

#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/linearforms/shell/supg_rhs.hpp"
#include "fe/wedge/operators/shell/div_k_grad.hpp"
#include "fe/wedge/operators/shell/entropy_viscosity.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/pbicgstab.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using ScalarType = double;

/// Local signed-max reduction (kernels::common provides min_entry and max_abs_entry
/// but no signed max_entry; we need it to spot over/undershoots: exact T ∈ [0,1],
/// so min(T) < 0 signals an undershoot, max(T) > 1 signals an overshoot).
template < typename T >
T max_entry( const grid::Grid4DDataScalar< T >& x )
{
    T max_val = std::numeric_limits< T >::lowest();
    Kokkos::parallel_reduce(
        "max_entry",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >( { 0, 0, 0, 0 },
                                                    { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int s, int i, int j, int k, T& lmax ) {
            lmax = Kokkos::max( lmax, x( s, i, j, k ) );
        },
        Kokkos::Max< T >( max_val ) );
    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &max_val, 1, mpi::mpi_datatype< T >(), MPI_MAX, MPI_COMM_WORLD );
    return max_val;
}

/// Rotating-Gaussian benchmark for the advection-diffusion equation.
/// ASPECT-style (Kronbichler-Heister-Bangerth, GJI 2012); 3D shell extension.
///
/// Eq.
///
///   ∂_t T + u · ∇T − κ ∇²T = H
///
/// Exact solution (Green's function of 3D heat equation advected by rigid rotation):
///
///   u(x, y, z)    = (−Ωy, Ωx, 0)                     solid-body rotation about z
///
///   x_c(t) = x₀ cos(Ωt) − y₀ sin(Ωt),
///   y_c(t) = x₀ sin(Ωt) + y₀ cos(Ωt),
///   z_c(t) = z₀                                       Gaussian centre on the
///                                                     rotating circle (x₀,y₀,z₀)
///   σ(t)   = √(σ₀² + 2κt)                             width spreads by diffusion
///
///   T(x, y, z, t) = (σ₀/σ(t))³ · exp(−|x − x_c(t)|² / (2σ(t)²))
///
///   H(x, y, z, t) = 0                                 ← no forcing anywhere!
///
/// Why H = 0: in the rotating frame this is just the 3D fundamental solution of
/// the heat equation, which satisfies ∂_t T = κ ∇²T identically. Rotation commutes
/// with Laplacian, so ∂_t T + u·∇T − κ∇²T = 0 holds for all κ ≥ 0. No κ/σ² forcing
/// pathology. At κ = 0 the Gaussian is pure advection (width constant); at κ > 0
/// it spreads naturally during the simulation.
///
/// When σ₀ < h, SUPG alone produces Gibbs-like over/undershoots — the YZβ target.
struct VelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_;
    ScalarType                     omega_;

    VelocityInterpolator(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataVec< ScalarType, 3 >& data,
        const ScalarType&                     omega )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , omega_( omega )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        data_( local_subdomain_id, x, y, r, 0 ) = -coords( 1 ) * omega_;
        data_( local_subdomain_id, x, y, r, 1 ) = coords( 0 ) * omega_;
        data_( local_subdomain_id, x, y, r, 2 ) = 0.0;
    }
};

struct SolutionAndRHSInterpolator
{
    Grid3DDataVec< ScalarType, 3 >                     grid_;
    Grid2DDataScalar< ScalarType >                     radii_;
    Grid4DDataScalar< ScalarType >                     data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    ScalarType t_;
    ScalarType kappa_;
    ScalarType omega_;
    ScalarType x0_;     // initial Gaussian-centre coordinates
    ScalarType y0_;
    ScalarType z0_;
    ScalarType sigma0_; // initial Gaussian width

    bool is_rhs_;
    bool only_boundary_;

    SolutionAndRHSInterpolator(
        const Grid3DDataVec< ScalarType, 3 >&                     grid,
        const Grid2DDataScalar< ScalarType >&                     radii,
        const Grid4DDataScalar< ScalarType >&                     data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const ScalarType&                                         t,
        const ScalarType&                                         kappa,
        const ScalarType&                                         omega,
        const ScalarType&                                         x0,
        const ScalarType&                                         y0,
        const ScalarType&                                         z0,
        const ScalarType&                                         sigma0,
        const bool                                                is_rhs,
        const bool                                                only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , boundary_mask_( boundary_mask )
    , t_( t )
    , kappa_( kappa )
    , omega_( omega )
    , x0_( x0 )
    , y0_( y0 )
    , z0_( z0 )
    , sigma0_( sigma0 )
    , is_rhs_( is_rhs )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x_idx, const int y_idx, const int r_idx ) const
    {
        const dense::Vec< ScalarType, 3 > coords =
            grid::shell::coords( local_subdomain_id, x_idx, y_idx, r_idx, grid_, radii_ );

        const auto x = coords( 0 );
        const auto y = coords( 1 );
        const auto z = coords( 2 );

        if ( !only_boundary_ ||
             util::has_flag(
                 boundary_mask_( local_subdomain_id, x_idx, y_idx, r_idx ), grid::shell::ShellBoundaryFlag::BOUNDARY ) )
        {
            if ( is_rhs_ )
            {
                // H = 0 — exact solution of the rotating-diffusing Gaussian needs no source.
                data_( local_subdomain_id, x_idx, y_idx, r_idx ) = ScalarType( 0 );
                return;
            }

            // Rotated centre and diffusively-spread width.
            const ScalarType ct   = Kokkos::cos( omega_ * t_ );
            const ScalarType st   = Kokkos::sin( omega_ * t_ );
            const ScalarType xc   = x0_ * ct - y0_ * st;
            const ScalarType yc   = x0_ * st + y0_ * ct;
            const ScalarType zc   = z0_;
            const ScalarType sig2 = sigma0_ * sigma0_ + ScalarType( 2 ) * kappa_ * t_;
            const ScalarType sig  = Kokkos::sqrt( sig2 );
            const ScalarType r0   = sigma0_ / sig;
            const ScalarType amp  = r0 * r0 * r0; // (σ₀/σ)³ — volume-conservation factor
            const ScalarType dx   = x - xc;
            const ScalarType dy   = y - yc;
            const ScalarType dz   = z - zc;
            const ScalarType r2   = dx * dx + dy * dy + dz * dz;
            data_( local_subdomain_id, x_idx, y_idx, r_idx ) =
                amp * Kokkos::exp( -r2 / ( ScalarType( 2 ) * sig2 ) );
        }
    }
};

struct SweepResult
{
    int        level;
    long       num_dofs;
    ScalarType h_min;
    ScalarType l2_error_final;
    ScalarType t_min;       // worst (smallest) value of T across the run  — signals undershoot if < 0
    ScalarType t_max;       // worst (largest)  value of T across the run  — signals overshoot  if > 1
};

SweepResult test( int level, ScalarType kappa, const std::shared_ptr< util::Table >& table )
{
    constexpr int timesteps = 50;
    constexpr int restart   = 10;

    // Rotating-Gaussian parameters.
    //   omega  : angular rotation speed (rigid-body rotation about z)
    //   kappa  : diffusivity  (passed in — sweep variable)
    //   x0/y0/z0 : initial Gaussian centre. (0.75, 0, 0) sits in the equatorial
    //              plane, well inside the shell (0.5 < r < 1.0).
    //   sigma0 : initial Gaussian half-width. 0.005 gives:
    //              L=2 (h=5.25e-2): σ/h ≈ 0.10 (deeply sub-grid; SUPG rings)
    //              L=3 (h=1.53e-2): σ/h ≈ 0.33
    //              L=4 (h=5.87e-3): σ/h ≈ 0.85 (marginally resolved)
    //              L=5 (h=2.59e-3): σ/h ≈ 1.93 (resolved)
    //            → persistent SUPG failure at coarse L, recovery at fine L.
    constexpr auto omega  = 1.0;
    constexpr auto x0     = static_cast< ScalarType >( 0.75 );
    constexpr auto y0     = static_cast< ScalarType >( 0.0 );
    constexpr auto z0     = static_cast< ScalarType >( 0.0 );
    constexpr auto sigma0 = static_cast< ScalarType >( 0.005 );

    // XDMF disabled during level sweeps: per-level output dirs would collide and
    // each write adds noticeable overhead for the coarser levels.
    constexpr auto xdmf = false;

    ScalarType t = 0.0;

    const auto domain = DistributedDomain::create_uniform(
        level,
        grid::shell::mapped_shell_radii( 0.5, 1.0, ( 1 << level ) + 1, grid::shell::make_tanh_boundary_cluster( 2.0 ) ),
        0,
        0 );

    const auto h_min = grid::shell::min_radial_h( domain.domain_info().radii() );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorQ1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorQ1Scalar< ScalarType > T_prev( "T_prev", domain, mask_data );      // for EV ∂_t E
    VectorQ1Scalar< ScalarType > nu_h_nodal( "nu_h_nodal", domain, mask_data );
    VectorQ1Scalar< ScalarType > rhs_ev( "rhs_ev", domain, mask_data );
    VectorQ1Scalar< ScalarType > kappa_nodal( "kappa_nodal", domain, mask_data );    // EV: κ as a Q1 field for DivKGrad
    VectorQ1Scalar< ScalarType > M_lumped( "M_lumped", domain, mask_data );          // EV: lumped mass diagonal
    VectorQ1Scalar< ScalarType > ones_vec( "ones", domain, mask_data );              // EV: helper for M_lumped
    VectorQ1Scalar< ScalarType > lap_T( "lap_T", domain, mask_data );                // EV: -κ∇²T (pointwise, projected)
    VectorQ1Scalar< ScalarType > g( "g", domain, mask_data );
    VectorQ1Scalar< ScalarType > f_strong( "f_strong", domain, mask_data );
    VectorQ1Scalar< ScalarType > f( "f", domain, mask_data );
    VectorQ1Vec< ScalarType >    u( "u", domain, mask_data );
    VectorQ1Scalar< ScalarType > solution( "solution", domain, mask_data );
    VectorQ1Scalar< ScalarType > error( "error", domain, mask_data );

    // Per-cell ν_h field.  Cell extents: (n_sub, N-1, N-1, N_r-1).
    const auto num_sub = static_cast< long long >( domain.subdomains().size() );
    const auto nx_c    = domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
    const auto nr_c    = domain.domain_info().subdomain_num_nodes_radially() - 1;
    grid::Grid4DDataScalar< ScalarType > nu_h( "nu_h", num_sub, nx_c, nx_c, nr_c );

    // Stabilization mode for this test:
    //   ev_apply=true  →  pure Galerkin advection-diffusion on the LHS +
    //                     explicit ν_h contribution on the RHS (ASPECT-like).
    //   ev_apply=false →  plain SUPG (existing baseline).
    //
    // NOTE: forced false during the per-wedge EV refactor.  The old per-hex
    // ν_h API has been removed; this test should be re-migrated to the
    // per-wedge path once WedgeConstantDivKGrad lands.
    constexpr bool ev_apply = false;
    const fe::wedge::operators::shell::EntropyViscosityParameters< ScalarType > ev_params{};

    std::vector< VectorQ1Scalar< ScalarType > > tmps;
    for ( int i = 0; i < 2 * restart + 4; ++i )
    {
        tmps.emplace_back( "tmpp", domain, mask_data );
    }

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    std::cout << "Number of dofs: " << num_dofs << std::endl;

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;

    AD A( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, kappa, 0, true );
    AD A_neumann( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, kappa, 0, false );
    AD A_diagonal( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, kappa, 0, false, true );

    // Pure Galerkin on the LHS when EV is driving the stabilization.
    if constexpr ( ev_apply )
    {
        A.set_supg_enabled( false );
        A_neumann.set_supg_enabled( false );
        A_diagonal.set_supg_enabled( false );
    }

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;

    Mass                                   M( domain, subdomain_shell_coords, subdomain_radii );
    fe::wedge::linearforms::shell::SUPGRHS supg_rhs(
        domain, subdomain_shell_coords, subdomain_radii, f_strong, u, kappa );

    // EV explicit-RHS operator: rhs_ev_i = ∫ ν_h ∇T · ∇φ_i dx.
    using DivKGrad = fe::wedge::operators::shell::DivKGrad< ScalarType >;
    DivKGrad A_ev( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data,
                   nu_h_nodal.grid_data(),
                   /*treat_boundary=*/false,
                   /*diagonal=*/false );

    // EV residual diffusion: lap_T = (DivKGrad(κ) · T) / M_lumped  ≈  -κ∇²T (pointwise).
    // This makes the entropy residual r_E include the full KHB form
    //   ∂_t E + (T-T_m)·(u·∇T - κ∇²T)
    // rather than dropping the diffusion piece.  At κ = 0 the operator is the
    // zero map so lap_T stays at zero and r_E reduces to the pure-advection form.
    linalg::assign( kappa_nodal, kappa );
    DivKGrad A_kappa( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data,
                      kappa_nodal.grid_data(),
                      /*treat_boundary=*/false,
                      /*diagonal=*/false );

    // Lumped mass = M · 1.  Computed once at setup; constant in time.
    linalg::assign( ones_vec, ScalarType( 1 ) );
    linalg::apply( M, ones_vec, M_lumped );

    // Set up solution data.
    Kokkos::parallel_for(
        "velocity interpolation",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator( subdomain_shell_coords, subdomain_radii, u.grid_data(), omega ) );

    Kokkos::fence();

    // Set up the initial temperature.
    Kokkos::parallel_for(
        "initial temp interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionAndRHSInterpolator(
            subdomain_shell_coords,
            subdomain_radii,
            T.grid_data(),
            boundary_mask_data,
            t,
            kappa,
            omega,
            x0,
            y0,
            z0,
            sigma0,
            false,
            false ) );

    Kokkos::fence();

    // EV bootstrap: start T_prev = T so ∂_t E = 0 on step 1.
    Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() );

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionAndRHSInterpolator(
            subdomain_shell_coords,
            subdomain_radii,
            solution.grid_data(),
            boundary_mask_data,
            t,
            kappa,
            omega,
            x0,
            y0,
            z0,
            sigma0,
            false,
            false ) );

    Kokkos::fence();

    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionAndRHSInterpolator(
            subdomain_shell_coords,
            subdomain_radii,
            f_strong.grid_data(),
            boundary_mask_data,
            t,
            kappa,
            omega,
            x0,
            y0,
            z0,
            sigma0,
            true,
            false ) );

    Kokkos::fence();

    linalg::solvers::FGMRES< AD > solver(
        tmps,
        { .restart                     = restart,
          .relative_residual_tolerance = 1e-6,
          .absolute_residual_tolerance = 1e-12,
          .max_iterations              = 100 },
        table );

    io::XDMFOutput xdmf_output(
        "test_unsteady_advection_diffusion_supg_2_output", domain, subdomain_shell_coords, subdomain_radii );
    xdmf_output.add( T.grid_data() );
    xdmf_output.add( solution.grid_data() );
    xdmf_output.add( error.grid_data() );

    if ( xdmf )
    {
        xdmf_output.write();
    }

    util::logroot << "Timestep " << 0 << std::endl;
    util::logroot << "  dt =     " << "-" << std::endl;
    util::logroot << "  h =      " << h_min << std::endl;
    util::logroot << "  kappa =  " << kappa << std::endl;

    linalg::lincomb( error, { 1.0, -1.0 }, { solution, T } );
    auto l2_error = linalg::norm_2_scaled( error, 1.0 / static_cast< ScalarType >( num_dofs ) );
    auto t_min_step = kernels::common::min_entry( T.grid_data() );
    auto t_max_step = max_entry( T.grid_data() );
    ScalarType t_min_run = t_min_step;
    ScalarType t_max_run = t_max_step;
    util::logroot << "L2 error: " << l2_error
                  << "  |  T min: " << t_min_step << "  T max: " << t_max_step << std::endl;

    for ( int ts = 1; ts < timesteps; ++ts )
    {
        const auto max_vel = kernels::common::max_vector_magnitude( u.grid_data() );

        // Choose "suitable" small dt for accuracy - we have and implicit time-stepping scheme so we do not really need
        // a CFL in the classical sense. Still useful for time-step size restriction.
        const auto dt_advection = h_min / max_vel;
        // const auto dt_diffusion = ( h * h ) / prm.diffusivity;
        // const auto dt           = prm.pseudo_cfl * std::min( dt_advection, dt_diffusion );
        auto dt = 0.5 * dt_advection;

        A.dt()          = dt;
        A_neumann.dt()  = dt;
        A_diagonal.dt() = dt;

        util::logroot << "Timestep " << ts << std::endl;
        util::logroot << "  dt =     " << dt << std::endl;
        util::logroot << "  h =      " << h_min << std::endl;
        util::logroot << "  kappa =  " << kappa << std::endl;
        t += dt;

        Kokkos::parallel_for(
            "rhs interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionAndRHSInterpolator(
                subdomain_shell_coords,
                subdomain_radii,
                f_strong.grid_data(),
                boundary_mask_data,
                t,
                kappa,
                omega,
                x0,
                y0,
                z0,
                sigma0,
                true,
                false ) );

        assign( g, 0.0 );
        Kokkos::parallel_for(
            "boundary temp interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionAndRHSInterpolator(
                subdomain_shell_coords,
                subdomain_radii,
                g.grid_data(),
                boundary_mask_data,
                t,
                kappa,
                omega,
                x0,
                y0,
                z0,
                sigma0,
                false,
                true ) );

        Kokkos::fence();

        // --- Entropy-viscosity stabilization (explicit, lagged) -----------
        // On entry T = T^{n-1}, T_prev = T^{n-2}.  Compute ν_h from this pair,
        // assemble the explicit term rhs_ev = ∫ ν_h ∇T · ∇φ dx (applied to T^{n-1}),
        // then snapshot T → T_prev BEFORE the solve overwrites T.
        if constexpr ( ev_apply )
        {
            // Pointwise -κ∇²T at nodes via lumped-mass-projected weak Laplacian:
            //   lap_T_weak = (DivKGrad(κ) · T)             (units of stiffness vector)
            //   lap_T      = lap_T_weak / M_lumped         ≈ -κ∇²T (pointwise, units of T/time)
            linalg::apply( A_kappa, T, lap_T );
            {
                auto       lap_view = lap_T.grid_data();
                const auto m_view   = M_lumped.grid_data();
                Kokkos::parallel_for(
                    "lap_T_lumped_mass_project",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                        { 0, 0, 0, 0 },
                        { lap_view.extent( 0 ), lap_view.extent( 1 ),
                          lap_view.extent( 2 ), lap_view.extent( 3 ) } ),
                    KOKKOS_LAMBDA( int s, int i, int j, int k ) {
                        const ScalarType m = m_view( s, i, j, k );
                        lap_view( s, i, j, k ) = ( m > ScalarType( 0 ) )
                                                     ? lap_view( s, i, j, k ) / m
                                                     : ScalarType( 0 );
                    } );
                Kokkos::fence();
            }

            const auto stats =
                fe::wedge::operators::shell::compute_entropy_stats( T, mask_data, ev_params );
            fe::wedge::operators::shell::compute_nu_h(
                nu_h, T, T_prev, u, lap_T.grid_data(), domain, subdomain_shell_coords, subdomain_radii,
                static_cast< ScalarType >( dt ), stats, ev_params );
            fe::wedge::operators::shell::project_nu_h_to_nodes( nu_h_nodal, nu_h, domain );

            linalg::apply( A_ev, T, rhs_ev );
        }

        // History rotation before the solve overwrites T.
        Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() );

        assign( tmps[0], 0.0 );
        assign( tmps[1], 0.0 );
        assign( tmps[2], 0.0 );

        linalg::apply( M, T, tmps[0] );
        linalg::apply( M, f_strong, tmps[1] );
        // linalg::apply( supg_rhs, tmps[2] );

        // f = M T^n + dt · (M · f_strong + 0)  (VectorQ1Scalar::lincomb supports ≤ 3 inputs).
        lincomb( f, { 1.0, dt, dt }, { tmps[0], tmps[1], tmps[2] } );

        // f -= dt · rhs_ev  (explicit EV stabilization, applied in a second pass).
        if constexpr ( ev_apply )
        {
            lincomb( f, { 1.0, -dt }, { f, rhs_ev } );
        }

        assign( tmps[0], 0.0 );
        fe::strong_algebraic_dirichlet_enforcement_poisson_like(
            A_neumann, A_diagonal, g, tmps[0], f, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

        linalg::solvers::solve( solver, A, T, f );

        // table->print_pretty();
        table->clear();

        Kokkos::parallel_for(
            "solution interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionAndRHSInterpolator(
                subdomain_shell_coords,
                subdomain_radii,
                solution.grid_data(),
                boundary_mask_data,
                t,
                kappa,
                omega,
                x0,
                y0,
                z0,
                sigma0,
                false,
                false ) );

        Kokkos::fence();

        linalg::lincomb( error, { 1.0, -1.0 }, { solution, T } );
        l2_error   = linalg::norm_2_scaled( error, 1.0 / static_cast< ScalarType >( num_dofs ) );
        t_min_step = kernels::common::min_entry( T.grid_data() );
        t_max_step = max_entry( T.grid_data() );
        t_min_run  = std::min( t_min_run, t_min_step );
        t_max_run  = std::max( t_max_run, t_max_step );
        util::logroot << "L2 error: " << l2_error
                      << "  |  T min: " << t_min_step << "  T max: " << t_max_step << std::endl;

        if ( xdmf )
        {
            xdmf_output.write();
        }
    }

    return SweepResult{ level, num_dofs, h_min, l2_error, t_min_run, t_max_run };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    constexpr int level_min = 2;
    constexpr int level_max = 5;

    const std::vector< std::pair< std::string, ScalarType > > kappa_cases = {
        { "kappa = 0       (pure advection — sharp Gaussian stays sharp; SUPG should ring at coarse L)", ScalarType( 0 ) },
        { "kappa = 1e-4    (nearly pure advection; width almost constant)",                                ScalarType( 1e-4 ) },
        { "kappa = 1e-2    (moderate diffusion; width roughly doubles over the run — clean baseline)",     ScalarType( 1e-2 ) },
    };

    std::vector< std::vector< SweepResult > > all_results;

    for ( const auto& kc : kappa_cases )
    {
        util::logroot << "\n################################################################\n"
                      << "# " << kc.first << "\n"
                      << "################################################################" << std::endl;

        std::vector< SweepResult > results;
        for ( int L = level_min; L <= level_max; ++L )
        {
            util::logroot << "================================================================\n"
                          << " Level sweep: L=" << L << "  (kappa=" << kc.second << ")\n"
                          << "================================================================" << std::endl;
            results.push_back( test( L, kc.second, table ) );
        }
        all_results.push_back( std::move( results ) );
    }

    for ( size_t k = 0; k < kappa_cases.size(); ++k )
    {
        util::logroot << "\n================================================================\n"
                      << " Level-sweep summary  (" << kappa_cases[k].first << ")\n"
                      << "================================================================" << std::endl;
        util::logroot << "  level |  #DoFs     |    h_min   |    L2 error    |   rate  |   T min    |   T max   "
                      << std::endl;
        util::logroot << "  ------+------------+------------+----------------+---------+------------+-----------"
                      << std::endl;

        const auto& results = all_results[k];
        for ( size_t i = 0; i < results.size(); ++i )
        {
            const auto& r = results[i];
            std::ostringstream line;
            line << "  " << std::setw( 5 ) << r.level
                 << " | "  << std::setw( 10 ) << r.num_dofs
                 << " | "  << std::scientific << std::setprecision( 3 ) << std::setw( 10 ) << r.h_min
                 << " | "  << std::scientific << std::setprecision( 6 ) << std::setw( 14 ) << r.l2_error_final;
            if ( i == 0 )
            {
                line << " |    —   ";
            }
            else
            {
                const double rate =
                    std::log( results[i - 1].l2_error_final / r.l2_error_final ) /
                    std::log( results[i - 1].h_min / r.h_min );
                line << " | "  << std::fixed << std::setprecision( 2 ) << std::setw( 7 ) << rate;
            }
            line << " | "  << std::scientific << std::setprecision( 3 ) << std::setw( 10 ) << r.t_min
                 << " | "  << std::scientific << std::setprecision( 3 ) << std::setw( 10 ) << r.t_max;
            util::logroot << line.str() << std::endl;
        }
        util::logroot << "  (exact T in [0,1]; T_min < 0 = undershoot, T_max > 1 = overshoot)" << std::endl;
    }

    return 0;
}