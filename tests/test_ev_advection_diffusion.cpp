// Test: rotating diffusing Gaussian with the entropy-viscosity (EV)
// energy-equation stabilization.
//
// Mirrors test_unsteady_advection_diffusion_supg_2.cpp (single inner case)
// but replaces the disabled-during-refactor EV scaffolding with the actual
// per-wedge EV pipeline used by EVSolver in apps/mantlecirculation/src/
// energy_solver.hpp:
//
//   LHS:  (M + dt·A_pureGalerkin) · T^{n+1} = f        (kerngen A, set_supg_enabled(false))
//   RHS:  f = M·T^n  -  dt·∫ ν_h ∇T^n · ∇φ_i dx          (per-wedge EV diffusion)
//   ν_h:  computed each step via per-wedge compute_nu_h    (new API)
//   lap_T (Q1-nodal): (A_kappa · T) / M_lumped, Dirichlet zeroed
//   T_{n-1}: rotated from T BEFORE the solve overwrites T
//   Dirichlet BCs: time-dependent, set from the exact rotated-Gaussian solution.
//
// Exact solution: T(x, t) = (σ₀/σ(t))³ · exp(-|x - x_c(t)|² / (2σ(t)²))
// with x_c(t) = rotated by ωt and σ²(t) = σ₀² + 2κt.
//
// Assertion: final scaled L2 error must be finite and below a generous bound,
// and T must stay near [0, 1] (no large over/undershoot).

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/entropy_viscosity.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg_kerngen.hpp"
#include "fe/wedge/operators/shell/wedge_constant_div_k_grad.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/solver.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::Grid5DDataScalar;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using ScalarType = double;

ScalarType max_entry( const Grid4DDataScalar< ScalarType >& x )
{
    ScalarType max_val = std::numeric_limits< ScalarType >::lowest();
    Kokkos::parallel_reduce(
        "max_entry",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 },
                                                    { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int id, int i, int j, int k, ScalarType& lmax ) {
            lmax = Kokkos::max( lmax, x( id, i, j, k ) );
        },
        Kokkos::Max< ScalarType >( max_val ) );
    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &max_val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return max_val;
}

// Rigid-body rotation u = ω · (-y, x, 0).
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
        ScalarType                            omega )
    : grid_( grid ), radii_( radii ), data_( data ), omega_( omega )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > c = grid::shell::coords( id, x, y, r, grid_, radii_ );
        data_( id, x, y, r, 0 ) = -c( 1 ) * omega_;
        data_( id, x, y, r, 1 ) =  c( 0 ) * omega_;
        data_( id, x, y, r, 2 ) =  0.0;
    }
};

// Exact rotating-diffusing-Gaussian solution.  If only_boundary, only writes
// at boundary nodes (used to build the time-dependent Dirichlet g vector).
struct SolutionAndRHSInterpolator
{
    Grid3DDataVec< ScalarType, 3 >                     grid_;
    Grid2DDataScalar< ScalarType >                     radii_;
    Grid4DDataScalar< ScalarType >                     data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;
    ScalarType                                         t_, kappa_, omega_, x0_, y0_, z0_, sigma0_;
    bool                                               only_boundary_;

    SolutionAndRHSInterpolator(
        const Grid3DDataVec< ScalarType, 3 >&                     grid,
        const Grid2DDataScalar< ScalarType >&                     radii,
        const Grid4DDataScalar< ScalarType >&                     data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        ScalarType t, ScalarType kappa, ScalarType omega,
        ScalarType x0, ScalarType y0, ScalarType z0, ScalarType sigma0,
        bool only_boundary )
    : grid_( grid ), radii_( radii ), data_( data ), boundary_mask_( boundary_mask )
    , t_( t ), kappa_( kappa ), omega_( omega )
    , x0_( x0 ), y0_( y0 ), z0_( z0 ), sigma0_( sigma0 )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x_idx, const int y_idx, const int r_idx ) const
    {
        if ( only_boundary_ && !util::has_flag(
                 boundary_mask_( id, x_idx, y_idx, r_idx ), grid::shell::ShellBoundaryFlag::BOUNDARY ) )
            return;

        const dense::Vec< ScalarType, 3 > c = grid::shell::coords( id, x_idx, y_idx, r_idx, grid_, radii_ );
        const ScalarType ct   = Kokkos::cos( omega_ * t_ );
        const ScalarType st   = Kokkos::sin( omega_ * t_ );
        const ScalarType xc   = x0_ * ct - y0_ * st;
        const ScalarType yc   = x0_ * st + y0_ * ct;
        const ScalarType zc   = z0_;
        const ScalarType sig2 = sigma0_ * sigma0_ + ScalarType( 2 ) * kappa_ * t_;
        const ScalarType sig  = Kokkos::sqrt( sig2 );
        const ScalarType r0   = sigma0_ / sig;
        const ScalarType amp  = r0 * r0 * r0;
        const ScalarType dx   = c( 0 ) - xc;
        const ScalarType dy   = c( 1 ) - yc;
        const ScalarType dz   = c( 2 ) - zc;
        const ScalarType r2   = dx * dx + dy * dy + dz * dz;
        data_( id, x_idx, y_idx, r_idx ) = amp * Kokkos::exp( -r2 / ( ScalarType( 2 ) * sig2 ) );
    }
};

int test( const int level, const ScalarType kappa )
{
    constexpr int        timesteps = 50;
    constexpr int        restart   = 10;
    constexpr ScalarType omega     = 1.0;
    constexpr ScalarType x0        = 0.75;
    constexpr ScalarType y0        = 0.0;
    constexpr ScalarType z0        = 0.0;
    constexpr ScalarType sigma0    = 0.005;

    const auto domain = DistributedDomain::create_uniform(
        level,
        grid::shell::mapped_shell_radii( 0.5, 1.0, ( 1 << level ) + 1, grid::shell::make_tanh_boundary_cluster( 2.0 ) ),
        0, 0 );

    const auto h_min = grid::shell::min_radial_h( domain.domain_info().radii() );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorQ1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorQ1Scalar< ScalarType > T_prev( "T_prev", domain, mask_data );
    VectorQ1Scalar< ScalarType > rhs_ev( "rhs_ev", domain, mask_data );
    VectorQ1Scalar< ScalarType > M_lumped( "M_lumped", domain, mask_data );
    VectorQ1Scalar< ScalarType > lap_T( "lap_T", domain, mask_data );
    VectorQ1Scalar< ScalarType > g( "g", domain, mask_data );
    VectorQ1Scalar< ScalarType > f( "f", domain, mask_data );
    VectorQ1Vec< ScalarType >    u( "u", domain, mask_data );
    VectorQ1Scalar< ScalarType > solution( "solution", domain, mask_data );
    VectorQ1Scalar< ScalarType > error( "error", domain, mask_data );

    // Per-wedge ν_h and κ_wedge for the new EV API.
    const auto num_sub = static_cast< long long >( domain.subdomains().size() );
    const auto nx_c    = domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
    const auto nr_c    = domain.domain_info().subdomain_num_nodes_radially() - 1;
    Grid5DDataScalar< ScalarType > nu_h_wedge(
        "nu_h_wedge", num_sub, nx_c, nx_c, nr_c, fe::wedge::num_wedges_per_hex_cell );
    Grid5DDataScalar< ScalarType > kappa_wedge(
        "kappa_wedge", num_sub, nx_c, nx_c, nr_c, fe::wedge::num_wedges_per_hex_cell );
    kernels::common::set_constant( kappa_wedge, kappa );

    const fe::wedge::operators::shell::EntropyViscosityParameters< ScalarType > ev_params{};

    std::vector< VectorQ1Scalar< ScalarType > > tmps;
    tmps.reserve( 2 * restart + 4 );
    for ( int i = 0; i < 2 * restart + 4; ++i )
        tmps.emplace_back( "tmpp", domain, mask_data );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );

    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;
    AD A(   domain, coords_shell, coords_radii, boundary_mask_data, u, kappa, 0.0, /*treat_boundary=*/true );
    AD A_n( domain, coords_shell, coords_radii, boundary_mask_data, u, kappa, 0.0, /*treat_boundary=*/false );
    AD A_d( domain, coords_shell, coords_radii, boundary_mask_data, u, kappa, 0.0, /*treat_boundary=*/false, /*diagonal=*/true );
    A.set_supg_enabled( false );
    A_n.set_supg_enabled( false );
    A_d.set_supg_enabled( false );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;
    Mass M( domain, coords_shell, coords_radii );

    using EVDiffOp = fe::wedge::operators::shell::WedgeConstantDivKGrad< ScalarType >;
    EVDiffOp A_kappa(  domain, coords_shell, coords_radii, kappa_wedge );
    EVDiffOp A_evdiff( domain, coords_shell, coords_radii, nu_h_wedge );

    // M_lumped = M · 1.
    {
        VectorQ1Scalar< ScalarType > ones( "ones", domain, mask_data );
        linalg::assign( ones, ScalarType( 1 ) );
        linalg::assign( M_lumped, ScalarType( 0 ) );
        linalg::apply( M, ones, M_lumped );
    }

    auto table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< AD > solver(
        tmps,
        { .restart                     = restart,
          .relative_residual_tolerance = 1e-6,
          .absolute_residual_tolerance = 1e-12,
          .max_iterations              = 100 },
        table );

    // Initial T = exact at t=0.
    ScalarType t = 0.0;
    Kokkos::parallel_for( "velocity", local_domain_md_range_policy_nodes( domain ),
                          VelocityInterpolator( coords_shell, coords_radii, u.grid_data(), omega ) );
    Kokkos::parallel_for( "init temp", local_domain_md_range_policy_nodes( domain ),
                          SolutionAndRHSInterpolator( coords_shell, coords_radii, T.grid_data(),
                                                      boundary_mask_data, t, kappa, omega,
                                                      x0, y0, z0, sigma0, /*only_boundary=*/false ) );
    Kokkos::fence();
    Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() );

    util::logroot << "Running EV advection-diffusion test"
                  << "  level=" << level << "  h_min=" << h_min << "  kappa=" << kappa
                  << "  steps=" << timesteps << "  alpha_max=" << ev_params.alpha_max << "\n";

    ScalarType l2_error  = 0;
    ScalarType t_min_run = +std::numeric_limits< ScalarType >::infinity();
    ScalarType t_max_run = -std::numeric_limits< ScalarType >::infinity();

    for ( int ts = 1; ts < timesteps; ++ts )
    {
        const auto max_vel      = kernels::common::max_vector_magnitude( u.grid_data() );
        const auto dt_advection = h_min / max_vel;
        const auto dt           = 0.5 * dt_advection;

        A.dt()   = dt;
        A_n.dt() = dt;
        A_d.dt() = dt;
        t       += dt;

        // Time-dependent Dirichlet vector g from the exact solution.
        linalg::assign( g, 0.0 );
        Kokkos::parallel_for(
            "boundary temp", local_domain_md_range_policy_nodes( domain ),
            SolutionAndRHSInterpolator( coords_shell, coords_radii, g.grid_data(),
                                        boundary_mask_data, t, kappa, omega,
                                        x0, y0, z0, sigma0, /*only_boundary=*/true ) );
        Kokkos::fence();

        // --- EV pipeline (mirrors EVSolver::step) ---------------------------
        linalg::apply( A_kappa, T, lap_T );
        {
            auto       lap_v = lap_T.grid_data();
            const auto m_v   = M_lumped.grid_data();
            const auto bm    = boundary_mask_data;
            Kokkos::parallel_for(
                "lap_T_div_M",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 }, { lap_v.extent( 0 ), lap_v.extent( 1 ), lap_v.extent( 2 ), lap_v.extent( 3 ) } ),
                KOKKOS_LAMBDA( int s, int i, int j, int k ) {
                    if ( util::has_flag( bm( s, i, j, k ), grid::shell::ShellBoundaryFlag::BOUNDARY ) )
                    {
                        lap_v( s, i, j, k ) = ScalarType( 0 );
                        return;
                    }
                    const ScalarType m = m_v( s, i, j, k );
                    lap_v( s, i, j, k ) = ( m > ScalarType( 0 ) ) ? ( lap_v( s, i, j, k ) / m ) : ScalarType( 0 );
                } );
            Kokkos::fence();
        }

        const auto stats = fe::wedge::operators::shell::compute_entropy_stats(
            T, mask_data, domain, coords_shell, coords_radii, ev_params );
        fe::wedge::operators::shell::compute_nu_h(
            nu_h_wedge, T, T_prev, u, lap_T.grid_data(),
            domain, coords_shell, coords_radii, dt, stats, ev_params );

        linalg::apply( A_evdiff, T, rhs_ev );

        // f = M·T  -  dt · rhs_ev.
        linalg::apply( M, T, f );
        linalg::lincomb( f, { ScalarType( 1 ), -dt }, { f, rhs_ev } );

        // History rotation BEFORE the solve.
        Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() );

        // Apply time-dependent Dirichlet enforcement to f using g.
        linalg::assign( tmps[ 0 ], 0.0 );
        fe::strong_algebraic_dirichlet_enforcement_poisson_like(
            A_n, A_d, g, tmps[ 0 ], f, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

        linalg::solvers::solve( solver, A, T, f );
        table->clear();

        // Exact solution at t for error.
        Kokkos::parallel_for(
            "exact", local_domain_md_range_policy_nodes( domain ),
            SolutionAndRHSInterpolator( coords_shell, coords_radii, solution.grid_data(),
                                        boundary_mask_data, t, kappa, omega,
                                        x0, y0, z0, sigma0, /*only_boundary=*/false ) );
        Kokkos::fence();

        linalg::lincomb( error, { 1.0, -1.0 }, { solution, T } );
        l2_error = linalg::norm_2_scaled( error, 1.0 / static_cast< ScalarType >( num_dofs ) );

        const auto t_min_step = kernels::common::min_entry( T.grid_data() );
        const auto t_max_step = max_entry( T.grid_data() );
        t_min_run = std::min( t_min_run, t_min_step );
        t_max_run = std::max( t_max_run, t_max_step );

        util::logroot << "  ts=" << ts << "  t=" << t << "  dt=" << dt
                      << "  L2 err=" << l2_error
                      << "  T_min=" << t_min_step << "  T_max=" << t_max_step << "\n";
    }

    util::logroot << "\n================================================================\n"
                  << " EV advection-diffusion summary (level=" << level << ", kappa=" << kappa << ")\n"
                  << "================================================================\n"
                  << "  final scaled L2 error: " << std::scientific << std::setprecision( 6 ) << l2_error << "\n"
                  << "  worst T min:           " << std::scientific << std::setprecision( 3 ) << t_min_run << "\n"
                  << "  worst T max:           " << std::scientific << std::setprecision( 3 ) << t_max_run << "\n";

    // Assertion: trajectory must be physical.
    //   * Finite L2 error.
    //   * Bounded scaled L2 error (loose: scale-aware threshold).
    //   * No huge over/undershoot — exact T ∈ [0, ≤1], allow modest excursion.
    if ( !std::isfinite( l2_error ) )
    {
        util::logroot << "FAIL: L2 error is not finite\n";
        return 1;
    }
    if ( l2_error > 1e-2 )
    {
        util::logroot << "FAIL: final scaled L2 error " << l2_error << " exceeds 1e-2 threshold\n";
        return 1;
    }
    if ( t_min_run < -0.5 || t_max_run > 1.5 )
    {
        util::logroot << "FAIL: T excursion (" << t_min_run << ", " << t_max_run
                      << ") outside [-0.5, 1.5]\n";
        return 1;
    }
    return 0;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    // Single (L, κ) case: L=3 (σ/h ≈ 0.33, sub-grid Gaussian), κ=1e-4 (nearly
    // pure advection, where EV should outperform SUPG on shape preservation).
    return test( /*level=*/3, /*kappa=*/1e-4 );
}
