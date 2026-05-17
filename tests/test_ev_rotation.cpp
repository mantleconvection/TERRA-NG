// Test: solid-body rotation of a cone using the entropy-viscosity (EV)
// energy-equation stabilization.
//
// Mirrors test_supg_rotation.cpp in problem setup (cone IC, ω=1 rotation,
// homogeneous Dirichlet, κ=0, dt=0.5·0.1·h, T_end=2π) but replaces the
// pure-SUPG advection-diffusion solve with the EV pipeline as used by
// EVSolver in apps/mantlecirculation/src/energy_solver.hpp:
//
//   LHS:  (M + dt·A_pureGalerkin) · T^{n+1} = q                 (kerngen A, set_supg_enabled(false))
//   RHS:  q = M · T^n  -  dt · ∫ ν_h ∇T^n · ∇φ_i dx              (per-wedge EV diffusion)
//   ν_h:  computed each step from (T^n, T^{n-1}, u, lap_T)        (compute_nu_h, per-wedge)
//   lap_T (Q1-nodal): (A_kappa · T) / M_lumped, with Dirichlet zeroed
//   T_{n-1}: rotated from T BEFORE the solve overwrites T
//
// Assertion: after one full revolution, the relative L2 error vs. the initial
// condition must be finite (no NaN/Inf) and bounded.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>

#include "../src/terra/communication/shell/communication.hpp"
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

// Solid-body rotation velocity:  u = (-y, x, 0).
struct VelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > c = grid::shell::coords( id, x, y, r, grid_, radii_ );
        data_( id, x, y, r, 0 )             = -c( 1 );
        data_( id, x, y, r, 1 )             =  c( 0 );
        data_( id, x, y, r, 2 )             =  0.0;
    }
};

// Cone IC at (0.75, 0, 0) with radius 0.2.
struct InitialConditionInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > c      = grid::shell::coords( id, x, y, r, grid_, radii_ );
        const dense::Vec< ScalarType, 3 > center{ 0.75, 0.0, 0.0 };
        const ScalarType                  radius = 0.2;
        const ScalarType                  dist   = ( c - center ).norm();
        if ( dist < radius )
        {
            const ScalarType s   = ScalarType( 1 ) - dist / radius;
            data_( id, x, y, r ) = s * s;
        }
    }
};

ScalarType compute_l2_relative_error_nodes(
    const Grid4DDataScalar< ScalarType >&             T,
    const Grid4DDataScalar< ScalarType >&             T_ref,
    const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask )
{
    ScalarType sum_sq_diff = 0;
    ScalarType sum_sq_ref  = 0;

    Kokkos::parallel_reduce(
        "l2_err_num",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 },
                                                    { T.extent( 0 ), T.extent( 1 ), T.extent( 2 ), T.extent( 3 ) } ),
        KOKKOS_LAMBDA( int id, int i, int j, int k, ScalarType& acc ) {
            if ( util::has_flag( mask( id, i, j, k ), grid::NodeOwnershipFlag::OWNED ) )
            {
                const ScalarType d = T( id, i, j, k ) - T_ref( id, i, j, k );
                acc += d * d;
            }
        },
        sum_sq_diff );

    Kokkos::parallel_reduce(
        "l2_err_den",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 },
                                                    { T.extent( 0 ), T.extent( 1 ), T.extent( 2 ), T.extent( 3 ) } ),
        KOKKOS_LAMBDA( int id, int i, int j, int k, ScalarType& acc ) {
            if ( util::has_flag( mask( id, i, j, k ), grid::NodeOwnershipFlag::OWNED ) )
            {
                const ScalarType v = T_ref( id, i, j, k );
                acc += v * v;
            }
        },
        sum_sq_ref );
    Kokkos::fence();

    ScalarType global_num = 0, global_den = 0;
    MPI_Allreduce( &sum_sq_diff, &global_num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( &sum_sq_ref,  &global_den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    return ( global_den < 1e-30 ) ? 0.0 : std::sqrt( global_num / global_den );
}

ScalarType max_entry_nodes( const Grid4DDataScalar< ScalarType >& x )
{
    ScalarType max_val = std::numeric_limits< ScalarType >::lowest();
    Kokkos::parallel_reduce(
        "max_entry_nodes",
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

int test( const int level )
{
    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Vec< ScalarType >    u( "u", domain, mask_data );
    VectorQ1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorQ1Scalar< ScalarType > T_prev( "T_prev", domain, mask_data );
    VectorQ1Scalar< ScalarType > T_ref( "T_ref", domain, mask_data );
    VectorQ1Scalar< ScalarType > f( "f", domain, mask_data );
    VectorQ1Scalar< ScalarType > rhs_ev( "rhs_ev", domain, mask_data );
    VectorQ1Scalar< ScalarType > lap_T( "lap_T", domain, mask_data );
    VectorQ1Scalar< ScalarType > M_lumped( "M_lumped", domain, mask_data );

    // Per-wedge ν_h field (new EV API): extents (n_sub, N-1, N-1, N_r-1, num_wedges).
    const auto num_sub = static_cast< long long >( domain.subdomains().size() );
    const auto nx_c    = domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
    const auto nr_c    = domain.domain_info().subdomain_num_nodes_radially() - 1;
    Grid5DDataScalar< ScalarType > nu_h_wedge(
        "nu_h_wedge", num_sub, nx_c, nx_c, nr_c, fe::wedge::num_wedges_per_hex_cell );

    // κ as a per-wedge field for the Galerkin Laplacian projection.  κ=0 for
    // the pure-rotation test ⇒ A_kappa·T = 0 ⇒ lap_T = 0 ⇒ ν_h reduces to its
    // velocity-only bound  (alpha_max · h_w · |u|).
    const ScalarType diffusivity = ScalarType( 0 );
    Grid5DDataScalar< ScalarType > kappa_wedge(
        "kappa_wedge", num_sub, nx_c, nx_c, nr_c, fe::wedge::num_wedges_per_hex_cell );
    kernels::common::set_constant( kappa_wedge, diffusivity );

    // Initialise velocity and temperature.
    Kokkos::parallel_for( "velocity_init", local_domain_md_range_policy_nodes( domain ),
                          VelocityInterpolator{ coords_shell, coords_radii, u.grid_data() } );
    Kokkos::parallel_for( "temperature_init", local_domain_md_range_policy_nodes( domain ),
                          InitialConditionInterpolator{ coords_shell, coords_radii, T.grid_data() } );
    Kokkos::fence();
    Kokkos::deep_copy( T_ref.grid_data(),  T.grid_data() );
    Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() ); // first-step bootstrap: ∂_t E ≈ 0

    // Time stepping.
    const ScalarType h           = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt          = 0.5 * 0.1 * h;
    const ScalarType T_end       = 2.0 * M_PI;
    const int        n_timesteps = static_cast< int >( std::ceil( T_end / dt ) );

    // Operators (mirror EVSolver constructor).
    //   A: pure-Galerkin advection-diffusion (SUPG kerngen with set_supg_enabled(false)).
    //   M: Mass.
    //   A_kappa: WedgeConstantDivKGrad(κ-uniform) — Galerkin Laplacian → projected to Q1 by /M_lumped.
    //   A_evdiff: WedgeConstantDivKGrad(ν_h_wedge) — explicit EV stabilization on the RHS.
    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;
    AD A(   domain, coords_shell, coords_radii, boundary_mask_data, u, diffusivity, dt, /*treat_boundary=*/true );
    AD A_n( domain, coords_shell, coords_radii, boundary_mask_data, u, diffusivity, dt, /*treat_boundary=*/false );
    AD A_d( domain, coords_shell, coords_radii, boundary_mask_data, u, diffusivity, dt, /*treat_boundary=*/false, /*diagonal=*/true );
    A.set_supg_enabled( false );
    A_n.set_supg_enabled( false );
    A_d.set_supg_enabled( false );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;
    Mass M( domain, coords_shell, coords_radii, false );

    using EVDiffOp = fe::wedge::operators::shell::WedgeConstantDivKGrad< ScalarType >;
    EVDiffOp A_kappa(  domain, coords_shell, coords_radii, kappa_wedge );
    EVDiffOp A_evdiff( domain, coords_shell, coords_radii, nu_h_wedge );

    // M_lumped = M · 1_vec  (used to project K·T → Q1-nodal lap field).
    {
        VectorQ1Scalar< ScalarType > ones( "ones", domain, mask_data );
        linalg::assign( ones, ScalarType( 1 ) );
        linalg::assign( M_lumped, ScalarType( 0 ) );
        linalg::apply( M, ones, M_lumped );
    }

    // FGMRES setup for the (M + dt·A)·T = f solve.
    constexpr int restart = 30;
    auto          table   = std::make_shared< util::Table >();
    std::vector< VectorQ1Scalar< ScalarType > > fgmres_tmps;
    fgmres_tmps.reserve( 2 * restart + 4 );
    for ( int i = 0; i < 2 * restart + 4; ++i )
        fgmres_tmps.emplace_back( "fgmres_tmp", domain, mask_data );

    const linalg::solvers::FGMRESOptions< ScalarType > fgmres_opts{
        .restart                     = restart,
        .relative_residual_tolerance = 1e-10,
        .absolute_residual_tolerance = 1e-12,
        .max_iterations              = 200,
    };
    linalg::solvers::FGMRES< AD > fgmres( fgmres_tmps, fgmres_opts );
    fgmres.set_tag( "ev_fgmres" );

    const fe::wedge::operators::shell::EntropyViscosityParameters< ScalarType > ev_params{};

    constexpr int vtk_interval = 10;
    io::XDMFOutput xdmf( "test_ev_rotation_out", domain, coords_shell, coords_radii );
    xdmf.add( T.grid_data() );
    xdmf.write();

    util::logroot << "Running EV rotation test"
                  << "  level=" << level << "  dt=" << dt
                  << "  steps=" << n_timesteps << "  T_end=" << T_end
                  << "  alpha_max=" << ev_params.alpha_max
                  << "  alpha_E=" << ev_params.alpha_E << "\n";

    ScalarType t_min_worst = kernels::common::min_entry( T.grid_data() );
    ScalarType t_max_worst = max_entry_nodes( T.grid_data() );

    for ( int ts = 1; ts <= n_timesteps; ++ts )
    {
        // --- EV pipeline (mirrors EVSolver::step) ---------------------------
        // 1) lap_T = (A_kappa · T) / M_lumped  with Dirichlet nodes zeroed.
        linalg::apply( A_kappa, T, lap_T );
        {
            auto lap_v = lap_T.grid_data();
            auto m_v   = M_lumped.grid_data();
            auto bm    = boundary_mask_data;
            Kokkos::parallel_for(
                "ev_lap_T_lumped_mass_divide",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 }, { lap_v.extent( 0 ), lap_v.extent( 1 ), lap_v.extent( 2 ), lap_v.extent( 3 ) } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                    if ( util::has_flag( bm( s, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY ) )
                    {
                        lap_v( s, x, y, r ) = ScalarType( 0 );
                        return;
                    }
                    const ScalarType m  = m_v( s, x, y, r );
                    lap_v( s, x, y, r ) = ( m > ScalarType( 0 ) ) ? ( lap_v( s, x, y, r ) / m ) : ScalarType( 0 );
                } );
            Kokkos::fence();
        }

        // 2) entropy stats and ν_h per wedge.
        const auto stats = fe::wedge::operators::shell::compute_entropy_stats(
            T, mask_data, domain, coords_shell, coords_radii, ev_params );
        fe::wedge::operators::shell::compute_nu_h(
            nu_h_wedge, T, T_prev, u, lap_T.grid_data(),
            domain, coords_shell, coords_radii, dt, stats, ev_params );

        // 3) rhs_ev = ∫ ν_h ∇T · ∇φ_i  (explicit EV stabilization).
        linalg::apply( A_evdiff, T, rhs_ev );

        // 4) f = M · T  -  dt · rhs_ev.
        linalg::apply( M, T, f );
        linalg::lincomb( f, { ScalarType( 1 ), -dt }, { f, rhs_ev } );

        // 5) History rotation BEFORE the solve overwrites T.
        Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() );

        // 6) Zero out f at boundary dofs (homogeneous Dirichlet).
        fe::strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like(
            f, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

        // 7) Solve (M + dt · A_pureGalerkin) · T^{n+1} = f.
        linalg::solvers::solve( fgmres, A, T, f );
        table->clear();

        const ScalarType l2_rel_err = compute_l2_relative_error_nodes( T.grid_data(), T_ref.grid_data(), mask_data );
        const ScalarType t_min_step = kernels::common::min_entry( T.grid_data() );
        const ScalarType t_max_step = max_entry_nodes( T.grid_data() );
        t_min_worst                 = std::min( t_min_worst, t_min_step );
        t_max_worst                 = std::max( t_max_worst, t_max_step );

        if ( ts % vtk_interval == 0 )
        {
            xdmf.write();
            util::logroot << "  ts=" << ts << "  t=" << ts * dt
                          << "  l2_rel_err=" << l2_rel_err
                          << "  T_min=" << t_min_step << "  T_max=" << t_max_step << "\n";
        }
    }
    xdmf.write();

    const ScalarType final_l2_rel = compute_l2_relative_error_nodes( T.grid_data(), T_ref.grid_data(), mask_data );

    util::logroot << "\n================================================================\n"
                  << " EV rotation summary (level=" << level << ")\n"
                  << "================================================================\n"
                  << "  final L2 relative error vs. initial: "
                  << std::scientific << std::setprecision( 6 ) << final_l2_rel << "\n"
                  << "  worst T min:                         "
                  << std::scientific << std::setprecision( 3 ) << t_min_worst << "\n"
                  << "  worst T max:                         "
                  << std::scientific << std::setprecision( 3 ) << t_max_worst << "\n";

    // Assertion: trajectory must be physical.
    //   * No NaN/Inf in final field (caught by std::isfinite of L2 error).
    //   * Final L2 relative error bounded by 0.5 — generous threshold for EV
    //     on a level-5 mesh after one full revolution with κ=0.
    if ( !std::isfinite( final_l2_rel ) )
    {
        util::logroot << "FAIL: final L2 error is not finite\n";
        return 1;
    }
    if ( final_l2_rel > 0.5 )
    {
        util::logroot << "FAIL: final L2 relative error = " << final_l2_rel
                      << " exceeds 0.5 threshold\n";
        return 1;
    }
    return 0;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    return test( 5 );
}
