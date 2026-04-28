
// Test: solid-body rotation of a cone using the SUPG advection-diffusion operator.
//
// Mirrors test_finite_volume_rotation.cpp exactly in problem setup so the XDMF
// outputs can be compared side-by-side:
//   - Domain: spherical shell r_min=0.5, r_max=1.0, level 5.
//   - Velocity: solid-body rotation u = (-y, x, 0)  (ω = 1 rad/s).
//   - Initial condition: cone centred at (0.75, 0, 0), radius 0.2.
//   - Dirichlet BCs: T = 0 at CMB and surface (treat_boundary=true).
//   - Zero diffusivity.
//   - Same dt = 0.5 * 0.1 * h_min as the FV test.
//   - T_end = 2π (one full revolution → exact solution = initial condition).
//   - XDMF written every 100 steps.
//
// In addition to the FV test, reports per-timestep L2 error vs. the initial
// condition, and min/max of T across the simulation (undershoot/overshoot
// diagnostics for SUPG vs. FCT comparison).

#include <cmath>
#include <iostream>
#include <mpi.h>

#include <iomanip>
#include <limits>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/div_k_grad.hpp"
#include "fe/wedge/operators/shell/entropy_viscosity.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
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
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using ScalarType = double;

// ============================================================================
// Solid-body rotation velocity:  u = (-y, x, 0)  (angular velocity ω = 1 rad/s)
// ============================================================================

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

// ============================================================================
// Cone initial condition at (0.75, 0, 0) with radius 0.2 — evaluated at Q1 nodes.
// ============================================================================

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

// ============================================================================
// Node-based L2 relative error: ||T - T_ref||_2 / ||T_ref||_2 over owned Q1 nodes.
// ============================================================================

ScalarType compute_l2_relative_error_nodes(
    const Grid4DDataScalar< ScalarType >&                     T,
    const Grid4DDataScalar< ScalarType >&                     T_ref,
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
    MPI_Allreduce( &sum_sq_ref, &global_den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    if ( global_den < 1e-30 )
        return 0.0;
    return std::sqrt( global_num / global_den );
}

// Signed max reduction over a node-scalar field (the kernels::common lib only has
// min_entry and max_abs_entry; we need signed max to detect overshoot T_max > 1).
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

// ============================================================================
// Main test
// ============================================================================

void test( const int level )
{
    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Vec< ScalarType >    u( "u", domain, mask_data );
    VectorQ1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorQ1Scalar< ScalarType > T_prev( "T_prev", domain, mask_data ); // for EV ∂_t E
    VectorQ1Scalar< ScalarType > T_ref( "T_ref", domain, mask_data );
    VectorQ1Scalar< ScalarType > f( "f", domain, mask_data );

    // Per-cell entropy viscosity ν_h.  Shape matches
    // local_domain_md_range_policy_cells (extents: n_sub × (N−1) × (N−1) × (N_r−1)).
    const auto num_sub = static_cast< long long >( domain.subdomains().size() );
    const auto nx_c    = domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
    const auto nr_c    = domain.domain_info().subdomain_num_nodes_radially() - 1;
    grid::Grid4DDataScalar< ScalarType > nu_h( "nu_h", num_sub, nx_c, nx_c, nr_c );

    // Nodal projection of ν_h (for the DivKGrad RHS operator).
    VectorQ1Scalar< ScalarType > nu_h_nodal( "nu_h_nodal", domain, mask_data );

    // Buffer for the explicit EV stabilization contribution  rhs_ev_i = ∫ ν_h ∇T · ∇φ_i dx.
    VectorQ1Scalar< ScalarType > rhs_ev( "rhs_ev", domain, mask_data );

    // Toggle for this first experiment.  When false, we compute ν_h each step
    // for diagnostic purposes but do NOT feed it back — reproduces the plain
    // SUPG baseline.  When true, subtract dt·DivKGrad(ν_h)·T^n from the RHS.
    // Compile-time for now so we can A/B inside one test binary later.
    constexpr bool ev_apply = true;

    // Initialise velocity.
    Kokkos::parallel_for(
        "velocity_init",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator{ coords_shell, coords_radii, u.grid_data() } );
    Kokkos::fence();

    // Initialise temperature (cone at Q1 nodes) and save the initial state as
    // reference — after one full revolution with zero diffusion the exact solution
    // equals the initial condition, so T_ref doubles as "exact solution at T_end".
    Kokkos::parallel_for(
        "temperature_init",
        local_domain_md_range_policy_nodes( domain ),
        InitialConditionInterpolator{ coords_shell, coords_radii, T.grid_data() } );
    Kokkos::fence();
    Kokkos::deep_copy( T_ref.grid_data(), T.grid_data() );
    Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() ); // first-step bootstrap: ∂_t E = 0

    // Time-stepping parameters — identical to the FV rotation test.
    const ScalarType h           = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt          = 0.5 * 0.1 * h;
    const ScalarType T_end       = 2.0 * M_PI; // one full revolution
    const int        n_timesteps = static_cast< int >( std::ceil( T_end / dt ) );

    const ScalarType diffusivity = ScalarType( 0 );

    // SUPG operator:  A * T = (M + dt * A_advdiff) * T
    // treat_boundary=true enforces zero Dirichlet BCs at CMB and surface.
    //
    // For the EV pipeline we DISABLE SUPG stabilization at runtime — ASPECT
    // (KHB 2012 §3.2.6) uses pure Galerkin for advection-diffusion, relying
    // entirely on ν_h for stabilization.  Keeping the SUPG τ-term active in
    // parallel double-counts the stabilization and over-diffuses (as seen in
    // the first hybrid experiment: peak decayed 45% faster than the SUPG
    // baseline at t=1000).
    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;
    AD A( domain, coords_shell, coords_radii, boundary_mask_data, u, diffusivity, dt,
          /*treat_boundary=*/true );
    if constexpr ( ev_apply )
    {
        A.set_supg_enabled( false ); // pure Galerkin advection-diffusion on the LHS
    }

    // Mass operator for RHS assembly:  f = M * T^n.
    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;
    Mass M( domain, coords_shell, coords_radii, false );

    // DivKGrad operator used ONLY to assemble the explicit EV contribution
    //   rhs_ev_i = ∫ ν_h ∇T · ∇φ_i dx.
    // treat_boundary=false: we handle Dirichlet via the separate zeroing pass
    // on `f` below, so the DivKGrad contribution should be an ordinary
    // interior integral.  diagonal=false: we want the full matvec, not a
    // diagonal preconditioner.
    using DivKGrad = fe::wedge::operators::shell::DivKGrad< ScalarType >;
    DivKGrad A_ev( domain, coords_shell, coords_radii, boundary_mask_data,
                   nu_h_nodal.grid_data(),
                   /*treat_boundary=*/false,
                   /*diagonal=*/false );

    // FGMRES(30) — 2*30 + 4 = 64 temporaries.
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
    fgmres.set_tag( "supg_fgmres" );

    // XDMF output.
    constexpr int vtk_interval = 10;
    io::XDMFOutput xdmf( "test_supg_rotation_out", domain, coords_shell, coords_radii );
    xdmf.add( T.grid_data() );
    xdmf.write(); // initial condition

    util::logroot << "Running SUPG rotation test"
                  << "  level=" << level << "  dt=" << dt
                  << "  steps=" << n_timesteps << "  T_end=" << T_end << "\n";

    // Sanity: at t=0, T == T_ref so the L2-relative error should be zero.
    // Worst-case min/max over the simulation (initial cone has min=0, max=1).
    ScalarType t_min_worst = kernels::common::min_entry( T.grid_data() );
    ScalarType t_max_worst = max_entry_nodes( T.grid_data() );
    util::logroot << "  ts=0  t=0  l2_rel_err=0  T_min=" << t_min_worst
                  << "  T_max=" << t_max_worst << "\n";

    // Entropy-viscosity diagnostics (standalone; not yet fed back into the solve).
    // Computes T_min, T_max, T_m, E_avg, D once per step so we can verify the
    // values before wiring the per-cell ν_h kernel into the RHS.
    const fe::wedge::operators::shell::EntropyViscosityParameters< ScalarType > ev_params{};
    {
        const auto stats =
            fe::wedge::operators::shell::compute_entropy_stats( T, mask_data, ev_params );
        util::logroot << "  ev  ts=0  t=0"
                      << "  T_min=" << stats.T_min << "  T_max=" << stats.T_max
                      << "  T_m="   << stats.T_m
                      << "  E_avg=" << stats.E_avg << "  D=" << stats.D << "\n";
    }

    for ( int ts = 1; ts <= n_timesteps; ++ts )
    {
        // --- Entropy-viscosity stabilization (explicit, lagged) -----------
        // On entry T = T^{n-1}, T_prev = T^{n-2}.  Compute ν_h from this pair
        // (so ∂_t E uses both past timesteps), then capture T^{n-1} into T_prev
        // BEFORE the solve overwrites T — that way next iteration sees the
        // correct (T, T_prev) = (T^n, T^{n-1}).
        if constexpr ( ev_apply )
        {
            const auto stats =
                fe::wedge::operators::shell::compute_entropy_stats( T, mask_data, ev_params );
            fe::wedge::operators::shell::compute_nu_h(
                nu_h, T, T_prev, u, domain, coords_shell, coords_radii, dt, stats, ev_params );
            fe::wedge::operators::shell::project_nu_h_to_nodes( nu_h_nodal, nu_h, domain );

            // rhs_ev = ∫ ν_h ∇T^n · ∇φ dx (assembled by DivKGrad with
            // src=T^n; reads nu_h_nodal.grid_data() internally).
            linalg::apply( A_ev, T, rhs_ev );
        }

        // History rotation: capture current T (= T^{n-1}) into T_prev now,
        // BEFORE the solve overwrites T.  Doing this unconditionally keeps
        // T_prev tracking (T) regardless of ev_apply.
        Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() );

        // f = M * T^n
        linalg::apply( M, T, f );

        // f -= dt * rhs_ev  (explicit EV stabilization moved to RHS)
        if constexpr ( ev_apply )
        {
            linalg::lincomb( f, { ScalarType( 1 ), -dt }, { f, rhs_ev } );
        }

        // Zero out f at boundary dofs (homogeneous Dirichlet BCs).
        // A has identity rows at boundary nodes; without this, T^{n+1}[boundary] = f[boundary] = (M*T^n)[boundary] ≠ 0.
        fe::strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like(
            f, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );
        // Solve (M + dt * A_advdiff) * T^{n+1} = f
        linalg::solvers::solve( fgmres, A, T, f );

        table->clear();

        // Diagnostics: relative L2 error against T_ref (exact solution after 2π),
        // and running min/max for over/undershoot tracking.
        const ScalarType l2_rel_err = compute_l2_relative_error_nodes( T.grid_data(), T_ref.grid_data(), mask_data );
        const ScalarType t_min_step = kernels::common::min_entry( T.grid_data() );
        const ScalarType t_max_step = max_entry_nodes( T.grid_data() );
        t_min_worst                 = std::min( t_min_worst, t_min_step );
        t_max_worst                 = std::max( t_max_worst, t_max_step );

        if ( ts % vtk_interval == 0 )
            xdmf.write();

        // Entropy-viscosity diagnostics (ν_h was already computed above).
        if ( ts % vtk_interval == 0 )
        {
            ScalarType nu_min = kernels::common::min_entry( nu_h );
            ScalarType nu_max = kernels::common::max_abs_entry( nu_h );
            util::logroot << "  ev  ts=" << ts << "  t=" << ts * dt
                          << "  nu_h_min=" << nu_min << "  nu_h_max=" << nu_max << "\n";
        }

        util::logroot << "  ts=" << ts << "  t=" << ts * dt
                      << "  l2_rel_err=" << l2_rel_err
                      << "  T_min=" << t_min_step << "  T_max=" << t_max_step << "\n";
    }

    xdmf.write(); // final state

    // After one full revolution, T should equal T_ref. The final L2 error is the
    // headline accuracy metric; t_{min,max}_worst capture the worst over/undershoot.
    const ScalarType final_l2_rel = compute_l2_relative_error_nodes( T.grid_data(), T_ref.grid_data(), mask_data );

    util::logroot << "\n================================================================\n"
                  << " SUPG rotation summary (level=" << level << ")\n"
                  << "================================================================\n"
                  << "  final L2 relative error vs. initial condition: "
                  << std::scientific << std::setprecision( 6 ) << final_l2_rel << "\n"
                  << "  worst T min (exact range [0,1]):               "
                  << std::scientific << std::setprecision( 3 ) << t_min_worst << "\n"
                  << "  worst T max (exact range [0,1]):               "
                  << std::scientific << std::setprecision( 3 ) << t_max_worst << std::endl;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    // Match test_finite_volume_rotation.cpp: level 5.
    test( 5 );
    return 0;
}
