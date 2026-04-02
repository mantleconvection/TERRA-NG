
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
//   - T_end = 2π (one full revolution).
//   - XDMF written every 100 steps.

#include <cmath>
#include <iostream>
#include <mpi.h>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/solver.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
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
    VectorQ1Scalar< ScalarType > f( "f", domain, mask_data );

    // Initialise velocity.
    Kokkos::parallel_for(
        "velocity_init",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator{ coords_shell, coords_radii, u.grid_data() } );
    Kokkos::fence();

    // Initialise temperature (cone at Q1 nodes).
    Kokkos::parallel_for(
        "temperature_init",
        local_domain_md_range_policy_nodes( domain ),
        InitialConditionInterpolator{ coords_shell, coords_radii, T.grid_data() } );
    Kokkos::fence();

    // Time-stepping parameters — identical to the FV rotation test.
    const ScalarType h           = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt          = 0.5 * 0.1 * h;
    const ScalarType T_end       = 2.0 * M_PI; // one full revolution
    const int        n_timesteps = static_cast< int >( std::ceil( T_end / dt ) );

    const ScalarType diffusivity = ScalarType( 0 );

    // SUPG operator:  A * T = (M + dt * A_advdiff) * T
    // treat_boundary=true enforces zero Dirichlet BCs at CMB and surface.
    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;
    AD A( domain, coords_shell, coords_radii, boundary_mask_data, u, diffusivity, dt,
          /*treat_boundary=*/true );

    // Mass operator for RHS assembly:  f = M * T^n.
    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;
    Mass M( domain, coords_shell, coords_radii, false );

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

    for ( int ts = 1; ts <= n_timesteps; ++ts )
    {
        // f = M * T^n
        linalg::apply( M, T, f );
        // Zero out f at boundary dofs (homogeneous Dirichlet BCs).
        // A has identity rows at boundary nodes; without this, T^{n+1}[boundary] = f[boundary] = (M*T^n)[boundary] ≠ 0.
        fe::strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like(
            f, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );
        // Solve (M + dt * A_advdiff) * T^{n+1} = f
        linalg::solvers::solve( fgmres, A, T, f );

        table->clear();

        if ( ts % vtk_interval == 0 )
            xdmf.write();

        util::logroot << "  ts=" << ts << "  t=" << ts * dt << "\n";
    }

    xdmf.write(); // final state
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    test( 4 );
    return 0;
}
