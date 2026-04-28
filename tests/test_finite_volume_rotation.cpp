
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <string>

#include "../src/terra/communication/shell/communication.hpp"
#include "fv/hex/conversion.hpp"
#include "fv/hex/helpers.hpp"
#include "fv/hex/operators/advection_diffusion.hpp"
#include "fv/hex/operators/fct_advection_diffusion.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/solver.hpp"
#include "linalg/vector_fv.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/bit_masks.hpp"
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
using grid::shell::DistributedDomain;
using linalg::VectorFVScalar;
using linalg::VectorFVVec;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using ScalarType = double;

// ============================================================================
// Method selection
// ============================================================================

enum class Method
{
    Upwind,
    FCTExplicit,
    FCTSemiImplicit,
};

std::string method_name( Method m )
{
    switch ( m )
    {
    case Method::Upwind:
        return "upwind";
    case Method::FCTExplicit:
        return "fct_explicit";
    case Method::FCTSemiImplicit:
        return "fct_semiimplicit";
    }
    return "unknown";
}

// ============================================================================
// Solid-body rotation velocity:  u = (-y, x, 0)  (angular velocity ω = 1 rad/s)
// One full revolution takes T = 2π seconds.
// ============================================================================

struct VelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 >       grid_;
    Grid2DDataScalar< ScalarType >       radii_;
    grid::Grid4DDataVec< ScalarType, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > c = grid::shell::coords( id, x, y, r, grid_, radii_ );
        data_( id, x, y, r, 0 )             = -c( 1 );
        data_( id, x, y, r, 1 )             = c( 0 );
        data_( id, x, y, r, 2 )             = 0.0;
    }
};

// ============================================================================
// Cone initial condition centred at (0.75, 0, 0) with radius 0.2.
// ============================================================================

struct InitialConditionInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        // Approximate cell centre as average of its 8 corner nodes.
        dense::Vec< ScalarType, 3 > c = {};
        for ( int dx = -1; dx < 1; ++dx )
            for ( int dy = -1; dy < 1; ++dy )
                for ( int dr = -1; dr < 1; ++dr )
                    c = c + grid::shell::coords( id, x + dx, y + dy, r + dr, grid_, radii_ );
        c = ScalarType( 0.125 ) * c;

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
// L2 error relative to a reference field (cell-count based, no cell volume weighting).
// ============================================================================

ScalarType compute_l2_relative_error(
    const DistributedDomain&              domain,
    const Grid4DDataScalar< ScalarType >& T,
    const Grid4DDataScalar< ScalarType >& T_ref )
{
    ScalarType sum_sq_diff = 0;
    ScalarType sum_sq_ref  = 0;

    Kokkos::parallel_reduce(
        "l2_err_num",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( const int id, const int x, const int y, const int r, ScalarType& acc ) {
            const ScalarType d = T( id, x, y, r ) - T_ref( id, x, y, r );
            acc += d * d;
        },
        sum_sq_diff );

    Kokkos::parallel_reduce(
        "l2_err_den",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( const int id, const int x, const int y, const int r, ScalarType& acc ) {
            const ScalarType v = T_ref( id, x, y, r );
            acc += v * v;
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

// Min / signed-max of an FV cell field, skipping ghost layers, with MPI reduction.
// Used to diagnose over/undershoot of T (exact range [0, 1]).
ScalarType fv_min_cells( const DistributedDomain& domain, const Grid4DDataScalar< ScalarType >& T )
{
    ScalarType mn = std::numeric_limits< ScalarType >::max();
    Kokkos::parallel_reduce(
        "fv_min_cells",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( int id, int x, int y, int r, ScalarType& lmn ) {
            lmn = Kokkos::min( lmn, T( id, x, y, r ) );
        },
        Kokkos::Min< ScalarType >( mn ) );
    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &mn, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );
    return mn;
}

ScalarType fv_max_cells( const DistributedDomain& domain, const Grid4DDataScalar< ScalarType >& T )
{
    ScalarType mx = std::numeric_limits< ScalarType >::lowest();
    Kokkos::parallel_reduce(
        "fv_max_cells",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( int id, int x, int y, int r, ScalarType& lmx ) {
            lmx = Kokkos::max( lmx, T( id, x, y, r ) );
        },
        Kokkos::Max< ScalarType >( mx ) );
    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return mx;
}

// ============================================================================
// Main test function
// ============================================================================

void test( const int level, const Method method )
{
    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    // FE velocity lives on Q1 nodes.
    VectorQ1Vec< ScalarType > u( "u", domain, mask_data );

    // FV scalar fields.
    VectorFVScalar< ScalarType > T( "T", domain );         // current solution
    VectorFVScalar< ScalarType > T_ref( "T_ref", domain ); // reference (initial condition)

    // Cell centres for geometry.
    VectorFVVec< ScalarType, 3 > cell_centers( "cell_centers", domain );

    // FE field for XDMF output.
    VectorQ1Scalar< ScalarType >                T_fe( "T_fe", domain, mask_data );
    std::vector< VectorQ1Scalar< ScalarType > > tmps_fe;
    for ( int i = 0; i < 5; ++i )
        tmps_fe.emplace_back( "tmpfe", domain, mask_data );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    // Compute cell centres once (also populates ghost layers via MPI).
    fv::hex::initialize_cell_centers( cell_centers, domain, coords_shell, coords_radii );

    // Initialise velocity field on nodes.
    Kokkos::parallel_for(
        "velocity_init",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator{ coords_shell, coords_radii, u.grid_data() } );
    Kokkos::fence();

    // Initialise temperature (cone) and save as reference.
    Kokkos::parallel_for(
        "temperature_init",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        InitialConditionInterpolator{ coords_shell, coords_radii, T.grid_data() } );
    Kokkos::fence();
    Kokkos::deep_copy( T_ref.grid_data(), T.grid_data() );

    // Pre-allocated scratch buffers (reused every timestep).
    fv::hex::operators::FVFCTBuffers< ScalarType > bufs( domain );

    // Time-stepping parameters.
    // CFL safety: at level 5, max |u| ≈ 1 at R=1.  Cell size ≈ π/(2*2^level).
    // Estimated CFL ≈ dt * 27 at level 5; dt=0.01 → CFL ≈ 0.27.
    const auto       h           = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt          = 0.5 * 0.1 * h;
    const ScalarType T_end       = 2.0 * M_PI; // one full revolution
    const int        n_timesteps = static_cast< int >( std::ceil( T_end / dt ) );

    const auto diffusivity = ScalarType( 0 );

    // Semi-implicit setup (constructed only if needed).
    using AD     = fv::hex::operators::UnsteadyAdvectionDiffusion< ScalarType >;
    using FGMRES = linalg::solvers::FGMRES< AD >;

    std::unique_ptr< AD >                           ad_op;
    std::unique_ptr< FGMRES >                       fgmres;
    std::unique_ptr< VectorFVScalar< ScalarType > > T_L;
    std::unique_ptr< VectorFVScalar< ScalarType > > rhs;

    if ( method == Method::FCTSemiImplicit )
    {
        ad_op = std::make_unique< AD >(
            domain, coords_shell, coords_radii, cell_centers.grid_data(), boundary_mask_data, u, diffusivity, dt );
        T_L = std::make_unique< VectorFVScalar< ScalarType > >( "T_L", domain );
        rhs = std::make_unique< VectorFVScalar< ScalarType > >( "rhs", domain );

        // FGMRES(m) requires 2*m + 4 temporary vectors.
        constexpr int                               restart = 30;
        std::vector< VectorFVScalar< ScalarType > > fgmres_tmps;
        fgmres_tmps.reserve( 2 * restart + 4 );
        for ( int i = 0; i < 2 * restart + 4; ++i )
            fgmres_tmps.emplace_back( "fgmres_tmp", domain );

        const linalg::solvers::FGMRESOptions< ScalarType > fgmres_opts{
            .restart                     = restart,
            .relative_residual_tolerance = 1e-10,
            .absolute_residual_tolerance = 1e-12,
            .max_iterations              = 200,
        };
        fgmres = std::make_unique< FGMRES >( fgmres_tmps, fgmres_opts );
    }

    // VTK output every 'vtk_interval' timesteps.
    constexpr int  vtk_interval = 100;
    constexpr bool vtk          = true;

    const std::string out_name = "test_fv_rotation_" + method_name( method ) + "_out";
    io::XDMFOutput    xdmf( out_name, domain, coords_shell, coords_radii );
    xdmf.add( T_fe.grid_data() );

    if ( vtk )
    {
        fv::hex::l2_project_fv_to_fe( T_fe, T, domain, coords_shell, coords_radii, tmps_fe );
        xdmf.write();
    }

    // Dirichlet BCs: pin T = 0 at both the CMB and the outer surface.
    const fv::hex::DirichletBCs< ScalarType > bcs{
        .T_cmb        = ScalarType( 0 ),
        .T_surface    = ScalarType( 0 ),
        .apply_cmb    = true,
        .apply_surface = true,
    };

    util::logroot << "Running solid-body rotation test [" << method_name( method ) << "]"
                  << "  level=" << level << "  dt=" << dt << "  steps=" << n_timesteps << "  T_end=" << T_end << "\n";

    // Worst-case under-/overshoot across the run (initial cone is in [0, 1]).
    ScalarType t_min_worst = fv_min_cells( domain, T.grid_data() );
    ScalarType t_max_worst = fv_max_cells( domain, T.grid_data() );

    for ( int ts = 1; ts <= n_timesteps; ++ts )
    {
        switch ( method )
        {
        case Method::Upwind:
            fv::hex::operators::upwind_explicit_step(
                domain, T, u, cell_centers.grid_data(), coords_shell, coords_radii, dt, bufs, diffusivity );
            break;

        case Method::FCTExplicit:
            fv::hex::operators::fct_explicit_step(
                domain, T, u, cell_centers.grid_data(), coords_shell, coords_radii, dt, bufs, diffusivity );
            break;

        case Method::FCTSemiImplicit:
            // RHS = M * T^n
            ad_op->compute_rhs( T, *rhs );
            // Solve (M + dt*A_upwind) * T_L = M * T^n
            linalg::solvers::solve( *fgmres, *ad_op, *T_L, *rhs );
            // FCT correction: antidiff from T^n, Zalesak on T_L
            fv::hex::operators::fct_semiimplicit_step(
                domain, T, *T_L, u, cell_centers.grid_data(), coords_shell, coords_radii, dt, bufs );
            break;
        }

        fv::hex::apply_dirichlet_bcs( T, boundary_mask_data, bcs, domain );

        if ( vtk && ts % vtk_interval == 0 )
        {
            fv::hex::l2_project_fv_to_fe( T_fe, T, domain, coords_shell, coords_radii, tmps_fe );
            xdmf.write();
        }

        const ScalarType t     = ts * dt;
        const ScalarType error = compute_l2_relative_error( domain, T.grid_data(), T_ref.grid_data() );
        const ScalarType t_min = fv_min_cells( domain, T.grid_data() );
        const ScalarType t_max = fv_max_cells( domain, T.grid_data() );
        t_min_worst            = std::min( t_min_worst, t_min );
        t_max_worst            = std::max( t_max_worst, t_max );
        util::logroot << "  ts=" << ts << "  t=" << t << "  rel_l2_err=" << error
                      << "  T_min=" << t_min << "  T_max=" << t_max << "\n";
    }

    // Final relative L2 error after one full revolution.
    const ScalarType final_error = compute_l2_relative_error( domain, T.grid_data(), T_ref.grid_data() );

    util::logroot << "\n================================================================\n"
                  << " FV rotation summary (method=" << method_name( method )
                  << ", level=" << level << ")\n"
                  << "================================================================\n"
                  << "  final L2 relative error vs. initial condition: "
                  << std::scientific << std::setprecision( 6 ) << final_error << "\n"
                  << "  worst T min (exact range [0,1]):               "
                  << std::scientific << std::setprecision( 3 ) << t_min_worst << "\n"
                  << "  worst T max (exact range [0,1]):               "
                  << std::scientific << std::setprecision( 3 ) << t_max_worst << std::endl;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    // Select method via command-line argument:
    //   argv[1] = "upwind" | "fct_explicit" | "fct_semiimplicit"  (default: upwind)
    Method method = Method::Upwind;
    if ( argc > 1 )
    {
        const std::string arg( argv[1] );
        if ( arg == "fct_explicit" )
            method = Method::FCTExplicit;
        else if ( arg == "fct_semiimplicit" )
            method = Method::FCTSemiImplicit;
    }

    const int level = 5;
    test( level, method );

    return 0;
}