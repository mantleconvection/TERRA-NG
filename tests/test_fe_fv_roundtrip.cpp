// Test: FE -> FV -> FE round-trip consistency.
//
// Interpolate a smooth analytical function u(x, y, z) onto Q1 nodes, project
// that Q1 field to the FV space via `l2_project_fe_to_fv`, project it back to
// Q1 via `l2_project_fv_to_fe`, and measure the relative error between the
// original and the round-tripped field.  We do this at several refinement
// levels and print the per-level error so the user can check the convergence
// rate (expected: O(h^2) for smooth u, since both projections are L2 onto a
// space with local approximation power h^2).

#include <cmath>
#include <cstdio>
#include <mpi.h>
#include <vector>

#include "communication/shell/communication.hpp"
#include "fv/hex/conversion.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"
#include "mpi/mpi.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using linalg::VectorFVScalar;
using linalg::VectorQ1Scalar;

using ScalarType = double;

// Smooth analytical function u: R^3 -> R used by both the Q1 nodal interpolator
// and the FV analytical projector.  Separate KOKKOS_INLINE_FUNCTION free
// function so both callers share one definition.
KOKKOS_INLINE_FUNCTION
ScalarType smooth_u( const dense::Vec< ScalarType, 3 >& x )
{
    return Kokkos::sin( 2.0 * x( 0 ) ) * Kokkos::cos( 2.0 * x( 1 ) ) * Kokkos::sin( 2.0 * x( 2 ) );
}

// Q1 nodal interpolator — writes u(node_position) into every (sd, x, y, r) in
// the grid.  Since u is a deterministic function of node coordinates, all
// subdomain copies of a shared node receive the same value, so no ghost
// exchange is needed afterwards.
struct Q1NodalInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords =
            grid::shell::coords( sd, x, y, r, grid_, radii_ );
        data_( sd, x, y, r ) = smooth_u( coords );
    }
};

// Relative L-infinity norm over the Q1 grid.  Taking a max over all subdomain
// copies of shared nodes is robust to double-counting (copies agree by
// construction for the input field and after `l2_project_fv_to_fe` +
// `communication::shell::send_recv` for the round-tripped field).
ScalarType linf_norm_q1( const VectorQ1Scalar< ScalarType >& v )
{
    auto        grid = v.grid_data();
    ScalarType  local_max = 0.0;
    Kokkos::parallel_reduce(
        "linf_q1",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { static_cast< long long >( grid.extent( 0 ) ),
              static_cast< long long >( grid.extent( 1 ) ),
              static_cast< long long >( grid.extent( 2 ) ),
              static_cast< long long >( grid.extent( 3 ) ) } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, ScalarType& m ) {
            const ScalarType val = Kokkos::abs( grid( sd, x, y, r ) );
            if ( val > m ) m = val;
        },
        Kokkos::Max< ScalarType >( local_max ) );
    Kokkos::fence();

    ScalarType global_max = 0.0;
    MPI_Allreduce(
        &local_max, &global_max, 1, mpi::mpi_datatype< ScalarType >(), MPI_MAX, MPI_COMM_WORLD );
    return global_max;
}

struct LevelResult
{
    int        level;
    ScalarType err_linf;
    ScalarType norm_linf;
};

LevelResult run_level( const int level )
{
    const auto domain = DistributedDomain::create_uniform( level, level, 0.5, 1.0, 0, 0 );

    auto mask_data = grid::setup_node_ownership_mask_data( domain );

    const auto coords_shell =
        grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Scalar< ScalarType > u_q1_original( "u_q1_original", domain, mask_data );
    VectorQ1Scalar< ScalarType > u_q1_back( "u_q1_back", domain, mask_data );
    VectorFVScalar< ScalarType > u_fv( "u_fv", domain );

    // Step 1: interpolate the smooth function onto Q1 nodes.
    Kokkos::parallel_for(
        "interp_smooth_to_q1",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        Q1NodalInterpolator{ coords_shell, coords_radii, u_q1_original.grid_data() } );
    Kokkos::fence();

    // Step 2: project Q1 -> FV.
    fv::hex::l2_project_fe_to_fv( u_fv, u_q1_original, domain, coords_shell, coords_radii );

    // Step 3: project FV -> Q1.  Needs 5 scratch Q1 vectors.
    std::vector< VectorQ1Scalar< ScalarType > > tmps;
    for ( int i = 0; i < 5; i++ )
    {
        tmps.emplace_back( "tmp_roundtrip", domain, mask_data );
    }
    fv::hex::l2_project_fv_to_fe( u_q1_back, u_fv, domain, coords_shell, coords_radii, tmps );

    // Step 4: form the difference and measure the relative L-infinity error.
    VectorQ1Scalar< ScalarType > diff( "diff", domain, mask_data );
    linalg::lincomb( diff, { 1.0, -1.0 }, { u_q1_back, u_q1_original } );

    const ScalarType err_linf  = linf_norm_q1( diff );
    const ScalarType norm_linf = linf_norm_q1( u_q1_original );

    return { level, err_linf, norm_linf };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int mpi_rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );

    std::vector< LevelResult > results;
    for ( const int level : { 2, 3, 4, 5 } )
    {
        results.push_back( run_level( level ) );
    }

    if ( mpi_rank == 0 )
    {
        std::printf( "\nFE -> FV -> FE round-trip errors (smooth u = sin(2x)cos(2y)sin(2z)):\n" );
        std::printf( "  %5s  %12s  %12s  %12s  %10s\n",
                     "level", "err_Linf", "norm_Linf", "rel_err", "rate" );

        ScalarType prev_rel = 0.0;
        for ( std::size_t i = 0; i < results.size(); i++ )
        {
            const auto& r       = results[i];
            const auto  rel_err = r.err_linf / r.norm_linf;
            if ( i == 0 )
            {
                std::printf( "  %5d  %12.4e  %12.4e  %12.4e  %10s\n",
                             r.level, r.err_linf, r.norm_linf, rel_err, "-" );
            }
            else
            {
                const auto rate = std::log2( prev_rel / rel_err );
                std::printf( "  %5d  %12.4e  %12.4e  %12.4e  %10.3f\n",
                             r.level, r.err_linf, r.norm_linf, rel_err, rate );
            }
            prev_rel = rel_err;
        }
    }

    return 0;
}
