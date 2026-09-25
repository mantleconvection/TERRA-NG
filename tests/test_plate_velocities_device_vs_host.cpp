/// @file
///
/// Regression test: the device plate velocity kernel must reproduce the host query exactly.
///
/// terra/plates/plate_velocity_device.hpp evaluates surface plate velocities inside a Kokkos kernel from the
/// packed stage views, where PlateVelocityProvider::getLocallyAveragedPointVelocity walks the polygon
/// containers on the host. They are independent implementations of the same formula, so any disagreement is a
/// bug in one of them -- and since the device path is the one intended to replace the host path in the mantle
/// circulation app, that has to stay true as either side changes.
///
/// Both regimes of the host query are covered, and they are checked separately because they exercise different
/// code:
///
///   * away from plate boundaries the host takes a rigid-rotation shortcut, and the kernel must match it;
///   * close to a boundary both average over the same stencil and project out the radial component, which
///     additionally requires the kernel to agree about *which* samples contribute -- a sample landing on a
///     plate with no rotation data must be skipped, not counted as a zero velocity.
///
/// That second point is not hypothetical: the packed stage stores a zero Euler vector for such plates, and an
/// earlier version of the kernel read those zeros as genuine velocities, which moved the average by ~25% near
/// the affected boundary while leaving the unaveraged regime exact.
///
/// Needs the plate reconstruction data, which is not in the repository. Pass a directory holding one .geojson
/// and one .rot, or the two files explicitly, or set TERRA_PLATE_DATA_DIR; without it the test skips (77).

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include <mpi.h>

#include "grid/bit_masks.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/plates/local_averaging_point_weight_provider.hpp"
#include "terra/plates/plate_velocity_device.hpp"
#include "terra/plates/plate_velocity_provider.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using namespace terra;
using util::logroot;

using ScalarType = double;

namespace
{

constexpr int kSkipExitCode = 77;

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        logroot << "  FAIL: " << what << std::endl;
    }
}

/// Runs both extractions on one mesh and compares them, split by averaging regime.
void compare_at_level(
    plates::PlateVelocityProvider& oracle,
    const int                      level,
    const double                   age )
{
    const auto radii = grid::shell::uniform_shell_radii< double >( 0.55, 1.0, ( 1 << level ) + 1 );
    const auto domain = grid::shell::DistributedDomain::create_uniform( level, radii, 0, 0 );

    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto radii_grid = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    auto ownership = grid::setup_node_ownership_mask_data( domain );
    auto boundary  = grid::shell::setup_boundary_mask_data( domain );

    const int num_sub = static_cast< int >( domain.subdomains().size() );
    const int n_lat   = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int n_rad   = domain.domain_info().subdomain_num_nodes_radially();

    // The stencil the mantle circulation app uses: centre plus one ring of six at 1/100 rad, sigma 0.1.
    const plates::UniformCirclesPointWeightProvider stencil_host( { { 1.0 / 100.0, 6 } }, 1e-1 );

    oracle.prepareEulerVectors( age );

    // ---- host path: the same call extract_plate_velocities() makes -----------------------------------------
    grid::Grid4DDataVec< ScalarType, 3 > host_v( "host_v", num_sub, n_lat, n_lat, n_rad );
    {
        // create_mirror_view (not ..._and_copy) so the types are exactly the host_mirror_type that
        // grid::shell::coords() static_asserts on.
        auto coords_h = Kokkos::create_mirror_view( coords );
        auto radii_h  = Kokkos::create_mirror_view( radii_grid );
        Kokkos::deep_copy( coords_h, coords );
        Kokkos::deep_copy( radii_h, radii_grid );
        auto host_h   = create_mirror( Kokkos::HostSpace{}, host_v );
        for ( int d = 0; d < 3; ++d )
            Kokkos::deep_copy( host_h.comp_[d], ScalarType( 0 ) );

        plates::DefaultPlateNotFoundHandler handler;

        for ( int sd = 0; sd < num_sub; ++sd )
            for ( int x = 0; x < n_lat; ++x )
                for ( int y = 0; y < n_lat; ++y )
                {
                    const auto c = grid::shell::coords( sd, x, y, n_rad - 1, coords_h, radii_h );
                    const auto v = oracle.getLocallyAveragedPointVelocity( c, age, stencil_host, handler );
                    for ( int d = 0; d < 3; ++d )
                        host_h( sd, x, y, n_rad - 1, d ) = v( d );
                }

        deep_copy( host_v, host_h );
    }

    // ---- device path --------------------------------------------------------------------------------------
    grid::Grid4DDataVec< ScalarType, 3 > device_v( "device_v", num_sub, n_lat, n_lat, n_rad );
    for ( int d = 0; d < 3; ++d )
        Kokkos::deep_copy( device_v.comp_[d], ScalarType( 0 ) );

    const auto& stage         = oracle.stageFor( age );
    const auto  stencil_device = plates::make_device_averaging_stencil( stencil_host );

    plates::extract_plate_velocities_device< ScalarType >(
        domain, coords, radii_grid, stage.device(), stencil_device, device_v, ScalarType( 1 ) );

    // ---- compare, split by regime -------------------------------------------------------------------------
    const auto stage_views = stage.device();
    const auto reach_km    = stencil_device.maxDistanceKm;

    ScalarType max_plain = 0, max_avg = 0, max_mag = 0;
    long long  n_plain = 0, n_avg = 0;

    Kokkos::parallel_reduce(
        "compare_host_device",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, ScalarType& plain, ScalarType& avg,
                       ScalarType& mag, long long& cnt_plain, long long& cnt_avg ) {
            if ( boundary( sd, x, y, r ) != grid::shell::ShellBoundaryFlag::SURFACE )
                return;
            if ( !util::has_flag( ownership( sd, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                return;

            const auto c = grid::shell::coords( sd, x, y, r, coords, radii_grid );
            const dense::Vec< double, 3 > lonLat =
                plates::conversions::cart2sph( dense::Vec< double, 3 >{ c( 0 ), c( 1 ), c( 2 ) } );
            const auto hit = plates::findPlateInStage(
                stage_views, plates::geometry::lonLatDegToUnit( lonLat( 0 ), lonLat( 1 ) ) );

            const bool averaged =
                hit.found && reach_km >= hit.distanceRad * plates::constants::earthRadiusInKm;

            if ( averaged )
                cnt_avg += 1;
            else
                cnt_plain += 1;

            for ( int d = 0; d < 3; ++d )
            {
                const ScalarType e =
                    Kokkos::abs( host_v( sd, x, y, r, d ) - device_v( sd, x, y, r, d ) );
                if ( averaged )
                    avg = Kokkos::max( avg, e );
                else
                    plain = Kokkos::max( plain, e );
                mag = Kokkos::max( mag, Kokkos::abs( host_v( sd, x, y, r, d ) ) );
            }
        },
        Kokkos::Max< ScalarType >( max_plain ),
        Kokkos::Max< ScalarType >( max_avg ),
        Kokkos::Max< ScalarType >( max_mag ),
        n_plain,
        n_avg );
    Kokkos::fence();

    const ScalarType rel_plain = max_mag > 0 ? max_plain / max_mag : 0;
    const ScalarType rel_avg   = max_mag > 0 ? max_avg / max_mag : 0;

    logroot << "  level " << level << ", age " << age << " Ma\n"
            << "    unaveraged nodes " << n_plain << ", max rel diff " << std::scientific
            << std::setprecision( 4 ) << rel_plain << "\n"
            << "    averaged   nodes " << n_avg << ", max rel diff " << rel_avg << std::defaultfloat
            << std::endl;

    check( max_mag > 0, "host produced an all-zero velocity field, so the comparison is vacuous" );
    check( n_plain > 0, "no nodes exercised the unaveraged path" );
    check( n_avg > 0, "no nodes exercised the averaged path, so the stencil is untested" );
    check( rel_plain < 1e-10, "device disagrees with host away from plate boundaries" );
    check( rel_avg < 1e-10, "device disagrees with host near plate boundaries (stencil or skipped samples)" );
}

} // namespace

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    std::string topologies;
    std::string reconstructions;

    if ( argc > 2 )
    {
        topologies      = argv[1];
        reconstructions = argv[2];
    }
    else
    {
        std::string dir;
        if ( argc > 1 )
            dir = argv[1];
        else if ( const char* env = std::getenv( "TERRA_PLATE_DATA_DIR" ) )
            dir = env;

        if ( !dir.empty() && std::filesystem::is_directory( dir ) )
        {
            for ( const auto& entry : std::filesystem::directory_iterator( dir ) )
            {
                const auto ext = entry.path().extension().string();
                if ( ext == ".geojson" && topologies.empty() )
                    topologies = entry.path().string();
                else if ( ext == ".rot" && reconstructions.empty() )
                    reconstructions = entry.path().string();
            }
        }
    }

    if ( topologies.empty() || reconstructions.empty() || !std::filesystem::exists( topologies ) ||
         !std::filesystem::exists( reconstructions ) )
    {
        logroot << "SKIP: plate reconstruction data not found.\n"
                << "      Pass a directory holding one .geojson and one .rot as the first argument,\n"
                << "      or the two files explicitly, or set TERRA_PLATE_DATA_DIR." << std::endl;
        return kSkipExitCode;
    }

    logroot << "topologies      : " << topologies << "\n"
            << "reconstructions : " << reconstructions << std::endl;

    plates::PlateVelocityProvider oracle( topologies, reconstructions );

    // Two resolutions: the coarse one is quick, the finer one puts many more nodes near plate boundaries and
    // so exercises the averaged branch harder.
    compare_at_level( oracle, 4, 0.0 );
    compare_at_level( oracle, 5, 0.0 );

    int failures = g_failures;
    MPI_Allreduce( MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    logroot << "\ntest_plate_velocities_device_vs_host: " << ( failures == 0 ? "PASSED" : "FAILED" )
            << std::endl;
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
