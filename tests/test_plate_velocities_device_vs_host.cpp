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

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>

#include "grid/bit_masks.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/plates/local_averaging_point_weight_provider.hpp"
#include "terra/plates/plate_velocity_calculator.hpp"
#include "terra/plates/plate_velocity_provider.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using namespace terra;
using util::logroot;

using ScalarType = double;

namespace {

constexpr int kSkipExitCode = 77;

int g_failures = 0;

/// XDMF output controls (set in main from the environment).
std::string g_xdmf_dir    = "test_plate_velocities_xdmf"; // empty => no output
int         g_xdmf_stride = 50;                           // write every Nth age; ages with mismatches always written

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        logroot << "  FAIL: " << what << std::endl;
    }
}

/// "Not found" sentinel shared by both paths, so found/not-found disagreements show up as id mismatches.
constexpr long long kNoPlate = -1;

/// Runs both extractions on one mesh and compares them, split by averaging regime.
void compare_at_level( plates::PlateVelocityProvider& oracle, const int level, const double age )
{
    const auto radii  = grid::shell::uniform_shell_radii< double >( 0.55, 1.0, ( 1 << level ) + 1 );
    const auto domain = grid::shell::DistributedDomain::create_uniform( level, radii, 0, 0 );

    const auto coords     = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
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
        auto host_h = create_mirror( Kokkos::HostSpace{}, host_v );
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

    const auto& stage          = oracle.stageFor( age );
    const auto  stencil_device = plates::make_averaging_stencil( stencil_host );

    plates::extract_plate_velocities< ScalarType >(
        domain, coords, radii_grid, stage.data(), stencil_device, device_v, ScalarType( 1 ) );

    // ---- compare, split by regime -------------------------------------------------------------------------
    const auto stage_views = stage.data();
    const auto reach_km    = stencil_device.maxDistanceKm;

    ScalarType max_plain = 0, max_avg = 0, max_mag = 0;
    long long  n_plain = 0, n_avg = 0;

    Kokkos::parallel_reduce(
        "compare_host_device",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA(
            const int   sd,
            const int   x,
            const int   y,
            const int   r,
            ScalarType& plain,
            ScalarType& avg,
            ScalarType& mag,
            long long&  cnt_plain,
            long long&  cnt_avg ) {
            if ( boundary( sd, x, y, r ) != grid::shell::ShellBoundaryFlag::SURFACE )
                return;
            if ( !util::has_flag( ownership( sd, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                return;

            const auto                    c = grid::shell::coords( sd, x, y, r, coords, radii_grid );
            const dense::Vec< double, 3 > lonLat =
                plates::conversions::cart2sph( dense::Vec< double, 3 >{ c( 0 ), c( 1 ), c( 2 ) } );
            const auto hit =
                plates::findPlateInStage( stage_views, plates::geometry::lonLatDegToUnit( lonLat( 0 ), lonLat( 1 ) ) );

            const bool averaged = hit.found && reach_km >= hit.distanceRad * plates::constants::earthRadiusInKm;

            if ( averaged )
                cnt_avg += 1;
            else
                cnt_plain += 1;

            for ( int d = 0; d < 3; ++d )
            {
                const ScalarType e = Kokkos::abs( host_v( sd, x, y, r, d ) - device_v( sd, x, y, r, d ) );
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
            << "    unaveraged nodes " << n_plain << ", max rel diff " << std::scientific << std::setprecision( 4 )
            << rel_plain << "\n"
            << "    averaged   nodes " << n_avg << ", max rel diff " << rel_avg << std::defaultfloat << std::endl;

    // ---- plate id comparison at ALL surface nodes ----------------------------------------------------------
    using Field = grid::Grid4DDataScalar< ScalarType >;

    // Plate id fields (device views + host mirrors) so the ids can be written to XDMF.
    // Interior nodes stay 0; only the surface layer is meaningful. Not-found is kNoPlate (-1).
    Field host_pid( "host_plate_id", num_sub, n_lat, n_lat, n_rad );
    Field dev_pid( "device_plate_id", num_sub, n_lat, n_lat, n_rad );
    Field pid_diff( "plate_id_differs", num_sub, n_lat, n_lat, n_rad );

    auto host_pid_m = Kokkos::create_mirror_view( host_pid );
    auto dev_pid_m  = Kokkos::create_mirror_view( dev_pid );
    auto diff_pid_m = Kokkos::create_mirror_view( pid_diff );
    Kokkos::deep_copy( host_pid_m, ScalarType( 0 ) );
    Kokkos::deep_copy( dev_pid_m, ScalarType( 0 ) );
    Kokkos::deep_copy( diff_pid_m, ScalarType( 0 ) );

    // Device: run findPlateInStage on every surface node (owned or not).
    Kokkos::View< long long***, Kokkos::DefaultExecutionSpace > dev_ids( "dev_plate_ids", num_sub, n_lat, n_lat );

    const int r_surf = n_rad - 1;
    Kokkos::parallel_for(
        "plate_ids_device",
        Kokkos::MDRangePolicy< Kokkos::DefaultExecutionSpace, Kokkos::Rank< 3 > >(
            { 0, 0, 0 }, { num_sub, n_lat, n_lat } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y ) {
            const auto                    c = grid::shell::coords( sd, x, y, r_surf, coords, radii_grid );
            const dense::Vec< double, 3 > lonLat =
                plates::conversions::cart2sph( dense::Vec< double, 3 >{ c( 0 ), c( 1 ), c( 2 ) } );
            const auto hit =
                plates::findPlateInStage( stage_views, plates::geometry::lonLatDegToUnit( lonLat( 0 ), lonLat( 1 ) ) );
            dev_ids( sd, x, y ) = hit.found ? static_cast< long long >( hit.plateId ) : kNoPlate;
        } );
    Kokkos::fence();

    auto dev_ids_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, dev_ids );

    // Host: query the oracle at the same points.
    long long n_id_total = 0, n_id_mismatch = 0, n_id_found_mismatch = 0;
    {
        auto coords_h = Kokkos::create_mirror_view( coords );
        auto radii_h  = Kokkos::create_mirror_view( radii_grid );
        Kokkos::deep_copy( coords_h, coords );
        Kokkos::deep_copy( radii_h, radii_grid );

        constexpr int kMaxReport = 5;
        int           reported   = 0;

        for ( int sd = 0; sd < num_sub; ++sd )
            for ( int x = 0; x < n_lat; ++x )
                for ( int y = 0; y < n_lat; ++y )
                {
                    const auto          c = grid::shell::coords( sd, x, y, r_surf, coords_h, radii_h );
                    const dense::Vec< double, 3 > cart{ c( 0 ), c( 1 ), c( 2 ) };

                    const auto      raw = oracle.findPlateID( cart, age );
                    const long long id_host =
                        ( raw == oracle.idWhenNoPlateFound ) ? kNoPlate : static_cast< long long >( raw );
                    const long long id_dev = dev_ids_h( sd, x, y );

                    host_pid_m( sd, x, y, r_surf ) = static_cast< ScalarType >( id_host );
                    dev_pid_m( sd, x, y, r_surf )  = static_cast< ScalarType >( id_dev );

                    ++n_id_total;
                    if ( id_host != id_dev )
                    {
                        diff_pid_m( sd, x, y, r_surf ) = ScalarType( 1 );
                        ++n_id_mismatch;
                        if ( ( id_host == kNoPlate ) != ( id_dev == kNoPlate ) )
                            ++n_id_found_mismatch;

                        if ( reported++ < kMaxReport )
                        {
                            const auto ll = plates::conversions::cart2sph( cart );
                            logroot << "    plate id mismatch at sd " << sd << " (" << x << "," << y << ")"
                                    << " lon/lat " << ll( 0 ) << "/" << ll( 1 ) << ": host " << id_host << ", device "
                                    << id_dev << std::endl;
                        }
                    }
                }
    }

    Kokkos::deep_copy( host_pid, host_pid_m );
    Kokkos::deep_copy( dev_pid, dev_pid_m );
    Kokkos::deep_copy( pid_diff, diff_pid_m );

    logroot << "    plate ids: " << n_id_total << " nodes, " << n_id_mismatch << " mismatches (" << n_id_found_mismatch
            << " found/not-found disagreements)" << std::endl;

    check( n_id_total > 0, "no nodes were compared for plate ids" );
    check( n_id_mismatch == 0, "device and host disagree on the plate id at some surface nodes" );

    // ---- XDMF output of host vs device velocities and plate ids -------------------------------------------
    {
        // All ranks must agree on whether to write (the write may be collective).
        int local_bad = ( rel_plain >= 1e-10 || rel_avg >= 1e-10 || n_id_mismatch > 0 ) ? 1 : 0;
        int any_bad   = 0;
        MPI_Allreduce( &local_bad, &any_bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );

        const bool periodic = g_xdmf_stride > 0 && ( static_cast< int >( age ) % g_xdmf_stride ) == 0;

        if ( !g_xdmf_dir.empty() && ( periodic || any_bad ) )
        {
            Field host_vx( "host_vel_x", num_sub, n_lat, n_lat, n_rad );
            Field host_vy( "host_vel_y", num_sub, n_lat, n_lat, n_rad );
            Field host_vz( "host_vel_z", num_sub, n_lat, n_lat, n_rad );
            Field dev_vx( "device_vel_x", num_sub, n_lat, n_lat, n_rad );
            Field dev_vy( "device_vel_y", num_sub, n_lat, n_lat, n_rad );
            Field dev_vz( "device_vel_z", num_sub, n_lat, n_lat, n_rad );
            Field vel_diff( "vel_diff_magnitude", num_sub, n_lat, n_lat, n_rad );

            Kokkos::parallel_for(
                "fill_xdmf_velocity_fields",
                grid::shell::local_domain_md_range_policy_nodes( domain ),
                KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                    const ScalarType hx = host_v( sd, x, y, r, 0 );
                    const ScalarType hy = host_v( sd, x, y, r, 1 );
                    const ScalarType hz = host_v( sd, x, y, r, 2 );
                    const ScalarType dx = device_v( sd, x, y, r, 0 );
                    const ScalarType dy = device_v( sd, x, y, r, 1 );
                    const ScalarType dz = device_v( sd, x, y, r, 2 );

                    host_vx( sd, x, y, r ) = hx;
                    host_vy( sd, x, y, r ) = hy;
                    host_vz( sd, x, y, r ) = hz;
                    dev_vx( sd, x, y, r )  = dx;
                    dev_vy( sd, x, y, r )  = dy;
                    dev_vz( sd, x, y, r )  = dz;

                    vel_diff( sd, x, y, r ) = Kokkos::sqrt(
                        ( hx - dx ) * ( hx - dx ) + ( hy - dy ) * ( hy - dy ) + ( hz - dz ) * ( hz - dz ) );
                } );
            Kokkos::fence();

            std::ostringstream dir;
            dir << g_xdmf_dir << "/level" << level << "_age" << std::setw( 3 ) << std::setfill( '0' )
                << static_cast< int >( age );
            std::filesystem::create_directories( dir.str() );

            io::XDMFOutput xdmf_output( dir.str(), domain, coords, radii_grid );
            xdmf_output.add( host_vx );
            xdmf_output.add( host_vy );
            xdmf_output.add( host_vz );
            xdmf_output.add( dev_vx );
            xdmf_output.add( dev_vy );
            xdmf_output.add( dev_vz );
            xdmf_output.add( vel_diff );
            xdmf_output.add( host_pid );
            xdmf_output.add( dev_pid );
            xdmf_output.add( pid_diff );
            xdmf_output.write( age );

            logroot << "    wrote XDMF to " << dir.str() << std::endl;
        }
    }

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

    if ( const char* env = std::getenv( "TERRA_TEST_XDMF_DIR" ) )
        g_xdmf_dir = env; // set to an empty string to disable output
    if ( const char* env = std::getenv( "TERRA_TEST_XDMF_STRIDE" ) )
        g_xdmf_stride = std::max( 1, std::atoi( env ) );

    // Two resolutions: the coarse one is quick, the finer one puts many more nodes near plate boundaries and
    // so exercises the averaged branch harder.
    for ( int i = 0; i <= 410; i++ )
    {
        // compare_at_level( oracle, 4, i );
        compare_at_level( oracle, 5, i );
    }

    int failures = g_failures;
    MPI_Allreduce( MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    logroot << "\ntest_plate_velocities_device_vs_host: " << ( failures == 0 ? "PASSED" : "FAILED" ) << std::endl;
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
