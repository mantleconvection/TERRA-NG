/*
 * Copyright (c) 2017-2026 Dominik Thoennes, Nils Kohl, Marcus Mohr, Fatemeh Rezaei.
 *
 * This file is part of HyTeG
 * (see https://i10git.cs.fau.de/hyteg/hyteg).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

/// @file
///
/// End-to-end demo of the plate machinery: reconstruction data in, enforced Stokes right-hand side out.
///
/// The two older demos each cover one piece -- `plate_velocity_demo` samples velocities on the host,
/// `plate_velocity_kokkos` runs the plate-id lookup on the device -- and neither touches the boundary
/// condition. This one walks the whole chain and checks the pieces against each other:
///
///   1. build the Q1isoQ2/Q1 mesh pair the Stokes operator needs
///   2. load the reconstruction (the "oracle")
///   3. plate-id lookup **on the device**, from flattened polygon views
///   4. plate velocity extraction **on the host**, through the same `extract_plate_velocities()` the mantle
///      circulation app uses
///   5. cross-check 3 against 4 -- a surface node has a plate id exactly when it has a velocity
///   6. enforce the velocities as an inhomogeneous Dirichlet condition on a Stokes right-hand side, and verify
///      the enforced rows against an independent evaluation
///
/// Step 5 is the point of the exercise. The device lookup and the host oracle are separate implementations of
/// the same predicate -- winding number over the flattened polygons versus the polygon containers in
/// `src/terra/plates` -- and nothing in the tree currently compares them.
///
/// Step 6 builds only the two Stokes operators the enforcement needs, not the multigrid preconditioner: the
/// point is to exercise the new boundary condition, not to re-test the existing solver.

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <optional>
#include <vector>

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "grid/bit_masks.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "mpi/mpi.hpp"
#include "src/plates.hpp"
#include "terra/plates/plate_velocity_device.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/plates/plate_velocity_provider.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/timer.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using util::logroot;

using ScalarType = double;

namespace
{

struct Parameters
{
    int         level      = 5;
    double      r_min      = 0.55;
    double      r_max      = 1.0;
    double      age        = 0.0;
    bool        interpolate_in_time = false;
    double      velocity_scale      = 1.0;
    std::string topologies = "./topologies0-100Ma.geojson";
    std::string reconstructions = "./Global_EarthByte_230-0Ma_GK07_AREPS.rot";
    std::string outdir     = "./plate_bc_demo_out";
};

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        logroot << "  FAIL: " << what << std::endl;
    }
}

} // namespace

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    Parameters prm;

    CLI::App app{ "Plate boundary-condition demo" };
    util::add_option_with_default( app, "--level", prm.level );
    util::add_option_with_default( app, "--r-min", prm.r_min );
    util::add_option_with_default( app, "--r-max", prm.r_max );
    util::add_option_with_default( app, "--age", prm.age );
    util::add_option_with_default( app, "--interpolate-in-time", prm.interpolate_in_time );
    util::add_option_with_default( app, "--velocity-scale", prm.velocity_scale );
    util::add_option_with_default( app, "--topologies", prm.topologies );
    util::add_option_with_default( app, "--reconstructions", prm.reconstructions );
    util::add_option_with_default( app, "--outdir", prm.outdir );
    CLI11_PARSE( app, argc, argv );

    // ==========================================================================================================
    //  1. Mesh
    // ==========================================================================================================
    // The Stokes operator is Q1isoQ2/Q1: velocities live on `level`, pressures one level coarser.
    logroot << "\n=== 1. Mesh ===" << std::endl;
    std::optional< util::Timer > timer_total;
    timer_total.emplace( "total" );
    std::optional< util::Timer > timer_phase;
    timer_phase.emplace( "mesh_setup" );

    const auto radii_fine   = grid::shell::uniform_shell_radii< double >( prm.r_min, prm.r_max, ( 1 << prm.level ) + 1 );
    const auto radii_coarse = grid::shell::uniform_shell_radii< double >(
        prm.r_min, prm.r_max, ( 1 << ( prm.level - 1 ) ) + 1 );

    const auto domain_fine   = DistributedDomain::create_uniform( prm.level, radii_fine, 0, 0 );
    const auto domain_coarse = DistributedDomain::create_uniform( prm.level - 1, radii_coarse, 0, 0 );

    const auto coords_fine = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_fine );
    const auto radii_grid  = grid::shell::subdomain_shell_radii< ScalarType >( domain_fine );

    auto ownership_fine   = grid::setup_node_ownership_mask_data( domain_fine );
    auto ownership_coarse = grid::setup_node_ownership_mask_data( domain_coarse );
    auto boundary_fine    = grid::shell::setup_boundary_mask_data( domain_fine );

    const int num_sub  = static_cast< int >( domain_fine.subdomains().size() );
    const int n_lat    = domain_fine.domain_info().subdomain_num_nodes_per_side_laterally();
    const int n_rad    = domain_fine.domain_info().subdomain_num_nodes_radially();

    logroot << "  level " << prm.level << ", shell [" << prm.r_min << ", " << prm.r_max << "], "
            << num_sub << " subdomains of " << n_lat << "^2 x " << n_rad << std::endl;

    // ==========================================================================================================
    //  2. Oracle
    // ==========================================================================================================
    logroot << "\n=== 2. Reconstruction data ===" << std::endl;
    logroot << "  topologies      : " << prm.topologies << std::endl;
    logroot << "  reconstructions : " << prm.reconstructions << std::endl;

    timer_phase.reset();
    timer_phase.emplace( "oracle_load" );
    auto oracle = mantlecirculation::initialise_plates( prm.topologies, prm.reconstructions );
    timer_phase.reset();

    logroot << "  age range available: [" << oracle->getMinAge() << ", " << oracle->getMaxAge() << "] Ma"
            << std::endl;

    if ( prm.age < oracle->getMinAge() || prm.age > oracle->getMaxAge() )
    {
        logroot << "  requested age " << prm.age << " Ma is outside the available range." << std::endl;
        return EXIT_FAILURE;
    }

    // ==========================================================================================================
    //  3. Plate ids on the device
    // ==========================================================================================================
    logroot << "\n=== 3. Plate id lookup (device) ===" << std::endl;

    Grid4DDataScalar< ScalarType > plate_id( "plate_id", num_sub, n_lat, n_lat, n_rad );

    {
        util::Timer timer( "device_plate_id_lookup" );

        // The stage must be prepared before its packed views (and Euler vectors) exist.
        oracle->prepareEulerVectors( prm.age );

        const auto& stage = oracle->stageFor( prm.age );

        logroot << "  " << stage.device().nPlates << " plates at stage " << static_cast< int >( prm.age )
                << " Ma" << std::endl;

        plates::extract_plate_ids_device< ScalarType >(
            domain_fine, coords_fine, radii_grid, stage.device(), plate_id );
    }

    // ==========================================================================================================
    //  3b. The same plate ids on the host
    // ==========================================================================================================
    // One lookup per surface node, the same shape as the device pass above and as the Boost reference build,
    // so that the three plate-id timings compare like for like.
    logroot << "\n=== 3b. Plate id lookup (host) ===" << std::endl;

    Grid4DDataScalar< ScalarType > plate_id_host( "plate_id_host", num_sub, n_lat, n_lat, n_rad );

    {
        auto coords_h = Kokkos::create_mirror_view( coords_fine );
        Kokkos::deep_copy( coords_h, coords_fine );
        auto radii_h = Kokkos::create_mirror_view( radii_grid );
        Kokkos::deep_copy( radii_h, radii_grid );
        auto pid_h = Kokkos::create_mirror_view( plate_id_host );
        Kokkos::deep_copy( pid_h, ScalarType( 0 ) );

        {
            util::Timer timer( "host_plate_id_lookup" );

            for ( int sd = 0; sd < num_sub; ++sd )
                for ( int x = 0; x < n_lat; ++x )
                    for ( int y = 0; y < n_lat; ++y )
                    {
                        const auto c = grid::shell::coords( sd, x, y, n_rad - 1, coords_h, radii_h );
                        pid_h( sd, x, y, n_rad - 1 ) = static_cast< ScalarType >( oracle->findPlateID(
                            dense::Vec< double, 3 >{ c( 0 ), c( 1 ), c( 2 ) }, prm.age ) );
                    }
        }

        Kokkos::deep_copy( plate_id_host, pid_h );

        auto pid_dev_h = Kokkos::create_mirror_view( plate_id );
        Kokkos::deep_copy( pid_dev_h, plate_id );

        long long id_mismatches = 0;
        for ( int sd = 0; sd < num_sub; ++sd )
            for ( int x = 0; x < n_lat; ++x )
                for ( int y = 0; y < n_lat; ++y )
                    if ( pid_h( sd, x, y, n_rad - 1 ) != pid_dev_h( sd, x, y, n_rad - 1 ) )
                        ++id_mismatches;

        logroot << "  host vs device plate ids: " << id_mismatches << " of "
                << ( static_cast< long long >( num_sub ) * n_lat * n_lat ) << " surface nodes differ"
                << std::endl;
    }

    // ==========================================================================================================
    //  4. Plate velocities on the host, through the production path
    // ==========================================================================================================
    logroot << "\n=== 4. Plate velocity extraction (host) ===" << std::endl;

    linalg::VectorQ1IsoQ2Q1< ScalarType > plate_velocities(
        "plate_velocities", domain_fine, domain_coarse, ownership_fine, ownership_coarse );

    {
        util::Timer timer( "host_plate_velocity_extraction" );

        mantlecirculation::extract_plate_velocities(
            prm.age,
            plate_velocities.block_1().grid_data(),
            *oracle,
            coords_fine,
            radii_grid,
            prm.interpolate_in_time,
            prm.velocity_scale );
    }

    // ==========================================================================================================
    //  4b. The same extraction on the device
    // ==========================================================================================================
    // Same quantity, evaluated from the packed stage views inside a Kokkos kernel instead of on the host. The
    // point of the comparison below is that these are two independent evaluations of the same formula.
    logroot << "\n=== 4b. Plate velocity extraction (device) ===" << std::endl;

    linalg::VectorQ1IsoQ2Q1< ScalarType > plate_velocities_device(
        "plate_velocities_device", domain_fine, domain_coarse, ownership_fine, ownership_coarse );
    linalg::assign( plate_velocities_device, ScalarType( 0 ) );

    long long needing_averaging = 0;

    {
        timer_phase.reset();
        timer_phase.emplace( "device_plate_velocity_extraction" );

        const auto& stage = oracle->stageFor( prm.age );

        // Same stencil the host path uses, uploaded once.
        const plates::UniformCirclesPointWeightProvider weights( { { 1.0 / 100.0, 6 } }, 1e-1 );
        const auto stencil = plates::make_device_averaging_stencil( weights );

        plates::extract_plate_velocities_device< ScalarType >(
            domain_fine, coords_fine, radii_grid, stage.device(), stencil,
            plate_velocities_device.block_1().grid_data(),
            static_cast< ScalarType >( prm.velocity_scale ) );

        timer_phase.reset();

        needing_averaging = plates::surface_points_needing_averaging< ScalarType >(
            domain_fine, coords_fine, radii_grid, stage.device(), stencil.maxDistanceKm );
    }

    // Compare the two extractions node by node.
    {
        const auto h    = plate_velocities.block_1().grid_data();
        const auto dv   = plate_velocities_device.block_1().grid_data();
        const auto mask = boundary_fine;
        const auto own  = ownership_fine;

        // Split the comparison by regime. The device kernel deliberately omits the host's local averaging, so
        // a disagreement near a plate boundary is expected and says nothing about whether the port is right.
        // What must agree to round-off is everywhere else, where the host takes the same unaveraged shortcut.
        const auto   stage_views   = oracle->stageFor( prm.age ).device();
        const plates::UniformCirclesPointWeightProvider weights_cmp( { { 1.0 / 100.0, 6 } }, 1e-1 );
        const double stencil_reach = weights_cmp.maxDistance( dense::Vec< double, 3 >{ 0, 0, 1 } );

        ScalarType max_diff_plain = 0, max_diff_avg = 0, max_host = 0;

        Kokkos::parallel_reduce(
            "host_vs_device_velocity",
            grid::shell::local_domain_md_range_policy_nodes( domain_fine ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, ScalarType& plain,
                           ScalarType& avg, ScalarType& mag ) {
                if ( mask( sd, x, y, r ) != grid::shell::ShellBoundaryFlag::SURFACE )
                    return;
                if ( !util::has_flag( own( sd, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                    return;

                const auto c = grid::shell::coords( sd, x, y, r, coords_fine, radii_grid );
                const dense::Vec< double, 3 > lonLat =
                    plates::conversions::cart2sph( dense::Vec< double, 3 >{ c( 0 ), c( 1 ), c( 2 ) } );
                const auto hit = plates::findPlateInStage(
                    stage_views, plates::geometry::lonLatDegToUnit( lonLat( 0 ), lonLat( 1 ) ) );

                const bool averaged =
                    hit.found && stencil_reach >= hit.distanceRad * plates::constants::earthRadiusInKm;

                for ( int d = 0; d < 3; ++d )
                {
                    const ScalarType e = Kokkos::abs( h( sd, x, y, r, d ) - dv( sd, x, y, r, d ) );
                    if ( averaged )
                        avg = Kokkos::max( avg, e );
                    else
                        plain = Kokkos::max( plain, e );
                    mag = Kokkos::max( mag, Kokkos::abs( h( sd, x, y, r, d ) ) );
                }
            },
            Kokkos::Max< ScalarType >( max_diff_plain ),
            Kokkos::Max< ScalarType >( max_diff_avg ),
            Kokkos::Max< ScalarType >( max_host ) );
        Kokkos::fence();

        MPI_Allreduce( MPI_IN_PLACE, &max_diff_plain, 1, MPI_DOUBLE, MPI_MAX, domain_fine.comm() );
        MPI_Allreduce( MPI_IN_PLACE, &max_diff_avg, 1, MPI_DOUBLE, MPI_MAX, domain_fine.comm() );
        MPI_Allreduce( MPI_IN_PLACE, &max_host, 1, MPI_DOUBLE, MPI_MAX, domain_fine.comm() );
        MPI_Allreduce( MPI_IN_PLACE, &needing_averaging, 1, MPI_LONG_LONG, MPI_SUM, domain_fine.comm() );

        const ScalarType rel_plain = max_host > 0 ? max_diff_plain / max_host : 0;
        const ScalarType rel_avg   = max_host > 0 ? max_diff_avg / max_host : 0;

        logroot << "  points in the averaged regime       : " << needing_averaging << std::endl;
        logroot << "  max |host - device|, unaveraged     : " << std::scientific << std::setprecision( 4 )
                << max_diff_plain << "   (rel " << rel_plain << ")" << std::endl;
        logroot << "  max |host - device|, averaged       : " << max_diff_avg << "   (rel " << rel_avg << ")"
                << std::defaultfloat << std::endl;

        check( rel_plain < 1e-10, "device and host velocities disagree away from plate boundaries" );
        check( rel_avg < 1e-10, "device and host velocities disagree near plate boundaries, where both average "
                                "over the same stencil" );
    }

    // ==========================================================================================================
    //  5. Do the two paths agree?
    // ==========================================================================================================
    // A surface node should carry a plate id exactly when the oracle gave it a velocity. Disagreement means the
    // device winding-number test and the host polygon containers classify that point differently.
    logroot << "\n=== 5. Cross-check: device ids vs host velocities ===" << std::endl;
    timer_phase.emplace( "cross_check" );

    long long num_surface = 0, num_with_id = 0, num_with_velocity = 0, num_disagree = 0;
    ScalarType max_speed = 0;

    {
        const auto ids  = plate_id;
        const auto vel  = plate_velocities.block_1().grid_data();
        const auto mask = boundary_fine;
        const auto own  = ownership_fine;

        Kokkos::parallel_reduce(
            "plate_cross_check",
            grid::shell::local_domain_md_range_policy_nodes( domain_fine ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, long long& surface,
                           long long& with_id, long long& with_v, long long& disagree, ScalarType& speed ) {
                if ( mask( sd, x, y, r ) != grid::shell::ShellBoundaryFlag::SURFACE )
                    return;
                if ( !util::has_flag( own( sd, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                    return;

                surface += 1;

                const bool has_id = ids( sd, x, y, r ) != ScalarType( 0 );

                ScalarType v2 = 0;
                for ( int d = 0; d < 3; ++d )
                    v2 += vel( sd, x, y, r, d ) * vel( sd, x, y, r, d );
                const ScalarType v = Kokkos::sqrt( v2 );

                const bool has_v = v > ScalarType( 0 );

                if ( has_id )
                    with_id += 1;
                if ( has_v )
                    with_v += 1;
                if ( has_id != has_v )
                    disagree += 1;

                speed = Kokkos::max( speed, v );
            },
            num_surface,
            num_with_id,
            num_with_velocity,
            num_disagree,
            Kokkos::Max< ScalarType >( max_speed ) );
        Kokkos::fence();
    }

    MPI_Allreduce( MPI_IN_PLACE, &num_surface, 1, MPI_LONG_LONG, MPI_SUM, domain_fine.comm() );
    MPI_Allreduce( MPI_IN_PLACE, &num_with_id, 1, MPI_LONG_LONG, MPI_SUM, domain_fine.comm() );
    MPI_Allreduce( MPI_IN_PLACE, &num_with_velocity, 1, MPI_LONG_LONG, MPI_SUM, domain_fine.comm() );
    MPI_Allreduce( MPI_IN_PLACE, &num_disagree, 1, MPI_LONG_LONG, MPI_SUM, domain_fine.comm() );
    MPI_Allreduce( MPI_IN_PLACE, &max_speed, 1, MPI_DOUBLE, MPI_MAX, domain_fine.comm() );

    const double coverage_id = 100.0 * static_cast< double >( num_with_id ) / static_cast< double >( num_surface );
    const double coverage_v  = 100.0 * static_cast< double >( num_with_velocity ) / static_cast< double >( num_surface );

    logroot << "  owned surface nodes         : " << num_surface << std::endl;
    logroot << "  with a plate id (device)    : " << num_with_id << "  (" << std::fixed << std::setprecision( 2 )
            << coverage_id << "%)" << std::endl;
    logroot << "  with a velocity (host)      : " << num_with_velocity << "  (" << coverage_v << "%)" << std::endl;
    logroot << "  disagreeing nodes           : " << num_disagree << "  ("
            << ( 100.0 * static_cast< double >( num_disagree ) / static_cast< double >( num_surface ) ) << "%)"
            << std::endl;
    logroot << "  max |v| (nondim)            : " << std::scientific << std::setprecision( 4 ) << max_speed
            << std::defaultfloat << std::endl;

    check( num_surface > 0, "no owned surface nodes found" );
    check( max_speed > 0, "every plate velocity is zero -- extraction produced nothing" );

    // ==========================================================================================================
    //  6. Enforce the velocities as a Dirichlet condition on a Stokes right-hand side
    // ==========================================================================================================
    timer_phase.reset();
    logroot << "\n=== 6. Dirichlet enforcement on the Stokes rhs ===" << std::endl;
    timer_phase.emplace( "bc_enforcement" );

    using Stokes = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;

    // Unit viscosity: the enforcement is linear in the operator, so the material model is irrelevant here.
    linalg::VectorQ1Scalar< ScalarType > eta( "eta", domain_fine, ownership_fine );
    linalg::assign( eta, ScalarType( 1 ) );

    grid::shell::BoundaryConditions bcs_neumann = {
        { grid::shell::ShellBoundaryFlag::CMB, grid::shell::BoundaryConditionFlag::NEUMANN },
        { grid::shell::ShellBoundaryFlag::SURFACE, grid::shell::BoundaryConditionFlag::NEUMANN },
    };

    Stokes K_neumann(
        domain_fine, domain_coarse, coords_fine, radii_grid, boundary_fine, eta.grid_data(), bcs_neumann, false );
    Stokes K_neumann_diag(
        domain_fine, domain_coarse, coords_fine, radii_grid, boundary_fine, eta.grid_data(), bcs_neumann, true );

    linalg::VectorQ1IsoQ2Q1< ScalarType > f( "f", domain_fine, domain_coarse, ownership_fine, ownership_coarse );
    linalg::VectorQ1IsoQ2Q1< ScalarType > tmp( "tmp", domain_fine, domain_coarse, ownership_fine, ownership_coarse );
    linalg::VectorQ1IsoQ2Q1< ScalarType > expected(
        "expected", domain_fine, domain_coarse, ownership_fine, ownership_coarse );

    // A non-zero starting rhs, so that the enforcement has something to overwrite and we can tell that it did.
    linalg::assign( f, ScalarType( 1 ) );

    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann,
        K_neumann_diag,
        plate_velocities,
        tmp,
        f,
        boundary_fine,
        grid::shell::ShellBoundaryFlag::SURFACE );

    // The enforcement sets the Dirichlet rows of the rhs to diag(K) * g. Recompute that independently and
    // compare -- this is what makes the step a check rather than a smoke test.
    linalg::apply( K_neumann_diag, plate_velocities, expected );

    ScalarType max_row_error = 0;
    ScalarType max_row_value = 0;
    {
        const auto f_data = f.block_1().grid_data();
        const auto e_data = expected.block_1().grid_data();
        const auto mask   = boundary_fine;
        const auto own    = ownership_fine;

        Kokkos::parallel_reduce(
            "check_enforced_rows",
            grid::shell::local_domain_md_range_policy_nodes( domain_fine ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, ScalarType& err, ScalarType& val ) {
                if ( mask( sd, x, y, r ) != grid::shell::ShellBoundaryFlag::SURFACE )
                    return;
                if ( !util::has_flag( own( sd, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                    return;

                for ( int d = 0; d < 3; ++d )
                {
                    err = Kokkos::max( err, Kokkos::abs( f_data( sd, x, y, r, d ) - e_data( sd, x, y, r, d ) ) );
                    val = Kokkos::max( val, Kokkos::abs( f_data( sd, x, y, r, d ) ) );
                }
            },
            Kokkos::Max< ScalarType >( max_row_error ),
            Kokkos::Max< ScalarType >( max_row_value ) );
        Kokkos::fence();
    }

    MPI_Allreduce( MPI_IN_PLACE, &max_row_error, 1, MPI_DOUBLE, MPI_MAX, domain_fine.comm() );
    MPI_Allreduce( MPI_IN_PLACE, &max_row_value, 1, MPI_DOUBLE, MPI_MAX, domain_fine.comm() );

    logroot << "  max |rhs - diag(K)*g| on the surface : " << std::scientific << std::setprecision( 4 )
            << max_row_error << std::endl;
    logroot << "  max |rhs| on the surface             : " << max_row_value << std::defaultfloat << std::endl;

    check( max_row_error < 1e-12, "enforced Dirichlet rows do not match diag(K) * g" );
    check( max_row_value > 0, "enforced Dirichlet rows are all zero -- the plate velocities did not reach the rhs" );

    // ==========================================================================================================
    //  7. Output
    // ==========================================================================================================
    timer_phase.reset();
    logroot << "\n=== 7. Output ===" << std::endl;
    timer_phase.emplace( "xdmf_output" );

    io::XDMFOutput xdmf( prm.outdir, domain_fine, coords_fine, radii_grid );
    xdmf.add( plate_id );
    xdmf.add( plate_velocities.block_1().grid_data() );
    xdmf.add( f.block_1().grid_data() );
    xdmf.write( 0 );

    logroot << "  wrote " << prm.outdir << std::endl;

    // Surface velocities from the device path, on the mesh nodes, so the field can be differenced against
    // another implementation evaluated at exactly the same points.
    {
        auto v_h = create_mirror( Kokkos::HostSpace{}, plate_velocities_device.block_1().grid_data() );
        deep_copy( v_h, plate_velocities_device.block_1().grid_data() );

        const std::string csv = prm.outdir + "_surface.csv";
        std::FILE*        out = std::fopen( csv.c_str(), "w" );
        std::fprintf( out, "sd,x,y,vx,vy,vz\n" );
        for ( int sd = 0; sd < num_sub; ++sd )
            for ( int x = 0; x < n_lat; ++x )
                for ( int y = 0; y < n_lat; ++y )
                    std::fprintf( out, "%d,%d,%d,%.17e,%.17e,%.17e\n", sd, x, y,
                                  v_h( sd, x, y, n_rad - 1, 0 ), v_h( sd, x, y, n_rad - 1, 1 ),
                                  v_h( sd, x, y, n_rad - 1, 2 ) );
        std::fclose( out );
        logroot << "  wrote " << csv << std::endl;
    }

    timer_phase.reset();
    timer_total.reset();

    logroot << "\n=== Timings (s) ===\n" << util::TimerTree::instance().json() << std::endl;

    int failures = g_failures;
    MPI_Allreduce( MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    logroot << "\nplate_bc_demo: " << ( failures == 0 ? "PASSED" : "FAILED" ) << std::endl;
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
