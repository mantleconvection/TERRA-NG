// Test: closed-form wedge point location and Q1 evaluation used by the MMOC transport scheme.
//
// Checks, on a single rank over all 10 diamonds:
//   1. Round-trip: for reference coordinates inside every wedge cell, map forward to a physical point,
//      locate it starting from a deliberately distant seed cell, and verify that the located cell and the
//      recovered reference coordinates reproduce the original point.
//   2. Q1 exactness: the wedge map is isoparametric (sum_j N_j x_j = x), so the Q1 interpolant of a function
//      that is linear in the physical coordinates must be exact. Interpolating  g(x) = 1 + 2x + 3y - 0.5z
//      at arbitrary interior points must reproduce g to round-off.
//   3. Radial clamping: points below r_min / above r_max are pulled back onto the boundary shell.
//   4. Subdomain boundary: points lying exactly on a lateral diamond edge, on a diamond corner, or on the
//      innermost/outermost shell are still located (either adjacent wedge is an acceptable answer), while
//      points just outside the diamond are reported as escaped, since this field has no ghost layer.
//   5. O(1) location: locate_point_direct() (barycentric prediction over the two spherical triangles of the
//      diamond, plus affine refinement) reproduces the walking locate_point() exactly, using a walk budget
//      far too small to cross the subdomain.
//   6. The device (Kokkos) instantiation compiles and produces the same result as the host path.
//
// Random sampling, the subdomain search, and the host-versus-GPU comparison live in
// test_mmoc_point_location_random.

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include <mpi.h>

#include "fe/wedge/sl/point_location.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;
using ScalarType = double;

using fe::wedge::sl::IndexBounds;
using fe::wedge::sl::WedgeCell;

namespace
{

ScalarType linear_field( const dense::Vec< ScalarType, 3 >& x )
{
    return 1.0 + 2.0 * x( 0 ) + 3.0 * x( 1 ) - 0.5 * x( 2 );
}

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        if ( g_failures < 20 )
            std::cout << "  FAIL: " << what << std::endl;
    }
}

} // namespace

void test( const int level )
{
    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond(
        level, level, 0.5, 1.0 );

    const auto coords_shell_d = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii_d = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    auto coords_shell = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_shell_d );
    auto coords_radii = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_radii_d );

    const int num_subdomains = static_cast< int >( domain.subdomains().size() );
    const int num_nodes_lat  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int num_nodes_rad  = domain.domain_info().subdomain_num_nodes_radially();

    const IndexBounds bounds{ num_nodes_lat, num_nodes_lat, num_nodes_rad };

    // No ghost layer here, so every node is usable.
    Kokkos::View< uint8_t*** > all_valid( "all_valid", num_subdomains, num_nodes_lat, num_nodes_lat );
    Kokkos::deep_copy( all_valid, static_cast< uint8_t >( 1 ) );
    auto all_valid_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, all_valid );

    std::cout << "level=" << level << "  subdomains=" << num_subdomains << "  nodes_lat=" << num_nodes_lat
              << "  nodes_rad=" << num_nodes_rad << std::endl;

    // A generous walk budget: from the corner seed we may have to cross the whole diamond.
    const int max_steps = 8 * num_nodes_lat;
    const ScalarType eps = 1e-12;

    // Reference points strictly inside the reference wedge (xi, eta > 0, xi + eta < 1, |zeta| < 1).
    const ScalarType ref_points[][3] = {
        { 1.0 / 3.0, 1.0 / 3.0, 0.0 },
        { 0.1, 0.1, -0.8 },
        { 0.7, 0.2, 0.6 },
        { 0.15, 0.7, 0.9 },
        { 0.45, 0.45, -0.95 },
    };
    const int num_ref_points = sizeof( ref_points ) / sizeof( ref_points[0] );

    ScalarType max_position_error = 0.0;
    ScalarType max_linear_error   = 0.0;
    long long  num_located        = 0;
    long long  num_escaped        = 0;

    for ( int sd = 0; sd < num_subdomains; ++sd )
    {
        for ( int cx = 0; cx < num_nodes_lat - 1; ++cx )
        {
            for ( int cy = 0; cy < num_nodes_lat - 1; ++cy )
            {
                for ( int cr = 0; cr < num_nodes_rad - 1; ++cr )
                {
                    for ( int w = 0; w < 2; ++w )
                    {
                        const WedgeCell cell{ cx, cy, cr, w };

                        for ( int p = 0; p < num_ref_points; ++p )
                        {
                            const ScalarType xi   = ref_points[p][0];
                            const ScalarType eta  = ref_points[p][1];
                            const ScalarType zeta = ref_points[p][2];

                            const auto X = fe::wedge::sl::wedge_forward_map(
                                sd, cell, coords_shell, coords_radii, xi, eta, zeta );

                            // Seed the search far away from the true cell (opposite corner of the diamond).
                            const WedgeCell seed{ 0, 0, 0, 0 };

                            const auto res = fe::wedge::sl::locate_point(
                                X, sd, seed, coords_shell, coords_radii, bounds, max_steps, eps,
                                /*clamp_radially=*/false, ScalarType( 0 ), ScalarType( 1e30 ), all_valid_h );

                            if ( !res.found )
                            {
                                ++num_escaped;
                                check( false,
                                       "not located: sd=" + std::to_string( sd ) + " cell=(" +
                                           std::to_string( cx ) + "," + std::to_string( cy ) + "," +
                                           std::to_string( cr ) + "," + std::to_string( w ) + ")" );
                                continue;
                            }
                            ++num_located;

                            check( res.cell.x == cell.x && res.cell.y == cell.y && res.cell.r == cell.r &&
                                       res.cell.w == cell.w,
                                   "wrong cell: sd=" + std::to_string( sd ) + " expected (" +
                                       std::to_string( cx ) + "," + std::to_string( cy ) + "," +
                                       std::to_string( cr ) + "," + std::to_string( w ) + ") got (" +
                                       std::to_string( res.cell.x ) + "," + std::to_string( res.cell.y ) + "," +
                                       std::to_string( res.cell.r ) + "," + std::to_string( res.cell.w ) + ")" );

                            // Round-trip the recovered reference coordinates back to a physical point.
                            const auto X_back = fe::wedge::sl::wedge_forward_map(
                                sd, res.cell, coords_shell, coords_radii, res.xi, res.eta, res.zeta );
                            max_position_error = std::max( max_position_error, ( X_back - X ).norm() );

                            // Q1 exactness for a linear field: build the six nodal values on the fly.
                            int nx[3], ny[3];
                            fe::wedge::sl::wedge_lateral_node_indices( res.cell, nx, ny );
                            ScalarType interpolated = 0.0;
                            for ( int j = 0; j < 6; ++j )
                            {
                                const int lateral = j % 3;
                                const int radial  = j / 3;
                                dense::Vec< ScalarType, 3 > node;
                                for ( int d = 0; d < 3; ++d )
                                    node( d ) = coords_shell( sd, nx[lateral], ny[lateral], d );
                                node = node * coords_radii( sd, res.cell.r + radial );

                                interpolated += linear_field( node ) *
                                                fe::wedge::shape_lat( j, res.xi, res.eta ) *
                                                fe::wedge::shape_rad( j, res.zeta );
                            }
                            max_linear_error =
                                std::max( max_linear_error, std::abs( interpolated - linear_field( X ) ) );
                        }
                    }
                }
            }
        }
    }

    std::cout << std::scientific << std::setprecision( 4 );
    std::cout << "  located          : " << num_located << "  (escaped: " << num_escaped << ")" << std::endl;
    std::cout << "  max |X_back - X| : " << max_position_error << std::endl;
    std::cout << "  max linear err   : " << max_linear_error << std::endl;

    check( max_position_error < 1e-12, "round-trip position error too large" );
    check( max_linear_error < 1e-12, "Q1 linear reproduction error too large" );
    check( num_escaped == 0, "some points escaped the index space" );

    // ---- radial clamping ---------------------------------------------------------------------------------
    {
        const WedgeCell cell{ 3, 3, 0, 0 };
        const auto      X_inner =
            fe::wedge::sl::wedge_forward_map( 0, cell, coords_shell, coords_radii, 0.3, 0.3, -1.0 );

        const auto X_below = X_inner * ScalarType( 0.5 ); // well below r_min

        const auto res_no_clamp = fe::wedge::sl::locate_point(
            X_below, 0, WedgeCell{ 0, 0, 0, 0 }, coords_shell, coords_radii, bounds, max_steps, eps, false,
            coords_radii( 0, 0 ), coords_radii( 0, num_nodes_rad - 1 ), all_valid_h );
        check( !res_no_clamp.found, "point below r_min should not be found without clamping" );

        const auto res_clamp = fe::wedge::sl::locate_point(
            X_below, 0, WedgeCell{ 0, 0, 0, 0 }, coords_shell, coords_radii, bounds, max_steps, eps, true,
            coords_radii( 0, 0 ), coords_radii( 0, num_nodes_rad - 1 ), all_valid_h );
        check( res_clamp.found && res_clamp.clamped_radially, "point below r_min should clamp onto r_min" );
        check( res_clamp.cell.r == 0 && std::abs( res_clamp.zeta + 1.0 ) < 1e-12,
               "clamped point should sit on the innermost shell" );
    }

    // ---- subdomain boundary --------------------------------------------------------------------------------
    // MMOC departure points routinely land on the boundary of a subdomain, so two things must hold there.
    //
    // First, a point sitting exactly *on* the boundary must still be located. It lies on the shared face of two
    // wedge cells, so one of its cone coordinates vanishes and the containment test passes for either of them.
    // Which cell comes back is therefore arbitrary and is deliberately not asserted; what is asserted is that
    // the walk does not report an escape and that the recovered reference coordinates reproduce the point.
    //
    // Second, a point just *outside* the boundary must be reported as escaped. This field carries no ghost
    // layer, so there is no geometry beyond the diamond edge, and quietly returning the nearest interior wedge
    // would be an unbounded extrapolation. (The ghosted case is covered by test_mmoc_ghost_exchange.)
    {
        const int last_cell = num_nodes_lat - 2; // last hex cell index; it spans nodes N-2 and N-1
        const int mid_cell  = num_nodes_lat / 2; // an index safely away from the diamond corners
        const int last_rad  = num_nodes_rad - 2;

        const ScalarType tol = 1e-11;

        struct Probe
        {
            const char* name;
            WedgeCell   cell;
            ScalarType  xi, eta, zeta;
        };

        // Reference coordinates chosen so that the forward map lands exactly on the named boundary. For w = 0
        // the triangle is ( (x,y), (x+1,y), (x,y+1) ), so eta = 0 traces the low-y edge and xi = 0 the low-x
        // one; for w = 1 it is ( (x+1,y+1), (x,y+1), (x+1,y) ), where eta = 0 traces the high-y edge and
        // xi = 0 the high-x one. zeta = -1 / +1 sit on the CMB / surface shell.
        const Probe probes[] = {
            { "lateral x = 0",       WedgeCell{ 0, mid_cell, 0, 0 }, 0.0, 0.4, 0.0 },
            { "lateral y = 0",       WedgeCell{ mid_cell, 0, 0, 0 }, 0.4, 0.0, 0.0 },
            { "lateral x = N-1",     WedgeCell{ last_cell, mid_cell, 0, 1 }, 0.0, 0.4, 0.0 },
            { "lateral y = N-1",     WedgeCell{ mid_cell, last_cell, 0, 1 }, 0.4, 0.0, 0.0 },
            { "corner (0, 0)",       WedgeCell{ 0, 0, 0, 0 }, 0.0, 0.0, 0.0 },
            { "corner (N-1, 0)",     WedgeCell{ last_cell, 0, 0, 1 }, 0.0, 1.0, 0.0 },
            { "corner (0, N-1)",     WedgeCell{ 0, last_cell, 0, 0 }, 0.0, 1.0, 0.0 },
            { "corner (N-1, N-1)",   WedgeCell{ last_cell, last_cell, 0, 1 }, 0.0, 0.0, 0.0 },
            { "radial r_min",        WedgeCell{ mid_cell, mid_cell, 0, 0 }, 0.3, 0.3, -1.0 },
            { "radial r_max",        WedgeCell{ mid_cell, mid_cell, last_rad, 0 }, 0.3, 0.3, 1.0 },
            { "edge x = 0 at r_min", WedgeCell{ 0, mid_cell, 0, 0 }, 0.0, 0.4, -1.0 },
        };

        ScalarType max_boundary_error = 0.0;

        for ( const auto& probe : probes )
        {
            // A point on the innermost / outermost shell is a knife-edge for the radial containment test: rho is
            // reconstructed from the cone coordinates, so it may land an ulp outside the shell. Clamping is what
            // the transport scheme uses there, and it is exact to round-off for a point already on the shell.
            const bool on_radial_boundary = std::abs( probe.zeta ) == 1.0;

            for ( int sd = 0; sd < num_subdomains; ++sd )
            {
                const auto X = fe::wedge::sl::wedge_forward_map(
                    sd, probe.cell, coords_shell, coords_radii, probe.xi, probe.eta, probe.zeta );

                const auto res = fe::wedge::sl::locate_point(
                    X, sd, WedgeCell{ 0, 0, 0, 0 }, coords_shell, coords_radii, bounds, max_steps, eps,
                    /*clamp_radially=*/on_radial_boundary, coords_radii( sd, 0 ),
                    coords_radii( sd, num_nodes_rad - 1 ), all_valid_h );

                const std::string what = std::string( probe.name ) + " (sd=" + std::to_string( sd ) + ")";

                check( res.found, "boundary point not located: " + what );
                check( !res.escaped_laterally, "boundary point escaped laterally: " + what );
                if ( !on_radial_boundary )
                    check( !res.clamped_radially, "lateral boundary point should not clamp radially: " + what );
                if ( !res.found )
                    continue;

                const auto X_back = fe::wedge::sl::wedge_forward_map(
                    sd, res.cell, coords_shell, coords_radii, res.xi, res.eta, res.zeta );
                const ScalarType err = ( X_back - X ).norm();
                max_boundary_error   = std::max( max_boundary_error, err );
                check( err < tol, "boundary round-trip error: " + what );

                // Whichever of the two adjacent cells came back, the reference coordinates must be a valid
                // point of it.
                check( res.xi >= -tol && res.eta >= -tol && res.xi + res.eta <= 1.0 + tol &&
                           std::abs( res.zeta ) <= 1.0 + tol,
                       "boundary reference coordinates outside the reference wedge: " + what );

                // The interpolation the transport scheme would perform there has to stay exact for a linear
                // field, which is the property a wrongly located boundary cell would silently break.
                int nx[3], ny[3];
                fe::wedge::sl::wedge_lateral_node_indices( res.cell, nx, ny );
                ScalarType interpolated = 0.0;
                for ( int j = 0; j < 6; ++j )
                {
                    const int lateral = j % 3;
                    const int radial  = j / 3;
                    dense::Vec< ScalarType, 3 > node;
                    for ( int d = 0; d < 3; ++d )
                        node( d ) = coords_shell( sd, nx[lateral], ny[lateral], d );
                    node = node * coords_radii( sd, res.cell.r + radial );

                    interpolated += linear_field( node ) * fe::wedge::shape_lat( j, res.xi, res.eta ) *
                                    fe::wedge::shape_rad( j, res.zeta );
                }
                check( std::abs( interpolated - linear_field( X ) ) < tol,
                       "Q1 linear reproduction on the boundary: " + what );
            }
        }

        std::cout << "  max boundary err : " << max_boundary_error << "  ("
                  << sizeof( probes ) / sizeof( probes[0] ) << " probes x " << num_subdomains
                  << " subdomains)" << std::endl;

        // Just outside the diamond. The walk has to run into the index-space bound and report an escape.
        const Probe outside[] = {
            { "outside x < 0",   WedgeCell{ 0, mid_cell, 0, 0 }, -0.25, 0.4, 0.0 },
            { "outside y < 0",   WedgeCell{ mid_cell, 0, 0, 0 }, 0.4, -0.25, 0.0 },
            { "outside x > N-1", WedgeCell{ last_cell, mid_cell, 0, 1 }, -0.25, 0.4, 0.0 },
            { "outside y > N-1", WedgeCell{ mid_cell, last_cell, 0, 1 }, 0.4, -0.25, 0.0 },
        };

        for ( const auto& probe : outside )
        {
            for ( int sd = 0; sd < num_subdomains; ++sd )
            {
                const auto X = fe::wedge::sl::wedge_forward_map(
                    sd, probe.cell, coords_shell, coords_radii, probe.xi, probe.eta, probe.zeta );

                const auto res = fe::wedge::sl::locate_point(
                    X, sd, WedgeCell{ 0, 0, 0, 0 }, coords_shell, coords_radii, bounds, max_steps, eps, false,
                    coords_radii( sd, 0 ), coords_radii( sd, num_nodes_rad - 1 ), all_valid_h );

                const std::string what = std::string( probe.name ) + " (sd=" + std::to_string( sd ) + ")";
                check( !res.found, "point outside the subdomain was located: " + what );
                check( res.escaped_laterally, "point outside the subdomain did not report an escape: " + what );
            }
        }
    }

    // ---- O(1) point location -------------------------------------------------------------------------------
    // locate_point_direct() must agree with the walking locate_point() everywhere, while only ever taking a
    // handful of steps. We also record how far the pure barycentric prediction alone is off, in cells, since
    // that number is what decides whether the walk budget can stay constant as the refinement level grows.
    {
        const auto box = fe::wedge::sl::corner_box_from_bounds( bounds );

        // Enough to finish from a prediction that is a cell or two off, and far too few to cross the diamond.
        const int direct_max_refinements = 2;
        const int direct_max_walk_steps  = 4;

        ScalarType max_prediction_error = 0.0; // |predicted - true| in cells, barycentric stage only
        ScalarType max_direct_error     = 0.0; // |X_back - X| after the full direct location
        long long  num_disagreements    = 0;

        for ( int sd = 0; sd < num_subdomains; ++sd )
        {
            for ( int cx = 0; cx < num_nodes_lat - 1; ++cx )
            {
                for ( int cy = 0; cy < num_nodes_lat - 1; ++cy )
                {
                    for ( int w = 0; w < 2; ++w )
                    {
                        const WedgeCell cell{ cx, cy, num_nodes_rad / 2, w };

                        for ( int p = 0; p < num_ref_points; ++p )
                        {
                            const ScalarType xi   = ref_points[p][0];
                            const ScalarType eta  = ref_points[p][1];
                            const ScalarType zeta = ref_points[p][2];

                            const auto X = fe::wedge::sl::wedge_forward_map(
                                sd, cell, coords_shell, coords_radii, xi, eta, zeta );

                            // Continuous index coordinates the prediction should reproduce. The wedge map is
                            // affine in index space, so these follow from the reference coordinates directly.
                            const ScalarType u_true = ( w == 0 ) ? cx + xi : cx + 1.0 - xi;
                            const ScalarType v_true = ( w == 0 ) ? cy + eta : cy + 1.0 - eta;

                            const auto pred = fe::wedge::sl::predict_lateral_cell(
                                X, sd, coords_shell, box, bounds, eps );
                            check( !pred.outside_quad, "interior point predicted outside the corner quad" );
                            max_prediction_error = std::max(
                                max_prediction_error,
                                std::max( std::abs( pred.u - u_true ), std::abs( pred.v - v_true ) ) );

                            const auto direct = fe::wedge::sl::locate_point_direct(
                                X, sd, coords_shell, coords_radii, box, bounds, direct_max_refinements,
                                direct_max_walk_steps, eps, /*clamp_radially=*/false, ScalarType( 0 ),
                                ScalarType( 1e30 ), all_valid_h );

                            const auto walked = fe::wedge::sl::locate_point(
                                X, sd, WedgeCell{ 0, 0, 0, 0 }, coords_shell, coords_radii, bounds, max_steps,
                                eps, false, ScalarType( 0 ), ScalarType( 1e30 ), all_valid_h );

                            if ( direct.found != walked.found || direct.cell.x != walked.cell.x ||
                                 direct.cell.y != walked.cell.y || direct.cell.r != walked.cell.r ||
                                 direct.cell.w != walked.cell.w )
                            {
                                ++num_disagreements;
                                check( false,
                                       "direct vs walked mismatch: sd=" + std::to_string( sd ) + " cell=(" +
                                           std::to_string( cx ) + "," + std::to_string( cy ) + "," +
                                           std::to_string( w ) + ")" );
                                continue;
                            }

                            check( direct.found, "direct location failed on an interior point" );
                            if ( !direct.found )
                                continue;

                            const auto X_back = fe::wedge::sl::wedge_forward_map(
                                sd, direct.cell, coords_shell, coords_radii, direct.xi, direct.eta,
                                direct.zeta );
                            max_direct_error = std::max( max_direct_error, ( X_back - X ).norm() );
                        }
                    }
                }
            }
        }

        std::cout << "  direct: max prediction err = " << max_prediction_error << " cells"
                  << "   max |X_back - X| = " << max_direct_error
                  << "   disagreements = " << num_disagreements << std::endl;

        check( max_direct_error < 1e-12, "direct location round-trip error too large" );
        check( num_disagreements == 0, "direct location disagrees with the walking location" );

        // The same on the subdomain boundary, where the prediction sits exactly on the quad's edge or corner.
        {
            const int last_cell = num_nodes_lat - 2;
            const int mid_cell  = num_nodes_lat / 2;

            const WedgeCell edge_cells[] = { WedgeCell{ 0, mid_cell, 0, 0 },
                                             WedgeCell{ mid_cell, 0, 0, 0 },
                                             WedgeCell{ last_cell, mid_cell, 0, 1 },
                                             WedgeCell{ mid_cell, last_cell, 0, 1 },
                                             WedgeCell{ 0, 0, 0, 0 },
                                             WedgeCell{ last_cell, last_cell, 0, 1 } };
            const ScalarType edge_xi[]  = { 0.0, 0.4, 0.0, 0.4, 0.0, 0.0 };
            const ScalarType edge_eta[] = { 0.4, 0.0, 0.4, 0.0, 0.0, 0.0 };

            for ( int p = 0; p < 6; ++p )
            {
                for ( int sd = 0; sd < num_subdomains; ++sd )
                {
                    const auto X = fe::wedge::sl::wedge_forward_map(
                        sd, edge_cells[p], coords_shell, coords_radii, edge_xi[p], edge_eta[p],
                        ScalarType( 0 ) );

                    const auto direct = fe::wedge::sl::locate_point_direct(
                        X, sd, coords_shell, coords_radii, box, bounds, direct_max_refinements,
                        direct_max_walk_steps, eps, false, coords_radii( sd, 0 ),
                        coords_radii( sd, num_nodes_rad - 1 ), all_valid_h );

                    check( direct.found, "direct location failed on a subdomain boundary point" );
                    if ( !direct.found )
                        continue;

                    const auto X_back = fe::wedge::sl::wedge_forward_map(
                        sd, direct.cell, coords_shell, coords_radii, direct.xi, direct.eta, direct.zeta );
                    check( ( X_back - X ).norm() < 1e-11, "direct boundary round-trip error" );
                }
            }
        }

        // And outside the diamond: the prediction is clamped to a boundary cell, from which the short walk must
        // still reach the index-space bound and report the escape rather than silently returning that cell.
        {
            const int  mid_cell = num_nodes_lat / 2;
            const auto X_out    = fe::wedge::sl::wedge_forward_map(
                0, WedgeCell{ 0, mid_cell, 0, 0 }, coords_shell, coords_radii, ScalarType( -0.25 ),
                ScalarType( 0.4 ), ScalarType( 0 ) );

            const auto pred = fe::wedge::sl::predict_lateral_cell( X_out, 0, coords_shell, box, bounds, eps );
            check( pred.outside_quad, "point outside the diamond not flagged by the predictor" );

            const auto direct = fe::wedge::sl::locate_point_direct(
                X_out, 0, coords_shell, coords_radii, box, bounds, direct_max_refinements,
                direct_max_walk_steps, eps, false, coords_radii( 0, 0 ),
                coords_radii( 0, num_nodes_rad - 1 ), all_valid_h );
            check( !direct.found && direct.escaped_laterally,
                   "direct location should report an escape outside the diamond" );
        }

        // Device instantiation of the O(1) path.
        {
            int  device_failures = 0;
            auto cs              = coords_shell_d;
            auto cr_             = coords_radii_d;

            Kokkos::parallel_reduce(
                "mmoc_locate_direct_device_check",
                Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
                    { 0, 0, 0 }, { num_subdomains, num_nodes_lat - 1, num_nodes_lat - 1 } ),
                KOKKOS_LAMBDA( const int sd, const int cx, const int cy, int& acc ) {
                    const WedgeCell cell{ cx, cy, 0, 0 };
                    const auto      X = fe::wedge::sl::wedge_forward_map(
                        sd, cell, cs, cr_, ScalarType( 0.25 ), ScalarType( 0.25 ), ScalarType( 0.0 ) );

                    const auto res = fe::wedge::sl::locate_point_direct(
                        X, sd, cs, cr_, box, bounds, direct_max_refinements, direct_max_walk_steps,
                        ScalarType( 1e-12 ), false, ScalarType( 0 ), ScalarType( 1e30 ), all_valid );

                    if ( !res.found || res.cell.x != cx || res.cell.y != cy || res.cell.w != 0 )
                        acc += 1;
                },
                device_failures );
            Kokkos::fence();

            std::cout << "  direct: device mismatches = " << device_failures << std::endl;
            check( device_failures == 0, "device direct location disagrees with the host" );
        }
    }

    // ---- device instantiation ----------------------------------------------------------------------------
    // Runs the same location on the default execution space and counts mismatches, proving the kernel is
    // device-compilable and branch-free enough to run inside a parallel_for.
    {
        int  device_failures = 0;
        auto cs              = coords_shell_d;
        auto cr_             = coords_radii_d;

        Kokkos::parallel_reduce(
            "mmoc_locate_device_check",
            Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 },
                                                        { num_subdomains, num_nodes_lat - 1, num_nodes_lat - 1 } ),
            KOKKOS_LAMBDA( const int sd, const int cx, const int cy, int& acc ) {
                const WedgeCell cell{ cx, cy, 0, 0 };
                const auto      X = fe::wedge::sl::wedge_forward_map(
                    sd, cell, cs, cr_, ScalarType( 0.25 ), ScalarType( 0.25 ), ScalarType( 0.0 ) );

                const auto res = fe::wedge::sl::locate_point(
                    X, sd, WedgeCell{ 0, 0, 0, 0 }, cs, cr_, bounds, max_steps, ScalarType( 1e-12 ), false,
                    ScalarType( 0 ), ScalarType( 1e30 ), all_valid );

                if ( !res.found || res.cell.x != cx || res.cell.y != cy || res.cell.w != 0 )
                    acc += 1;
            },
            device_failures );
        Kokkos::fence();

        std::cout << "  device mismatches: " << device_failures << std::endl;
        check( device_failures == 0, "device location disagrees with host location" );
    }
}

/// Checks that the cost of locate_point_direct() really is independent of the refinement level.
///
/// The barycentric prediction of stage 1 is off by a fixed *fraction* of the diamond, so its error in cells
/// grows linearly with the level and a scheme that only predicted and then walked would be O(N). What has to
/// stay bounded is the work after the prediction, so this runs with the same small refinement and walk budgets
/// at every level and requires exact agreement with a full walk. It samples cells sparsely, since the reference
/// walk it compares against is the expensive O(N) one.
void test_direct_scaling( const int level )
{
    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond(
        level, level, 0.5, 1.0 );

    const auto coords_shell_d = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii_d = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    auto coords_shell = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_shell_d );
    auto coords_radii = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_radii_d );

    const int num_subdomains = static_cast< int >( domain.subdomains().size() );
    const int num_nodes_lat  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int num_nodes_rad  = domain.domain_info().subdomain_num_nodes_radially();

    const IndexBounds bounds{ num_nodes_lat, num_nodes_lat, num_nodes_rad };

    Kokkos::View< uint8_t***, Kokkos::HostSpace > all_valid_h(
        "all_valid_h", num_subdomains, num_nodes_lat, num_nodes_lat );
    Kokkos::deep_copy( all_valid_h, static_cast< uint8_t >( 1 ) );

    const auto box = fe::wedge::sl::corner_box_from_bounds( bounds );

    // Budgets held fixed across levels -- this is the property under test.
    const int direct_max_refinements = 2;
    const int direct_max_walk_steps  = 4;

    const ScalarType eps       = 1e-12;
    const int        max_steps = 8 * num_nodes_lat; // reference walk, allowed to cross the whole diamond

    const int num_cells = num_nodes_lat - 1;
    const int stride    = std::max( 1, num_cells / 12 );

    ScalarType max_prediction_error = 0.0;
    ScalarType max_direct_error     = 0.0;
    long long  num_sampled          = 0;
    long long  num_disagreements    = 0;

    for ( int sd = 0; sd < num_subdomains; ++sd )
    {
        for ( int cx = 0; cx < num_cells; cx += stride )
        {
            for ( int cy = 0; cy < num_cells; cy += stride )
            {
                for ( int w = 0; w < 2; ++w )
                {
                    const WedgeCell  cell{ cx, cy, num_nodes_rad / 2, w };
                    const ScalarType xi   = 0.3;
                    const ScalarType eta  = 0.25;
                    const ScalarType zeta = 0.1;

                    const auto X =
                        fe::wedge::sl::wedge_forward_map( sd, cell, coords_shell, coords_radii, xi, eta, zeta );

                    const ScalarType u_true = ( w == 0 ) ? cx + xi : cx + 1.0 - xi;
                    const ScalarType v_true = ( w == 0 ) ? cy + eta : cy + 1.0 - eta;

                    const auto pred =
                        fe::wedge::sl::predict_lateral_cell( X, sd, coords_shell, box, bounds, eps );
                    max_prediction_error =
                        std::max( max_prediction_error,
                                  std::max( std::abs( pred.u - u_true ), std::abs( pred.v - v_true ) ) );

                    const auto direct = fe::wedge::sl::locate_point_direct(
                        X, sd, coords_shell, coords_radii, box, bounds, direct_max_refinements,
                        direct_max_walk_steps, eps, false, ScalarType( 0 ), ScalarType( 1e30 ), all_valid_h );

                    const auto walked = fe::wedge::sl::locate_point(
                        X, sd, WedgeCell{ 0, 0, 0, 0 }, coords_shell, coords_radii, bounds, max_steps, eps,
                        false, ScalarType( 0 ), ScalarType( 1e30 ), all_valid_h );

                    ++num_sampled;

                    if ( !direct.found || direct.found != walked.found || direct.cell.x != walked.cell.x ||
                         direct.cell.y != walked.cell.y || direct.cell.r != walked.cell.r ||
                         direct.cell.w != walked.cell.w )
                    {
                        ++num_disagreements;
                        continue;
                    }

                    const auto X_back = fe::wedge::sl::wedge_forward_map(
                        sd, direct.cell, coords_shell, coords_radii, direct.xi, direct.eta, direct.zeta );
                    max_direct_error = std::max( max_direct_error, ( X_back - X ).norm() );
                }
            }
        }
    }

    std::cout << std::scientific << std::setprecision( 4 );
    std::cout << "level=" << level << "  cells/side=" << num_cells << "  sampled=" << num_sampled
              << "  prediction err=" << max_prediction_error << " cells"
              << "  |X_back - X|=" << max_direct_error << "  disagreements=" << num_disagreements << std::endl;

    check( num_disagreements == 0,
           "direct location needed more than the fixed budget at level " + std::to_string( level ) );
    check( max_direct_error < 1e-12, "direct location round-trip error at level " + std::to_string( level ) );
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    test( 3 );
    test( 4 );

    // The budgets inside locate_point_direct() are held fixed here while the level grows.
    for ( int level = 3; level <= 7; ++level )
        test_direct_scaling( level );

    if ( g_failures == 0 )
    {
        std::cout << "\ntest_mmoc_point_location: PASSED" << std::endl;
    }
    else
    {
        std::cout << "\ntest_mmoc_point_location: FAILED (" << g_failures << " checks)" << std::endl;
    }
    return g_failures == 0 ? 0 : 1;
}
