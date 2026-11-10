// Test: width-1 ghost layers for Q1 nodal fields, and point location across subdomain / diamond seams.
//
// The ghost exchange is what lets a semi-Lagrangian departure point that leaves the local subdomain still be
// interpolated. It is validated in three steps:
//
//   1. Consistency. Coordinates and a scalar field are filled by the *same* exchange, so every ghost node's
//      coordinate and value must belong to the same physical node: with f a function of position, the ghosted
//      f must equal f(ghosted coordinates) everywhere the ghost was written.
//
//   2. Geometry. Ghost node directions must be unit vectors, and must sit roughly one mesh width from the
//      owned boundary node they extend -- this catches a sweep that deposited a plausible-looking but
//      geometrically wrong value.
//
//   3. End to end. Nodes near the subdomain boundaries are displaced by a fraction of the mesh width in
//      several directions (which is exactly what a departure point does), located in the ghosted index space,
//      and the Q1-interpolated linear field is compared against its exact value. This is the property the
//      transport actually relies on.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <mpi.h>

#include "fe/wedge/sl/ghost_exchange.hpp"
#include "fe/wedge/sl/ghosted_geometry.hpp"
#include "fe/wedge/sl/point_location.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;
using ScalarType = double;

using fe::wedge::sl::GhostExchange;
using fe::wedge::sl::IndexBounds;
using fe::wedge::sl::WedgeCell;
using fe::wedge::sl::ghost_width;
using fe::wedge::sl::to_ghosted_index;

namespace
{

constexpr ScalarType kUnwritten = 1e30;

KOKKOS_INLINE_FUNCTION ScalarType linear_field( const dense::Vec< ScalarType, 3 >& x )
{
    return 1.0 + 2.0 * x( 0 ) + 3.0 * x( 1 ) - 0.5 * x( 2 );
}

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        std::cout << "  FAIL: " << what << std::endl;
    }
}

} // namespace

void test( const int level, const int subdomain_level = 0, const ScalarType r_lo = 0.5,
           const ScalarType r_hi = 1.0 )
{
    // subdomain_level > 0 splits every diamond into (2^k)^2 x 2^k subdomains, which is what the mantle
    // circulation app runs with -- and which creates many more interface corners than one subdomain per
    // diamond does.
    const auto domain = grid::shell::DistributedDomain::create_uniform(
        level, level, r_lo, r_hi, subdomain_level, subdomain_level );

    const GhostExchange exchange( domain );

    const int num_sub  = exchange.num_subdomains();
    const int n_lat    = exchange.num_nodes_lateral();
    const int n_rad    = exchange.num_nodes_radial();
    const int n_lat_g  = exchange.num_nodes_lateral_ghosted();
    const int n_rad_g  = exchange.num_nodes_radial_ghosted();

    const auto coords_owned = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto radii_owned  = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    const auto coords_g = fe::wedge::sl::ghosted_unit_sphere_coords< ScalarType >( domain, exchange );

    // The unit-sphere directions do not vary radially; pin an owned radial layer for point location.
    const fe::wedge::sl::RadialSliceCoords< decltype( coords_g ) > lateral_g{ coords_g, ghost_width };

    const auto lateral_valid = fe::wedge::sl::ghosted_lateral_validity< ScalarType >( exchange, coords_g );
    const auto radii_g  = fe::wedge::sl::ghosted_shell_radii< ScalarType >( domain, exchange );
    const auto [r_min, r_max] = fe::wedge::sl::shell_radius_bounds< ScalarType >( domain );

    std::cout << "level=" << level << " sub_lvl=" << subdomain_level << "  subdomains=" << num_sub << "  nodes " << n_lat << "^2 x " << n_rad
              << "  ghosted " << n_lat_g << "^2 x " << n_rad_g << std::endl;

    // ---- a scalar field carrying a linear function of position -------------------------------------------
    grid::Grid4DDataScalar< ScalarType > f_owned( "f_owned", num_sub, n_lat, n_lat, n_rad );
    Kokkos::parallel_for(
        "f_init",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_sub, n_lat, n_lat, n_rad } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
            dense::Vec< ScalarType, 3 > p;
            for ( int d = 0; d < 3; ++d )
                p( d ) = coords_owned( sd, x, y, d );
            f_owned( sd, x, y, r ) = linear_field( p * radii_owned( sd, r ) );
        } );
    Kokkos::fence();

    auto f_g = exchange.allocate_scalar< ScalarType >( "f_ghosted" );
    Kokkos::deep_copy( f_g, kUnwritten );
    exchange.fill( f_owned, f_g );

    // ---- 1 + 2: consistency and geometry of the ghost layer ----------------------------------------------
    {
        long long num_ghost_written = 0;
        long long num_ghost_total   = 0;
        ScalarType max_consistency  = 0.0;
        ScalarType max_norm_error   = 0.0;

        Kokkos::parallel_reduce(
            "ghost_consistency",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 },
                                                        { num_sub, n_lat_g, n_lat_g, n_rad_g } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, long long& written,
                           long long& total, ScalarType& consistency, ScalarType& norm_err ) {
                const bool interior = x >= ghost_width && x < n_lat_g - ghost_width && y >= ghost_width &&
                                      y < n_lat_g - ghost_width && r >= ghost_width && r < n_rad_g - ghost_width;
                if ( interior )
                    return;

                total += 1;

                if ( f_g( sd, x, y, r ) == kUnwritten )
                    return;
                written += 1;

                dense::Vec< ScalarType, 3 > p;
                for ( int d = 0; d < 3; ++d )
                    p( d ) = coords_g( sd, x, y, r, d );

                norm_err = Kokkos::max( norm_err, Kokkos::abs( p.norm() - ScalarType( 1 ) ) );

                const ScalarType expected = linear_field( p * radii_g( sd, r ) );
                consistency = Kokkos::max( consistency, Kokkos::abs( f_g( sd, x, y, r ) - expected ) );
            },
            num_ghost_written,
            num_ghost_total,
            Kokkos::Max< ScalarType >( max_consistency ),
            Kokkos::Max< ScalarType >( max_norm_error ) );
        Kokkos::fence();

        std::cout << std::scientific << std::setprecision( 4 );
        std::cout << "  ghost nodes written : " << num_ghost_written << " / " << num_ghost_total << std::endl;
        std::cout << "  max |f_g - f(x_g)|  : " << max_consistency << std::endl;
        std::cout << "  max ||p_g| - 1|     : " << max_norm_error << std::endl;

        check( max_consistency < 1e-11, "ghosted field inconsistent with ghosted coordinates" );
        check( max_norm_error < 1e-11, "ghosted direction is not a unit vector" );
    }

    // ---- 3: point location and interpolation across the seams --------------------------------------------
    {
        const IndexBounds bounds{ n_lat_g, n_lat_g, n_rad_g };

        long long  num_samples = 0;
        long long  num_escaped = 0;
        long long  num_clamped = 0;
        ScalarType max_error   = 0.0;

        // Sample the nodes adjacent to the lateral subdomain boundaries -- the only ones whose departure
        // points can leave the subdomain -- and displace them in eight lateral directions plus two radial.
        Kokkos::parallel_reduce(
            "locate_across_seams",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_sub, n_lat, n_lat, n_rad } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, long long& samples,
                           long long& escaped, long long& clamped, ScalarType& err ) {
                const bool near_boundary = x <= 1 || y <= 1 || x >= n_lat - 2 || y >= n_lat - 2;
                if ( !near_boundary )
                    return;

                const int gx = to_ghosted_index( x );
                const int gy = to_ghosted_index( y );
                const int gr = to_ghosted_index( r );

                dense::Vec< ScalarType, 3 > p;
                for ( int d = 0; d < 3; ++d )
                    p( d ) = coords_g( sd, gx, gy, gr, d );
                const auto X0 = p * radii_g( sd, gr );

                // Local mesh width from an adjacent lateral node.
                const int nx = ( x + 1 < n_lat ) ? x + 1 : x - 1;
                dense::Vec< ScalarType, 3 > q;
                for ( int d = 0; d < 3; ++d )
                    q( d ) = coords_g( sd, to_ghosted_index( nx ), gy, gr, d );
                const ScalarType h = ( q * radii_g( sd, gr ) - X0 ).norm();

                // An orthonormal tangent frame at X0.
                dense::Vec< ScalarType, 3 > e_r = X0;
                e_r                             = e_r * ( ScalarType( 1 ) / e_r.norm() );
                dense::Vec< ScalarType, 3 > tmp{ ScalarType( 0 ), ScalarType( 0 ), ScalarType( 1 ) };
                if ( Kokkos::abs( e_r( 2 ) ) > ScalarType( 0.9 ) )
                    tmp = dense::Vec< ScalarType, 3 >{ ScalarType( 1 ), ScalarType( 0 ), ScalarType( 0 ) };
                dense::Vec< ScalarType, 3 > e1{ e_r( 1 ) * tmp( 2 ) - e_r( 2 ) * tmp( 1 ),
                                                e_r( 2 ) * tmp( 0 ) - e_r( 0 ) * tmp( 2 ),
                                                e_r( 0 ) * tmp( 1 ) - e_r( 1 ) * tmp( 0 ) };
                e1 = e1 * ( ScalarType( 1 ) / e1.norm() );
                dense::Vec< ScalarType, 3 > e2{ e_r( 1 ) * e1( 2 ) - e_r( 2 ) * e1( 1 ),
                                                e_r( 2 ) * e1( 0 ) - e_r( 0 ) * e1( 2 ),
                                                e_r( 0 ) * e1( 1 ) - e_r( 1 ) * e1( 0 ) };

                const WedgeCell seed{ to_ghosted_index( Kokkos::min( x, n_lat - 2 ) ),
                                      to_ghosted_index( Kokkos::min( y, n_lat - 2 ) ),
                                      to_ghosted_index( Kokkos::min( r, n_rad - 2 ) ),
                                      0 };

                for ( int k = 0; k < 8; ++k )
                {
                    const ScalarType angle = ScalarType( k ) * ScalarType( M_PI ) / ScalarType( 4 );
                    const auto       dir   = e1 * Kokkos::cos( angle ) + e2 * Kokkos::sin( angle );
                    const auto       X     = X0 + dir * ( ScalarType( 0.75 ) * h );

                    const auto res = fe::wedge::sl::locate_point(
                        X, sd, seed, lateral_g, radii_g, bounds, 8, ScalarType( 1e-12 ), true, r_min, r_max,
                        lateral_valid );

                    samples += 1;
                    if ( !res.found )
                    {
                        escaped += 1;
                        continue;
                    }
                    if ( res.clamped_radially )
                    {
                        clamped += 1;
                        continue;
                    }

                    const ScalarType interpolated = fe::wedge::sl::evaluate_q1_scalar(
                        f_g, sd, res.cell, res.xi, res.eta, res.zeta );
                    err = Kokkos::max( err, Kokkos::abs( interpolated - linear_field( X ) ) );
                }
            },
            num_samples,
            num_escaped,
            num_clamped,
            Kokkos::Max< ScalarType >( max_error ) );
        Kokkos::fence();

        std::cout << "  seam samples        : " << num_samples << std::endl;
        std::cout << "  escaped             : " << num_escaped << "  ("
                  << ( 100.0 * static_cast< double >( num_escaped ) / static_cast< double >( num_samples ) )
                  << "%)" << std::endl;
        std::cout << "  radially clamped    : " << num_clamped << std::endl;
        std::cout << "  max interp error    : " << max_error << std::endl;

        check( max_error < 1e-11, "Q1 interpolation across the ghost layer is not exact for a linear field" );
    }

    // ---- 3b: geometry of the four diagonal corner ghosts --------------------------------------------------
    // The corner ghost is reached only indirectly, by the second lateral pass propagating a neighbour's ghost
    // row. Where two seam rules of opposite orientation meet, two different neighbours can both write it, so
    // the result need not be the node that geometrically belongs there. Measure the distance from each corner
    // ghost to the owned node it diagonally extends, in units of the local mesh width: a correct corner sits at
    // about sqrt(2), a wrong one does not.
    {
        const int last = n_lat_g - 1;
        const int corners[4][2] = { { 0, 0 }, { 0, last }, { last, 0 }, { last, last } };
        // Reference against the nearest *owned* diagonal node, which is ghost_width steps in; at width 1 that
        // is the immediate neighbour, as before.
        constexpr int gw        = fe::wedge::sl::ghost_width;
        const int     inner[4][2] = { { gw, gw }, { gw, last - gw }, { last - gw, gw }, { last - gw, last - gw } };

        auto coords_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_g.comp_[0] );
        auto cy_h     = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_g.comp_[1] );
        auto cz_h     = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_g.comp_[2] );

        auto dir = [&]( int sd, int x, int y ) {
            return dense::Vec< ScalarType, 3 >{ coords_h( sd, x, y, ghost_width ), cy_h( sd, x, y, ghost_width ),
                                                cz_h( sd, x, y, ghost_width ) };
        };

        int    num_bad = 0;
        double worst   = 0;
        for ( int sd = 0; sd < num_sub; ++sd )
        {
            // Local mesh width, from an owned lateral edge near the middle of the block.
            const int  m = n_lat_g / 2;
            const auto h = ( dir( sd, m, m ) - dir( sd, m + 1, m ) ).norm();

            for ( int k = 0; k < 4; ++k )
            {
                const auto d =
                    ( dir( sd, corners[k][0], corners[k][1] ) - dir( sd, inner[k][0], inner[k][1] ) ).norm() / h;
                // The inner reference is one node diagonally inside the *owned* block, so the separation grows
                // with the layer width.
                const double expected = std::sqrt( 2.0 ) * fe::wedge::sl::ghost_width;
                worst                 = std::max( worst, std::abs( d - expected ) );
                if ( std::abs( d - expected ) > 0.5 * fe::wedge::sl::ghost_width )
                {
                    ++num_bad;
                    if ( num_bad <= 6 )
                        std::cout << "    corner ghost off: sd=" << sd << " corner(" << corners[k][0] << ","
                                  << corners[k][1] << ")  d/h=" << d << " (expected ~" << expected << ")"
                                  << std::endl;
                }
            }
        }
        // What actually distinguishes a usable corner: the orientation of the two wedges it forms. Report the
        // signed area ratio against a reference owned wedge, alongside d/h.
        {
            auto det_ratio = [&]( int sd, int cx_, int cy_ ) {
                auto tri_det = [&]( dense::Vec< ScalarType, 3 > a, dense::Vec< ScalarType, 3 > b,
                                    dense::Vec< ScalarType, 3 > c ) {
                    return dense::Mat< ScalarType, 3, 3 >::from_col_vecs( a, b, c ).det();
                };
                const int  m   = n_lat_g / 2;
                const auto ref = tri_det( dir( sd, m, m ), dir( sd, m + 1, m ), dir( sd, m, m + 1 ) );
                // The hex cell whose far corner is this ghost corner.
                const int  hx  = ( cx_ == 0 ) ? 0 : cx_ - 1;
                const int  hy  = ( cy_ == 0 ) ? 0 : cy_ - 1;
                const auto d0 =
                    tri_det( dir( sd, hx, hy ), dir( sd, hx + 1, hy ), dir( sd, hx, hy + 1 ) ) / ref;
                const auto d1 =
                    tri_det( dir( sd, hx + 1, hy + 1 ), dir( sd, hx, hy + 1 ), dir( sd, hx + 1, hy ) ) / ref;
                return std::min( d0, d1 );
            };

            double worst_good = 1e30, best_bad = -1e30;
            for ( int sd = 0; sd < num_sub; ++sd )
                for ( int k = 0; k < 4; ++k )
                {
                    const auto d =
                        ( dir( sd, corners[k][0], corners[k][1] ) - dir( sd, inner[k][0], inner[k][1] ) ).norm() /
                        ( dir( sd, n_lat_g / 2, n_lat_g / 2 ) - dir( sd, n_lat_g / 2 + 1, n_lat_g / 2 ) ).norm();
                    const auto q = det_ratio( sd, corners[k][0], corners[k][1] );
                    if ( std::abs( d - std::sqrt( 2.0 ) ) > 0.5 )
                        best_bad = std::max( best_bad, q );
                    else
                        worst_good = std::min( worst_good, q );
                }
            std::cout << "  wedge area ratio: worst good corner = " << worst_good
                      << ", best degenerate corner = " << best_bad << std::endl;
        }

        std::cout << "  corner ghosts inconsistent: " << num_bad << " / " << ( 4 * num_sub )
                  << "   worst deviation = " << worst << std::endl;

        // The degenerate corners sit at the pentagonal points of the icosahedron, where no node diagonally
        // extends the block. The d/h figure above is only a rough indicator -- the mask itself uses the wedge
        // orientation -- so what is asserted here is structural: only diagonal corners may ever be rejected,
        // some must be (a mask that rejects nothing means the detection has silently stopped working), and the
        // rejected set must stay a vanishing fraction of the block.
        const auto validity    = fe::wedge::sl::ghosted_lateral_validity< ScalarType >( exchange, coords_g );
        int        num_invalid = 0;
        int        num_non_corner_invalid = 0;
        Kokkos::parallel_reduce(
            "count_invalid",
            Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_sub, n_lat_g, n_lat_g } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, int& acc, int& non_corner ) {
                if ( validity( sd, x, y ) != 0 )
                    return;
                acc += 1;
                // The degenerate region is the ghost_width x ghost_width block at each corner: with a layer
                // wider than one node every node of that block lies diagonally outside the subdomain.
                constexpr int gw        = fe::wedge::sl::ghost_width;
                const bool    corner_x  = ( x < gw ) || ( x >= n_lat_g - gw );
                const bool    corner_y  = ( y < gw ) || ( y >= n_lat_g - gw );
                const bool    is_corner = corner_x && corner_y;
                if ( !is_corner )
                    non_corner += 1;
            },
            num_invalid,
            num_non_corner_invalid );
        Kokkos::fence();

        std::cout << "  nodes marked invalid: " << num_invalid << " of " << ( num_sub * n_lat_g * n_lat_g )
                  << std::endl;

        check( num_non_corner_invalid == 0, "validity mask rejected a node that is not a diagonal corner" );
        check( num_invalid > 0, "validity mask rejected nothing -- degenerate-corner detection is not firing" );
        check( num_invalid <= 4 * fe::wedge::sl::ghost_width * fe::wedge::sl::ghost_width * num_sub,
               "validity mask rejected more than the four corner blocks per subdomain" );
    }

    // ---- 4: every owned node must be locatable from the transport's seed rule -----------------------------
    // A departure point degenerates to the node itself as dt -> 0, so a node that cannot locate itself would
    // escape at *every* timestep regardless of the flow.
    {
        const IndexBounds bounds{ n_lat_g, n_lat_g, n_rad_g };

        constexpr int                   max_reported = 16;
        Kokkos::View< int* [4] >        failed_nodes( "failed_nodes", max_reported );
        Kokkos::View< int >             num_reported( "num_reported" );

        long long num_failed = 0;

        Kokkos::parallel_reduce(
            "self_location",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_sub, n_lat, n_lat, n_rad } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, long long& acc ) {
                const int gx = to_ghosted_index( x );
                const int gy = to_ghosted_index( y );
                const int gr = to_ghosted_index( r );

                dense::Vec< ScalarType, 3 > p;
                for ( int d = 0; d < 3; ++d )
                    p( d ) = coords_g( sd, gx, gy, gr, d );
                const auto X = p * radii_g( sd, gr );

                const WedgeCell seed{ to_ghosted_index( Kokkos::min( x, n_lat - 2 ) ),
                                      to_ghosted_index( Kokkos::min( y, n_lat - 2 ) ),
                                      to_ghosted_index( Kokkos::min( r, n_rad - 2 ) ),
                                      0 };

                const auto res = fe::wedge::sl::locate_point(
                    X, sd, seed, lateral_g, radii_g, bounds, 8, ScalarType( 1e-12 ), true, r_min, r_max,
                    lateral_valid );

                if ( !res.found )
                {
                    acc += 1;
                    const int slot = Kokkos::atomic_fetch_add( &num_reported(), 1 );
                    if ( slot < max_reported )
                    {
                        failed_nodes( slot, 0 ) = sd;
                        failed_nodes( slot, 1 ) = x;
                        failed_nodes( slot, 2 ) = y;
                        failed_nodes( slot, 3 ) = r;
                    }
                }
            },
            num_failed );
        Kokkos::fence();

        auto failed_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, failed_nodes );
        std::cout << "  self-location failures: " << num_failed << std::endl;
        for ( int i = 0; i < std::min< long long >( num_failed, max_reported ); ++i )
        {
            std::cout << "    node cannot locate itself: sd=" << failed_h( i, 0 ) << " (" << failed_h( i, 1 )
                      << "," << failed_h( i, 2 ) << "," << failed_h( i, 3 ) << ")  [n_lat=" << n_lat
                      << ", n_rad=" << n_rad << "]" << std::endl;
        }

        check( num_failed == 0, "some owned nodes cannot locate themselves" );
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    test( 3 );
    test( 4 );
    test( 4, 1 );                  // the mantle circulation app's decomposition
    test( 4, 1, 1.22, 2.22 );      // ... and its geometry

    int failures = g_failures;
    MPI_Allreduce( MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    int rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
    {
        std::cout << "\ntest_mmoc_ghost_exchange: " << ( failures == 0 ? "PASSED" : "FAILED" ) << std::endl;
    }
    return failures == 0 ? 0 : 1;
}
