#pragma once

#include "fe/wedge/sl/ghost_exchange.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"

/// @file
///
/// Ghosted mesh geometry for the semi-Lagrangian transport: the unit-sphere node directions and the shell
/// radii, extended by the ghost layer so that a departure point just outside a subdomain can be located and
/// interpolated.
///
/// **The degenerate diagonal corners.** A corner ghost node is never sent directly; it arrives because the
/// second lateral pass propagates a neighbour's already-filled ghost row. That works wherever the index space
/// is locally a regular grid. It does not work at the corners of a diamond, where five diamonds meet at a
/// pentagonal point of the icosahedron: there is no fourth quadrant, so no node diagonally extends the corner,
/// and the sweep deposits whichever face-ghost node its last writer happened to carry. Measured, such a corner
/// sits about 0.88 mesh widths from the node it should extend instead of sqrt(2).
///
/// \ref ghosted_lateral_validity detects exactly those nodes geometrically and marks them invalid, and point
/// location refuses any wedge that touches an invalid node. The values are self-consistent (coordinates and
/// field data are filled by the same exchange, so both belong to the same physical node), so nothing would be
/// interpolated at a wrong *place* without the mask -- but the wedge they span is degenerate, and rejecting it
/// explicitly is what lets the transport fall back in a controlled way instead of letting the walk wander.

namespace terra::fe::wedge::sl
{

/// @brief Unit-sphere node directions on the ghosted lateral index space.
///
/// The value is independent of the radial index and is replicated along it, so that the lateral ghost exchange
/// (which transports whole radial columns) fills every radial layer including the radial ghosts.
template < std::floating_point T >
grid::Grid4DDataVec< T, 3 > ghosted_unit_sphere_coords(
    const grid::shell::DistributedDomain& domain,
    const GhostExchange&                  exchange )
{
    const auto owned = grid::shell::subdomain_unit_sphere_single_shell_coords< T >( domain );

    auto ghosted = exchange.allocate_vec< T, 3 >( "ghosted_unit_sphere_coords" );

    const int num_sub  = exchange.num_subdomains();
    const int num_lat  = exchange.num_nodes_lateral();
    const int num_radg = exchange.num_nodes_radial_ghosted();

    // Replicate the lateral directions over every radial layer, ghosts included.
    Kokkos::parallel_for(
        "sl_ghosted_coords_interior",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_sub, num_lat, num_lat, num_radg } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
            for ( int d = 0; d < 3; ++d )
                ghosted( sd, to_ghosted_index( x ), to_ghosted_index( y ), r, d ) = owned( sd, x, y, d );
        } );
    Kokkos::fence();

    exchange.exchange_lateral( ghosted );

    return ghosted;
}

/// @brief Shell radii on the ghosted radial index space.
///
/// Radial ghosts inside the domain hold the neighbouring radial subdomain's node radius. Beyond the CMB and the
/// surface there is no such node; those entries are linearly extrapolated so that the array stays strictly
/// monotone (the radial search assumes it). They are never interpolated at: departure points outside the shell
/// are clamped onto the physical boundary radii, which are returned separately by \ref shell_radius_bounds.
template < std::floating_point T >
grid::Grid2DDataScalar< T > ghosted_shell_radii(
    const grid::shell::DistributedDomain& domain,
    const GhostExchange&                  exchange )
{
    const int  num_nodes_rad   = exchange.num_nodes_radial();
    const int  num_nodes_rad_g = exchange.num_nodes_radial_ghosted();
    const int  layers          = num_nodes_rad - 1;
    const auto radii           = domain.domain_info().radii();
    const int  num_global      = static_cast< int >( radii.size() );

    grid::Grid2DDataScalar< T > device( "sl_ghosted_shell_radii", exchange.num_subdomains(), num_nodes_rad_g );
    auto                        host = Kokkos::create_mirror_view( device );

    for ( const auto& [subdomain_info, data] : domain.subdomains() )
    {
        const int subdomain_idx = std::get< 0 >( data );
        const int innermost     = subdomain_info.subdomain_r() * layers;

        for ( int j = 0; j < num_nodes_rad_g; ++j )
        {
            const int global_idx = innermost + j - ghost_width;

            if ( global_idx >= 0 && global_idx < num_global )
            {
                host( subdomain_idx, j ) = static_cast< T >( radii[global_idx] );
            }
            else if ( global_idx < 0 )
            {
                // Extrapolate below the CMB.
                const T dr = static_cast< T >( radii[1] - radii[0] );
                host( subdomain_idx, j ) = static_cast< T >( radii[0] ) + global_idx * dr;
            }
            else
            {
                // Extrapolate above the surface.
                const T dr = static_cast< T >( radii[num_global - 1] - radii[num_global - 2] );
                host( subdomain_idx, j ) =
                    static_cast< T >( radii[num_global - 1] ) + ( global_idx - ( num_global - 1 ) ) * dr;
            }
        }
    }

    Kokkos::deep_copy( device, host );
    return device;
}

/// @brief Marks ghost lateral nodes that do not geometrically extend the block.
///
/// Only the four diagonal corners of the ghosted lateral index space can be wrong (see the file comment), and
/// only at the corners of a diamond. The test is on the *orientation* of the two wedges the corner completes,
/// measured against a reference wedge of the same winding taken from the middle of the owned block. A corner
/// that genuinely extends the block yields a ratio near one; a degenerate one is either exactly zero (the node
/// deposited there duplicates another vertex of the same triangle, so the triangle collapses) or negative (the
/// triangle is inverted). Both were observed -- collapsed with one subdomain per diamond, inverted with four --
/// and both are caught by requiring a positive ratio, which needs no tolerance on mesh stretching.
template < std::floating_point T >
Kokkos::View< uint8_t*** > ghosted_lateral_validity(
    const GhostExchange&               exchange,
    const grid::Grid4DDataVec< T, 3 >& coords_g )
{
    const int num_sub = exchange.num_subdomains();
    const int n_lat_g = exchange.num_nodes_lateral_ghosted();

    Kokkos::View< uint8_t*** > valid( "sl_lateral_validity", num_sub, n_lat_g, n_lat_g );
    Kokkos::deep_copy( valid, static_cast< uint8_t >( 1 ) );

    const int last = n_lat_g - 1;

    Kokkos::parallel_for(
        "sl_lateral_validity",
        // Every node of the ghost_width x ghost_width block at each of the four corners, not just the
        // outermost one: with a layer wider than one node the whole block sits diagonally outside the
        // subdomain, where the two-pass face exchange has nothing consistent to propagate.
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { num_sub, 4, ghost_width, ghost_width } ),
        KOKKOS_LAMBDA( const int sd, const int k, const int i, const int j ) {
            const int corner_x = ( k & 1 ) ? last - i : i;
            const int corner_y = ( k & 2 ) ? last - j : j;

            auto dir = [&]( const int x, const int y ) {
                dense::Vec< T, 3 > v;
                for ( int d = 0; d < 3; ++d )
                    v( d ) = coords_g( sd, x, y, ghost_width, d );
                return v;
            };

            auto det3 = [&]( const dense::Vec< T, 3 >& a,
                             const dense::Vec< T, 3 >& b,
                             const dense::Vec< T, 3 >& c ) {
                return dense::Mat< T, 3, 3 >::from_col_vecs( a, b, c ).det();
            };

            // Reference wedges of each winding, from a hex cell well inside the owned block.
            const int m = n_lat_g / 2;
            const T   ref_0 = det3( dir( m, m ), dir( m + 1, m ), dir( m, m + 1 ) );
            const T   ref_1 = det3( dir( m + 1, m + 1 ), dir( m, m + 1 ), dir( m + 1, m ) );

            // The hex cell that has this ghost node as its outer corner: on the low side the cell starts at
            // the node, on the high side it ends there.
            const int hx = ( corner_x < ghost_width ) ? corner_x : corner_x - 1;
            const int hy = ( corner_y < ghost_width ) ? corner_y : corner_y - 1;

            const T q_0 = det3( dir( hx, hy ), dir( hx + 1, hy ), dir( hx, hy + 1 ) ) / ref_0;
            const T q_1 = det3( dir( hx + 1, hy + 1 ), dir( hx, hy + 1 ), dir( hx + 1, hy ) ) / ref_1;

            constexpr T min_area_ratio = T( 0.1 );

            if ( q_0 < min_area_ratio || q_1 < min_area_ratio )
                valid( sd, corner_x, corner_y ) = 0;
        } );
    Kokkos::fence();

    return valid;
}

/// @brief Physical radii of the CMB and the surface, i.e. the clamp bounds for departure points.
template < std::floating_point T >
std::pair< T, T > shell_radius_bounds( const grid::shell::DistributedDomain& domain )
{
    const auto& radii = domain.domain_info().radii();
    return { static_cast< T >( radii.front() ), static_cast< T >( radii.back() ) };
}

} // namespace terra::fe::wedge::sl
