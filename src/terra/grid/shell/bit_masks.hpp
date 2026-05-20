

#pragma once
#include <Kokkos_UnorderedMap.hpp>

#include "terra/util/bit_masking.hpp"

namespace terra::grid::shell {

/// \ref FlagLike that indicates boundary types for the thick spherical shell.
enum class ShellBoundaryFlag : uint8_t
{
    NO_FLAG  = 0,
    INNER    = 1 << 0,
    BOUNDARY = 1 << 1,
    CMB      = BOUNDARY | ( 1 << 2 ),
    SURFACE  = BOUNDARY | ( 1 << 3 ),
    ALL      = INNER | BOUNDARY | CMB | SURFACE,
};

static_assert( util::FlagLike< ShellBoundaryFlag > );

/// \ref FlagLike that indicates the type of boundary condition
enum class BoundaryConditionFlag : uint8_t
{
    NEUMANN   = 0, // not sure we need this one, implemented as teat_boundary == false in operators
    DIRICHLET = 1,
    FREESLIP  = 2,
};

struct BoundaryConditionMapping
{
    ShellBoundaryFlag     sbf;
    BoundaryConditionFlag bcf;
};

using BoundaryConditions = BoundaryConditionMapping[2];

/// @brief Retrieve the boundary condition flag that is associated with a location in the shell
///        e.g. SURFACE -> DIRICHLET
KOKKOS_INLINE_FUNCTION
BoundaryConditionFlag get_boundary_condition_flag( const BoundaryConditions bcs, ShellBoundaryFlag sbf )
{
    for ( int i = 0; i < 2; ++i ) // might become larger for more bc types
    {
        if ( bcs[i].sbf == sbf )
            return bcs[i].bcf;
    }
    return BoundaryConditionFlag::NEUMANN;
}

/// @brief Set the boundary condition flag that is associated with a location in the shell
///        e.g. SURFACE -> DIRICHLET
KOKKOS_INLINE_FUNCTION
void set_boundary_condition_flag( BoundaryConditions& bcs, ShellBoundaryFlag sbf, BoundaryConditionFlag bcf )
{
    for ( auto& [_sbf, _bcf] : bcs ) // might become larger for more bc types
    {
        if ( _sbf == sbf )
        {
            _bcf = bcf;
        }
    }
}

/// @brief Retrieve the ShellBoundary flag associated with a certain boundary condition type/flag
KOKKOS_INLINE_FUNCTION
ShellBoundaryFlag get_shell_boundary_flag( const BoundaryConditions bcs, BoundaryConditionFlag bcf )
{
    for ( int i = 0; i < 2; ++i ) // might become larger for more bc types
    {
        if ( bcs[i].bcf == bcf )
            return bcs[i].sbf;
    }
    return ShellBoundaryFlag::NO_FLAG;
}

/// @brief Set up mask data for a distributed shell domain.
/// The mask encodes boundary information for each grid node.
/// @param domain Distributed shell domain.
/// @return Mask data grid.
inline Grid4DDataScalar< ShellBoundaryFlag > setup_boundary_mask_data( const DistributedDomain& domain )
{
    Grid4DDataScalar< ShellBoundaryFlag > mask_data =
        grid::shell::allocate_scalar_grid< ShellBoundaryFlag >( "mask_data_shell_boundary", domain );

    auto tmp_data_for_global_subdomain_indices =
        grid::shell::allocate_scalar_grid< int64_t >( "tmp_data_for_global_subdomain_indices", domain );

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< int64_t > send_buffers( domain );
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< int64_t > recv_buffers( domain );

    // Setting boundary flags.
    // First set all nodes to inner.
    // Then overwrite for outer and inner subdomains if the nodes are at the actual boundary.

    Kokkos::parallel_for(
        "set_boundary_flags",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0 },
            { mask_data.extent( 0 ), mask_data.extent( 1 ), mask_data.extent( 2 ), mask_data.extent( 3 ) } ),
        KOKKOS_LAMBDA( const int local_subdomain_id, const int x, const int y, const int r ) {
            mask_data( local_subdomain_id, x, y, r ) = ShellBoundaryFlag::INNER;
        } );

    const int num_radial_subdomains = domain.domain_info().num_subdomains_in_radial_direction();

    for ( const auto& [subdomain_info, data] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = data;

        if ( subdomain_info.subdomain_r() == 0 )
        {
            Kokkos::parallel_for(
                "set_boundary_flags",
                Kokkos::MDRangePolicy( { 0, 0 }, { mask_data.extent( 1 ), mask_data.extent( 2 ) } ),
                KOKKOS_LAMBDA( const int x, const int y ) {
                    mask_data( local_subdomain_id, x, y, 0 ) = ShellBoundaryFlag::CMB;
                } );
        }

        if ( subdomain_info.subdomain_r() == num_radial_subdomains - 1 )
        {
           //Set here proper bounds.
            Kokkos::parallel_for(
                "set_boundary_flags",
                Kokkos::MDRangePolicy( { 0, 0 }, { mask_data.extent( 1 ), mask_data.extent( 2 ) } ),
                KOKKOS_LAMBDA( const int x, const int y ) {
                    mask_data( local_subdomain_id, x, y, mask_data.extent( 3 ) - 1 ) = ShellBoundaryFlag::SURFACE;
                } );
        }
    }

    Kokkos::fence();

    return mask_data;
}

} // namespace terra::grid::shell