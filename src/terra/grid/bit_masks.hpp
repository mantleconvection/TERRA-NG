

#pragma once
#include "terra/communication/shell/communication.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/util/bit_masking.hpp"

namespace terra::grid {

/// @brief \ref FlagLike enum class that indicates whether a node is owned on a subdomain.
///
/// Each node of the grid is either owned or not owned. Nodes that are duplicated due to the domain partitioning
/// into subdomains must somehow be treated properly in kernels like dot products etc. Values at duplicate nodes
/// typically must not be added twice. This enum shall mark exactly one node of each set of duplicated nodes (and all
/// the non-duplicated nodes in the subdomain interior) as owned.
///
enum class NodeOwnershipFlag : uint8_t
{
    NO_FLAG = 0,
    OWNED   = 1
};

static_assert( util::FlagLike< NodeOwnershipFlag > );

/// @brief Set up mask data for a distributed shell domain.
/// The mask encodes ownership information for each grid node.
/// @param domain Distributed shell domain.
/// @return Mask data grid.
inline Grid4DDataScalar< NodeOwnershipFlag > setup_node_ownership_mask_data( const shell::DistributedDomain& domain )
{
    Grid4DDataScalar< NodeOwnershipFlag > mask_data =
        grid::shell::allocate_scalar_grid< NodeOwnershipFlag >( "mask_data_node_ownership", domain );

    auto tmp_data_for_global_subdomain_indices =
        grid::shell::allocate_scalar_grid< int64_t >( "tmp_data_for_global_subdomain_indices", domain );

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< int64_t > send_buffers( domain );
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< int64_t > recv_buffers( domain );

    // Interpolate the unique subdomain ID.
    for ( const auto& [subdomain_info, value] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = value;

        const auto global_subdomain_id = subdomain_info.global_id();

        Kokkos::parallel_for(
            "set_global_subdomain_id",
            Kokkos::MDRangePolicy< Kokkos::Rank< 3, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 0, 0 }, { mask_data.extent( 1 ), mask_data.extent( 2 ), mask_data.extent( 3 ) } ),
            KOKKOS_LAMBDA( const int x, const int y, const int r ) {
                tmp_data_for_global_subdomain_indices( local_subdomain_id, x, y, r ) = global_subdomain_id;
            } );
    }

    // Communicate and reduce with minimum.
    terra::communication::shell::pack_send_and_recv_local_subdomain_boundaries(
        domain, tmp_data_for_global_subdomain_indices, send_buffers, recv_buffers );

    terra::communication::shell::unpack_and_reduce_local_subdomain_boundaries(
        domain, tmp_data_for_global_subdomain_indices, recv_buffers, communication::CommunicationReduction::MIN );

    // Set all nodes to 1 if the global_subdomain_id matches - 0 otherwise.
    for ( const auto& [subdomain_info, value] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = value;

        const auto global_subdomain_id = subdomain_info.global_id();

        Kokkos::parallel_for(
            "set_node_owner_flags",
            Kokkos::MDRangePolicy< Kokkos::Rank< 3, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 0, 0 }, { mask_data.extent( 1 ), mask_data.extent( 2 ), mask_data.extent( 3 ) } ),
            KOKKOS_LAMBDA( const int x, const int y, const int r ) {
                if ( tmp_data_for_global_subdomain_indices( local_subdomain_id, x, y, r ) == global_subdomain_id )
                {
                    mask_data( local_subdomain_id, x, y, r ) = NodeOwnershipFlag::OWNED;
                }
            } );

        Kokkos::fence();
    }

    return mask_data;
}

} // namespace terra::grid