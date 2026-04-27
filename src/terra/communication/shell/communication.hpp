#pragma once

#include <ranges>
#include <variant>
#include <vector>

#include "dense/vec.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "terra/communication/buffer_copy_kernels.hpp"

using terra::grid::shell::SubdomainNeighborhood;

namespace terra::communication::shell {

constexpr int MPI_TAG_BOUNDARY_DATA = 100;

namespace detail {

// Build an unmanaged view with the *same* data_type/layout/device as a Grid*DDataVec,
// pointing into a raw pointer slice.
template < class GridViewT >
auto make_unmanaged_like( typename GridViewT::value_type* ptr, int n0 = 0, int n1 = 0, int n2 = 0 )
{
    using data_type    = typename GridViewT::data_type;
    using array_layout = typename GridViewT::array_layout;
    using device_type  = typename GridViewT::device_type;

    using unmanaged_view =
        Kokkos::View< data_type, array_layout, device_type, Kokkos::MemoryTraits< Kokkos::Unmanaged > >;

    if constexpr ( GridViewT::rank == 1 )
        return unmanaged_view( ptr, n0 );
    else if constexpr ( GridViewT::rank == 2 )
        return unmanaged_view( ptr, n0, n1 );
    else if constexpr ( GridViewT::rank == 3 )
        return unmanaged_view( ptr, n0, n1, n2 );
    else
        static_assert( GridViewT::rank >= 1 && GridViewT::rank <= 3, "Unsupported rank for unmanaged-like helper." );
}

} // namespace detail

/// @brief Send and receive buffers for all process-local subdomain boundaries.
///
/// Allocates views for all boundaries of local subdomains. Those are the nodes that overlap with values from
/// neighboring subdomains.
///
/// One buffer per local boundary + neighbor is allocated. So, for instance, for an edge shared with several
/// neighbors, just as many buffers as neighbors are allocated. This facilitates the receiving step since all
/// neighbors that a subdomain receives data from can send their data simultaneously.
///
/// Can be reused after communication (send + recv) has been completed to avoid unnecessary reallocation.
template < typename ScalarType, int VecDim = 1 >
class SubdomainNeighborhoodSendRecvBuffer
{
  public:
    /// @brief Constructs a SubdomainNeighborhoodSendRecvBuffer for the passed distributed domain object.
    explicit SubdomainNeighborhoodSendRecvBuffer( const grid::shell::DistributedDomain& domain )
    {
        setup_buffers( domain );
    }

    /// @brief Const reference to the view that is a buffer for a vertex of a subdomain.
    ///
    /// @param local_subdomain the SubdomainInfo identifying the local subdomain
    /// @param local_boundary_vertex the boundary vertex of the local subdomain
    /// @param neighbor_subdomain the SubdomainInfo identifying the neighboring subdomain
    /// @param neighbor_boundary_vertex the boundary vertex of the neighboring subdomain
    ///
    /// @return A const ref to a Kokkos::View with shape (VecDim), where VecDim is the number of scalars per node (class
    ///         template parameter).
    const grid::Grid0DDataVec< ScalarType, VecDim >& buffer_vertex(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryVertex        local_boundary_vertex,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryVertex        neighbor_boundary_vertex ) const
    {
        return buffers_vertex_.at(
            { local_subdomain, local_boundary_vertex, neighbor_subdomain, neighbor_boundary_vertex } );
    }

    /// @brief Const reference to the view that is a buffer for an edge of a subdomain.
    ///
    /// @param local_subdomain the SubdomainInfo identifying the local subdomain
    /// @param local_boundary_edge the boundary edge of the local subdomain
    /// @param neighbor_subdomain the SubdomainInfo identifying the neighboring subdomain
    /// @param neighbor_boundary_edge the boundary edge of the neighboring subdomain
    ///
    /// @return A const ref to a Kokkos::View with shape (N, VecDim), where N is the number of grid nodes on the edge
    ///         and VecDim is the number of scalars per node (class template parameter).
    const grid::Grid1DDataVec< ScalarType, VecDim >& buffer_edge(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryEdge          local_boundary_edge,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryEdge          neighbor_boundary_edge ) const
    {
        return buffers_edge_.at( { local_subdomain, local_boundary_edge, neighbor_subdomain, neighbor_boundary_edge } );
    }

    /// @brief Const reference to the view that is a buffer for a face of a subdomain.
    ///
    /// @param local_subdomain the SubdomainInfo identifying the local subdomain
    /// @param local_boundary_face the boundary face of the local subdomain
    /// @param neighbor_subdomain the SubdomainInfo identifying the neighboring subdomain
    /// @param neighbor_boundary_face the boundary face of the neighboring subdomain
    ///
    /// @return A const ref to a Kokkos::View with shape (N, M, VecDim), where N, M are the number of grid nodes on
    ///         each side of the face and VecDim is the number of scalars per node (class template parameter).
    const grid::Grid2DDataVec< ScalarType, VecDim >& buffer_face(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryFace          local_boundary_face,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryFace          neighbor_boundary_face ) const
    {
        return buffers_face_.at( { local_subdomain, local_boundary_face, neighbor_subdomain, neighbor_boundary_face } );
    }

    /// @brief Mutable reference to the view that is a buffer for a vertex of a subdomain.
    grid::Grid0DDataVec< ScalarType, VecDim >& buffer_vertex(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryVertex        local_boundary_vertex,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryVertex        neighbor_boundary_vertex )
    {
        return buffers_vertex_.at(
            { local_subdomain, local_boundary_vertex, neighbor_subdomain, neighbor_boundary_vertex } );
    }

    /// @brief Mutable reference to the view that is a buffer for an edge of a subdomain.
    grid::Grid1DDataVec< ScalarType, VecDim >& buffer_edge(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryEdge          local_boundary_edge,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryEdge          neighbor_boundary_edge )
    {
        return buffers_edge_.at( { local_subdomain, local_boundary_edge, neighbor_subdomain, neighbor_boundary_edge } );
    }

    /// @brief Mutable reference to the view that is a buffer for a face of a subdomain.
    grid::Grid2DDataVec< ScalarType, VecDim >& buffer_face(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryFace          local_boundary_face,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryFace          neighbor_boundary_face )
    {
        return buffers_face_.at( { local_subdomain, local_boundary_face, neighbor_subdomain, neighbor_boundary_face } );
    }

  private:
    /// @brief Helper called in the ctor that allocates the buffers.
    void setup_buffers( const grid::shell::DistributedDomain& domain )
    {
        for ( const auto& [subdomain_info, data] : domain.subdomains() )
        {
            const auto& [local_subdomain_idx, neighborhood] = data;

            for ( const auto& [local_boundary_vertex, neighbor] : neighborhood.neighborhood_vertex() )
            {
                for ( const auto& [neighbor_subdomain, neighbor_boundary_vertex, mpi_rank] : neighbor )
                {
                    buffers_vertex_[{
                        subdomain_info, local_boundary_vertex, neighbor_subdomain, neighbor_boundary_vertex }] =
                        grid::Grid0DDataVec< ScalarType, VecDim >( "recv_buffer" );
                }
            }

            for ( const auto& [local_boundary_edge, neighbor] : neighborhood.neighborhood_edge() )
            {
                for ( const auto& [neighbor_subdomain, neighbor_boundary_edge, _, mpi_rank] : neighbor )
                {
                    const int buffer_size = grid::is_edge_boundary_radial( local_boundary_edge ) ?
                                                domain.domain_info().subdomain_num_nodes_radially() :
                                                domain.domain_info().subdomain_num_nodes_per_side_laterally();

                    buffers_edge_[{ subdomain_info, local_boundary_edge, neighbor_subdomain, neighbor_boundary_edge }] =
                        grid::Grid1DDataVec< ScalarType, VecDim >( "recv_buffer", buffer_size );
                }
            }

            for ( const auto& [local_boundary_face, neighbor] : neighborhood.neighborhood_face() )
            {
                const auto& [neighbor_subdomain, neighbor_boundary_face, _, mpi_rank] = neighbor;

                const int buffer_size_i = domain.domain_info().subdomain_num_nodes_per_side_laterally();
                const int buffer_size_j = grid::is_face_boundary_normal_to_radial_direction( local_boundary_face ) ?
                                              domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                              domain.domain_info().subdomain_num_nodes_radially();

                buffers_face_[{ subdomain_info, local_boundary_face, neighbor_subdomain, neighbor_boundary_face }] =
                    grid::Grid2DDataVec< ScalarType, VecDim >( "recv_buffer", buffer_size_i, buffer_size_j );
            }
        }
    }

    /// Key ordering: local subdomain, local boundary, neighbor subdomain, neighbor-local boundary
    std::map<
        std::
            tuple< grid::shell::SubdomainInfo, grid::BoundaryVertex, grid::shell::SubdomainInfo, grid::BoundaryVertex >,
        grid::Grid0DDataVec< ScalarType, VecDim > >
        buffers_vertex_;
    std::map<
        std::tuple< grid::shell::SubdomainInfo, grid::BoundaryEdge, grid::shell::SubdomainInfo, grid::BoundaryEdge >,
        grid::Grid1DDataVec< ScalarType, VecDim > >
        buffers_edge_;
    std::map<
        std::tuple< grid::shell::SubdomainInfo, grid::BoundaryFace, grid::shell::SubdomainInfo, grid::BoundaryFace >,
        grid::Grid2DDataVec< ScalarType, VecDim > >
        buffers_face_;
};

/// @brief Packs, sends and recvs local subdomain boundaries using two sets of buffers.
///
/// Communication works like this:
/// - data is packed from the boundaries of the grid data structure into send buffers
/// - the send buffers are sent via MPI
/// - the data is received in receive buffers
/// - the receive buffers are unpacked into the grid data structure (and the data is potentially rotated if necessary)
///
/// If the sending and receiving subdomains are on the same process, the data is directly packed into the recv buffers.
/// However, not yet directly written from subdomain A to subdomain B. This and further optimizations are obviously
/// possible.
///
/// @note Must be complemented with `unpack_and_reduce_local_subdomain_boundaries()` to complete communication.
///       This function waits until all recv buffers are filled - but does not unpack.
///
/// Performs "additive" communication. Nodes at the subdomain interfaces overlap and will be reduced using some
/// reduction mode during the receiving phase. This is typically required for matrix-free matrix-vector multiplications
/// in a finite element context: nodes that are shared by elements of two neighboring subdomains receive contributions
/// from both subdomains that need to be added. In this case, the required reduction mode is `CommunicationReduction::SUM`.
///
/// The send buffers are only required until this function returns.
/// The recv buffers must be passed to the corresponding unpacking function `recv_unpack_and_add_local_subdomain_boundaries()`.
///
/// @param domain the DistributedDomain that this works on
/// @param data the data (Kokkos::View) to be communicated
/// @param boundary_send_buffers SubdomainNeighborhoodSendRecvBuffer instance that serves for sending data - can be
///                              reused after this function returns
/// @param boundary_recv_buffers SubdomainNeighborhoodSendRecvBuffer instance that serves for receiving data - must be
///                              passed to `unpack_and_reduce_local_subdomain_boundaries()`
template < typename GridDataType >
void pack_send_and_recv_local_subdomain_boundaries(
    const grid::shell::DistributedDomain& domain,
    const GridDataType&                   data,
    SubdomainNeighborhoodSendRecvBuffer< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() >&
        boundary_send_buffers,
    SubdomainNeighborhoodSendRecvBuffer< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() >&
        boundary_recv_buffers )
{
    // Left this switch here (for debugging). Toggle to choose between
    // - copying data directly into recv buffers if sender and receiver subdomains
    //   are on the same rank (enable_local_comm == true) and
    // - still simply sending via MPI: first copy to send buffer, send, then copy from recv buffer to data
    //   (enable_local_comm == false)
    // Further optimizations are possible ofc.
    constexpr bool enable_local_comm = true;

    // Since it is not clear whether a static last dimension of 1 impacts performance, we want to support both
    // scalar and vector-valued grid data views. To simplify matters, we always use the vector-valued versions for the
    // buffers.

    static_assert(
        std::is_same_v< GridDataType, grid::Grid4DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v<
            GridDataType,
            grid::Grid4DDataVec< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() > > );

    using ScalarType = typename GridDataType::value_type;

    // std::vector< MPI_Request >                              metadata_send_requests;
    // std::vector< std::unique_ptr< std::array< int, 11 > > > metadata_send_buffers;

    std::vector< MPI_Request > data_send_requests;

    ////////////////////////////////////////////
    // Collecting and sorting send-recv pairs //
    ////////////////////////////////////////////

    // First, we collect all the send-recv pairs and sort them.
    // This ensures the same message order per process.
    // We need to post the Isends and Irecvs in that correct order (per process pair).

    struct SendRecvPair
    {
        int                        boundary_type = -1;
        mpi::MPIRank               local_rank;
        grid::shell::SubdomainInfo local_subdomain;
        int                        local_subdomain_boundary;
        int                        local_subdomain_id;
        mpi::MPIRank               neighbor_rank;
        grid::shell::SubdomainInfo neighbor_subdomain;
        int                        neighbor_subdomain_boundary;

        std::string to_string() const
        {
            std::stringstream ss;
            ss << "boundary_type: " << boundary_type << ", ";
            ss << "local_subdomain: " << local_subdomain << ", ";
            if ( boundary_type == 0 )
            {
                ss << "local_subdomain_boundary: " << static_cast< grid::BoundaryVertex >( local_subdomain_boundary )
                   << ", ";
            }
            else if ( boundary_type == 1 )
            {
                ss << "local_subdomain_boundary: " << static_cast< grid::BoundaryEdge >( local_subdomain_boundary )
                   << ", ";
            }
            else if ( boundary_type == 2 )
            {
                ss << "local_subdomain_boundary: " << static_cast< grid::BoundaryFace >( local_subdomain_boundary )
                   << ", ";
            }
            ss << "neighbor_subdomain: " << neighbor_subdomain << ", ";
            if ( boundary_type == 0 )
            {
                ss << "neighbor_subdomain_boundary: "
                   << static_cast< grid::BoundaryVertex >( neighbor_subdomain_boundary ) << ", ";
            }
            else if ( boundary_type == 1 )
            {
                ss << "neighbor_subdomain_boundary: "
                   << static_cast< grid::BoundaryEdge >( neighbor_subdomain_boundary ) << ", ";
            }
            else if ( boundary_type == 2 )
            {
                ss << "neighbor_subdomain_boundary: "
                   << static_cast< grid::BoundaryFace >( neighbor_subdomain_boundary ) << ", ";
            }

            return ss.str();
        }
    };

    std::vector< SendRecvPair > send_recv_pairs;

    for ( const auto& [local_subdomain_info, idx_and_neighborhood] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = idx_and_neighborhood;

        for ( const auto& [local_vertex_boundary, neighbors] : neighborhood.neighborhood_vertex() )
        {
            // Multiple neighbor subdomains per vertex.
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

                SendRecvPair send_recv_pair{
                    .boundary_type               = 0,
                    .local_rank                  = mpi::rank( domain.comm() ),
                    .local_subdomain             = local_subdomain_info,
                    .local_subdomain_boundary    = static_cast< int >( local_vertex_boundary ),
                    .local_subdomain_id          = local_subdomain_id,
                    .neighbor_rank               = neighbor_rank,
                    .neighbor_subdomain          = neighbor_subdomain_info,
                    .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ) };
                send_recv_pairs.push_back( send_recv_pair );
            }
        }

        for ( const auto& [local_edge_boundary, neighbors] : neighborhood.neighborhood_edge() )
        {
            // Multiple neighbor subdomains per edge.
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, _, neighbor_rank] = neighbor;

                SendRecvPair send_recv_pair{
                    .boundary_type               = 1,
                    .local_rank                  = mpi::rank( domain.comm() ),
                    .local_subdomain             = local_subdomain_info,
                    .local_subdomain_boundary    = static_cast< int >( local_edge_boundary ),
                    .local_subdomain_id          = local_subdomain_id,
                    .neighbor_rank               = neighbor_rank,
                    .neighbor_subdomain          = neighbor_subdomain_info,
                    .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ) };
                send_recv_pairs.push_back( send_recv_pair );
            }
        }

        for ( const auto& [local_face_boundary, neighbor] : neighborhood.neighborhood_face() )
        {
            // Single neighbor subdomain per facet.
            const auto& [neighbor_subdomain_info, neighbor_local_boundary, _, neighbor_rank] = neighbor;

            SendRecvPair send_recv_pair{
                .boundary_type               = 2,
                .local_rank                  = mpi::rank( domain.comm() ),
                .local_subdomain             = local_subdomain_info,
                .local_subdomain_boundary    = static_cast< int >( local_face_boundary ),
                .local_subdomain_id          = local_subdomain_id,
                .neighbor_rank               = neighbor_rank,
                .neighbor_subdomain          = neighbor_subdomain_info,
                .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ) };
            send_recv_pairs.push_back( send_recv_pair );
        }
    }

    ////////////////////
    // Posting Irecvs //
    ////////////////////

    // Sort the pairs by sender subdomains.
    std::sort( send_recv_pairs.begin(), send_recv_pairs.end(), []( const SendRecvPair& a, const SendRecvPair& b ) {
        if ( a.boundary_type != b.boundary_type )
            return a.boundary_type < b.boundary_type;
        if ( a.neighbor_subdomain != b.neighbor_subdomain )
            return a.neighbor_subdomain < b.neighbor_subdomain;
        if ( a.neighbor_subdomain_boundary != b.neighbor_subdomain_boundary )
            return a.neighbor_subdomain_boundary < b.neighbor_subdomain_boundary;
        if ( a.local_subdomain != b.local_subdomain )
            return a.local_subdomain < b.local_subdomain;
        return a.local_subdomain_boundary < b.local_subdomain_boundary;
    } );

    std::vector< MPI_Request > data_recv_requests;

    for ( const auto& send_recv_pair : send_recv_pairs )
    {
        // We will handle local communication via direct copies in the send loop.
        if ( enable_local_comm && send_recv_pair.local_rank == send_recv_pair.neighbor_rank )
        {
            continue;
        }

        ScalarType* recv_buffer_ptr  = nullptr;
        int         recv_buffer_size = 0;

        if ( send_recv_pair.boundary_type == 0 )
        {
            const auto& recv_buffer = boundary_recv_buffers.buffer_vertex(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryVertex >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryVertex >( send_recv_pair.neighbor_subdomain_boundary ) );

            recv_buffer_ptr  = recv_buffer.data();
            recv_buffer_size = recv_buffer.span();
        }
        else if ( send_recv_pair.boundary_type == 1 )
        {
            const auto& recv_buffer = boundary_recv_buffers.buffer_edge(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryEdge >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryEdge >( send_recv_pair.neighbor_subdomain_boundary ) );

            recv_buffer_ptr  = recv_buffer.data();
            recv_buffer_size = recv_buffer.span();
        }
        else if ( send_recv_pair.boundary_type == 2 )
        {
            const auto& recv_buffer = boundary_recv_buffers.buffer_face(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryFace >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryFace >( send_recv_pair.neighbor_subdomain_boundary ) );

            recv_buffer_ptr  = recv_buffer.data();
            recv_buffer_size = recv_buffer.span();
        }
        else
        {
            Kokkos::abort( "Unknown boundary type" );
        }

        MPI_Request data_recv_request;
        MPI_Irecv(
            recv_buffer_ptr,
            recv_buffer_size,
            mpi::mpi_datatype< ScalarType >(),
            send_recv_pair.neighbor_rank,
            MPI_TAG_BOUNDARY_DATA,
            domain.comm(),
            &data_recv_request );
        data_recv_requests.push_back( data_recv_request );
    }

    /////////////////////////////////////////////////
    // Packing send data buffers and posting sends //
    /////////////////////////////////////////////////

    // Sort the pairs by sender subdomains.
    std::sort( send_recv_pairs.begin(), send_recv_pairs.end(), []( const SendRecvPair& a, const SendRecvPair& b ) {
        if ( a.boundary_type != b.boundary_type )
            return a.boundary_type < b.boundary_type;
        if ( a.local_subdomain != b.local_subdomain )
            return a.local_subdomain < b.local_subdomain;
        if ( a.local_subdomain_boundary != b.local_subdomain_boundary )
            return a.local_subdomain_boundary < b.local_subdomain_boundary;
        if ( a.neighbor_subdomain != b.neighbor_subdomain )
            return a.neighbor_subdomain < b.neighbor_subdomain;
        return a.neighbor_subdomain_boundary < b.neighbor_subdomain_boundary;
    } );

    for ( const auto& send_recv_pair : send_recv_pairs )
    {
        const auto local_comm = enable_local_comm && send_recv_pair.local_rank == send_recv_pair.neighbor_rank;

        // Packing buffer.

        const auto local_subdomain_id = send_recv_pair.local_subdomain_id;

        // Deep-copy into device-side send buffer.

        ScalarType* send_buffer_ptr  = nullptr;
        int         send_buffer_size = 0;

        if ( send_recv_pair.boundary_type == 0 )
        {
            const auto local_vertex_boundary =
                static_cast< grid::BoundaryVertex >( send_recv_pair.local_subdomain_boundary );

            if ( local_comm )
            {
                // Handling local communication and moving on to next send_recv_pair afterward.
                const auto& recv_buffer = boundary_recv_buffers.buffer_vertex(
                    send_recv_pair.local_subdomain,
                    static_cast< grid::BoundaryVertex >( send_recv_pair.local_subdomain_boundary ),
                    send_recv_pair.neighbor_subdomain,
                    static_cast< grid::BoundaryVertex >( send_recv_pair.neighbor_subdomain_boundary ) );

                if ( !domain.subdomains().contains( send_recv_pair.neighbor_subdomain ) )
                {
                    Kokkos::abort( "Subdomain not found locally - but it should be there..." );
                }

                const auto local_subdomain_id_of_neighboring_subdomain =
                    std::get< 0 >( domain.subdomains().at( send_recv_pair.neighbor_subdomain ) );

                copy_to_buffer(
                    recv_buffer,
                    data,
                    local_subdomain_id_of_neighboring_subdomain,
                    static_cast< grid::BoundaryVertex >( send_recv_pair.neighbor_subdomain_boundary ) );
                continue;
            }

            auto& send_buffer = boundary_send_buffers.buffer_vertex(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryVertex >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryVertex >( send_recv_pair.neighbor_subdomain_boundary ) );

            send_buffer_ptr  = send_buffer.data();
            send_buffer_size = send_buffer.span();

            copy_to_buffer( send_buffer, data, local_subdomain_id, local_vertex_boundary );
        }
        else if ( send_recv_pair.boundary_type == 1 )
        {
            const auto local_edge_boundary =
                static_cast< grid::BoundaryEdge >( send_recv_pair.local_subdomain_boundary );

            if ( local_comm )
            {
                // Handling local communication and moving on to next send_recv_pair afterward.
                const auto& recv_buffer = boundary_recv_buffers.buffer_edge(
                    send_recv_pair.local_subdomain,
                    static_cast< grid::BoundaryEdge >( send_recv_pair.local_subdomain_boundary ),
                    send_recv_pair.neighbor_subdomain,
                    static_cast< grid::BoundaryEdge >( send_recv_pair.neighbor_subdomain_boundary ) );

                if ( !domain.subdomains().contains( send_recv_pair.neighbor_subdomain ) )
                {
                    Kokkos::abort( "Subdomain not found locally - but it should be there..." );
                }

                const auto local_subdomain_id_of_neighboring_subdomain =
                    std::get< 0 >( domain.subdomains().at( send_recv_pair.neighbor_subdomain ) );

                copy_to_buffer(
                    recv_buffer,
                    data,
                    local_subdomain_id_of_neighboring_subdomain,
                    static_cast< grid::BoundaryEdge >( send_recv_pair.neighbor_subdomain_boundary ) );
                continue;
            }

            auto& send_buffer = boundary_send_buffers.buffer_edge(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryEdge >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryEdge >( send_recv_pair.neighbor_subdomain_boundary ) );

            send_buffer_ptr  = send_buffer.data();
            send_buffer_size = send_buffer.span();

            copy_to_buffer( send_buffer, data, local_subdomain_id, local_edge_boundary );
        }
        else if ( send_recv_pair.boundary_type == 2 )
        {
            const auto local_face_boundary =
                static_cast< grid::BoundaryFace >( send_recv_pair.local_subdomain_boundary );

            if ( local_comm )
            {
                // Handling local communication and moving on to next send_recv_pair afterward.
                const auto& recv_buffer = boundary_recv_buffers.buffer_face(
                    send_recv_pair.local_subdomain,
                    static_cast< grid::BoundaryFace >( send_recv_pair.local_subdomain_boundary ),
                    send_recv_pair.neighbor_subdomain,
                    static_cast< grid::BoundaryFace >( send_recv_pair.neighbor_subdomain_boundary ) );

                if ( !domain.subdomains().contains( send_recv_pair.neighbor_subdomain ) )
                {
                    Kokkos::abort( "Subdomain not found locally - but it should be there..." );
                }

                const auto local_subdomain_id_of_neighboring_subdomain =
                    std::get< 0 >( domain.subdomains().at( send_recv_pair.neighbor_subdomain ) );

                copy_to_buffer(
                    recv_buffer,
                    data,
                    local_subdomain_id_of_neighboring_subdomain,
                    static_cast< grid::BoundaryFace >( send_recv_pair.neighbor_subdomain_boundary ) );
                continue;
            }

            const auto& send_buffer = boundary_send_buffers.buffer_face(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryFace >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryFace >( send_recv_pair.neighbor_subdomain_boundary ) );

            send_buffer_ptr  = send_buffer.data();
            send_buffer_size = send_buffer.span();

            copy_to_buffer( send_buffer, data, local_subdomain_id, local_face_boundary );
        }
        else
        {
            Kokkos::abort( "Unknown boundary type" );
        }

        Kokkos::fence( "deep_copy_into_send_buffer" );

        // Schedule Isend (non-local comm).
        MPI_Request data_send_request;
        MPI_Isend(
            send_buffer_ptr,
            send_buffer_size,
            mpi::mpi_datatype< ScalarType >(),
            send_recv_pair.neighbor_rank,
            MPI_TAG_BOUNDARY_DATA,
            domain.comm(),
            &data_send_request );
        data_send_requests.push_back( data_send_request );
    }

    /////////////////////////////////////
    // Wait for all sends to complete. //
    /////////////////////////////////////

    MPI_Waitall( data_send_requests.size(), data_send_requests.data(), MPI_STATUSES_IGNORE );
    MPI_Waitall( data_recv_requests.size(), data_recv_requests.data(), MPI_STATUSES_IGNORE );
}

/// @brief Unpacks and reduces local subdomain boundaries.
///
/// The recv buffers must be the same instances as used during sending in `pack_send_and_recv_local_subdomain_boundaries()`.
///
/// See `pack_send_and_recv_local_subdomain_boundaries()` for more details on how the communication works.
///
/// @param domain the DistributedDomain that this works on
/// @param data the data (Kokkos::View) to be communicated
/// @param boundary_recv_buffers SubdomainNeighborhoodSendRecvBuffer instance that serves for receiving data - must be
///                              the same that was previously populated by `pack_send_and_recv_local_subdomain_boundaries()`
/// @param reduction reduction mode
template < typename GridDataType >
void unpack_and_reduce_local_subdomain_boundaries(
    const grid::shell::DistributedDomain& domain,
    const GridDataType&                   data,
    SubdomainNeighborhoodSendRecvBuffer< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() >&
                           boundary_recv_buffers,
    CommunicationReduction reduction = CommunicationReduction::SUM )
{
    // Since it is not clear whether a static last dimension of 1 impacts performance, we want to support both
    // scalar and vector-valued grid data views. To simplify matters, we always use the vector-valued versions for the
    // buffers.

    static_assert(
        std::is_same_v< GridDataType, grid::Grid4DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v<
            GridDataType,
            grid::Grid4DDataVec< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() > > );

    for ( const auto& [local_subdomain_info, idx_and_neighborhood] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = idx_and_neighborhood;

        for ( const auto& [local_vertex_boundary, neighbors] : neighborhood.neighborhood_vertex() )
        {
            // Multiple neighbor subdomains per vertex.
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

                auto recv_buffer = boundary_recv_buffers.buffer_vertex(
                    local_subdomain_info, local_vertex_boundary, neighbor_subdomain_info, neighbor_local_boundary );

                copy_from_buffer_rotate_and_reduce(
                    recv_buffer, data, local_subdomain_id, local_vertex_boundary, reduction );
            }
        }

        for ( const auto& [local_edge_boundary, neighbors] : neighborhood.neighborhood_edge() )
        {
            // Multiple neighbor subdomains per edge.
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, boundary_direction, neighbor_rank] =
                    neighbor;

                auto recv_buffer = boundary_recv_buffers.buffer_edge(
                    local_subdomain_info, local_edge_boundary, neighbor_subdomain_info, neighbor_local_boundary );

                copy_from_buffer_rotate_and_reduce(
                    recv_buffer, data, local_subdomain_id, local_edge_boundary, boundary_direction, reduction );
            }
        }

        for ( const auto& [local_face_boundary, neighbor] : neighborhood.neighborhood_face() )
        {
            // Single neighbor subdomain per facet.
            const auto& [neighbor_subdomain_info, neighbor_local_boundary, boundary_directions, neighbor_rank] =
                neighbor;

            auto recv_buffer = boundary_recv_buffers.buffer_face(
                local_subdomain_info, local_face_boundary, neighbor_subdomain_info, neighbor_local_boundary );

            copy_from_buffer_rotate_and_reduce(
                recv_buffer, data, local_subdomain_id, local_face_boundary, boundary_directions, reduction );
        }
    }

    Kokkos::fence();
}

/// @brief Executes packing, sending, receiving, and unpacking operations for the shell.
///
/// @note THIS MAY COME WITH A PERFORMANCE PENALTY.
///       This function (re-)allocates send and receive buffers for each call, which could be inefficient.
///       Use only where performance does not matter (e.g. in tests).
///       Better: reuse the buffers for subsequent send-recv calls through overloads of this function.
///
/// Essentially just calls `pack_send_and_recv_local_subdomain_boundaries()` and `unpack_and_reduce_local_subdomain_boundaries()`.
template < typename ScalarType >
void send_recv(
    const grid::shell::DistributedDomain& domain,
    grid::Grid4DDataScalar< ScalarType >& grid,
    const CommunicationReduction          reduction = CommunicationReduction::SUM )
{
    SubdomainNeighborhoodSendRecvBuffer< ScalarType > send_buffers( domain );
    SubdomainNeighborhoodSendRecvBuffer< ScalarType > recv_buffers( domain );

    shell::pack_send_and_recv_local_subdomain_boundaries( domain, grid, send_buffers, recv_buffers );
    shell::unpack_and_reduce_local_subdomain_boundaries( domain, grid, recv_buffers, reduction );
}

/// @brief Executes packing, sending, receiving, and unpacking operations for the shell.
///
/// Send and receive buffers must be passed. This is the preferred way to execute communication since the buffers
/// can be reused.
///
/// Essentially just calls `pack_send_and_recv_local_subdomain_boundaries()` and `unpack_and_reduce_local_subdomain_boundaries()`.
template < typename ScalarType >
void send_recv(
    const grid::shell::DistributedDomain&              domain,
    grid::Grid4DDataScalar< ScalarType >&              grid,
    SubdomainNeighborhoodSendRecvBuffer< ScalarType >& send_buffers,
    SubdomainNeighborhoodSendRecvBuffer< ScalarType >& recv_buffers,
    const CommunicationReduction                       reduction = CommunicationReduction::SUM )
{
    shell::pack_send_and_recv_local_subdomain_boundaries( domain, grid, send_buffers, recv_buffers );
    shell::unpack_and_reduce_local_subdomain_boundaries( domain, grid, recv_buffers, reduction );
}

} // namespace terra::communication::shell