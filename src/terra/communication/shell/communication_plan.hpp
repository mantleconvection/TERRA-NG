#pragma once

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

#include "dense/vec.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "terra/communication/buffer_copy_kernels.hpp"
#include "terra/communication/shell/communication.hpp"
#include "util/timer.hpp"

namespace terra::communication::shell {



namespace detail {

// Device-side description for one packed boundary chunk.
struct DevicePackChunk
{
    int                    boundary_type;
    int                    local_subdomain_id;
    int                    offset;
    int                    n0;
    int                    n1;
    grid::BoundaryPosition boundary_position_x;
    grid::BoundaryPosition boundary_position_y;
    grid::BoundaryPosition boundary_position_r;
};

// Device-side description for one unpack/reduction boundary chunk.
struct DeviceUnpackChunk
{
    int                     boundary_type;
    int                     local_subdomain_id;
    int                     offset;
    int                     n0;
    int                     n1;
    grid::BoundaryPosition  boundary_position_x;
    grid::BoundaryPosition  boundary_position_y;
    grid::BoundaryPosition  boundary_position_r;
    grid::BoundaryDirection direction_0;
    grid::BoundaryDirection direction_1;
};

template < int VecDim, typename GridDataType, typename FlatBufferView, typename DeviceChunkView >
void launch_pack_chunks_batched(
    const char*         kernel_name,
    const GridDataType& data,
    FlatBufferView      flat_buffer,
    DeviceChunkView     chunks )
{
    using ScalarType      = typename GridDataType::value_type;
    using execution_space = typename FlatBufferView::execution_space;
    using team_policy     = Kokkos::TeamPolicy< execution_space >;
    using member_type     = typename team_policy::member_type;

    constexpr bool is_scalar = ( GridDataType::rank == 4 );

    const int size_x = static_cast< int >( data.extent( 1 ) );
    const int size_y = static_cast< int >( data.extent( 2 ) );
    const int size_r = static_cast< int >( data.extent( 3 ) );

    Kokkos::parallel_for(
        kernel_name,
        team_policy( static_cast< int >( chunks.extent( 0 ) ), Kokkos::AUTO ),
        KOKKOS_LAMBDA( const member_type& team ) {
            const DevicePackChunk ch       = chunks( team.league_rank() );
            ScalarType* const     base_ptr = flat_buffer.data() + ch.offset;

            // A manual strided team-thread loop avoids an extra nested lambda
            // and keeps the whole batch in this single TeamPolicy kernel.
            const int team_rank = team.team_rank();
            const int team_size = team.team_size();

            if ( ch.boundary_type == 0 )
            {
                using BufT = grid::Grid0DDataVec< ScalarType, VecDim >;
                auto buffer = make_unmanaged_like< BufT >( base_ptr );

                for ( int d = team_rank; d < VecDim; d += team_size )
                {
                    const int x = terra::communication::detail::idx(
                        0, size_x, ch.boundary_position_x, grid::BoundaryDirection::FORWARD );
                    const int y = terra::communication::detail::idx(
                        0, size_y, ch.boundary_position_y, grid::BoundaryDirection::FORWARD );
                    const int r = terra::communication::detail::idx(
                        0, size_r, ch.boundary_position_r, grid::BoundaryDirection::FORWARD );

                    buffer( d ) = terra::communication::detail::value< GridDataType, is_scalar >(
                        data, ch.local_subdomain_id, x, y, r, d );
                }
            }
            else if ( ch.boundary_type == 1 )
            {
                using BufT = grid::Grid1DDataVec< ScalarType, VecDim >;
                auto buffer = make_unmanaged_like< BufT >( base_ptr, ch.n0 );

                const int total = ch.n0 * VecDim;
                for ( int linear = team_rank; linear < total; linear += team_size )
                {
                    const int idx = linear / VecDim;
                    const int d   = linear % VecDim;

                    const int x = terra::communication::detail::idx(
                        idx, size_x, ch.boundary_position_x, grid::BoundaryDirection::FORWARD );
                    const int y = terra::communication::detail::idx(
                        idx, size_y, ch.boundary_position_y, grid::BoundaryDirection::FORWARD );
                    const int r = terra::communication::detail::idx(
                        idx, size_r, ch.boundary_position_r, grid::BoundaryDirection::FORWARD );

                    buffer( idx, d ) = terra::communication::detail::value< GridDataType, is_scalar >(
                        data, ch.local_subdomain_id, x, y, r, d );
                }
            }
            else if ( ch.boundary_type == 2 )
            {
                using BufT = grid::Grid2DDataVec< ScalarType, VecDim >;
                auto buffer = make_unmanaged_like< BufT >( base_ptr, ch.n0, ch.n1 );

                const int total = ch.n0 * ch.n1 * VecDim;
                for ( int linear = team_rank; linear < total; linear += team_size )
                {
                    const int i   = linear / ( ch.n1 * VecDim );
                    const int rem = linear % ( ch.n1 * VecDim );
                    const int j   = rem / VecDim;
                    const int d   = rem % VecDim;

                    int x = 0;
                    int y = 0;
                    int r = 0;

                    if ( ch.boundary_position_x != grid::BoundaryPosition::PV )
                    {
                        x = terra::communication::detail::idx(
                            0, size_x, ch.boundary_position_x, grid::BoundaryDirection::FORWARD );
                        y = terra::communication::detail::idx(
                            i, size_y, ch.boundary_position_y, grid::BoundaryDirection::FORWARD );
                        r = terra::communication::detail::idx(
                            j, size_r, ch.boundary_position_r, grid::BoundaryDirection::FORWARD );
                    }
                    else if ( ch.boundary_position_y != grid::BoundaryPosition::PV )
                    {
                        x = terra::communication::detail::idx(
                            i, size_x, ch.boundary_position_x, grid::BoundaryDirection::FORWARD );
                        y = terra::communication::detail::idx(
                            0, size_y, ch.boundary_position_y, grid::BoundaryDirection::FORWARD );
                        r = terra::communication::detail::idx(
                            j, size_r, ch.boundary_position_r, grid::BoundaryDirection::FORWARD );
                    }
                    else
                    {
                        x = terra::communication::detail::idx(
                            i, size_x, ch.boundary_position_x, grid::BoundaryDirection::FORWARD );
                        y = terra::communication::detail::idx(
                            j, size_y, ch.boundary_position_y, grid::BoundaryDirection::FORWARD );
                        r = terra::communication::detail::idx(
                            0, size_r, ch.boundary_position_r, grid::BoundaryDirection::FORWARD );
                    }

                    buffer( i, j, d ) = terra::communication::detail::value< GridDataType, is_scalar >(
                        data, ch.local_subdomain_id, x, y, r, d );
                }
            }
        } );
}

template < int VecDim, typename GridDataType, typename FlatBufferView, typename DeviceChunkView >
void launch_unpack_chunks_batched(
    const char*             kernel_name,
    const GridDataType&     data,
    FlatBufferView          flat_buffer,
    DeviceChunkView         chunks,
    CommunicationReduction  reduction )
{
    using ScalarType      = typename GridDataType::value_type;
    using execution_space = typename FlatBufferView::execution_space;
    using team_policy     = Kokkos::TeamPolicy< execution_space >;
    using member_type     = typename team_policy::member_type;

    constexpr bool is_scalar = ( GridDataType::rank == 4 );

    const int size_x = static_cast< int >( data.extent( 1 ) );
    const int size_y = static_cast< int >( data.extent( 2 ) );
    const int size_r = static_cast< int >( data.extent( 3 ) );

    Kokkos::parallel_for(
        kernel_name,
        team_policy( static_cast< int >( chunks.extent( 0 ) ), Kokkos::AUTO ),
        KOKKOS_LAMBDA( const member_type& team ) {
            const DeviceUnpackChunk ch     = chunks( team.league_rank() );
            ScalarType* const       base_ptr = flat_buffer.data() + ch.offset;

            const int team_rank = team.team_rank();
            const int team_size = team.team_size();

            if ( ch.boundary_type == 0 )
            {
                using BufT = grid::Grid0DDataVec< ScalarType, VecDim >;
                auto buffer = make_unmanaged_like< BufT >( base_ptr );

                for ( int d = team_rank; d < VecDim; d += team_size )
                {
                    const int x = terra::communication::detail::idx(
                        0, size_x, ch.boundary_position_x, grid::BoundaryDirection::FORWARD );
                    const int y = terra::communication::detail::idx(
                        0, size_y, ch.boundary_position_y, grid::BoundaryDirection::FORWARD );
                    const int r = terra::communication::detail::idx(
                        0, size_r, ch.boundary_position_r, grid::BoundaryDirection::FORWARD );

                    terra::communication::detail::reduction_function(
                        &terra::communication::detail::value_ref< GridDataType, is_scalar >(
                            data, ch.local_subdomain_id, x, y, r, d ),
                        buffer( d ),
                        reduction );
                }
            }
            else if ( ch.boundary_type == 1 )
            {
                using BufT = grid::Grid1DDataVec< ScalarType, VecDim >;
                auto buffer = make_unmanaged_like< BufT >( base_ptr, ch.n0 );

                const int total = ch.n0 * VecDim;
                for ( int linear = team_rank; linear < total; linear += team_size )
                {
                    const int idx = linear / VecDim;
                    const int d   = linear % VecDim;

                    const int x = terra::communication::detail::idx(
                        idx, size_x, ch.boundary_position_x, ch.direction_0 );
                    const int y = terra::communication::detail::idx(
                        idx, size_y, ch.boundary_position_y, ch.direction_0 );
                    const int r = terra::communication::detail::idx(
                        idx, size_r, ch.boundary_position_r, ch.direction_0 );

                    terra::communication::detail::reduction_function(
                        &terra::communication::detail::value_ref< GridDataType, is_scalar >(
                            data, ch.local_subdomain_id, x, y, r, d ),
                        buffer( idx, d ),
                        reduction );
                }
            }
            else if ( ch.boundary_type == 2 )
            {
                using BufT = grid::Grid2DDataVec< ScalarType, VecDim >;
                auto buffer = make_unmanaged_like< BufT >( base_ptr, ch.n0, ch.n1 );

                const int total = ch.n0 * ch.n1 * VecDim;
                for ( int linear = team_rank; linear < total; linear += team_size )
                {
                    const int i   = linear / ( ch.n1 * VecDim );
                    const int rem = linear % ( ch.n1 * VecDim );
                    const int j   = rem / VecDim;
                    const int d   = rem % VecDim;

                    int x = 0;
                    int y = 0;
                    int r = 0;

                    if ( ch.boundary_position_x != grid::BoundaryPosition::PV )
                    {
                        x = terra::communication::detail::idx(
                            0, size_x, ch.boundary_position_x, grid::BoundaryDirection::FORWARD );
                        y = terra::communication::detail::idx(
                            i, size_y, ch.boundary_position_y, ch.direction_0 );
                        r = terra::communication::detail::idx(
                            j, size_r, ch.boundary_position_r, ch.direction_1 );
                    }
                    else if ( ch.boundary_position_y != grid::BoundaryPosition::PV )
                    {
                        x = terra::communication::detail::idx(
                            i, size_x, ch.boundary_position_x, ch.direction_0 );
                        y = terra::communication::detail::idx(
                            0, size_y, ch.boundary_position_y, grid::BoundaryDirection::FORWARD );
                        r = terra::communication::detail::idx(
                            j, size_r, ch.boundary_position_r, ch.direction_1 );
                    }
                    else
                    {
                        x = terra::communication::detail::idx(
                            i, size_x, ch.boundary_position_x, ch.direction_0 );
                        y = terra::communication::detail::idx(
                            j, size_y, ch.boundary_position_y, ch.direction_1 );
                        r = terra::communication::detail::idx(
                            0, size_r, ch.boundary_position_r, grid::BoundaryDirection::FORWARD );
                    }

                    terra::communication::detail::reduction_function(
                        &terra::communication::detail::value_ref< GridDataType, is_scalar >(
                            data, ch.local_subdomain_id, x, y, r, d ),
                        buffer( i, j, d ),
                        reduction );
                }
            }
        } );
}

} // namespace detail

// --------------------------------------------------------------------------------------
// Reusable, precomputed plan
// --------------------------------------------------------------------------------------
//
// Goal: avoid rebuilding send/recv pair lists, sorting, chunk layout, and per-rank buffer sizes
// on every halo exchange. We do that once in the ctor, then `exchange_and_reduce(...)` just runs
// the hot path: local copies, pack, fence, post isends/irecvs, wait, scatter, unpack.
//
// Notes:
// - Remote communication keeps the existing per-rank aggregation optimization.
// - Local communication uses one reusable flat snapshot buffer so packing and
//   unpack/reduction can each be batched into one GPU kernel.
// - The boundary_recv_buffers argument is kept for API compatibility but is no
//   longer used by this fully batched plan.
// - It does not depend on send_buffers_ (your argument is unused currently anyway).
//
template < class GridDataType >
class ShellBoundaryCommPlan
{
  public:
    using ScalarType            = typename GridDataType::value_type;
    static constexpr int VecDim = grid::grid_data_vec_dim< GridDataType >();
    using memory_space          = typename GridDataType::memory_space;
    using rank_buffer_view      = Kokkos::View< ScalarType*, memory_space >;

    explicit ShellBoundaryCommPlan( const grid::shell::DistributedDomain& domain, bool enable_local_comm = true )
        : domain_( &domain ), enable_local_comm_( enable_local_comm )
    {
        build_plan_();
        allocate_rank_buffers_();
    }

    // Call this each timestep/iteration.
    void exchange_and_reduce(
        const GridDataType& data,
        SubdomainNeighborhoodSendRecvBuffer< ScalarType, VecDim >& boundary_recv_buffers,
        CommunicationReduction reduction = CommunicationReduction::SUM ) const
    {
        util::Timer timer_all( "shell_boundary_exchange_and_reduce" );

        // Kept in the public API for compatibility.  The fully batched plan
        // now uses its own flat local scratch buffer and the per-rank buffers.
        (void)boundary_recv_buffers;

        post_irecvs_();

        // Local and remote packing both read the original grid state.  Keep
        // those reads in the packing phase before any reduction modifies data.
        pack_local_( data );
        pack_remote_sends_( data );

        post_isends_();

        // Reduce the snapshotted local chunks while MPI progresses.
        unpack_local_( data, reduction );

        // Waitany loop: for each remote recv as it lands, unpack its chunks
        // directly from the flat per-rank recv buffer. Sends are drained at end.
        wait_and_unpack_remote_( data, reduction );

        Kokkos::fence();
    }

    // Optional: if domain topology changes (rare), rebuild everything.
    void rebuild()
    {
        build_plan_();
        allocate_rank_buffers_();
    }

  private:
    struct SendRecvPair
    {
        int                        boundary_type = -1; // 0 vertex, 1 edge, 2 face
        mpi::MPIRank               local_rank;
        grid::shell::SubdomainInfo local_subdomain;
        int                        local_subdomain_boundary;
        int                        local_subdomain_id;

        mpi::MPIRank               neighbor_rank;
        grid::shell::SubdomainInfo neighbor_subdomain;
        int                        neighbor_subdomain_boundary;

        // Orientation for rotate step in unpack. direction_0 is used for edges
        // and as the first component for faces; direction_1 only for faces.
        grid::BoundaryDirection    direction_0 = grid::BoundaryDirection::FORWARD;
        grid::BoundaryDirection    direction_1 = grid::BoundaryDirection::FORWARD;
    };

    struct ChunkInfo
    {
        SendRecvPair pair;
        int         offset = 0; // in scalars
        int         size   = 0; // in scalars
    };

    using device_pack_chunk_view   = Kokkos::View< detail::DevicePackChunk*, memory_space >;
    using device_unpack_chunk_view = Kokkos::View< detail::DeviceUnpackChunk*, memory_space >;

    // --------------------------
    // Plan build / layout
    // --------------------------
    int piece_num_scalars_( const SendRecvPair& p ) const
    {
        const auto& domain = *domain_;

        if ( p.boundary_type == 0 )
        {
            return VecDim;
        }
        else if ( p.boundary_type == 1 )
        {
            const auto local_edge_boundary = static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );
            const int  n_nodes             = grid::is_edge_boundary_radial( local_edge_boundary ) ?
                                                 domain.domain_info().subdomain_num_nodes_radially() :
                                                 domain.domain_info().subdomain_num_nodes_per_side_laterally();
            return n_nodes * VecDim;
        }
        else if ( p.boundary_type == 2 )
        {
            const auto local_face_boundary = static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );
            const int  ni                  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
            const int  nj                  = grid::is_face_boundary_normal_to_radial_direction( local_face_boundary ) ?
                                                 domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                                 domain.domain_info().subdomain_num_nodes_radially();
            return ni * nj * VecDim;
        }
        Kokkos::abort( "Unknown boundary type" );
        return 0;
    }

    void build_plan_()
    {
        util::Timer timer( "ShellBoundaryCommPlan::build_plan" );

        const auto& domain = *domain_;

        send_recv_pairs_.clear();
        send_recv_pairs_.reserve( 1024 );

        // Build the full (unsorted) pair list once.
        for ( const auto& [local_subdomain_info, idx_and_neighborhood] : domain.subdomains() )
        {
            const auto& [local_subdomain_id, neighborhood] = idx_and_neighborhood;

            for ( const auto& [local_vertex_boundary, neighbors] : neighborhood.neighborhood_vertex() )
            {
                for ( const auto& neighbor : neighbors )
                {
                    const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;
                    send_recv_pairs_.push_back( SendRecvPair{
                        .boundary_type               = 0,
                        .local_rank                  = mpi::rank( domain.comm() ),
                        .local_subdomain             = local_subdomain_info,
                        .local_subdomain_boundary    = static_cast< int >( local_vertex_boundary ),
                        .local_subdomain_id          = local_subdomain_id,
                        .neighbor_rank               = neighbor_rank,
                        .neighbor_subdomain          = neighbor_subdomain_info,
                        .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ) } );
                }
            }

            for ( const auto& [local_edge_boundary, neighbors] : neighborhood.neighborhood_edge() )
            {
                for ( const auto& neighbor : neighbors )
                {
                    const auto& [neighbor_subdomain_info, neighbor_local_boundary, edge_direction, neighbor_rank] =
                        neighbor;
                    send_recv_pairs_.push_back( SendRecvPair{
                        .boundary_type               = 1,
                        .local_rank                  = mpi::rank( domain.comm() ),
                        .local_subdomain             = local_subdomain_info,
                        .local_subdomain_boundary    = static_cast< int >( local_edge_boundary ),
                        .local_subdomain_id          = local_subdomain_id,
                        .neighbor_rank               = neighbor_rank,
                        .neighbor_subdomain          = neighbor_subdomain_info,
                        .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ),
                        .direction_0                 = edge_direction } );
                }
            }

            for ( const auto& [local_face_boundary, neighbor] : neighborhood.neighborhood_face() )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, face_directions, neighbor_rank] =
                    neighbor;
                send_recv_pairs_.push_back( SendRecvPair{
                    .boundary_type               = 2,
                    .local_rank                  = mpi::rank( domain.comm() ),
                    .local_subdomain             = local_subdomain_info,
                    .local_subdomain_boundary    = static_cast< int >( local_face_boundary ),
                    .local_subdomain_id          = local_subdomain_id,
                    .neighbor_rank               = neighbor_rank,
                    .neighbor_subdomain          = neighbor_subdomain_info,
                    .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ),
                    .direction_0                 = std::get< 0 >( face_directions ),
                    .direction_1                 = std::get< 1 >( face_directions ) } );
            }
        }

        // Precompute local-comm subset (fixed list).
        local_pairs_.clear();
        local_pairs_.reserve( send_recv_pairs_.size() );
        for ( const auto& p : send_recv_pairs_ )
        {
            if ( enable_local_comm_ && p.local_rank == p.neighbor_rank )
                local_pairs_.push_back( p );
        }

        // LOCAL layout.  Keep a flat snapshot buffer so local packing and
        // local unpack/reduction can each be executed by one batched kernel
        // while preserving the original two-phase semantics.
        local_chunks_.clear();
        local_total_ = 0;
        for ( const auto& p : local_pairs_ )
        {
            const int sz  = piece_num_scalars_( p );
            const int off = local_total_;
            local_total_ += sz;
            local_chunks_.push_back( ChunkInfo{ .pair = p, .offset = off, .size = sz } );
        }

        // SEND layout (sorted and chunked per rank, remote only)
        {
            auto send_pairs = send_recv_pairs_;
            std::sort( send_pairs.begin(), send_pairs.end(), []( const SendRecvPair& a, const SendRecvPair& b ) {
                if ( a.boundary_type != b.boundary_type ) return a.boundary_type < b.boundary_type;
                if ( a.local_subdomain != b.local_subdomain ) return a.local_subdomain < b.local_subdomain;
                if ( a.local_subdomain_boundary != b.local_subdomain_boundary )
                    return a.local_subdomain_boundary < b.local_subdomain_boundary;
                if ( a.neighbor_subdomain != b.neighbor_subdomain ) return a.neighbor_subdomain < b.neighbor_subdomain;
                return a.neighbor_subdomain_boundary < b.neighbor_subdomain_boundary;
            } );

            send_chunks_by_rank_.clear();
            send_total_by_rank_.clear();

            for ( const auto& p : send_pairs )
            {
                if ( enable_local_comm_ && p.local_rank == p.neighbor_rank )
                    continue;

                const int sz = piece_num_scalars_( p );
                auto&     chunks = send_chunks_by_rank_[p.neighbor_rank];

                const int off = send_total_by_rank_[p.neighbor_rank];
                send_total_by_rank_[p.neighbor_rank] += sz;

                chunks.push_back( ChunkInfo{ .pair = p, .offset = off, .size = sz } );
            }
        }

        // RECV layout (sorted and chunked per rank, remote only)
        {
            auto recv_pairs = send_recv_pairs_;
            std::sort( recv_pairs.begin(), recv_pairs.end(), []( const SendRecvPair& a, const SendRecvPair& b ) {
                if ( a.boundary_type != b.boundary_type ) return a.boundary_type < b.boundary_type;
                if ( a.neighbor_subdomain != b.neighbor_subdomain ) return a.neighbor_subdomain < b.neighbor_subdomain;
                if ( a.neighbor_subdomain_boundary != b.neighbor_subdomain_boundary )
                    return a.neighbor_subdomain_boundary < b.neighbor_subdomain_boundary;
                if ( a.local_subdomain != b.local_subdomain ) return a.local_subdomain < b.local_subdomain;
                return a.local_subdomain_boundary < b.local_subdomain_boundary;
            } );

            recv_chunks_by_rank_.clear();
            recv_total_by_rank_.clear();

            for ( const auto& p : recv_pairs )
            {
                if ( enable_local_comm_ && p.local_rank == p.neighbor_rank )
                    continue;

                const int sz = piece_num_scalars_( p );
                auto&     chunks = recv_chunks_by_rank_[p.neighbor_rank];

                const int off = recv_total_by_rank_[p.neighbor_rank];
                recv_total_by_rank_[p.neighbor_rank] += sz;

                chunks.push_back( ChunkInfo{ .pair = p, .offset = off, .size = sz } );
            }
        }
    }

    void allocate_rank_buffers_()
    {
        util::Timer timer( "ShellBoundaryCommPlan::allocate_rank_buffers" );

        send_rank_buffers_.clear();
        recv_rank_buffers_.clear();

        for ( const auto& [rank, total] : send_total_by_rank_ )
        {
            if ( total > 0 )
                send_rank_buffers_[rank] = rank_buffer_view( "rank_send_buffer", total );
        }
        for ( const auto& [rank, total] : recv_total_by_rank_ )
        {
            if ( total > 0 )
                recv_rank_buffers_[rank] = rank_buffer_view( "rank_recv_buffer", total );
        }

        if ( local_total_ > 0 )
            local_buffer_ = rank_buffer_view( "local_comm_buffer", local_total_ );
        else
            local_buffer_ = rank_buffer_view();

        data_send_requests_.resize( send_rank_buffers_.size() );
        data_recv_requests_.resize( recv_rank_buffers_.size() );
        recv_request_ranks_.resize( recv_rank_buffers_.size() );

        build_device_chunks_();
    }

    void build_device_chunks_()
    {
        send_pack_chunks_device_.clear();
        recv_unpack_chunks_device_.clear();
        local_pack_chunks_device_   = device_pack_chunk_view();
        local_unpack_chunks_device_ = device_unpack_chunk_view();

        const auto& domain = *domain_;
        const int   nlat   = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        const int   nrad   = domain.domain_info().subdomain_num_nodes_radially();

        // Remote send packing metadata.
        for ( const auto& [rank, chunks] : send_chunks_by_rank_ )
        {
            device_pack_chunk_view device_chunks( "send_pack_chunks", chunks.size() );
            auto host_chunks = Kokkos::create_mirror_view( device_chunks );

            for ( std::size_t i = 0; i < chunks.size(); ++i )
            {
                const auto& ch = chunks[i];
                const auto& p  = ch.pair;

                detail::DevicePackChunk dc{};
                dc.boundary_type      = p.boundary_type;
                dc.local_subdomain_id = p.local_subdomain_id;
                dc.offset             = ch.offset;

                if ( p.boundary_type == 0 )
                {
                    const auto boundary = static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary );
                    dc.n0 = VecDim;
                    dc.n1 = 1;
                    dc.boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary );
                    dc.boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary );
                    dc.boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary );
                }
                else if ( p.boundary_type == 1 )
                {
                    const auto boundary = static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );
                    dc.n0 = grid::is_edge_boundary_radial( boundary ) ? nrad : nlat;
                    dc.n1 = 1;
                    dc.boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary );
                    dc.boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary );
                    dc.boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary );
                }
                else if ( p.boundary_type == 2 )
                {
                    const auto boundary = static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );
                    dc.n0 = nlat;
                    dc.n1 = grid::is_face_boundary_normal_to_radial_direction( boundary ) ? nlat : nrad;
                    dc.boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary );
                    dc.boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary );
                    dc.boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary );
                }
                else
                {
                    Kokkos::abort( "Unknown boundary type" );
                }

                host_chunks( i ) = dc;
            }

            Kokkos::deep_copy( device_chunks, host_chunks );
            send_pack_chunks_device_[rank] = device_chunks;
        }

        // Remote receive/unpack metadata.
        for ( const auto& [rank, chunks] : recv_chunks_by_rank_ )
        {
            device_unpack_chunk_view device_chunks( "recv_unpack_chunks", chunks.size() );
            auto host_chunks = Kokkos::create_mirror_view( device_chunks );

            for ( std::size_t i = 0; i < chunks.size(); ++i )
            {
                const auto& ch = chunks[i];
                const auto& p  = ch.pair;

                detail::DeviceUnpackChunk dc{};
                dc.boundary_type      = p.boundary_type;
                dc.local_subdomain_id = p.local_subdomain_id;
                dc.offset             = ch.offset;
                dc.direction_0        = p.direction_0;
                dc.direction_1        = p.direction_1;

                if ( p.boundary_type == 0 )
                {
                    const auto boundary = static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary );
                    dc.n0 = VecDim;
                    dc.n1 = 1;
                    dc.boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary );
                    dc.boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary );
                    dc.boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary );
                }
                else if ( p.boundary_type == 1 )
                {
                    const auto boundary = static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );
                    dc.n0 = grid::is_edge_boundary_radial( boundary ) ? nrad : nlat;
                    dc.n1 = 1;
                    dc.boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary );
                    dc.boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary );
                    dc.boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary );
                }
                else if ( p.boundary_type == 2 )
                {
                    const auto boundary = static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );
                    dc.n0 = nlat;
                    dc.n1 = grid::is_face_boundary_normal_to_radial_direction( boundary ) ? nlat : nrad;
                    dc.boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary );
                    dc.boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary );
                    dc.boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary );
                }
                else
                {
                    Kokkos::abort( "Unknown boundary type" );
                }

                host_chunks( i ) = dc;
            }

            Kokkos::deep_copy( device_chunks, host_chunks );
            recv_unpack_chunks_device_[rank] = device_chunks;
        }

        // Local communication metadata.  Packing reads the neighboring
        // subdomain/boundary into a flat snapshot; unpacking later reduces
        // that snapshot into the local subdomain/boundary.
        if ( !local_chunks_.empty() )
        {
            local_pack_chunks_device_ =
                device_pack_chunk_view( "local_pack_chunks", local_chunks_.size() );
            local_unpack_chunks_device_ =
                device_unpack_chunk_view( "local_unpack_chunks", local_chunks_.size() );

            auto host_pack   = Kokkos::create_mirror_view( local_pack_chunks_device_ );
            auto host_unpack = Kokkos::create_mirror_view( local_unpack_chunks_device_ );

            for ( std::size_t i = 0; i < local_chunks_.size(); ++i )
            {
                const auto& ch = local_chunks_[i];
                const auto& p  = ch.pair;

                if ( !domain.subdomains().contains( p.neighbor_subdomain ) )
                    Kokkos::abort( "Subdomain not found locally - but it should be there..." );

                const int neighbor_subdomain_id =
                    std::get< 0 >( domain.subdomains().at( p.neighbor_subdomain ) );

                detail::DevicePackChunk   pc{};
                detail::DeviceUnpackChunk uc{};

                pc.boundary_type      = p.boundary_type;
                pc.local_subdomain_id = neighbor_subdomain_id;
                pc.offset             = ch.offset;

                uc.boundary_type      = p.boundary_type;
                uc.local_subdomain_id = p.local_subdomain_id;
                uc.offset             = ch.offset;
                uc.direction_0        = p.direction_0;
                uc.direction_1        = p.direction_1;

                if ( p.boundary_type == 0 )
                {
                    const auto src_boundary =
                        static_cast< grid::BoundaryVertex >( p.neighbor_subdomain_boundary );
                    const auto dst_boundary =
                        static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary );

                    pc.n0 = uc.n0 = VecDim;
                    pc.n1 = uc.n1 = 1;

                    pc.boundary_position_x = grid::boundary_position_from_boundary_type_x( src_boundary );
                    pc.boundary_position_y = grid::boundary_position_from_boundary_type_y( src_boundary );
                    pc.boundary_position_r = grid::boundary_position_from_boundary_type_r( src_boundary );

                    uc.boundary_position_x = grid::boundary_position_from_boundary_type_x( dst_boundary );
                    uc.boundary_position_y = grid::boundary_position_from_boundary_type_y( dst_boundary );
                    uc.boundary_position_r = grid::boundary_position_from_boundary_type_r( dst_boundary );
                }
                else if ( p.boundary_type == 1 )
                {
                    const auto src_boundary =
                        static_cast< grid::BoundaryEdge >( p.neighbor_subdomain_boundary );
                    const auto dst_boundary =
                        static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );

                    // The original local path iterates over the receive buffer,
                    // whose shape is determined by the local (destination)
                    // boundary, while using the neighbor boundary for source
                    // indexing.  Preserve that exactly here.
                    pc.n0 = uc.n0 =
                        grid::is_edge_boundary_radial( dst_boundary ) ? nrad : nlat;
                    pc.n1 = uc.n1 = 1;

                    pc.boundary_position_x = grid::boundary_position_from_boundary_type_x( src_boundary );
                    pc.boundary_position_y = grid::boundary_position_from_boundary_type_y( src_boundary );
                    pc.boundary_position_r = grid::boundary_position_from_boundary_type_r( src_boundary );

                    uc.boundary_position_x = grid::boundary_position_from_boundary_type_x( dst_boundary );
                    uc.boundary_position_y = grid::boundary_position_from_boundary_type_y( dst_boundary );
                    uc.boundary_position_r = grid::boundary_position_from_boundary_type_r( dst_boundary );
                }
                else if ( p.boundary_type == 2 )
                {
                    const auto src_boundary =
                        static_cast< grid::BoundaryFace >( p.neighbor_subdomain_boundary );
                    const auto dst_boundary =
                        static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );

                    pc.n0 = uc.n0 = nlat;
                    pc.n1 = uc.n1 =
                        grid::is_face_boundary_normal_to_radial_direction( dst_boundary ) ? nlat : nrad;

                    pc.boundary_position_x = grid::boundary_position_from_boundary_type_x( src_boundary );
                    pc.boundary_position_y = grid::boundary_position_from_boundary_type_y( src_boundary );
                    pc.boundary_position_r = grid::boundary_position_from_boundary_type_r( src_boundary );

                    uc.boundary_position_x = grid::boundary_position_from_boundary_type_x( dst_boundary );
                    uc.boundary_position_y = grid::boundary_position_from_boundary_type_y( dst_boundary );
                    uc.boundary_position_r = grid::boundary_position_from_boundary_type_r( dst_boundary );
                }
                else
                {
                    Kokkos::abort( "Unknown boundary type" );
                }

                host_pack( i )   = pc;
                host_unpack( i ) = uc;
            }

            Kokkos::deep_copy( local_pack_chunks_device_, host_pack );
            Kokkos::deep_copy( local_unpack_chunks_device_, host_unpack );
        }
    }

    // --------------------------
    // Hot path
    // --------------------------
    void post_irecvs_() const
    {
        util::Timer timer( "ShellBoundaryCommPlan::post_irecvs" );

        int i = 0;
        for ( const auto& [rank, buf] : recv_rank_buffers_ )
        {
            const int total_sz = static_cast< int >( buf.extent( 0 ) );
            MPI_Irecv(
                buf.data(),
                total_sz,
                mpi::mpi_datatype< ScalarType >(),
                rank,
                MPI_TAG_BOUNDARY_DATA,
                domain_->comm(),
                &data_recv_requests_[i] );
            recv_request_ranks_[i] = rank;
            ++i;
        }
        recv_req_count_ = i;
    }

    void pack_local_( const GridDataType& data ) const
    {
        util::Timer timer( "ShellBoundaryCommPlan::local_comm" );

        if ( local_total_ == 0 )
            return;

        detail::launch_pack_chunks_batched< VecDim >(
            "pack_local_batched",
            data,
            local_buffer_,
            local_pack_chunks_device_ );
    }

    void pack_remote_sends_( const GridDataType& data ) const
    {
        util::Timer timer( "ShellBoundaryCommPlan::pack_remote" );

        // Keep this member private.  The actual KOKKOS_LAMBDA lives in the
        // namespace-scope helper above, because NVCC forbids extended
        // host/device lambdas whose enclosing class member is private/protected.
        for ( const auto& [rank, rank_buf_stored] : send_rank_buffers_ )
        {
            auto rank_buf = rank_buf_stored;
            auto chunks   = send_pack_chunks_device_.at( rank );

            detail::launch_pack_chunks_batched< VecDim >(
                "pack_remote_rank_batched",
                data,
                rank_buf,
                chunks );
        }

        // MPI_Isend follows this call, so all rank buffers must be ready first.
        Kokkos::fence( "pack_rank_send_buffers" );
    }

    void post_isends_() const
    {
        util::Timer timer( "ShellBoundaryCommPlan::post_isends" );

        int i = 0;
        for ( const auto& [rank, buf] : send_rank_buffers_ )
        {
            const int total_sz = static_cast< int >( buf.extent( 0 ) );
            MPI_Isend(
                buf.data(),
                total_sz,
                mpi::mpi_datatype< ScalarType >(),
                rank,
                MPI_TAG_BOUNDARY_DATA,
                domain_->comm(),
                &data_send_requests_[i] );
            ++i;
        }
        send_req_count_ = i;
    }

    // Unpack/reduce the snapshotted local chunks in one kernel.  This remains a
    // separate phase from local packing so every source value is read before
    // any local reduction can modify the grid.
    void unpack_local_(
        const GridDataType& data,
        CommunicationReduction reduction ) const
    {
        util::Timer timer( "ShellBoundaryCommPlan::unpack_local" );

        if ( local_total_ == 0 )
            return;

        detail::launch_unpack_chunks_batched< VecDim >(
            "unpack_local_batched",
            data,
            local_buffer_,
            local_unpack_chunks_device_,
            reduction );
    }

    // Unpack all chunks received from one remote rank in one kernel.  This
    // preserves the existing MPI_Waitany overlap: each rank is dispatched as
    // soon as its aggregated receive completes.
    void unpack_remote_rank_(
        const GridDataType&    data,
        const mpi::MPIRank     rank,
        CommunicationReduction reduction ) const
    {
        auto rank_buf = recv_rank_buffers_.at( rank );
        auto chunks   = recv_unpack_chunks_device_.at( rank );

        detail::launch_unpack_chunks_batched< VecDim >(
            "unpack_remote_rank_batched",
            data,
            rank_buf,
            chunks,
            reduction );
    }

    // Wait for remote recvs, dispatching unpack_remote_rank_ as each message
    // lands. Finally waits on pending sends.
    //
    // Sub-timers:
    //   - mpi_waitany_first: time from start until the first recv completes.
    //   - mpi_waitany_rest : per-call time for subsequent recv completions
    //                        (count = (num_msgs - 1) * num_iters across runs).
    // Comparing their aggregates tells us whether waitall is dominated by the
    // initial handshake round-trip or by tail latency from stragglers.
    void wait_and_unpack_remote_(
        const GridDataType&    data,
        CommunicationReduction reduction ) const
    {
        util::Timer timer( "ShellBoundaryCommPlan::waitall" );

        for ( int completed = 0; completed < recv_req_count_; ++completed )
        {
            int idx = MPI_UNDEFINED;
            if ( completed == 0 )
            {
                util::Timer t( "ShellBoundaryCommPlan::mpi_waitany_first" );
                MPI_Waitany( recv_req_count_, data_recv_requests_.data(), &idx, MPI_STATUS_IGNORE );
            }
            else
            {
                util::Timer t( "ShellBoundaryCommPlan::mpi_waitany_rest" );
                MPI_Waitany( recv_req_count_, data_recv_requests_.data(), &idx, MPI_STATUS_IGNORE );
            }
            unpack_remote_rank_( data, recv_request_ranks_[idx], reduction );
        }

        if ( send_req_count_ > 0 )
        {
            util::Timer t( "ShellBoundaryCommPlan::mpi_waitall_sends" );
            MPI_Waitall( send_req_count_, data_send_requests_.data(), MPI_STATUSES_IGNORE );
        }
    }

  private:
    const grid::shell::DistributedDomain* domain_            = nullptr;
    bool                                  enable_local_comm_ = true;

    // Precomputed full list
    std::vector< SendRecvPair > send_recv_pairs_;

    // Precomputed local-only subset and its flat snapshot layout.
    std::vector< SendRecvPair > local_pairs_;
    std::vector< ChunkInfo >    local_chunks_;
    int                         local_total_ = 0;

    // Precomputed rank aggregation layouts
    std::map< mpi::MPIRank, std::vector< ChunkInfo > > send_chunks_by_rank_;
    std::map< mpi::MPIRank, std::vector< ChunkInfo > > recv_chunks_by_rank_;
    std::map< mpi::MPIRank, int >                      send_total_by_rank_;
    std::map< mpi::MPIRank, int >                      recv_total_by_rank_;

    // Device-side metadata for all four batched copy/reduction paths.
    std::map< mpi::MPIRank, device_pack_chunk_view >   send_pack_chunks_device_;
    std::map< mpi::MPIRank, device_unpack_chunk_view > recv_unpack_chunks_device_;
    device_pack_chunk_view                              local_pack_chunks_device_;
    device_unpack_chunk_view                            local_unpack_chunks_device_;

    // Reused flat buffers.  local_buffer_ snapshots same-rank boundary data
    // between the local pack and local unpack/reduction phases.
    rank_buffer_view                                      local_buffer_;
    mutable std::map< mpi::MPIRank, rank_buffer_view >   send_rank_buffers_;
    mutable std::map< mpi::MPIRank, rank_buffer_view >   recv_rank_buffers_;

    // Reused request storage
    mutable std::vector< MPI_Request >  data_send_requests_;
    mutable std::vector< MPI_Request >  data_recv_requests_;
    mutable std::vector< mpi::MPIRank > recv_request_ranks_;
    mutable int                         send_req_count_ = 0;
    mutable int                         recv_req_count_ = 0;
};

// --------------------------------------------------------------------------------------
// Unified one-call routine (plan is built once, then just executed each call)
// --------------------------------------------------------------------------------------
template < typename GridDataType >
void send_recv_with_plan(
    const ShellBoundaryCommPlan< GridDataType >& plan,
    const GridDataType&                         data,
    SubdomainNeighborhoodSendRecvBuffer< typename GridDataType::value_type,
                                         grid::grid_data_vec_dim< GridDataType >() >& recv_buffers,
    CommunicationReduction reduction = CommunicationReduction::SUM )
{
    plan.exchange_and_reduce( data, recv_buffers, reduction );
}

} // namespace terra::communication::shell
