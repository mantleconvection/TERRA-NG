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


// --------------------------------------------------------------------------------------
// Reusable, precomputed plan
// --------------------------------------------------------------------------------------
//
// Goal: avoid rebuilding send/recv pair lists, sorting, chunk layout, and per-rank buffer sizes
// on every halo exchange. We do that once in the ctor, then `exchange_and_reduce(...)` just runs
// the hot path: local copies, pack, fence, post isends/irecvs, wait, scatter, unpack.
//
// Notes:
// - This keeps your existing per-boundary recv buffers intact and still used by unpack.
// - It keeps your per-rank aggregation optimization.
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

        post_irecvs_();

        local_comm_copy_into_recv_buffers_( data, boundary_recv_buffers );

        pack_remote_sends_( data );

        post_isends_();

        // Unpack local pairs from boundary_recv_buffers while MPI progresses.
        unpack_local_( data, boundary_recv_buffers, reduction );

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

        data_send_requests_.resize( send_rank_buffers_.size() );
        data_recv_requests_.resize( recv_rank_buffers_.size() );
        recv_request_ranks_.resize( recv_rank_buffers_.size() );
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

    void local_comm_copy_into_recv_buffers_(
        const GridDataType& data,
        SubdomainNeighborhoodSendRecvBuffer< ScalarType, VecDim >& boundary_recv_buffers ) const
    {
        util::Timer timer( "ShellBoundaryCommPlan::local_comm" );

        const auto& domain = *domain_;

        for ( const auto& p : local_pairs_ )
        {
            if ( !domain.subdomains().contains( p.neighbor_subdomain ) )
                Kokkos::abort( "Subdomain not found locally - but it should be there..." );

            const auto neighbor_subdomain_id = std::get< 0 >( domain.subdomains().at( p.neighbor_subdomain ) );

            if ( p.boundary_type == 0 )
            {
                auto& recv_buf = boundary_recv_buffers.buffer_vertex(
                    p.local_subdomain,
                    static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary ),
                    p.neighbor_subdomain,
                    static_cast< grid::BoundaryVertex >( p.neighbor_subdomain_boundary ) );

                copy_to_buffer<VecDim>(
                    recv_buf,
                    data,
                    neighbor_subdomain_id,
                    static_cast< grid::BoundaryVertex >( p.neighbor_subdomain_boundary ) );
            }
            else if ( p.boundary_type == 1 )
            {
                auto& recv_buf = boundary_recv_buffers.buffer_edge(
                    p.local_subdomain,
                    static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary ),
                    p.neighbor_subdomain,
                    static_cast< grid::BoundaryEdge >( p.neighbor_subdomain_boundary ) );

                copy_to_buffer<VecDim>(
                    recv_buf,
                    data,
                    neighbor_subdomain_id,
                    static_cast< grid::BoundaryEdge >( p.neighbor_subdomain_boundary ) );
            }
            else if ( p.boundary_type == 2 )
            {
                auto& recv_buf = boundary_recv_buffers.buffer_face(
                    p.local_subdomain,
                    static_cast< grid::BoundaryFace >( p.local_subdomain_boundary ),
                    p.neighbor_subdomain,
                    static_cast< grid::BoundaryFace >( p.neighbor_subdomain_boundary ) );

                copy_to_buffer<VecDim>(
                    recv_buf,
                    data,
                    neighbor_subdomain_id,
                    static_cast< grid::BoundaryFace >( p.neighbor_subdomain_boundary ) );
            }
            else
            {
                Kokkos::abort( "Unknown boundary type" );
            }
        }
    }

    void pack_remote_sends_( const GridDataType& data ) const
    {
        util::Timer timer( "ShellBoundaryCommPlan::pack_remote" );

        const auto& domain = *domain_;

        for ( const auto& [rank, chunks] : send_chunks_by_rank_ )
        {
            auto& rank_buf = send_rank_buffers_.at( rank );

            for ( const auto& ch : chunks )
            {
                const auto& p = ch.pair;
                ScalarType* base_ptr = rank_buf.data() + ch.offset;

                if ( p.boundary_type == 0 )
                {
                    using BufT = grid::Grid0DDataVec< ScalarType, VecDim >;
                    auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr );

                    copy_to_buffer<VecDim>(
                        unmanaged,
                        data,
                        p.local_subdomain_id,
                        static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary ) );
                }
                else if ( p.boundary_type == 1 )
                {
                    using BufT = grid::Grid1DDataVec< ScalarType, VecDim >;
                    const auto local_edge_boundary = static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );
                    const int  n_nodes             = grid::is_edge_boundary_radial( local_edge_boundary ) ?
                                                         domain.domain_info().subdomain_num_nodes_radially() :
                                                         domain.domain_info().subdomain_num_nodes_per_side_laterally();

                    auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr, n_nodes );
                    copy_to_buffer<VecDim>( unmanaged, data, p.local_subdomain_id, local_edge_boundary );
                }
                else if ( p.boundary_type == 2 )
                {
                    using BufT = grid::Grid2DDataVec< ScalarType, VecDim >;
                    const auto local_face_boundary = static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );
                    const int  ni                  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
                    const int  nj                  = grid::is_face_boundary_normal_to_radial_direction( local_face_boundary ) ?
                                                         domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                                         domain.domain_info().subdomain_num_nodes_radially();

                    auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr, ni, nj );
                    copy_to_buffer<VecDim>( unmanaged, data, p.local_subdomain_id, local_face_boundary );
                }
                else
                {
                    Kokkos::abort( "Unknown boundary type" );
                }
            }
        }

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

    // Unpack the per-boundary recv buffers for local pairs (filled by
    // local_comm_copy_into_recv_buffers_). Runs before the MPI wait so the
    // host/GPU are not idle while remote messages are still in flight.
    void unpack_local_(
        const GridDataType& data,
        SubdomainNeighborhoodSendRecvBuffer< ScalarType, VecDim >& boundary_recv_buffers,
        CommunicationReduction reduction ) const
    {
        util::Timer timer( "ShellBoundaryCommPlan::unpack_local" );

        for ( const auto& p : local_pairs_ )
        {
            if ( p.boundary_type == 0 )
            {
                const auto local_boundary    = static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary );
                const auto neighbor_boundary = static_cast< grid::BoundaryVertex >( p.neighbor_subdomain_boundary );
                const auto& recv_buffer      = boundary_recv_buffers.buffer_vertex(
                    p.local_subdomain, local_boundary, p.neighbor_subdomain, neighbor_boundary );
                copy_from_buffer_rotate_and_reduce(
                    recv_buffer, data, p.local_subdomain_id, local_boundary, reduction );
            }
            else if ( p.boundary_type == 1 )
            {
                const auto local_boundary    = static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );
                const auto neighbor_boundary = static_cast< grid::BoundaryEdge >( p.neighbor_subdomain_boundary );
                const auto& recv_buffer      = boundary_recv_buffers.buffer_edge(
                    p.local_subdomain, local_boundary, p.neighbor_subdomain, neighbor_boundary );
                copy_from_buffer_rotate_and_reduce(
                    recv_buffer, data, p.local_subdomain_id, local_boundary, p.direction_0, reduction );
            }
            else if ( p.boundary_type == 2 )
            {
                const auto local_boundary    = static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );
                const auto neighbor_boundary = static_cast< grid::BoundaryFace >( p.neighbor_subdomain_boundary );
                const auto& recv_buffer      = boundary_recv_buffers.buffer_face(
                    p.local_subdomain, local_boundary, p.neighbor_subdomain, neighbor_boundary );
                copy_from_buffer_rotate_and_reduce(
                    recv_buffer,
                    data,
                    p.local_subdomain_id,
                    local_boundary,
                    std::make_tuple( p.direction_0, p.direction_1 ),
                    reduction );
            }
            else
            {
                Kokkos::abort( "Unknown boundary type" );
            }
        }
    }

    // Unpack all chunks received from one remote rank. Reads directly from the
    // per-rank flat recv buffer via inline unmanaged views.
    void unpack_remote_rank_(
        const GridDataType&    data,
        const mpi::MPIRank     rank,
        CommunicationReduction reduction ) const
    {
        const auto& domain   = *domain_;
        auto&       rank_buf = recv_rank_buffers_.at( rank );
        const auto& chunks   = recv_chunks_by_rank_.at( rank );

        for ( const auto& ch : chunks )
        {
            const auto& p        = ch.pair;
            ScalarType* base_ptr = rank_buf.data() + ch.offset;

            if ( p.boundary_type == 0 )
            {
                using BufT     = grid::Grid0DDataVec< ScalarType, VecDim >;
                auto unmanaged = detail::make_unmanaged_like< BufT >( base_ptr );
                copy_from_buffer_rotate_and_reduce(
                    unmanaged,
                    data,
                    p.local_subdomain_id,
                    static_cast< grid::BoundaryVertex >( p.local_subdomain_boundary ),
                    reduction );
            }
            else if ( p.boundary_type == 1 )
            {
                using BufT             = grid::Grid1DDataVec< ScalarType, VecDim >;
                const auto local_edge  = static_cast< grid::BoundaryEdge >( p.local_subdomain_boundary );
                const int  n_nodes     = grid::is_edge_boundary_radial( local_edge ) ?
                                             domain.domain_info().subdomain_num_nodes_radially() :
                                             domain.domain_info().subdomain_num_nodes_per_side_laterally();
                auto unmanaged         = detail::make_unmanaged_like< BufT >( base_ptr, n_nodes );
                copy_from_buffer_rotate_and_reduce(
                    unmanaged, data, p.local_subdomain_id, local_edge, p.direction_0, reduction );
            }
            else if ( p.boundary_type == 2 )
            {
                using BufT             = grid::Grid2DDataVec< ScalarType, VecDim >;
                const auto local_face  = static_cast< grid::BoundaryFace >( p.local_subdomain_boundary );
                const int  ni          = domain.domain_info().subdomain_num_nodes_per_side_laterally();
                const int  nj          = grid::is_face_boundary_normal_to_radial_direction( local_face ) ?
                                             domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                             domain.domain_info().subdomain_num_nodes_radially();
                auto unmanaged         = detail::make_unmanaged_like< BufT >( base_ptr, ni, nj );
                copy_from_buffer_rotate_and_reduce(
                    unmanaged,
                    data,
                    p.local_subdomain_id,
                    local_face,
                    std::make_tuple( p.direction_0, p.direction_1 ),
                    reduction );
            }
            else
            {
                Kokkos::abort( "Unknown boundary type" );
            }
        }
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

    // Precomputed local-only subset
    std::vector< SendRecvPair > local_pairs_;

    // Precomputed rank aggregation layouts
    std::map< mpi::MPIRank, std::vector< ChunkInfo > > send_chunks_by_rank_;
    std::map< mpi::MPIRank, std::vector< ChunkInfo > > recv_chunks_by_rank_;
    std::map< mpi::MPIRank, int >                      send_total_by_rank_;
    std::map< mpi::MPIRank, int >                      recv_total_by_rank_;


    // Reused rank buffers
    mutable std::map< mpi::MPIRank, rank_buffer_view > send_rank_buffers_;
    mutable std::map< mpi::MPIRank, rank_buffer_view > recv_rank_buffers_;

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
