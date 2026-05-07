
#pragma once

#include <algorithm>
#include <cassert>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <Kokkos_Core.hpp>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "terra/mpi/mpi.hpp"
#include "util/timer.hpp"

namespace terra::communication::shell {

/// @brief Move a distributed grid vector from a fine DistributedDomain to a coarse one
/// whose comm is a subset of the fine domain's comm. Used by agglomerated multigrid to
/// collapse the active rank set at the descent into a coarse V-cycle level, and then
/// to broadcast the coarse correction back to the fine rank set on the way up.
///
/// Assumptions:
/// - Both domains describe the same mesh (same num_global_subdomains, same per-subdomain
///   node layout). Only the owner->rank mapping differs.
/// - The coarse domain's comm is a subset of the fine domain's comm. Ranks that are not
///   in the coarse comm get MPI_COMM_NULL for `domain_coarse.comm()` and own zero
///   subdomains on the coarse side.
/// - Data layout: grid data is a 4D/5D Kokkos view indexed by (local_sdr, i, j, k[, c]),
///   with a fixed block size per subdomain. We use the fine domain's layout to determine
///   that block size (same on both sides by the same-mesh assumption).
///
/// The class is stateful: it precomputes send/recv counts and displacements once at
/// construction, then reuses them across solves. apply() and apply_transpose() are the
/// hot paths; they pack/Alltoallv/unpack.
template < class GridDataType >
class Redistribute
{
  public:
    using ScalarType            = typename GridDataType::value_type;
    static constexpr int VecDim = grid::grid_data_vec_dim< GridDataType >();
    using memory_space          = typename GridDataType::memory_space;
    using buffer_view           = Kokkos::View< ScalarType*, memory_space >;
    using host_buffer_view      = Kokkos::View< ScalarType*, Kokkos::HostSpace >;

    /// @brief Build a redistribute plan between two distributed domains.
    /// @param domain_fine   Source side; holds data before apply() and after apply_transpose().
    /// @param domain_coarse Destination side; holds data after apply() and before apply_transpose().
    /// @param subdomain_to_rank_fine   Fine-side owner function (sub-comm-local ranks on fine comm).
    /// @param subdomain_to_rank_coarse Coarse-side owner function (sub-comm-local ranks on coarse comm).
    Redistribute(
        const grid::shell::DistributedDomain& domain_fine,
        const grid::shell::DistributedDomain& domain_coarse,
        const grid::shell::SubdomainToRankDistributionFunction& subdomain_to_rank_fine,
        const grid::shell::SubdomainToRankDistributionFunction& subdomain_to_rank_coarse )
    : domain_fine_( &domain_fine )
    , domain_coarse_( &domain_coarse )
    , union_comm_( domain_fine.comm() )
    {
        build_plan_( subdomain_to_rank_fine, subdomain_to_rank_coarse );
    }

    /// @brief True when the fine and coarse domains have the same comm AND
    /// every subdomain has the same owner on both sides. In that case there is
    /// nothing to do — the caller can route restriction output directly to the
    /// coarse-side buffer and skip calling apply/apply_transpose entirely.
    bool is_identity() const { return identity_plan_; }

    /// @brief Move data from fine-owned subdomains to coarse-owned subdomains.
    /// Collective on the fine comm; every rank in it must call this.
    void apply( const GridDataType& src_fine, GridDataType& dst_coarse )
    {
        if ( union_comm_ == MPI_COMM_NULL ) return;  // rank not in this redistribute's comm
        if ( identity_plan_ ) return;                // factor=1 — caller already routed src/dst to the same buffer
        util::Timer t( "redistribute_apply" );
        run_alltoallv_( src_fine, dst_coarse, /*transpose=*/false );
    }

    /// @brief Move data back from coarse-owned subdomains to fine-owned subdomains.
    /// Collective on the fine comm. Used on the way up in a V-cycle after the coarse
    /// correction has been computed on the reduced rank set.
    void apply_transpose( const GridDataType& src_coarse, GridDataType& dst_fine )
    {
        if ( union_comm_ == MPI_COMM_NULL ) return;
        if ( identity_plan_ ) return;
        util::Timer t( "redistribute_apply_transpose" );
        run_alltoallv_( src_coarse, dst_fine, /*transpose=*/true );
    }

  private:
    static constexpr int Ni_( const grid::shell::DistributedDomain& d )
    {
        return d.domain_info().subdomain_num_nodes_per_side_laterally();
    }
    static constexpr int Nr_( const grid::shell::DistributedDomain& d )
    {
        return d.domain_info().subdomain_num_nodes_radially();
    }

    int subdomain_block_size_() const
    {
        // Same for both domains by the same-mesh assumption; pick fine.
        const int ni = Ni_( *domain_fine_ );
        const int nr = Nr_( *domain_fine_ );
        return ni * ni * nr * VecDim;
    }

  public:
    // A message: ships one subdomain's DoFs between a send and a receive rank.
    // Public so NVCC's cudafe1 stub (generated for device-side captures of
    // send_messages_ / recv_messages_) can reference the type.
    struct Message
    {
        int send_rank;               // rank on union_comm_
        int recv_rank;               // rank on union_comm_
        int local_subdomain_on_fine; // >= 0 if the sender uses the fine domain's local index
        int local_subdomain_on_coarse; // >= 0 if the receiver uses the coarse domain's local index
    };

  private:
    void build_plan_(
        const grid::shell::SubdomainToRankDistributionFunction& fn_fine,
        const grid::shell::SubdomainToRankDistributionFunction& fn_coarse )
    {
        // On ranks that are entirely outside this redistribute's union comm
        // (common in a hierarchical agglomeration ladder: a rank that dropped
        // out at descent Li+1 must still hold a per-level Redistribute slot for
        // descent Li, but never participates in it), skip plan construction.
        // The apply/apply_transpose guards below also early-return.
        if ( union_comm_ == MPI_COMM_NULL )
            return;

        const auto& dom_info = domain_fine_->domain_info();
        const int   n_diam   = dom_info.num_subdomains_per_diamond_side();
        const int   n_rad    = dom_info.num_subdomains_in_radial_direction();

        // Translate coarse-comm ranks to union-comm ranks using MPI groups.
        // Ranks not on the coarse comm still need to resolve its ownership so they
        // know where to send their locally-owned subdomains.
        MPI_Group union_group = MPI_GROUP_NULL;
        MPI_Group coarse_group = MPI_GROUP_NULL;
        MPI_Comm_group( union_comm_, &union_group );

        const MPI_Comm coarse_comm = domain_coarse_->comm();
        const bool     have_coarse = ( coarse_comm != MPI_COMM_NULL );
        if ( have_coarse )
        {
            MPI_Comm_group( coarse_comm, &coarse_group );
        }

        int coarse_size = 0;
        if ( have_coarse )
        {
            MPI_Group_size( coarse_group, &coarse_size );
        }
        // Broadcast coarse_size from the lowest-ranked coarse member so non-members know it.
        MPI_Allreduce( MPI_IN_PLACE, &coarse_size, 1, MPI_INT, MPI_MAX, union_comm_ );

        // Build the coarse->union rank translation on every rank: every rank on the
        // union comm needs to know, for each coarse-comm rank c, what the corresponding
        // union-comm rank is. We do this by having the lowest union-rank coarse member
        // broadcast the full translation table.
        std::vector< int > coarse_to_union( coarse_size, MPI_PROC_NULL );
        if ( have_coarse )
        {
            std::vector< int > coarse_ranks( coarse_size );
            std::iota( coarse_ranks.begin(), coarse_ranks.end(), 0 );
            MPI_Group_translate_ranks( coarse_group, coarse_size, coarse_ranks.data(),
                                        union_group, coarse_to_union.data() );
        }

        // Elect the first coarse member to broadcast the table. That rank's union index is 0
        // if we kept rank 0 on the coarse side (which build_level_comms always does by construction).
        // Use Allreduce(MIN) over each rank's "I am coarse rank 0 → my union rank else INT_MAX" to discover the broadcaster.
        int bcaster_union_rank = std::numeric_limits< int >::max();
        if ( have_coarse )
        {
            int my_coarse_rank = mpi::rank( coarse_comm );
            if ( my_coarse_rank == 0 )
            {
                bcaster_union_rank = mpi::rank( union_comm_ );
            }
        }
        MPI_Allreduce( MPI_IN_PLACE, &bcaster_union_rank, 1, MPI_INT, MPI_MIN, union_comm_ );
        MPI_Bcast( coarse_to_union.data(), coarse_size, MPI_INT, bcaster_union_rank, union_comm_ );

        coarse_to_union_ = std::move( coarse_to_union );

        if ( union_group != MPI_GROUP_NULL ) MPI_Group_free( &union_group );
        if ( coarse_group != MPI_GROUP_NULL ) MPI_Group_free( &coarse_group );

        // Compute local-subdomain index maps for fine and coarse (so pack/unpack know
        // where each subdomain's DoFs live in the 4D view).
        //
        // The existing DistributedDomain caches subdomains() with their local indices.
        std::map< grid::shell::SubdomainInfo, int > fine_local_idx;
        for ( const auto& [sdr, info] : domain_fine_->subdomains() )
        {
            fine_local_idx[sdr] = std::get< 0 >( info );
        }
        std::map< grid::shell::SubdomainInfo, int > coarse_local_idx;
        for ( const auto& [sdr, info] : domain_coarse_->subdomains() )
        {
            coarse_local_idx[sdr] = std::get< 0 >( info );
        }

        const int my_union_rank = mpi::rank( union_comm_ );
        const int block         = subdomain_block_size_();

        // Walk all subdomains globally. For each one, determine send rank (fine owner
        // translated to union) and recv rank (coarse owner translated to union).
        // Record the message on every rank that participates in either side, and count
        // the per-peer send/recv bytes on this rank.
        int union_size = 0;
        MPI_Comm_size( union_comm_, &union_size );
        send_counts_.assign( union_size, 0 );
        recv_counts_.assign( union_size, 0 );

        send_messages_.clear();
        recv_messages_.clear();

        for ( int diamond_id = 0; diamond_id < 10; ++diamond_id )
        {
            for ( int x = 0; x < n_diam; ++x )
            {
                for ( int y = 0; y < n_diam; ++y )
                {
                    for ( int r = 0; r < n_rad; ++r )
                    {
                        grid::shell::SubdomainInfo s( diamond_id, x, y, r );

                        const int fine_owner   = fn_fine( s, n_diam, n_rad );   // union-comm rank
                        const int coarse_owner = fn_coarse( s, n_diam, n_rad ); // coarse-comm rank
                        if ( coarse_owner < 0 || coarse_owner >= coarse_size )
                        {
                            throw std::runtime_error(
                                "Redistribute: coarse subdomain_to_rank produced out-of-range rank" );
                        }
                        const int recv_union = coarse_to_union_[coarse_owner];

                        const bool i_send = ( fine_owner == my_union_rank );
                        const bool i_recv = ( recv_union == my_union_rank );

                        if ( i_send )
                        {
                            Message m;
                            m.send_rank                 = my_union_rank;
                            m.recv_rank                 = recv_union;
                            m.local_subdomain_on_fine   = fine_local_idx.at( s );
                            m.local_subdomain_on_coarse = -1;
                            send_messages_.push_back( m );
                            send_counts_[recv_union] += block;
                        }
                        if ( i_recv )
                        {
                            Message m;
                            m.send_rank                 = fine_owner;
                            m.recv_rank                 = my_union_rank;
                            m.local_subdomain_on_fine   = -1;
                            m.local_subdomain_on_coarse = coarse_local_idx.at( s );
                            recv_messages_.push_back( m );
                            recv_counts_[fine_owner] += block;
                        }
                    }
                }
            }
        }

        // Sort messages by peer and subdomain index so pack/unpack order matches on both sides.
        auto by_peer_and_subdomain = []( const Message& a, const Message& b ) {
            if ( a.send_rank != b.send_rank ) return a.send_rank < b.send_rank;
            if ( a.recv_rank != b.recv_rank ) return a.recv_rank < b.recv_rank;
            // Use whichever local index is set as tie-breaker
            const int ai = a.local_subdomain_on_fine >= 0 ? a.local_subdomain_on_fine : a.local_subdomain_on_coarse;
            const int bi = b.local_subdomain_on_fine >= 0 ? b.local_subdomain_on_fine : b.local_subdomain_on_coarse;
            return ai < bi;
        };

        // For sends, we pack per recv_rank. For recvs, we unpack per send_rank.
        // To make the two sides match, both sides must order by the SAME primary key
        // (the peer rank across the wire) and the SAME secondary key (subdomain identity).
        // Sorting by SubdomainInfo is natural but we've already lost that; use the peer's
        // local subdomain index instead. That's fine because on each (sender, receiver)
        // pair, a subdomain's local index on its owner's domain is uniquely determined.

        // Since send_messages_ always has local_subdomain_on_fine set and recv_messages_
        // has local_subdomain_on_coarse set, we sort each independently by a consistent
        // key (peer rank) and accept that the per-peer chunk order is whatever the loop
        // produced — that order is globally deterministic (diamond, x, y, r iteration),
        // so both sides see the same sequence.
        std::stable_sort( send_messages_.begin(), send_messages_.end(),
                          []( const Message& a, const Message& b ) { return a.recv_rank < b.recv_rank; } );
        std::stable_sort( recv_messages_.begin(), recv_messages_.end(),
                          []( const Message& a, const Message& b ) { return a.send_rank < b.send_rank; } );

        // Build Alltoallv displacements.
        send_displs_.assign( union_size, 0 );
        recv_displs_.assign( union_size, 0 );
        for ( int r = 1; r < union_size; ++r )
        {
            send_displs_[r] = send_displs_[r - 1] + send_counts_[r - 1];
            recv_displs_[r] = recv_displs_[r - 1] + recv_counts_[r - 1];
        }
        const int total_send = send_displs_.back() + send_counts_.back();
        const int total_recv = recv_displs_.back() + recv_counts_.back();

        // Always allocate at least 1 element so .data() is a valid pointer even
        // on ranks that have nothing to send or receive — nvpcx MPI_Alltoallv does
        // not accept nullptr buffer arguments even when the per-rank count is 0.
        const int send_alloc = std::max( total_send, 1 );
        const int recv_alloc = std::max( total_recv, 1 );
        send_buf_ = buffer_view( Kokkos::view_alloc( Kokkos::WithoutInitializing, "redistribute_send" ), send_alloc );
        recv_buf_ = buffer_view( Kokkos::view_alloc( Kokkos::WithoutInitializing, "redistribute_recv" ), recv_alloc );
        // Host staging buffers: nvpcx MPI_Alltoallv on GPU pointers hangs on this
        // cluster (whereas Isend/Irecv is CUDA-aware). Stage through host memory.
        send_host_ = host_buffer_view( Kokkos::view_alloc( Kokkos::WithoutInitializing, "redistribute_send_host" ),
                                        send_alloc );
        recv_host_ = host_buffer_view( Kokkos::view_alloc( Kokkos::WithoutInitializing, "redistribute_recv_host" ),
                                        recv_alloc );

        // Identity detection: if every subdomain I own stays on my rank (i.e.
        // fine owner == coarse owner on this rank), and this holds on every
        // rank, the Redistribute has nothing to do at runtime — the caller
        // should just reuse the same buffer on both sides of the descent.
        // Check locally and MPI_Allreduce LAND across union_comm_ to make sure
        // every rank agrees before we claim identity. Otherwise one rank
        // would skip apply while others wait in Alltoallv.
        int local_identity = 1;
        for ( const auto& m : send_messages_ )
            if ( m.send_rank != m.recv_rank ) { local_identity = 0; break; }
        if ( local_identity )
            for ( const auto& m : recv_messages_ )
                if ( m.send_rank != m.recv_rank ) { local_identity = 0; break; }
        MPI_Allreduce( MPI_IN_PLACE, &local_identity, 1, MPI_INT, MPI_LAND, union_comm_ );
        identity_plan_ = ( local_identity != 0 );
    }

  public:
    // Pack / unpack are public so nvcc permits extended __host__ __device__ lambdas
    // inside them (private/protected enclosing functions are disallowed for extended
    // lambdas). Not part of the intended API surface — call apply()/apply_transpose().
    void pack_( const GridDataType& src, const buffer_view& buf, const std::vector< Message >& messages,
                bool use_fine_index ) const
    {
        const int ni    = Ni_( *domain_fine_ );
        const int nr    = Nr_( *domain_fine_ );
        const int block = subdomain_block_size_();

        // Copy message list to a device view so the kernel can index it.
        Kokkos::View< int*, memory_space > msg_sdr( Kokkos::view_alloc( Kokkos::WithoutInitializing, "msg_sdr" ),
                                                    messages.size() );
        auto msg_sdr_host = Kokkos::create_mirror_view( msg_sdr );
        for ( size_t i = 0; i < messages.size(); ++i )
        {
            msg_sdr_host( i ) =
                use_fine_index ? messages[i].local_subdomain_on_fine : messages[i].local_subdomain_on_coarse;
        }
        Kokkos::deep_copy( msg_sdr, msg_sdr_host );

        const int num_messages = static_cast< int >( messages.size() );
        if ( num_messages == 0 ) return;

        // Dispatch at call site, not inside the lambda — nvcc rejects first-capture
        // inside `if constexpr` bodies of extended __device__ lambdas.
        if constexpr ( VecDim == 1 )
        {
            Kokkos::parallel_for(
                "redistribute_pack",
                Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >( { 0, 0, 0, 0, 0 }, { num_messages, ni, ni, nr, VecDim } ),
                KOKKOS_LAMBDA( int m, int i, int j, int k, int c ) {
                    const int local_sdr = msg_sdr( m );
                    const int flat      = m * block + ( ( i * ni + j ) * nr + k ) * VecDim + c;
                    buf( flat )         = src( local_sdr, i, j, k );
                } );
        }
        else
        {
            Kokkos::parallel_for(
                "redistribute_pack",
                Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >( { 0, 0, 0, 0, 0 }, { num_messages, ni, ni, nr, VecDim } ),
                KOKKOS_LAMBDA( int m, int i, int j, int k, int c ) {
                    const int local_sdr = msg_sdr( m );
                    const int flat      = m * block + ( ( i * ni + j ) * nr + k ) * VecDim + c;
                    buf( flat )         = src( local_sdr, i, j, k, c );
                } );
        }
        Kokkos::fence( "redistribute_pack" );
    }

    void unpack_( GridDataType& dst, const buffer_view& buf, const std::vector< Message >& messages,
                  bool use_fine_index ) const
    {
        const int ni    = Ni_( *domain_fine_ );
        const int nr    = Nr_( *domain_fine_ );
        const int block = subdomain_block_size_();

        Kokkos::View< int*, memory_space > msg_sdr( Kokkos::view_alloc( Kokkos::WithoutInitializing, "msg_sdr" ),
                                                    messages.size() );
        auto msg_sdr_host = Kokkos::create_mirror_view( msg_sdr );
        for ( size_t i = 0; i < messages.size(); ++i )
        {
            msg_sdr_host( i ) =
                use_fine_index ? messages[i].local_subdomain_on_fine : messages[i].local_subdomain_on_coarse;
        }
        Kokkos::deep_copy( msg_sdr, msg_sdr_host );

        const int num_messages = static_cast< int >( messages.size() );
        if ( num_messages == 0 ) return;

        if constexpr ( VecDim == 1 )
        {
            Kokkos::parallel_for(
                "redistribute_unpack",
                Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >( { 0, 0, 0, 0, 0 }, { num_messages, ni, ni, nr, VecDim } ),
                KOKKOS_LAMBDA( int m, int i, int j, int k, int c ) {
                    const int local_sdr       = msg_sdr( m );
                    const int flat            = m * block + ( ( i * ni + j ) * nr + k ) * VecDim + c;
                    dst( local_sdr, i, j, k ) = buf( flat );
                } );
        }
        else
        {
            Kokkos::parallel_for(
                "redistribute_unpack",
                Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >( { 0, 0, 0, 0, 0 }, { num_messages, ni, ni, nr, VecDim } ),
                KOKKOS_LAMBDA( int m, int i, int j, int k, int c ) {
                    const int local_sdr          = msg_sdr( m );
                    const int flat               = m * block + ( ( i * ni + j ) * nr + k ) * VecDim + c;
                    dst( local_sdr, i, j, k, c ) = buf( flat );
                } );
        }
        Kokkos::fence( "redistribute_unpack" );
    }

  private:
    void run_alltoallv_( const GridDataType& src, GridDataType& dst, bool transpose )
    {
        // Forward pass: pack from fine-indexed subdomains, send to coarse-indexed subdomains.
        // Transpose: pack from coarse-indexed subdomains, send to fine-indexed subdomains.
        const std::vector< Message >& pack_msgs   = transpose ? recv_messages_ : send_messages_;
        const std::vector< Message >& unpack_msgs = transpose ? send_messages_ : recv_messages_;
        const std::vector< int >&     s_counts    = transpose ? recv_counts_   : send_counts_;
        const std::vector< int >&     s_displs    = transpose ? recv_displs_   : send_displs_;
        const std::vector< int >&     r_counts    = transpose ? send_counts_   : recv_counts_;
        const std::vector< int >&     r_displs    = transpose ? send_displs_   : recv_displs_;

        const bool pack_uses_fine_idx   = !transpose; // forward: pack from fine-owned subdomains
        const bool unpack_uses_fine_idx = transpose;

        // Select staging buffers. "send" side of this direction is the buffer we pack into.
        buffer_view&      device_send = transpose ? recv_buf_  : send_buf_;
        buffer_view&      device_recv = transpose ? send_buf_  : recv_buf_;
        host_buffer_view& host_send   = transpose ? recv_host_ : send_host_;
        host_buffer_view& host_recv   = transpose ? send_host_ : recv_host_;

        pack_( src, device_send, pack_msgs, pack_uses_fine_idx );

        // Device -> host staging (Alltoallv is not GPU-aware on nvpcx).
        Kokkos::deep_copy( host_send, device_send );

        MPI_Alltoallv( host_send.data(),
                        s_counts.data(),
                        s_displs.data(),
                        mpi::mpi_datatype< ScalarType >(),
                        host_recv.data(),
                        r_counts.data(),
                        r_displs.data(),
                        mpi::mpi_datatype< ScalarType >(),
                        union_comm_ );

        // Host -> device; then unpack.
        Kokkos::deep_copy( device_recv, host_recv );

        unpack_( dst, device_recv, unpack_msgs, unpack_uses_fine_idx );
    }

    const grid::shell::DistributedDomain* domain_fine_   = nullptr;
    const grid::shell::DistributedDomain* domain_coarse_ = nullptr;
    MPI_Comm                              union_comm_    = MPI_COMM_NULL;

    std::vector< int > coarse_to_union_; // index = coarse-comm rank, value = union-comm rank

    std::vector< Message > send_messages_;
    std::vector< Message > recv_messages_;

    std::vector< int > send_counts_;
    std::vector< int > send_displs_;
    std::vector< int > recv_counts_;
    std::vector< int > recv_displs_;

    buffer_view      send_buf_;
    buffer_view      recv_buf_;
    host_buffer_view send_host_;
    host_buffer_view recv_host_;

    /// True when every rank on union_comm_ owns the same subdomains on both
    /// sides of the descent — apply/apply_transpose are pure no-ops. Set at
    /// plan construction via a collective check across union_comm_.
    bool identity_plan_ = false;
};

} // namespace terra::communication::shell
