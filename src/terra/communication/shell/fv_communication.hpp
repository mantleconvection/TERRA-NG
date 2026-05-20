
#pragma once

/// @file fv_communication.hpp
/// @brief Ghost layer communication for scalar finite volume (FV) fields on the shell grid.
///
/// FV cells are owned exclusively by one subdomain.  Each cell array has a one-cell-wide ghost
/// layer on every side so that operator stencils can read neighbour values without extra index
/// checks.  Unlike the FE communication in communication.hpp — which *adds* contributions from
/// shared nodes — here we simply *copy* the innermost real cells of the neighbour into our ghost
/// cells (no reduction required).
///
/// Layout reminder (Grid4DDataScalar, shape [n_subdomains, N_lat+1, N_lat+1, N_rad+1]):
///   - Real cells:  indices [1, N_lat-1] × [1, N_lat-1] × [1, N_rad-1]
///   - Ghost cells: index 0 and index N (for each axis, N = last extent index)
///
/// Only face-to-face communication is implemented here.  A 6-neighbour cell stencil (±x, ±y, ±r)
/// never touches edge or vertex ghost cells, so face communication is sufficient.
///
/// Typical usage (operator that is applied repeatedly):
/// @code
///   // Construct once (allocates Kokkos views for send/recv buffers):
///   FVGhostLayerBuffers<double> ghost_bufs(domain);
///
///   // Call every time the field changes (no allocation, just pack/MPI/unpack):
///   update_fv_ghost_layers(domain, field, ghost_bufs);
/// @endcode

#include <algorithm>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "mpi/mpi.hpp"

namespace terra::communication::shell {

/// MPI tag for FV ghost layer messages (distinct from MPI_TAG_BOUNDARY_DATA = 100).
constexpr int MPI_TAG_FV_GHOST_LAYERS = 200;

// ============================================================================
// Internal pack / unpack kernels
// ============================================================================

namespace fv_detail {

/// @brief Packs the innermost real-cell layer adjacent to `face` into `buffer`.
///
/// For a face whose normal is in direction d:
///   - Ghost layer lives at data index 0 (P0 side) or extent-1 (P1 side) in d.
///   - The innermost *real* cells are at index 1 (P0) or extent-2 (P1).
///
/// The two varying dimensions are packed in *forward* order:
///   buffer(i, j) = data(id, ..., i+1, j+1, ...)
///
/// The receiver applies BoundaryDirection reversal during unpack if needed.
template < typename ScalarType >
void pack_inner_cells(
    const grid::Grid2DDataScalar< ScalarType >& buffer,
    const grid::Grid4DDataScalar< ScalarType >& data,
    const int                                   local_subdomain_id,
    const grid::BoundaryFace                    face )
{
    using namespace grid;

    const auto pos_x = boundary_position_from_boundary_type_x( face );
    const auto pos_y = boundary_position_from_boundary_type_y( face );
    const auto pos_r = boundary_position_from_boundary_type_r( face );

    const int sx = static_cast< int >( data.extent( 1 ) );
    const int sy = static_cast< int >( data.extent( 2 ) );
    const int sr = static_cast< int >( data.extent( 3 ) );
    const int ni = static_cast< int >( buffer.extent( 0 ) );
    const int nj = static_cast< int >( buffer.extent( 1 ) );
    const int id = local_subdomain_id;

    Kokkos::parallel_for(
        "fv_pack_inner_cells",
        Kokkos::MDRangePolicy< Kokkos::Rank< 2, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >( { 0, 0 }, { ni, nj } ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            int x = 0, y = 0, r = 0;

            if ( pos_x != BoundaryPosition::PV )
            {
                x = ( pos_x == BoundaryPosition::P0 ) ? 1 : ( sx - 2 );
                y = i + 1;
                r = j + 1;
            }
            else if ( pos_y != BoundaryPosition::PV )
            {
                x = i + 1;
                y = ( pos_y == BoundaryPosition::P0 ) ? 1 : ( sy - 2 );
                r = j + 1;
            }
            else
            {
                x = i + 1;
                y = j + 1;
                r = ( pos_r == BoundaryPosition::P0 ) ? 1 : ( sr - 2 );
            }

            buffer( i, j ) = data( id, x, y, r );
        } );

    Kokkos::fence();
}

/// @brief Writes `buffer` into the ghost cell layer at `face`.
///
/// @param dir0  Iteration direction for the first  varying dimension.
/// @param dir1  Iteration direction for the second varying dimension.
///
/// Direction mapping (from SubdomainNeighborhood::neighborhood_face()):
///   FORWARD:  buffer index i  →  data index i+1
///   BACKWARD: buffer index i  →  data index (extent - 2 - i)
template < typename ScalarType >
void unpack_to_ghost(
    const grid::Grid2DDataScalar< ScalarType >& buffer,
    const grid::Grid4DDataScalar< ScalarType >& data,
    const int                                   local_subdomain_id,
    const grid::BoundaryFace                    face,
    const grid::BoundaryDirection               dir0,
    const grid::BoundaryDirection               dir1 )
{
    using namespace grid;

    const auto pos_x = boundary_position_from_boundary_type_x( face );
    const auto pos_y = boundary_position_from_boundary_type_y( face );
    const auto pos_r = boundary_position_from_boundary_type_r( face );

    const int sx = static_cast< int >( data.extent( 1 ) );
    const int sy = static_cast< int >( data.extent( 2 ) );
    const int sr = static_cast< int >( data.extent( 3 ) );
    const int ni = static_cast< int >( buffer.extent( 0 ) );
    const int nj = static_cast< int >( buffer.extent( 1 ) );
    const int id = local_subdomain_id;

    Kokkos::parallel_for(
        "fv_unpack_to_ghost",
        Kokkos::MDRangePolicy< Kokkos::Rank< 2, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >( { 0, 0 }, { ni, nj } ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            int x = 0, y = 0, r = 0;

            if ( pos_x != BoundaryPosition::PV )
            {
                x = ( pos_x == BoundaryPosition::P0 ) ? 0 : ( sx - 1 );
                y = ( dir0 == BoundaryDirection::FORWARD ) ? ( i + 1 ) : ( sy - 2 - i );
                r = ( dir1 == BoundaryDirection::FORWARD ) ? ( j + 1 ) : ( sr - 2 - j );
            }
            else if ( pos_y != BoundaryPosition::PV )
            {
                x = ( dir0 == BoundaryDirection::FORWARD ) ? ( i + 1 ) : ( sx - 2 - i );
                y = ( pos_y == BoundaryPosition::P0 ) ? 0 : ( sy - 1 );
                r = ( dir1 == BoundaryDirection::FORWARD ) ? ( j + 1 ) : ( sr - 2 - j );
            }
            else
            {
                x = ( dir0 == BoundaryDirection::FORWARD ) ? ( i + 1 ) : ( sx - 2 - i );
                y = ( dir1 == BoundaryDirection::FORWARD ) ? ( j + 1 ) : ( sy - 2 - j );
                r = ( pos_r == BoundaryPosition::P0 ) ? 0 : ( sr - 1 );
            }

            data( id, x, y, r ) = buffer( i, j );
        } );

    Kokkos::fence();
}

} // namespace fv_detail

// ============================================================================
// FVGhostLayerBuffers — pre-allocated communication state
// ============================================================================

/// @brief Pre-allocated send/recv buffers and sorted face-pair lists for FV ghost layer comm.
///
/// Construct once per operator (or wherever ghost layer updates are needed), then pass to
/// `update_fv_ghost_layers` on every call.  No heap allocation happens after construction.
template < typename ScalarType >
class FVGhostLayerBuffers
{
  public:
    using Buffer = grid::Grid2DDataScalar< ScalarType >;

    /// @brief All metadata needed to drive communication for one subdomain face.
    struct FacePair
    {
        int                        buf_idx;            ///< Index into send_bufs_ / recv_bufs_.
        mpi::MPIRank               local_rank;
        grid::shell::SubdomainInfo local_subdomain;
        int                        local_subdomain_id;
        grid::BoundaryFace         local_face;
        mpi::MPIRank               neighbor_rank;
        grid::shell::SubdomainInfo neighbor_subdomain;
        int                        neighbor_subdomain_id; ///< Local array index; -1 if on remote rank.
        grid::BoundaryFace         neighbor_face;
        grid::BoundaryDirection    dir0; ///< Unpack direction for first  varying dimension.
        grid::BoundaryDirection    dir1; ///< Unpack direction for second varying dimension.
    };

    /// @brief Allocates all send/recv buffers and builds the sorted face pair lists.
    explicit FVGhostLayerBuffers( const grid::shell::DistributedDomain& domain ) { setup( domain ); }

    /// @brief Face pairs sorted for consistent MPI_Irecv posting order (by sending subdomain).
    const std::vector< FacePair >& recv_ordered() const { return recv_ordered_; }

    /// @brief Face pairs sorted for consistent MPI_Isend / local-comm packing order.
    const std::vector< FacePair >& send_ordered() const { return send_ordered_; }

    Buffer& send_buf( const FacePair& fp ) { return send_bufs_[fp.buf_idx]; }
    Buffer& recv_buf( const FacePair& fp ) { return recv_bufs_[fp.buf_idx]; }

  private:
    std::vector< Buffer >   send_bufs_; ///< One entry per face pair (indexed by FacePair::buf_idx).
    std::vector< Buffer >   recv_bufs_;
    std::vector< FacePair > recv_ordered_;
    std::vector< FacePair > send_ordered_;

    void setup( const grid::shell::DistributedDomain& domain )
    {
        using namespace grid;

        const int N_lat       = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        const int N_rad       = domain.domain_info().subdomain_num_nodes_radially();
        const int n_lat_cells = N_lat - 1;
        const int n_rad_cells = N_rad - 1;

        const mpi::MPIRank my_rank = mpi::rank();

        // Build the unsorted master list of face pairs, allocating one buffer per entry.
        std::vector< FacePair > all_pairs;
        all_pairs.reserve( domain.subdomains().size() * 6 );

        for ( const auto& [subdomain_info, idx_and_neighborhood] : domain.subdomains() )
        {
            const auto& [local_id, neighborhood] = idx_and_neighborhood;

            for ( const auto& [local_face, neighbor_tuple] : neighborhood.neighborhood_face() )
            {
                const auto& [neighbor_info, neighbor_face, unpack_dirs, neighbor_rank] = neighbor_tuple;

                // Resolve the neighbour's local array index (only valid for on-rank neighbours).
                int neighbor_local_id = -1;
                if ( neighbor_rank == my_rank && domain.subdomains().contains( neighbor_info ) )
                    neighbor_local_id = std::get< 0 >( domain.subdomains().at( neighbor_info ) );

                // Buffer dimensions: lateral×radial for x/y-normal faces, lateral×lateral for r-normal.
                const auto px = boundary_position_from_boundary_type_x( local_face );
                const auto py = boundary_position_from_boundary_type_y( local_face );
                const bool r_normal = ( px == BoundaryPosition::PV && py == BoundaryPosition::PV );
                const int  ni       = n_lat_cells;
                const int  nj       = r_normal ? n_lat_cells : n_rad_cells;

                const int buf_idx = static_cast< int >( all_pairs.size() );
                send_bufs_.emplace_back( "fv_ghost_send", ni, nj );
                recv_bufs_.emplace_back( "fv_ghost_recv", ni, nj );

                all_pairs.push_back( FacePair{
                    .buf_idx               = buf_idx,
                    .local_rank            = my_rank,
                    .local_subdomain       = subdomain_info,
                    .local_subdomain_id    = local_id,
                    .local_face            = local_face,
                    .neighbor_rank         = neighbor_rank,
                    .neighbor_subdomain    = neighbor_info,
                    .neighbor_subdomain_id = neighbor_local_id,
                    .neighbor_face         = neighbor_face,
                    .dir0                  = std::get< 0 >( unpack_dirs ),
                    .dir1                  = std::get< 1 >( unpack_dirs ),
                } );
            }
        }

        // Sort for MPI_Irecv: order by sending (neighbour) subdomain so all ranks post recvs
        // from the same sender in the same order.
        recv_ordered_ = all_pairs;
        std::sort( recv_ordered_.begin(), recv_ordered_.end(), []( const FacePair& a, const FacePair& b ) {
            if ( a.neighbor_subdomain != b.neighbor_subdomain )
                return a.neighbor_subdomain < b.neighbor_subdomain;
            if ( a.neighbor_face != b.neighbor_face )
                return static_cast< int >( a.neighbor_face ) < static_cast< int >( b.neighbor_face );
            if ( a.local_subdomain != b.local_subdomain )
                return a.local_subdomain < b.local_subdomain;
            return static_cast< int >( a.local_face ) < static_cast< int >( b.local_face );
        } );

        // Sort for MPI_Isend: order by local subdomain.
        send_ordered_ = all_pairs;
        std::sort( send_ordered_.begin(), send_ordered_.end(), []( const FacePair& a, const FacePair& b ) {
            if ( a.local_subdomain != b.local_subdomain )
                return a.local_subdomain < b.local_subdomain;
            if ( a.local_face != b.local_face )
                return static_cast< int >( a.local_face ) < static_cast< int >( b.local_face );
            if ( a.neighbor_subdomain != b.neighbor_subdomain )
                return a.neighbor_subdomain < b.neighbor_subdomain;
            return static_cast< int >( a.neighbor_face ) < static_cast< int >( b.neighbor_face );
        } );
    }
};

// ============================================================================
// update_fv_ghost_layers — the actual communication call
// ============================================================================

/// @brief Fills all ghost layers of a scalar FV field from neighbouring subdomains.
///
/// Pre-allocated buffers and sorted face-pair lists from `bufs` are used; no allocation occurs.
///
/// @param domain  The distributed domain.
/// @param data    FV scalar field — ghost cells are written in-place.
/// @param bufs    Pre-allocated buffers constructed for this domain (constructed once, reused).
template < typename ScalarType >
void update_fv_ghost_layers(
    const grid::shell::DistributedDomain&       domain,
    const grid::Grid4DDataScalar< ScalarType >& data,
    FVGhostLayerBuffers< ScalarType >&          bufs )
{
    using namespace fv_detail;

    std::vector< MPI_Request > recv_requests;
    std::vector< MPI_Request > send_requests;

    // Post Irecvs first (recv_ordered guarantees consistent ordering across ranks).
    for ( const auto& fp : bufs.recv_ordered() )
    {
        if ( fp.local_rank == fp.neighbor_rank )
            continue; // on-rank comm handled below

        auto& recv_buf = bufs.recv_buf( fp );
        MPI_Request req;
        MPI_Irecv(
            recv_buf.data(),
            static_cast< int >( recv_buf.span() ),
            mpi::mpi_datatype< ScalarType >(),
            fp.neighbor_rank,
            MPI_TAG_FV_GHOST_LAYERS,
            MPI_COMM_WORLD,
            &req );
        recv_requests.push_back( req );
    }

    // Pack and Isend (send_ordered guarantees consistent ordering).
    for ( const auto& fp : bufs.send_ordered() )
    {
        if ( fp.local_rank == fp.neighbor_rank )
        {
            // On-rank: read the neighbour's inner cells directly into our recv buffer.
            pack_inner_cells( bufs.recv_buf( fp ), data, fp.neighbor_subdomain_id, fp.neighbor_face );
        }
        else
        {
            // Remote: pack our inner cells and send.
            auto& send_buf = bufs.send_buf( fp );
            pack_inner_cells( send_buf, data, fp.local_subdomain_id, fp.local_face );
            // pack_inner_cells() calls Kokkos::fence(), so data is ready for MPI.

            MPI_Request req;
            MPI_Isend(
                send_buf.data(),
                static_cast< int >( send_buf.span() ),
                mpi::mpi_datatype< ScalarType >(),
                fp.neighbor_rank,
                MPI_TAG_FV_GHOST_LAYERS,
                MPI_COMM_WORLD,
                &req );
            send_requests.push_back( req );
        }
    }

    MPI_Waitall( static_cast< int >( send_requests.size() ), send_requests.data(), MPI_STATUSES_IGNORE );
    MPI_Waitall( static_cast< int >( recv_requests.size() ), recv_requests.data(), MPI_STATUSES_IGNORE );

    // Unpack all recv buffers into ghost layers (order does not matter here).
    for ( const auto& fp : bufs.recv_ordered() )
    {
        unpack_to_ghost( bufs.recv_buf( fp ), data, fp.local_subdomain_id, fp.local_face, fp.dir0, fp.dir1 );
    }
}

/// @brief Convenience overload — allocates temporary buffers internally.
///
/// @note THIS ALLOCATES ON EVERY CALL.  Use the overload with FVGhostLayerBuffers for any
///       code that runs repeatedly (iterative solvers, time-stepping loops).
template < typename ScalarType >
void update_fv_ghost_layers(
    const grid::shell::DistributedDomain&       domain,
    const grid::Grid4DDataScalar< ScalarType >& data )
{
    FVGhostLayerBuffers< ScalarType > tmp_bufs( domain );
    update_fv_ghost_layers( domain, data, tmp_bufs );
}

// ============================================================================
// Vector-valued FV ghost layer communication
// ============================================================================

/// MPI tag for vector-valued FV ghost layer messages.
constexpr int MPI_TAG_FV_VEC_GHOST_LAYERS = 201;

namespace fv_detail {

/// @brief Packs the innermost real-cell layer adjacent to `face` into `buffer` for a vector field.
///
/// buffer has shape [ni, nj, VecDim] and packs all VecDim components together so a single
/// MPI message per face suffices.
template < typename ScalarType, int VecDim >
void pack_inner_cells_vec(
    const grid::Grid3DDataScalar< ScalarType >&      buffer,
    const grid::Grid4DDataVec< ScalarType, VecDim >& data,
    const int                                        local_subdomain_id,
    const grid::BoundaryFace                         face )
{
    using namespace grid;

    const auto pos_x = boundary_position_from_boundary_type_x( face );
    const auto pos_y = boundary_position_from_boundary_type_y( face );
    const auto pos_r = boundary_position_from_boundary_type_r( face );

    const int sx = static_cast< int >( data.extent( 1 ) );
    const int sy = static_cast< int >( data.extent( 2 ) );
    const int sr = static_cast< int >( data.extent( 3 ) );
    const int ni = static_cast< int >( buffer.extent( 0 ) );
    const int nj = static_cast< int >( buffer.extent( 1 ) );
    const int id = local_subdomain_id;

    Kokkos::parallel_for(
        "fv_pack_inner_cells_vec",
        Kokkos::MDRangePolicy< Kokkos::Rank< 2, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >( { 0, 0 }, { ni, nj } ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            int x = 0, y = 0, r = 0;

            if ( pos_x != BoundaryPosition::PV )
            {
                x = ( pos_x == BoundaryPosition::P0 ) ? 1 : ( sx - 2 );
                y = i + 1;
                r = j + 1;
            }
            else if ( pos_y != BoundaryPosition::PV )
            {
                x = i + 1;
                y = ( pos_y == BoundaryPosition::P0 ) ? 1 : ( sy - 2 );
                r = j + 1;
            }
            else
            {
                x = i + 1;
                y = j + 1;
                r = ( pos_r == BoundaryPosition::P0 ) ? 1 : ( sr - 2 );
            }

            for ( int d = 0; d < VecDim; ++d )
                buffer( i, j, d ) = data( id, x, y, r, d );
        } );

    Kokkos::fence();
}

/// @brief Writes `buffer` into the ghost cell layer at `face` for a vector field.
template < typename ScalarType, int VecDim >
void unpack_to_ghost_vec(
    const grid::Grid3DDataScalar< ScalarType >&      buffer,
    const grid::Grid4DDataVec< ScalarType, VecDim >& data,
    const int                                        local_subdomain_id,
    const grid::BoundaryFace                         face,
    const grid::BoundaryDirection                    dir0,
    const grid::BoundaryDirection                    dir1 )
{
    using namespace grid;

    const auto pos_x = boundary_position_from_boundary_type_x( face );
    const auto pos_y = boundary_position_from_boundary_type_y( face );
    const auto pos_r = boundary_position_from_boundary_type_r( face );

    const int sx = static_cast< int >( data.extent( 1 ) );
    const int sy = static_cast< int >( data.extent( 2 ) );
    const int sr = static_cast< int >( data.extent( 3 ) );
    const int ni = static_cast< int >( buffer.extent( 0 ) );
    const int nj = static_cast< int >( buffer.extent( 1 ) );
    const int id = local_subdomain_id;

    Kokkos::parallel_for(
        "fv_unpack_to_ghost_vec",
        Kokkos::MDRangePolicy< Kokkos::Rank< 2, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >( { 0, 0 }, { ni, nj } ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            int x = 0, y = 0, r = 0;

            if ( pos_x != BoundaryPosition::PV )
            {
                x = ( pos_x == BoundaryPosition::P0 ) ? 0 : ( sx - 1 );
                y = ( dir0 == BoundaryDirection::FORWARD ) ? ( i + 1 ) : ( sy - 2 - i );
                r = ( dir1 == BoundaryDirection::FORWARD ) ? ( j + 1 ) : ( sr - 2 - j );
            }
            else if ( pos_y != BoundaryPosition::PV )
            {
                x = ( dir0 == BoundaryDirection::FORWARD ) ? ( i + 1 ) : ( sx - 2 - i );
                y = ( pos_y == BoundaryPosition::P0 ) ? 0 : ( sy - 1 );
                r = ( dir1 == BoundaryDirection::FORWARD ) ? ( j + 1 ) : ( sr - 2 - j );
            }
            else
            {
                x = ( dir0 == BoundaryDirection::FORWARD ) ? ( i + 1 ) : ( sx - 2 - i );
                y = ( dir1 == BoundaryDirection::FORWARD ) ? ( j + 1 ) : ( sy - 2 - j );
                r = ( pos_r == BoundaryPosition::P0 ) ? 0 : ( sr - 1 );
            }

            for ( int d = 0; d < VecDim; ++d )
                data( id, x, y, r, d ) = buffer( i, j, d );
        } );

    Kokkos::fence();
}

} // namespace fv_detail

/// @brief Pre-allocated send/recv buffers for vector-valued FV ghost layer communication.
///
/// Mirrors FVGhostLayerBuffers but uses 3D buffers [ni, nj, VecDim] so all VecDim components
/// are packed and sent in a single MPI message per face.
template < typename ScalarType, int VecDim >
class FVGhostLayerVecBuffers
{
  public:
    using Buffer = grid::Grid3DDataScalar< ScalarType >;

    struct FacePair
    {
        int                        buf_idx;
        mpi::MPIRank               local_rank;
        grid::shell::SubdomainInfo local_subdomain;
        int                        local_subdomain_id;
        grid::BoundaryFace         local_face;
        mpi::MPIRank               neighbor_rank;
        grid::shell::SubdomainInfo neighbor_subdomain;
        int                        neighbor_subdomain_id;
        grid::BoundaryFace         neighbor_face;
        grid::BoundaryDirection    dir0;
        grid::BoundaryDirection    dir1;
    };

    explicit FVGhostLayerVecBuffers( const grid::shell::DistributedDomain& domain ) { setup( domain ); }

    const std::vector< FacePair >& recv_ordered() const { return recv_ordered_; }
    const std::vector< FacePair >& send_ordered() const { return send_ordered_; }

    Buffer& send_buf( const FacePair& fp ) { return send_bufs_[fp.buf_idx]; }
    Buffer& recv_buf( const FacePair& fp ) { return recv_bufs_[fp.buf_idx]; }

  private:
    std::vector< Buffer >   send_bufs_;
    std::vector< Buffer >   recv_bufs_;
    std::vector< FacePair > recv_ordered_;
    std::vector< FacePair > send_ordered_;

    void setup( const grid::shell::DistributedDomain& domain )
    {
        using namespace grid;

        const int N_lat       = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        const int N_rad       = domain.domain_info().subdomain_num_nodes_radially();
        const int n_lat_cells = N_lat - 1;
        const int n_rad_cells = N_rad - 1;

        const mpi::MPIRank my_rank = mpi::rank();

        std::vector< FacePair > all_pairs;
        all_pairs.reserve( domain.subdomains().size() * 6 );

        for ( const auto& [subdomain_info, idx_and_neighborhood] : domain.subdomains() )
        {
            const auto& [local_id, neighborhood] = idx_and_neighborhood;

            for ( const auto& [local_face, neighbor_tuple] : neighborhood.neighborhood_face() )
            {
                const auto& [neighbor_info, neighbor_face, unpack_dirs, neighbor_rank] = neighbor_tuple;

                int neighbor_local_id = -1;
                if ( neighbor_rank == my_rank && domain.subdomains().contains( neighbor_info ) )
                    neighbor_local_id = std::get< 0 >( domain.subdomains().at( neighbor_info ) );

                const auto px      = boundary_position_from_boundary_type_x( local_face );
                const auto py      = boundary_position_from_boundary_type_y( local_face );
                const bool r_normal = ( px == BoundaryPosition::PV && py == BoundaryPosition::PV );
                const int  ni       = n_lat_cells;
                const int  nj       = r_normal ? n_lat_cells : n_rad_cells;

                const int buf_idx = static_cast< int >( all_pairs.size() );
                send_bufs_.emplace_back( "fv_ghost_vec_send", ni, nj, VecDim );
                recv_bufs_.emplace_back( "fv_ghost_vec_recv", ni, nj, VecDim );

                all_pairs.push_back( FacePair{
                    .buf_idx               = buf_idx,
                    .local_rank            = my_rank,
                    .local_subdomain       = subdomain_info,
                    .local_subdomain_id    = local_id,
                    .local_face            = local_face,
                    .neighbor_rank         = neighbor_rank,
                    .neighbor_subdomain    = neighbor_info,
                    .neighbor_subdomain_id = neighbor_local_id,
                    .neighbor_face         = neighbor_face,
                    .dir0                  = std::get< 0 >( unpack_dirs ),
                    .dir1                  = std::get< 1 >( unpack_dirs ),
                } );
            }
        }

        recv_ordered_ = all_pairs;
        std::sort( recv_ordered_.begin(), recv_ordered_.end(), []( const FacePair& a, const FacePair& b ) {
            if ( a.neighbor_subdomain != b.neighbor_subdomain )
                return a.neighbor_subdomain < b.neighbor_subdomain;
            if ( a.neighbor_face != b.neighbor_face )
                return static_cast< int >( a.neighbor_face ) < static_cast< int >( b.neighbor_face );
            if ( a.local_subdomain != b.local_subdomain )
                return a.local_subdomain < b.local_subdomain;
            return static_cast< int >( a.local_face ) < static_cast< int >( b.local_face );
        } );

        send_ordered_ = all_pairs;
        std::sort( send_ordered_.begin(), send_ordered_.end(), []( const FacePair& a, const FacePair& b ) {
            if ( a.local_subdomain != b.local_subdomain )
                return a.local_subdomain < b.local_subdomain;
            if ( a.local_face != b.local_face )
                return static_cast< int >( a.local_face ) < static_cast< int >( b.local_face );
            if ( a.neighbor_subdomain != b.neighbor_subdomain )
                return a.neighbor_subdomain < b.neighbor_subdomain;
            return static_cast< int >( a.neighbor_face ) < static_cast< int >( b.neighbor_face );
        } );
    }
};

/// @brief Fills all ghost layers of a vector-valued FV field from neighbouring subdomains.
///
/// @param domain  The distributed domain.
/// @param data    FV vector field — ghost cells are written in-place.
/// @param bufs    Pre-allocated buffers (constructed once, reused).
template < typename ScalarType, int VecDim >
void update_fv_ghost_layers(
    const grid::shell::DistributedDomain&            domain,
    const grid::Grid4DDataVec< ScalarType, VecDim >& data,
    FVGhostLayerVecBuffers< ScalarType, VecDim >&    bufs )
{
    using namespace fv_detail;

    std::vector< MPI_Request > recv_requests;
    std::vector< MPI_Request > send_requests;

    for ( const auto& fp : bufs.recv_ordered() )
    {
        if ( fp.local_rank == fp.neighbor_rank )
            continue;

        auto& recv_buf = bufs.recv_buf( fp );
        MPI_Request req;
        MPI_Irecv(
            recv_buf.data(),
            static_cast< int >( recv_buf.span() ),
            mpi::mpi_datatype< ScalarType >(),
            fp.neighbor_rank,
            MPI_TAG_FV_VEC_GHOST_LAYERS,
            MPI_COMM_WORLD,
            &req );
        recv_requests.push_back( req );
    }

    for ( const auto& fp : bufs.send_ordered() )
    {
        if ( fp.local_rank == fp.neighbor_rank )
        {
            pack_inner_cells_vec( bufs.recv_buf( fp ), data, fp.neighbor_subdomain_id, fp.neighbor_face );
        }
        else
        {
            auto& send_buf = bufs.send_buf( fp );
            pack_inner_cells_vec( send_buf, data, fp.local_subdomain_id, fp.local_face );

            MPI_Request req;
            MPI_Isend(
                send_buf.data(),
                static_cast< int >( send_buf.span() ),
                mpi::mpi_datatype< ScalarType >(),
                fp.neighbor_rank,
                MPI_TAG_FV_VEC_GHOST_LAYERS,
                MPI_COMM_WORLD,
                &req );
            send_requests.push_back( req );
        }
    }

    MPI_Waitall( static_cast< int >( send_requests.size() ), send_requests.data(), MPI_STATUSES_IGNORE );
    MPI_Waitall( static_cast< int >( recv_requests.size() ), recv_requests.data(), MPI_STATUSES_IGNORE );

    for ( const auto& fp : bufs.recv_ordered() )
    {
        unpack_to_ghost_vec( bufs.recv_buf( fp ), data, fp.local_subdomain_id, fp.local_face, fp.dir0, fp.dir1 );
    }
}

/// @brief Convenience overload for vector fields — allocates temporary buffers internally.
///
/// @note THIS ALLOCATES ON EVERY CALL.  Prefer the overload with FVGhostLayerVecBuffers for
///       any code that runs repeatedly.
template < typename ScalarType, int VecDim >
void update_fv_ghost_layers(
    const grid::shell::DistributedDomain&            domain,
    const grid::Grid4DDataVec< ScalarType, VecDim >& data )
{
    FVGhostLayerVecBuffers< ScalarType, VecDim > tmp_bufs( domain );
    update_fv_ghost_layers( domain, data, tmp_bufs );
}

} // namespace terra::communication::shell
