#pragma once

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "mpi/mpi.hpp"

/// @file
///
/// Width-1 ghost (halo) layers for Q1 nodal subdomain fields.
///
/// The ordinary shell communication (\ref terra::communication::shell) treats subdomain interfaces as *shared*
/// nodes and reduces them additively -- it never makes a neighbouring subdomain's *interior* data available
/// locally. The semi-Lagrangian transport needs exactly that: a departure point may fall into the first cell
/// beyond the local subdomain, and evaluating the Q1 basis there requires the six nodes of that cell.
///
/// A ghosted field of a subdomain block with `n x n x n_r` nodes is stored as `(n+2) x (n+2) x (n_r+2)`, with
/// the owned block at ghosted indices `[1, n]`. Ghost plane `0` (respectively `n+1`) holds the neighbouring
/// subdomain's *depth-1* plane, i.e. the layer just inside its own shared boundary plane -- the shared plane
/// itself is already present in the owned block.
///
/// **Filling edges and corners.** Only the *face* neighbourhood is used. The lateral face exchange is run
/// **twice**: the first pass fills the four lateral face ghosts, and because each pass sends the full extent of
/// the perpendicular directions *including their ghosts*, the second pass propagates a neighbour's ghost rows
/// into our corner regions. The radial exchange then runs once over the full (already ghosted) lateral extent,
/// completing every remaining edge and corner. This avoids having to reason about the edge and vertex
/// neighbourhood tables, whose entries are ambiguous at the twelve pentagonal points of the icosahedral grid.
///
/// @note Ghost regions with no face neighbour (at the CMB and the surface) are never written. Validity is not
///       tracked here -- see \ref terra::fe::wedge::sl::GhostedGeometry, which derives a validity mask
///       geometrically from the ghosted coordinates and therefore also rejects the degenerate pentagonal
///       corners that the sweep fills with a geometrically meaningless value.

namespace terra::fe::wedge::sl
{

/// @brief Ghost layer width. Bounds the admissible per-substep Courant number of the transport.
inline constexpr int ghost_width = 3;

/// @brief Ghosted index of an owned node index.
KOKKOS_INLINE_FUNCTION constexpr int to_ghosted_index( const int owned_index )
{
    return owned_index + ghost_width;
}

namespace detail
{

/// Detects the SoA vector grid data (which carries `vec_dim`) as opposed to a plain scalar Kokkos view.
template < typename V, typename = void >
struct view_vec_dim
{
    static constexpr int  value  = 1;
    static constexpr bool is_vec = false;
};

template < typename V >
struct view_vec_dim< V, std::void_t< decltype( V::vec_dim ) > >
{
    static constexpr int  value  = V::vec_dim;
    static constexpr bool is_vec = true;
};

template < typename V >
KOKKOS_INLINE_FUNCTION auto& view_element( const V& v, const int sd, const int x, const int y, const int r, const int d )
{
    if constexpr ( view_vec_dim< V >::is_vec )
        return v( sd, x, y, r, d );
    else
        return v( sd, x, y, r );
}

/// Index of the ghost plane `depth` outside a boundary at `position`, for `depth` in `1 .. ghost_width`.
KOKKOS_INLINE_FUNCTION constexpr int
    ghost_plane_index( const grid::BoundaryPosition position, const int size_ghosted, const int depth )
{
    return position == grid::BoundaryPosition::P0 ? ghost_width - depth : size_ghosted - 1 - ghost_width + depth;
}

/// Index of the owned plane `depth` inside a boundary at `position` -- the data a neighbour needs for its ghost
/// plane at the same depth, the boundary plane itself being shared and already present on both sides.
KOKKOS_INLINE_FUNCTION constexpr int
    depth_plane_index( const grid::BoundaryPosition position, const int size_ghosted, const int depth )
{
    return position == grid::BoundaryPosition::P0 ? ghost_width + depth : size_ghosted - 1 - ghost_width - depth;
}

KOKKOS_INLINE_FUNCTION constexpr int
    varying_index( const int loop_index, const int size_ghosted, const grid::BoundaryDirection direction )
{
    return direction == grid::BoundaryDirection::FORWARD ? loop_index : size_ghosted - 1 - loop_index;
}

} // namespace detail


/// @brief Packs the depth-`ghost_width` plane at `face` of subdomain `sd`, iterating the varying directions
///        forward -- the buffer index *is* the sender's own coordinate, which is what lets the receiver apply
///        its own unpack ordering.
///
/// @note Free function rather than a member: CUDA does not allow an extended `__host__ __device__` lambda
///       inside a private or protected member function.
template < typename FieldView, typename BufferView >
void pack_face_plane(
    const FieldView&         field,
    const int                subdomain,
    const grid::BoundaryFace face,
    const BufferView&        buffer,
    const int                n0,
    const int                n1,
    const int                num_lat_ghosted,
    const int                num_rad_ghosted,
    const int                depth,
    const size_t             buffer_offset )
{
    constexpr int vec_dim = detail::view_vec_dim< FieldView >::value;

    const auto px = grid::boundary_position_from_boundary_type_x( face );
    const auto py = grid::boundary_position_from_boundary_type_y( face );
    const auto pr = grid::boundary_position_from_boundary_type_r( face );

    Kokkos::parallel_for(
        "sl_ghost_pack",
        Kokkos::MDRangePolicy< Kokkos::Rank< 2 > >( { 0, 0 }, { n0, n1 } ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            int x = 0, y = 0, r = 0;
            if ( px != grid::BoundaryPosition::PV )
            {
                x = detail::depth_plane_index( px, num_lat_ghosted, depth );
                y = i;
                r = j;
            }
            else if ( py != grid::BoundaryPosition::PV )
            {
                x = i;
                y = detail::depth_plane_index( py, num_lat_ghosted, depth );
                r = j;
            }
            else
            {
                x = i;
                y = j;
                r = detail::depth_plane_index( pr, num_rad_ghosted, depth );
            }

            for ( int d = 0; d < vec_dim; ++d )
                buffer( buffer_offset + ( static_cast< size_t >( i ) * n1 + j ) * vec_dim + d ) =
                    detail::view_element( field, subdomain, x, y, r, d );
        } );
}

/// @brief Unpacks a neighbour's depth plane into our ghost plane at `face`, applying the unpack ordering.
template < typename FieldView, typename BufferView >
void unpack_ghost_plane(
    const FieldView&              field,
    const int                     subdomain,
    const grid::BoundaryFace      face,
    const grid::BoundaryDirection d0,
    const grid::BoundaryDirection d1,
    const BufferView&             buffer,
    const int                     n0,
    const int                     n1,
    const int                     num_lat_ghosted,
    const int                     num_rad_ghosted,
    const int                     depth,
    const size_t                  buffer_offset )
{
    constexpr int vec_dim = detail::view_vec_dim< FieldView >::value;

    const auto px = grid::boundary_position_from_boundary_type_x( face );
    const auto py = grid::boundary_position_from_boundary_type_y( face );
    const auto pr = grid::boundary_position_from_boundary_type_r( face );

    Kokkos::parallel_for(
        "sl_ghost_unpack",
        Kokkos::MDRangePolicy< Kokkos::Rank< 2 > >( { 0, 0 }, { n0, n1 } ),
        KOKKOS_LAMBDA( const int i, const int j ) {
            int x = 0, y = 0, r = 0;
            if ( px != grid::BoundaryPosition::PV )
            {
                x = detail::ghost_plane_index( px, num_lat_ghosted, depth );
                y = detail::varying_index( i, num_lat_ghosted, d0 );
                r = detail::varying_index( j, num_rad_ghosted, d1 );
            }
            else if ( py != grid::BoundaryPosition::PV )
            {
                x = detail::varying_index( i, num_lat_ghosted, d0 );
                y = detail::ghost_plane_index( py, num_lat_ghosted, depth );
                r = detail::varying_index( j, num_rad_ghosted, d1 );
            }
            else
            {
                x = detail::varying_index( i, num_lat_ghosted, d0 );
                y = detail::varying_index( j, num_lat_ghosted, d1 );
                r = detail::ghost_plane_index( pr, num_rad_ghosted, depth );
            }

            for ( int d = 0; d < vec_dim; ++d )
                detail::view_element( field, subdomain, x, y, r, d ) =
                    buffer( buffer_offset + ( static_cast< size_t >( i ) * n1 + j ) * vec_dim + d );
        } );
}

/// @brief Precomputed plan for filling width-1 ghost layers of Q1 subdomain fields.
///
/// Construct once per domain and reuse for every field and every timestep.
class GhostExchange
{
  public:
    GhostExchange() = default;

    explicit GhostExchange( const grid::shell::DistributedDomain& domain )
    : comm_( domain.comm() )
    {
        num_subdomains_ = static_cast< int >( domain.subdomains().size() );
        num_lat_        = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        num_rad_        = domain.domain_info().subdomain_num_nodes_radially();
        num_lat_ghost_  = num_lat_ + 2 * ghost_width;
        num_rad_ghost_  = num_rad_ + 2 * ghost_width;

        // Map every subdomain that lives on this rank to its local index, so that a neighbour on the same rank
        // can be served by a straight device-to-device copy instead of a message.
        std::map< int64_t, int > local_id_of;
        for ( const auto& [info, data] : domain.subdomains() )
            local_id_of[info.global_id()] = std::get< 0 >( data );

        for ( const auto& [info, data] : domain.subdomains() )
        {
            const int   my_sd         = std::get< 0 >( data );
            const auto& neighborhood  = std::get< 1 >( data );
            const auto  my_global_id  = info.global_id();

            for ( const auto& [my_face, neighbor] : neighborhood.neighborhood_face() )
            {
                const auto& [nb_info, nb_face, ordering, nb_rank] = neighbor;

                Link link;
                link.my_sd        = my_sd;
                link.my_face      = my_face;
                link.nb_face      = nb_face;
                link.d0           = std::get< 0 >( ordering );
                link.d1           = std::get< 1 >( ordering );
                link.nb_rank      = static_cast< int >( nb_rank );
                link.my_global_id = my_global_id;
                link.nb_global_id = nb_info.global_id();
                link.radial       = grid::is_face_boundary_normal_to_radial_direction( my_face );

                const auto it     = local_id_of.find( link.nb_global_id );
                link.nb_local_sd  = ( it == local_id_of.end() ) ? -1 : it->second;

                buffer_extents( my_face, link.buf_n0, link.buf_n1 );

                links_.push_back( link );
            }
        }

        assign_tags();
    }

    [[nodiscard]] int num_subdomains() const { return num_subdomains_; }
    [[nodiscard]] int num_nodes_lateral() const { return num_lat_; }
    [[nodiscard]] int num_nodes_radial() const { return num_rad_; }
    [[nodiscard]] int num_nodes_lateral_ghosted() const { return num_lat_ghost_; }
    [[nodiscard]] int num_nodes_radial_ghosted() const { return num_rad_ghost_; }

    /// @brief Allocates a ghosted scalar field matching this plan.
    template < typename ScalarType >
    [[nodiscard]] grid::Grid4DDataScalar< ScalarType > allocate_scalar( const std::string& label ) const
    {
        return grid::Grid4DDataScalar< ScalarType >(
            label, num_subdomains_, num_lat_ghost_, num_lat_ghost_, num_rad_ghost_ );
    }

    /// @brief Allocates a ghosted vector field matching this plan.
    template < typename ScalarType, int VecDim >
    [[nodiscard]] grid::Grid4DDataVec< ScalarType, VecDim > allocate_vec( const std::string& label ) const
    {
        return grid::Grid4DDataVec< ScalarType, VecDim >(
            label, num_subdomains_, num_lat_ghost_, num_lat_ghost_, num_rad_ghost_ );
    }

    /// @brief Copies an owned subdomain field into the interior of a ghosted field.
    template < typename SrcView, typename DstView >
    void copy_interior( const SrcView& src, const DstView& dst ) const
    {
        constexpr int vec_dim = detail::view_vec_dim< DstView >::value;

        Kokkos::parallel_for(
            "sl_ghost_copy_interior",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 },
                                                        { num_subdomains_, num_lat_, num_lat_, num_rad_ } ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                for ( int d = 0; d < vec_dim; ++d )
                {
                    detail::view_element(
                        dst, sd, to_ghosted_index( x ), to_ghosted_index( y ), to_ghosted_index( r ), d ) =
                        detail::view_element( src, sd, x, y, r, d );
                }
            } );
        Kokkos::fence();
    }

    /// @brief Fills the ghost layers of `field` (whose interior must already hold the owned data).
    ///
    /// Runs the lateral exchange twice and the radial exchange once; see the file comment.
    template < typename FieldView >
    void exchange( const FieldView& field ) const
    {
        exchange_pass( field, /*radial=*/false );
        exchange_pass( field, /*radial=*/false );
        exchange_pass( field, /*radial=*/true );
    }

    /// @brief Runs only the lateral part of the exchange.
    ///
    /// Used for quantities that are constant along the radial direction (the unit-sphere node directions), for
    /// which the radial pass would be a no-op.
    template < typename FieldView >
    void exchange_lateral( const FieldView& field ) const
    {
        exchange_pass( field, /*radial=*/false );
        exchange_pass( field, /*radial=*/false );
    }

    /// @brief Convenience: interior copy followed by the ghost exchange.
    template < typename SrcView, typename DstView >
    void fill( const SrcView& src, const DstView& dst ) const
    {
        copy_interior( src, dst );
        exchange( dst );
    }

  private:
    struct Link
    {
        int                     my_sd        = 0;
        int                     nb_local_sd  = -1;
        int                     nb_rank      = -1;
        int64_t                 my_global_id = 0;
        int64_t                 nb_global_id = 0;
        grid::BoundaryFace      my_face      = grid::BoundaryFace::F_0YR;
        grid::BoundaryFace      nb_face      = grid::BoundaryFace::F_0YR;
        grid::BoundaryDirection d0           = grid::BoundaryDirection::FORWARD;
        grid::BoundaryDirection d1           = grid::BoundaryDirection::FORWARD;
        bool                    radial       = false;
        int                     buf_n0       = 0;
        int                     buf_n1       = 0;
        int                     send_tag     = 0;
        int                     recv_tag     = 0;
    };

    /// Extents of the 2D buffer for a face: the two "varying" directions, at their *ghosted* sizes.
    void buffer_extents( const grid::BoundaryFace face, int& n0, int& n1 ) const
    {
        const auto px = grid::boundary_position_from_boundary_type_x( face );
        const auto py = grid::boundary_position_from_boundary_type_y( face );

        if ( px != grid::BoundaryPosition::PV )
        {
            n0 = num_lat_ghost_; // y
            n1 = num_rad_ghost_; // r
        }
        else if ( py != grid::BoundaryPosition::PV )
        {
            n0 = num_lat_ghost_; // x
            n1 = num_rad_ghost_; // r
        }
        else
        {
            n0 = num_lat_ghost_; // x
            n1 = num_lat_ghost_; // y
        }
    }

    static int face_code( const grid::BoundaryFace face )
    {
        switch ( face )
        {
        case grid::BoundaryFace::F_XY0: return 0;
        case grid::BoundaryFace::F_XY1: return 1;
        case grid::BoundaryFace::F_X0R: return 2;
        case grid::BoundaryFace::F_X1R: return 3;
        case grid::BoundaryFace::F_0YR: return 4;
        default: return 5;
        }
    }

    /// Both ends of a link must agree on the MPI tag. The key of a message is always
    /// `(sending subdomain, sending face)`, which the receiver knows as `(nb_global_id, nb_face)`. Sorting the
    /// keys per peer rank and per direction therefore yields the same numbering on both ranks.
    void assign_tags()
    {
        std::map< int, std::vector< std::pair< int64_t, int > > > send_keys, recv_keys;

        for ( const auto& link : links_ )
        {
            if ( link.nb_local_sd >= 0 )
                continue;
            send_keys[link.nb_rank].emplace_back( link.my_global_id, face_code( link.my_face ) );
            recv_keys[link.nb_rank].emplace_back( link.nb_global_id, face_code( link.nb_face ) );
        }

        for ( auto& [rank, keys] : send_keys )
            std::sort( keys.begin(), keys.end() );
        for ( auto& [rank, keys] : recv_keys )
            std::sort( keys.begin(), keys.end() );

        for ( auto& link : links_ )
        {
            if ( link.nb_local_sd >= 0 )
                continue;

            const auto& sk = send_keys.at( link.nb_rank );
            const auto& rk = recv_keys.at( link.nb_rank );

            link.send_tag = static_cast< int >(
                std::lower_bound(
                    sk.begin(), sk.end(), std::make_pair( link.my_global_id, face_code( link.my_face ) ) ) -
                sk.begin() );
            link.recv_tag = static_cast< int >(
                std::lower_bound(
                    rk.begin(), rk.end(), std::make_pair( link.nb_global_id, face_code( link.nb_face ) ) ) -
                rk.begin() );
        }
    }

    /// One exchange pass over either the lateral or the radial faces.
    template < typename FieldView >
    void exchange_pass( const FieldView& field, const bool radial ) const
    {
        using ScalarType      = typename FieldView::value_type;
        constexpr int vec_dim = detail::view_vec_dim< FieldView >::value;

        std::vector< const Link* > active;
        for ( const auto& link : links_ )
        {
            if ( link.radial == radial )
                active.push_back( &link );
        }
        if ( active.empty() )
            return;

        // Pack every active link's outgoing plane. Local links are packed too and then unpacked directly from
        // the device buffer, which keeps one code path for the index arithmetic.
        std::vector< Kokkos::View< ScalarType* > > send_buffers( active.size() );
        std::vector< Kokkos::View< ScalarType* > > recv_buffers( active.size() );

        for ( size_t i = 0; i < active.size(); ++i )
        {
            const Link& link = *active[i];
            // One buffer per link carrying every depth, so a wider ghost layer costs a fatter message rather
            // than more of them: the exchange is latency-bound at these sizes.
            const size_t plane = static_cast< size_t >( link.buf_n0 ) * link.buf_n1 * vec_dim;
            const size_t n     = plane * ghost_width;

            send_buffers[i] = Kokkos::View< ScalarType* >( Kokkos::view_alloc( "sl_ghost_send", Kokkos::WithoutInitializing ), n );
            recv_buffers[i] = Kokkos::View< ScalarType* >( Kokkos::view_alloc( "sl_ghost_recv", Kokkos::WithoutInitializing ), n );

            for ( int d = 1; d <= ghost_width; ++d )
                pack_face_plane( field, link.my_sd, link.my_face, send_buffers[i], link.buf_n0,
                                 link.buf_n1, num_lat_ghost_, num_rad_ghost_, d, ( d - 1 ) * plane );
        }
        Kokkos::fence();

        // Local links: the neighbour's outgoing plane is packed by its own link, so we cannot simply reuse a
        // buffer -- pack it here from the neighbour's subdomain instead.
        std::vector< MPI_Request > requests;
        std::vector< Kokkos::View< ScalarType*, Kokkos::HostSpace > > host_send, host_recv;
        host_send.resize( active.size() );
        host_recv.resize( active.size() );

        for ( size_t i = 0; i < active.size(); ++i )
        {
            const Link& link = *active[i];

            if ( link.nb_local_sd >= 0 )
            {
                // Same rank: pack the neighbour's depth planes directly into our receive buffer.
                const size_t plane = static_cast< size_t >( link.buf_n0 ) * link.buf_n1 * vec_dim;
                for ( int d = 1; d <= ghost_width; ++d )
                    pack_face_plane( field, link.nb_local_sd, link.nb_face, recv_buffers[i], link.buf_n0,
                                     link.buf_n1, num_lat_ghost_, num_rad_ghost_, d, ( d - 1 ) * plane );
                continue;
            }

            const size_t n = send_buffers[i].extent( 0 );

            host_send[i] = Kokkos::View< ScalarType*, Kokkos::HostSpace >(
                Kokkos::view_alloc( "sl_ghost_send_h", Kokkos::WithoutInitializing ), n );
            host_recv[i] = Kokkos::View< ScalarType*, Kokkos::HostSpace >(
                Kokkos::view_alloc( "sl_ghost_recv_h", Kokkos::WithoutInitializing ), n );

            Kokkos::deep_copy( host_send[i], send_buffers[i] );

            MPI_Request req_recv, req_send;
            MPI_Irecv( host_recv[i].data(), static_cast< int >( n ), mpi_type< ScalarType >(), link.nb_rank,
                       link.recv_tag, comm_, &req_recv );
            MPI_Isend( host_send[i].data(), static_cast< int >( n ), mpi_type< ScalarType >(), link.nb_rank,
                       link.send_tag, comm_, &req_send );
            requests.push_back( req_recv );
            requests.push_back( req_send );
        }
        Kokkos::fence();

        if ( !requests.empty() )
            MPI_Waitall( static_cast< int >( requests.size() ), requests.data(), MPI_STATUSES_IGNORE );

        for ( size_t i = 0; i < active.size(); ++i )
        {
            const Link& link = *active[i];
            if ( link.nb_local_sd < 0 )
                Kokkos::deep_copy( recv_buffers[i], host_recv[i] );
            const size_t plane = static_cast< size_t >( link.buf_n0 ) * link.buf_n1 * vec_dim;
            for ( int d = 1; d <= ghost_width; ++d )
                unpack_ghost_plane( field, link.my_sd, link.my_face, link.d0, link.d1, recv_buffers[i],
                                    link.buf_n0, link.buf_n1, num_lat_ghost_, num_rad_ghost_, d,
                                    ( d - 1 ) * plane );
        }
        Kokkos::fence();
    }

    template < typename ScalarType >
    static MPI_Datatype mpi_type()
    {
        if constexpr ( std::is_same_v< ScalarType, double > )
            return MPI_DOUBLE;
        else if constexpr ( std::is_same_v< ScalarType, float > )
            return MPI_FLOAT;
        else
            return MPI_BYTE;
    }

    MPI_Comm          comm_           = MPI_COMM_WORLD;
    int               num_subdomains_ = 0;
    int               num_lat_        = 0;
    int               num_rad_        = 0;
    int               num_lat_ghost_  = 0;
    int               num_rad_ghost_  = 0;
    std::vector< Link > links_;
};

} // namespace terra::fe::wedge::sl
