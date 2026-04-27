#pragma once

#include "grid/grid_types.hpp"

namespace terra::communication {

/// @brief Communication reduction modes.
enum class CommunicationReduction
{
    /// Sums up the node values during receive.
    SUM,

    /// Stores the min of all received values during receive.
    MIN,

    /// Stores the max of all received values during receive.
    MAX,
};

namespace detail {

/// @brief Helper function to defer to the respective Kokkos::atomic_xxx() reduction function.
template < typename T >
KOKKOS_INLINE_FUNCTION void reduction_function( T* ptr, const T& val, const CommunicationReduction reduction_type )
{
    if ( reduction_type == CommunicationReduction::SUM )
    {
        Kokkos::atomic_add( ptr, val );
    }
    else if ( reduction_type == CommunicationReduction::MIN )
    {
        Kokkos::atomic_min( ptr, val );
    }
    else if ( reduction_type == CommunicationReduction::MAX )
    {
        Kokkos::atomic_max( ptr, val );
    }
}

} // namespace detail

namespace detail {
template < typename DataView, bool is_scalar >
KOKKOS_INLINE_FUNCTION DataView::value_type
                       value( const DataView& data, int local_subdomain_id, int x, int y, int r, int d )
{
    if constexpr ( is_scalar )
    {
        return data( local_subdomain_id, x, y, r );
    }
    else
    {
        return data( local_subdomain_id, x, y, r, d );
    }
}

template < typename DataView, bool is_scalar >
KOKKOS_INLINE_FUNCTION DataView::value_type&
                       value_ref( const DataView& data, int local_subdomain_id, int x, int y, int r, int d )
{
    if constexpr ( is_scalar )
    {
        return data( local_subdomain_id, x, y, r );
    }
    else
    {
        return data( local_subdomain_id, x, y, r, d );
    }
}

KOKKOS_INLINE_FUNCTION
constexpr int
    idx( const int                     loop_idx,
         const int                     size,
         const grid::BoundaryPosition  position,
         const grid::BoundaryDirection direction )
{
    if ( position == grid::BoundaryPosition::P0 )
    {
        return 0;
    }
    else if ( position == grid::BoundaryPosition::P1 )
    {
        return size - 1;
    }
    else
    {
        if ( direction == grid::BoundaryDirection::FORWARD )
        {
            return loop_idx;
        }
        else
        {
            return size - 1 - loop_idx;
        }
    }
}

} // namespace detail

namespace detail_view_constraints {

template < class T >
struct is_kokkos_view : std::false_type
{};

template < class... Args >
struct is_kokkos_view< Kokkos::View< Args... > > : std::true_type
{};

template < class V >
constexpr bool is_kokkos_view_v = is_kokkos_view< std::decay_t< V > >::value;

} // namespace detail_view_constraints

// ---------------------------
// Generic 0D copy_to_buffer
// buffer: rank-1 view of length VecDim
// ---------------------------
template < int VecDim, typename BufferView, typename ViewType >
std::enable_if_t< detail_view_constraints::is_kokkos_view_v< BufferView > && ( std::decay_t< BufferView >::rank == 1 ) >
    copy_to_buffer(
        const BufferView&          buffer,
        const ViewType&            data,
        const int                  local_subdomain_id,
        const grid::BoundaryVertex boundary_vertex )
{
    // Heuristic: scalar grid data is rank-4 (sd,x,y,r), vector-valued is rank-5 (sd,x,y,r,d)
    static_assert(
        std::decay_t< ViewType >::rank == 4 || std::decay_t< ViewType >::rank == 5,
        "copy_to_buffer expects ViewType rank 4 (scalar) or 5 (vector-valued)." );

    constexpr bool is_scalar = ( std::decay_t< ViewType >::rank == 4 );

    if ( buffer.extent( 0 ) != VecDim )
        Kokkos::abort( "The buffer VecDim should match its respective extent. This abort should not happen." );

    const auto boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary_vertex );
    const auto boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary_vertex );
    const auto boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary_vertex );

    const auto size_x = data.extent( 1 );
    const auto size_y = data.extent( 2 );
    const auto size_r = data.extent( 3 );

    Kokkos::parallel_for(
        "copy_to_buffer_0D", Kokkos::RangePolicy( 0, buffer.extent( 0 ) ), KOKKOS_LAMBDA( const int d ) {
            auto x      = detail::idx( 0, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
            auto y      = detail::idx( 0, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
            auto r      = detail::idx( 0, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            buffer( d ) = detail::value< ViewType, is_scalar >( data, local_subdomain_id, x, y, r, d );
        } );
}

// ---------------------------
// Generic 1D copy_to_buffer
// buffer: rank-2 view of shape (N, VecDim)
// ---------------------------
template < int VecDim, typename BufferView, typename ViewType >
std::enable_if_t< detail_view_constraints::is_kokkos_view_v< BufferView > && ( std::decay_t< BufferView >::rank == 2 ) >
    copy_to_buffer(
        const BufferView&        buffer,
        const ViewType&          data,
        const int                local_subdomain_id,
        const grid::BoundaryEdge boundary_edge )
{
    static_assert(
        std::decay_t< ViewType >::rank == 4 || std::decay_t< ViewType >::rank == 5,
        "copy_to_buffer expects ViewType rank 4 (scalar) or 5 (vector-valued)." );

    constexpr bool is_scalar = ( std::decay_t< ViewType >::rank == 4 );

    if ( buffer.extent( 1 ) != VecDim )
        Kokkos::abort( "The buffer VecDim should match its respective extent. This abort should not happen." );

    const auto boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary_edge );
    const auto boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary_edge );
    const auto boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary_edge );

    const auto size_x = data.extent( 1 );
    const auto size_y = data.extent( 2 );
    const auto size_r = data.extent( 3 );

    Kokkos::parallel_for(
        "copy_to_buffer_1D",
        Kokkos::MDRangePolicy( { 0, 0 }, { buffer.extent( 0 ), buffer.extent( 1 ) } ),
        KOKKOS_LAMBDA( const int idx, const int d ) {
            auto x           = detail::idx( idx, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
            auto y           = detail::idx( idx, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
            auto r           = detail::idx( idx, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            buffer( idx, d ) = detail::value< ViewType, is_scalar >( data, local_subdomain_id, x, y, r, d );
        } );
}

// ---------------------------
// Generic 2D copy_to_buffer
// buffer: rank-3 view of shape (Ni, Nj, VecDim)
// ---------------------------
template < int VecDim, typename BufferView, typename ViewType >
std::enable_if_t< detail_view_constraints::is_kokkos_view_v< BufferView > && ( std::decay_t< BufferView >::rank == 3 ) >
    copy_to_buffer(
        const BufferView&        buffer,
        const ViewType&          data,
        const int                local_subdomain_id,
        const grid::BoundaryFace boundary_face )
{
    static_assert(
        std::decay_t< ViewType >::rank == 4 || std::decay_t< ViewType >::rank == 5,
        "copy_to_buffer expects ViewType rank 4 (scalar) or 5 (vector-valued)." );

    constexpr bool is_scalar = ( std::decay_t< ViewType >::rank == 4 );

    if ( buffer.extent( 2 ) != VecDim )
        Kokkos::abort( "The buffer VecDim should match its respective extent. This abort should not happen." );

    const auto boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary_face );
    const auto boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary_face );
    const auto boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary_face );

    const auto size_x = data.extent( 1 );
    const auto size_y = data.extent( 2 );
    const auto size_r = data.extent( 3 );

    Kokkos::parallel_for(
        "copy_to_buffer_2D",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { buffer.extent( 0 ), buffer.extent( 1 ), buffer.extent( 2 ) } ),
        KOKKOS_LAMBDA( const int i, const int j, const int d ) {
            int x = 0, y = 0, r = 0;

            if ( boundary_position_x != grid::BoundaryPosition::PV )
            {
                x = detail::idx( 0, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
                y = detail::idx( i, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
                r = detail::idx( j, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            }
            else if ( boundary_position_y != grid::BoundaryPosition::PV )
            {
                x = detail::idx( i, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
                y = detail::idx( 0, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
                r = detail::idx( j, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            }
            else
            {
                x = detail::idx( i, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
                y = detail::idx( j, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
                r = detail::idx( 0, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            }

            buffer( i, j, d ) = detail::value< ViewType, is_scalar >( data, local_subdomain_id, x, y, r, d );
        } );
}

template < typename ScalarType, int VecDim, typename ViewType >
void copy_to_buffer(
    const grid::Grid0DDataVec< ScalarType, VecDim >& buffer,
    const ViewType&                                  data,
    const int                                        local_subdomain_id,
    const grid::BoundaryVertex                       boundary_vertex )
{
    static_assert(
        std::is_same_v< ViewType, grid::Grid4DDataScalar< ScalarType > > ||
        std::is_same_v< ViewType, grid::Grid4DDataVec< ScalarType, VecDim > > );

    constexpr bool is_scalar = std::is_same_v< ViewType, grid::Grid4DDataScalar< ScalarType > >;

    if ( buffer.extent( 0 ) != VecDim )
    {
        Kokkos::abort( "The buffer VecDim should match its respective extent. This abort should not happen." );
    }

    const auto boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary_vertex );
    const auto boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary_vertex );
    const auto boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary_vertex );

    const auto size_x = data.extent( 1 );
    const auto size_y = data.extent( 2 );
    const auto size_r = data.extent( 3 );

    Kokkos::parallel_for(
        "copy_to_buffer_0D", Kokkos::RangePolicy( 0, buffer.extent( 0 ) ), KOKKOS_LAMBDA( const int d ) {
            auto x      = detail::idx( 0, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
            auto y      = detail::idx( 0, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
            auto r      = detail::idx( 0, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            buffer( d ) = detail::value< ViewType, is_scalar >( data, local_subdomain_id, x, y, r, d );
        } );
}

template < typename ScalarType, int VecDim, typename ViewType >
void copy_to_buffer(
    const grid::Grid1DDataVec< ScalarType, VecDim >& buffer,
    const ViewType&                                  data,
    const int                                        local_subdomain_id,
    const grid::BoundaryEdge                         boundary_edge )
{
    static_assert(
        std::is_same_v< ViewType, grid::Grid4DDataScalar< ScalarType > > ||
        std::is_same_v< ViewType, grid::Grid4DDataVec< ScalarType, VecDim > > );

    constexpr bool is_scalar = std::is_same_v< ViewType, grid::Grid4DDataScalar< ScalarType > >;

    if ( buffer.extent( 1 ) != VecDim )
    {
        Kokkos::abort( "The buffer VecDim should match its respective extent. This abort should not happen." );
    }

    const auto boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary_edge );
    const auto boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary_edge );
    const auto boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary_edge );

    const auto size_x = data.extent( 1 );
    const auto size_y = data.extent( 2 );
    const auto size_r = data.extent( 3 );

    Kokkos::parallel_for(
        "copy_to_buffer_1D",
        Kokkos::MDRangePolicy( { 0, 0 }, { buffer.extent( 0 ), buffer.extent( 1 ) } ),
        KOKKOS_LAMBDA( const int idx, const int d ) {
            auto x           = detail::idx( idx, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
            auto y           = detail::idx( idx, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
            auto r           = detail::idx( idx, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            buffer( idx, d ) = detail::value< ViewType, is_scalar >( data, local_subdomain_id, x, y, r, d );
        } );
}

template < typename ScalarType, int VecDim, typename ViewType >
void copy_to_buffer(
    const grid::Grid2DDataVec< ScalarType, VecDim >& buffer,
    const ViewType&                                  data,
    const int                                        local_subdomain_id,
    const grid::BoundaryFace                         boundary_face )
{
    static_assert(
        std::is_same_v< ViewType, grid::Grid4DDataScalar< ScalarType > > ||
        std::is_same_v< ViewType, grid::Grid4DDataVec< ScalarType, VecDim > > );

    constexpr bool is_scalar = std::is_same_v< ViewType, grid::Grid4DDataScalar< ScalarType > >;

    if ( buffer.extent( 2 ) != VecDim )
    {
        Kokkos::abort( "The buffer VecDim should match its respective extent. This abort should not happen." );
    }

    const auto boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary_face );
    const auto boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary_face );
    const auto boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary_face );

    const auto size_x = data.extent( 1 );
    const auto size_y = data.extent( 2 );
    const auto size_r = data.extent( 3 );

    Kokkos::parallel_for(
        "copy_to_buffer_2D",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { buffer.extent( 0 ), buffer.extent( 1 ), buffer.extent( 2 ) } ),
        KOKKOS_LAMBDA( const int i, const int j, const int d ) {
            int x = 0;
            int y = 0;
            int r = 0;

            if ( boundary_position_x != grid::BoundaryPosition::PV )
            {
                x = detail::idx(
                    0 /* can be set to anything */, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
                y = detail::idx( i, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
                r = detail::idx( j, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            }
            else if ( boundary_position_y != grid::BoundaryPosition::PV )
            {
                x = detail::idx( i, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
                y = detail::idx(
                    0 /* can be set to anything */, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
                r = detail::idx( j, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            }
            else
            {
                x = detail::idx( i, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
                y = detail::idx( j, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
                r = detail::idx(
                    0 /* can be set to anything */, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            }

            buffer( i, j, d ) = detail::value< ViewType, is_scalar >( data, local_subdomain_id, x, y, r, d );
        } );
}

template < typename BufferView, typename ViewType >
void copy_from_buffer_rotate_and_reduce(
    const BufferView&            buffer,
    const ViewType&              data,
    const int                    local_subdomain_id,
    const grid::BoundaryVertex   boundary_vertex,
    const CommunicationReduction reduction )
{
    using ScalarType             = typename BufferView::value_type;
    constexpr bool is_scalar = std::is_same_v< ViewType, grid::Grid4DDataScalar< ScalarType > >;

    const auto boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary_vertex );
    const auto boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary_vertex );
    const auto boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary_vertex );

    const auto size_x = data.extent( 1 );
    const auto size_y = data.extent( 2 );
    const auto size_r = data.extent( 3 );

    Kokkos::parallel_for(
        "copy_from_buffer_0D", Kokkos::RangePolicy( 0, buffer.extent( 0 ) ), KOKKOS_LAMBDA( const int d ) {
            auto x = detail::idx( 0, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
            auto y = detail::idx( 0, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
            auto r = detail::idx( 0, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            detail::reduction_function(
                &detail::value_ref< ViewType, is_scalar >( data, local_subdomain_id, x, y, r, d ),
                buffer( d ),
                reduction );
        } );
}

template < typename BufferView, typename ViewType >
void copy_from_buffer_rotate_and_reduce(
    const BufferView&             buffer,
    const ViewType&               data,
    const int                     local_subdomain_id,
    const grid::BoundaryEdge      boundary_edge,
    const grid::BoundaryDirection boundary_direction,
    const CommunicationReduction  reduction )
{
    constexpr bool is_scalar = std::is_same_v< ViewType, grid::Grid4DDataScalar< ScalarType > >;

    if ( buffer.extent( 1 ) != VecDim )
    {
        Kokkos::abort( "The buffer VecDim should match its respective extent. This abort should not happen." );
    }

    const auto boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary_edge );
    const auto boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary_edge );
    const auto boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary_edge );

    const auto size_x = data.extent( 1 );
    const auto size_y = data.extent( 2 );
    const auto size_r = data.extent( 3 );

    Kokkos::parallel_for(
        "copy_from_buffer_1D",
        Kokkos::MDRangePolicy( { 0, 0 }, { buffer.extent( 0 ), buffer.extent( 1 ) } ),
        KOKKOS_LAMBDA( const int idx, const int d ) {
            auto x = detail::idx( idx, size_x, boundary_position_x, boundary_direction );
            auto y = detail::idx( idx, size_y, boundary_position_y, boundary_direction );
            auto r = detail::idx( idx, size_r, boundary_position_r, boundary_direction );
            detail::reduction_function(
                &detail::value_ref< ViewType, is_scalar >( data, local_subdomain_id, x, y, r, d ),
                buffer( idx, d ),
                reduction );
        } );
}

template < typename BufferView, typename ViewType >
void copy_from_buffer_rotate_and_reduce(
    const BufferView&                                                    buffer,
    const ViewType&                                                      data,
    const int                                                            local_subdomain_id,
    const grid::BoundaryFace                                             boundary_face,
    const std::tuple< grid::BoundaryDirection, grid::BoundaryDirection > boundary_directions,
    const CommunicationReduction                                         reduction )
{
    using ScalarType             = typename BufferView::value_type;
    constexpr bool is_scalar = std::is_same_v< ViewType, grid::Grid4DDataScalar< ScalarType > >;

    const auto boundary_position_x = grid::boundary_position_from_boundary_type_x( boundary_face );
    const auto boundary_position_y = grid::boundary_position_from_boundary_type_y( boundary_face );
    const auto boundary_position_r = grid::boundary_position_from_boundary_type_r( boundary_face );

    const auto size_x = data.extent( 1 );
    const auto size_y = data.extent( 2 );
    const auto size_r = data.extent( 3 );

    const auto boundary_direction_0 = std::get< 0 >( boundary_directions );
    const auto boundary_direction_1 = std::get< 1 >( boundary_directions );

    Kokkos::parallel_for(
        "copy_from_buffer_2D",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { buffer.extent( 0 ), buffer.extent( 1 ), buffer.extent( 2 ) } ),
        KOKKOS_LAMBDA( const int i, const int j, const int d ) {
            int x = 0;
            int y = 0;
            int r = 0;

            if ( boundary_position_x != grid::BoundaryPosition::PV )
            {
                x = detail::idx( 0, size_x, boundary_position_x, grid::BoundaryDirection::FORWARD );
                y = detail::idx( i, size_y, boundary_position_y, boundary_direction_0 );
                r = detail::idx( j, size_r, boundary_position_r, boundary_direction_1 );
            }
            else if ( boundary_position_y != grid::BoundaryPosition::PV )
            {
                x = detail::idx( i, size_x, boundary_position_x, boundary_direction_0 );
                y = detail::idx( 0, size_y, boundary_position_y, grid::BoundaryDirection::FORWARD );
                r = detail::idx( j, size_r, boundary_position_r, boundary_direction_1 );
            }
            else
            {
                x = detail::idx( i, size_x, boundary_position_x, boundary_direction_0 );
                y = detail::idx( j, size_y, boundary_position_y, boundary_direction_1 );
                r = detail::idx( 0, size_r, boundary_position_r, grid::BoundaryDirection::FORWARD );
            }

            detail::reduction_function(
                &detail::value_ref< ViewType, is_scalar >( data, local_subdomain_id, x, y, r, d ),
                buffer( i, j, d ),
                reduction );
        } );
}

} // namespace terra::communication