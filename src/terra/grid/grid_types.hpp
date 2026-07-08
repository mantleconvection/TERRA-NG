
#pragma once

#include "../kokkos/kokkos_wrapper.hpp"
#include "../types.hpp"
#include "dense/mat.hpp"

#include <string>

namespace terra::grid {

using Layout = Kokkos::LayoutRight;

template < typename ScalarType >
using Grid0DDataScalar = Kokkos::View< ScalarType, Layout >;

template < typename ScalarType >
using Grid1DDataScalar = Kokkos::View< ScalarType*, Layout >;

template < typename ScalarType >
using Grid2DDataScalar = Kokkos::View< ScalarType**, Layout >;

template < typename ScalarType >
using Grid3DDataScalar = Kokkos::View< ScalarType***, Layout >;

template < typename ScalarType >
using Grid4DDataScalar = Kokkos::View< ScalarType****, Layout >;

template < typename ScalarType >
using Grid5DDataScalar = Kokkos::View< ScalarType*****, Layout >;

template < typename ScalarType, int VecDim >
using Grid0DDataVec = Kokkos::View< ScalarType[VecDim], Layout >;

template < typename ScalarType, int VecDim >
using Grid1DDataVec = Kokkos::View< ScalarType* [VecDim], Layout >;

template < typename ScalarType, int VecDim >
using Grid2DDataVec = Kokkos::View< ScalarType** [VecDim], Layout >;

template < typename ScalarType, int VecDim >
using Grid3DDataVec = Kokkos::View< ScalarType*** [VecDim], Layout >;

/// @brief SoA (Structure-of-Arrays) 4D vector grid data.
///
/// Stores VecDim separate Grid4DDataScalar views for GPU memory coalescing.
/// Provides the same `operator()(sd, x, y, r, d)` interface as the former
/// AoS Kokkos::View<ScalarType****[VecDim]>.
template < typename ScalarType, int VecDim >
struct Grid4DDataVec
{
    using value_type                  = ScalarType;
    using memory_space                = typename Grid4DDataScalar< ScalarType >::memory_space;
    static constexpr int vec_dim      = VecDim;
    static constexpr int rank         = 5;

    Grid4DDataScalar< ScalarType > comp_[VecDim];

    Grid4DDataVec() = default;

    Grid4DDataVec( const std::string& label, int s0, int s1, int s2, int s3 )
    {
        for ( int d = 0; d < VecDim; ++d )
            comp_[d] = Grid4DDataScalar< ScalarType >(
                label + "_d" + std::to_string( d ), s0, s1, s2, s3 );
    }

    KOKKOS_INLINE_FUNCTION
    ScalarType& operator()( int i0, int i1, int i2, int i3, int d ) const
    {
        return comp_[d]( i0, i1, i2, i3 );
    }

    KOKKOS_INLINE_FUNCTION
    auto extent( int i ) const
    {
        if ( i < 4 )
            return comp_[0].extent( i );
        return static_cast< decltype( comp_[0].extent( 0 ) ) >( VecDim );
    }

    /// @brief Get the label (derived from first component by stripping "_d0" suffix).
    std::string label() const
    {
        std::string l   = comp_[0].label();
        auto        pos = l.rfind( "_d0" );
        return ( pos != std::string::npos ) ? l.substr( 0, pos ) : l;
    }

    /// @brief Host mirror type for I/O.
    struct host_mirror_type
    {
        using value_type = ScalarType;
        typename Grid4DDataScalar< ScalarType >::host_mirror_type comp_[VecDim];

        ScalarType& operator()( int i0, int i1, int i2, int i3, int d )
        {
            return comp_[d]( i0, i1, i2, i3 );
        }

        const ScalarType& operator()( int i0, int i1, int i2, int i3, int d ) const
        {
            return comp_[d]( i0, i1, i2, i3 );
        }

        auto extent( int i ) const
        {
            if ( i < 4 )
                return comp_[0].extent( i );
            return static_cast< decltype( comp_[0].extent( 0 ) ) >( VecDim );
        }
    };
};

/// @brief Create a host mirror for Grid4DDataVec.
template < typename ScalarType, int VecDim >
typename Grid4DDataVec< ScalarType, VecDim >::host_mirror_type
    create_mirror( Kokkos::HostSpace space, const Grid4DDataVec< ScalarType, VecDim >& src )
{
    typename Grid4DDataVec< ScalarType, VecDim >::host_mirror_type result;
    for ( int d = 0; d < VecDim; ++d )
        result.comp_[d] = Kokkos::create_mirror( space, src.comp_[d] );
    return result;
}

/// @brief Create a host mirror for Grid4DDataScalar (delegates to Kokkos).
template < typename ScalarType >
typename Grid4DDataScalar< ScalarType >::host_mirror_type
    create_mirror( Kokkos::HostSpace space, const Grid4DDataScalar< ScalarType >& src )
{
    return Kokkos::create_mirror( space, src );
}

/// @brief Deep copy from device Grid4DDataVec to host mirror.
template < typename ScalarType, int VecDim >
void deep_copy(
    typename Grid4DDataVec< ScalarType, VecDim >::host_mirror_type& dst,
    const Grid4DDataVec< ScalarType, VecDim >&                src )
{
    for ( int d = 0; d < VecDim; ++d )
        Kokkos::deep_copy( dst.comp_[d], src.comp_[d] );
}

/// @brief Deep copy from host mirror to device Grid4DDataVec.
template < typename ScalarType, int VecDim >
void deep_copy(
    Grid4DDataVec< ScalarType, VecDim >&                             dst,
    const typename Grid4DDataVec< ScalarType, VecDim >::host_mirror_type&  src )
{
    for ( int d = 0; d < VecDim; ++d )
        Kokkos::deep_copy( dst.comp_[d], src.comp_[d] );
}

/// @brief Deep copy for Grid4DDataScalar host mirror to device (delegates to Kokkos).
template < typename ScalarType >
void deep_copy(
    Grid4DDataScalar< ScalarType >&                                  dst,
    const typename Grid4DDataScalar< ScalarType >::host_mirror_type&       src )
{
    Kokkos::deep_copy( dst, src );
}

/// @brief Deep copy for Grid4DDataScalar device to host mirror (delegates to Kokkos).
template < typename ScalarType >
void deep_copy(
    typename Grid4DDataScalar< ScalarType >::host_mirror_type&  dst,
    const Grid4DDataScalar< ScalarType >&                 src )
{
    Kokkos::deep_copy( dst, src );
}

template < typename ScalarType, int Rows, int Cols, int NumMatrices >
using Grid4DDataMatrices = Kokkos::View< dense::Mat< ScalarType, Rows, Cols >****[NumMatrices], Layout >;

template < typename GridDataType >
constexpr int grid_data_vec_dim()
{
    if constexpr (
        std::is_same_v< GridDataType, Grid0DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v< GridDataType, Grid1DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v< GridDataType, Grid2DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v< GridDataType, Grid3DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v< GridDataType, Grid4DDataScalar< typename GridDataType::value_type > > )
    {
        return 1;
    }

    else if constexpr (
        std::is_same_v< GridDataType, Grid0DDataVec< typename GridDataType::value_type, 1 > > ||
        std::is_same_v< GridDataType, Grid1DDataVec< typename GridDataType::value_type, 1 > > ||
        std::is_same_v< GridDataType, Grid2DDataVec< typename GridDataType::value_type, 1 > > ||
        std::is_same_v< GridDataType, Grid3DDataVec< typename GridDataType::value_type, 1 > > ||
        std::is_same_v< GridDataType, Grid4DDataVec< typename GridDataType::value_type, 1 > > )
    {
        return 1;
    }

    else if constexpr (
        std::is_same_v< GridDataType, Grid0DDataVec< typename GridDataType::value_type, 2 > > ||
        std::is_same_v< GridDataType, Grid1DDataVec< typename GridDataType::value_type, 2 > > ||
        std::is_same_v< GridDataType, Grid2DDataVec< typename GridDataType::value_type, 2 > > ||
        std::is_same_v< GridDataType, Grid3DDataVec< typename GridDataType::value_type, 2 > > ||
        std::is_same_v< GridDataType, Grid4DDataVec< typename GridDataType::value_type, 2 > > )
    {
        return 2;
    }

    else if constexpr (
        std::is_same_v< GridDataType, Grid0DDataVec< typename GridDataType::value_type, 3 > > ||
        std::is_same_v< GridDataType, Grid1DDataVec< typename GridDataType::value_type, 3 > > ||
        std::is_same_v< GridDataType, Grid2DDataVec< typename GridDataType::value_type, 3 > > ||
        std::is_same_v< GridDataType, Grid3DDataVec< typename GridDataType::value_type, 3 > > ||
        std::is_same_v< GridDataType, Grid4DDataVec< typename GridDataType::value_type, 3 > > )
    {
        return 3;
    }

    return -1;
}

/// @brief Enum for encoding the boundary type tuples (in \ref BoundaryVertex, \ref BoundaryEdge, \ref BoundaryFace).
enum class BoundaryPosition : int
{
    /// start (`== 0`)
    P0 = 0,

    /// end (`== size - 1`)
    P1 = 1,

    /// variable
    PV = 2
};

constexpr int boundary_position_encoding( const BoundaryPosition x, const BoundaryPosition y, const BoundaryPosition r )
{
    return ( static_cast< int >( x ) << 4 ) | ( static_cast< int >( y ) << 2 ) | ( static_cast< int >( r ) << 0 );
}

/// @brief Enum for identification of the 8 boundary vertices of a subdomain.
///
/// @code
/// V_011
/// => x = 0,
///    y = size - 1,
///    r = size - 1
/// @endcode
enum class BoundaryVertex : int
{
    // (x=0, y=0, r=0)
    V_000 = boundary_position_encoding( BoundaryPosition::P0, BoundaryPosition::P0, BoundaryPosition::P0 ),
    V_100 = boundary_position_encoding( BoundaryPosition::P1, BoundaryPosition::P0, BoundaryPosition::P0 ),
    V_010 = boundary_position_encoding( BoundaryPosition::P0, BoundaryPosition::P1, BoundaryPosition::P0 ),
    V_110 = boundary_position_encoding( BoundaryPosition::P1, BoundaryPosition::P1, BoundaryPosition::P0 ),
    V_001 = boundary_position_encoding( BoundaryPosition::P0, BoundaryPosition::P0, BoundaryPosition::P1 ),
    V_101 = boundary_position_encoding( BoundaryPosition::P1, BoundaryPosition::P0, BoundaryPosition::P1 ),
    V_011 = boundary_position_encoding( BoundaryPosition::P0, BoundaryPosition::P1, BoundaryPosition::P1 ),
    V_111 = boundary_position_encoding( BoundaryPosition::P1, BoundaryPosition::P1, BoundaryPosition::P1 ),
};

/// @brief Enum for identification of the 12 boundary edges of a subdomain.
///
/// @code
/// E_1Y0
/// => x = size - 1,
///    y = variable,
///    r = 0
/// @endcode
enum class BoundaryEdge : int
{
    // edge along x, y=0, r=0, (:,BoundaryPosition::START,BoundaryPosition::START) in slice notation
    E_X00 = boundary_position_encoding( BoundaryPosition::PV, BoundaryPosition::P0, BoundaryPosition::P0 ),
    E_X10 = boundary_position_encoding( BoundaryPosition::PV, BoundaryPosition::P1, BoundaryPosition::P0 ),
    E_X01 = boundary_position_encoding( BoundaryPosition::PV, BoundaryPosition::P0, BoundaryPosition::P1 ),
    E_X11 = boundary_position_encoding( BoundaryPosition::PV, BoundaryPosition::P1, BoundaryPosition::P1 ),

    // (0, :,BoundaryPosition::START) in slice notation
    E_0Y0 = boundary_position_encoding( BoundaryPosition::P0, BoundaryPosition::PV, BoundaryPosition::P0 ),
    E_1Y0 = boundary_position_encoding( BoundaryPosition::P1, BoundaryPosition::PV, BoundaryPosition::P0 ),
    E_0Y1 = boundary_position_encoding( BoundaryPosition::P0, BoundaryPosition::PV, BoundaryPosition::P1 ),
    E_1Y1 = boundary_position_encoding( BoundaryPosition::P1, BoundaryPosition::PV, BoundaryPosition::P1 ),

    E_00R = boundary_position_encoding( BoundaryPosition::P0, BoundaryPosition::P0, BoundaryPosition::PV ),
    E_10R = boundary_position_encoding( BoundaryPosition::P1, BoundaryPosition::P0, BoundaryPosition::PV ),
    E_01R = boundary_position_encoding( BoundaryPosition::P0, BoundaryPosition::P1, BoundaryPosition::PV ),
    E_11R = boundary_position_encoding( BoundaryPosition::P1, BoundaryPosition::P1, BoundaryPosition::PV ),
};

/// @brief Enum for identification of the 6 boundary faces of a subdomain.
///
/// @code
/// F_X1R
/// => x = variable,
///    y = size - 1,
///    r = variable
/// @endcode
enum class BoundaryFace : int
{
    // facet orthogonal to r, r=0
    F_XY0 = boundary_position_encoding( BoundaryPosition::PV, BoundaryPosition::PV, BoundaryPosition::P0 ),
    F_XY1 = boundary_position_encoding( BoundaryPosition::PV, BoundaryPosition::PV, BoundaryPosition::P1 ),

    F_X0R = boundary_position_encoding( BoundaryPosition::PV, BoundaryPosition::P0, BoundaryPosition::PV ),
    F_X1R = boundary_position_encoding( BoundaryPosition::PV, BoundaryPosition::P1, BoundaryPosition::PV ),

    F_0YR = boundary_position_encoding( BoundaryPosition::P0, BoundaryPosition::PV, BoundaryPosition::PV ),
    F_1YR = boundary_position_encoding( BoundaryPosition::P1, BoundaryPosition::PV, BoundaryPosition::PV ),
};

/// @brief Enum for the iteration direction at a boundary.
enum class BoundaryDirection : int
{
    FORWARD = 0,
    BACKWARD
};

template < typename BoundaryType >
constexpr BoundaryPosition boundary_position_from_boundary_type_x( const BoundaryType& boundary_type )
{
    static_assert(
        std::is_same_v< BoundaryType, BoundaryVertex > || std::is_same_v< BoundaryType, BoundaryEdge > ||
        std::is_same_v< BoundaryType, BoundaryFace > );

    return static_cast< BoundaryPosition >( ( static_cast< int >( boundary_type ) & 0b110000 ) >> 4 );
}

template < typename BoundaryType >
constexpr BoundaryPosition boundary_position_from_boundary_type_y( const BoundaryType& boundary_type )
{
    static_assert(
        std::is_same_v< BoundaryType, BoundaryVertex > || std::is_same_v< BoundaryType, BoundaryEdge > ||
        std::is_same_v< BoundaryType, BoundaryFace > );

    return static_cast< BoundaryPosition >( ( static_cast< int >( boundary_type ) & 0b001100 ) >> 2 );
}

template < typename BoundaryType >
constexpr BoundaryPosition boundary_position_from_boundary_type_r( const BoundaryType& boundary_type )
{
    static_assert(
        std::is_same_v< BoundaryType, BoundaryVertex > || std::is_same_v< BoundaryType, BoundaryEdge > ||
        std::is_same_v< BoundaryType, BoundaryFace > );

    return static_cast< BoundaryPosition >( ( static_cast< int >( boundary_type ) & 0b000011 ) >> 0 );
}

constexpr bool is_edge_boundary_radial( const BoundaryEdge id )
{
    return id == BoundaryEdge::E_00R || id == BoundaryEdge::E_10R || id == BoundaryEdge::E_01R ||
           id == BoundaryEdge::E_11R;
}

constexpr bool is_face_boundary_normal_to_radial_direction( const BoundaryFace id )
{
    return id == BoundaryFace::F_XY0 || id == BoundaryFace::F_XY1;
}

constexpr BoundaryVertex other_side_r( BoundaryVertex boundary_vertex )
{
    return static_cast< BoundaryVertex >( boundary_position_encoding(
        boundary_position_from_boundary_type_x( boundary_vertex ),
        boundary_position_from_boundary_type_y( boundary_vertex ),
        boundary_position_from_boundary_type_r( boundary_vertex ) == BoundaryPosition::P0 ? BoundaryPosition::P1 :
                                                                                            BoundaryPosition::P0 ) );
}

constexpr std::array all_boundary_vertices = {
    BoundaryVertex::V_000,
    BoundaryVertex::V_100,
    BoundaryVertex::V_010,
    BoundaryVertex::V_110,
    BoundaryVertex::V_001,
    BoundaryVertex::V_101,
    BoundaryVertex::V_011,
    BoundaryVertex::V_111 };

constexpr std::array all_boundary_edges = {
    BoundaryEdge::E_X00,
    BoundaryEdge::E_X10,
    BoundaryEdge::E_X01,
    BoundaryEdge::E_X11,

    BoundaryEdge::E_0Y0,
    BoundaryEdge::E_1Y0,
    BoundaryEdge::E_0Y1,
    BoundaryEdge::E_1Y1,

    BoundaryEdge::E_00R,
    BoundaryEdge::E_10R,
    BoundaryEdge::E_01R,
    BoundaryEdge::E_11R,
};

constexpr std::array all_boundary_faces = {
    BoundaryFace::F_XY0,
    BoundaryFace::F_XY1,
    BoundaryFace::F_X0R,
    BoundaryFace::F_X1R,
    BoundaryFace::F_0YR,
    BoundaryFace::F_1YR,
};

// String conversion functions
inline std::string to_string( BoundaryVertex v )
{
    switch ( v )
    {
    case BoundaryVertex::V_000:
        return "V_000";
    case BoundaryVertex::V_100:
        return "V_100";
    case BoundaryVertex::V_010:
        return "V_010";
    case BoundaryVertex::V_110:
        return "V_110";
    case BoundaryVertex::V_001:
        return "V_001";
    case BoundaryVertex::V_101:
        return "V_101";
    case BoundaryVertex::V_011:
        return "V_011";
    case BoundaryVertex::V_111:
        return "V_111";
    default:
        return "<unknown LocalBoundaryVertex>";
    }
}

inline std::string to_string( BoundaryEdge e )
{
    switch ( e )
    {
    case BoundaryEdge::E_X00:
        return "E_X00";
    case BoundaryEdge::E_X10:
        return "E_X10";
    case BoundaryEdge::E_X01:
        return "E_X01";
    case BoundaryEdge::E_X11:
        return "E_X11";
    case BoundaryEdge::E_0Y0:
        return "E_0Y0";
    case BoundaryEdge::E_1Y0:
        return "E_1Y0";
    case BoundaryEdge::E_0Y1:
        return "E_0Y1";
    case BoundaryEdge::E_1Y1:
        return "E_1Y1";
    case BoundaryEdge::E_00R:
        return "E_00R";
    case BoundaryEdge::E_10R:
        return "E_10R";
    case BoundaryEdge::E_01R:
        return "E_01R";
    case BoundaryEdge::E_11R:
        return "E_11R";
    default:
        return "<unknown LocalBoundaryEdge>";
    }
}

inline std::string to_string( BoundaryFace f )
{
    switch ( f )
    {
    case BoundaryFace::F_XY0:
        return "F_XY0";
    case BoundaryFace::F_XY1:
        return "F_XY1";
    case BoundaryFace::F_X0R:
        return "F_X0R";
    case BoundaryFace::F_X1R:
        return "F_X1R";
    case BoundaryFace::F_0YR:
        return "F_0YR";
    case BoundaryFace::F_1YR:
        return "F_1YR";
    default:
        return "<unknown LocalBoundaryFace>";
    }
}

inline std::ostream& operator<<( std::ostream& os, BoundaryVertex v )
{
    return os << to_string( v );
}

inline std::ostream& operator<<( std::ostream& os, BoundaryEdge e )
{
    return os << to_string( e );
}

inline std::ostream& operator<<( std::ostream& os, BoundaryFace f )
{
    return os << to_string( f );
}

} // namespace terra::grid
