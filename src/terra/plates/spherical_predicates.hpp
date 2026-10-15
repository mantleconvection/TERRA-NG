#pragma once

/// \file
/// Point-in-spherical-polygon and point-to-great-circle-arc distance.
///
/// These replace the two things Boost.Geometry was used for -- `within()` and
/// `distance()` on a `spherical_equatorial` polygon -- and are the only geometry the plate
/// lookup needs. Everything here works on unit vectors in R^3, never on longitude and
/// latitude, so the antimeridian seam does not exist and no equivalent of
/// `boost::geometry::correct()` is required: the winding test is orientation-agnostic.
///
/// Every function is KOKKOS_INLINE_FUNCTION and free of allocation, exceptions and virtual
/// dispatch, so the same code runs on host and device.
///
/// Derived from the Kokkos prototype on suganth1997/plates-gpu (`is_point_in_plate`), with the
/// distance returned rather than discarded, the vertex accessor templated so the same sweep
/// serves both `std::vector<vec3D>` and a flat `Kokkos::View`, and the endpoint angles taken
/// with atan2 rather than acos -- see angleBetweenUnit().
///
/// The inside test itself is NOT the prototype's. See the note on pointInSphericalPolygon().

#include "terra/dense/vec.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"

namespace terra {
namespace plates {
namespace geometry {

using UnitVec = dense::Vec< double, 3 >;

/// Unit cartesian vector from longitude and latitude in degrees
KOKKOS_INLINE_FUNCTION
UnitVec lonLatDegToUnit( const double lonDeg, const double latDeg )
{
    constexpr double degToRad = 3.14159265358979323846 / 180.0;

    const double lon    = lonDeg * degToRad;
    const double lat    = latDeg * degToRad;
    const double coslat = Kokkos::cos( lat );

    return UnitVec{ coslat * Kokkos::cos( lon ), coslat * Kokkos::sin( lon ), Kokkos::sin( lat ) };
}

/// Angle in radians between two unit vectors
///
/// atan2( |a x b|, a.b ) rather than acos( a.b ): acos loses roughly half the significant digits
/// for small angles, which is exactly the regime that matters here, since the distance to a
/// plate boundary is only interesting when it is small.
KOKKOS_INLINE_FUNCTION
double angleBetweenUnit( const UnitVec& a, const UnitVec& b )
{
    return Kokkos::atan2( a.cross( b ).norm(), a.dot( b ) );
}

/// Angular distance in radians from a unit vector to the great-circle arc from \p a to \p b
///
/// Drops the perpendicular onto the great circle through a and b; if its foot falls within the
/// arc the distance is to the circle, otherwise to the nearer endpoint. Degenerate edges (a and
/// b numerically coincident, which is what a closed ring's wrap-around edge looks like) fall
/// back to the distance to the vertex.
KOKKOS_INLINE_FUNCTION
double distancePointToArc( const UnitVec& p, const UnitVec& a, const UnitVec& b )
{
    const UnitVec cross = a.cross( b );
    const double  len   = cross.norm();

    if ( len < 1e-14 )
    {
        return angleBetweenUnit( p, a );
    }

    const UnitVec n     = cross * ( 1.0 / len );
    const double  pDotN = p.dot( n );

    // Foot of the perpendicular from p onto the great circle through a and b.
    const UnitVec foot = ( p - pDotN * n ).normalized();

    // The foot lies within the arc when it is no further from either endpoint than the
    // endpoints are from each other.
    const double abDot = a.dot( b );
    const bool   onArc = ( a.dot( foot ) >= abDot ) && ( b.dot( foot ) >= abDot );

    if ( onArc )
    {
        const double s = pDotN < 0.0 ? -pDotN : pDotN;
        return Kokkos::asin( s > 1.0 ? 1.0 : s );
    }

    const double dA = angleBetweenUnit( p, a );
    const double dB = angleBetweenUnit( p, b );

    return dA < dB ? dA : dB;
}

struct PointInPolygonResult
{
    /// Whether the point lies inside the polygon
    bool inside;

    /// Angular distance in radians to the nearest polygon edge. Always computed, and meaningful
    /// whether or not the point is inside.
    double distanceRad;
};

/// Whether the arcs p->q and a->b cross
///
/// Both great circles are formed, their line of intersection gives two antipodal candidate
/// points, and each is tested for lying within both arcs. Coplanar arcs are reported as not
/// crossing, which is a measure-zero case.
KOKKOS_INLINE_FUNCTION
bool arcsCross( const UnitVec& p, const UnitVec& q, const UnitVec& a, const UnitVec& b )
{
    const UnitVec n1 = p.cross( q );
    const UnitVec n2 = a.cross( b );

    const UnitVec d    = n1.cross( n2 );
    const double  dLen = d.norm();

    if ( dLen < 1e-14 )
    {
        return false;
    }

    const UnitVec x = d * ( 1.0 / dLen );

    for ( int sign = 0; sign < 2; ++sign )
    {
        const UnitVec c = sign == 0 ? x : x * -1.0;

        // Within an arc when the candidate turns the same way from each endpoint as the arc does.
        const bool onPQ = ( p.cross( c ).dot( n1 ) >= 0.0 ) && ( c.cross( q ).dot( n1 ) >= 0.0 );
        const bool onAB = ( a.cross( c ).dot( n2 ) >= 0.0 ) && ( c.cross( b ).dot( n2 ) >= 0.0 );

        if ( onPQ && onAB )
        {
            return true;
        }
    }

    return false;
}

/// Tests a unit vector against a spherical polygon and measures its distance to the boundary
///
/// The inside test counts how many polygon edges the arc from \p p to \p q crosses; an odd count
/// means the two points are on opposite sides, so p is inside whenever q is outside.
///
/// \note A turning-angle (winding) sum does NOT work here, though it is the obvious thing to
///       reach for and is what the prototype this code descends from used. A closed curve splits
///       the sphere into two regions and the turning angles sum to a full turn in *both* of them,
///       so every point in the complement of a polygon enclosing a pole is reported inside. That
///       showed up as most of the northern hemisphere being assigned to Antarctica.
///
/// \param p        the test point, a unit vector
/// \param q        a reference direction known to lie outside the polygon; the antipode of the
///                 polygon's bounding-cap centre serves whenever that cap is under a hemisphere
/// \param numVerts number of polygon vertices
/// \param vertex   callable mapping a vertex index to its unit vector; the ring is closed
///                 implicitly, so a ring whose last vertex repeats its first is fine
template < typename VertexFn >
KOKKOS_INLINE_FUNCTION PointInPolygonResult
    pointInSphericalPolygon( const UnitVec& p, const UnitVec& q, const int numVerts, const VertexFn& vertex )
{
    double minDist   = Kokkos::Experimental::finite_max_v< double >;
    int    crossings = 0;

    if ( numVerts < 3 )
    {
        return PointInPolygonResult{ false, minDist };
    }

    UnitVec a = vertex( 0 );

    for ( int i = 0; i < numVerts; ++i )
    {
        const int     j = ( i + 1 == numVerts ) ? 0 : i + 1;
        const UnitVec b = vertex( j );

        if ( arcsCross( p, q, a, b ) )
        {
            ++crossings;
        }

        const double edgeDist = distancePointToArc( p, a, b );
        minDist               = edgeDist < minDist ? edgeDist : minDist;

        a = b;
    }

    return PointInPolygonResult{ ( crossings % 2 ) == 1, minDist };
}

/// Direction opposite the polygon's vertex centroid
///
/// Serves as the outside reference point for pointInSphericalPolygon(). Sound whenever the
/// polygon fits in a cap narrower than a hemisphere, which is the case for plate polygons.
template < typename VertexFn >
KOKKOS_INLINE_FUNCTION UnitVec outsideReference( const int numVerts, const VertexFn& vertex )
{
    UnitVec sum{ 0.0, 0.0, 0.0 };
    for ( int i = 0; i < numVerts; ++i )
    {
        const UnitVec v = vertex( i );
        for ( int d = 0; d < 3; ++d )
        {
            sum( d ) += v( d );
        }
    }

    const double len = sum.norm();
    if ( len < 1e-12 )
    {
        return UnitVec{ 0.0, 0.0, 1.0 };
    }

    return sum * ( -1.0 / len );
}

/// Convenience overload deriving the outside reference from the polygon itself
template < typename VertexFn >
KOKKOS_INLINE_FUNCTION PointInPolygonResult
    pointInSphericalPolygon( const UnitVec& p, const int numVerts, const VertexFn& vertex )
{
    return pointInSphericalPolygon( p, outsideReference( numVerts, vertex ), numVerts, vertex );
}

} // namespace geometry
} // namespace plates
} // namespace terra
