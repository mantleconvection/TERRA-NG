#pragma once

/// \file
/// Flat, device-ready description of the plate polygons of one age stage.
///
/// PlateStorage keeps plates as a std::map of std::vector<PlateInfo>, each holding a
/// std::vector<vec3D> of (longitude, latitude, 0) vertices. That is fine for loading and
/// hopeless for a kernel: pointer chasing, host-only containers, and four transcendentals per
/// vertex on every query to get back to a cartesian direction.
///
/// PlateStageData packs one stage into flat Kokkos::Views with the vertices already converted
/// to unit cartesian, so the trigonometry leaves the inner loop entirely, and adds two things
/// the polygon sweep alone does not have:
///
///   - a bounding cap per plate, so a point far from a plate is rejected by one dot product
///     instead of a full edge sweep;
///   - a coarse longitude/latitude bin grid mapping a bin to its candidate plates, so a point
///     does not even look at most plates.
///
/// The Euler vector of each plate is carried alongside, indexed the same way, so a lookup hands
/// back everything the velocity needs in one place.
///
/// The struct is a POD of Views and is captured by value into a kernel. Phase 5 of the port
/// swaps host() for device() at the call site and nothing else changes.

#include <cmath>
#include <stdexcept>
#include <vector>

#include "terra/dense/vec.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/plates/plate_rotation_provider.hpp"
#include "terra/plates/plate_storage.hpp"
#include "terra/plates/functions_for_plates.hpp"
#include "terra/plates/spherical_predicates.hpp"

namespace terra {
namespace plates {

/// Views over one age stage's plates. Copyable by value into a kernel.
///
/// The layout is pinned to LayoutRight rather than left to the default, which differs between
/// HostSpace and a CUDA space and would make the host and device view types incompatible. It is
/// also the layout the sweep wants: a thread walks all three components of one vertex, and a
/// warp reads the same vertex of the same plate at once, so this is a broadcast rather than a
/// strided read.
template < class MemSpace >
struct PlateStageViews
{
    /// Polygon vertices as unit cartesian vectors, CSR-packed by plate
    Kokkos::View< double* [3], Kokkos::LayoutRight, MemSpace > vertices;

    /// Offsets into \c vertices, size nPlates + 1
    Kokkos::View< int*, Kokkos::LayoutRight, MemSpace > ringBegin;

    /// Plate ID as it appears in the datafile, size nPlates
    Kokkos::View< unsigned int*, Kokkos::LayoutRight, MemSpace > plateId;

    /// Bounding cap per plate: centre in 0..2, cos of the angular radius in 3
    Kokkos::View< double* [4], Kokkos::LayoutRight, MemSpace > cap;

    /// Euler vector per plate, cartesian, degrees per Ma
    Kokkos::View< double* [3], Kokkos::LayoutRight, MemSpace > omega;

    /// Whether \c omega is meaningful for a plate, size nPlates.
    ///
    /// Zero when the topologies name a plate that the rotation data does not describe. Such a plate has no
    /// velocity at all, which is not the same as a velocity of zero, so consumers must skip it rather than
    /// read its zero \c omega as "this plate does not move". The host query path expresses the same thing by
    /// demoting a hit on such a plate to a miss.
    Kokkos::View< unsigned char*, Kokkos::LayoutRight, MemSpace > hasRotation;

    /// Offsets into \c binPlate, size nLonBins * nLatBins + 1
    Kokkos::View< int*, Kokkos::LayoutRight, MemSpace > binBegin;

    /// Candidate plate indices per bin
    Kokkos::View< int*, Kokkos::LayoutRight, MemSpace > binPlate;

    int nPlates{ 0 };
    int nLonBins{ 0 };
    int nLatBins{ 0 };
};

struct PlateLookupResult
{
    bool         found{ false };
    int          plateIndex{ -1 };
    unsigned int plateId{ 0 };

    /// Angular distance to the plate boundary, in radians. Only meaningful when found.
    double distanceRad{ 0.0 };
};

/// Bin containing a direction, or -1 if the grid is empty
template < class MemSpace >
KOKKOS_INLINE_FUNCTION int binIndexOf( const PlateStageViews< MemSpace >& s, const geometry::UnitVec& p )
{
    if ( s.nLonBins <= 0 || s.nLatBins <= 0 )
    {
        return -1;
    }

    constexpr double pi = 3.14159265358979323846;

    const double lon = Kokkos::atan2( p( 1 ), p( 0 ) );                    // [-pi, pi]
    const double lat = Kokkos::asin( p( 2 ) > 1.0 ? 1.0 : ( p( 2 ) < -1.0 ? -1.0 : p( 2 ) ) );

    int iLon = static_cast< int >( ( lon + pi ) / ( 2.0 * pi ) * s.nLonBins );
    int iLat = static_cast< int >( ( lat + 0.5 * pi ) / pi * s.nLatBins );

    iLon = iLon < 0 ? 0 : ( iLon >= s.nLonBins ? s.nLonBins - 1 : iLon );
    iLat = iLat < 0 ? 0 : ( iLat >= s.nLatBins ? s.nLatBins - 1 : iLat );

    return iLat * s.nLonBins + iLon;
}

/// Finds the plate containing \p p and its distance to that plate's boundary
///
/// Mirrors findPlateAndDistance(): the first matching plate wins, and the distance is the
/// minimum over that plate's edges. Candidates come from the bin grid where one is built and
/// from the whole stage otherwise, and each candidate is cap-rejected before its edges are
/// swept.
template < class MemSpace >
KOKKOS_INLINE_FUNCTION PlateLookupResult findPlateInStage( const PlateStageViews< MemSpace >& s,
                                                           const geometry::UnitVec&           p )
{
    const int bin       = binIndexOf( s, p );
    const bool binned   = ( bin >= 0 ) && ( s.binBegin.extent( 0 ) > 0 );
    const int candBegin = binned ? s.binBegin( bin ) : 0;
    const int candEnd   = binned ? s.binBegin( bin + 1 ) : s.nPlates;

    for ( int c = candBegin; c < candEnd; ++c )
    {
        const int plate = binned ? s.binPlate( c ) : c;

        // Cap rejection. cos(radius) < 0 marks a cap wider than a hemisphere, which is not
        // geodesically convex and so cannot be used to reject; those plates are always swept.
        const double cosRadius = s.cap( plate, 3 );
        if ( cosRadius > 0.0 )
        {
            const double d = p( 0 ) * s.cap( plate, 0 ) + p( 1 ) * s.cap( plate, 1 ) + p( 2 ) * s.cap( plate, 2 );
            if ( d < cosRadius )
            {
                continue;
            }
        }

        const int begin = s.ringBegin( plate );
        const int count = s.ringBegin( plate + 1 ) - begin;

        // The cap centre's antipode is the outside reference the parity test needs, and it is
        // already computed -- no extra pass over the vertices.
        const geometry::UnitVec outside{ -s.cap( plate, 0 ), -s.cap( plate, 1 ), -s.cap( plate, 2 ) };

        const auto r = geometry::pointInSphericalPolygon( p, outside, count, [&s, begin]( const int i ) {
            return geometry::UnitVec{ s.vertices( begin + i, 0 ), s.vertices( begin + i, 1 ), s.vertices( begin + i, 2 ) };
        } );

        if ( r.inside )
        {
            return PlateLookupResult{ true, plate, s.plateId( plate ), r.distanceRad };
        }
    }

    return PlateLookupResult{ false, -1, 0, 0.0 };
}

/// Builds and owns the flat stage description for one age, on host and on device.
class PlateStageData
{
  public:
    using HostSpace   = Kokkos::HostSpace;
    using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

    PlateStageData() = default;

    /// \param age      the age stage to pack; plates are taken from std::ceil( age ), matching
    ///                 findPlateAndDistance()
    /// \param nLonBins longitude bins for the broad phase; 0 disables binning
    /// \param nLatBins latitude bins for the broad phase; 0 disables binning
    PlateStageData( const PlateStorage&          topologies,
                    const PlateRotationProvider& rotations,
                    const double                 age,
                    const int                    nLonBins = 64,
                    const int                    nLatBins = 32 )
    : age_( age )
    {
        const auto& plates = topologies.getPlatesForStage( std::ceil( age ) );

        const int nPlates = static_cast< int >( plates.size() );

        int nVerts = 0;
        for ( const auto& plate : plates )
        {
            nVerts += static_cast< int >( plate.boundary.size() );
        }

        allocate( host_, nPlates, nVerts, nLonBins, nLatBins );

        // --- vertices, CSR offsets, plate IDs, Euler vectors ---

        int offset = 0;
        for ( int i = 0; i < nPlates; ++i )
        {
            host_.ringBegin( i ) = offset;
            host_.plateId( i )   = plates[i].id;

            // A topology file may name plates the rotation file does not describe. Such a plate has no Euler
            // vector; pack a zero and record the id, so that the caller can route points on it through its
            // PlateNotFoundHandler instead of reading the zero as a genuine "this plate does not move".
            vec3D w{ 0, 0, 0 };
            host_.hasRotation( i ) = 1;
            try
            {
                w = computeEulerVector( rotations, static_cast< int >( plates[i].id ), age );
            }
            catch ( const std::runtime_error& )
            {
                platesWithoutRotations_.push_back( plates[i].id );
                host_.hasRotation( i ) = 0;
            }

            for ( int d = 0; d < 3; ++d )
            {
                host_.omega( i, d ) = w( d );
            }

            for ( const auto& vertex : plates[i].boundary )
            {
                const auto unit = geometry::lonLatDegToUnit( vertex( 0 ), vertex( 1 ) );
                for ( int d = 0; d < 3; ++d )
                {
                    host_.vertices( offset, d ) = unit( d );
                }
                ++offset;
            }
        }
        host_.ringBegin( nPlates ) = offset;

        buildCaps( nPlates );
        buildBins( nPlates, nLonBins, nLatBins );

        mirrorToDevice();
    }

    double age() const { return age_; }

    /// Plates in this stage whose reconstruction circuit could not be built from the rotation data. Their
    /// packed Euler vector is zero and must not be used.
    const std::vector< uint_t >& platesWithoutRotations() const { return platesWithoutRotations_; }

    const PlateStageViews< HostSpace >&   host() const { return host_; }
    const PlateStageViews< DeviceSpace >& device() const { return device_; }

  private:
    std::vector< uint_t > platesWithoutRotations_;


    template < class Space >
    static void allocate( PlateStageViews< Space >& v, int nPlates, int nVerts, int nLonBins, int nLatBins )
    {
        v.vertices  = Kokkos::View< double* [3], Kokkos::LayoutRight, Space >( "plate_vertices", nVerts );
        v.ringBegin = Kokkos::View< int*, Kokkos::LayoutRight, Space >( "plate_ring_begin", nPlates + 1 );
        v.plateId   = Kokkos::View< unsigned int*, Kokkos::LayoutRight, Space >( "plate_id", nPlates );
        v.cap       = Kokkos::View< double* [4], Kokkos::LayoutRight, Space >( "plate_cap", nPlates );
        v.omega     = Kokkos::View< double* [3], Kokkos::LayoutRight, Space >( "plate_omega", nPlates );
        v.hasRotation =
            Kokkos::View< unsigned char*, Kokkos::LayoutRight, Space >( "plate_has_rotation", nPlates );
        v.nPlates   = nPlates;
        v.nLonBins  = nLonBins;
        v.nLatBins  = nLatBins;
    }

    /// Smallest-ish spherical cap containing a plate's vertices
    ///
    /// The centre is the normalised vertex sum and the radius reaches the furthest vertex. A
    /// great-circle arc between two points of a cap narrower than a hemisphere stays inside it,
    /// so containing the vertices is enough to contain the edges -- and, for the small regions
    /// plate polygons describe, the interior too. Wider caps are marked with a negative
    /// cos(radius) and never used for rejection.
    void buildCaps( const int nPlates )
    {
        for ( int i = 0; i < nPlates; ++i )
        {
            const int begin = host_.ringBegin( i );
            const int end   = host_.ringBegin( i + 1 );

            geometry::UnitVec sum{ 0.0, 0.0, 0.0 };
            for ( int k = begin; k < end; ++k )
            {
                for ( int d = 0; d < 3; ++d )
                {
                    sum( d ) += host_.vertices( k, d );
                }
            }

            const double len = sum.norm();
            if ( end == begin || len < 1e-12 )
            {
                // Degenerate or antipodally balanced: no usable cap.
                host_.cap( i, 0 ) = 1.0;
                host_.cap( i, 1 ) = 0.0;
                host_.cap( i, 2 ) = 0.0;
                host_.cap( i, 3 ) = -1.0;
                continue;
            }

            const geometry::UnitVec centre = sum * ( 1.0 / len );

            double cosRadius = 1.0;
            for ( int k = begin; k < end; ++k )
            {
                const double d = centre( 0 ) * host_.vertices( k, 0 ) + centre( 1 ) * host_.vertices( k, 1 ) +
                                 centre( 2 ) * host_.vertices( k, 2 );
                cosRadius = d < cosRadius ? d : cosRadius;
            }

            host_.cap( i, 0 ) = centre( 0 );
            host_.cap( i, 1 ) = centre( 1 );
            host_.cap( i, 2 ) = centre( 2 );
            host_.cap( i, 3 ) = cosRadius > 0.0 ? cosRadius : -1.0;
        }
    }

    /// Marks every bin whose spherical extent can touch a plate's cap
    ///
    /// Conservative by construction: a bin is kept whenever its centre lies within the cap
    /// radius plus the bin's own angular half-diagonal, so no containing plate is ever missed.
    /// Plates without a usable cap are added to every bin.
    void buildBins( const int nPlates, const int nLonBins, const int nLatBins )
    {
        if ( nLonBins <= 0 || nLatBins <= 0 )
        {
            return;
        }

        constexpr double pi     = 3.14159265358979323846;
        const int        nBins  = nLonBins * nLatBins;
        const double     dLon   = 2.0 * pi / nLonBins;
        const double     dLat   = pi / nLatBins;
        const double     margin = 0.5 * std::sqrt( dLon * dLon + dLat * dLat );

        std::vector< std::vector< int > > candidates( nBins );

        for ( int b = 0; b < nBins; ++b )
        {
            const int    iLon = b % nLonBins;
            const int    iLat = b / nLonBins;
            const double lon  = -pi + ( iLon + 0.5 ) * dLon;
            const double lat  = -0.5 * pi + ( iLat + 0.5 ) * dLat;

            const geometry::UnitVec c{ std::cos( lat ) * std::cos( lon ), std::cos( lat ) * std::sin( lon ), std::sin( lat ) };

            for ( int i = 0; i < nPlates; ++i )
            {
                const double cosRadius = host_.cap( i, 3 );
                if ( cosRadius <= 0.0 )
                {
                    candidates[b].push_back( i );
                    continue;
                }

                const double d = c( 0 ) * host_.cap( i, 0 ) + c( 1 ) * host_.cap( i, 1 ) + c( 2 ) * host_.cap( i, 2 );

                // cos( radius + margin ), clamped: beyond pi the cap covers everything.
                const double radius   = std::acos( cosRadius > 1.0 ? 1.0 : cosRadius );
                const double extended = radius + margin;
                const double cosLimit = extended >= pi ? -1.0 : std::cos( extended );

                if ( d >= cosLimit )
                {
                    candidates[b].push_back( i );
                }
            }
        }

        int total = 0;
        for ( const auto& c : candidates )
        {
            total += static_cast< int >( c.size() );
        }

        host_.binBegin = Kokkos::View< int*, Kokkos::LayoutRight, HostSpace >( "plate_bin_begin", nBins + 1 );
        host_.binPlate = Kokkos::View< int*, Kokkos::LayoutRight, HostSpace >( "plate_bin_plate", total );

        int offset = 0;
        for ( int b = 0; b < nBins; ++b )
        {
            host_.binBegin( b ) = offset;
            for ( const int i : candidates[b] )
            {
                host_.binPlate( offset++ ) = i;
            }
        }
        host_.binBegin( nBins ) = offset;
    }

    void mirrorToDevice()
    {
        device_.nPlates  = host_.nPlates;
        device_.nLonBins = host_.nLonBins;
        device_.nLatBins = host_.nLatBins;

        device_.vertices  = Kokkos::create_mirror_view_and_copy( DeviceSpace{}, host_.vertices );
        device_.ringBegin = Kokkos::create_mirror_view_and_copy( DeviceSpace{}, host_.ringBegin );
        device_.plateId   = Kokkos::create_mirror_view_and_copy( DeviceSpace{}, host_.plateId );
        device_.cap       = Kokkos::create_mirror_view_and_copy( DeviceSpace{}, host_.cap );
        device_.omega       = Kokkos::create_mirror_view_and_copy( DeviceSpace{}, host_.omega );
        device_.hasRotation = Kokkos::create_mirror_view_and_copy( DeviceSpace{}, host_.hasRotation );

        if ( host_.binBegin.extent( 0 ) > 0 )
        {
            device_.binBegin = Kokkos::create_mirror_view_and_copy( DeviceSpace{}, host_.binBegin );
            device_.binPlate = Kokkos::create_mirror_view_and_copy( DeviceSpace{}, host_.binPlate );
        }
    }

    double                          age_{ 0.0 };
    PlateStageViews< HostSpace >    host_;
    PlateStageViews< DeviceSpace >  device_;
};

} // namespace plates
} // namespace terra
