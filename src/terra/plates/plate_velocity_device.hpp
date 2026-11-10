#pragma once

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/plates/conversions.hpp"
#include "terra/plates/plate_rotation_provider.hpp"
#include "terra/plates/plate_stage_data.hpp"
#include "terra/plates/spherical_predicates.hpp"
#include "terra/plates/types.hpp"

// preserve ordering of includes: functions_for_plates.hpp needs PlateRotationProvider to be declared already
#include "terra/plates/functions_for_geometry.hpp"
#include "terra/plates/local_averaging_point_weight_provider.hpp"
#include "terra/plates/functions_for_plates.hpp"

/// @file
///
/// Plate surface velocities evaluated on the device.
///
/// The host path (PlateVelocityProvider::getLocallyAveragedPointVelocity, driven by
/// extract_plate_velocities) walks the polygon containers and is not device-callable, so it mirrors the mesh
/// coordinates to the host, computes there, and copies the field back. Measured on a level-6 shell that step
/// costs ~2.3 s and grows about fourfold per refinement level, while the equivalent plate-id lookup already
/// running on the device costs ~8 ms and is flat.
///
/// Everything needed to close that gap is already packed: \ref PlateStageData builds a
/// `PlateStageViews<DeviceSpace>` holding the boundary vertices, the CSR ring offsets, per-plate bounding caps,
/// a lon/lat bin index **and the per-plate Euler vectors**, and \ref findPlateInStage is device-callable. This
/// header just uses them.
///
/// Both regimes of the host query are reproduced: points comfortably inside a plate take the plate's rigid
/// rotation directly, and points close enough to a boundary that the averaging stencil could straddle it are
/// averaged over that stencil, with the radial component projected out afterwards. The stencil offsets are
/// independent of the point they surround -- only the local tangent frame is -- so they are uploaded once.

namespace terra::plates
{

/// @brief An orthonormal pair spanning the tangent plane at `normalCart`.
///
/// Device-callable counterpart of \ref findOrthogonalVectorsCart, which returns a std::pair and uses std::fabs.
KOKKOS_INLINE_FUNCTION void tangent_frame( const vec3D& normalCart, vec3D& first, vec3D& second )
{
    // Start from whichever axis is least aligned with the normal, so the cross product is well conditioned.
    double smallest = Kokkos::abs( normalCart( 0 ) );
    vec3D  u1{ 1, 0, 0 };

    if ( Kokkos::abs( normalCart( 1 ) ) < smallest )
    {
        smallest = Kokkos::abs( normalCart( 1 ) );
        u1       = vec3D{ 0, 1, 0 };
    }
    if ( Kokkos::abs( normalCart( 2 ) ) < smallest )
    {
        u1 = vec3D{ 0, 0, 1 };
    }

    first  = normalCart.cross( u1 ).normalized();
    second = normalCart.cross( first ).normalized();
}

/// @brief The averaging stencil, flattened for the device: 2D tangent-plane offsets and their weights.
template < typename MemSpace >
struct DeviceAveragingStencil
{
    /// (offset along the first tangent direction, offset along the second, weight)
    Kokkos::View< double* [3], Kokkos::LayoutRight, MemSpace > offsets;

    /// Reach of the stencil in km, matching the units of PlateLookupResult::distanceRad * earthRadiusInKm.
    double maxDistanceKm{ 0 };

    KOKKOS_INLINE_FUNCTION int size() const { return static_cast< int >( offsets.extent( 0 ) ); }
};

/// @brief Uploads a UniformCirclesPointWeightProvider's stencil once, for reuse across all points.
inline DeviceAveragingStencil< typename Kokkos::DefaultExecutionSpace::memory_space >
    make_device_averaging_stencil( const UniformCirclesPointWeightProvider& provider )
{
    using DeviceSpace = typename Kokkos::DefaultExecutionSpace::memory_space;

    const auto& offsets = provider.sampleOffsets2DCart();

    Kokkos::View< double* [3], Kokkos::LayoutRight, DeviceSpace > device(
        "plate_averaging_stencil", offsets.size() );
    auto host = Kokkos::create_mirror_view( device );

    for ( size_t i = 0; i < offsets.size(); ++i )
    {
        host( i, 0 ) = offsets[i].first( 0 );
        host( i, 1 ) = offsets[i].first( 1 );
        host( i, 2 ) = offsets[i].second;
    }
    Kokkos::deep_copy( device, host );

    return { device, provider.maxDistance( vec3D{ 0, 0, 1 } ) };
}

/// @brief Writes the id of the plate under every node of the outermost shell.
///
/// Uses the same packed-stage lookup as the velocity kernel, i.e. the arc-crossing test in
/// \ref pointInSphericalPolygon. That matters: the obvious turning-angle (winding) sum reports every point in
/// the complement of a pole-enclosing polygon as inside, so with "first matching plate wins" the early plates
/// swallow points belonging to later ones. Measured against boost::geometry::within on a level-5 shell at
/// 5 Ma, a winding-sum version of this kernel used only 18 distinct plates where the correct predicate uses 40,
/// gave several plates exactly twice their true area, and never assigned two of them at all.
template < typename ScalarType >
struct PlateIDInterpolator
{
    using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

    grid::Grid3DDataVec< ScalarType, 3 > coords_shell;
    grid::Grid2DDataScalar< ScalarType > coords_radii;
    grid::Grid4DDataScalar< ScalarType > plate_id;
    PlateStageViews< DeviceSpace >       stage;
    int                                  surface_r;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        if ( r != surface_r )
        {
            plate_id( sd, x, y, r ) = ScalarType( 0 );
            return;
        }

        const auto coords = grid::shell::coords( sd, x, y, r, coords_shell, coords_radii );
        const vec3D lonLatRad =
            conversions::cart2sph( vec3D{ coords( 0 ), coords( 1 ), coords( 2 ) } );

        const auto hit =
            findPlateInStage( stage, geometry::lonLatDegToUnit( lonLatRad( 0 ), lonLatRad( 1 ) ) );

        plate_id( sd, x, y, r ) =
            hit.found ? static_cast< ScalarType >( stage.plateId( hit.plateIndex ) ) : ScalarType( 0 );
    }
};

/// @brief Fills `plate_id` with the plate under each outermost-shell node, on the device.
template < typename ScalarType >
void extract_plate_ids_device(
    const grid::shell::DistributedDomain&                                          domain,
    const grid::Grid3DDataVec< ScalarType, 3 >&                                    coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                                    coords_radii,
    const PlateStageViews< typename Kokkos::DefaultExecutionSpace::memory_space >& stage,
    grid::Grid4DDataScalar< ScalarType >&                                          plate_id )
{
    const int surface_r = domain.domain_info().subdomain_num_nodes_radially() - 1;

    Kokkos::parallel_for(
        "extract_plate_ids_device",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        PlateIDInterpolator< ScalarType >{ coords_shell, coords_radii, plate_id, stage, surface_r } );
    Kokkos::fence();
}

/// @brief Evaluates the rigid-plate surface velocity at every node of the outermost shell.
template < typename ScalarType >
struct DevicePlateVelocityInterpolator
{
    using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;

    grid::Grid3DDataVec< ScalarType, 3 >  coords_shell;
    grid::Grid2DDataScalar< ScalarType >  coords_radii;
    grid::Grid4DDataVec< ScalarType, 3 >  velocity;
    PlateStageViews< DeviceSpace >        stage;
    DeviceAveragingStencil< DeviceSpace > stencil;
    int                                   surface_r;
    ScalarType                            scale;

    /// Rigid-plate velocity at a lon/lat/radius point, or false if no plate covers it.
    KOKKOS_INLINE_FUNCTION
    bool velocity_at( const vec3D& lonLatRad, vec3D& out ) const
    {
        const auto hit = findPlateInStage( stage, geometry::lonLatDegToUnit( lonLatRad( 0 ), lonLatRad( 1 ) ) );

        // A plate with no rotation data has no velocity, which is not a velocity of zero: skip it, exactly as
        // the host query does by demoting such a hit to a miss.
        if ( !hit.found || stage.hasRotation( hit.plateIndex ) == 0 )
            return false;

        vec3D omega;
        for ( int d = 0; d < 3; ++d )
            omega( d ) = stage.omega( hit.plateIndex, d );

        out = eulerVectorToVelocity( lonLatRad, omega, 1.0 );
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y ) const
    {
        const auto coords = grid::shell::coords( sd, x, y, surface_r, coords_shell, coords_radii );
        const vec3D point{ coords( 0 ), coords( 1 ), coords( 2 ) };

        const vec3D lonLatRad = conversions::cart2sph( point );
        const auto  hit = findPlateInStage( stage, geometry::lonLatDegToUnit( lonLatRad( 0 ), lonLatRad( 1 ) ) );

        vec3D v{ 0, 0, 0 };

        if ( hit.found && stage.hasRotation( hit.plateIndex ) != 0 )
        {
            const double distanceKm = hit.distanceRad * plates::constants::earthRadiusInKm;

            if ( stencil.maxDistanceKm < distanceKm )
            {
                // Comfortably inside a plate: the whole stencil would land on the same plate, so averaging
                // would change nothing. This is the shortcut the host query takes as well.
                vec3D omega;
                for ( int d = 0; d < 3; ++d )
                    omega( d ) = stage.omega( hit.plateIndex, d );

                v = eulerVectorToVelocity( lonLatRad, omega, 1.0 );
            }
            else
            {
                // Close enough to a boundary that the stencil may straddle it. Average over the stencil,
                // skipping samples that land on no plate, then project out the radial component: blending
                // velocities of differently-rotating plates sampled at different points does not in general
                // leave a tangential vector.
                vec3D first, second;
                tangent_frame( point, first, second );

                vec3D  accumulated{ 0, 0, 0 };
                double weight_sum = 0;

                for ( int k = 0; k < stencil.size(); ++k )
                {
                    const vec3D sample_cart =
                        point + stencil.offsets( k, 0 ) * first + stencil.offsets( k, 1 ) * second;

                    vec3D sample_lonLatRad = conversions::cart2sph( sample_cart );
                    sample_lonLatRad( 2 )  = lonLatRad( 2 );

                    vec3D sample_v;
                    if ( velocity_at( sample_lonLatRad, sample_v ) )
                    {
                        const double w = stencil.offsets( k, 2 );
                        for ( int d = 0; d < 3; ++d )
                            accumulated( d ) += w * sample_v( d );
                        weight_sum += w;
                    }
                }

                if ( weight_sum > 0 )
                {
                    const vec3D  n   = point.normalized();
                    const double dot = n( 0 ) * accumulated( 0 ) + n( 1 ) * accumulated( 1 ) +
                                       n( 2 ) * accumulated( 2 );

                    for ( int d = 0; d < 3; ++d )
                        v( d ) = accumulated( d ) / weight_sum - ( dot / weight_sum ) * n( d );
                }
            }
        }

        for ( int d = 0; d < 3; ++d )
            velocity( sd, x, y, surface_r, d ) = static_cast< ScalarType >( v( d ) ) * scale;
    }
};

/// @brief Fills the outermost shell of `velocity` with rigid-plate velocities, entirely on the device.
///
/// `stage` must have been prepared for the requested age (PlateVelocityProvider::prepareEulerVectors), which is
/// also what fills in the per-plate Euler vectors this reads.
template < typename ScalarType >
void extract_plate_velocities_device(
    const grid::shell::DistributedDomain&                                          domain,
    const grid::Grid3DDataVec< ScalarType, 3 >&                                    coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                                    coords_radii,
    const PlateStageViews< typename Kokkos::DefaultExecutionSpace::memory_space >& stage,
    const DeviceAveragingStencil< typename Kokkos::DefaultExecutionSpace::memory_space >& stencil,
    grid::Grid4DDataVec< ScalarType, 3 >&                                          velocity,
    const ScalarType                                                               scale )
{
    const int num_sub   = static_cast< int >( domain.subdomains().size() );
    const int n_lat     = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int surface_r = domain.domain_info().subdomain_num_nodes_radially() - 1;

    Kokkos::parallel_for(
        "extract_plate_velocities_device",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_sub, n_lat, n_lat } ),
        DevicePlateVelocityInterpolator< ScalarType >{
            coords_shell, coords_radii, velocity, stage, stencil, surface_r, scale } );
    Kokkos::fence();
}

/// @brief Counts surface nodes close enough to a plate boundary that the host path would average over its
///        stencil rather than take the unaveraged shortcut.
///
/// Diagnostic: reports how many points take the averaged branch rather than the rigid-rotation shortcut.
/// `max_distance_km` is the stencil reach, in the same units as the lookup distance.
template < typename ScalarType >
long long surface_points_needing_averaging(
    const grid::shell::DistributedDomain&                                          domain,
    const grid::Grid3DDataVec< ScalarType, 3 >&                                    coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                                    coords_radii,
    const PlateStageViews< typename Kokkos::DefaultExecutionSpace::memory_space >& stage,
    const double                                                                   max_distance_km )
{
    const int num_sub   = static_cast< int >( domain.subdomains().size() );
    const int n_lat     = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int surface_r = domain.domain_info().subdomain_num_nodes_radially() - 1;

    long long count = 0;

    Kokkos::parallel_reduce(
        "surface_points_needing_averaging",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_sub, n_lat, n_lat } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, long long& acc ) {
            const auto coords = grid::shell::coords( sd, x, y, surface_r, coords_shell, coords_radii );
            const vec3D lonLatRad = conversions::cart2sph( vec3D{ coords( 0 ), coords( 1 ), coords( 2 ) } );
            const auto  unit      = geometry::lonLatDegToUnit( lonLatRad( 0 ), lonLatRad( 1 ) );

            const auto hit = findPlateInStage( stage, unit );
            if ( hit.found && max_distance_km >= hit.distanceRad * plates::constants::earthRadiusInKm )
                acc += 1;
        },
        count );
    Kokkos::fence();

    return count;
}

} // namespace terra::plates
