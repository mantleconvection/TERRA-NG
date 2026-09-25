#pragma once

#include <string>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1.hpp"
#include "parameters.hpp"
#include "terra/plates/plate_velocity_device.hpp"
#include "terra/plates/plate_velocity_provider.hpp"
#include "terra/plates/types.hpp"
#include "util/logging.hpp"
#include "util/timer.hpp"

namespace terra::mantlecirculation {

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataVec;

template < typename GridType, typename RadiiType, typename DataType, typename VelocityFn >
struct Computeplate_velocities
{
    GridType   grid_;
    RadiiType  radii_;
    DataType   plate_data_;
    VelocityFn computeVelocity;

    Computeplate_velocities(
        const GridType&  grid,
        const RadiiType& radii,
        const DataType&  plate_data,
        VelocityFn       velocityFn )
    : grid_( grid )
    , radii_( radii )
    , plate_data_( plate_data )
    , computeVelocity( std::move( velocityFn ) )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y ) const
    {
        const dense::Vec< ScalarType, 3 > coords =
            grid::shell::coords( id, x, y, radii_.extent( 1 ) - 1, grid_, radii_ );

        const vec3D v = computeVelocity( coords );
        // Grid4DDataVec is SoA here (comp_[d]), and its host mirror has no const-qualified
        // operator(), so go through the component views directly.
        for ( int d = 0; d < 3; ++d )
            plate_data_.comp_[d]( id, x, y, radii_.extent( 1 ) - 1 ) = v( d );
    }
};

/// @param on_device evaluate the velocities in a Kokkos kernel from the packed stage views rather than by
///                  querying the oracle per point on the host. Ignored when interpolating in time, which
///                  blends two stages and has no device path yet.
void extract_plate_velocities(
    ScalarType                            plate_age,
    Grid4DDataVec< ScalarType, 3 >&       plate_velocities,
    plates::PlateVelocityProvider&        oracle,
    const Grid3DDataVec< ScalarType, 3 >& coords_shell,
    const Grid2DDataScalar< ScalarType >& coords_radii,
    const bool                            interpolate_in_time,
    const ScalarType                      scale_factor,
    const grid::shell::DistributedDomain*  domain_for_device = nullptr,
    const bool                             on_device         = false )
{
    util::Timer timer_plates( "plate_velocities" );

    using HostExecSpace = Kokkos::DefaultHostExecutionSpace;

    plates::StatisticsPlateNotFoundHandler    errorHandler;
    plates::UniformCirclesPointWeightProvider pointWeightProvider( { { 1.0 / 100.0, 6 } }, 1e-1 );

    if ( !interpolate_in_time )
        plate_age = std::ceil( plate_age );

    util::logroot << "Updating plates..... Plate age: " << plate_age << " Ma." << std::endl;

    // Hoist the reconstruction-tree walk out of the per-point loop below. Without this every
    // sample point re-derives the stage pole of its plate, which dominates the extraction cost.
    if ( interpolate_in_time )
        oracle.prepareEulerVectorsInterpolatedInTime( plate_age );
    else
        oracle.prepareEulerVectors( plate_age );

    if ( on_device && !interpolate_in_time && domain_for_device != nullptr )
    {
        const plates::UniformCirclesPointWeightProvider weights( { { 1.0 / 100.0, 6 } }, 1e-1 );
        const auto stencil = plates::make_device_averaging_stencil( weights );

        plates::extract_plate_velocities_device< ScalarType >(
            *domain_for_device,
            coords_shell,
            coords_radii,
            oracle.stageFor( plate_age ).device(),
            stencil,
            plate_velocities,
            scale_factor );

        util::logroot << "Plate data extracted (device)." << std::endl;
        return;
    }

    // Mirror the needed Kokkos::Views to the host
    auto coords_host = Kokkos::create_mirror_view( coords_shell );
    Kokkos::deep_copy( coords_host, coords_shell );
    auto radii_host = Kokkos::create_mirror_view( coords_radii );
    Kokkos::deep_copy( radii_host, coords_radii );

    // Copy plate data object to host, using Grid4DDataVec's own overloads
    // and make sure to zero-initialize on host-side.
    auto plate_velocities_host = create_mirror( Kokkos::HostSpace{}, plate_velocities );
    for ( int d = 0; d < 3; ++d )
        Kokkos::deep_copy( plate_velocities_host.comp_[d], ScalarType( 0 ) );

    // Callback function for computing velocity components
    auto getPointVelocity =
        [&oracle, plate_age, &pointWeightProvider, &errorHandler, interpolate_in_time, scale_factor](
            const vec3D& point ) {
            vec3D velocity;
            if ( interpolate_in_time )
            {
                velocity = oracle.getLocallyAveragedPointVelocityInterpolatedInTime(
                    point, plate_age, pointWeightProvider, errorHandler );
            }

            else
            {
                velocity =
                    oracle.getLocallyAveragedPointVelocity( point, plate_age, pointWeightProvider, errorHandler );
            }

            // Nondimensionalise and scale before returning
            return velocity * scale_factor;
        };

    // Extract plate velocities from data
    // Explicitly on host since underlying plates functionality is not device-callable.
    Kokkos::parallel_for(
        "Computeplate_velocities",
        Kokkos::MDRangePolicy< HostExecSpace, Kokkos::Rank< 3 > >(
            { 0, 0, 0 }, { coords_host.extent( 0 ), coords_host.extent( 1 ), coords_host.extent( 2 ) } ),
        Computeplate_velocities( coords_host, radii_host, plate_velocities_host, getPointVelocity ) );
    Kokkos::fence();

    // Copy to device
    deep_copy( plate_velocities, plate_velocities_host );

    util::logroot << "Plate data extracted." << std::endl;
}

inline std::shared_ptr< plates::PlateVelocityProvider >
    initialise_plates( const std::string& fnameTopologies, const std::string& fnameReconstructions )
{
    std::shared_ptr< plates::PlateVelocityProvider > oracle;

    util::logroot << "Setting up Oracle for plates." << std::endl;

    oracle = std::make_shared< plates::PlateVelocityProvider >( fnameTopologies, fnameReconstructions );

    return oracle;
}

} // namespace terra::mantlecirculation