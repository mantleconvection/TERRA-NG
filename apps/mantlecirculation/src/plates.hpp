#pragma once

#include <string>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1.hpp"
#include "parameters.hpp"
#include "terra/plates/plate_velocity_calculator.hpp"
#include "terra/plates/plate_velocity_provider.hpp"
#include "terra/plates/types.hpp"
#include "util/logging.hpp"
#include "util/timer.hpp"

namespace terra::mantlecirculation {

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataVec;

template < typename GridType, typename RadiiType, typename DataType, typename VelocityFn >
struct Compute_plate_velocities
{
    GridType   grid_;
    RadiiType  radii_;
    DataType   plate_data_;
    VelocityFn computeVelocity;

    Compute_plate_velocities(
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
        for ( int d = 0; d < 3; ++d )
            plate_data_( id, x, y, radii_.extent( 1 ) - 1, d ) = v( d );
    }
};

void extract_plate_velocities(
    ScalarType                            plate_age,
    Grid4DDataVec< ScalarType, 3 >&       plate_velocities,
    plates::PlateVelocityProvider&        oracle,
    const Grid3DDataVec< ScalarType, 3 >& coords_shell,
    const Grid2DDataScalar< ScalarType >& coords_radii,
    const bool                            interpolate_in_time,
    const ScalarType                      scale_factor,
    const grid::shell::DistributedDomain& domain )
{
    util::Timer timer_plates( "plate_velocities" );

    plates::StatisticsPlateNotFoundHandler errorHandler;

    const plates::UniformCirclesPointWeightProvider weights( { { 1.0 / 100.0, 6 } }, 1e-1 );
    const auto                                      stencil = plates::make_averaging_stencil( weights );

    ScalarType plate_age_ceil;
    ScalarType plate_age_floor;
    ScalarType interpolation_factor;

    const ScalarType remainder = std::ceil( plate_age ) - plate_age;

    if ( !interpolate_in_time )
        plate_age = std::ceil( plate_age );
    else
    {
        plate_age_ceil       = std::ceil( plate_age );
        plate_age_floor      = std::ceil( plate_age ) - 1;
        interpolation_factor = ( plate_age - plate_age_floor ) / ( plate_age_ceil - plate_age_floor );
    }

    util::logroot << "Updating plates..... Plate age: " << plate_age << " Ma." << std::endl;

    // Hoist the reconstruction-tree walk out of the per-point loop below. Without this every
    // sample point re-derives the stage pole of its plate, which dominates the extraction cost.
    if ( interpolate_in_time )
        oracle.prepareEulerVectorsInterpolatedInTime( plate_age );
    else
        oracle.prepareEulerVectors( plate_age );

    if ( !interpolate_in_time || remainder == 0 )
    {
        plates::extract_plate_velocities< ScalarType >(
            domain,
            coords_shell,
            coords_radii,
            oracle.stageFor( plate_age ).data(),
            stencil,
            plate_velocities,
            scale_factor );

        util::logroot << "Plate data extracted." << std::endl;
        return;
    }
    else
    {
        plates::extract_plate_velocities_interpolated_in_time< ScalarType >(
            domain,
            coords_shell,
            coords_radii,
            plate_age,
            oracle.stageFor( plate_age_ceil ).data(),
            oracle.stageFor( plate_age_floor ).data(),
            stencil,
            plate_velocities,
            scale_factor,
            interpolation_factor );

        util::logroot << "Plate data extracted (interpolated in time)." << std::endl;
        return;
    }
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