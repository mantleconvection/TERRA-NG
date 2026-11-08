#pragma once

#include <string>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_w1.hpp"
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

    // Set plate_age to the ceiling integer value if no interpolation in time
    // is wanted. Holds only as long as plate data is defined in 1 Ma steps.
    // Everything else regarding time-averaging is handled inside the plates
    // module.
    if ( !interpolate_in_time )
        plate_age = std::ceil( plate_age );

    util::logroot << "Updating plates..... Plate age: " << plate_age << " Ma." << std::endl;

    plates::extract_plate_velocities< ScalarType >(
        plate_age, plate_velocities, oracle, coords_shell, coords_radii, scale_factor, domain, stencil );

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