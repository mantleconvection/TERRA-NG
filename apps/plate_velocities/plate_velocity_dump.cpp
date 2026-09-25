/// Dumps plate surface velocities on a fixed lon/lat grid, for comparing implementations.
///
/// The grid is independent of any mesh so that two builds of different revisions sample exactly the same
/// points. Output is CSV: lon,lat,vx,vy,vz.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "terra/plates/local_averaging_point_weight_provider.hpp"
#include "terra/plates/plate_not_found_handlers.hpp"
#include "terra/plates/plate_velocity_provider.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using namespace terra;


int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    if ( argc < 5 )
    {
        std::printf( "usage: %s <topologies> <rotations> <out.csv> <step_deg> [age]\n", argv[0] );
        return EXIT_FAILURE;
    }

    const std::string topologies = argv[1];
    const std::string rotations  = argv[2];
    const std::string outPath    = argv[3];
    const double      step       = std::atof( argv[4] );
    const double      age        = argc > 5 ? std::atof( argv[5] ) : 0.0;

    plates::PlateVelocityProvider oracle( topologies, rotations );

    // The stencil the mantle circulation app uses.
    const plates::UniformCirclesPointWeightProvider stencil( { { 1.0 / 100.0, 6 } }, 1e-1 );

    // The current implementation only takes its packed (binned, capped) lookup path once the stage has been
    // prepared; without this it would fall back to the unpacked polygon search and we would be comparing the
    // wrong thing.
    oracle.prepareEulerVectors( age );

    std::FILE* out = std::fopen( outPath.c_str(), "w" );
    std::fprintf( out, "lon,lat,vx,vy,vz\n" );

    const double deg2rad = M_PI / 180.0;

    for ( double lat = -89.0; lat <= 89.0 + 1e-9; lat += step )
    {
        for ( double lon = -180.0; lon < 180.0 - 1e-9; lon += step )
        {
            // Unit-radius cartesian point, matching what the shell mesh hands to the query at r_max = 1.
            const double clat = std::cos( lat * deg2rad );
            const vec3D  point{ clat * std::cos( lon * deg2rad ), clat * std::sin( lon * deg2rad ),
                               std::sin( lat * deg2rad ) };

            const vec3D v =
                oracle.getLocallyAveragedPointVelocity( point, age, stencil,
                                                        plates::DefaultPlateNotFoundHandler{} );

            std::fprintf( out, "%.6f,%.6f,%.17e,%.17e,%.17e\n", lon, lat, v( 0 ), v( 1 ), v( 2 ) );
        }
    }

    std::fclose( out );
    util::logroot << "wrote " << outPath << std::endl;
    return EXIT_SUCCESS;
}
