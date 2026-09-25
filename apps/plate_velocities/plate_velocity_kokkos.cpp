/*
 * Copyright (c) 2017-2022 Dominik Thoennes, Nils Kohl, Marcus Mohr, Fatemeh Rezaei.
 *
 * This file is part of HyTeG
 * (see https://i10git.cs.fau.de/hyteg/hyteg).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <cmath>
#include <csignal>
#include <sstream>
#include <string>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "mpi/mpi.hpp"
#include "src/parameters.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/plates/plate_velocity_device.hpp"
#include "terra/plates/plate_velocity_provider.hpp"
#include "terra/plates/types.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/timer.hpp"

namespace terra {
using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
// using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;

using grid::shell::DistributedDomain;

using util::logall;
using util::logroot;

// using linalg::VectorQ1IsoQ2Q1;

namespace plates {

typedef enum
{
    PLATE_IDS,
    VELOCITIES,
    VELOCITIES_AND_IDS
} job_t;

// The device-side plate lookup lives in src/terra/plates/plate_velocity_device.hpp and uses the packed
// stage views. The winding-sum test this file used to carry reported every point in the complement of a
// pole-enclosing polygon as inside, which with first-match-wins let early plates swallow later ones: 18
// distinct plates instead of 40 on a level-5 shell at 5 Ma.

} // namespace plates
} // namespace terra

using namespace terra;
using namespace plates;

struct Parameters
{
    uint_t      max_level = 6;
    uint_t      min_level = 1;
    double      r_min     = 0.55;
    double      r_max     = 1.0;
    std::string jobType   = "both";
    int         beginAge  = 10;
    int         endAge    = 0;
    std::string outdir    = "./output";
    std::string dataDir   = "../../../TERRA-NG/data/plates/Chen2025-tomopac/";
};

// ========
//  Driver
// ========
int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    // ------------
    //  Parameters
    // ------------

    logroot << "*** STEP 1: Obtaining Steering Parameters" << std::endl;

    // TO-DO : implement reading a parameter file
    // Fill with default parameters.

    Parameters parameters;

    CLI::App app{ "Plate Velocity Demo" };

    terra::util::add_option_with_default( app, "--max-level", parameters.max_level );
    terra::util::add_option_with_default( app, "--min-level", parameters.min_level );
    terra::util::add_option_with_default( app, "--r-min", parameters.r_min );
    terra::util::add_option_with_default( app, "--r-max", parameters.r_max );
    terra::util::add_option_with_default( app, "--jobType", parameters.jobType );
    terra::util::add_option_with_default( app, "--beginAge", parameters.beginAge );
    terra::util::add_option_with_default( app, "--endAge", parameters.endAge );
    terra::util::add_option_with_default( app, "--data-dir", parameters.dataDir );

    CLI11_PARSE( app, argc, argv );

    terra::util::prepare_empty_directory_or_abort( parameters.outdir );

    const auto xdmf_dir            = parameters.outdir + "/xdmf";
    const auto radial_profiles_dir = parameters.outdir + "/radial_profiles";
    const auto timer_trees_dir     = parameters.outdir + "/timer_trees";

    terra::util::prepare_empty_directory_or_abort( xdmf_dir );
    terra::util::prepare_empty_directory_or_abort( radial_profiles_dir );
    terra::util::prepare_empty_directory_or_abort( timer_trees_dir );

    logroot << "Running with the following steering parameters:" << std::endl;
    logroot << "--max-level " << parameters.max_level << std::endl;
    logroot << "--min-level " << parameters.min_level << std::endl;
    logroot << "--r-min " << parameters.r_min << std::endl;
    logroot << "--r-max " << parameters.r_max << std::endl;
    logroot << "--jobType " << parameters.jobType << std::endl;
    logroot << "--beginAge " << parameters.beginAge << std::endl;
    logroot << "--endAge " << parameters.endAge << std::endl;
    logroot << "--data-dir " << parameters.dataDir << std::endl;

    // make sure beginAge > endAge -- we simulate forward in time
    if ( parameters.endAge > parameters.beginAge )
    {
        logroot << "## Specified endAge is larger than beginAge. Aborting.." << std::endl;
        std::abort();
    }

    // determine job type
    terra::plates::job_t jobType;
    jobType = terra::plates::VELOCITIES_AND_IDS;

    // ---------
    //  Meshing
    // ---------

    logroot << "*** STEP 2: Generating Mesh" << std::endl;

    std::vector< terra::grid::shell::DistributedDomain >   domains;
    std::vector< terra::grid::Grid3DDataVec< double, 3 > > coords_shell;
    std::vector< terra::grid::Grid2DDataScalar< double > > coords_radii;

    for ( int level = parameters.min_level; level <= parameters.max_level; level++ ) // not needed
    {
        const int idx = level - parameters.min_level;

        domains.push_back( terra::grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond(
            level, level, parameters.r_min, parameters.r_max ) );
        coords_shell.push_back(
            terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domains[idx] ) );
        coords_radii.push_back( terra::grid::shell::subdomain_shell_radii< double >( domains[idx] ) );
    }

    const auto   num_levels     = domains.size();
    const uint_t velocity_level = num_levels - 1;
    logroot << "velocity level : " << velocity_level << std::endl;

    // Mirror mesh details to host
    auto coords_shell_host = Kokkos::create_mirror_view( coords_shell[velocity_level] );
    Kokkos::deep_copy( coords_shell_host, coords_shell[velocity_level] );
    auto coords_radii_host = Kokkos::create_mirror_view( coords_radii[velocity_level] );
    Kokkos::deep_copy( coords_radii_host, coords_radii[velocity_level] );

    terra::grid::Grid4DDataScalar< double > plate_id_fe(
        "plate_id_fe",
        domains[velocity_level].subdomains().size(),
        domains[velocity_level].domain_info().subdomain_num_nodes_per_side_laterally(),
        domains[velocity_level].domain_info().subdomain_num_nodes_per_side_laterally(),
        domains[velocity_level].domain_info().subdomain_num_nodes_radially() );

    io::XDMFOutput< double > xdmf_output(
        "./output-xdmf/", domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level] );
    xdmf_output.add( plate_id_fe );

    // --------
    //  Oracle
    // --------

    logroot << "*** STEP 3: Generating an Oracle" << std::endl;

    //std::string dataDir{ "/import/freenas-m-04-students/frezaei/TerraNeoX/TERRA-NG/apps/plate_velocities/data/plates/" };
    const std::string& dataDir = parameters.dataDir;
    std::string fnameTopologies      = dataDir + "topologies_0-410Ma.geojson";
    std::string fnameReconstructions = dataDir + "TomoPAC2.rot";
    terra::plates::PlateVelocityProvider oracle( fnameTopologies, fnameReconstructions );

    logroot << "*** STEP 4: Checking plate stages to work with" << std::endl;

    auto stages = oracle.getListOfPlateStages();

    if ( parameters.beginAge > oracle.getMaxAge() || parameters.endAge < oracle.getMinAge() )
    {
        logroot << "Specified age range extends beyond available plate data. " << std::endl;
        std::abort();
    }

    logroot << " - Running from " << parameters.beginAge << " Ma to " << parameters.endAge << " Ma" << std::endl;

    // Prepare the stage: this is what packs the plate polygons (and Euler vectors) into the device views the
    // lookup reads.
    oracle.prepareEulerVectors( static_cast< double >( parameters.endAge ) );
    const auto& stage = oracle.stageFor( static_cast< double >( parameters.endAge ) );

    logroot << " - " << stage.device().nPlates << " plates at stage " << parameters.endAge << " Ma" << std::endl;

    terra::plates::extract_plate_ids_device< double >(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        stage.device(),
        plate_id_fe );

    xdmf_output.write( 0 );

    return EXIT_SUCCESS;
}
