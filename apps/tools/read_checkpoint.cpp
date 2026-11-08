
#include <bit>
#include <fstream>
#include <iomanip>
#include <linalg/vector_w1.hpp>
#include <optional>

#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "util/cli11_helper.hpp"
#include "util/cli11_wrapper.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using terra::util::add_flag_with_default;
using terra::util::add_option_with_default;
using terra::util::logroot;

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    const auto description = "Parses checkpoint. \n\n"
                             "Can be used to print metadata or to check integrity via reading in\n"
                             "the full checkpoint and writing the checkpoint again. The resulting\n"
                             "XDMF can be inspected via Paraview or similar tools.\n\n"
                             "Run with -h for help.";
    CLI::App   app{ description };

    std::string checkpoint_directory;
    app.add_option( "checkpoint-directory", checkpoint_directory, "Checkpoint directory" )->required();

    std::string output_directory;
    add_option_with_default(
        app,
        "-o,--output-directory",
        output_directory,
        "Output directory. If omitted this tool only parses the checkpoint metadata and exits. "
        "If supplied with a path, then the checkpoint is read and then the corresponding domain is built and an XDMF "
        "'copy' of the checkpoint is written." );

    CLI11_PARSE( app, argc, argv );

    logroot << description << "\n\n";

    const auto checkpoint_metadata_result = terra::io::read_xdmf_checkpoint_metadata( checkpoint_directory );
    if ( checkpoint_metadata_result.is_err() )
    {
        logroot << checkpoint_metadata_result.error() << std::endl;
        return EXIT_FAILURE;
    }

    const auto& checkpoint_metadata = checkpoint_metadata_result.unwrap();

    logroot << "===================================================================================\n";
    logroot << "checkpoint directory:                         " << checkpoint_directory << std::endl;
    logroot << "===================================================================================\n";
    logroot << "version:                                      " << checkpoint_metadata.version << std::endl;
    logroot << "num_subdomains_per_diamond_lateral_direction: "
            << checkpoint_metadata.num_subdomains_per_diamond_lateral_direction << std::endl;
    logroot << "num_subdomains_per_diamond_radial_direction:  "
            << checkpoint_metadata.num_subdomains_per_diamond_radial_direction << std::endl;
    logroot << "subdomain_size_x:                             " << checkpoint_metadata.size_x << std::endl;
    logroot << "subdomain_size_y:                             " << checkpoint_metadata.size_y << std::endl;
    logroot << "subdomain_size_r:                             " << checkpoint_metadata.size_r << std::endl;

    for ( const auto& grid_data_file : checkpoint_metadata.grid_data_files )
    {
        logroot << " + grid_data_file:                            " << grid_data_file.grid_name_string << std::endl;
        logroot << "   scalar_data_type:                          " << grid_data_file.scalar_data_type
                << " (0 = int, 1 = uint, 2 = float)" << std::endl;
        logroot << "   scalar_bytes:                              " << grid_data_file.scalar_bytes << std::endl;
        logroot << "   vec_dim:                                   " << grid_data_file.vec_dim << std::endl;
    }

    logroot << "checkpoint_subdomain_ordering:                " << checkpoint_metadata.checkpoint_subdomain_ordering
            << " (relevant for checkpoint data format)" << std::endl;

    logroot << "subdomains:                                   "
            << checkpoint_metadata.checkpoint_ordering_0_global_subdomain_ids.size() << std::endl;
    for ( const auto& global_subdomain_id : checkpoint_metadata.checkpoint_ordering_0_global_subdomain_ids )
    {
        logroot << " + " << terra::grid::shell::SubdomainInfo( global_subdomain_id ) << std::endl;
    }

    logroot << "===================================================================================\n";

    const auto lateral_refinement_level = std::bit_width(
                                              static_cast< unsigned int >(
                                                  checkpoint_metadata.num_subdomains_per_diamond_lateral_direction *
                                                  ( checkpoint_metadata.size_x - 1 ) ) ) -
                                          1;
    const auto radial_refinement_level = std::bit_width(
                                             static_cast< unsigned int >(
                                                 checkpoint_metadata.num_subdomains_per_diamond_radial_direction *
                                                 ( checkpoint_metadata.size_r - 1 ) ) ) -
                                         1;

    const auto lateral_subdomain_refinement_level =
        std::bit_width(
            static_cast< unsigned int >( checkpoint_metadata.num_subdomains_per_diamond_lateral_direction ) ) -
        1;

    const auto radial_subdomain_refinement_level =
        std::bit_width(
            static_cast< unsigned int >( checkpoint_metadata.num_subdomains_per_diamond_radial_direction ) ) -
        1;

    logroot << "Some derived quantities: " << std::endl;
    logroot << "  lateral refinement level (diamond):                                          "
            << lateral_refinement_level << std::endl;
    logroot << "  radial refinement level (diamond):                                           "
            << radial_refinement_level << std::endl;
    logroot << "  lateral subdomain refinement level:                                          "
            << lateral_subdomain_refinement_level << std::endl;
    logroot << "  radial subdomain refinement level:                                           "
            << radial_subdomain_refinement_level << std::endl;
    logroot << "  number of hex cells (total):                                                 "
            << 10 * ( 1 << lateral_refinement_level ) * ( 1 << lateral_refinement_level ) *
                   ( 1 << radial_refinement_level )
            << std::endl;
    logroot << "  number of nodes (total, including duplicates at diamond-diamond boundaries): "
            << 10 * ( ( 1 << lateral_refinement_level ) + 1 ) * ( ( 1 << lateral_refinement_level ) + 1 ) *
                   ( ( 1 << radial_refinement_level ) + 1 )
            << std::endl;
    logroot << "===================================================================================\n";

    if ( !output_directory.empty() )
    {
        logroot << "\nReading full checkpoint and writing XDMF to directory: " << output_directory << "\n\n";

        const auto domain = terra::grid::shell::DistributedDomain::create_uniform(
            lateral_refinement_level,
            checkpoint_metadata.radii,
            lateral_subdomain_refinement_level,
            radial_subdomain_refinement_level );

        const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
        const auto coords_radii = terra::grid::shell::subdomain_shell_radii< double >( domain );

        auto mask_data = terra::grid::setup_node_ownership_mask_data( domain );

        terra::io::XDMFOutput xdmf( output_directory, domain, coords_shell, coords_radii );

        for ( const auto& grid_data_file : checkpoint_metadata.grid_data_files )
        {
            if ( grid_data_file.vec_dim == 1 )
            {
                if ( grid_data_file.scalar_data_type == 2 )
                {
                    if ( grid_data_file.scalar_bytes == 4 )
                    {
                        terra::linalg::VectorW1Scalar< float > vec(
                            grid_data_file.grid_name_string, domain, mask_data );
                        const auto result = terra::io::read_xdmf_checkpoint_grid(
                            checkpoint_directory, grid_data_file.grid_name_string, 0, domain, vec.grid_data() );
                        if ( result.is_err() )
                        {
                            logroot << "Failed to read checkpoint for " << grid_data_file.grid_name_string << ": "
                                    << result.error() << std::endl;
                            return EXIT_FAILURE;
                        }
                        xdmf.add( vec.grid_data() );
                    }
                    else if ( grid_data_file.scalar_bytes == 8 )
                    {
                        terra::linalg::VectorW1Scalar< double > vec(
                            grid_data_file.grid_name_string, domain, mask_data );
                        const auto result = terra::io::read_xdmf_checkpoint_grid(
                            checkpoint_directory, grid_data_file.grid_name_string, 0, domain, vec.grid_data() );
                        if ( result.is_err() )
                        {
                            logroot << "Failed to read checkpoint for " << grid_data_file.grid_name_string << ": "
                                    << result.error() << std::endl;
                            return EXIT_FAILURE;
                        }
                        xdmf.add( vec.grid_data() );
                    }
                    else
                    {
                        logroot << "Unknown scalar bytes: " << grid_data_file.scalar_bytes << std::endl;
                        return EXIT_FAILURE;
                    }
                }
            }
        }

        xdmf.write( 0 );
    }
    logroot << "\nBye :)" << std::endl;

    return 0;
}
