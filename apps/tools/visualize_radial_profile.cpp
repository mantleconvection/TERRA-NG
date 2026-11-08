
#include <fstream>
#include <iomanip>
#include <linalg/vector_w1.hpp>
#include <optional>

#include "shell/radial_profiles.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/shell/spherical_harmonics.hpp"
#include "util/cli11_helper.hpp"
#include "util/cli11_wrapper.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"

using terra::util::add_flag_with_default;
using terra::util::add_option_with_default;
using terra::util::logroot;

struct Parameters
{
    double r_min = 0.5;
    double r_max = 1.0;

    int lateral_refinement_level = 4;
    int radial_refinement_level  = 4;

    std::string                radial_profile_filename;
    std::string                radial_profile_radii_key;
    std::vector< std::string > radial_profile_data_keys;

    std::string output_directory = "visualize_radial_profile_output";
};

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    const auto description = "Visualizes radial profile read from CSV (writes XDMF).";
    CLI::App   app{ description };

    Parameters parameters{};

    app.add_option( "-f,--profile-file", parameters.radial_profile_filename, "Path to the profile .csv." )->required();

    app.add_option(
           "-r,--profile-radii-keys",
           parameters.radial_profile_radii_key,
           "Key in the CSV file that corresponds to profile radii." )
        ->required();
    app.add_option(
           "-k,--profile-data-keys",
           parameters.radial_profile_data_keys,
           "Keys in the CSV file that correspond to profile values." )
        ->required();

    add_option_with_default( app, "--output-dir", parameters.output_directory, "XDMF output directory." );

    add_option_with_default( app, "--r-min", parameters.r_min, "Inner radius of thick spherical shell." );
    add_option_with_default( app, "--r-max", parameters.r_max, "Outer radius of thick spherical shell." );

    add_option_with_default(
        app,
        "--lateral-refinement-level",
        parameters.lateral_refinement_level,
        "Refinement level in lateral direction." );

    add_option_with_default(
        app, "--radial-refinement-level", parameters.radial_refinement_level, "Refinement level in radial direction." );

    // Parse arguments (it's a simple macro that handles exceptions nicely).
    CLI11_PARSE( app, argc, argv );

    logroot << "\n" << description << "\n\n";

    // Print overview.

    terra::util::print_cli_summary( app, logroot );
    logroot << std::endl;

    // Load csv and print as table.

    auto profile_table_result = terra::util::read_table_from_csv( parameters.radial_profile_filename );
    if ( profile_table_result.is_err() )
    {
        logroot << profile_table_result.error() << std::endl;
        return EXIT_FAILURE;
    }
    const auto& profile_table = profile_table_result.unwrap();
    auto        all_keys      = parameters.radial_profile_data_keys;
    all_keys.insert( all_keys.begin(), parameters.radial_profile_radii_key );
    profile_table.select_columns( all_keys ).print_pretty();
    logroot << std::endl;

    const auto profile_radii = profile_table.column_as_vector< double >( parameters.radial_profile_radii_key );

    // Create domain for visualization.

    const auto domain = terra::grid::shell::DistributedDomain::create_uniform(
        parameters.lateral_refinement_level,
        parameters.radial_refinement_level,
        parameters.r_min,
        parameters.r_max,
        0,
        0,
        terra::grid::shell::subdomain_to_rank_iterate_diamond_subdomains );

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< double >( domain );

    auto mask_data = terra::grid::setup_node_ownership_mask_data( domain );

    terra::io::XDMFOutput xdmf( parameters.output_directory, domain, subdomain_shell_coords, subdomain_radii );

    for ( const auto& key : parameters.radial_profile_data_keys )
    {
        logroot << "Processing profile data for key: " << key << std::endl;

        auto profile = profile_table.column_as_vector< double >( key );

        const auto profiles_device =
            terra::shell::interpolate_radial_profile_into_subdomains( key, subdomain_radii, profile_radii, profile );

        const auto profile_interpolated_on_shell = terra::grid::shell::allocate_scalar_grid< double >( key, domain );

        Kokkos::parallel_for(
            "radial_profile_interpolation",
            terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( int local_subdomain_idx, int x, int y, int r ) {
                profile_interpolated_on_shell( local_subdomain_idx, x, y, r ) =
                    profiles_device( local_subdomain_idx, r );
            } );

        xdmf.add( profile_interpolated_on_shell );
    }

    logroot << "\nWriting output to directory: " << parameters.output_directory << "\n\n";

    xdmf.write( 0 );

    logroot << "Bye :)" << std::endl;

    return 0;
}
