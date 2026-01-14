
#include <fstream>
#include <iomanip>
#include <linalg/vector_q1.hpp>
#include <optional>

#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/vtk.hpp"
#include "terra/io/xdmf.hpp"
#include "util/cli11_helper.hpp"
#include "util/cli11_wrapper.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using terra::util::add_flag_with_default;
using terra::util::add_option_with_default;
using terra::util::logroot;

struct Parameters
{
    double r_min = 0.5;
    double r_max = 1.0;

    int                   lateral_refinement_level = 2;
    int                   radial_refinement_level  = 2;
    std::vector< double > radii                    = { -1.0 };

    int lateral_subdomain_level = 0;
    int radial_subdomain_level  = 0;

    std::string output_directory = "visualize_thick_spherical_shell_mesh_output";
};

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    const int rank = terra::mpi::rank();

    const auto description = "Thick Spherical Mesh Visualizer - fancy XDMF output to illustrate partitioning.";
    CLI::App   app{ description };

    Parameters parameters{};

    add_option_with_default( app, "--output-dir", parameters.output_directory, "XDMF output directory." );

    add_option_with_default(
        app,
        "--lateral-refinement-level",
        parameters.lateral_refinement_level,
        "Refinement level in lateral direction." );

    auto radial_level_option = add_option_with_default(
        app,
        "--radial-refinement-level",
        parameters.radial_refinement_level,
        "Radial refinement level (uniform refinement)." );

    add_option_with_default( app, "--radii", parameters.radii, "Explicit list of shell radii." )
        ->excludes( radial_level_option );

    add_option_with_default(
        app,
        "--lateral-subdomain-level",
        parameters.lateral_subdomain_level,
        "Subdomain refinement level in lateral direction." );

    add_option_with_default(
        app,
        "--radial-subdomain-level",
        parameters.radial_subdomain_level,
        "Subdomain refinement level in radial direction." );

    // Parse arguments (it's a simple macro that handles exceptions nicely).
    CLI11_PARSE( app, argc, argv );

    logroot << "\n" << description << "\n\n";

    // Print overview.

    terra::util::print_cli_summary( app, logroot );
    logroot << "\n";

    if ( parameters.radii.size() < 2 )
    {
        logroot << "Applying uniform radial refinement." << std::endl;

        parameters.radii = terra::grid::shell::uniform_shell_radii(
            parameters.r_min, parameters.r_max, ( 1 << parameters.radial_refinement_level ) + 1 );
    }
    else
    {
        logroot << "Using specified shell radii." << std::endl;
    }
    logroot << "Shell radii:" << std::endl;
    for ( auto r : parameters.radii )
    {
        logroot << "  " << r << std::endl;
    }

    logroot << std::endl;

    const auto domain = terra::grid::shell::DistributedDomain::create_uniform(
        parameters.lateral_refinement_level,
        parameters.radii,
        parameters.lateral_subdomain_level,
        parameters.radial_subdomain_level,
        terra::grid::shell::subdomain_to_rank_iterate_diamond_subdomains );

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< double >( domain );

    auto mask_data = terra::grid::setup_node_ownership_mask_data( domain );

    auto grid_rank               = terra::grid::shell::allocate_scalar_grid< double >( "rank", domain );
    auto grid_diamond_id         = terra::grid::shell::allocate_scalar_grid< double >( "diamond_id", domain );
    auto grid_local_subdomain_id = terra::grid::shell::allocate_scalar_grid< double >( "local_subdomain_id", domain );

    Kokkos::parallel_for(
        "rank_interpolation",
        terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( int local_subdomain_idx, int x, int y, int r ) {
            grid_rank( local_subdomain_idx, x, y, r ) = rank;
        } );

    for ( const auto& [subdomain, local_subdomain_id_and_neighborhood] : domain.subdomains() )
    {
        auto [local_subdomain_id, neighborhood] = local_subdomain_id_and_neighborhood;

        const auto diamond_id = subdomain.diamond_id();

        Kokkos::parallel_for(
            "diamond_interpolation",
            terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( int local_subdomain_idx, int x, int y, int r ) {
                if ( local_subdomain_idx == local_subdomain_id )
                {
                    grid_diamond_id( local_subdomain_idx, x, y, r ) = diamond_id;
                }
            } );
    }

    Kokkos::parallel_for(
        "diamond_interpolation",
        terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( int local_subdomain_idx, int x, int y, int r ) {
            grid_local_subdomain_id( local_subdomain_idx, x, y, r ) = local_subdomain_idx;
        } );

    terra::io::XDMFOutput xdmf( parameters.output_directory, domain, subdomain_shell_coords, subdomain_radii );

    xdmf.add( grid_rank );
    xdmf.add( grid_diamond_id );
    xdmf.add( grid_local_subdomain_id );

    logroot << "Writing output to directory: " << parameters.output_directory << "\n\n";

    xdmf.write();

    logroot << "Bye :)" << std::endl;

    return 0;
}