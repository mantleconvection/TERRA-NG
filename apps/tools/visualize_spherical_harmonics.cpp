
#include <fstream>
#include <iomanip>
#include <linalg/vector_q1.hpp>
#include <optional>

#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/vtk.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/shell/spherical_harmonics.hpp"
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
    int lateral_refinement_level = 4;
    int lateral_subdomain_level  = 0;

    int degree_l;
    int order_m;

    std::string output_directory = "visualize_spherical_harmonics_output";
};

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    const auto description = "Visualize Spherical Harmonics - XDMF output of coefficient grids.";
    CLI::App   app{ description };

    Parameters parameters{};

    app.add_option( "-l,--degree", parameters.degree_l, "Degree of spherical harmonics." )->required();
    app.add_option( "-m,--order", parameters.order_m, "Order of spherical harmonics." )->required();

    add_option_with_default( app, "--output-dir", parameters.output_directory, "XDMF output directory." );

    add_option_with_default(
        app,
        "--lateral-refinement-level",
        parameters.lateral_refinement_level,
        "Refinement level in lateral direction." );

    add_option_with_default(
        app,
        "--lateral-subdomain-level",
        parameters.lateral_subdomain_level,
        "Subdomain refinement level in lateral direction." );

    // Parse arguments (it's a simple macro that handles exceptions nicely).
    CLI11_PARSE( app, argc, argv );

    logroot << "\n" << description << "\n\n";

    // Print overview.

    terra::util::print_cli_summary( app, logroot );
    logroot << "\n";
    logroot << std::endl;

    const auto domain = terra::grid::shell::DistributedDomain::create_uniform(
        parameters.lateral_refinement_level,
        0,
        0.5,
        1.0,
        parameters.lateral_subdomain_level,
        0,
        terra::grid::shell::subdomain_to_rank_iterate_diamond_subdomains );

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< double >( domain );

    auto mask_data = terra::grid::setup_node_ownership_mask_data( domain );

    auto sph_data = terra::shell::spherical_harmonics_coefficients_grid< double >(
        parameters.degree_l, parameters.order_m, subdomain_shell_coords );

    auto sph_data_for_xdmf = allocate_scalar_grid< double >( "sph_data_for_xdmf", domain );

    Kokkos::parallel_for(
        "sph_interpolation_for_xdmf",
        terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( int local_subdomain_idx, int x, int y, int r ) {
            sph_data_for_xdmf( local_subdomain_idx, x, y, r ) = sph_data( local_subdomain_idx, x, y );
        } );

    terra::io::XDMFOutput xdmf( parameters.output_directory, domain, subdomain_shell_coords, subdomain_radii );

    xdmf.add( sph_data_for_xdmf );

    logroot << "Writing output to directory: " << parameters.output_directory << "\n\n";

    xdmf.write();

    logroot << "Bye :)" << std::endl;

    return 0;
}