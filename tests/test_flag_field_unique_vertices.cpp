#include "io/xdmf.hpp"
#include "kernels/common/grid_operations.hpp"
#include "linalg/vector_w1.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"

/// For dot products to work correctly, we need to properly define vertex ownership.
/// Neighboring subdomains share vertices, and to uniquely mark them, we create a mask View, that is either 1 or 0
/// at each vertex, such that each logically identical vertex is only marked 1 exactly once.
///
/// This test checks some properties to ensure this is implemented correctly.
void test( const int level, const int subdomain_level )
{
    // The refinement level defines the number of unknowns (for equal refinement in all directions).

    const int number_of_nodes_along_each_diamond_edge = ( 1 << level ) + 1;

    const int number_of_nodes =
        ( 10 * ( number_of_nodes_along_each_diamond_edge - 1 ) * ( number_of_nodes_along_each_diamond_edge - 1 ) + 2 ) *
        number_of_nodes_along_each_diamond_edge;

    const auto domain = terra::grid::shell::DistributedDomain::create_uniform(
        level, level, 0.5, 1.0, subdomain_level, subdomain_level );

    auto mask_data = terra::grid::setup_node_ownership_mask_data( domain );

    auto coords_lat = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    auto coords_rad = terra::grid::shell::subdomain_shell_radii< double >( domain );

    // Summing up all the mask entries should be equal to the number of nodes.
    // First casting to larger type to enable adding those up.

    auto ones           = terra::grid::shell::allocate_scalar_grid< long >( "ones", domain );
    auto ones_comm      = terra::grid::shell::allocate_scalar_grid< double >( "ones_comm", domain );
    auto mask_data_long = terra::grid::shell::allocate_scalar_grid< long >( "mask_data_long", domain );
    terra::kernels::common::set_constant( ones, 1l );
    terra::kernels::common::set_constant( ones_comm, 1.0 );
    terra::kernels::common::assign_masked_else_keep_old(
        mask_data_long, ones, mask_data, terra::grid::NodeOwnershipFlag::OWNED );

    const auto number_of_nodes_mask = terra::kernels::common::sum_of_absolutes( mask_data_long );

    std::cout << "Level:                                      " << level << std::endl;
    std::cout << "Subdomain level:                            " << subdomain_level << std::endl;
    std::cout << "Number of nodes (analytical):               " << number_of_nodes << std::endl;
    std::cout << "Number of nodes (mask):                     " << number_of_nodes_mask << std::endl;

    if ( number_of_nodes_mask != number_of_nodes )
    {
        throw std::logic_error( "Number of nodes does not match number of mask entries." );
    }

    const auto min_mask = terra::kernels::common::min_abs_entry( mask_data_long );
    const auto max_mask = terra::kernels::common::max_abs_entry( mask_data_long );

    if ( min_mask != 0 || max_mask != 1 )
    {
        throw std::logic_error( "Mask entries are not in [0, 1]." );
    }

    // Now we can also test additive communication of the mask. This should set all entries to one.
    terra::communication::shell::send_recv( domain, mask_data_long );

    const auto min_mask_after = terra::kernels::common::min_abs_entry( mask_data_long );
    const auto max_mask_after = terra::kernels::common::max_abs_entry( mask_data_long );

    if ( min_mask_after != 1 || max_mask_after != 1 )
    {
        throw std::logic_error( "Mask entries are not 1 (after communication)." );
    }

    // Now we check if the sum is equal to the actual number of allocated nodes ("with overlap").
    const auto sum_mask_after = terra::kernels::common::sum_of_absolutes( mask_data_long );

    const auto number_of_nodes_with_overlap = domain.subdomains().size() *
                                              domain.domain_info().subdomain_num_nodes_per_side_laterally() *
                                              domain.domain_info().subdomain_num_nodes_per_side_laterally() *
                                              domain.domain_info().subdomain_num_nodes_radially();

    std::cout << "Number of nodes (with duplicate nodes, analytical): " << number_of_nodes_with_overlap << std::endl;
    std::cout << "Number of nodes (with duplicate nodes, mask):       " << sum_mask_after << std::endl;

    if ( sum_mask_after != number_of_nodes_with_overlap )
    {
        throw std::logic_error( "Sum of mask entries does not match (after communication)." );
    }

    std::cout << std::endl;

    auto grid_diamond_id = terra::grid::shell::allocate_scalar_grid< double >( "diamond_id", domain );

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

    terra::communication::shell::send_recv( domain, ones_comm );

    auto mask_data_double = terra::grid::shell::allocate_scalar_grid< double >( "mask_data_double", domain );

    terra::kernels::common::cast( mask_data_double, mask_data_long );

    const auto xdmf_dir = "test_flag_field_unique_vertices_out";
    terra::util::prepare_empty_directory( xdmf_dir );

    terra::io::XDMFOutput xdmf( xdmf_dir, domain, coords_lat, coords_rad );
    xdmf.add( mask_data_double );
    xdmf.add( ones_comm );
    xdmf.add( grid_diamond_id );
    xdmf.write( 0 );
}

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    for ( int level = 0; level < 5; level++ )
    {
        for ( int subdomain_level = 0; subdomain_level < level; subdomain_level++ )
        {
            test( level, subdomain_level );
        }
    }

    return EXIT_SUCCESS;
}
