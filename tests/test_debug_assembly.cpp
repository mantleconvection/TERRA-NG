

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/util/debug_sparse_assembly.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;

struct SomeFunctionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    SomeFunctionInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        bool                              only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double value = Kokkos::sinh( coords( 0 ) ) * Kokkos::cosh( coords( 1 ) ) * Kokkos::tanh( coords( 2 ) );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

void test_laplace( int level )
{
    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorQ1Scalar< ScalarType > u_src( "u_src", domain, mask_data );
    VectorQ1Scalar< ScalarType > u_dst( "u_dst", domain, mask_data );

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    using Laplace = fe::wedge::operators::shell::Laplace< ScalarType >;

    Laplace A( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, false, false );

    // First let's get the sparse matrix and vector

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SomeFunctionInterpolator( subdomain_shell_coords, subdomain_radii, u_src.grid_data(), false ) );

    Eigen::SparseVector< ScalarType > u_src_assembled =
        linalg::util::debug_sparse_assembly_vector_vec_q1_scalar< ScalarType >( u_src );

    Eigen::SparseMatrix< ScalarType > A_assembled =
        linalg::util::debug_sparse_assembly_operator_vec_q1_scalar( domain, A, u_src, u_dst );

    Eigen::SparseVector< ScalarType > u_dst_assembled = A_assembled * u_src_assembled;

    // Now come the matrix-free mult.

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SomeFunctionInterpolator( subdomain_shell_coords, subdomain_radii, u_src.grid_data(), false ) );

    apply( A, u_src, u_dst );

    Eigen::SparseVector< ScalarType > u_dst_mat_free =
        linalg::util::debug_sparse_assembly_vector_vec_q1_scalar< ScalarType >( u_dst );

    std::cout << A_assembled.toDense() << std::endl;

    std::cout << u_dst_assembled.toDense() << std::endl;
    std::cout << u_dst_mat_free.toDense() << std::endl;

    Eigen::SparseVector< ScalarType > error = u_dst_assembled - u_dst_mat_free;

    std::cout << error.toDense() << std::endl;

    const auto error_norm = error.norm();
    std::cout << "error norm: " << error_norm << std::endl;

    if ( error_norm > 1e-15 )
    {
        throw std::runtime_error( "error is not zero" );
    }
}

void test_prolongation( int level )
{
    using ScalarType = double;

    if ( level < 1 )
    {
        throw std::runtime_error( "level must be >= 1" );
    }

    const auto domain_fine = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );
    const auto domain_coarse =
        DistributedDomain::create_uniform_single_subdomain_per_diamond( level - 1, level - 1, 0.5, 1.0 );

    auto mask_data_fine   = grid::setup_node_ownership_mask_data( domain_fine );
    auto mask_data_coarse = grid::setup_node_ownership_mask_data( domain_coarse );

    VectorQ1Scalar< ScalarType > u_coarse( "u_coarse", domain_coarse, mask_data_coarse );
    VectorQ1Scalar< ScalarType > u_fine( "u_fine", domain_fine, mask_data_fine );

    VectorQ1Scalar< ScalarType > solution_fine( "solution_fine", domain_fine, mask_data_fine );
    VectorQ1Scalar< ScalarType > error_fine( "error_fine", domain_fine, mask_data_fine );

    const auto subdomain_shell_coords_fine =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_fine );
    const auto subdomain_radii_fine = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain_fine );

    const auto subdomain_shell_coords_coarse =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_coarse );
    const auto subdomain_radii_coarse = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain_coarse );

    using Prolongation = fe::wedge::operators::shell::ProlongationConstant< ScalarType >;

    Prolongation P;

    // First let's get the sparse matrix and vector

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain_coarse ),
        SomeFunctionInterpolator(
            subdomain_shell_coords_coarse, subdomain_radii_coarse, u_coarse.grid_data(), false ) );

    Eigen::SparseVector< ScalarType > u_coarse_assembled =
        linalg::util::debug_sparse_assembly_vector_vec_q1_scalar< ScalarType >( u_coarse );

    Eigen::SparseMatrix< ScalarType > P_assembled =
        linalg::util::debug_sparse_assembly_operator_vec_q1_scalar( domain_coarse, P, u_coarse, u_fine );

    Eigen::SparseVector< ScalarType > u_fine_assembled = P_assembled * u_coarse_assembled;

    // Now come the matrix-free mult.

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain_coarse ),
        SomeFunctionInterpolator(
            subdomain_shell_coords_coarse, subdomain_radii_coarse, u_coarse.grid_data(), false ) );

    apply( P, u_coarse, u_fine );

    Eigen::SparseVector< ScalarType > u_fine_mat_free =
        linalg::util::debug_sparse_assembly_vector_vec_q1_scalar< ScalarType >( u_fine );

    std::cout << P_assembled.toDense() << std::endl;

    std::cout << "u_coarse" << u_coarse_assembled.toDense() << std::endl;
    std::cout << "u_fine_assembled" << u_fine_assembled.toDense() << std::endl;
    std::cout << "u_fine_mat_free" << u_fine_mat_free.toDense() << std::endl;

    Eigen::SparseVector< ScalarType > error = u_fine_assembled - u_fine_mat_free;

    std::cout << error.toDense() << std::endl;

    const auto error_norm = error.norm();
    std::cout << "error norm: " << error_norm << std::endl;

    if ( error_norm > 1e-15 )
    {
        throw std::runtime_error( "error is not zero" );
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    test_laplace( 1 );
    test_prolongation( 1 );

    return 0;
}