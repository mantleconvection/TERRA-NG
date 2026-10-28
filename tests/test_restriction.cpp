

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/util/debug_sparse_assembly.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/fe/wedge/operators/shell/prolongation_linear.hpp"
#include "terra/fe/wedge/operators/shell/restriction_linear.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorW1Scalar;

void test_constant( int level )
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

    VectorW1Scalar< ScalarType > u_coarse( "u_coarse", domain_coarse, mask_data_coarse );
    VectorW1Scalar< ScalarType > u_fine( "u_fine", domain_fine, mask_data_fine );

    VectorW1Scalar< ScalarType > solution_fine( "solution_fine", domain_fine, mask_data_fine );
    VectorW1Scalar< ScalarType > error_fine( "error_fine", domain_fine, mask_data_fine );

    const auto subdomain_shell_coords_fine =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_fine );
    const auto subdomain_radii_fine = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain_fine );

    const auto subdomain_shell_coords_coarse =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_coarse );
    const auto subdomain_radii_coarse = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain_coarse );

    using Prolongation = fe::wedge::operators::shell::ProlongationConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionConstant< ScalarType >;

    Prolongation P;
    Restriction  R( domain_coarse );

    Eigen::SparseMatrix< double > P_assembled =
        linalg::util::debug_sparse_assembly_operator_vec_q1_scalar( domain_coarse, P, u_coarse, u_fine );

    Eigen::SparseMatrix< double > R_assembled =
        linalg::util::debug_sparse_assembly_operator_vec_q1_scalar( domain_fine, R, u_fine, u_coarse );

    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > error =
        R_assembled.transpose().toDense() - P_assembled.toDense();

    const auto error_norm = error.norm();
    std::cout << "R^T - P (const., level " << level << "): error norm: " << error_norm << std::endl;

    if ( error_norm > 1e-15 )
    {
        throw std::runtime_error( "error is not zero" );
    }
}

void test_linear( int level )
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

    VectorW1Scalar< ScalarType > u_coarse( "u_coarse", domain_coarse, mask_data_coarse );
    VectorW1Scalar< ScalarType > u_fine( "u_fine", domain_fine, mask_data_fine );

    VectorW1Scalar< ScalarType > solution_fine( "solution_fine", domain_fine, mask_data_fine );
    VectorW1Scalar< ScalarType > error_fine( "error_fine", domain_fine, mask_data_fine );

    const auto subdomain_shell_coords_fine =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_fine );
    const auto subdomain_radii_fine = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain_fine );

    const auto subdomain_shell_coords_coarse =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_coarse );
    const auto subdomain_radii_coarse = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain_coarse );

    using Prolongation = fe::wedge::operators::shell::ProlongationLinear< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionLinear< ScalarType >;

    Prolongation P( subdomain_shell_coords_fine, subdomain_radii_fine );
    Restriction  R( domain_coarse, subdomain_shell_coords_fine, subdomain_radii_fine );

    Eigen::SparseMatrix< double > P_assembled =
        linalg::util::debug_sparse_assembly_operator_vec_q1_scalar( domain_coarse, P, u_coarse, u_fine );

    Eigen::SparseMatrix< double > R_assembled =
        linalg::util::debug_sparse_assembly_operator_vec_q1_scalar( domain_fine, R, u_fine, u_coarse );

    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > error =
        R_assembled.transpose().toDense() - P_assembled.toDense();

    const auto error_norm = error.norm();
    std::cout << "R^T - P (linear, level " << level << "): error norm: " << error_norm << std::endl;

    if ( error_norm > 1e-15 )
    {
        throw std::runtime_error( "error is not zero" );
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    test_constant( 1 );
    test_constant( 2 );
    test_constant( 3 );

    test_linear( 1 );
    test_linear( 2 );
    test_linear( 3 );

    return 0;
}