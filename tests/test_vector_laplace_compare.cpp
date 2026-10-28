

#include <fe/wedge/operators/shell/laplace_simple.hpp>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/vector_laplace.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;
using grid::shell::BoundaryConditions;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;

struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    bool                       only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data,
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
        // const double                  value  = coords( 0 ) * Kokkos::sin( coords( 1 ) ) * Kokkos::sinh( coords( 2 ) );
        const auto   rr = radii_( local_subdomain_id, r );
        const double value_x =
            ( rr - 1.0 ) * ( rr - 0.5 ) * ( 1.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        const double value_y =
            ( rr - 1.0 ) * ( rr - 0.5 ) * ( 1.0 / 2.0 ) * Kokkos::sin( 3 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        const double value_z =
            ( rr - 1.0 ) * ( rr - 0.5 ) * ( 1.0 / 2.0 ) * Kokkos::sin( 4 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );

        data_( local_subdomain_id, x, y, r, 0 ) = value_x;
        data_( local_subdomain_id, x, y, r, 1 ) = value_y;
        data_( local_subdomain_id, x, y, r, 2 ) = value_z;
    }
};

void test( int level, bool treat_boundary, bool diagonal )
{
    Kokkos::Timer timer;

    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorW1Vec< ScalarType > src( "src", domain, mask_data );
    VectorW1Vec< ScalarType > dst_a( "dst_a", domain, mask_data );
    VectorW1Vec< ScalarType > dst_b( "dst_b", domain, mask_data );
    VectorW1Vec< ScalarType > dst_c( "dst_c", domain, mask_data );
    VectorW1Vec< ScalarType > dst_d( "dst_d", domain, mask_data );
    VectorW1Vec< ScalarType > error_b( "error_b", domain, mask_data );
    VectorW1Vec< ScalarType > error_c( "error_c", domain, mask_data );
    VectorW1Vec< ScalarType > error_d( "error_d", domain, mask_data );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );
    BoundaryConditions bcs  = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };
    using LaplaceA = fe::wedge::operators::shell::VectorLaplaceSimple< ScalarType >;
    using LaplaceB = fe::wedge::operators::shell::VectorLaplace< ScalarType >;

    LaplaceA A( domain, coords_shell, coords_radii, treat_boundary, diagonal );
    LaplaceB B( domain, coords_shell, coords_radii, boundary_mask_data, bcs, diagonal );

    // Set up solution data.
    Kokkos::parallel_for(
        "src interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( coords_shell, coords_radii, src.grid_data(), false ) );

    Kokkos::fence();

    linalg::apply( A, src, dst_a );
    linalg::apply( B, src, dst_b );

    linalg::lincomb( error_b, { 1.0, -1.0 }, { dst_a, dst_b } );

    const auto l2_error_b  = std::sqrt( dot( error_b, error_b ) / num_dofs );
    const auto inf_error_b = linalg::norm_inf( error_b );

    std::cout << "L2 error:           " << l2_error_b << std::endl;
    std::cout << "inf error:          " << inf_error_b << std::endl;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    for ( auto treat_boundary : { true, false } )
    {
        for ( auto diagonal : { true, false } )
        {
            std::cout << "treat_boundary = " << treat_boundary << ", diagonal = " << diagonal << std::endl;

            for ( int level = 0; level < 6; ++level )
            {
                std::cout << "level = " << level << std::endl;

                test( level, treat_boundary, diagonal );
            }
        }
    }

    return 0;
}