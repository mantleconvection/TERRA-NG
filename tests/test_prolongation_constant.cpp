

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/fe/wedge/operators/shell/prolongation_constant.hpp"
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
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorW1Scalar;

struct ConstantFunctionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    ConstantFunctionInterpolator(
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
        const double value = 1.0;

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

struct LinearFunctionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    LinearFunctionInterpolator(
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

        const double value = coords( 0 ) + 2.3 * coords( 1 ) - 0.8 * coords( 2 ) + 1.0;
        // const double value = coords( 2 );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

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

        const double value =
            Kokkos::sin( 10 * coords( 0 ) ) * Kokkos::cos( 10 * coords( 1 ) ); //* Kokkos::tanh( coords( 2 ) );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

template < typename FunctionInterpolator >
double test( int level, const std::shared_ptr< util::Table >& table )
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

    Prolongation P;

    // Set up solution data.
    Kokkos::parallel_for(
        "coarse interpolation",
        local_domain_md_range_policy_nodes( domain_coarse ),
        FunctionInterpolator( subdomain_shell_coords_coarse, subdomain_radii_coarse, u_coarse.grid_data(), false ) );

    Kokkos::fence();

    Kokkos::parallel_for(
        "fine solution interpolation",
        local_domain_md_range_policy_nodes( domain_fine ),
        FunctionInterpolator( subdomain_shell_coords_fine, subdomain_radii_fine, solution_fine.grid_data(), false ) );

    Kokkos::fence();

    linalg::apply( P, u_coarse, u_fine );

    linalg::lincomb( error_fine, { 1.0, -1.0 }, { u_fine, solution_fine } );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data_fine, grid::NodeOwnershipFlag::OWNED );

    const auto error_norm = linalg::norm_2_scaled( error_fine, 1.0 / num_dofs );

    if ( true )
    {
        io::XDMFOutput xdmf_output_fine( ".", domain_fine, subdomain_shell_coords_fine, subdomain_radii_fine );
        xdmf_output_fine.add( u_fine.grid_data() );
        xdmf_output_fine.add( solution_fine.grid_data() );
        xdmf_output_fine.add( error_fine.grid_data() );

        xdmf_output_fine.write( 0 );

        io::XDMFOutput xdmf_output_coarse( ".", domain_coarse, subdomain_shell_coords_coarse, subdomain_radii_coarse );
        xdmf_output_coarse.add( u_coarse.grid_data() );

        xdmf_output_coarse.write( 0 );
    }

    return error_norm;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    std::cout << "Testing prolongation: constant function" << std::endl;
    {
        for ( int level = 1; level <= 5; ++level )
        {
            double error = test< ConstantFunctionInterpolator >( level, table );

            std::cout << "error (fine level " << level << ") = " << error << std::endl;

            if ( error > 1e-12 )
            {
                throw std::runtime_error( "constants must be prolongated exactly" );
            }
        }
    }

    std::cout << std::endl;

    std::cout << "Testing prolongation: linear function" << std::endl;
    {
        double prev_error = 1.0;
        for ( int level = 2; level <= 5; ++level )
        {
            double error = test< SomeFunctionInterpolator >( level, table );
            if ( level > 3 )
            {
                const auto order = prev_error / error;
                std::cout << "order (fine level " << level << ") = " << order << std::endl;
                if ( order < 3.2 )
                {
                    throw std::runtime_error( "order too low" );
                }
            }
            prev_error = error;
        }
    }

    std::cout << std::endl;

    std::cout << "Testing prolongation: arbitrary function" << std::endl;
    {
        double prev_error = 1.0;
        for ( int level = 2; level <= 5; ++level )
        {
            double error = test< SomeFunctionInterpolator >( level, table );
            if ( level > 3 )
            {
                const auto order = prev_error / error;
                std::cout << "order (fine level " << level << ") = " << order << std::endl;
                if ( order < 3.2 )
                {
                    throw std::runtime_error( "order too low" );
                }
            }
            prev_error = error;
        }
    }

    return 0;
}
