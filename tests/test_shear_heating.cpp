#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/shear_heating_simple.hpp"
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
#include "util/timer.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;

using uint_t = unsigned int;

struct UxInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    UxInterpolator(
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

        data_( local_subdomain_id, x, y, r ) = coords( 0 ) + coords( 1 ) + coords( 2 );
    }
};

struct UyInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    UyInterpolator(
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

        data_( local_subdomain_id, x, y, r ) = 2.0 * coords( 0 ) + coords( 1 ) + coords( 2 );
    }
};

struct UzInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    UzInterpolator(
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

        data_( local_subdomain_id, x, y, r ) = coords( 0 ) + 3.0 * coords( 1 ) + coords( 2 );
    }
};

struct ViscosityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    ViscosityInterpolator(
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

        data_( local_subdomain_id, x, y, r ) =
            coords( 0 ) * coords( 0 ) + coords( 1 ) * coords( 1 ) + coords( 2 ) * coords( 2 );
    }
};

struct TrialTestFunctionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    TrialTestFunctionInterpolator(
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
        // const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        // const double x_coord = coords( 0 );
        // const double y_coord = coords( 1 );
        // const double z_coord = coords( 2 );

        // const double value = x_coord * x_coord + y_coord * y_coord + z_coord * z_coord;

        data_( local_subdomain_id, x, y, r ) = 1.0;
    }
};

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    using ScalarType = double;

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;

    const uint_t level = 5u;

    const auto rMin = 0.5;
    const auto rMax = 1.0;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, rMin, rMax );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Scalar< ScalarType > T_h( "T_h", domain, mask_data );
    VectorQ1Scalar< ScalarType > s_h( "s_h", domain, mask_data );

    VectorQ1Scalar< ScalarType > mu( "mu", domain, mask_data );

    VectorQ1Scalar< ScalarType > ux( "ux", domain, mask_data );
    VectorQ1Scalar< ScalarType > uy( "uy", domain, mask_data );
    VectorQ1Scalar< ScalarType > uz( "uz", domain, mask_data );

    VectorQ1Scalar< ScalarType > f_dst( "f_dst", domain, mask_data );

    using ShearHeatingOperator = fe::wedge::operators::shell::ShearHeatingSimple< ScalarType >;

    ShearHeatingOperator shear_heating_operator(
        domain, coords_shell, coords_radii, mu.grid_data(), ux.grid_data(), uy.grid_data(), uz.grid_data(), false, false );

    Kokkos::parallel_for(
        "u_interpolation",
        local_domain_md_range_policy_nodes( domain ),
        TrialTestFunctionInterpolator( coords_shell, coords_radii, T_h.grid_data(), false ) );

    Kokkos::parallel_for(
        "v_interpolation",
        local_domain_md_range_policy_nodes( domain ),
        TrialTestFunctionInterpolator( coords_shell, coords_radii, s_h.grid_data(), false ) );

    Kokkos::parallel_for(
        "mu_interpolation",
        local_domain_md_range_policy_nodes( domain ),
        ViscosityInterpolator( coords_shell, coords_radii, mu.grid_data(), false ) );

    Kokkos::parallel_for(
        "ux_interpolation",
        local_domain_md_range_policy_nodes( domain ),
        UxInterpolator( coords_shell, coords_radii, ux.grid_data(), false ) );

    Kokkos::parallel_for(
        "uy_interpolation",
        local_domain_md_range_policy_nodes( domain ),
        UyInterpolator( coords_shell, coords_radii, uy.grid_data(), false ) );

    Kokkos::parallel_for(
        "uz_interpolation",
        local_domain_md_range_policy_nodes( domain ),
        UzInterpolator( coords_shell, coords_radii, uz.grid_data(), false ) );

    linalg::apply( shear_heating_operator, T_h, f_dst );

    const auto shear_heating_integral_analytical =
        14.5 * ( 4.0 / 5.0 ) * M_PI * ( rMax * rMax * rMax * rMax * rMax - rMin * rMin * rMin * rMin * rMin );
    const auto shear_heating_integral = linalg::dot( s_h, f_dst );

    const auto shear_heating_integral_error = std::abs( shear_heating_integral - shear_heating_integral_analytical );

    if ( shear_heating_integral_error > 0.025 )
    {
        Kokkos::abort( "Integration error too high!" );
    }

    // std::cout << "shear_heating_integral_analytical = " << shear_heating_integral_analytical << std::endl;
    // std::cout << "shear_heating_integral            = " << shear_heating_integral << std::endl;
    // std::cout << "shear_heating_integral_error      = " << shear_heating_integral_error << std::endl;

    return 0;
}