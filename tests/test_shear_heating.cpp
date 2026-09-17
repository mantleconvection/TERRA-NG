#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/linearforms/shell/shear_heating_term.hpp"
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
#include "util/logging.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

using namespace terra;

using grid::Grid4DDataVec;
using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using uint_t = unsigned int;

struct VelocityInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataVec< double, 3 >                         data_u_;
    bool                                               only_boundary_;

    VelocityInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataVec< double, 3 >&                         data_u,
        const bool                                                only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        data_u_( local_subdomain_id, x, y, r, 0 ) = coords( 0 ) + coords( 1 ) + coords( 2 );
        data_u_( local_subdomain_id, x, y, r, 1 ) = 2.0 * coords( 0 ) + coords( 1 ) + coords( 2 );
        data_u_( local_subdomain_id, x, y, r, 2 ) = coords( 0 ) + 3.0 * coords( 1 ) + coords( 2 );
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

struct ShearHeatingCoefficientTest
{
    KOKKOS_INLINE_FUNCTION double operator()(
        const int                     ,
        const int                     ,
        const int                     ,
        const int                     ,
        const int                     ,
        const dense::Vec< double, 3 >  ) const
    {
        return 1.0;
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
        // The test function is identically 1, so dot( s_h, f_dst ) integrates
        // the linear form over the whole shell.
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

    VectorQ1Scalar< ScalarType > s_h( "s_h", domain, mask_data );
    VectorQ1Scalar< ScalarType > mu( "mu", domain, mask_data );
    VectorQ1Scalar< ScalarType > f_dst( "f_dst", domain, mask_data );

    VectorQ1Vec< ScalarType, 3 > velocity("velocity", domain, mask_data);

    using ShearHeatingOperator = fe::wedge::linearforms::shell::ShearHeatingTerm< ScalarType, ShearHeatingCoefficientTest >;

    ShearHeatingOperator shear_heating_operator(
        domain,
        coords_shell,
        coords_radii,
        mu,
        velocity,
        ShearHeatingCoefficientTest()
    );

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
        VelocityInterpolator( coords_shell, coords_radii, velocity.grid_data(), false ) );

    linalg::apply( shear_heating_operator, f_dst );

    const auto shear_heating_integral_analytical =
        14.5 * ( 4.0 / 5.0 ) * M_PI * ( rMax * rMax * rMax * rMax * rMax - rMin * rMin * rMin * rMin * rMin );
    const auto shear_heating_integral = linalg::dot( s_h, f_dst ) / 2.0; // Divided by 2.0 because of 2 \nu definition

    const auto shear_heating_integral_error = std::abs( shear_heating_integral - shear_heating_integral_analytical );

    util::logroot << "shear heating integral (analytical) = " << shear_heating_integral_analytical << "\n"
                  << "shear heating integral (computed)   = " << shear_heating_integral << "\n"
                  << "absolute error                      = " << shear_heating_integral_error << "\n";

    if ( shear_heating_integral_error > 0.025 )
    {
        Kokkos::abort( "Integration error too high!" );
    }

    return 0;
}
