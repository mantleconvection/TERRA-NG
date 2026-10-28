

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
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

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorW1Vec;
using util::logroot;

struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    Grid4DDataVec< double, 3 >                         data_;
    double                                             r_min_;
    double                                             r_max_;
    bool                                               only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const Grid4DDataVec< double, 3 >&                         data,
        const double                                              r_min,
        const double                                              r_max,
        bool                                                      only_boundary )
    : grid_( grid )
    , radii_( radii )
    , mask_( mask )
    , data_( data )
    , r_min_( r_min )
    , r_max_( r_max )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const auto                    cx     = coords( 0 );
        const auto                    cy     = coords( 1 );
        const auto                    cr     = radii_( local_subdomain_id, r );

        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        const auto k      = 3;
        const auto lambda = ( Kokkos::numbers::pi * static_cast< double >( k ) ) / ( r_max_ - r_min_ );
        const auto a      = Kokkos::cos( lambda * ( cr - r_min_ ) );

        if ( !only_boundary_ || on_boundary )
        {
            data_( local_subdomain_id, x, y, r, 0 ) = a * -( cy / cr );
            data_( local_subdomain_id, x, y, r, 1 ) = a * ( cx / cr );
            data_( local_subdomain_id, x, y, r, 2 ) = 0;
        }
    }
};

struct DotNormalBoundary
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    bool                       only_boundary_;

    DotNormalBoundary(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        // const double value = coords( 0 );
        const double value = ( 3.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        for ( int d = 0; d < 3; ++d )
        {
            data_( local_subdomain_id, x, y, r, d ) = value;
        }
    }
};

void test( int level )
{
    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorW1Vec< ScalarType > a( "a", domain, mask_data );
    VectorW1Vec< ScalarType > b( "b", domain, mask_data );

    VectorW1Vec< ScalarType > no_normal( "no_normal", domain, mask_data );
    VectorW1Vec< ScalarType > no_tangent( "no_tangent", domain, mask_data );

    linalg::VectorW1Scalar< ScalarType > scalar_func( "scalar_func", domain, mask_data );

    VectorW1Vec< ScalarType > error( "error", domain, mask_data );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    // First test: transform back and forth and compare.

    linalg::assign( a, 1.0 );
    linalg::assign( b, 1.0 );

    linalg::trafo::cartesian_to_normal_tangential_in_place(
        a, coords_shell, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    linalg::lincomb( error, { 1.0, -1.0 }, { a, b } );
    auto error_inf = linalg::norm_inf( error );
    logroot << "Error cart vs normal-tangential:            " << error_inf << std::endl;

    if ( error_inf < 1.0 )
    {
        Kokkos::abort( "Error should be large." );
    }

    linalg::trafo::normal_tangential_to_cartesian_in_place(
        a, coords_shell, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );
    linalg::lincomb( error, { 1.0, -1.0 }, { a, b } );
    error_inf = linalg::norm_inf( error );
    logroot << "Error cart vs cart (after trafo backwards): " << error_inf << std::endl;

    if ( error_inf > 1e-15 )
    {
        Kokkos::abort( "Error should be close to 0." );
    }

    // Second test: checking values of normals at the boundary

    auto no_normal_data = no_normal.grid_data();
    Kokkos::parallel_for(
        "zero normal function",
        local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( const int local_subdomain_id, const int x, const int y, const int r ) {
            auto c = grid::shell::coords( local_subdomain_id, x, y, r, coords_shell, coords_radii );
            no_normal_data( local_subdomain_id, x, y, r, 0 ) = -c( 1 );
            no_normal_data( local_subdomain_id, x, y, r, 1 ) = c( 0 );
            no_normal_data( local_subdomain_id, x, y, r, 2 ) = 0.0;
        } );

    linalg::trafo::cartesian_to_normal_tangential_in_place(
        no_normal, coords_shell, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    kernels::common::extract_vector_component( scalar_func.grid_data(), no_normal.grid_data(), 0 );

    auto max_normal = kernels::common::max_abs_entry(
        scalar_func.grid_data(), boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    logroot << "Max normal: " << max_normal << std::endl;
    if ( max_normal > 1e-15 )
    {
        Kokkos::abort( "Max normal should be close to 0." );
    }

    linalg::trafo::normal_tangential_to_cartesian_in_place(
        no_normal, coords_shell, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    // Third test: checking values of tangents at the boundary

    auto no_tangent_data = no_tangent.grid_data();
    Kokkos::parallel_for(
        "zero tangent function",
        local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( const int local_subdomain_id, const int x, const int y, const int r ) {
            auto c = grid::shell::coords( local_subdomain_id, x, y, r, coords_shell, coords_radii );
            no_tangent_data( local_subdomain_id, x, y, r, 0 ) = c( 0 );
            no_tangent_data( local_subdomain_id, x, y, r, 1 ) = c( 1 );
            no_tangent_data( local_subdomain_id, x, y, r, 2 ) = c( 2 );
        } );

    linalg::trafo::cartesian_to_normal_tangential_in_place(
        no_tangent, coords_shell, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    kernels::common::extract_vector_component( scalar_func.grid_data(), no_tangent.grid_data(), 1 );

    auto max_tangent_0 = kernels::common::max_abs_entry(
        scalar_func.grid_data(), boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    logroot << "Max tangent (0): " << max_tangent_0 << std::endl;

    if ( max_tangent_0 > 1e-15 )
    {
        Kokkos::abort( "Max tangent (component 0) should be close to 0." );
    }

    kernels::common::extract_vector_component( scalar_func.grid_data(), no_tangent.grid_data(), 2 );

    auto max_tangent_1 = kernels::common::max_abs_entry(
        scalar_func.grid_data(), boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    logroot << "Max tangent (1): " << max_tangent_1 << std::endl;

    if ( max_tangent_1 > 1e-15 )
    {
        Kokkos::abort( "Max tangent (component 1) should be close to 0." );
    }

    linalg::trafo::normal_tangential_to_cartesian_in_place(
        no_tangent, coords_shell, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    ///////////////////

    io::XDMFOutput xdmf( "test_basis_trafo_normal_tangential_out", domain, coords_shell, coords_radii );
    xdmf.add( a.grid_data() );
    xdmf.add( b.grid_data() );
    xdmf.add( no_normal.grid_data() );
    xdmf.add( no_tangent.grid_data() );
    xdmf.write( 0 );
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    test( 3 );

    return 0;
}
