

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/pminres.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
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

struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    SolutionInterpolator(
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
        // const double                  value  = coords( 0 ) * Kokkos::sin( coords( 1 ) ) * Kokkos::sinh( coords( 2 ) );
        const double value = ( 1.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

struct RHSInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    RHSInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data )
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
        data_( local_subdomain_id, x, y, r ) = value;
    }
};

struct SetOnBoundary
{
    Grid4DDataScalar< double > src_;
    Grid4DDataScalar< double > dst_;
    int                        num_shells_;

    SetOnBoundary( const Grid4DDataScalar< double >& src, const Grid4DDataScalar< double >& dst, const int num_shells )
    : src_( src )
    , dst_( dst )
    , num_shells_( num_shells )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_idx, const int x, const int y, const int r ) const
    {
        if ( ( r == 0 || r == num_shells_ - 1 ) )
        {
            dst_( local_subdomain_idx, x, y, r ) = src_( local_subdomain_idx, x, y, r );
        }
    }
};

double test( int level, const std::shared_ptr< util::Table >& table )
{
    Kokkos::Timer timer;

    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    constexpr auto                              num_tmps_fgmres = 64;
    std::vector< VectorW1Scalar< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < num_tmps_fgmres; ++i )
    {
        tmp_fgmres.emplace_back( "tmp_" + std::to_string( i ), domain, mask_data );
    }

    constexpr auto                              num_tmps_cg = 5;
    std::vector< VectorW1Scalar< ScalarType > > tmp_cg;
    for ( int i = 0; i < num_tmps_cg; ++i )
    {
        tmp_cg.emplace_back( "tmp_" + std::to_string( i ), domain, mask_data );
    }

    VectorW1Scalar< ScalarType > u( "u", domain, mask_data );
    VectorW1Scalar< ScalarType > g( "g", domain, mask_data );

    VectorW1Scalar< ScalarType > solution( "solution", domain, mask_data );
    VectorW1Scalar< ScalarType > error( "error", domain, mask_data );
    VectorW1Scalar< ScalarType > b( "b", domain, mask_data );
    VectorW1Scalar< ScalarType > r( "r", domain, mask_data );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    using Laplace = fe::wedge::operators::shell::Laplace< ScalarType >;

    Laplace A( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, true, false );
    Laplace A_neumann( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, false, false );
    Laplace A_neumann_diag( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, false, true );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;

    Mass M( domain, subdomain_shell_coords, subdomain_radii, false );

    // Set up solution data.
    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, solution.grid_data(), false ) );

    // Set up boundary data.
    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, g.grid_data(), true ) );

    // Set up rhs data.
    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domain ),
        RHSInterpolator( subdomain_shell_coords, subdomain_radii, tmp_fgmres[0].grid_data() ) );

    linalg::apply( M, tmp_fgmres[0], b );

    fe::strong_algebraic_dirichlet_enforcement_poisson_like(
        A_neumann, A_neumann_diag, g, tmp_fgmres[1], b, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    linalg::solvers::IterativeSolverParameters solver_params{ 100, 1e-12, 1e-12 };

    linalg::solvers::PCG< Laplace >                                     cg( { 8, 1e-12, 1e-12 }, table, tmp_cg );
    linalg::solvers::FGMRES< Laplace, linalg::solvers::PCG< Laplace > > fgmres( tmp_fgmres, {}, nullptr, cg );

    Kokkos::fence();
    timer.reset();
    linalg::solvers::solve( fgmres, A, u, b );
    Kokkos::fence();
    const auto time_solver = timer.seconds();

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error = std::sqrt( dot( error, error ) / num_dofs );

    table->add_row(
        { { "level", level }, { "dofs", num_dofs }, { "l2_error", l2_error }, { "time_solver", time_solver } } );

    return l2_error;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    double prev_l2_error = 1.0;

    for ( int level = 0; level < 5; ++level )
    {
        Kokkos::Timer timer;
        timer.reset();
        double     l2_error   = test( level, table );
        const auto time_total = timer.seconds();
        table->add_row( { { "level", level }, { "time_total", time_total } } );

        if ( level > 1 )
        {
            const double order = prev_l2_error / l2_error;
            std::cout << "order = " << order << ", error = " << l2_error << std::endl;
            if ( order < 3.4 )
            {
                return EXIT_FAILURE;
            }

            table->add_row( { { "level", level }, { "order", prev_l2_error / l2_error } } );
        }
        prev_l2_error = l2_error;
    }

    table->query_rows_not_none( "order" ).select_columns( { "level", "order" } ).print_pretty();
    table->query_rows_not_none( "dofs" ).select_columns( { "level", "dofs", "l2_error" } ).print_pretty();
}