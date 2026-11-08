

#include <fe/wedge/operators/shell/restriction_linear.hpp>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/util/debug_sparse_assembly.hpp"
#include "terra/dense/mat.hpp"
#include "terra/eigen/eigen_wrapper.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/fe/wedge/operators/shell/prolongation_linear.hpp"
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

template < std::floating_point T >
struct SolutionInterpolator
{
    Grid3DDataVec< T, 3 > grid_;
    Grid2DDataScalar< T > radii_;
    Grid4DDataScalar< T > data_;
    bool                  only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< T, 3 >& grid,
        const Grid2DDataScalar< T >& radii,
        const Grid4DDataScalar< T >& data,
        bool                         only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< T, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        // const T                  value  = coords( 0 ) * Kokkos::sin( coords( 1 ) ) * Kokkos::sinh( coords( 2 ) );
        const T value = ( 1.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        // const T value = 0.0;
        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

template < std::floating_point T >
struct RHSInterpolator
{
    Grid3DDataVec< T, 3 > grid_;
    Grid2DDataScalar< T > radii_;
    Grid4DDataScalar< T > data_;

    RHSInterpolator(
        const Grid3DDataVec< T, 3 >& grid,
        const Grid2DDataScalar< T >& radii,
        const Grid4DDataScalar< T >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< T, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        // const T value = coords( 0 );
        const T value = ( 3.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        // const T value                   = 0.0;
        data_( local_subdomain_id, x, y, r ) = value;
    }
};

template < std::floating_point T >
struct SetOnBoundary
{
    Grid4DDataScalar< T > src_;
    Grid4DDataScalar< T > dst_;
    int                   num_shells_;

    SetOnBoundary( const Grid4DDataScalar< T >& src, const Grid4DDataScalar< T >& dst, const int num_shells )
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

template < std::floating_point T, typename Prolongation, typename Restriction >
T test( int min_level, int max_level, const std::shared_ptr< util::Table >& table, T omega, int prepost_smooth )
{
    using ScalarType       = T;
    using Laplace          = fe::wedge::operators::shell::LaplaceSimple< ScalarType >;
    using Smoother         = linalg::solvers::Jacobi< Laplace >;
    using CoarseGridSolver = linalg::solvers::PCG< Laplace >;

    std::cout << "min_level = " << min_level << ", max_level = " << max_level << std::endl;

    std::vector< DistributedDomain > domains;

    std::vector< Grid3DDataVec< ScalarType, 3 > > subdomain_shell_coords;
    std::vector< Grid2DDataScalar< ScalarType > > subdomain_radii;

    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    std::vector< VectorW1Scalar< ScalarType > > tmp_r_c;
    std::vector< VectorW1Scalar< ScalarType > > tmp_e_c;
    std::vector< VectorW1Scalar< ScalarType > > tmp;
    std::vector< Laplace >                      A_c;
    std::vector< Prolongation >                 P_additive;
    std::vector< Restriction >                  R;

    std::vector< Smoother > smoothers;

    std::vector< VectorW1Scalar< ScalarType > > coarse_grid_tmps;

    for ( int level = 0; level <= max_level; level++ )
    {
        auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );
        //auto level_subdomains = 2;
        //auto domain = DistributedDomain::create_uniform( level, level, 0.5, 1.0, level_subdomains, level_subdomains );

        domains.push_back( domain );

        subdomain_shell_coords.push_back(
            terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain ) );
        subdomain_radii.push_back( terra::grid::shell::subdomain_shell_radii< ScalarType >( domain ) );

        mask_data.push_back( grid::setup_node_ownership_mask_data( domain ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domain ) );
    }

    for ( int level = min_level; level <= max_level; level++ )
    {
        tmp.emplace_back( "tmp_level_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorW1Scalar< ScalarType > tmp_smoother(
            "tmp_smoothers_level_" + std::to_string( level ), domains[level], mask_data[level] );
        VectorW1Scalar< ScalarType > inverse_diagonal(
            "inv_diag_level_" + std::to_string( level ), domains[level], mask_data[level] );
        Laplace A_diag( domains[level], subdomain_shell_coords[level], subdomain_radii[level], true, true );
        assign( tmp_smoother, 1.0 );
        apply( A_diag, tmp_smoother, inverse_diagonal );

        linalg::invert_entries( inverse_diagonal );

        smoothers.emplace_back( inverse_diagonal, prepost_smooth, tmp_smoother, omega );

        if ( level == min_level )
        {
            constexpr int num_coarse_grid_tmps = 4;
            for ( int i = 0; i < num_coarse_grid_tmps; ++i )
            {
                coarse_grid_tmps.emplace_back(
                    "coarse_grid_tmps_" + std::to_string( i ), domains[level], mask_data[level] );
            }
        }

        if ( level < max_level )
        {
            tmp_r_c.emplace_back( "tmp_r_c_level_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_e_c.emplace_back( "tmp_e_c_level_" + std::to_string( level ), domains[level], mask_data[level] );

            A_c.emplace_back( domains[level], subdomain_shell_coords[level], subdomain_radii[level], true, false );

            if constexpr ( std::is_same_v<
                               Prolongation,
                               fe::wedge::operators::shell::ProlongationLinear< ScalarType > > )
            {
                P_additive.emplace_back(
                    subdomain_shell_coords[level + 1], subdomain_radii[level + 1], linalg::OperatorApplyMode::Add );
                //R.emplace_back( domains[level] );
                R.emplace_back( domains[level], subdomain_shell_coords[level + 1], subdomain_radii[level + 1] );
            }
            else if constexpr ( std::is_same_v<
                                    Prolongation,
                                    fe::wedge::operators::shell::ProlongationConstant< ScalarType > > )
            {
                P_additive.emplace_back( linalg::OperatorApplyMode::Add );
                R.emplace_back( domains[level] );
            }
            else
            {
                throw std::runtime_error( "Unknown prolongation type." );
            }
        }
    }

    VectorW1Scalar< ScalarType > u( "u", domains.back(), mask_data.back() );
    VectorW1Scalar< ScalarType > f( "f", domains.back(), mask_data.back() );
    VectorW1Scalar< ScalarType > solution( "solution", domains.back(), mask_data.back() );
    VectorW1Scalar< ScalarType > error( "error", domains.back(), mask_data.back() );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data.back(), grid::NodeOwnershipFlag::OWNED );
    std::cout << "num_dofs = " << num_dofs << std::endl;

    Laplace A( domains.back(), subdomain_shell_coords.back(), subdomain_radii.back(), true, false );
    Laplace A_neumann( domains.back(), subdomain_shell_coords.back(), subdomain_radii.back(), false, false );
    Laplace A_neumann_diag( domains.back(), subdomain_shell_coords.back(), subdomain_radii.back(), false, true );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;

    Mass M( domains.back(), subdomain_shell_coords.back(), subdomain_radii.back(), false );

    // Set up solution data.
    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domains.back() ),
        SolutionInterpolator( subdomain_shell_coords.back(), subdomain_radii.back(), solution.grid_data(), false ) );

    Kokkos::fence();

    // Set up rhs data.
    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domains.back() ),
        RHSInterpolator( subdomain_shell_coords.back(), subdomain_radii.back(), error.grid_data() ) );

    Kokkos::fence();

    linalg::apply( M, error, f );

    assign( error, 0.0 );
    assign( u, 0.0 );

    // Set up boundary data.
    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domains.back() ),
        SolutionInterpolator( subdomain_shell_coords.back(), subdomain_radii.back(), u.grid_data(), true ) );

    Kokkos::fence();

    fe::strong_algebraic_dirichlet_enforcement_poisson_like(
        A_neumann, A_neumann_diag, u, error, f, boundary_mask_data.back(), grid::shell::ShellBoundaryFlag::BOUNDARY );

    assign( u, 0.0 );
    assign( error, 0.0 );

    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 100, 1e-7, 1e-7 };

    CoarseGridSolver coarse_grid_solver( solver_params, table, coarse_grid_tmps );

    linalg::solvers::Multigrid< Laplace, Prolongation, Restriction, Smoother, CoarseGridSolver > multigrid_solver(
        P_additive, R, A_c, tmp_r_c, tmp_e_c, tmp, smoothers, smoothers, coarse_grid_solver, 20, 1e-6 );

    multigrid_solver.collect_statistics( table );

    assign( u, 1.0 );

    Kokkos::fence();
    Kokkos::Timer timer;
    timer.reset();
    linalg::solvers::solve( multigrid_solver, A, u, f );
    Kokkos::fence();
    const auto time_solver = timer.seconds();

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error = linalg::norm_2_scaled( error, 1.0 / static_cast< T >( num_dofs ) );

    if ( true )
    {
        io::XDMFOutput xdmf( ".", domains.back(), subdomain_shell_coords.back(), subdomain_radii.back() );
        xdmf.add( u.grid_data() );
        xdmf.add( solution.grid_data() );
        xdmf.add( error.grid_data() );
        xdmf.write( 0 );
    }

    table->add_row(
        { { "tag", "time_solver" },
          { "level", max_level },
          { "dofs", num_dofs },
          { "l2_error", l2_error },
          { "time_solver", time_solver } } );

    return l2_error;
}

template < std::floating_point T >
int run_test()
{
    T prev_l2_error = 1.0;

    const int max_level = 7;

    constexpr T   omega          = 0.666;
    constexpr int prepost_smooth = 2;

    for ( int level = 1; level <= max_level; level++ )
    {
        auto table = std::make_shared< util::Table >();

        Kokkos::Timer timer;
        timer.reset();

        T l2_error = test<
            T,
            fe::wedge::operators::shell::ProlongationLinear< T >,
            fe::wedge::operators::shell::RestrictionLinear< T > >( 0, level, table, omega, prepost_smooth );

        const auto time_total = timer.seconds();
        table->add_row( { { "tag", "time_total" }, { "level", level }, { "time_total", time_total } } );

        if ( level > 1 )
        {
            std::cout << "l2_error = " << l2_error << std::endl;
            const T order = prev_l2_error / l2_error;
            std::cout << "order = " << order << std::endl;
            if ( order < 3.4 )
            {
              //  return EXIT_FAILURE;
            }

            table->add_row( { { "level", level }, { "order", prev_l2_error / l2_error } } );
        }
        prev_l2_error = l2_error;

        table->query_rows_equals( "tag", "multigrid" ).print_pretty();
        // table->query_rows_equals( "tag", "pcg_solver" ).print_pretty();
        table->query_rows_equals( "tag", "time_solver" ).print_pretty();
        table->query_rows_equals( "tag", "time_total" ).print_pretty();
    }

    return EXIT_SUCCESS;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    return run_test< double >();
}
