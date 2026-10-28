

#include <fe/wedge/operators/shell/restriction_linear.hpp>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/div_k_grad.hpp"
#include "fe/wedge/operators/shell/div_k_grad_simple.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/prolongation_linear.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/restriction_linear.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/gca/gca_elements_collector.hpp"
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
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
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
using linalg::solvers::TwoGridGCA;
using terra::linalg::DiagonallyScaledOperator;
using terra::linalg::solvers::power_iteration;

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
        const T                  value  = ( 1.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
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
    const double               alpha_;
    const double               r_min_;
    const double               r_max_;
    const double               k_max_;

    RHSInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        const double                      r_min,
        const double                      r_max,
        const double                      alpha,
        const double                      k_max )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , alpha_( alpha )
    , r_min_( r_min )
    , r_max_( r_max )
    , k_max_( k_max )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double                  x0     = Kokkos::sinh( coords( 1 ) );
        const double                  x1     = 2 * coords( 0 );
        const double                  x2     = Kokkos::sin( x1 );
        const double                  x3     = 0.5 * r_max_;
        const double                  x4     = 0.5 * r_min_;
        const double                  x5     = Kokkos::sqrt(
            Kokkos::pow( coords( 0 ), 2 ) + Kokkos::pow( coords( 1 ), 2 ) + Kokkos::pow( coords( 2 ), 2 ) );
        const double x6                      = alpha_ / ( x3 - x4 );
        const double x7                      = Kokkos::tanh( x6 * ( -x3 - x4 + x5 ) );
        const double x8                      = 0.5 * k_max_;
        const double x9                      = x6 * ( 1 - Kokkos::pow( x7, 2 ) ) / x5;
        data_( local_subdomain_id, x, y, r ) = -0.25 * k_max_ * x2 * x9 * coords( 1 ) * Kokkos::cosh( coords( 1 ) ) -
                                               coords( 0 ) * x0 * x8 * x9 * Kokkos::cos( x1 ) +
                                               1.5 * x0 * x2 * ( k_max_ + x8 * ( x7 + 1 ) );
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

struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    const double               alpha_;
    const double               r_min_;
    const double               r_max_;
    const double               k_max_;

    KInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        const double                      r_min,
        const double                      r_max,
        const double                      alpha,
        const double                      k_max )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , alpha_( alpha )
    , r_min_( r_min )
    , r_max_( r_max )
    , k_max_( k_max )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double rad                     = coords.norm();
        const double x0                      = 0.5 * r_max_;
        const double x1                      = 0.5 * r_min_;
        data_( local_subdomain_id, x, y, r ) = 
        0.5 * k_max_*( Kokkos::tanh( alpha_ * ( -x0 - x1 + rad ) / ( x0 - x1 ) ) + 1 ) + k_max_;
    }
};

template < std::floating_point T, typename Prolongation, typename Restriction >
T test(
    int                                   min_level,
    int                                   max_level,
    const std::shared_ptr< util::Table >& table,
    int                                   prepost_smooth,
    double                                alpha,
    double                                k_max,
    int                                   gca )
{
    using ScalarType   = T;
    const double r_min = 0.5;
    const double r_max = 1.0;

    using DivKGrad         = fe::wedge::operators::shell::DivKGrad< ScalarType >;
    using Smoother         = linalg::solvers::Jacobi< DivKGrad >;
    using CoarseGridSolver = linalg::solvers::PCG< DivKGrad >;

    std::cout << "min_level = " << min_level << ", max_level = " << max_level << std::endl;

    std::vector< DistributedDomain >              domains;
    std::vector< Grid3DDataVec< ScalarType, 3 > > subdomain_shell_coords;
    std::vector< Grid2DDataScalar< ScalarType > > subdomain_radii;

    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    std::vector< VectorW1Scalar< ScalarType > > tmp_r_c;
    std::vector< VectorW1Scalar< ScalarType > > tmp_e_c;
    std::vector< VectorW1Scalar< ScalarType > > tmp;
    std::vector< DivKGrad >                     A_c;
    std::vector< Prolongation >                 P_additive;
    std::vector< Restriction >                  R;

    std::vector< Smoother > smoothers;

    std::vector< VectorW1Scalar< ScalarType > > coarse_grid_tmps;

    std::cout << "Domain setup " << std::endl;
    for ( int level = 0; level <= max_level; level++ )
    {
        auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );
        domains.push_back( domain );

        subdomain_shell_coords.push_back(
            terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain ) );
        subdomain_radii.push_back( terra::grid::shell::subdomain_shell_radii< ScalarType >( domain ) );

        mask_data.push_back( grid::setup_node_ownership_mask_data( domain ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domain ) );
    }

    VectorW1Scalar< ScalarType > k( "k", domains.back(), mask_data.back() );
    // Set up coefficient data.
    std::cout << "K interpolation " << std::endl;

    Kokkos::parallel_for(
        "coefficient interpolation",
        local_domain_md_range_policy_nodes( domains.back() ),
        KInterpolator(
            subdomain_shell_coords.back(), subdomain_radii.back(), k.grid_data(), r_min, r_max, alpha, k_max ) );

    Kokkos::fence();

    // agca stuff
    VectorW1Scalar< ScalarType > GCAElements( "GCAElements", domains[min_level], mask_data[min_level] );

    
    if ( gca == 2 )
    {
        linalg::assign( GCAElements, 0 );
        std::cout << "adaptive gca: determining gca elements on level " << max_level << std::endl;
        terra::linalg::solvers::GCAElementsCollector< ScalarType >(
            domains.back(), k.grid_data(), max_level - min_level, GCAElements.grid_data() );

    }

    DivKGrad A(
        domains.back(),
        subdomain_shell_coords.back(),
        subdomain_radii.back(),
        boundary_mask_data.back(),
        k.grid_data(),
        true,
        false );
      
    // A.set_single_quadpoint( true );
    DivKGrad A_neumann(
        domains.back(),
        subdomain_shell_coords.back(),
        subdomain_radii.back(),
        boundary_mask_data.back(),
        k.grid_data(),
        false,
        false );
    //A_neumann.set_single_quadpoint( true );
    DivKGrad A_neumann_diag(
        domains.back(),
        subdomain_shell_coords.back(),
        subdomain_radii.back(),
        boundary_mask_data.back(),
        k.grid_data(),
        false,
        true );
   
    //A_neumann_diag.set_single_quadpoint( true );

    // setup operators (prolongation, restriction, matrix storage)
    for ( int level = min_level; level <= max_level; level++ )
    {
        std::cout << "MG hierarchy level " << level << std::endl;
        tmp.emplace_back( "tmp_level_" + std::to_string( level ), domains[level], mask_data[level] );

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
            VectorW1Scalar< ScalarType > k_c( "k_c", domains[level], mask_data[level] );
            Kokkos::parallel_for(
                "coefficient interpolation",
                local_domain_md_range_policy_nodes( domains[level] ),
                KInterpolator(
                    subdomain_shell_coords[level],
                    subdomain_radii[level],
                    k_c.grid_data(),
                    r_min,
                    r_max,
                    alpha,
                    k_max ) );

            Kokkos::fence();
            A_c.emplace_back(
                domains[level],
                subdomain_shell_coords[level],
                subdomain_radii[level],
                boundary_mask_data[level],
                k_c.grid_data(),
                true,
                false );

            if ( gca > 0 )
            {
                if ( gca == 2 )
                {
                    std::cout << "Adaptive gca: store gca matrices only on selected elements " << std::endl;
                    A_c.back().set_stored_matrix_mode(
                        linalg::OperatorStoredMatrixMode::Selective, level - min_level, GCAElements.grid_data() );
                }
                else if ( gca == 1 )
                {
                    std::cout << "Gca on all elements " << std::endl;
                    linalg::assign( GCAElements, 1 );
                    A_c.back().set_stored_matrix_mode(
                        linalg::OperatorStoredMatrixMode::Full, level - min_level, GCAElements.grid_data() );
                }
            }
            if constexpr ( std::is_same_v<
                               Prolongation,
                               fe::wedge::operators::shell::ProlongationLinear< ScalarType > > )
            {
                P_additive.emplace_back(
                    subdomain_shell_coords[level + 1], subdomain_radii[level + 1], linalg::OperatorApplyMode::Add );
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

    // setup gca coarse ops
    if ( gca )
    {
        std::cout << "Assembling GCA operators" << std::endl;
        //linalg::assign( GCAElements, 1 );

        for ( int level = max_level - 1; level >= min_level; level-- )
        {
            if ( level == max_level - 1 )
            {
                TwoGridGCA< ScalarType, DivKGrad >(
                    A_neumann, A_c[level - min_level], level - min_level, GCAElements.grid_data() );
            }
            else
            {
                TwoGridGCA< ScalarType, DivKGrad >(
                    A_c[level + 1 - min_level], A_c[level - min_level], level - min_level, GCAElements.grid_data() );
            }
        }
    }

    // setup smoothers
    for ( int level = min_level; level <= max_level; level++ )
    {
        VectorW1Scalar< ScalarType > tmp_smoother(
            "tmp_smoothers_level_" + std::to_string( level ), domains[level], mask_data[level] );
        VectorW1Scalar< ScalarType > inverse_diagonal(
            "inv_diag_level_" + std::to_string( level ), domains[level], mask_data[level] );
        VectorW1Scalar< ScalarType > tmp_pi_0(
            "tmp_pi_0_level_" + std::to_string( level ), domains[level], mask_data[level] );
        VectorW1Scalar< ScalarType > tmp_pi_1(
            "tmp_pi_1_level_" + std::to_string( level ), domains[level], mask_data[level] );
        assign( tmp_smoother, 1.0 );
        if ( level < max_level )
        {
            A_c[level - min_level].set_diagonal( true );
            apply( A_c[level - min_level], tmp_smoother, inverse_diagonal );
            A_c[level - min_level].set_diagonal( false );
        }
        else
        {
            A.set_diagonal( true );
            apply( A, tmp_smoother, inverse_diagonal );
            A.set_diagonal( false );
        }

        linalg::invert_entries( inverse_diagonal );

        // determine estimate for maximum eigenvalue
        T max_ev = 0.0;
        if ( level < max_level )
        {
            DiagonallyScaledOperator< DivKGrad > inv_diag_A( A_c[level - min_level], inverse_diagonal );
            max_ev = power_iteration< DiagonallyScaledOperator< DivKGrad > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
        }
        else
        {
            DiagonallyScaledOperator< DivKGrad > inv_diag_A( A, inverse_diagonal );
            max_ev = power_iteration< DiagonallyScaledOperator< DivKGrad > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
        }

        // compute optimal jacobi weight: omega_opt = 2/(lambda_min + lambda_max)
        T omega_opt = 2.0 / ( 1.5 * max_ev );
        std::cout << "Maximum ev on level " << level << ": " << max_ev << ", optimal omega: " << omega_opt << std::endl;

        smoothers.emplace_back( inverse_diagonal, prepost_smooth, tmp_smoother, 2.0 / 3.0 );
    }

    VectorW1Scalar< ScalarType > u( "u", domains.back(), mask_data.back() );
    VectorW1Scalar< ScalarType > f( "f", domains.back(), mask_data.back() );
    VectorW1Scalar< ScalarType > solution( "solution", domains.back(), mask_data.back() );
    VectorW1Scalar< ScalarType > error( "error", domains.back(), mask_data.back() );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data.back(), grid::NodeOwnershipFlag::OWNED );
    std::cout << "num_dofs = " << num_dofs << std::endl;

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
        RHSInterpolator(
            subdomain_shell_coords.back(), subdomain_radii.back(), error.grid_data(), r_min, r_max, alpha, k_max ) );

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

    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 100, 1e-7, 1e-7 };

    CoarseGridSolver coarse_grid_solver( solver_params, table, coarse_grid_tmps );

    linalg::solvers::Multigrid< DivKGrad, Prolongation, Restriction, Smoother, CoarseGridSolver > multigrid_solver(
        P_additive, R, A_c, tmp_r_c, tmp_e_c, tmp, smoothers, smoothers, coarse_grid_solver, 100, 1e-6 );

    multigrid_solver.collect_statistics( table );

    Kokkos::fence();
    Kokkos::Timer timer;
    timer.reset();
    linalg::solvers::solve( multigrid_solver, A, u, f );
    Kokkos::fence();
    const auto time_solver = timer.seconds();

    assign( error, 0.0 );
    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error = linalg::norm_2_scaled( error, 1.0 / static_cast< T >( num_dofs ) );

    if ( true )
    {
        io::XDMFOutput xdmf( "gca_coeffs", domains.back(), subdomain_shell_coords.back(), subdomain_radii.back() );
        xdmf.add( u.grid_data() );
        xdmf.add( solution.grid_data() );
        xdmf.add( error.grid_data() );
        xdmf.add( k.grid_data() );
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

    const int max_level = 6;

    constexpr int         prepost_smooth = 3;
    std::vector< double > alphas         = { 1 };
    std::vector< int >    pows           = { 1 };
    std::vector< int >    gcas           = { 1, 0 }; //, 1 };

    auto table_dca  = std::make_shared< util::Table >();
    auto table_gca  = std::make_shared< util::Table >();
    auto table_agca = std::make_shared< util::Table >();
    for ( int minlevel = 0; minlevel < 1; minlevel++ )
    {
        std::cout << "Minlevel=" << minlevel << std::endl;
        for ( int gca : gcas )
        {
            for ( int alpha : alphas )
            {
                terra::util::Table::Row cycles;
                cycles["alpha"] = alpha;
                for ( int pow : pows )
                {
                    int k_max = Kokkos::pow( 10, pow );
                    auto table = std::make_shared< util::Table >();

                    for ( int level = minlevel + 1; level <= max_level; level++ )
                    {
                        
                        Kokkos::Timer timer;
                        timer.reset();

                        std::cout << " <<<<<<<< alpha = " << alpha << ", k_max = " << k_max << ", gca = " << gca
                                  << " >>>>>>>>>" << std::endl;
                        T l2_error = test<
                            T,
                            fe::wedge::operators::shell::ProlongationConstant< T >,
                            fe::wedge::operators::shell::RestrictionConstant< T > >(
                            minlevel, level, table, prepost_smooth, alpha, k_max, gca );

                        const auto time_total = timer.seconds();
                        table->add_row( { { "tag", "time_total" }, { "level", level }, { "time_total", time_total } } );

                        double order = prev_l2_error / l2_error;
                        
                        if (level > 1 && order <= 3.6)
                            Kokkos::abort("Convergence order not reached.");

                        cycles[std::string( "k_max=" ) + std::to_string( k_max )] =
                            table->query_rows_equals( "tag", "multigrid" ).rows().size();
                        cycles["level"] = level;
                        cycles["order"] = order;
                        cycles["l2_error"] = l2_error;    
                        prev_l2_error = l2_error;

                        if ( gca == 1 )
                        {
                            table_gca->add_row( cycles );
                        }
                        else if ( gca == 2 )
                        {
                            table_agca->add_row( cycles );
                        }
                        else
                        {
                            table_dca->add_row( cycles );
                        }
                    }
                }
            }
        }
        std::cout << "DCA: " << std::endl;
        table_dca->print_pretty();
        std::cout << "GCA: " << std::endl;
        table_gca->print_pretty();
        std::cout << "AGCA: " << std::endl;
        table_agca->print_pretty();
    }
    return EXIT_SUCCESS;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    return run_test< double >();
}
