

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/pcg.hpp"
#include "grid/shell/bit_masks.hpp"
#include "linalg/solvers/pminres.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/vector_w1_iso_w2_w1.hpp"
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
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorW1IsoW2W1;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;
using grid::shell::BoundaryConditions;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;

#define SOLUTION_TYPE 1

struct SolutionVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_u_;
    bool                       only_boundary_;

    SolutionVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data_u,
        const bool                        only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        dense::Vec< double, 3 > u;

        if ( SOLUTION_TYPE == 0 )
        {
            u( 0 ) = Kokkos::sin( cy );
            u( 1 ) = Kokkos::sin( cz );
            u( 2 ) = Kokkos::sin( cx );
        }

        else if ( SOLUTION_TYPE == 1 )
        {
            u( 0 ) = -4 * Kokkos::cos( 4 * cz );
            u( 1 ) = 8 * Kokkos::cos( 8 * cx );
            u( 2 ) = -2 * Kokkos::cos( 2 * cy );
        }

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            for ( int d = 0; d < 3; d++ )
            {
                data_u_( local_subdomain_id, x, y, r, d ) = u( d );
            }
        }
    }
};

struct SolutionPressureInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_p_;
    bool                       only_boundary_;

    SolutionPressureInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data_p,
        const bool                        only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_p_( data_p )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        double p = 0.0;

        if ( SOLUTION_TYPE == 0 )
        {
            p = 0;
        }

        else if ( SOLUTION_TYPE == 1 )
        {
            p = Kokkos::sin( 4 * cx ) * Kokkos::sin( 8 * cy ) * Kokkos::sin( 2 * cz );
        }

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_p_( local_subdomain_id, x, y, r ) = p;
        }
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_u_;

    RHSVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data_u )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )

    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        dense::Vec< double, 3 > u;

        if ( SOLUTION_TYPE == 0 )
        {
            u( 0 ) = Kokkos::sin( cy );
            u( 1 ) = Kokkos::sin( cz );
            u( 2 ) = Kokkos::sin( cx );
        }

        else if ( SOLUTION_TYPE == 1 )
        {
            u( 0 ) =
                4 * Kokkos::sin( 8 * cy ) * Kokkos::sin( 2 * cz ) * Kokkos::cos( 4 * cx ) - 64 * Kokkos::cos( 4 * cz );
            u( 1 ) =
                8 * Kokkos::sin( 4 * cx ) * Kokkos::sin( 2 * cz ) * Kokkos::cos( 8 * cy ) + 512 * Kokkos::cos( 8 * cx );
            u( 2 ) =
                2 * Kokkos::sin( 4 * cx ) * Kokkos::sin( 8 * cy ) * Kokkos::cos( 2 * cz ) - 8 * Kokkos::cos( 2 * cy );
        }

        for ( int d = 0; d < 3; d++ )
        {
            data_u_( local_subdomain_id, x, y, r, d ) = u( d );
        }
    }
};

struct SetOnBoundary
{
    Grid4DDataVec< double, 3 > src_;
    Grid4DDataVec< double, 3 > dst_;
    int                        num_shells_;

    SetOnBoundary( const Grid4DDataVec< double, 3 >& src, const Grid4DDataVec< double, 3 >& dst, const int num_shells )
    : src_( src )
    , dst_( dst )
    , num_shells_( num_shells )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_idx, const int x, const int y, const int r ) const
    {
        if ( ( r == 0 || r == num_shells_ - 1 ) )
        {
            for ( int d = 0; d < 3; ++d )
            {
                dst_( local_subdomain_idx, x, y, r, d ) = src_( local_subdomain_idx, x, y, r, d );
            }
        }
    }
};

std::pair< double, double > test( int level, const std::shared_ptr< util::Table >& table )
{
    using ScalarType = double;

    const auto domain_fine = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );
    const auto domain_coarse =
        DistributedDomain::create_uniform_single_subdomain_per_diamond( level - 1, level - 1, 0.5, 1.0 );

    const auto subdomain_fine_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_fine );
    const auto subdomain_fine_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain_fine );

    const auto subdomain_coarse_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_coarse );
    const auto subdomain_coarse_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain_coarse );

    auto mask_data_fine   = grid::setup_node_ownership_mask_data( domain_fine );
    auto mask_data_coarse = grid::setup_node_ownership_mask_data( domain_coarse );

    auto boundary_mask_data_fine = grid::shell::setup_boundary_mask_data( domain_fine );

    // K w = b

    VectorW1IsoW2W1< ScalarType > w( "w", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > g( "g", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > Kdiagg( "Kdiagg", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > tmp( "tmp", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > solution( "solution", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > error( "error", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > b( "b", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > r( "r", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );

    VectorW1IsoW2W1< ScalarType > tmp_0( "tmp_0", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > tmp_1( "tmp_1", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > tmp_2( "tmp_2", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > tmp_3( "tmp_3", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > tmp_4( "tmp_4", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > tmp_5( "tmp_5", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );
    VectorW1IsoW2W1< ScalarType > tmp_6( "tmp_6", domain_fine, domain_coarse, mask_data_fine, mask_data_coarse );

    const auto num_dofs_velocity =
        3 * kernels::common::count_masked< long >( mask_data_fine, grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( mask_data_coarse, grid::NodeOwnershipFlag::OWNED );

    using Stokes           = fe::wedge::operators::shell::Stokes< ScalarType >;
    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };
    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };

    Stokes K(
        domain_fine,
        domain_coarse,
        subdomain_fine_shell_coords,
        subdomain_fine_radii,
        boundary_mask_data_fine,
        bcs,
        false );
    Stokes K_neumann(
        domain_fine,
        domain_coarse,
        subdomain_fine_shell_coords,
        subdomain_fine_radii,
        boundary_mask_data_fine,
        bcs_neumann,
        false );
    Stokes K_neumann_diag(
        domain_fine,
        domain_coarse,
        subdomain_fine_shell_coords,
        subdomain_fine_radii,
        boundary_mask_data_fine,
        bcs_neumann,
        true );

    using Mass = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;

    Mass M( domain_fine, subdomain_fine_shell_coords, subdomain_fine_radii, false );

    // Set up solution data.
    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain_fine ),
        SolutionVelocityInterpolator(
            subdomain_fine_shell_coords, subdomain_fine_radii, solution.block_1().grid_data(), false ) );

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain_coarse ),
        SolutionPressureInterpolator(
            subdomain_coarse_shell_coords, subdomain_coarse_radii, solution.block_2().grid_data(), false ) );

    // Set up boundary data.
    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domain_fine ),
        SolutionVelocityInterpolator(
            subdomain_fine_shell_coords, subdomain_fine_radii, g.block_1().grid_data(), true ) );

    // Set up rhs data.
    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domain_fine ),
        RHSVelocityInterpolator( subdomain_fine_shell_coords, subdomain_fine_radii, tmp.block_1().grid_data() ) );

    linalg::apply( M, tmp.block_1(), b.block_1() );

    linalg::apply( K_neumann_diag, g, Kdiagg );
    linalg::apply( K_neumann, g, tmp );

    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );

    Kokkos::parallel_for(
        "set on boundary",
        grid::shell::local_domain_md_range_policy_nodes( domain_fine ),
        SetOnBoundary(
            Kdiagg.block_1().grid_data(),
            b.block_1().grid_data(),
            domain_fine.domain_info().subdomain_num_nodes_radially() ) );

    linalg::solvers::IterativeSolverParameters solver_params{ 3000, 1e-8, 1e-12 };

    linalg::solvers::PMINRES< Stokes > pminres(
        solver_params, table, { tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6 } );
    pminres.set_tag( "pminres_solver_level_" + std::to_string( level ) );

    solve( pminres, K, w, b );

    const double avg_pressure_solution =
        kernels::common::masked_sum(
            solution.block_2().grid_data(), solution.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;
    const double avg_pressure_approximation =
        kernels::common::masked_sum(
            w.block_2().grid_data(), w.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;

    linalg::lincomb( solution.block_2(), { 1.0 }, { solution.block_2() }, -avg_pressure_solution );
    linalg::lincomb( w.block_2(), { 1.0 }, { w.block_2() }, -avg_pressure_approximation );

    linalg::apply( K, w, tmp_6 );
    linalg::lincomb( r, { 1.0, -1.0 }, { b, tmp_6 } );
    const auto inf_residual_vel = linalg::norm_inf( r.block_1() );
    const auto inf_residual_pre = linalg::norm_inf( r.block_2() );

    linalg::lincomb( error, { 1.0, -1.0 }, { w, solution } );
    const auto l2_error_velocity =
        std::sqrt( dot( error.block_1(), error.block_1() ) / static_cast< double >( num_dofs_velocity ) );
    const auto l2_error_pressure =
        std::sqrt( dot( error.block_2(), error.block_2() ) / static_cast< double >( num_dofs_pressure ) );

    table->add_row(
        { { "level", level },
          { "dofs_vel", num_dofs_velocity },
          { "l2_error_vel", l2_error_velocity },
          { "dofs_pre", num_dofs_pressure },
          { "l2_error_pre", l2_error_pressure },
          { "inf_res_vel", inf_residual_vel },
          { "inf_res_pre", inf_residual_pre } } );

    return { l2_error_velocity, l2_error_pressure };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    double prev_l2_error_vel = 1.0;
    double prev_l2_error_pre = 1.0;

    for ( int level = 1; level < 6; ++level )
    {
        std::cout << "level = " << level << std::endl;
        Kokkos::Timer timer;
        timer.reset();
        const auto [l2_error_vel, l2_error_pre] = test( level, table );
        const auto time_total                   = timer.seconds();
        table->add_row( { { "level", level }, { "time_total", time_total } } );

        if ( level > 2 )
        {
            const double order_vel = prev_l2_error_vel / l2_error_vel;
            const double order_pre = prev_l2_error_pre / l2_error_pre;

            std::cout << "order_vel = " << order_vel << std::endl;
            std::cout << "order_pre = " << order_pre << std::endl;

            if ( order_vel < 3.7 )
            {
                return EXIT_FAILURE;
            }

            if ( order_vel < 3.7 )
            {
                return EXIT_FAILURE;
            }

            table->add_row( { { "level", level }, { "order_vel", order_vel }, { "order_pre", order_pre } } );
            table->print_pretty();
        }
        prev_l2_error_vel = l2_error_vel;
        prev_l2_error_pre = l2_error_pre;
    }

    table->query_rows_not_none( "order_vel" ).select_columns( { "level", "order_vel", "order_pre" } ).print_pretty();
    table->query_rows_not_none( "dofs_vel" )
        .select_columns( { "level", "dofs_vel", "l2_error_vel", "l2_error_pre" } )
        .print_pretty();

    return 0;
}