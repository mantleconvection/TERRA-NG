

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/identity.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "grid/shell/bit_masks.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pbicgstab.hpp"
#include "linalg/solvers/pcg.hpp"
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

std::pair< double, double > test( int min_level, int max_level, const std::shared_ptr< util::Table >& table )
{
    using ScalarType = double;

    // Set up domains for all levels.

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    for ( int level = min_level; level <= max_level; level++ )
    {
        const int idx = level - min_level;

        domains.push_back( DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const auto num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    // Set up Stokes vectors for the finest grid.

    std::map< std::string, VectorW1IsoW2W1< ScalarType > > stok_vecs;
    std::vector< std::string >                             stok_vec_names = { "u", "f", "solution", "error" };
    constexpr int                                          num_stok_tmps  = 8;

    for ( int i = 0; i < num_stok_tmps; i++ )
    {
        stok_vec_names.push_back( "tmp_" + std::to_string( i ) );
    }

    for ( const auto& name : stok_vec_names )
    {
        stok_vecs[name] = VectorW1IsoW2W1< ScalarType >(
            name,
            domains[velocity_level],
            domains[pressure_level],
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    auto& u        = stok_vecs["u"];
    auto& f        = stok_vecs["f"];
    auto& solution = stok_vecs["solution"];
    auto& error    = stok_vecs["error"];

    // Set up tmp vecs for multigrid.

    std::vector< VectorW1Vec< ScalarType > > tmp_mg;
    std::vector< VectorW1Vec< ScalarType > > tmp_mg_r;
    std::vector< VectorW1Vec< ScalarType > > tmp_mg_e;

    for ( int level = 0; level < num_levels; level++ )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );
        }
    }

    // Counting DoFs.

    const auto num_dofs_velocity =
        3 * kernels::common::count_masked< long >( mask_data[num_levels - 1], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( mask_data[num_levels - 2], grid::NodeOwnershipFlag::OWNED );

    // Set up operators.
    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };
    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };
    using Stokes      = fe::wedge::operators::shell::Stokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        bcs,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        bcs_neumann,
        false );

    Stokes K_neumann_diag(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        bcs_neumann,
        true );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // Multigrid operators

    std::vector< Viscous >      A_diag;
    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;

    std::vector< VectorW1Vec< ScalarType > > inverse_diagonals;

    for ( int level = 0; level < num_levels; level++ )
    {
        A_diag.emplace_back(
            domains[level], coords_shell[level], coords_radii[level], boundary_mask_data[level], bcs, true );

        inverse_diagonals.emplace_back(
            "inverse_diagonal_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorW1Vec< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( level ), domains[level], mask_data[level] );

        linalg::assign( tmp, 1.0 );
        linalg::apply( A_diag[level], tmp, inverse_diagonals.back() );
        linalg::invert_entries( inverse_diagonals.back() );

        if ( level < num_levels - 1 )
        {
            A_c.emplace_back(
                domains[level], coords_shell[level], coords_radii[level], boundary_mask_data[level], bcs, false );
            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // Set up solution data.

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            stok_vecs["solution"].block_1().grid_data(),
            false ) );

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domains[pressure_level] ),
        SolutionPressureInterpolator(
            coords_shell[pressure_level],
            coords_radii[pressure_level],
            stok_vecs["solution"].block_2().grid_data(),
            false ) );

    // Set up rhs data.

    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        RHSVelocityInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level], stok_vecs["tmp_1"].block_1().grid_data() ) );

    linalg::apply( M, stok_vecs["tmp_1"].block_1(), stok_vecs["f"].block_1() );

    // Set up boundary data.

    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            stok_vecs["tmp_0"].block_1().grid_data(),
            true ) );

    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann,
        K_neumann_diag,
        stok_vecs["tmp_0"],
        stok_vecs["tmp_1"],
        stok_vecs["f"],
        boundary_mask_data[velocity_level],
        grid::shell::ShellBoundaryFlag::BOUNDARY );

    // Set up solvers.

    // Multigrid preconditioner.

    using Smoother = linalg::solvers::Jacobi< Viscous >;

    std::vector< Smoother > smoothers;
    for ( int level = 0; level < num_levels; level++ )
    {
        constexpr auto smoother_prepost = 3;
        constexpr auto omega            = 0.666;
        smoothers.emplace_back( inverse_diagonals[level], smoother_prepost, tmp_mg[level], omega );
    }

    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;

    std::vector< VectorW1Vec< ScalarType > > coarse_grid_tmps;
    for ( int i = 0; i < 4; i++ )
    {
        coarse_grid_tmps.emplace_back( "tmp_coarse_grid", domains[0], mask_data[0] );
    }

    CoarseGridSolver coarse_grid_solver(
        linalg::solvers::IterativeSolverParameters{ 100, 1e-8, 1e-16 }, table, coarse_grid_tmps );

    constexpr auto num_mg_cycles = 2;

    using PrecVisc = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;
    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_grid_solver, num_mg_cycles, 1e-8 );

    using PrecSchur = linalg::solvers::IdentitySolver< fe::wedge::operators::shell::Identity< ScalarType > >;
    PrecSchur prec_22;

    using PrecStokes = linalg::solvers::BlockDiagonalPreconditioner2x2<
        Stokes,
        Viscous,
        fe::wedge::operators::shell::Identity< ScalarType >,
        PrecVisc,
        PrecSchur >;

    PrecStokes prec_stokes( K.block_11(), fe::wedge::operators::shell::Identity< ScalarType >(), prec_11, prec_22 );

    linalg::solvers::IterativeSolverParameters solver_params{ 100, 1e-8, 1e-12 };

    std::vector< VectorW1IsoW2W1< ScalarType > > tmp_bicgstab( 8 );
    for ( int i = 0; i < 8; i++ )
    {
        tmp_bicgstab[i] = stok_vecs["tmp_" + std::to_string( i )];
    }

    linalg::solvers::PBiCGStab< Stokes, PrecStokes > pbicgstab( 2, solver_params, table, tmp_bicgstab, prec_stokes );

    linalg::solvers::solve( pbicgstab, K, u, f );

    const double avg_pressure_solution =
        kernels::common::masked_sum(
            solution.block_2().grid_data(), solution.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;
    const double avg_pressure_approximation =
        kernels::common::masked_sum(
            u.block_2().grid_data(), u.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;

    linalg::lincomb( solution.block_2(), { 1.0 }, { solution.block_2() }, -avg_pressure_solution );
    linalg::lincomb( u.block_2(), { 1.0 }, { u.block_2() }, -avg_pressure_approximation );

    linalg::apply( K, u, stok_vecs["tmp_6"] );
    linalg::lincomb( stok_vecs["tmp_5"], { 1.0, -1.0 }, { f, stok_vecs["tmp_6"] } );
    const auto inf_residual_vel = linalg::norm_inf( stok_vecs["tmp_5"].block_1() );
    const auto inf_residual_pre = linalg::norm_inf( stok_vecs["tmp_5"].block_2() );

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error_velocity =
        std::sqrt( dot( error.block_1(), error.block_1() ) / static_cast< double >( num_dofs_velocity ) );
    const auto l2_error_pressure =
        std::sqrt( dot( error.block_2(), error.block_2() ) / static_cast< double >( num_dofs_pressure ) );

    table->add_row(
        { { "level", max_level },
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

    for ( int level = 1; level < 5; ++level )
    {
        std::cout << "level = " << level << std::endl;
        Kokkos::Timer timer;
        timer.reset();
        const auto [l2_error_vel, l2_error_pre] = test( 0, level, table );
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
        }
        prev_l2_error_vel = l2_error_vel;
        prev_l2_error_pre = l2_error_pre;

        table->query_rows_equals( "tag", "pbicgstab_solver" ).print_pretty();
        table->clear();
    }

    table->query_rows_not_none( "order_vel" ).select_columns( { "level", "order_vel", "order_pre" } ).print_pretty();
    table->query_rows_not_none( "dofs_vel" )
        .select_columns( { "level", "dofs_vel", "l2_error_vel", "l2_error_pre" } )
        .print_pretty();

    return 0;
}