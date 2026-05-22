
#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_simple.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "grid/shell/bit_masks.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/vtk.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

#include <map>
#include <algorithm>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
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
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                       only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        bool                              only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , mask_( mask )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

          const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

          if ( !only_boundary_ || on_boundary )
            {
            data_( local_subdomain_id, x, y, r, 0 ) =
                Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sin( 2 * coords( 2 ) ) * Kokkos::sinh( coords( 1 ) );
            data_( local_subdomain_id, x, y, r, 1 ) =
                2 * Kokkos::sin( 2 * coords( 1 ) ) * Kokkos::sin( 2 * coords( 2 ) ) * Kokkos::sinh( coords( 0 ) );
            data_( local_subdomain_id, x, y, r, 2 ) =
                4 * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sin( 2 * coords( 1 ) ) * Kokkos::sinh( coords( 2 ) );
        }
    }
};

struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KInterpolator(
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
        const double                  value  = 2 + Kokkos::sin( coords( 2 ) );
        data_( local_subdomain_id, x, y, r ) = value;
    }
};

struct RHSInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    RHSInterpolator(
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

        // x component of rhs
        {
            const real_t x0  = Kokkos::sinh( coords( 1 ) );
            const real_t x1  = Kokkos::sin( coords( 2 ) ) + 2;
            const real_t x2  = 2 * coords( 0 );
            const real_t x3  = Kokkos::sin( x2 );
            const real_t x4  = 2 * coords( 2 );
            const real_t x5  = Kokkos::sin( x4 );
            const real_t x6  = x0 * x3;
            const real_t x7  = Kokkos::cos( x2 );
            const real_t x8  = 2 * coords( 1 );
            const real_t x9  = Kokkos::sin( x8 );
            const real_t x10 = x5 * x6;
            const real_t x11 = Kokkos::cosh( coords( 2 ) );
            const real_t x12 = 2 * x1;
            const real_t x13 = x5 * Kokkos::cos( x8 ) * Kokkos::cosh( coords( 0 ) );
            data_( local_subdomain_id, x, y, r, 0 ) =
                8.0 * x0 * x1 * x3 * x5 + 0.66666666666666663 * x1 * ( -4 * x10 + 8 * x11 * x7 * x9 + 4 * x13 ) -
                x12 * ( -2.0 * x10 + 4.0 * x11 * x7 * x9 ) - x12 * ( 0.5 * x10 + 2.0 * x13 ) -
                2 * ( 1.0 * x6 * Kokkos::cos( x4 ) + 4.0 * x7 * x9 * Kokkos::sinh( coords( 2 ) ) ) *
                    Kokkos::cos( coords( 2 ) );
        }

        // y component of rhs
        {
            const real_t x0  = Kokkos::sinh( coords( 0 ) );
            const real_t x1  = Kokkos::sin( coords( 2 ) ) + 2;
            const real_t x2  = 2 * coords( 1 );
            const real_t x3  = Kokkos::sin( x2 );
            const real_t x4  = 2 * coords( 2 );
            const real_t x5  = Kokkos::sin( x4 );
            const real_t x6  = Kokkos::cos( x2 );
            const real_t x7  = 2 * coords( 0 );
            const real_t x8  = Kokkos::sin( x7 );
            const real_t x9  = 4.0 * x6 * x8;
            const real_t x10 = x0 * x3;
            const real_t x11 = Kokkos::cosh( coords( 2 ) );
            const real_t x12 = x10 * x5;
            const real_t x13 = 2 * x1;
            const real_t x14 = 1.0 * x5;
            const real_t x15 = std::cos( x7 ) * Kokkos::cosh( coords( 1 ) );
            data_( local_subdomain_id, x, y, r, 1 ) =
                16.0 * x0 * x1 * x3 * x5 + 0.66666666666666663 * x1 * ( 8 * x11 * x6 * x8 - 8 * x12 + 2 * x15 * x5 ) -
                x13 * ( x10 * x14 + x14 * x15 ) - x13 * ( x11 * x9 - 4.0 * x12 ) -
                2 * ( 2.0 * x10 * Kokkos::cos( x4 ) + x9 * Kokkos::sinh( coords( 2 ) ) ) * Kokkos::cos( coords( 2 ) );
        }

        // z component of rhs
        {
            const real_t x0  = Kokkos::cos( coords( 2 ) );
            const real_t x1  = 2 * coords( 0 );
            const real_t x2  = 2 * coords( 1 );
            const real_t x3  = Kokkos::sin( x1 ) * Kokkos::sin( x2 );
            const real_t x4  = x3 * Kokkos::cosh( coords( 2 ) );
            const real_t x5  = Kokkos::sin( coords( 2 ) ) + 2;
            const real_t x6  = x3 * Kokkos::sinh( coords( 2 ) );
            const real_t x7  = 8.0 * x6;
            const real_t x8  = Kokkos::sinh( coords( 1 ) );
            const real_t x9  = Kokkos::cos( x1 );
            const real_t x10 = 2 * coords( 2 );
            const real_t x11 = Kokkos::cos( x10 );
            const real_t x12 = 2 * x5;
            const real_t x13 = Kokkos::sinh( coords( 0 ) );
            const real_t x14 = Kokkos::cos( x2 );
            const real_t x15 = Kokkos::sin( x10 );
            const real_t x16 = x8 * x9;
            const real_t x17 = x13 * x14;
            data_( local_subdomain_id, x, y, r, 2 ) =
                -8.0 * x0 * x4 + 0.66666666666666663 * x0 * ( 2 * x15 * x16 + 4 * x15 * x17 + 4 * x4 ) -
                x12 * ( 4.0 * x11 * x13 * x14 - x7 ) - x12 * ( 2.0 * x11 * x8 * x9 - x7 ) - x5 * x7 +
                0.66666666666666663 * x5 * ( 4 * x11 * x16 + 8 * x11 * x17 + 4 * x6 );
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

double test( int level, int level_subdomains, const std::shared_ptr< util::Table >& table )
{
    Kokkos::Timer timer;

    using ScalarType = double;

    // CHANGED: allow varying number of subdomains (refinement level)
    const auto domain =
        DistributedDomain::create_uniform( level, level, 0.5, 1.0, level_subdomains, level_subdomains );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorQ1Vec< ScalarType >    u( "u", domain, mask_data );
    VectorQ1Vec< ScalarType >    g( "g", domain, mask_data );
    VectorQ1Vec< ScalarType >    Adiagg( "Adiagg", domain, mask_data );
    VectorQ1Vec< ScalarType >    tmp( "tmp", domain, mask_data );
    VectorQ1Vec< ScalarType >    solution( "solution", domain, mask_data );
    VectorQ1Vec< ScalarType >    error( "error", domain, mask_data );
    VectorQ1Vec< ScalarType >    b( "b", domain, mask_data );
    VectorQ1Vec< ScalarType >    r( "r", domain, mask_data );
    VectorQ1Scalar< ScalarType > k( "k", domain, mask_data );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    // Set up coefficient data.
    Kokkos::parallel_for(
        "coefficient interpolation",
        local_domain_md_range_policy_nodes( domain ),
        KInterpolator( subdomain_shell_coords, subdomain_radii, k.grid_data() ) );

    Kokkos::fence();

    using Epsilon = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarType, 3 >;

    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };
    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };

    Epsilon A( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, k.grid_data(), bcs, false );
    Epsilon A_neumann(
        domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, k.grid_data(), bcs_neumann, false );
    Epsilon A_neumann_diag(
        domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, k.grid_data(), bcs_neumann, true );

    // Force the hex 1-pt Gauss DN path for all three operators.
    A.set_kernel_path( Epsilon::KernelPath::FastDirichletNeumannHex );
    A_neumann.set_kernel_path( Epsilon::KernelPath::FastDirichletNeumannHex );
    A_neumann_diag.set_kernel_path( Epsilon::KernelPath::FastDirichletNeumannHex );

    using Mass = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;
    Mass M( domain, subdomain_shell_coords, subdomain_radii, false );

    // Set up solution data.
    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, solution.grid_data(), boundary_mask_data, false ) );

    // Set up boundary data.
    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, g.grid_data(), boundary_mask_data, true ) );

    Kokkos::fence();
    // Set up rhs data.
    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domain ),
        RHSInterpolator( subdomain_shell_coords, subdomain_radii, tmp.grid_data() ) );

    Kokkos::fence();

    linalg::apply( M, tmp, b );

   fe::strong_algebraic_dirichlet_enforcement_vectorlaplace_like(
        A_neumann, A_neumann_diag, g, tmp, b, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

     Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 500, 1e-15, 1e-15 };

    linalg::solvers::PCG< Epsilon > pcg( solver_params, table, { tmp, Adiagg, error, r } );
    pcg.set_tag( "pcg_solver_level_" + std::to_string( level ) + "_sub_" + std::to_string( level_subdomains ) );

    Kokkos::fence();
    timer.reset();
    linalg::solvers::solve( pcg, A, u, b );
    Kokkos::fence();
    const auto time_solver = timer.seconds();

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error = std::sqrt( dot( error, error ) / num_dofs );

    table->add_row( { { "level", level },
                      { "level_subdomains", level_subdomains },
                      { "dofs", num_dofs },
                      { "l2_error", l2_error },
                      { "time_solver", time_solver } } );

    return l2_error;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    // errors[level][level_subdomains]
    std::map< int, std::map< int, double > > errors;

    for ( int level = 2; level <= 4; ++level )
    {
        errors[level];

        // requested: subdomain refinement levels 0-3
        // (optional clamp so we don't ask for more subdomain refinement than the global level)
        const int max_sub = 0;

        for ( int level_subdomains = 0; level_subdomains <= max_sub; ++level_subdomains )
        {
            Kokkos::Timer timer;
            timer.reset();

            errors[level][level_subdomains] = test( level, level_subdomains, table );
            const auto time_total           = timer.seconds();

            std::cout << "level: " << level << ", "
                      << "level_subdomains: " << level_subdomains << ", "
                      << "error: " << errors[level][level_subdomains] << ", "
                      << "time_total: " << time_total << std::endl;

            // sanity: same global level should give same error irrespective of subdomain partitioning
            if ( level_subdomains > 0 )
            {
                if ( std::abs( errors[level][level_subdomains] - errors[level][level_subdomains - 1] ) > 1e-12 )
                {
                    Kokkos::abort( "Same level should have same error - regardless of number of subdomains." );
                }
            }

            table->add_row( { { "level", level },
                              { "level_subdomains", level_subdomains },
                              { "time_total", time_total } } );
        }

        // convergence order across levels (use any subdomain level; they should all match)
        if ( level > 1 )
        {
            const int prev_sub = std::min( 3, level - 1 );
            const int curr_sub = std::min( 3, level );

            const double prev_err = errors[level - 1][prev_sub];
            const double curr_err = errors[level][curr_sub];
            const double order    = prev_err / curr_err;

            std::cout << "\nerror(level=" << level << ") = " << curr_err << std::endl;
            std::cout << "order = " << order << std::endl;

            // keep your threshold here (was commented in your original)
            // if ( order < 3.8 ) { return EXIT_FAILURE; }

            table->add_row( { { "level", level }, { "order", order } } );
        }

        std::cout << std::endl;
    }

    table->query_rows_not_none( "order" ).select_columns( { "level", "order" } ).print_pretty();
    table->query_rows_not_none( "dofs" )
        .select_columns( { "level", "level_subdomains", "dofs", "l2_error", "time_solver" } )
        .print_pretty();

    // PCG residual history — one row per iteration per (level, sub) tag.
    table->query_rows_not_none( "absolute_residual" )
        .select_columns( { "tag", "iteration", "relative_residual", "absolute_residual" } )
        .print_pretty();

    return 0;
}
