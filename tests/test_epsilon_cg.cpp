

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon.hpp"
#include "fe/wedge/operators/shell/epsilon_simple.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/pcg.hpp"
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
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;

struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    bool                       only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data,
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

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
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
    bool                       only_boundary_;

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
            const real_t x11 = 2 * x1;
            data_( local_subdomain_id, x, y, r, 0 ) =
                8.0 * x0 * x1 * x3 * x5 - x11 * ( -2.0 * x10 + 4.0 * x7 * x9 * Kokkos::cosh( coords( 2 ) ) ) -
                x11 * ( 0.5 * x10 + 2.0 * x5 * Kokkos::cos( x8 ) * std::cosh( coords( 0 ) ) ) -
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
            const real_t x6  = 2 * coords( 0 );
            const real_t x7  = 4.0 * Kokkos::sin( x6 ) * Kokkos::cos( x2 );
            const real_t x8  = x0 * x3;
            const real_t x9  = 2 * x1;
            const real_t x10 = 1.0 * x5;
            data_( local_subdomain_id, x, y, r, 1 ) =
                16.0 * x0 * x1 * x3 * x5 - x9 * ( x10 * x8 + x10 * Kokkos::cos( x6 ) * Kokkos::cosh( coords( 1 ) ) ) -
                x9 * ( -4.0 * x5 * x8 + x7 * Kokkos::cosh( coords( 2 ) ) ) -
                2 * ( x7 * Kokkos::sinh( coords( 2 ) ) + 2.0 * x8 * Kokkos::cos( x4 ) ) * Kokkos::cos( coords( 2 ) );
        }

        // z component of rhs
        {
            const real_t x0 = 2 * coords( 0 );
            const real_t x1 = 2 * coords( 1 );
            const real_t x2 = 8.0 * Kokkos::sin( x0 ) * Kokkos::sin( x1 );
            const real_t x3 = Kokkos::sin( coords( 2 ) ) + 2;
            const real_t x4 = x2 * Kokkos::sinh( coords( 2 ) );
            const real_t x5 = Kokkos::cos( 2 * coords( 2 ) );
            const real_t x6 = 2 * x3;
            data_( local_subdomain_id, x, y, r, 2 ) =
                -x2 * Kokkos::cos( coords( 2 ) ) * Kokkos::cosh( coords( 2 ) ) - x3 * x4 -
                x6 * ( -x4 + 2.0 * x5 * Kokkos::cos( x0 ) * Kokkos::sinh( coords( 1 ) ) ) -
                x6 * ( -x4 + 4.0 * x5 * Kokkos::cos( x1 ) * Kokkos::sinh( coords( 0 ) ) );
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

double test( int level, const std::shared_ptr< util::Table >& table )
{
    Kokkos::Timer timer;

    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data = grid::setup_node_ownership_mask_data( domain );

    VectorW1Vec< ScalarType >    u( "u", domain, mask_data );
    VectorW1Vec< ScalarType >    g( "g", domain, mask_data );
    VectorW1Vec< ScalarType >    Adiagg( "Adiagg", domain, mask_data );
    VectorW1Vec< ScalarType >    tmp( "tmp", domain, mask_data );
    VectorW1Vec< ScalarType >    solution( "solution", domain, mask_data );
    VectorW1Vec< ScalarType >    error( "error", domain, mask_data );
    VectorW1Vec< ScalarType >    b( "b", domain, mask_data );
    VectorW1Vec< ScalarType >    r( "r", domain, mask_data );
    VectorW1Scalar< ScalarType > k( "k", domain, mask_data );

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
    using Epsilon = fe::wedge::operators::shell::Epsilon< ScalarType, 3 >;

    Epsilon A( domain, subdomain_shell_coords, subdomain_radii, k.grid_data(), true, false );
    Epsilon A_neumann( domain, subdomain_shell_coords, subdomain_radii, k.grid_data(), false, false );
    Epsilon A_neumann_diag( domain, subdomain_shell_coords, subdomain_radii, k.grid_data(), false, true );

    using Mass = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;

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
        RHSInterpolator( subdomain_shell_coords, subdomain_radii, tmp.grid_data() ) );

    linalg::apply( M, tmp, b );

    linalg::apply( A_neumann_diag, g, Adiagg );
    linalg::apply( A_neumann, g, tmp );

    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );

    Kokkos::parallel_for(
        "set on boundary",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        SetOnBoundary( Adiagg.grid_data(), b.grid_data(), domain.domain_info().subdomain_num_nodes_radially() ) );

    linalg::solvers::IterativeSolverParameters solver_params{ 500, 1e-15, 1e-15 };

    linalg::solvers::PCG< Epsilon > pcg( solver_params, table, { tmp, Adiagg, error, r } );
    pcg.set_tag( "pcg_solver_level_" + std::to_string( level ) );

    Kokkos::fence();
    timer.reset();
    linalg::solvers::solve( pcg, A, u, b );
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

    for ( int level = 0; level < 6; ++level )
    {
        Kokkos::Timer timer;
        timer.reset();
        double     l2_error   = test( level, table );
        const auto time_total = timer.seconds();
        table->add_row( { { "level", level }, { "time_total", time_total } } );

        const double order = prev_l2_error / l2_error;
        std::cout << "error = " << l2_error << std::endl;
        std::cout << "order = " << order << std::endl;
        std::cout << "time_total = " << time_total << std::endl;
        /*if ( order < 3.8 )
            {
                return EXIT_FAILURE;
            }*/

        table->add_row( { { "level", level }, { "order", prev_l2_error / l2_error } } );
        prev_l2_error = l2_error;
    }

    table->query_rows_not_none( "order" ).select_columns( { "level", "order" } ).print_pretty();
    table->query_rows_not_none( "dofs" )
        .select_columns( { "level", "dofs", "l2_error", "time_solver" } )
        .print_pretty();
    return 0;
}
