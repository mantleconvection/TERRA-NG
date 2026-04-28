

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
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

using util::logroot;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;

struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataScalar< double >                         data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataScalar< double >&                         data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        bool                                                      only_boundary )
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
        // const double                  value  = coords( 0 ) * Kokkos::sin( coords( 1 ) ) * Kokkos::sinh( coords( 2 ) );
        const double value = ( 1.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );

        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || on_boundary )
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

double test( int level, int level_subdomains, const std::shared_ptr< util::Table >& table )
{
    Kokkos::Timer timer;

    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform( level, level, 0.5, 1.0, level_subdomains, level_subdomains );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorQ1Scalar< ScalarType > u( "u", domain, mask_data );
    VectorQ1Scalar< ScalarType > g( "g", domain, mask_data );
    VectorQ1Scalar< ScalarType > Adiagg( "Adiagg", domain, mask_data );
    VectorQ1Scalar< ScalarType > tmp( "tmp", domain, mask_data );
    VectorQ1Scalar< ScalarType > solution( "solution", domain, mask_data );
    VectorQ1Scalar< ScalarType > error( "error", domain, mask_data );
    VectorQ1Scalar< ScalarType > b( "b", domain, mask_data );
    VectorQ1Scalar< ScalarType > r( "r", domain, mask_data );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    logroot << "num_dofs = " << num_dofs << std::endl;

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    using Laplace = fe::wedge::operators::shell::Laplace< ScalarType >;

    Laplace A( domain, coords_shell, coords_radii, boundary_mask_data, true, false );
    Laplace A_neumann( domain, coords_shell, coords_radii, boundary_mask_data, false, false );
    Laplace A_neumann_diag( domain, coords_shell, coords_radii, boundary_mask_data, false, true );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;

    Mass M( domain, coords_shell, coords_radii, false );

    // Set up solution data.
    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( coords_shell, coords_radii, solution.grid_data(), boundary_mask_data, false ) );

    Kokkos::fence();

    // Set up boundary data.
    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( coords_shell, coords_radii, g.grid_data(), boundary_mask_data, true ) );

    Kokkos::fence();

    // Set up rhs data.
    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domain ),
        RHSInterpolator( coords_shell, coords_radii, tmp.grid_data() ) );

    Kokkos::fence();

    linalg::apply( M, tmp, b );

    fe::strong_algebraic_dirichlet_enforcement_poisson_like(
        A_neumann, A_neumann_diag, g, tmp, b, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );

    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 100, 1e-12, 1e-12 };

    linalg::solvers::PCG< Laplace > pcg( solver_params, table, { tmp, Adiagg, error, r } );
    pcg.set_tag( "pcg_solver_level_" + std::to_string( level ) );

    Kokkos::fence();
    timer.reset();
    linalg::solvers::solve( pcg, A, u, b );
    Kokkos::fence();
    const auto time_solver = timer.seconds();

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error = std::sqrt( dot( error, error ) / num_dofs );

    if ( false )
    {
        io::XDMFOutput< double > xdmf( ".", domain, coords_shell, coords_radii );
        xdmf.add( g.grid_data() );
        xdmf.add( u.grid_data() );
        xdmf.add( solution.grid_data() );
        xdmf.add( error.grid_data() );
        xdmf.write();
    }

    table->add_row(
        { { "level", level }, { "dofs", num_dofs }, { "l2_error", l2_error }, { "time_solver", time_solver } } );

    return l2_error;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    std::map< int, std::map< int, double > > errors;

    for ( int level = 0; level < 5; ++level )
    {
        errors[level];

        for ( int level_subdomains = 0; level_subdomains <= level; ++level_subdomains )
        {
            Kokkos::Timer timer;
            timer.reset();
            errors[level][level_subdomains] = test( level, level_subdomains, table );
            const auto time_total           = timer.seconds();

            logroot << "level: " << level << ", ";
            logroot << "level_subdomain: " << level_subdomains << ", ";
            logroot << "error: " << errors[level][level_subdomains] << ", ";
            logroot << "time: " << time_total << std::endl;

            if ( level_subdomains > 0 )
            {
                if ( errors[level][level_subdomains] - errors[level][level_subdomains - 1] > 1e-12 )
                {
                    Kokkos::abort( "Same level should have same error - regardless of number of subdomains." );
                }
            }
        }

        if ( level > 1 )
        {
            const double order = errors[level - 1][level - 1] / errors[level][level];
            logroot << std::endl;
            logroot << "error = " << errors[level][level] << std::endl;
            logroot << "order = " << order << std::endl;
            if ( order < 3.4 )
            {
                Kokkos::abort( "Grid convergence order too low." );
            }

            if ( level == 4 && errors[level][level] > 1e-4 )
            {
                Kokkos::abort( "Error at level 4 too large." );
            }
        }

        logroot << std::endl;
    }

    return 0;
}