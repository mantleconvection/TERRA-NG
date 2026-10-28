

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "linalg/solvers/pbicgstab.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
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
    Grid4DDataScalar< double > data_;
    double                     t_;
    bool                       only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        const double                      t,
        bool                              only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , t_( t )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double pi = Kokkos::numbers::pi;

        const double value = Kokkos::sin( pi * coords( 0 ) ) * Kokkos::sin( pi * coords( 1 ) ) *
                             Kokkos::sin( pi * coords( 2 ) ) * Kokkos::exp( -3.0 * pi * pi * t_ );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

void test( int level, int timesteps, double dt, const std::shared_ptr< util::Table >& table, double l2_error_threshold )
{
    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    const auto max_level = domain.domain_info().subdomain_max_refinement_level();
    std::cout << "Max level: " << max_level << std::endl;

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorW1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorW1Scalar< ScalarType > T_prev( "T_prev", domain, mask_data );
    VectorW1Scalar< ScalarType > f( "f", domain, mask_data );
    VectorW1Scalar< ScalarType > solution( "solution", domain, mask_data );
    VectorW1Scalar< ScalarType > error( "error", domain, mask_data );

    VectorW1Vec< ScalarType, 3 > u( "u", domain, mask_data );

    u.mask_data() = mask_data;

    std::vector< VectorW1Scalar< double > > tmps;
    for ( int i = 0; i < 8; ++i )
    {
        tmps.emplace_back( "tmpp", domain, mask_data );
    }

    linalg::assign( tmps[0], 1.0 );
    const auto num_dofs = linalg::dot( tmps[0], tmps[0] );
    std::cout << "Number of dofs: " << num_dofs << std::endl;

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;

    AD A_bdf1( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, 1.0, dt, true, false, 1.0 );
    AD A_bdf1_neumann_diag(
        domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, 1.0, dt, false, true, 1.0 );
    AD A_bdf1_neumann(
        domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, 1.0, dt, false, false, 1.0 );

    AD A_bdf2(
        domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, 1.0, dt, true, false, 3.0 / 2.0 );
    AD A_bdf2_neumann_diag(
        domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, 1.0, dt, false, true, 3.0 / 2.0 );
    AD A_bdf2_neumann(
        domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, 1.0, dt, false, false, 3.0 / 2.0 );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;

    Mass M( domain, subdomain_shell_coords, subdomain_radii, false );

    // Set up the initial temperature.
    Kokkos::parallel_for(
        "initial temp interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, T.grid_data(), 0.0, false ) );

    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 1000, 1e-12, 1e-12 };

    linalg::solvers::PBiCGStab< AD > bicgstab( 2, solver_params, table, tmps );
    bicgstab.set_tag( "bicgstab_solver_level_" + std::to_string( level ) );

    io::XDMFOutput< ScalarType > xdmf_output( ".", domain, subdomain_shell_coords, subdomain_radii );
    xdmf_output.add( T.grid_data() );
    xdmf_output.add( solution.grid_data() );
    xdmf_output.add( error.grid_data() );

    constexpr auto vtk = false;

    if ( vtk )
    {
        xdmf_output.write( 0 );
    }

    double l2_error = 0;

    for ( int ts = 1; ts <= 1; ++ts )
    {
        linalg::assign( T_prev, T );

        linalg::apply( M, T, f );

        auto& g = tmps[0];
        assign( g, 0.0 );
        Kokkos::parallel_for(
            "boundary temp interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionInterpolator( subdomain_shell_coords, subdomain_radii, g.grid_data(), dt * ts, true ) );

        fe::strong_algebraic_dirichlet_enforcement_poisson_like(
            A_bdf1_neumann,
            A_bdf1_neumann_diag,
            g,
            tmps[1],
            f,
            boundary_mask_data,
            grid::shell::ShellBoundaryFlag::BOUNDARY );

        linalg::solvers::solve( bicgstab, A_bdf1, T, f );

        Kokkos::parallel_for(
            "solution interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionInterpolator( subdomain_shell_coords, subdomain_radii, solution.grid_data(), dt * ts, false ) );

        linalg::lincomb( error, { 1.0, -1.0 }, { T, solution } );
        l2_error = std::sqrt( dot( error, error ) / num_dofs );

        if ( true )
        {
            std::cout << "L2 error (ts = " << ts << "): " << l2_error << std::endl;
        }

        // table.print_pretty();
        table->clear();

        if ( vtk )
        {
            xdmf_output.write( ts );
        }
    }

    for ( int ts = 2; ts <= timesteps; ++ts )
    {
        linalg::apply( M, T_prev, tmps[0] );
        linalg::apply( M, T, tmps[1] );

        linalg::lincomb( f, { 2.0, -0.5 }, { tmps[1], tmps[0] } );

        linalg::assign( T_prev, T );

        auto& g = tmps[0];
        assign( g, 0.0 );
        Kokkos::parallel_for(
            "boundary temp interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionInterpolator( subdomain_shell_coords, subdomain_radii, g.grid_data(), dt * ts, true ) );

        fe::strong_algebraic_dirichlet_enforcement_poisson_like(
            A_bdf2_neumann,
            A_bdf2_neumann_diag,
            g,
            tmps[1],
            f,
            boundary_mask_data,
            grid::shell::ShellBoundaryFlag::BOUNDARY );

        linalg::solvers::solve( bicgstab, A_bdf2, T, f );

        Kokkos::parallel_for(
            "solution interpolation",
            local_domain_md_range_policy_nodes( domain ),
            SolutionInterpolator( subdomain_shell_coords, subdomain_radii, solution.grid_data(), dt * ts, false ) );

        linalg::lincomb( error, { 1.0, -1.0 }, { T, solution } );
        l2_error = std::sqrt( dot( error, error ) / num_dofs );

        if ( true )
        {
            std::cout << "L2 error (ts = " << ts << "): " << l2_error << std::endl;
        }

        // table.print_pretty();
        table->clear();

        if ( vtk )
        {
            xdmf_output.write( ts );
        }
    }

    if ( l2_error > l2_error_threshold )
    {
        throw std::runtime_error( "Error too large!" );
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    test( 2, 10, 1e-3, table, 0.0059 );
    test( 3, 10, 1e-3, table, 0.0014 );
    test( 4, 10, 1e-3, table, 0.00028 );

    test( 4, 5, 4e-2, table, 0.0022 );
    test( 4, 10, 2e-2, table, 0.00017 );
    test( 4, 20, 1e-2, table, 3.9e-05 );

    return 0;
}
