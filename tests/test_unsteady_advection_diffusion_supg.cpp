

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

using ScalarType = float;

struct VelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_;
    bool                           only_boundary_;

    VelocityInterpolator(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataVec< ScalarType, 3 >& data,
        bool                                  only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        data_( local_subdomain_id, x, y, r, 0 ) = -coords( 1 );
        data_( local_subdomain_id, x, y, r, 1 ) = coords( 0 );
        data_( local_subdomain_id, x, y, r, 2 ) = 0.0;
    }
};

struct InitialConditionInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > data_;
    bool                           only_boundary_;

    InitialConditionInterpolator(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataScalar< ScalarType >& data,
        bool                                  only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const dense::Vec< ScalarType, 3 > center{ 0.75, 0.0, 0.0 };
        const ScalarType                  radius = 0.1;

        if ( ( coords - center ).norm() < radius )
        {
            data_( local_subdomain_id, x, y, r ) = 1.0;
        }
    }
};

void test( int level, const std::shared_ptr< util::Table >& table )
{
    Kokkos::Timer timer;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    VectorW1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorW1Scalar< ScalarType > f( "f", domain, mask_data );
    VectorW1Vec< ScalarType >    u( "u", domain, mask_data );

    std::vector< VectorW1Scalar< ScalarType > > tmps;
    for ( int i = 0; i < 8; ++i )
    {
        tmps.emplace_back( "tmpp", domain, mask_data );
    }

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    std::cout << "Number of dofs: " << num_dofs << std::endl;

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;

    AD A( domain, subdomain_shell_coords, subdomain_radii, boundary_mask_data, u, 1e-3, 1e-2, false, false, 1.0 );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;

    Mass M( domain, subdomain_shell_coords, subdomain_radii, false );

    // Set up solution data.
    Kokkos::parallel_for(
        "velocity interpolation",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator( subdomain_shell_coords, subdomain_radii, u.grid_data(), false ) );

    Kokkos::fence();

    // Set up the initial temperature.
    Kokkos::parallel_for(
        "initial temp interpolation",
        local_domain_md_range_policy_nodes( domain ),
        InitialConditionInterpolator( subdomain_shell_coords, subdomain_radii, T.grid_data(), false ) );

    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 10, 1e-12, 1e-12 };

    linalg::solvers::PBiCGStab< AD > bicgstab( 2, solver_params, table, tmps );
    bicgstab.set_tag( "bicgstab_solver_level_" + std::to_string( level ) );

    const int timesteps = 10;

    io::XDMFOutput xdmf_output( ".", domain, subdomain_shell_coords, subdomain_radii );
    xdmf_output.add( T.grid_data() );

    constexpr auto vtk = false;

    if ( vtk )
    {
        xdmf_output.write( 0 );
    }

    for ( int ts = 1; ts < timesteps; ++ts )
    {
        std::cout << "Timestep " << ts << std::endl;

        linalg::apply( M, T, f );
        linalg::solvers::solve( bicgstab, A, T, f );

        table->print_pretty();
        table->clear();

        if ( vtk )
        {
            xdmf_output.write( ts );
        }
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    const int level = 4;

    test( level, table );

    return 0;
}
