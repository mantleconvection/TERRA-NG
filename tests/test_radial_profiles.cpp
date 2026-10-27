

#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/pminres.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "shell/radial_profiles.hpp"
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
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

KOKKOS_INLINE_FUNCTION
double test_func( const dense::Vec< double, 3 >& coords )
{
    const auto x = coords( 0 );
    const auto r = coords.norm();
    return r + x / r;
}

struct ScalarInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    ScalarInterpolator(
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
        data_( local_subdomain_id, x, y, r ) = test_func( coords );
    }
};

void test( int level )
{
    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    const auto coords_shell  = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii  = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );
    const auto shell_indices = terra::grid::shell::subdomain_shell_idx( domain );

    auto mask_data = grid::setup_node_ownership_mask_data( domain );

    VectorQ1Scalar< ScalarType > u( "u", domain, mask_data );

    // Set up solution data.
    Kokkos::parallel_for(
        "function interpolation",
        local_domain_md_range_policy_nodes( domain ),
        ScalarInterpolator( coords_shell, coords_radii, u.grid_data() ) );

    Kokkos::fence();

    auto profiles_device = shell::radial_profiles( u, shell_indices, domain.domain_info().radii().size() );

    auto profiles = shell::radial_profiles_to_table(
        shell::radial_profiles( u, shell_indices, domain.domain_info().radii().size() ), domain.domain_info().radii() );

    profiles.print_pretty();

    {
        std::ofstream out( "/tmp/test_radial_profiles.csv" );
        profiles.print_csv( out );
    }

    {
        std::ofstream out( "/tmp/test_radial_profiles.jsonl" );
        profiles.print_jsonl( out );
    }

    auto profiles_host_min = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profiles_device.radial_min_ );
    auto profiles_host_avg = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profiles_device.radial_avg_ );
    auto profiles_host_max = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profiles_device.radial_max_ );

    for ( int shell_idx = 0; shell_idx < profiles_host_min.extent( 0 ); shell_idx++ )
    {
        const auto r = domain.domain_info().radii()[shell_idx];

        const auto expected_min = r - 1.0;
        const auto expected_max = r + 1.0;
        const auto expected_avg = r;

        if ( std::abs( profiles_host_min( shell_idx ) - expected_min ) > 1e-12 )
        {
            Kokkos::abort( "Min not correct." );
        }

        if ( std::abs( profiles_host_avg( shell_idx ) - expected_avg ) > 1e-12 )
        {
            Kokkos::abort( "Avg not correct." );
        }

        if ( std::abs( profiles_host_max( shell_idx ) - expected_max ) > 1e-12 )
        {
            Kokkos::abort( "Max not correct." );
        }
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    test( 4 );

    return 0;
}