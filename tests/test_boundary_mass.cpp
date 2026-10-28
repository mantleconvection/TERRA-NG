

#include "communication/shell/communication.hpp"
#include "fe/wedge/operators/shell/boundary_mass.hpp"
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
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorW1Scalar;

using ScalarType = double;

struct SurfaceData
{
    ScalarType surface;
    ScalarType surface_analytical;
    ScalarType error;
};

SurfaceData test( int level, grid::shell::ShellBoundaryFlag boundary_flag )
{
    constexpr ScalarType r_inner = 0.5;
    constexpr ScalarType r_outer = 1.0;

    const auto domain =
        DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_inner, r_outer );

    auto ownership_mask_data = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data  = grid::shell::setup_boundary_mask_data( domain );

    VectorW1Scalar< ScalarType > ones( "ones", domain, ownership_mask_data );
    VectorW1Scalar< ScalarType > dst( "dst", domain, ownership_mask_data );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    const auto r_boundary = boundary_flag == grid::shell::ShellBoundaryFlag::CMB ? r_inner : r_outer;

    using BoundaryMass = fe::wedge::operators::shell::BoundaryMass< ScalarType >;

    BoundaryMass M( domain, coords_shell, coords_radii, boundary_mask_data, boundary_flag );

    assign( ones, 1.0 );

    linalg::apply( M, ones, dst );

    const auto analytical_surface = 4.0 * Kokkos::numbers::pi * r_boundary * r_boundary;

    const auto surface = kernels::common::masked_sum(
        dst.grid_data(), ownership_mask_data, boundary_mask_data, grid::NodeOwnershipFlag::OWNED, boundary_flag );

    const auto error = std::abs( surface - analytical_surface );

    return SurfaceData{ .surface = surface, .surface_analytical = analytical_surface, .error = error };
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    for ( auto boundary_flag : { grid::shell::ShellBoundaryFlag::CMB, grid::shell::ShellBoundaryFlag::SURFACE } )
    {
        double prev_error = 1.0;
        for ( int level = 0; level < 8; ++level )
        {
            auto       data  = test( level, boundary_flag );
            const auto order = prev_error / data.error;
            if ( level > 0 )
            {
                std::cout << "level = " << level << " surface computed = " << data.surface
                          << ", analytical = " << data.surface_analytical << ", error = " << data.error
                          << ", order = " << order << std::endl;
                if ( order < 3.2 )
                {
                    return EXIT_FAILURE;
                }
            }
            prev_error = data.error;
        }
    }

    return 0;
}