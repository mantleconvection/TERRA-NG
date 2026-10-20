/// @file  test_freeslip_normals.cpp
/// @brief Compares the two candidate free-slip normals on the CMB and surface and writes them to XDMF:
///
///          n_rhat = r̂                  (spherical outward normal, the operators' default)
///          n_BT1  = g / |g|,  g = Bᵀ·1  (the normal implied by the discrete gradient)
///
///        Bᵀ is GradientKerngen with Neumann BCs, applied to a constant (coarse) pressure; it is
///        nonzero only at boundary nodes, where g_i = -∫ φ_i n ds. Both normals are oriented like r̂.
///        Free-slip projected with n_BT1 makes constant pressure an exact null mode (P·Bᵀ·1 = 0);
///        with n_rhat it is only approximately one.
///
///        Output (directory test_freeslip_normals_output/): vector fields n_rhat, n_BT1, scalar field
///        angle_deg (angle between them), all zero away from the two boundaries. Max/mean angle per
///        boundary are printed. Usage: test_freeslip_normals [level=5]

#include <mpi.h>

#include <cstdlib>
#include <iostream>

#include "fe/wedge/operators/shell/gradient_kerngen.hpp"
#include "linalg/vector_w1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"

using namespace terra;

using grid::shell::BoundaryConditions;
using grid::shell::DistributedDomain;
using grid::shell::ShellBoundaryFlag;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const int    level_fine = ( argc > 1 ) ? std::atoi( argv[1] ) : 5;
    const double r_min = 0.5, r_max = 1.0;

    auto domain_fine   = DistributedDomain::create_uniform( level_fine, level_fine, r_min, r_max, 0, 0 );
    auto domain_coarse = DistributedDomain::create_uniform( level_fine - 1, level_fine - 1, r_min, r_max, 0, 0 );

    const auto coords        = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain_fine );
    const auto radii         = grid::shell::subdomain_shell_radii< double >( domain_fine );
    const auto mask_fine     = grid::setup_node_ownership_mask_data( domain_fine );
    const auto mask_coarse   = grid::setup_node_ownership_mask_data( domain_coarse );
    const auto boundary_mask = grid::shell::setup_boundary_mask_data( domain_fine );

    // g = Bᵀ·1 (unconstrained gradient of a constant pressure).
    BoundaryConditions bcs_neumann = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };
    fe::wedge::operators::shell::GradientKerngen< double > BT(
        domain_fine, domain_coarse, coords, radii, boundary_mask, bcs_neumann );

    VectorW1Scalar< double > one( "one", domain_coarse, mask_coarse );
    VectorW1Vec< double, 3 > g( "BT1", domain_fine, mask_fine );
    linalg::assign( one, 1.0 );
    linalg::apply( BT, one, g );

    VectorW1Vec< double, 3 > n_rhat( "n_rhat", domain_fine, mask_fine );
    VectorW1Vec< double, 3 > n_bt1( "n_BT1", domain_fine, mask_fine );
    VectorW1Scalar< double > angle( "angle_deg", domain_fine, mask_fine );
    linalg::assign( n_rhat, 0.0 );
    linalg::assign( n_bt1, 0.0 );
    linalg::assign( angle, 0.0 );

    const auto gd = g.grid_data();
    const auto nr = n_rhat.grid_data();
    const auto nb = n_bt1.grid_data();
    const auto an = angle.grid_data();

    Kokkos::parallel_for(
        "compare_normals",
        grid::shell::local_domain_md_range_policy_nodes( domain_fine ),
        KOKKOS_LAMBDA( const int s, const int x, const int y, const int r ) {
            const auto flags = boundary_mask( s, x, y, r );
            if ( !util::has_flag( flags, CMB ) && !util::has_flag( flags, SURFACE ) )
                return;

            const double rx = coords( s, x, y, 0 ), ry = coords( s, x, y, 1 ), rz = coords( s, x, y, 2 );
            const double gx = gd( s, x, y, r, 0 ), gy = gd( s, x, y, r, 1 ), gz = gd( s, x, y, r, 2 );
            const double sign = ( gx * rx + gy * ry + gz * rz < 0.0 ) ? -1.0 : 1.0;
            const double inv  = sign / Kokkos::sqrt( gx * gx + gy * gy + gz * gz );
            const double bx = inv * gx, by = inv * gy, bz = inv * gz;

            nr( s, x, y, r, 0 ) = rx;
            nr( s, x, y, r, 1 ) = ry;
            nr( s, x, y, r, 2 ) = rz;
            nb( s, x, y, r, 0 ) = bx;
            nb( s, x, y, r, 1 ) = by;
            nb( s, x, y, r, 2 ) = bz;

            // atan2(|r̂ × n|, r̂·n): accurate for tiny angles, unlike acos.
            const double cx = ry * bz - rz * by, cy = rz * bx - rx * bz, cz = rx * by - ry * bx;
            an( s, x, y, r ) =
                Kokkos::atan2( Kokkos::sqrt( cx * cx + cy * cy + cz * cz ), rx * bx + ry * by + rz * bz ) * 180.0 /
                Kokkos::numbers::pi;
        } );
    Kokkos::fence();

    // Max / mean angle per boundary (owned nodes only).
    bool ok = true;
    for ( const auto boundary : { CMB, SURFACE } )
    {
        double local_max = 0.0, local_sum = 0.0, local_cnt = 0.0;
        Kokkos::parallel_reduce(
            "angle_stats",
            grid::shell::local_domain_md_range_policy_nodes( domain_fine ),
            KOKKOS_LAMBDA( const int s, const int x, const int y, const int r, double& mx, double& sum, double& cnt ) {
                if ( !util::has_flag( boundary_mask( s, x, y, r ), boundary ) ||
                     !util::has_flag( mask_fine( s, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                    return;
                mx = Kokkos::max( mx, an( s, x, y, r ) );
                sum += an( s, x, y, r );
                cnt += 1.0;
            },
            Kokkos::Max< double >( local_max ),
            local_sum,
            local_cnt );

        double gmax, gsum, gcnt;
        MPI_Allreduce( &local_max, &gmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
        MPI_Allreduce( &local_sum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
        MPI_Allreduce( &local_cnt, &gcnt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

        util::logroot << ( boundary == CMB ? "CMB    " : "SURFACE" ) << ": nodes " << static_cast< long >( gcnt )
                      << ", angle(r̂, Bᵀ·1) max " << gmax << " deg, mean " << gsum / gcnt << " deg" << std::endl;

        // Sanity only: the two normals must be close, but (generally) not identical.
        ok = ok && std::isfinite( gmax ) && gmax < 1.0;
    }

    util::prepare_empty_directory( "test_freeslip_normals_output" );
    io::XDMFOutput xdmf( "test_freeslip_normals_output", domain_fine, coords, radii );
    xdmf.add( n_rhat.grid_data() );
    xdmf.add( n_bt1.grid_data() );
    xdmf.add( angle.grid_data() );
    xdmf.write( 0 );
    util::logroot << "Wrote test_freeslip_normals_output/step_0.xmf (level " << level_fine << ")" << std::endl;

    return ok ? 0 : 1;
}
