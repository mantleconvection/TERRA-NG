#pragma once

#include "parameters.hpp"

#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "mpi/mpi.hpp"
#include "util/logging.hpp"

namespace terra::mantlecirculation {

/// @brief Compute ∫_Γ ∇T · n̂ dΓ on the surface or CMB boundary.
inline ScalarType compute_boundary_heat_flux_integral(
    const grid::shell::DistributedDomain&                domain,
    const grid::Grid4DDataScalar< ScalarType >&          T_grid,
    const grid::Grid3DDataVec< ScalarType, 3 >&          coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&          coords_radii,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& ownership_mask,
    const bool                                           at_surface )
{
    using namespace fe::wedge;

    const int num_subdomains = domain.subdomains().size();
    const int nx = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int nr = domain.domain_info().subdomain_num_nodes_radially();

    const int r_cell = at_surface ? ( nr - 2 ) : 0;
    const int r_boundary_node = at_surface ? ( nr - 1 ) : 0;
    const auto expected_flag = at_surface ? grid::shell::ShellBoundaryFlag::SURFACE : grid::shell::ShellBoundaryFlag::CMB;
    const ScalarType zeta_boundary = at_surface ? ScalarType( 1 ) : ScalarType( -1 );
    const ScalarType normal_sign = at_surface ? ScalarType( 1 ) : ScalarType( -1 );

    ScalarType local_integral = 0;

    Kokkos::parallel_reduce(
        "nusselt_surface_integral",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_subdomains, nx - 1, nx - 1 } ),
        KOKKOS_LAMBDA( const int sd, const int x_cell, const int y_cell, ScalarType& sum ) {
            // Skip subdomains that are not at the actual boundary (radial subdomain decomposition).
            if ( boundary_mask( sd, x_cell, y_cell, r_boundary_node ) != expected_flag )
                return;
            // Skip cells whose anchor node is not owned to avoid double-counting at lateral subdomain boundaries.
            if ( ownership_mask( sd, x_cell, y_cell, r_cell ) != grid::NodeOwnershipFlag::OWNED )
                return;
            constexpr int nqp = quadrature::quad_felippa_3x2_num_quad_points;
            dense::Vec< ScalarType, 3 > quad_points[nqp];
            ScalarType                  quad_weights[nqp];
            quadrature::quad_felippa_3x2_quad_points( quad_points );
            quadrature::quad_felippa_3x2_quad_weights( quad_weights );

            dense::Vec< ScalarType, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, coords_shell, sd, x_cell, y_cell );

            const ScalarType r_1 = coords_radii( sd, r_cell );
            const ScalarType r_2 = coords_radii( sd, r_cell + 1 );

            dense::Vec< ScalarType, 6 > local_T[num_wedges_per_hex_cell] = {};
            extract_local_wedge_scalar_coefficients( local_T, sd, x_cell, y_cell, r_cell, T_grid );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < nqp; ++q )
                {
                    dense::Vec< ScalarType, 3 > qp = quad_points[q];
                    qp( 2 ) = zeta_boundary;

                    const auto J       = jac( wedge_phy_surf[wedge], r_1, r_2, qp );
                    const auto det     = J.det();
                    const auto abs_det = Kokkos::abs( det );

                    if ( abs_det < ScalarType( 1e-20 ) )
                        continue;

                    const auto J_inv_T = J.inv().transposed();

                    dense::Vec< ScalarType, 3 > grad_T_phys{ 0, 0, 0 };
                    for ( int i = 0; i < num_nodes_per_wedge; ++i )
                    {
                        const auto grad_phi_ref = grad_shape< ScalarType >( i, qp );
                        grad_T_phys = grad_T_phys + ( J_inv_T * grad_phi_ref ) * local_T[wedge]( i );
                    }

                    const auto x_phys = forward_map(
                        wedge_phy_surf[wedge][0],
                        wedge_phy_surf[wedge][1],
                        wedge_phy_surf[wedge][2],
                        r_1, r_2,
                        qp( 0 ), qp( 1 ), qp( 2 ) );
                    const auto r_hat = x_phys.normalized();

                    const ScalarType grad_T_dot_n = normal_sign * grad_T_phys.dot( r_hat );

                    const ScalarType radial_scale = ( r_2 - r_1 ) / ScalarType( 2 );
                    // The 3×2 tensor-product quadrature has 2 radial points that collapse
                    // to the same (ξ,η) when ζ is fixed to the boundary, so divide by 2.
                    const ScalarType dA = abs_det * quad_weights[q] / radial_scale / ScalarType( 2 );

                    sum += grad_T_dot_n * dA;
                }
            }
        },
        Kokkos::Sum< ScalarType >( local_integral ) );

    Kokkos::fence();

    ScalarType global_integral = 0;
    MPI_Allreduce( &local_integral, &global_integral, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return global_integral;
}

/// @brief Compute the Nusselt number at the top (surface) or bottom (CMB) boundary.
///
/// Nu = ∫_Γ ∇T · n̂ dΓ  /  ∫_Γ ∇T_ref · n̂ dΓ
///
/// The denominator is computed numerically from the reference conductive profile T_ref,
/// matching the approach used in HyTeG (Ilangovan et al.).
///
/// @param at_surface  If true, compute Nu at the outer surface; if false, at the CMB.
inline ScalarType compute_nusselt(
    const grid::shell::DistributedDomain&       domain,
    const linalg::VectorQ1Scalar< ScalarType >& T,
    const linalg::VectorQ1Scalar< ScalarType >& T_ref,
    const grid::Grid3DDataVec< ScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&  coords_radii,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& ownership_mask,
    const bool                                   at_surface )
{
    const ScalarType numerator   = compute_boundary_heat_flux_integral( domain, T.grid_data(), coords_shell, coords_radii, boundary_mask, ownership_mask, at_surface );
    const ScalarType denominator = compute_boundary_heat_flux_integral( domain, T_ref.grid_data(), coords_shell, coords_radii, boundary_mask, ownership_mask, at_surface );

    return Kokkos::abs( numerator ) / Kokkos::abs( denominator );
}

/// @brief Volume-averaged root-mean-square velocity over the shell.
///
/// V_rms = sqrt( ∫_Ω |u|² dV / ∫_Ω dV )
///
/// Integrated with the Felippa 3×2 wedge quadrature on each hex cell.  Cells
/// are not shared between ranks (only nodes are), so each rank's contribution
/// is a partition of the global volume integral; the two partial sums are
/// MPI_Allreduce'd before the final ratio.
inline ScalarType compute_v_rms(
    const grid::shell::DistributedDomain&              domain,
    const linalg::VectorQ1Vec< ScalarType, 3 >&        velocity,
    const grid::Grid3DDataVec< ScalarType, 3 >&        coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&        coords_radii )
{
    using namespace fe::wedge;

    const auto u_data   = velocity.grid_data();
    const auto grid_lat = coords_shell;
    const auto radii_v  = coords_radii;

    constexpr int num_q = quadrature::quad_felippa_3x2_num_quad_points;

    ScalarType sum_u2_dV = 0;
    ScalarType sum_dV    = 0;

    Kokkos::parallel_reduce(
        "v_rms_volume_int",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int id, int xc, int yc, int rc, ScalarType& acc_u2, ScalarType& acc_V ) {
            dense::Vec< ScalarType, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_lat, id, xc, yc );

            const ScalarType r_1 = radii_v( id, rc );
            const ScalarType r_2 = radii_v( id, rc + 1 );

            dense::Vec< ScalarType, 3 > qp[num_q];
            ScalarType                  qw[num_q];
            quadrature::quad_felippa_3x2_quad_points( qp );
            quadrature::quad_felippa_3x2_quad_weights( qw );

            dense::Vec< ScalarType, num_nodes_per_wedge > u_w[num_wedges_per_hex_cell][3];
            for ( int d = 0; d < 3; ++d )
            {
                dense::Vec< ScalarType, num_nodes_per_wedge > comp[num_wedges_per_hex_cell];
                extract_local_wedge_vector_coefficients( comp, id, xc, yc, rc, d, u_data );
                u_w[0][d] = comp[0];
                u_w[1][d] = comp[1];
            }

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < num_q; ++q )
                {
                    const dense::Mat< ScalarType, 3, 3 > J =
                        jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                    const ScalarType abs_det = Kokkos::abs( J.det() );
                    if ( abs_det < ScalarType( 1e-30 ) )
                    {
                        continue;
                    }

                    ScalarType uq[3] = { 0, 0, 0 };
                    for ( int j = 0; j < num_nodes_per_wedge; ++j )
                    {
                        const ScalarType N_j = shape( j, qp[q] );
                        uq[0] += N_j * u_w[wedge][0]( j );
                        uq[1] += N_j * u_w[wedge][1]( j );
                        uq[2] += N_j * u_w[wedge][2]( j );
                    }
                    const ScalarType u_mag2 = uq[0]*uq[0] + uq[1]*uq[1] + uq[2]*uq[2];

                    acc_u2 += qw[q] * abs_det * u_mag2;
                    acc_V  += qw[q] * abs_det;
                }
            }
        },
        sum_u2_dV,
        sum_dV );
    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &sum_u2_dV, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &sum_dV,    1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return ( sum_dV > ScalarType( 0 ) ) ? Kokkos::sqrt( sum_u2_dV / sum_dV ) : ScalarType( 0 );
}

} // namespace terra::mantlecirculation
