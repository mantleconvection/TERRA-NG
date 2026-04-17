#pragma once

#include "parameters.hpp"

#include "fe/wedge/integrands.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "mpi/mpi.hpp"
#include "util/logging.hpp"

namespace terra::mantlecirculation {

/// @brief Compute the Nusselt number from the FV temperature field.
///
/// Uses the boundary-cell values and the Dirichlet BC to compute the radial gradient
/// at the boundary face.  The spherical-shell average gradient is then normalized by
/// the conductive reference.
///
/// Nu = < ∂T/∂r >_surface / < ∂T_ref/∂r >_surface
///
/// where the average is weighted by the surface area element 4π r².
inline ScalarType compute_nusselt_fv(
    const grid::shell::DistributedDomain&       domain,
    const linalg::VectorFVScalar< ScalarType >& T_fct,
    const ScalarType                            T_bc_surface,
    const ScalarType                            T_bc_cmb,
    const ScalarType                            r_min,
    const ScalarType                            r_max,
    const bool                                  at_surface )
{
    // The FV grid has ghost layers: indices run from 0..n+1 in each direction.
    // Interior cells (no ghost) are at indices 1..n.
    // Boundary cells at the surface are at radial index n (the outermost interior layer).
    // Boundary cells at the CMB are at radial index 1 (the innermost interior layer).

    const auto fv_grid = T_fct.grid_data();
    const int nsd = fv_grid.extent( 0 );
    const int nx_fv = fv_grid.extent( 1 );
    const int ny_fv = fv_grid.extent( 2 );
    const int nr_fv = fv_grid.extent( 3 );

    // The FV Dirichlet BC sets the outermost interior cell to T_bc.
    // The actual evolved temperature is in the cell BELOW the boundary cell.
    // For surface: boundary cell = nr_fv-2 (set to T_bc), first free cell = nr_fv-3.
    // For CMB: boundary cell = 1 (set to T_bc), first free cell = 2.
    const int r_cell_fv = at_surface ? ( nr_fv - 3 ) : 2;

    // The face radius is exactly at the boundary.
    const ScalarType r_face = at_surface ? r_max : r_min;

    // For a uniform radial mesh with nr_interior cells, the cell width is (r_max - r_min) / nr_interior.
    const int nr_interior = nr_fv - 2; // subtract 2 ghost layers
    const ScalarType dr_cell = ( r_max - r_min ) / nr_interior;
    // The cell center of the first free cell (the one we're reading).
    const ScalarType r_center = at_surface ? ( r_face - ScalarType( 1.5 ) * dr_cell ) : ( r_face + ScalarType( 1.5 ) * dr_cell );
    const ScalarType T_bc = at_surface ? T_bc_surface : T_bc_cmb;
    const ScalarType normal_sign = at_surface ? ScalarType( 1 ) : ScalarType( -1 );

    // Compute the area-weighted average of ∂T/∂r at the boundary.
    // Since all lateral cells have approximately equal area on the sphere,
    // a simple average over all boundary cells gives the shell-averaged gradient.

    ScalarType local_sum_gradT = 0;
    int local_count = 0;

    Kokkos::parallel_reduce(
        "nusselt_fv_surface",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 1, 1 }, { nsd, nx_fv - 1, ny_fv - 1 } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, ScalarType& sum ) {
            const ScalarType T_cell = fv_grid( sd, x, y, r_cell_fv );
            const ScalarType dTdr = normal_sign * ( T_bc - T_cell ) / ( r_face - r_center );
            sum += dTdr;
        },
        Kokkos::Sum< ScalarType >( local_sum_gradT ) );
    Kokkos::fence();

    Kokkos::parallel_reduce(
        "nusselt_fv_count",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 1, 1 }, { nsd, nx_fv - 1, ny_fv - 1 } ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, int& cnt ) {
            cnt += 1;
        },
        local_count );
    Kokkos::fence();

    ScalarType global_sum_gradT = 0;
    int global_count = 0;
    MPI_Allreduce( &local_sum_gradT, &global_sum_gradT, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( &local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    const ScalarType avg_gradT = global_sum_gradT / global_count;

    // Conductive profile gradient at the boundary.
    // T_ref(r) = (r_min * r_max / r - r_min) / (r_max - r_min)
    // ∂T_ref/∂r = -r_min * r_max / (r² * (r_max - r_min))
    const ScalarType D = r_max - r_min;
    const ScalarType avg_gradTref = -r_min * r_max / ( r_face * r_face * D );
    const ScalarType avg_gradTref_outward = normal_sign * avg_gradTref;

    return Kokkos::abs( avg_gradT ) / Kokkos::abs( avg_gradTref_outward );
}

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

    // Also compute ∫ dΓ for diagnostics (should be 4π*r² ≈ 61.89 at surface).
    // And compute a simple volume-averaged surface gradient for cross-check.
    const ScalarType area = [&]() {
        using namespace fe::wedge;
        const int num_subdomains = domain.subdomains().size();
        const int nx = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nr = domain.domain_info().subdomain_num_nodes_radially();
        const int r_cell = at_surface ? ( nr - 2 ) : 0;
        const int r_boundary_node = at_surface ? ( nr - 1 ) : 0;
        const auto expected_flag = at_surface ? grid::shell::ShellBoundaryFlag::SURFACE : grid::shell::ShellBoundaryFlag::CMB;
        const ScalarType zeta_boundary = at_surface ? ScalarType( 1 ) : ScalarType( -1 );

        ScalarType local_area = 0;
        Kokkos::parallel_reduce(
            "nusselt_surface_area",
            Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { num_subdomains, nx - 1, nx - 1 } ),
            KOKKOS_LAMBDA( const int sd, const int x_cell, const int y_cell, ScalarType& sum ) {
                // Skip subdomains that are not at the actual boundary.
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

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
                {
                    for ( int q = 0; q < nqp; ++q )
                    {
                        dense::Vec< ScalarType, 3 > qp = quad_points[q];
                        qp( 2 ) = zeta_boundary;

                        const auto J       = jac( wedge_phy_surf[wedge], r_1, r_2, qp );
                        const auto abs_det = Kokkos::abs( J.det() );
                        const ScalarType radial_scale = ( r_2 - r_1 ) / ScalarType( 2 );
                        // Divide by 2: same lateral-point doubling as in the flux integral.
                        sum += abs_det * quad_weights[q] / radial_scale / ScalarType( 2 );
                    }
                }
            },
            Kokkos::Sum< ScalarType >( local_area ) );
        Kokkos::fence();
        ScalarType global_area = 0;
        MPI_Allreduce( &local_area, &global_area, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );
        return global_area;
    }();

    const int nx_debug = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int nr_debug = domain.domain_info().subdomain_num_nodes_radially();
    const int r_cell_debug = at_surface ? ( nr_debug - 2 ) : 0;
    const ScalarType r_boundary = at_surface ? domain.domain_info().radii().back() : domain.domain_info().radii().front();
    const ScalarType expected_area = 4.0 * M_PI * r_boundary * r_boundary;

    // Sample some T values at the boundary for debugging.
    auto T_data = T.grid_data();
    auto T_ref_data = T_ref.grid_data();
    auto h_T = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), T_data );
    auto h_Tref = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), T_ref_data );
    const ScalarType T_surf_sample = h_T( 0, nx_debug / 2, nx_debug / 2, nr_debug - 1 );
    const ScalarType T_below_sample = h_T( 0, nx_debug / 2, nx_debug / 2, nr_debug - 2 );
    const ScalarType Tref_surf = h_Tref( 0, nx_debug / 2, nx_debug / 2, nr_debug - 1 );
    const ScalarType Tref_below = h_Tref( 0, nx_debug / 2, nx_debug / 2, nr_debug - 2 );

    util::logroot << "  Nusselt debug: numerator = " << numerator << ", denominator = " << denominator
                  << ", area = " << area << " (expected " << expected_area << ")"
                  << ", nx = " << nx_debug << ", nr = " << nr_debug << ", r_cell = " << r_cell_debug
                  << "\n  T_surf = " << T_surf_sample << ", T_below = " << T_below_sample
                  << ", Tref_surf = " << Tref_surf << ", Tref_below = " << Tref_below
                  << std::endl;

    return Kokkos::abs( numerator ) / Kokkos::abs( denominator );
}

} // namespace terra::mantlecirculation
