
#pragma once
#include "communication/shell/fv_communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/shell/bit_masks.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fv::hex {

// ============================================================================
// Dirichlet boundary conditions for radial shell boundaries
// ============================================================================

/// @brief Constant Dirichlet boundary conditions for the inner (CMB) and outer (surface)
///        radial boundaries of the spherical shell.
///
/// Used by `apply_dirichlet_bcs` to enforce prescribed temperatures after each time step.
/// The BCs are applied as a strong (pointwise) enforcement — boundary cells are overwritten
/// with the prescribed value at the end of each explicit or implicit step.
///
/// @tparam ScalarT  Floating-point scalar type.
template < typename ScalarT >
struct DirichletBCs
{
    ScalarT T_cmb         = ScalarT( 0 ); ///< Prescribed temperature at the core-mantle boundary.
    ScalarT T_surface     = ScalarT( 0 ); ///< Prescribed temperature at the outer surface.
    bool    apply_cmb     = false;        ///< Enforce CMB Dirichlet BC when true.
    bool    apply_surface = false;        ///< Enforce surface Dirichlet BC when true.
};

/// @brief Strongly enforce Dirichlet boundary conditions on the radial shell boundaries.
///
/// For each FV cell that lies on the CMB (\f$r = 1\f$, innermost real cell of the innermost
/// radial subdomain) or on the outer surface (\f$r = N_r - 1\f$, outermost real cell of the
/// outermost radial subdomain), the scalar field is overwritten with the prescribed value.
///
/// Boundary cells are identified via the `boundary_mask`:
/// - CMB cells: `r_cell == 1` and `boundary_mask(id, 0, 0, 0) == ShellBoundaryFlag::CMB`.
/// - Surface cells: `r_cell == last` and `boundary_mask(id, 0, 0, last_q1) == ShellBoundaryFlag::SURFACE`.
///
/// This strong enforcement is first-order in time (the boundary value is fixed after each step).
/// For cells *adjacent* to the boundary the scheme is spatially consistent: the next time step's
/// stencil will see the correct Dirichlet value at the boundary cell.
///
/// @param T             [in/out] Transported scalar field (boundary cells overwritten).
/// @param boundary_mask Node-based boundary flag array (Q1 layout, CMB at r=0, SURFACE at r=last).
/// @param bcs           Prescribed BC values and flags.
/// @param domain        Distributed domain (used for the loop range policy).
/// @brief Apply Dirichlet BCs directly to a raw FV grid view.
///
/// Sets both the real boundary cell and the adjacent ghost cell outside the physical domain.
/// The ghost cell is never filled by `update_fv_ghost_layers` (no subdomain neighbour exists
/// beyond a physical boundary) so it must be set here to give the correct inflow/diffusion
/// BC when FCT predictor stencils read across the boundary face.
///
/// This overload is the low-level implementation; the `VectorFVScalar` overload delegates to it.
/// It can also be called directly on intermediate FCT buffers (e.g. `T_L`) that are raw views.
template < typename ScalarT >
void apply_dirichlet_bcs(
    grid::Grid4DDataScalar< ScalarT >                               data,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const DirichletBCs< ScalarT >&                                  bcs,
    const grid::shell::DistributedDomain&                           domain )
{
    using Flag = grid::shell::ShellBoundaryFlag;

    const int fv_r_last   = static_cast< int >( data.extent( 3 ) ) - 2;
    const int mask_r_last = static_cast< int >( boundary_mask.extent( 3 ) ) - 1;

    const ScalarT T_cmb   = bcs.T_cmb;
    const ScalarT T_surf  = bcs.T_surface;
    const bool    do_cmb  = bcs.apply_cmb;
    const bool    do_surf = bcs.apply_surface;

    Kokkos::parallel_for(
        "apply_dirichlet_bcs",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        KOKKOS_LAMBDA( const int id, const int x, const int y, const int r ) {
            if ( do_cmb && r == 1 && util::has_flag( boundary_mask( id, 0, 0, 0 ), Flag::CMB ) )
            {
                data( id, x, y, 1 ) = T_cmb;
                data( id, x, y, 0 ) = T_cmb;
            }
            if ( do_surf && r == fv_r_last && util::has_flag( boundary_mask( id, 0, 0, mask_r_last ), Flag::SURFACE ) )
            {
                data( id, x, y, fv_r_last )     = T_surf;
                data( id, x, y, fv_r_last + 1 ) = T_surf;
            }
        } );

    Kokkos::fence();
}

template < typename ScalarT >
void apply_dirichlet_bcs(
    linalg::VectorFVScalar< ScalarT >&                              T,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const DirichletBCs< ScalarT >&                                  bcs,
    const grid::shell::DistributedDomain&                           domain )
{
    apply_dirichlet_bcs( T.grid_data(), boundary_mask, bcs, domain );
}

/// @brief Computes cell centers and writes to a vector valued finite volume function.
///
///
/// This function computes for every finite volume cell \f$K\f$ the value
/// \f[
///     u_K = \frac{1}{|K|} \int_K x \ dx
/// \f]
/// with
/// \f[
///     |K| = \int_K 1 \ dx
/// \f]
/// and writes \f$u_K\f$ into the respective finite volume cell.
///
///
/// @param dst [out] finite volume function that is being written to
/// @param coords_shell coords of the unit shell
/// @param coords_radii radii of all shells
template < typename ScalarType, typename GridScalarType >
void compute_cell_centers(
    linalg::VectorFVVec< ScalarType, 3 >&           dst,
    const grid::Grid3DDataVec< GridScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< GridScalarType >& coords_radii )
{
    grid::Grid4DDataVec< ScalarType, 3 > fv_grid = dst.grid_data();
    Kokkos::parallel_for(
        "l2_project_into_fv_function",
        Kokkos::MDRangePolicy(
            { 0, 1, 1, 1 },
            { fv_grid.extent( 0 ), fv_grid.extent( 1 ) - 1, fv_grid.extent( 2 ) - 1, fv_grid.extent( 3 ) - 1 } ),
        KOKKOS_LAMBDA(
            const int local_subdomain_id, const int hex_cell_x, const int hex_cell_y, const int hex_cell_r ) {
            constexpr auto              num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;
            dense::Vec< ScalarType, 3 > quad_points[num_quad_points];
            ScalarType                  quad_weights[num_quad_points];
            fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
            fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

            ScalarType                  volume   = 0.0;
            dense::Vec< ScalarType, 3 > integral = {};

            dense::Vec< ScalarType, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                                      [fe::wedge::num_nodes_per_wedge_surface] = {};
            fe::wedge::wedge_surface_physical_coords(
                wedge_phy_surf, coords_shell, local_subdomain_id, hex_cell_x - 1, hex_cell_y - 1 );

            const auto r_1 = coords_radii( local_subdomain_id, hex_cell_r - 1 );
            const auto r_2 = coords_radii( local_subdomain_id, hex_cell_r );

            for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < num_quad_points; ++q )
                {
                    const auto J       = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                    const auto det     = J.det();
                    const auto abs_det = Kokkos::abs( det );
                    const auto Jw      = abs_det * quad_weights[q];
                    volume             = volume + Jw;

                    const auto x = fe::wedge::forward_map(
                        wedge_phy_surf[wedge][0],
                        wedge_phy_surf[wedge][1],
                        wedge_phy_surf[wedge][2],
                        r_1,
                        r_2,
                        quad_points[q]( 0 ),
                        quad_points[q]( 1 ),
                        quad_points[q]( 2 ) );

                    integral = integral + x * Jw;
                }
            }

            for ( int d = 0; d < 3; ++d )
            {
                fv_grid( local_subdomain_id, hex_cell_x, hex_cell_y, hex_cell_r, d ) = integral( d ) / volume;
            }
        } );
}

/// @brief Computes cell center positions once and populates ghost layers via MPI communication.
///
/// This is the recommended one-time initialization for cell centers at application startup.
/// After this call, `dst` contains valid cell centers in both real cells and all ghost layers,
/// so kernels can look up neighbour cell centers without recomputing geometry.
///
/// @param dst          [out] FV vector field receiving cell centers (real cells + ghost layers).
/// @param domain       Distributed domain (used for ghost layer communication).
/// @param coords_shell Lateral node coordinates of the unit sphere surface.
/// @param coords_radii Radial shell radii.
template < typename ScalarType, typename GridScalarType >
void initialize_cell_centers(
    linalg::VectorFVVec< ScalarType, 3 >&           dst,
    const grid::shell::DistributedDomain&           domain,
    const grid::Grid3DDataVec< GridScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< GridScalarType >& coords_radii )
{
    compute_cell_centers( dst, coords_shell, coords_radii );
    Kokkos::fence();
    communication::shell::update_fv_ghost_layers( domain, dst.grid_data() );
}

} // namespace terra::fv::hex