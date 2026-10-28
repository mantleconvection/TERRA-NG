
#pragma once

#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
#include "linalg/vector_w1_iso_w2_w1.hpp"

namespace terra::fe {

/// @brief Helper function to modify the right-hand side vector accordingly for strong free-slip boundary condition
/// enforcement.
///
/// \note The framework documentation features [a detailed description](#boundary-conditions)
/// of the strong imposition of free-slip boundary conditions.
///
/// @param b [in/out] RHS coefficient vector before boundary elimination (but including forcing etc.) - will be modified
/// in this function to impose the free-slip BCs (after the function returns, this is what is called \f$b_\mathrm{elim}\f$
/// in the documentation)
/// @param coords_shell the coordinates of the unit shell, obtained e.g. via
/// @ref terra::grid::shell::subdomain_unit_sphere_single_shell_coords
/// @param mask_data the boundary mask data
/// @param freeslip_boundary_mask the flag that indicates where to apply the conditions
template < typename ScalarType, typename ScalarTypeGrid, util::FlagLike FlagType >
void strong_algebraic_freeslip_enforcement_in_place(
    linalg::VectorW1IsoW2W1< ScalarType >&          b,
    const grid::Grid3DDataVec< ScalarTypeGrid, 3 >& coords_shell,
    const grid::Grid4DDataScalar< FlagType >&       mask_data,
    const FlagType&                                 freeslip_boundary_mask )
{
    // b <- R b trafo to n-t space
    linalg::trafo::cartesian_to_normal_tangential_in_place( b.block_1(), coords_shell, mask_data, freeslip_boundary_mask );

    // b <- 0 for normal components at FS boundary
    kernels::common::assign_masked_else_keep_old<ScalarType, 3, FlagType>( b.block_1().grid_data(), 0, mask_data, freeslip_boundary_mask, 0 );

    // b <- R^T b trafo back to carth space
    linalg::trafo::normal_tangential_to_cartesian_in_place( b.block_1(), coords_shell, mask_data, freeslip_boundary_mask );

}

} // namespace terra::fe