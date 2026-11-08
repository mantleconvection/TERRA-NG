
#pragma once

#include "linalg/vector_w1.hpp"
#include "linalg/vector_w1_iso_w2_w1.hpp"
#include "terra/linalg/operator.hpp"

namespace terra::fe {

/// @brief Helper function to modify the right-hand side vector accordingly for strong Dirichlet boundary condition
/// enforcement.
///
/// \note The framework documentation features [a detailed description](#boundary-conditions)
/// of the strong imposition of Dirichlet boundary conditions.
///
/// @param A_neumann the "Neumann" operator (without any modification at the boundary)
/// @param A_neumann_diag the diagonal of the "Neumann" operator
/// @param g a coefficient vector with interpolated Dirichlet boundary conditions at the respective boundary nodes, zero
/// elsewhere
/// @param tmp a temporary vector
/// @param b [in/out] RHS coefficient vector before boundary elimination (but including forcing etc.) - will be modified
/// in this function to impose the Dirichlet BCs (after the function returns, this is what is called \f$b_\mathrm{elim}\f$
/// in the documentation)
/// @param mask_data the boundary mask data
/// @param dirichlet_boundary_mask the flag that indicates where to apply the conditions
template < typename ScalarType, linalg::OperatorLike OperatorType, typename FlagType >
void strong_algebraic_dirichlet_enforcement_poisson_like(
    OperatorType&                               A_neumann,
    OperatorType&                               A_neumann_diag,
    const linalg::VectorW1Scalar< ScalarType >& g,
    linalg::VectorW1Scalar< ScalarType >&       tmp,
    linalg::VectorW1Scalar< ScalarType >&       b,
    const grid::Grid4DDataScalar< FlagType >&   mask_data,
    const FlagType&                             dirichlet_boundary_mask )
{
    // g_A <- A * g
    linalg::apply( A_neumann, g, tmp );

    // b_elim <- b - g_A
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );

    // g_D <- diag(A) * g
    linalg::apply( A_neumann_diag, g, tmp );

    // b_elim <- g_D on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old( b.grid_data(), tmp.grid_data(), mask_data, dirichlet_boundary_mask );
}

template < typename ScalarType, linalg::OperatorLike OperatorType, typename FlagType >
void strong_algebraic_dirichlet_enforcement_vectorlaplace_like(
    OperatorType&                               A_neumann,
    OperatorType&                               A_neumann_diag,
    const linalg::VectorW1Vec< ScalarType >& g,
    linalg::VectorW1Vec< ScalarType >&       tmp,
    linalg::VectorW1Vec< ScalarType >&       b,
    const grid::Grid4DDataScalar< FlagType >&   mask_data,
    const FlagType&                             dirichlet_boundary_mask )
{
    // g_A <- A * g
    linalg::apply( A_neumann, g, tmp );

    // b_elim <- b - g_A
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );

    // g_D <- diag(A) * g
    linalg::apply( A_neumann_diag, g, tmp );

    // b_elim <- g_D on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old( b.grid_data(), tmp.grid_data(), mask_data, dirichlet_boundary_mask );
}

/// @brief Same as strong_algebraic_dirichlet_enforcement_poisson_like() for homogenous boundary conditions (\f$ g = 0 \f$).
///
/// Does not require most of the steps since \f$ g = g_A = g_D = 0 \f$.
/// Still requires solving \f$ A_\mathrm{elim} x = b_\mathrm{elim} \f$after this.
template < typename ScalarType, util::FlagLike FlagType >
void strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like(
    linalg::VectorW1Scalar< ScalarType >&     b,
    const grid::Grid4DDataScalar< FlagType >& mask_data,
    const FlagType&                           dirichlet_boundary_mask )
{
    // b_elim <- 0 on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old( b.grid_data(), 0.0, mask_data, dirichlet_boundary_mask );
}

/// @brief Same as strong_algebraic_dirichlet_enforcement_poisson_like() for Stokes-like systems (with strong
/// enforcement of velocity boundary conditions).
template < typename ScalarType, linalg::OperatorLike OperatorType, util::FlagLike FlagType >
void strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
    OperatorType&                                K_neumann,
    OperatorType&                                K_neumann_diag,
    const linalg::VectorW1IsoW2W1< ScalarType >& g,
    linalg::VectorW1IsoW2W1< ScalarType >&       tmp,
    linalg::VectorW1IsoW2W1< ScalarType >&       b,
    const grid::Grid4DDataScalar< FlagType >&    mask_data,
    const FlagType&                              dirichlet_boundary_mask )
{
    // g_A <- A * g
    linalg::apply( K_neumann, g, tmp );

    // b_elim <- b - g_A
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );

    // g_D <- diag(A) * g
    linalg::apply( K_neumann_diag, g, tmp );

    // b_elim <- g_D on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old(
        b.block_1().grid_data(), tmp.block_1().grid_data(), mask_data, dirichlet_boundary_mask );
}

/// @brief Same as strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like() for Stokes-like systems
/// (with strong enforcement of zero velocity boundary conditions).
template < typename ScalarType, util::FlagLike FlagType >
void strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
    linalg::VectorW1IsoW2W1< ScalarType >&    b,
    const grid::Grid4DDataScalar< FlagType >& mask_data,
    const FlagType&                           dirichlet_boundary_mask )
{
    // b_elim <- g_D on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old(
        b.block_1().grid_data(), ScalarType( 0 ), mask_data, dirichlet_boundary_mask );
}

} // namespace terra::fe