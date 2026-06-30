#pragma once

#include <memory>

#include "fe/wedge/operators/shell/kmass.hpp"
#include "grid/bit_masks.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::mantlecirculation {

/// @brief Velocity-space lumped diagonal of the sqrt(eta)-weighted mass matrix
///        C_w = M_u_lumped(sqrt(eta)) used by the w-BFBT Schur preconditioner.
///
/// In the Rudi/Stadler/Ghattas 2017 w-BFBT formula with w_l = w_r = sqrt(eta),
/// the same diagonal serves as both C_w and D_w; the preconditioner needs only
/// its inverse to scale velocity vectors element-wise between B^T/A and A/B.
///
/// Implementation note (v1): we reuse the scalar-Q1 KMass on the velocity-level
/// domain with k = sqrt(eta) and `lumped_diagonal=true`.  Applying it to a
/// 1-vector yields the row-sums (∫ sqrt(eta) phi_i dx, exactly the lumped
/// diagonal entries).  Inverting and replicating the result across the 3
/// velocity components gives a `VectorQ1Vec` suitable for
/// `linalg::scale_in_place` against any velocity vector.
template < typename ScalarType, int VecDim = 3 >
class WBFBTWeightedLumpedVelocityMass
{
  public:
    using ScalarField = linalg::VectorQ1Scalar< ScalarType >;
    using VectorField = linalg::VectorQ1Vec< ScalarType, VecDim >;
    using KMass       = fe::wedge::operators::shell::KMass< ScalarType >;

    /// @param velocity_domain          Velocity-level distributed domain.
    /// @param velocity_coords_shell    Velocity-level shell coords (unit-sphere positions per node).
    /// @param velocity_coords_radii    Velocity-level radial coords.
    /// @param velocity_ownership_mask  Velocity-level node-ownership mask.
    WBFBTWeightedLumpedVelocityMass(
        const grid::shell::DistributedDomain&                    velocity_domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&              velocity_coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&              velocity_coords_radii,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& velocity_ownership_mask )
    : domain_( velocity_domain )
    , coords_shell_( velocity_coords_shell )
    , coords_radii_( velocity_coords_radii )
    , ownership_mask_( velocity_ownership_mask )
    , sqrt_eta_( "wbfbt_sqrt_eta_velocity", velocity_domain, velocity_ownership_mask )
    , c_w_inv_diag_scalar_( "wbfbt_c_w_inv_diag_scalar", velocity_domain, velocity_ownership_mask )
    , c_w_inv_diag_velocity_( "wbfbt_c_w_inv_diag_velocity", velocity_domain, velocity_ownership_mask )
    {
        linalg::assign( sqrt_eta_, ScalarType( 1 ) );

        kmass_ = std::make_unique< KMass >(
            domain_,
            coords_shell_,
            coords_radii_,
            sqrt_eta_.grid_data(),
            /*diagonal=*/ false,
            /*lumped_diagonal=*/ true );

        recompute_inv_diag_();
    }

    /// Update the internal sqrt(eta) field, then recompute the lumped inverse
    /// diagonal and re-replicate into the per-component velocity vector.
    void refresh( const ScalarField& sqrt_eta_velocity_finest )
    {
        linalg::assign( sqrt_eta_, sqrt_eta_velocity_finest );
        recompute_inv_diag_();
    }

    /// Velocity-vector inverse diagonal for element-wise scaling.
    /// Each of the VecDim components carries the same scalar 1/(M_u_lumped_w)_i,
    /// so applying `linalg::scale_in_place(v, inv_diag_velocity())` to a
    /// velocity vector `v` is mathematically C_w^-1 v.
    const VectorField& inv_diag_velocity() const { return c_w_inv_diag_velocity_; }

  private:
    void recompute_inv_diag_()
    {
        // 1) Apply lumped KMass with k = sqrt(eta) to a ones-vector → row sums
        //    of M_u_w on the velocity mesh.  These are the lumped diagonal
        //    entries C_w_ii.
        ScalarField ones( "wbfbt_ones_velocity", domain_, ownership_mask_ );
        linalg::assign( ones, ScalarType( 1 ) );
        linalg::apply( *kmass_, ones, c_w_inv_diag_scalar_ );

        // 2) Invert in place → C_w_inv_diag.
        linalg::invert_entries( c_w_inv_diag_scalar_ );

        // 3) Replicate the scalar inverse-diagonal into each velocity component.
        const auto& src = c_w_inv_diag_scalar_.grid_data();
        auto&       dst = c_w_inv_diag_velocity_.grid_data();
        for ( int d = 0; d < VecDim; ++d )
        {
            Kokkos::deep_copy( dst.comp_[d], src );
        }
    }

    grid::shell::DistributedDomain                    domain_;
    grid::Grid3DDataVec< ScalarType, 3 >              coords_shell_;
    grid::Grid2DDataScalar< ScalarType >              coords_radii_;
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > ownership_mask_;

    ScalarField sqrt_eta_;
    ScalarField c_w_inv_diag_scalar_;
    VectorField c_w_inv_diag_velocity_;

    std::unique_ptr< KMass > kmass_;
};

} // namespace terra::mantlecirculation
