#pragma once

#include <memory>

#include "grid/bit_masks.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

#include "wbfbt_pressure_poisson.hpp"

namespace terra::mantlecirculation {

/// @brief Weighted-BFBT Schur complement preconditioner for the saddle-point
///        Stokes system, per Rudi, Stadler, Ghattas (SISC 2017, eq. 7):
///
///   S^{-1} ≈ K_w^{-1} . ( B C_w^{-1} A D_w^{-1} B^T ) . K_w^{-1}
///
/// with K_w = B C_w^{-1} B^T and C_w = D_w = M_u_lumped(sqrt(eta)).
///
/// On a pressure-space input y, the action is:
///   1) z1 := K_w^{-1}  y
///   2) v1 := B^T z1                  (velocity)
///   3) v1 := D_w^{-1} . v1           (elementwise)
///   4) v2 := A v1                    (velocity)
///   5) v2 := C_w^{-1} . v2           (elementwise)
///   6) z2 := B v2                    (pressure)
///   7) x  := K_w^{-1}  z2
///
/// Replaces the existing M_p(1/eta)-based Schur preconditioner
/// (`DiagonalSolver<PressureMass>`) at high viscosity contrast, where the
/// inverse-viscosity-weighted lumped pressure mass becomes spectrally
/// inadequate.
///
/// Templated on the four operator types so it can be reused with any Stokes
/// discretization that exposes the four blocks; mirrors the templating style
/// of `BlockTriangularPreconditioner2x2`.
///
/// `OperatorType` is set to the pressure-mass type only so this class
/// satisfies the `SolverLike` concept expected by
/// `BlockTriangularPreconditioner2x2` for its (2,2)-block preconditioner.
/// The operator argument to `solve_impl` is unused at solve time — the
/// w-BFBT action is independent of it.
template < linalg::OperatorLike ViscousOp,
           linalg::OperatorLike GradientOp,
           linalg::OperatorLike DivergenceOp,
           linalg::OperatorLike PressureMassOp >
class WBFBTSchurPreconditioner
{
  public:
    using OperatorType       = PressureMassOp;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType      = linalg::DstOf< OperatorType >;
    using ScalarType         = typename SolutionVectorType::ScalarType;

    using PressureVector = SolutionVectorType;
    using VelocityVector = linalg::SrcOf< ViscousOp >;

    /// @param viscous                   A (block_11).
    /// @param gradient                  B^T (block_12).
    /// @param divergence                B (block_21).
    /// @param kw_solver                 Solver for K_w = B C_w^{-1} B^T (any
    ///                                  implementation of WBFBTPressurePoissonSolver).
    ///                                  Shared so the orchestrating StokesContext
    ///                                  can drive `refresh()` externally on
    ///                                  viscosity changes.
    /// @param c_w_inv_diag_velocity     Velocity-space lumped inverse diagonal
    ///                                  1/(M_u_lumped(sqrt(eta)))_ii. Stored by
    ///                                  value (shallow handle copy); updates to
    ///                                  the underlying Kokkos buffer are seen
    ///                                  through this copy.
    /// @param velocity_domain           Velocity-level distributed domain.
    /// @param pressure_domain           Pressure-level distributed domain.
    /// @param velocity_ownership_mask   Velocity-level node-ownership mask.
    /// @param pressure_ownership_mask   Pressure-level node-ownership mask.
    WBFBTSchurPreconditioner(
        const ViscousOp&                                                viscous,
        const GradientOp&                                               gradient,
        const DivergenceOp&                                             divergence,
        std::shared_ptr< WBFBTPressurePoissonSolver< ScalarType > >     kw_solver,
        const VelocityVector&                                           c_w_inv_diag_velocity,
        const grid::shell::DistributedDomain&                           velocity_domain,
        const grid::shell::DistributedDomain&                           pressure_domain,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        velocity_ownership_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        pressure_ownership_mask )
    : A_( viscous )
    , B_T_( gradient )
    , B_( divergence )
    , kw_solver_( std::move( kw_solver ) )
    , c_w_inv_diag_velocity_( c_w_inv_diag_velocity )
    , z1_( "wbfbt_z1", pressure_domain, pressure_ownership_mask )
    , z2_( "wbfbt_z2", pressure_domain, pressure_ownership_mask )
    , v1_( "wbfbt_v1", velocity_domain, velocity_ownership_mask )
    , v2_( "wbfbt_v2", velocity_domain, velocity_ownership_mask )
    {}

    void solve_impl( OperatorType& /*M_p_unused*/, SolutionVectorType& x, const RHSVectorType& b )
    {
        // 1) z1 := K_w^{-1} b
        kw_solver_->solve( b, z1_ );

        // 2) v1 := B^T z1
        linalg::apply( B_T_, z1_, v1_ );

        // 3) v1 := D_w^{-1} . v1  (with D_w = C_w, same inv-diag)
        linalg::scale_in_place( v1_, c_w_inv_diag_velocity_ );

        // 4) v2 := A v1
        linalg::apply( A_, v1_, v2_ );

        // 5) v2 := C_w^{-1} . v2
        linalg::scale_in_place( v2_, c_w_inv_diag_velocity_ );

        // 6) z2 := B v2
        linalg::apply( B_, v2_, z2_ );

        // 7) x := K_w^{-1} z2
        kw_solver_->solve( z2_, x );
    }

  private:
    ViscousOp    A_;
    GradientOp   B_T_;
    DivergenceOp B_;

    std::shared_ptr< WBFBTPressurePoissonSolver< ScalarType > > kw_solver_;

    VelocityVector c_w_inv_diag_velocity_;

    PressureVector z1_;
    PressureVector z2_;
    VelocityVector v1_;
    VelocityVector v2_;
};

} // namespace terra::mantlecirculation
