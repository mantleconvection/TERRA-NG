#pragma once

#include "linalg/vector_q1.hpp"

namespace terra::mantlecirculation {

/// @brief Abstract solver for the w-BFBT pressure-space Poisson operator
///        K_w = B C_w^-1 B^T  where  C_w = M_u_lumped(sqrt(mu)).
///
/// The w-BFBT Schur complement approximation (Rudi, Stadler, Ghattas 2017,
/// eq. 7) applies this inverse twice per outer FGMRES iteration, with a
/// mat-vec sandwiched between.  Robustness at high viscosity contrast
/// depends almost entirely on the quality of this inverse.
///
/// Implementations are free to choose any assembly + inversion strategy:
/// - a re-discretized DivKGrad with k = 1/sqrt(mu) + scalar Q1 multigrid;
/// - a literal `B C_w^-1 B^T` matvec wrapped in a Krylov solver;
/// - AMG;
/// - a direct solver on small problems;
/// - ...
///
/// The orchestrator (WBFBTSchurPreconditioner) consumes only this interface.
template < typename ScalarType >
class WBFBTPressurePoissonSolver
{
  public:
    using PressureVector = linalg::VectorQ1Scalar< ScalarType >;

    virtual ~WBFBTPressurePoissonSolver() = default;

    /// @brief Apply K_w^-1 to `rhs`, write into `sol`.
    ///
    /// Must be reasonably accurate (residual reduction at least ~1e-3) but
    /// cheap; called twice per outer Stokes-FGMRES iteration.  Implementations
    /// should treat `sol` as an initial guess (zero unless the caller pre-
    /// seeded it).
    virtual void solve( const PressureVector& rhs, PressureVector& sol ) = 0;

    /// @brief Refresh internal state when the fine-level viscosity has
    /// changed.  Called from `WBFBTSchurPreconditioner` whenever the
    /// orchestrator computes a new pressure-level `sqrt(eta)` field (after
    /// `StokesContext::update_viscosity`).
    ///
    /// `sqrt_eta_pressure_finest` is the velocity-projected `sqrt(eta)` on
    /// the finest pressure-MG level.  Implementations restrict it to coarser
    /// pressure levels as needed and update their internal coefficient
    /// representations / smoother spectral estimates.
    virtual void refresh( const PressureVector& sqrt_eta_pressure_finest ) = 0;
};

} // namespace terra::mantlecirculation
