# Entropy-viscosity stabilization {#entropy-viscosity}

Residual-driven, isotropic artificial-diffusion stabilization for the Q1
advection–diffusion energy equation
\f[
    \partial_t T + \mathbf{u}\cdot\nabla T - \kappa\,\nabla^2 T = \gamma,
\f]
following Kronbichler, Heister & Bangerth (*GJI* **191**, 12–29, 2012) /
Guermond, Pasquetti & Popov (*JCP* **230**, 4248–4267, 2011).  Adds a strictly
local diffusion \f$\nu_h(\mathbf{x})\f$ that vanishes where the strong-form PDE
residual is small and grows only near layers.

## The entropy and its residual

\f[
    E(T) = \tfrac{1}{2}(T - T_m)^2,
    \qquad
    T_m = \tfrac{1}{2}(T_{\min} + T_{\max}).
\f]
Multiplying the PDE by \f$\mathrm{d}E/\mathrm{d}T = (T - T_m)\f$ and using
\f$\partial_t E = (T - T_m)\,\partial_t T\f$ gives the entropy equation
\f[
    \partial_t E + (T - T_m)\big(\mathbf{u}\cdot\nabla T - \kappa\,\nabla^2 T - \gamma\big) = 0,
\f]
identically zero on the exact solution.  Plug the discrete \f$T\f$ in to get
the per-element residual estimator
\f[
    r_E\big|_K
    = \big\|\,\partial_t E + (T - T_m)\big(\mathbf{u}\cdot\nabla T - \kappa\,\nabla^2 T - \gamma\big)\,\big\|_{\infty,\,K}.
\f]

## The ν_h formula

\f[
    \boxed{\;
    \nu_h\big|_K = \min\!\Big(
        \alpha_\mathrm{max}\,h_K\,\|\mathbf{u}\|_{\infty,K},
        \;\;
        \alpha_E\,h_K^2\,r_E\big|_K \,/\, D
    \Big)\;}
\f]
with global normalization
\f[
    E_\mathrm{avg} = \frac{1}{|\Omega|}\int_\Omega E\,\mathrm{d}V,
    \qquad
    D = \max\!\big(\|E - E_\mathrm{avg}\|_{\infty,\Omega},\,D_\mathrm{floor}\big),
\f]
and ASPECT constants \f$\alpha_\mathrm{max} = 0.026\,d = 0.078\f$ in 3D,
\f$\alpha_E = 1\f$.  The first branch is the first-order upwind cap; the
second is the residual-driven branch.

## Time-lagging

\f$\partial_t E\f$ uses the BDF-1 backward difference
\f[
    \partial_t E \approx \frac{E^n - E^{n-1}}{\Delta t},
\f]
all spatial residual terms use \f$T^n\f$, and \f$T_m, E_\mathrm{avg}, D\f$ are
computed from \f$T^n\f$.  \f$\nu_h\f$ is fully determined before the implicit
solve, which therefore stays linear and symmetric.  EV enters as an explicit
RHS contribution
\f[
    \mathrm{rhs}_\mathrm{ev} = -\Delta t\;\nabla\!\cdot\!\big(\nu_h\,\nabla T^n\big),
\f]
added to the Galerkin advection–diffusion RHS.

## Per-wedge discretisation

\f$\nu_h\f$ is **piecewise constant per wedge** (2 per hex), evaluated at the
Felippa 3×2 quadrature.  Per quadrature point \f$q\f$:
\f[
    T(q) = \sum_j N_j(q)\,T_j,
    \qquad
    \nabla T(q) = J^{-T}(q)\sum_j T_j\,\nabla_\xi N_j(q),
\f]
with analogous interpolants for \f$T^{n-1}, \mathbf{u}, \mathrm{Lap}\f$.  The
wedge-local residual and characteristic length are
\f[
    r_E\big|_w = \max_q r_E(q),
    \qquad
    h_w = V_\mathrm{wedge}^{1/3},
    \qquad
    V_\mathrm{wedge} = \sum_q w_q\,|\det J(q)|.
\f]

## Volume-weighted E_avg

True FE volume integral on the same Felippa quadrature:
\f[
    E_\mathrm{avg}
    = \frac{\sum_w \sum_q w_q\,|\det J(q)|\,E(q)}
           {\sum_w \sum_q w_q\,|\det J(q)|},
    \qquad
    E(q) = \tfrac{1}{2}(T(q) - T_m)^2.
\f]
Cells are not shared between MPI ranks, so the local sums are partition-of-unity
and the MPI sum is exact.

## Per-wedge lumped-mass Laplacian projection

Need a discrete surrogate for \f$\kappa\nabla^2 T\f$ inside \f$r_E\f$.  Per
wedge:
\f[
    K_w[i,j] = \int_w \nabla\phi_i\cdot\nabla\phi_j\,\mathrm{d}V,
    \qquad
    M_{w,\mathrm{lumped}}[i] = \sum_j \int_w \phi_i\,\phi_j\,\mathrm{d}V,
\f]
\f[
    P_w[i,j] = \kappa\,\frac{K_w[i,j]}{M_{w,\mathrm{lumped}}[i]},
    \qquad
    \mathrm{lap}_w = P_w\,T_w \quad \in \mathbb{R}^6.
\f]
\f$P_w\f$ is geometry- and \f$\kappa\f$-only and is assembled once at solver
construction.  Each step costs one 6×6 matrix–vector product per wedge.  No
halo exchange — strictly single-element local.  Inside \f$r_E\f$:
\f[
    \mathrm{Lap}(q) = \sum_j N_j(q)\,\mathrm{lap}_w[j].
\f]

## Picard locking

\f$\nu_h\f$ is locked across Picard iterations within a single timestep:
substep 0 of Picard iter 0 computes it from \f$T^n\f$, all later Picard iters
of the same step reuse it.  Substeps within one Picard iter recompute since
\f$T\f$ evolves between them.  This keeps the explicit-lagged stabilization
consistent across the \f$(T, \mathbf{u})\f$ Picard fixed point.

## Per-step summary

1. \f$\mathrm{lap}_w \leftarrow P_w\,T_w^n\f$ — one 6×6 mat-vec per wedge.
2. Reduce \f$T_{\min}, T_{\max}, E_\mathrm{avg}, D\f$ globally.
3. Per wedge: assemble \f$r_E|_w, h_w, \|\mathbf{u}\|_{\infty,w}\f$ and store
   \f$\nu_h|_w\f$.
4. \f$\mathrm{rhs}_\mathrm{ev} \leftarrow -\Delta t\,K_\mathrm{ev}(\nu_h)\,T^n\f$.
5. Linear solve \f$(M + \Delta t\,(K_\mathrm{adv} + K_\mathrm{diff}))\,T^{n+1}
   = M\,T^n + \mathrm{rhs}_\mathrm{ev}\f$.
