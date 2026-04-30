# Entropy-viscosity stabilization {#entropy-viscosity}

Entropy viscosity (EV) is a residual-driven, isotropic artificial-diffusion
stabilization for the advection–diffusion energy equation discretised in Q1 on
the wedge mesh.  We follow Kronbichler, Heister & Bangerth (*GJI* **191**,
12–29, 2012), which adapts Guermond, Pasquetti & Popov (*JCP* **230**,
4248–4267, 2011).  This is the alternative to SUPG used by the EV energy solver.

## Motivation: why not just SUPG (or pure Galerkin)?

The Q1 Galerkin discretization of \f$\partial_t T + \mathbf{u}\cdot\nabla T -
\kappa\nabla^2 T = \gamma\f$ is unstable in the advection-dominated regime that
mantle convection always sits in (Pe \f$\gg 1\f$).  Without stabilization the
solution develops O(1) wiggles around boundary layers, plumes, and shear-driven
fronts; with κ small enough, those wiggles destroy the temperature distribution
within a few time steps.

SUPG (streamline-upwind Petrov–Galerkin) is the textbook fix: add an anisotropic
diffusion proportional to \f$\mathbf{u}\otimes\mathbf{u}\f$ that smears only
along streamlines, leaving the cross-stream profile alone.  It is cheap,
unconditionally available, and *always on* — even in regions where the solution
is perfectly smooth and no stabilization is needed.  That last property is the
problem: SUPG keeps shaving off the small-scale structure of plumes and slabs
even where the Galerkin scheme would have been stable on its own, which
systematically biases the long-time thermal structure.

Entropy viscosity solves the same instability but from the opposite direction:
add an *isotropic* artificial diffusion \f$\nu_h(\mathbf{x})\f$ that is built to
be small wherever the solution is smooth and to grow only where the strong-form
PDE residual indicates that the discrete solution is not satisfying its own
equation.  The result is a stabilization that is essentially zero in 99% of the
domain and turns on locally near the layers and fronts that actually need it.

## The entropy residual

The construction starts from an *entropy* — a strictly convex functional of the
temperature whose evolution is sharpened wherever the strong-form PDE residual
is nonzero.  KHB use the simple quadratic entropy
\f[
    E(T) = \tfrac{1}{2}(T - T_m)^2,
    \qquad
    T_m = \tfrac{1}{2}\big(T_{\min} + T_{\max}\big),
\f]
with \f$T_{\min}, T_{\max}\f$ the global temperature extrema over the current
solution.  Centring on the midpoint of the physical range makes \f$E\f$
symmetric in \f$T - T_m\f$ and decouples it from the absolute temperature
level.

Multiplying the strong-form energy equation by the chain-rule factor \f$T - T_m
= \mathrm{d}E/\mathrm{d}T\f$ and using \f$\partial_t E = (T - T_m)\,\partial_t
T\f$ gives the equivalent entropy equation:
\f[
    \partial_t E + (T - T_m)\,\big(\mathbf{u}\cdot\nabla T - \kappa\,\nabla^2 T - \gamma\big) = 0.
\f]
This is identically zero on the exact solution.  Plugging the discrete solution
into the LHS yields a non-zero residual whose magnitude is a local estimator of
how badly the discrete \f$T\f$ violates its own PDE.  Define the per-element
estimator
\f[
    r_E\big|_K
    = \big\|\,\partial_t E + (T - T_m)\,\big(\mathbf{u}\cdot\nabla T - \kappa\,\nabla^2 T - \gamma\big)\,\big\|_{\infty,\,K}.
\f]
Where the strong form is well-resolved, every term on the right matches and
\f$r_E \to 0\f$.  Where the gradient is too steep for the local mesh — boundary
layers, shock-like plumes — the discrete \f$\nabla T\f$, \f$\nabla^2 T\f$, and
\f$\partial_t T\f$ disagree with each other and \f$r_E\f$ is large.  This is
the signal EV uses to decide where to add diffusion.

## The ν_h formula

The artificial viscosity per element is the smaller of two terms:
\f[
    \nu_h\big|_K
    = \min\!\Big(\;
        \alpha_\mathrm{max}\,h_K\,\|\mathbf{u}\|_{\infty,K},
        \;\;
        \alpha_E\,h_K^2\,\,r_E\big|_K\,/\,D
      \;\Big).
\f]
The first term is a **first-order upwind cap**: it is the diffusion that a
monotone first-order upwind scheme would introduce on this element with mesh
size \f$h_K\f$ and velocity \f$\|\mathbf{u}\|_{\infty,K}\f$.  EV will never add
more than this; in particular, \f$\nu_h\f$ is bounded above even in the limit
\f$r_E \to \infty\f$.  This guarantees that the EV-stabilized scheme is at most
as diffusive as first-order upwind and never worse.

The second term is the **residual-driven branch**.  The factor \f$h_K^2\f$
scales like a Galerkin discretization error, the residual \f$r_E\f$ is the
local error indicator, and the global normalization
\f[
    E_\mathrm{avg} = \frac{1}{|\Omega|}\int_\Omega E\,\mathrm{d}V,
    \qquad
    D = \big\|E - E_\mathrm{avg}\big\|_{\infty,\,\Omega},
\f]
makes the ratio dimensionless: \f$D\f$ is the entropy variation across the
domain, against which \f$r_E\f$ is measured.  When the temperature is nearly
constant, \f$D \to 0\f$ and the residual branch would diverge — a small floor
\f$D \ge D_\mathrm{floor}\f$ is applied and the upwind cap then takes over.

The constants \f$\alpha_\mathrm{max} = 0.026\,d\f$ (so 0.078 in 3D) and
\f$\alpha_E = 1\f$ are taken from ASPECT and have been thoroughly tested over
the past decade in the geodynamics community.  Tezduyar-style shock-capturing
work has explored \f$\alpha_E\f$ in \f$[0.1, 2]\f$ for related problems.

## Time-lagging and the linear implicit solve

If \f$\nu_h\f$ depended on the unknown \f$T^{n+1}\f$, the implicit
advection–diffusion solve would become nonlinear.  EV avoids that by evaluating
\f$\nu_h\f$ from the past states only.  Concretely \f$\partial_t E\f$ uses a
backward difference \f$(E^n - E^{n-1})/\Delta t\f$, the spatial residual terms
use \f$T^n\f$, and \f$T_m\f$, \f$E_\mathrm{avg}\f$, \f$D\f$ are likewise computed
from \f$T^n\f$.  This is the IMPES-style "explicit-lagged stabilization" used by
ASPECT: \f$\nu_h\f$ is fixed before the solve and the implicit solve sees a
linear, symmetric Galerkin operator.  The EV contribution enters as an explicit
RHS term \f$-\Delta t\,\nabla\!\cdot\!(\nu_h\,\nabla T^n)\f$; the LHS is just
\f$M + \Delta t\,(K_\mathrm{adv} + K_\mathrm{diff})\f$.

The price for keeping the solve linear is that the stabilization lags the
solution by one step.  In practice this is harmless under the usual
advection-CFL time step that the EV solver runs at: \f$T\f$ does not move
enough between consecutive steps for the lagged \f$\nu_h\f$ to be misplaced.

A BDF-1 lagging on \f$\partial_t E\f$ is used in the current implementation; a
BDF-2 swap is straightforward (one extra history slot) and would tighten the
residual estimator on smooth problems.

## Discrete realisation: per-wedge ν_h

The discrete unknowns live at Q1 nodes of the wedge mesh, but \f$\nu_h\f$ is
piecewise constant **per wedge** — two values per hex cell.  This matches the
true element granularity of the FE space (each hex is split into two prismatic
wedges) and gives the stabilization-field exactly the same resolution as the
operator assembly.  An older variant projected a per-cell \f$\nu_h\f$ to nodes
via a Q1 nodal average, which under-resolved the field by a factor of two and
required an extra halo exchange of the projected coefficients.

Per wedge \f$w\f$, \f$\nu_h\f$ is evaluated at the same Felippa 3×2 quadrature
points used by the rest of the wedge operator suite (Mass, Laplace,
DivKGrad, AD-SUPG).  At each quadrature point \f$q\f$ we form:

- field interpolants \f$T(q), T^{n-1}(q), \mathbf{u}(q), \mathrm{Lap}(q)\f$ via
  \f$\sum_j N_j(q)\,\cdot\f$ over the 6 wedge nodes,
- the FE-consistent physical-space gradient
  \f$\nabla T(q) = J^{-T}(q)\sum_j T_j\,\nabla_\xi N_j(q)\f$,
- the strong-form residual \f$r_{E}(q)\f$ from the same expression as above.

The wedge-local residual is the \f$\infty\f$-norm over the six quadrature
points, \f$r_E\big|_w = \max_q r_E(q)\f$, and the wedge characteristic length
is computed from the same quadrature as
\f[
    h_w = V_\mathrm{wedge}^{1/3},
    \qquad
    V_\mathrm{wedge} = \sum_q w_q\,|\det J(q)|.
\f]
The per-wedge \f$\nu_h\f$ is then the formula above with \f$h_K \to h_w\f$ and
\f$r_E\big|_K \to r_E\big|_w\f$.

## Volume-weighted normalisation

Both \f$T_m\f$ (range midpoint) and \f$D\f$ (\f$\infty\f$-norm of \f$E - E_\mathrm{avg}\f$
over owned nodes) come from ownership-masked nodal reductions, so each Q1 node
contributes exactly once across the partition.  The third global scalar,
\f$E_\mathrm{avg}\f$, is a true FE volume integral evaluated on the same Felippa
quadrature:
\f[
    E_\mathrm{avg}
    = \frac{\sum_w \sum_q w_q\,|\det J(q)|\,E(q)}
           {\sum_w \sum_q w_q\,|\det J(q)|},
    \qquad
    E(q) = \tfrac{1}{2}(T(q) - T_m)^2.
\f]
Cells are not shared between MPI ranks (only nodes are), so the local sums form
a partition-of-unity at the integration level and the MPI sum is exact.  This
matters: an earlier nodal average of \f$E\f$ would weight every node equally,
biasing \f$E_\mathrm{avg}\f$ toward regions with denser node distribution
(e.g.\ near the poles on a spherical shell).

## The Laplacian inside the residual

The residual \f$r_E\f$ contains the strong-form \f$\kappa\nabla^2 T\f$ term,
which has no direct meaning for a Q1 FE function (its second derivative is a
distribution).  We need a discrete surrogate.  The standard choice is to project
the *weak* Laplacian onto the FE space:
\f[
    \mathrm{lap}^{\mathrm{nodal}} = M^{-1}\,K\,T,
\f]
where \f$K_{ij} = \int \kappa\,\nabla\phi_i\cdot\nabla\phi_j\,\mathrm{d}V\f$ is
the Galerkin Laplacian and \f$M\f$ is some mass matrix.  Using the consistent
mass requires solving a linear system; using the lumped (row-sum) diagonal
gives an explicit pointwise formula and is the standard cheap choice.

The EV solver does this projection **per wedge with the wedge's own lumped
mass**:
\f[
    \mathrm{lap}_w[i] = \frac{\kappa\,K_w[i,j]\,T_w[j]}{M_{w,\mathrm{lumped}}[i]},
    \qquad
    K_w[i,j] = \int_w \nabla\phi_i\cdot\nabla\phi_j\,\mathrm{d}V,
    \qquad
    M_{w,\mathrm{lumped}}[i] = \sum_j \int_w \phi_i\,\phi_j\,\mathrm{d}V.
\f]
This is a *single-element* projection: each wedge sees only its own assembly,
not the contributions of neighbouring wedges that share its nodes.  The result
is a 6-vector of nodal lap values per wedge, consumed at every Felippa quadrature
point via the same \f$\sum_j N_j(q)\,\mathrm{lap}_w[j]\f$ interpolation as
every other field.

The per-wedge variant is mathematically distinct from the global lumped-mass
projection (which divides by \f$\sum_w M_{w,\mathrm{lumped}}\f$ at each shared
node).  Both are O(h²)-consistent estimators of \f$\nabla^2 T\f$; the per-wedge
version trades one source of accuracy (sharing across wedges that meet at a
node) for one source of cleanliness (no halo exchange, no need to pre-build a
global lumped mass, no special boundary handling).  Because \f$\mathrm{lap}_w\f$
only enters the residual estimator — *not* the operator that goes into the
implicit solve — the loss of consistency at element interfaces does not affect
solution accuracy; it only slightly biases where EV decides to add diffusion.

The 6×6 projector \f$P_w[i,j] = \kappa\,K_w[i,j]/M_{w,\mathrm{lumped}}[i]\f$ is
geometry- and \f$\kappa\f$-only, so it is assembled once at solver construction
and stored.  Each step costs one 6×6 matrix–vector product per wedge.

## Picard locking

The mantle-circulation outer loop is a Picard fixed-point iteration over the
coupled \f$(T, \mathbf{u})\f$ system: each Picard step solves Stokes for
\f$\mathbf{u}\f$ given \f$T\f$, then solves the energy equation for \f$T\f$
given \f$\mathbf{u}\f$.  The explicit-lagged \f$\nu_h\f$ is a function of the
*previous-step* \f$T\f$ and the *current* \f$\mathbf{u}\f$, but recomputing it
at every Picard iteration would make the energy solve re-converge to a
slightly different RHS each iteration, which slows or breaks the Picard
convergence.

The EV solver therefore locks \f$\nu_h\f$ across Picard iterations within a
single timestep: the first energy substep of the first Picard iteration
computes \f$\mathrm{lap}_w\f$ and \f$\nu_h\f$ from the current \f$T^n\f$, and
all subsequent Picard iterations of the same timestep reuse it.  Substeps
within one Picard iteration always recompute, since \f$T\f$ evolves from
substep to substep.  This keeps the explicit stabilization consistent across
the \f$(T, \mathbf{u})\f$ Picard fixed point while still letting it track the
intra-timestep evolution.

## Summary: what the EV scheme does each timestep

1. Project \f$\kappa\nabla^2 T^n\f$ to nodal lap values per wedge via the
   precomputed lumped-mass projector — one 6×6 mat-vec per wedge, no halos.
2. Reduce \f$T_{\min}, T_{\max}, E_\mathrm{avg}, D\f$ globally; one
   parallel scan and three MPI reductions.
3. For every wedge, compute \f$r_E\big|_w\f$, \f$h_w\f$, \f$\|\mathbf{u}\|_{\infty,w}\f$
   on the Felippa quadrature and store the per-wedge \f$\nu_h\f$.
4. Apply the Galerkin operator \f$\nabla\!\cdot\!(\nu_h\,\nabla\,)\f$ with the
   per-wedge \f$\nu_h\f$ as coefficient, and add its action on \f$T^n\f$ to the
   RHS of the implicit advection–diffusion solve as
   \f$-\Delta t\,K_\mathrm{ev}\,T^n\f$.
5. Do the standard linear, symmetric Galerkin solve for \f$T^{n+1}\f$.

The whole pipeline is single-element local up to the standard additive
neighbourhood reduction of the \f$\nu_h\f$-Galerkin operator's output — the
same reduction every other Galerkin operator in the framework already performs.
