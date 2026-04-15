# Stokes Discretization and Solver {#stokes}

Finite-element discretization of the Stokes equations on the thick spherical shell, together
with the block-structured solver used in applications such as `apps/mantlecirculation`.

---

## Physical problem

We solve the steady-state Stokes equations governing slow, viscous mantle flow:

\f[
\begin{aligned}
-\nabla \cdot \boldsymbol{\tau}(\mathbf{u})

+ \nabla p &= \mathbf{f}, \\
    - \nabla \cdot \mathbf{u} &= g.
      \end{aligned}
      \f]

> Note that we adhere to the convention of having a negative sign in front of the divergence term in the mass
> conservation equation. This is important for the correct implementation of the compressibility terms later.
> While this convection is not typical in the literature, it is more natural here because it reflects the implementation
> and is required to make the system symmetric. This comes from the fact that the weak form of the pressure gradient 
> in the momentum equation also introduces a negative sign 
> (\f$\int_\Omega \mathbf{v} \cdot \nabla p = - \int_\Omega p \nabla \cdot \mathbf{v} + 
> \int_{\partial \Omega} p \mathbf{n} \cdot \mathbf{v} \f$, with test function \f$\mathbf{v}\f$) 

In the incompressible (Boussinesq) case \f$g = 0\f$. For compressible/anelastic
extensions \f$g \neq 0\f$ but the momentum operator on the left-hand side stays the same
— see [Compressible and anelastic extensions](#stokes-compressible) below.

**Notation:**

- \f$\mathbf{u}\f$ — velocity field,
- \f$p\f$ — dynamic pressure,
- \f$\eta > 0\f$ — dynamic viscosity (scalar, possibly spatially varying),
- \f$\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2}(\nabla\mathbf{u} + (\nabla\mathbf{u})^T)\f$
  — symmetric strain-rate tensor,
-

\f$\boldsymbol{\tau}(\mathbf{u}) = 2\eta\bigl(\boldsymbol{\varepsilon}(\mathbf{u}) - \frac{1}{3}(\nabla\cdot\mathbf{u})\,\mathbf{I}\bigr)\f$
— deviatoric stress tensor,

- \f$\mathbf{f}\f$ — body force; for mantle convection the buoyancy
  \f$\mathbf{f} = \mathrm{Ra}\,T\,\hat{r}\f$ (Rayleigh number \f$\mathrm{Ra}\f$,
  temperature \f$T\f$, outward radial unit vector \f$\hat{r}\f$),
- \f$g\f$ — mass conservation source (\f$g = 0\f$ for incompressible flow).

The stress tensor \f$\boldsymbol{\tau}\f$ crucially does **not** assume
\f$\nabla\cdot\mathbf{u} = 0\f$: the \f$-\frac{2}{3}\eta(\nabla\cdot\mathbf{u})\mathbf{I}\f$
correction is always present in the discrete operator. This makes the same LHS operator
valid for both incompressible and compressible/anelastic formulations (and the same optimized implementation
can be used in either case).

The system is solved at every time step with the current temperature field to obtain the
instantaneous velocity and pressure, which are then fed into the advection-diffusion solver
for the next temperature update.

---

## Finite element discretization

### Element pair

The velocity–pressure pair uses the **P1-iso-P2 / P1** mixed finite element:

- **Velocity** \f$\mathbf{u}_h\f$: continuous piecewise-linear (P1) on a twice-refined mesh
  (`velocity_level = mesh_level`). Three components per node.
- **Pressure** \f$p_h\f$: continuous piecewise-linear (P1) on the coarser mesh
  (`pressure_level = mesh_level - 1`).

This pair satisfies the inf-sup (LBB) stability condition, preventing pressure oscillations
without requiring any stabilization on the pressure field.

The underlying geometry uses **wedge (prism) elements** formed by extruding triangular
surface elements in the radial direction. Each hexahedral shell cell is decomposed into
two wedges. The geometric mapping and basis functions are defined in `fe/wedge/integrands.hpp`.

### Discrete system

Assembling the variational form of the Stokes equations yields the **2×2 saddle-point system**

\f[
\begin{pmatrix} A & B^T \\ B & 0 \end{pmatrix}
\begin{pmatrix} \mathbf{u} \\ \mathbf{p} \end{pmatrix}
=
\begin{pmatrix} \mathbf{f} \\ \mathbf{g} \end{pmatrix},
\f]

where

- \f$A\f$ is the **viscous block** (velocity–velocity coupling),
- \f$B^T\f$ is the **gradient operator** (pressure-to-velocity coupling),
- \f$B\f$ is the **divergence operator** (velocity-to-pressure coupling),
- the lower-right block is zero (no pressure stabilization needed for the inf-sup stable pair),
- \f$\mathbf{g}\f$ is the discretised mass conservation source (\f$\mathbf{g} = \mathbf{0}\f$ for incompressible flow).

The corresponding C++ types from `EpsDivDivStokes` are:

- `Block11Type` = `EpsilonDivDivKerngen` (deviatoric viscous block, optimized kernel)
- `Block12Type` = `Gradient`
- `Block21Type` = `Divergence`
- `Block22Type` = `Zero`

### Viscous block variants

Two variants of the viscous block are available:

1. **`Stokes`** (`stokes.hpp`): uses the **vector Laplace** operator
   \f$A_{ij} = \int_\Omega 2\eta\,\nabla\phi_j : \nabla\phi_i\f$.
   Equivalent to the deviatoric form only when \f$\nabla\cdot\mathbf{u} = 0\f$; less
   accurate for spatially varying viscosity or compressible flow.

2. **`EpsDivDivStokes`** (`epsilon_divdiv_stokes.hpp`): uses the **deviatoric stress**
   bilinear form implemented in `EpsilonDivDivKerngen`:
   \f[
   a(\mathbf{u},\mathbf{v})
   = \int_\Omega 2\eta\,\boldsymbol{\varepsilon}(\mathbf{u}):\boldsymbol{\varepsilon}(\mathbf{v})
    - \frac{2}{3}\eta\,(\nabla\cdot\mathbf{u})(\nabla\cdot\mathbf{v})\,\mathrm{d}x,
      \f]
      which is the weak form of \f$-\nabla\cdot\boldsymbol{\tau}(\mathbf{u})\f$.
      The \f$-\frac{2}{3}(\nabla\cdot\mathbf{u})(\nabla\cdot\mathbf{v})\f$ term is the
      deviatoric correction; it vanishes only when \f$\nabla\cdot\mathbf{u} = 0\f$ but is
      always included in the discrete operator. This is the variant used in
      `mantlecirculation`.

The viscous block \f$A\f$ is implemented by the **optimized kernel** `EpsilonDivDivKerngen`
(`epsilon_divdiv_kerngen.hpp`). Unlike the reference implementation `EpsilonDivDiv`, the
kerngen variant fuses the loop over velocity components and quadrature points into a
single, highly optimized Kokkos kernel that avoids redundant memory traffic and allows the
compiler to vectorize the inner loop more aggressively.

### Matrix-free assembly

All fine-grid operators (\f$A\f$, \f$B\f$, \f$B^T\f$) are **matrix-free**: no global
sparse matrix is ever assembled or stored. Instead, each call to `apply_impl` recomputes
all element integrals from scratch using the current viscosity field and geometry.

The implementation follows the standard element-loop pattern but executed via
`Kokkos::parallel_for` over all locally owned cells with atomic contributions to shared
nodes. This means:

- **Memory footprint**: proportional to the number of degrees of freedom (vectors only),
  independent of the number of non-zeros in the stiffness matrix.
- **Re-use of geometry**: physical coordinates and Jacobians are recomputed per matrix-vector
  product — there is no cached geometry array.
- **Viscosity coupling**: the viscosity field \f$\eta\f$ is read from a `VectorQ1Scalar` at
  quadrature points every application, so changes to \f$\eta\f$ (e.g. between time steps)
  are automatically reflected without re-assembly.
- **Communication**: one ghost-layer exchange via
  `communication::shell::send_recv` (additive communication pattern) per `apply_impl` call.

---

## Solver

The saddle-point system is solved by a **preconditioned FGMRES** iteration with a
block-triangular preconditioner for the full 2×2 system.

### Outer Krylov solver: FGMRES

`linalg::solvers::FGMRES<Stokes, PrecStokes>` is used because the preconditioner is
non-stationary (it contains an inner multigrid V-cycle with Chebyshev smoothing).
FGMRES(m) with restart \f$m\f$ requires \f$2m + 4\f$ temporary vectors of the combined
velocity–pressure type `VectorQ1IsoQ2Q1`.

In a time-stepping context the velocity changes only slightly between time steps;
using a warm start (initial guess = previous solution) and a small number of time steps and restart should be enough.

**Robustness features.**  The FGMRES implementation includes several safeguards:

- **NaN/Inf in preconditioner output**: if the preconditioner produces a vector containing
  NaN or Inf entries (which can happen during the early iterations of a poorly conditioned
  problem or after a large viscosity change), the preconditioned direction is silently
  replaced by the raw input vector — equivalent to skipping the preconditioner for that
  step — and a warning is printed. This prevents divergence propagation while allowing
  the iteration to continue.
- **Arnoldi breakdown**: if the new Krylov basis vector has norm below machine epsilon
  times the initial residual (\f$h_{j+1,j} < \epsilon\,\|r_0\|\f$), the inner loop exits
  early and a restart is triggered.
- **Stagnation / near-singularity**: if the Givens-rotated diagonal entry \f$|H_{jj}|\f$
  falls below machine epsilon, the current column is dropped and a restart is triggered.

### Block-triangular preconditioner

The preconditioner has the upper-triangular form

\f[
\mathcal{P}^{-1} =
\begin{pmatrix} A^{-1} & -A^{-1} B^T \hat{S}^{-1} \\ 0 & \hat{S}^{-1} \end{pmatrix},
\f]

where \f$\hat{S}^{-1}\f$ approximates the inverse of the Schur complement
\f$S = -B A^{-1} B^T\f$. This is implemented in
`linalg::solvers::BlockTriangularPreconditioner2x2`.

**Velocity block** \f$A^{-1}\f$: **multigrid.**
The viscous block is inverted approximately by a geometric multigrid (GMG) V-cycle:

- **Smoother**: Chebyshev-accelerated Jacobi (`linalg::solvers::Chebyshev`, see below).
- **Prolongation / Restriction**: constant interpolation / restriction between mesh levels
  (`ProlongationVecConstant`, `RestrictionVecConstant`).
- **Coarse-grid operator**: assembled by Galerkin coarse-grid approximation (**GCA**),
  solved by PCG on the coarsest level.

#### Chebyshev smoother and eigenvalue estimation

The smoother is a **Chebyshev-accelerated Jacobi** iteration of polynomial order \f$p\f$
(`viscous_pc_chebyshev_order`). For \f$p = 1\f$ it reduces to standard weighted Jacobi;
for \f$p \geq 2\f$ it applies the Chebyshev three-term recurrence to \f$D^{-1}A\f$,
where \f$D = \mathrm{diag}(A)\f$. The recurrence coefficients depend on bounds
\f$[\lambda_{\min}, \lambda_{\max}]\f$ of the spectrum of \f$D^{-1}A\f$:
\f[
\lambda_{\max} = 1.5 \cdot \hat\lambda_{\max},
\qquad
\lambda_{\min} = 0.1 \cdot \hat\lambda_{\max},
\f]
where \f$\hat\lambda_{\max}\f$ is the spectral radius estimate obtained by
**power iteration** on \f$D^{-1}A\f$ (`viscous_pc_num_power_iterations` steps).
The 1.5 factor provides a safety margin above the estimate; the 0.1 factor is a
heuristic lower bound (about 10 % of the upper bound is a typical choice for elliptic
problems on quasi-uniform meshes).

The power iteration and eigenvalue estimation are performed **once** before the first
solve, and re-triggered automatically if `refresh_max_eigenvalue_estimate_in_next_solve()`
is called (e.g. after a viscosity update).

Each Chebyshev application performs \f$p\f$ matrix-vector products. The number of
pre/post-smoothing steps (iterations of the whole Chebyshev polynomial) is set by
`viscous_pc_num_smoothing_steps_prepost`.

#### Galerkin coarse-grid approximation (GCA)

Standard geometric multigrid constructs coarse-grid operators by re-discretising the PDE
on coarser meshes. For **strongly varying viscosity** (several orders of magnitude, as
typical in the mantle) this produces poor coarse-grid representations: a coarse element
spanning a viscosity jump will use the wrong effective viscosity, and the multigrid
convergence deteriorates badly.

The GCA avoids this by computing the coarse-grid operator as the Galerkin triple product
\f[
A_c = R\,A_f\,P,
\f]
where \f$P\f$ is the prolongation operator and \f$R = P^T\f$ is the restriction.
This guarantees that the coarse-grid operator is spectrally equivalent to the fine-grid
operator projected onto the coarse space, regardless of viscosity variations.

**What is stored.**  GCA is the only place in the code where a matrix is explicitly
assembled, and it is applied only on the **coarser grid levels** — the finest grid
always remains matrix-free. Because each refinement step multiplies the number of cells
by 8, the coarser levels together hold at most about \f$1/8\f$ of what a full fine-grid
matrix would require. For each coarse-grid element (wedge), `TwoGridGCA` computes and
stores the \f$18 \times 18\f$ local element matrix (6 velocity nodes × 3 components).
These local matrices are held in a `Grid4DDataMatrices` array on the device. During a
coarse-grid matrix-vector product the stored local matrices are applied element by element
— essentially a sparse matrix-vector product in local-matrix form.

**Two application modes** are supported via `OperatorStoredMatrixMode`:

- `Full` (`gca = 1`): GCA is applied on all elements. Safe default for any viscosity
  structure.
- `Selective` (`gca = 2`): GCA is applied only on elements identified by
  `GCAElementsCollector` as having significant viscosity contrast; remaining elements use
  the matrix-free kernel. Reduces memory and assembly cost when high-viscosity regions
  are localised.

**Pressure Schur complement** \f$\hat{S}^{-1}\f$: **lumped viscosity-weighted pressure mass.**
The exact Schur complement \f$S = -B A^{-1} B^T\f$ is dense and expensive. For a
Stokes problem with variable viscosity the standard approximation is
\f$S \approx -M_p(\eta^{-1})\f$, i.e. the pressure mass matrix weighted by the inverse
viscosity \f$\eta^{-1}\f$. This captures the viscosity dependence and is the
approximation used here (`KMass` with coefficient \f$k = \eta^{-1}\f$).

The approximation is then further simplified to a **lumped (row-summed) diagonal**:
\f$\hat{S}^{-1} \approx \mathrm{diag}(M_p(\eta^{-1}))^{-1}\f$.
This is a single scalar per pressure DOF, cheap to apply, and implemented as
`linalg::solvers::DiagonalSolver<KMass>`.

> **Note:** while the viscosity-weighted scaling is physically motivated and significantly
> better than a pure mass matrix, this preconditioner is not yet optimal. An improved
> approximation would invert the full (non-lumped) weighted pressure mass matrix, which
> would require a PCG iteration on the pressure block instead of a simple diagonal scaling.

### Parameter guidelines

**FGMRES**

| Parameter          | Steady-state | Time-stepping | Notes                                                                                                                                                                                            |
|--------------------|--------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| restart \f$m\f$    | 30           | 10            | Restart allocates \f$2m+4\f$ combined velocity–pressure vectors. Keep small for time-stepping; increase for ill-conditioned steady-state problems. Should be smaller/equal number of iterations. |
| max iterations     | 100–200      | 10–20         | With a warm start from the previous time step, convergence is fast and few iterations suffice.                                                                                                   |
| relative tolerance | 1e-6         | 1e-4 – 1e-6   | Tightening below 1e-8 is unlikely to massively improve the physical result.                                                                                                                      |
| absolute tolerance | 1e-12        | 1e-12         | Guards against stagnation when the residual is already very small. Applies to double precision ("close" to unit roundoff).                                                                       |

**Multigrid preconditioner (velocity block)**

| Parameter                                     | Possibly sensible range | Notes                                                                                                                                        |
|-----------------------------------------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| V-cycles per FGMRES iteration                 | 1-3                     | The outer FGMRES compensates for the inexact inversion. Typically discouraged to increase beyond a few iterations.                           |
| Chebyshev order \f$p\f$                       | 2-3                     | Order 1 is equivalent to weighted Jacobi. Order 2–3 gives better smoothing per iteration. Higher orders increase the matvec count per sweep. |
| Pre/post-smoothing steps                      | 2                       | Try increasing to 3–4 if convergence is slow. Note that the number of matvecs multiplies with \f$p\f$.                                       |
| Power iterations for \f$\hat\lambda_{\max}\f$ | 10-50                   | Sufficient for a few-percent estimate; the 1.5× safety factor absorbs the remaining error. Try increasing if the smoother diverges.          |

---

## Boundary conditions

For a general description of boundary condition enforcement in terraneo see the
[Boundary Conditions](#boundary-conditions) chapter.

Velocity boundary conditions are specified via the `BoundaryConditions` struct, which
maps each shell boundary (`CMB`, `SURFACE`) to one of:

- `DIRICHLET` (**no-slip**): the velocity is prescribed to zero. Enforced strongly
  (algebraic row replacement) via `fe::strong_algebraic_dirichlet_enforcement`.
- `FREESLIP`: the normal velocity component is zero and the tangential components are
  unconstrained (free-slip condition).
- `NEUMANN`: boundary rows are left untouched in the operator.

Two operator instances are typically constructed: one with the physical BCs (`K`, used
in the solver) and one with full Neumann BCs (`K_neumann`, used as the input to
`TwoGridGCA` when assembling the coarse-grid operators — ensuring the GCA triple product
reflects the unconstrained stiffness rather than the BC-modified rows).

---

## Compressible and anelastic extensions {#stokes-compressible}

The goal is to approximate the full compressible mass conservation
\f[
\nabla\cdot(\rho\,\mathbf{u}) = 0
\f]
without changing the left-hand side operators in the formulation at the top of this page.
Specific formulations (e.g. anelastic) then introduce a reference-state density \f$\bar\rho\f$.
For anelastic flow the mass conservation constraint is modified, but the momentum operator
and the entire solver infrastructure stay the same. Only the right-hand side of the
pressure equation gains a non-zero term \f$\mathbf{g}\f$:

\f[
\begin{pmatrix} A & B^T \\ B & 0 \end{pmatrix}
\begin{pmatrix} \mathbf{u} \\ \mathbf{p} \end{pmatrix}
=
\begin{pmatrix} \mathbf{f} \\ \mathbf{g} \end{pmatrix}.
\f]

\f$A\f$, \f$B\f$, \f$B^T\f$, and the block-triangular preconditioner are unchanged
(the Schur complement \f$S = -BA^{-1}B^T\f$ is independent of \f$\mathbf{g}\f$).

### Truncated anelastic liquid approximation (TALA) and the frozen-velocity approach

> **Following [Ilangovan et al. (2026)](https://doi.org/10.5194/gmd-19-1455-2026). Consult the paper for details and
> derivations.**


The frozen velocity approach applied to the truncated anelastic approximation (TALA) replaces
\f$-\nabla\cdot\mathbf{u} = 0\f$ with:
\f[
- \nabla\cdot\mathbf{u} = \frac{\nabla \rho}{\rho} \cdot \mathbf{u}^{\text{(old)}},
\f]
introducing a linearization that results in the left-hand side of the mass conservation equation
being the same as in the incompressible case. The downside of this linearization is that
the right-hand-side velocity is 'frozen' and typically taken from the previous time step,
or improved via an outer iteration over the Stokes solver.

The linear form that evaluates the right-hand side of the pressure equation is
\f[
\mathbf{g}^\text{TALA}_i = \int_\Omega \frac{1}{\rho} \nabla\rho \cdot \mathbf{u} \, \phi_i \, \mathrm{d}x,
\f]
and implemented in `terra::fe::wedge::linearforms::shell::InvRhoGradRhoDotU`.

### Projected density approximation (PDA)

> **Following [Gassmöller et al. (2020)](https://doi.org/10.1093/gji/ggaa078). Consult the paper for details and
> derivations.**

Therein, compressible flow is approximated by adding another term to the right-hand side of the mass conservation
equation:
\f[
-\nabla\cdot\mathbf{u} = \underbrace{\frac{1}{\rho} \frac{\partial \rho}{\partial t}}_{\text{new term}} \quad + \quad
\underbrace{\frac{\nabla \rho}{\rho} \cdot \mathbf{u}^{\text{(old)}}}_{\text{see TALA}}.
\f]
The new term involves a time derivative of the density \f$\rho\f$ that has to be approximated (via Euler or BDF2 for
example) before the linear form is evaluated.

The full right-hand side is evaluated via the PDA linear form:
\f[
\mathbf{g}^\text{PDA}_i = \mathbf{g}^\text{TALA}_i + \int_\Omega \frac{1}{\rho} \dot\rho \, \phi_i \, \mathrm{d}x.
\f]
The second term is implemented in `terra::fe::wedge::linearforms::shell::InvRhoDrhoDt`.
