# Entropy-viscosity stabilization for the energy equation

This document describes the entropy-viscosity (EV) stabilization scheme used by
the `mantlecirculation` app as an alternative to SUPG, and how it is wired into
the energy-equation solver.

## 1. Background and recipe

EV replaces SUPG's *anisotropic, always-on* streamline diffusion with an
*isotropic, residual-driven* artificial diffusion `ν_h(x)`:

* In smooth regions the entropy residual `r_E ≈ 0` and no diffusion is added.
* Near sharp features (boundary layers, plumes), `r_E` is large and `ν_h` is
  capped from above by a first-order upwind bound, so the scheme adds at most
  the diffusion of an upwind-stabilised first-order method.

We use the formulation of Kronbichler, Heister & Bangerth (KHB),
*Geophysical Journal International* **191**, 12–29 (2012), which itself adopts
Guermond, Pasquetti & Popov, *J. Comput. Phys.* **230**, 4248–4267 (2011).
The implementation evaluates `ν_h` **per wedge** (two wedges per hex cell) at
the same Felippa 3×2 quadrature points used by the rest of the wedge FE
operator suite:

```
E(T)     = ½ (T − T_m)²,                T_m = ½(T_min + T_max)
r_E|_w   = ‖∂_t E + (T − T_m)·(u·∇T − κ∇²T − γ)‖_{∞, w}
E_avg    = (1/|Ω|) ∫_Ω E(T)
D        = ‖E − E_avg‖_{∞, Ω}
ν_h|_w   = min( α_max · h_w · ‖u‖_{∞,w},
                α_E   · h_w² · r_E|_w / D )
α_max    = 0.026 · d  (= 0.078 in 3D)
α_E      = 1.0
h_w      = V_wedge^(1/3)   (wedge volume, integrated by the same quadrature)
```

The residual is **time-lagged**: `∂_t E` and the corner field values use
`T^{n−1}` and `T^{n−2}`, so `ν_h` is fully determined from past states and the
implicit solve for `T^{n+1}` stays linear. This is the IMPES-style "explicit
stabilisation" used by ASPECT.

The resulting `ν_h_w` is the per-wedge piecewise-constant coefficient of a
Galerkin `∇·(ν_h ∇T^n)` term added explicitly to the RHS of the
otherwise pure-Galerkin advection–diffusion solve, applied by the
`WedgeConstantDivKGrad` operator.

## 2. Files

| File | Role |
|---|---|
| `src/terra/fe/wedge/operators/shell/entropy_viscosity.hpp` | EV kernels: stats, per-wedge ν_h |
| `src/terra/fe/wedge/operators/shell/wedge_constant_div_k_grad.hpp` | `∇·(ν_h ∇·)` with per-wedge piecewise-constant ν |
| `src/terra/fe/wedge/operators/shell/wedge_lumped_lap_projector.hpp` | Per-wedge lumped-mass projection of κ·∇²T |
| `src/terra/fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp` | AD operator with a runtime toggle to disable SUPG |
| `apps/mantlecirculation/src/energy_solver.hpp` | `EVSolver` class — assembles the timestep |
| `apps/mantlecirculation/src/parameters.hpp` | CLI option mapping for `--energy-solver ev` |
| `apps/mantlecirculation/mantlecirculation.cpp` | Picks the energy solver via polymorphic dispatch |

## 3. Kernel surface (entropy_viscosity.hpp)

Two free functions, both in `terra::fe::wedge::operators::shell`:

### `compute_entropy_stats`

Computes the global scalars `T_min`, `T_max`, `T_m`, `E_avg`, `D` used by the
ν_h formula. `EntropyStats<ScalarT>` is the return type.

* `T_min` / `T_max` use ownership-masked min/max reductions over owned Q1 nodes
  so halo nodes do not double-count. Two helper reductions
  (`max_entry_owned` / `min_entry_owned`) are provided locally because
  `kernels::common` lacks ownership-aware variants.
* `T_m` uses the global `[min, max]` range (not the volume-average), matching
  KHB 2012 — keeps the entropy symmetric around the midpoint of the physical
  range.
* `E_avg` is a true FE volume integral evaluated on the same Felippa 3×2
  quadrature as `compute_nu_h`:
  `E_avg = (Σ_w Σ_q w_q·|det J(q)|·E(q)) / (Σ_w Σ_q w_q·|det J(q)|)`,
  where `E(q) = ½(T(q) − T_m)²` with `T(q) = Σ_j N_j(q)·T_j`.
  Cells are not shared between ranks — only nodes are — so the local sums form
  a partition-of-unity at the integration level and the MPI sum is exact.
* `D` is the signed-magnitude ∞-norm of `(E − E_avg)` over owned nodes,
  guarded with a small floor `params.D_floor` so the residual branch does not
  blow up when T is near-constant.

### `compute_nu_h`

Computes the per-wedge artificial viscosity `ν_h_w`, one scalar per wedge
(`num_wedges_per_hex_cell = 2` per hex), into a
`Grid5DDataScalar<ScalarT>` of extents
`(num_subdomains, Nc_x, Nc_y, Nc_r, num_wedges_per_hex_cell)`.

For each wedge of each hex, at every Felippa 3×2 quad point:

* The full 3D Jacobian `J = ∂x/∂(ξ,η,ζ)` is evaluated via the existing
  `jac()`, with `J_inv_t = J.inv_transposed(det)`.
* Field interpolants `T(q), Tp(q), u(q)` are obtained via
  `Σ_j N_j(q)·field_j` over the 6 wedge nodes; `Lap(q)` is interpolated the
  same way from the per-wedge nodal lap supplied by the caller (see §4).
* The FE-consistent physical-space gradient is
  `grad_T(q) = J_inv_t · (Σ_j T_j · ∇_ξ N_j(q))`.
* Strong-form residual at the quad point (full ASPECT form):
  `r_E_q = |∂_t E + (T − T_m)·(u·∇T + Lap − γ)|`,
  with `Lap` already encoding `−κ∇²T` (lumped-mass-projected per wedge,
  see §4) and `γ` a constant scalar parameter.
* The wedge accumulates `r_E_max,w = max_q r_E_q`, `‖u‖_∞,w = max_q ‖u(q)‖`,
  and the wedge volume `V_wedge = Σ_q w_q·|det J(q)|`.
* Output: `ν_h_w = min(α_max·h_w·‖u‖_∞,w, α_E·h_w²·r_E_max,w / D)` with
  `h_w = V_wedge^(1/3)`.

The previously separate `project_nu_h_to_nodes` step is **gone**: the per-
wedge ν_h is consumed directly by `WedgeConstantDivKGrad`. There is no nodal
projection and therefore no halo exchange of the stabilization field.

## 4. Per-wedge lap projection (wedge_lumped_lap_projector.hpp)

`WedgeLumpedLapProjector<ScalarT>` precomputes, at construction time, a per-
wedge 6×6 projector
`P_w[i][j] = κ · K_w[i][j] / M_w_lumped[i]`,
where `K_w` is the wedge-local Galerkin Laplacian and `M_w_lumped` is the
row-sum lumping of the wedge-local consistent mass — both assembled with the
same Felippa 3×2 quadrature as the rest of the wedge FE suite.

At apply time `lap_w[i] = (P_w · T_w)[i]` for each wedge — strictly local,
no halo exchange. Output shape:
`Kokkos::View<dense::Vec<ScalarT, 6>****[num_wedges_per_hex_cell]>`.
`compute_nu_h` reads `lap_w(s,x,y,r,wedge)(j)` and interpolates
`Lap(q) = Σ_j N_j(q)·lap_w[wedge](j)` at each Felippa quad point.

Per-wedge lumping (rather than the global lumped mass shared between wedges)
keeps the projection a single-element local computation and lines up with
the per-wedge ν_h granularity. κ is folded into `P_w` at construction time;
extending to spatially varying κ is a matter of carrying it inside the
assembly kernel.

## 5. The `WedgeConstantDivKGrad` operator

`WedgeConstantDivKGrad<ScalarT>` (`wedge_constant_div_k_grad.hpp`) implements
`∇·(ν ∇·)` with a per-wedge piecewise-constant coefficient `ν_w`, taken as a
`Grid5DDataScalar<ScalarT>` of shape
`(num_subdomains, Nc_x, Nc_y, Nc_r, num_wedges_per_hex_cell)`.

For each wedge, the local 6×6 Galerkin Laplacian is assembled on the fly with
the same Felippa 3×2 quadrature + `jac().inv_transposed() * grad_shape()`
pattern used by `Laplace` / `DivKGrad`, scaled by `ν_w`, applied to the
gathered local `T_w`, and scattered additively. No boundary treatment, no
diagonal mode, no GCA storage — the operator is applied exactly once per
substep as an explicit RHS contribution.

## 6. Per-step assembly (EVSolver)

`EVSolver` is the concrete subclass of `EnergySolver` for the EV scheme; see
`apps/mantlecirculation/src/energy_solver.hpp`.

### Construction

* Allocates `nu_h_wedge_` (`Grid5DDataScalar`, last dim = 2 wedges/hex) for
  the per-wedge ν_h.
* Builds three `UnsteadyAdvectionDiffusionSUPG` operators (`A_`,
  `A_neumann_`, `A_neumann_diag_`) and calls `set_supg_enabled(false)` on
  each — this turns the LHS into a pure Galerkin advection–diffusion
  operator. The toggle lives at `unsteady_advection_diffusion_supg.hpp`
  (member, setter, and the spot where `streamline_diffusivity[wedge]` is
  forced to 0 when off).
* Builds the temperature mass `M_`.
* Builds the per-wedge lap projector
  `lap_projector_ = WedgeLumpedLapProjector(domain, coords_shell_,
  coords_radii_, κ)` — assembles `P_w` once.
* Builds `A_evdiff_ = WedgeConstantDivKGrad(domain, coords_shell_,
  coords_radii_, nu_h_wedge_)`. The coefficient view is captured by
  reference, so updates to `nu_h_wedge_` are picked up automatically.
* Bootstraps `T_prev_ = T` so the BDF1 history rotation gives `∂_t E = 0`
  on step 1.

### `step(dt, …)`

Per substep:

1. **Per-wedge nodal lap** `lap_projector_->apply(T_, lap_w_)`:
   `lap_w[wedge](j) = (κ K_w T_w / M_w_lumped)(j)`. Strictly local.
2. **Stats + per-wedge ν_h** —
   `compute_entropy_stats` (volume-weighted `E_avg`),
   then `compute_nu_h(nu_h_wedge_, T_, T_prev_, velocity_, lap_w_, …,
   gamma)` where `γ` is `prm.physics_parameters.constant_internal_heating_value`
   when `constant_internal_heating` is enabled, else 0.
3. **Explicit EV diffusion** `rhs_ev = A_evdiff_·T = ∫ ν_h ∇T·∇φ_i`
   using `WedgeConstantDivKGrad`.
4. **RHS assembly**  `q = M·T^n − dt · rhs_ev`.
5. **History rotation**  `T_prev_ ← T` *before* the solve overwrites `T`.
6. **Dirichlet BC vector + strong elimination**.
7. **Solve** `(M + dt · A_galerkin) · T^{n+1} = q` with FGMRES + diagonal
   preconditioner.

### Picard correctness

`step()` mutates both `T_` (the solve output) and `T_prev_` (via the history
rotation), so `EVSolver` overrides both `snapshot_for_picard` and
`restore_for_picard` to deep-copy and restore the *pair* `(T, T_prev_)`.
Restoring only `T` would leave `T_prev_` reflecting the post-step state of
the previous Picard iteration, biasing `∂_t E` and hence `ν_h` on the next
iteration. A flag `nu_h_locked_for_step_` (cleared by `snapshot_for_picard`,
set at the end of substep 0) gates the lap + ν_h recompute on Picard
iterations > 0 of substep 0 so all sweeps see the same explicit-lagged
stabilization. Substeps > 0 always recompute since `T` evolves between them.

## 7. Wiring & CLI

* Enum value: `EnergySolverType::ENTROPY_VISCOSITY` —
  `apps/mantlecirculation/src/parameters.hpp`.
* CLI mapping: both `entropy_viscosity` and `ev` resolve to
  `ENTROPY_VISCOSITY`.
* Dispatch site in `mantlecirculation.cpp`:

  ```cpp
  case EnergySolverType::ENTROPY_VISCOSITY:
      energy = std::make_unique< EVSolver< ScalarType > >( ... );
      break;
  ```

## 8. Notes / known limitations

* The whole stabilization pipeline is now halo-free below the `compute_nu_h`
  output: lap projection is local per wedge, ν_h lives per wedge, and the
  `WedgeConstantDivKGrad` apply is the only step that does the standard
  `CommunicateAdditively` reduction (same as every other Galerkin operator).
  The old `project_nu_h_to_nodes` halo exchange is gone.
* `compute_nu_h` uses BDF-1 lagging on `∂_t E`; a BDF-2 swap requires a
  third history slot and is straightforward.
* κ is constant across the domain in the current solver and folded into
  `P_w` at construction. Spatially varying κ requires either re-assembling
  `P_w` whenever κ changes or carrying κ through the projector kernel
  per wedge.
* `WedgeConstantDivKGrad` assembles `K_w` on the fly each apply; if profiles
  show this dominating, the same `LocalMatrixStorage` machinery used by
  `DivKGrad` can be added (per-wedge `Mat<6,6>` cache).
