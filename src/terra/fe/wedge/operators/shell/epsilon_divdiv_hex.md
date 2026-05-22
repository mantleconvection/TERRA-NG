# Hex 1-point Gauss EpsDivDiv DN kernel — design plan

**Target:** add a third Dirichlet/Neumann path that integrates the ε(u):ε(v) + (-2/3) tr(ε)tr(δε) form **on the trilinear hex** with a **single Gauss point at the centroid**, instead of subdividing each hex into two Felippa-1×1 wedges. The wedge path (`FastDirichletNeumann`) and the wave-parallel wedge path (`FastDirichletNeumannWave`) stay in place; the new path is reachable via a new `KernelPath::FastDirichletNeumannHex` enum value so all three are runtime-selectable.

---

## 0. Why this design

The wedge wave kernel is **latency-bound, not bandwidth-bound** (HBM 12 % utilized, L2 95 % hit). Its hot path:

| Item | Wedge wave path | Hex 1-pt path |
|---|---:|---:|
| Quadrature points per cell | 2 (one per wedge) | 1 |
| Active lanes per cell | 6 of 8 (25 % SIMD waste) | 8 of 8 |
| Wedge outer loop iterations | 2 (sequential) | 0 |
| `wave_reduce8` calls per cell | 12 (6 acc × 2 wedges) | 6 (6 acc × 1 quadpt) |
| atomic_add scatter ops per cell | 12 (6 nodes × 2 wedges) | 8 (8 corners × 1 quadpt) |
| Per-lane `WEDGE_NODE_OFF` lookups | constant table, divergent | none (corner id = lane) |
| Persistent boundary state | 2 ints (`cmb_shift`/`surface_shift`) | 1 byte (`bnd_mask`) |

Cost savings are concentrated where the wedge wave kernel is slow: lane-divergent table lookups during gather, cross-lane reductions, and atomic traffic. Tradeoff: 1-pt Gauss is **only O(h²) consistent on distorted (non-affine) hexes**, and the operator is no longer the assembly-equivalent of the wedge-subdivided one. This is a **separate kernel**, not a replacement.

---

## 1. Lane / cell / wave mapping

Each wave (64 lanes) processes a 16-cell radial pencil at fixed `(x_cell, y_cell)`, in 2 sequential 8-cell blocks. Same outer geometry as the wave wedge path. Difference is **inside one block**:

```
Wave (64 lanes), block 0 of 2 — 8 hex cells, 8 lanes each:

  lane:  0 1 2 3 4 5 6 7  | 8 9 ...15 | 16..23 | 24..31 | 32..39 | 40..47 | 48..55 | 56..63
  cell:  c=0              | c=1       |  c=2   |  c=3   |  c=4   |  c=5   |  c=6   |  c=7
  node:  0 1 2 3 4 5 6 7  | 0 1 ... 7 |  ...   |  ...   |  ...   |  ...   |  ...   |  ...
                ↑↑↑↑↑↑↑↑    8 hex corners — all active (no idle lanes)
```

Lane id → `(cell_in_block, node_in_cell)`:
```cpp
const int cell_in_block = lane_id / 8;       // [0, 8)
const int node_in_cell  = lane_id % 8;       // [0, 8) — all active
```

The 8 hex corner ids `node_in_cell ∈ {0..7}` decode as `(dxn, dyn, dzn) = (n & 1, (n >> 1) & 1, (n >> 2) & 1)`. This matches the standard trilinear hex node order on `[0,1]³` ref:

| node | dxn | dyn | dzn | ξ | η | ζ |
|---:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 0 | 0 | 0 | −1 | −1 | −1 |
| 1 | 1 | 0 | 0 | +1 | −1 | −1 |
| 2 | 0 | 1 | 0 | −1 | +1 | −1 |
| 3 | 1 | 1 | 0 | +1 | +1 | −1 |
| 4 | 0 | 0 | 1 | −1 | −1 | +1 |
| 5 | 1 | 0 | 1 | +1 | −1 | +1 |
| 6 | 0 | 1 | 1 | −1 | +1 | +1 |
| 7 | 1 | 1 | 1 | +1 | +1 | +1 |

(The wedge kernel uses `WEDGE_NODE_OFF[w][node_in_cell]` lookup tables to map a wedge-local node to (dxn, dyn, dzn); here it is direct bit-decode.)

The outer block loop and the per-cell logic stay identical to the wave wedge path; only the per-cell body is different.

---

## 2. Quadrature: 1-pt Gauss at the hex centroid

On the reference hex `[-1, 1]³`, the 1-point Gauss rule is:

```
∫_{[-1,1]³} f(ξ, η, ζ) dξ dη dζ ≈ 8 · f(0, 0, 0)
```

So **all evaluations happen at the centroid** `(ξ, η, ζ) = (0, 0, 0)`. The eight shape-function gradients at the centroid have the closed form:

```
dN_i / dξ_j  |_centroid  =  (1/8) · sign(ξ_j for node i)  ·  (constants outside that direction)
                        =  ±1/4  for all 8 nodes × 3 directions
```

Explicitly:

```cpp
// node ordering matches the (dxn, dyn, dzn) bit decode above
constexpr double dN_ref[8][3] = {
    // dN/dξ                dN/dη                dN/dζ
    { -0.25, -0.25, -0.25 },  // node 0  ξ=-1, η=-1, ζ=-1
    {  0.25, -0.25, -0.25 },  // node 1  ξ=+1
    { -0.25,  0.25, -0.25 },  // node 2  η=+1
    {  0.25,  0.25, -0.25 },  // node 3
    { -0.25, -0.25,  0.25 },  // node 4  ζ=+1
    {  0.25, -0.25,  0.25 },  // node 5
    { -0.25,  0.25,  0.25 },  // node 6
    {  0.25,  0.25,  0.25 }   // node 7
};
```

The integration weight on `[-1, 1]³` is 8. Mapping reference `[-1, 1]³ → physical` uses the Jacobian at the centroid; the cell volume is `|J(0,0,0)| · 8`, so the per-cell integration weight is `kwJ = k_eval · |J| · 8`.

Equivalently, mapping `[0, 1]³ → physical` uses ref weight 1 with the same physical Jacobian and `dN_ref` entries become `±0.5` instead of `±0.25`. We keep `[-1, 1]³` to match standard Q1 references; either convention gives the same product `(kwJ × dN_ref²)` so the bilinear form is invariant.

---

## 3. Jacobian at the hex centroid

The physical position is `x(ξ, η, ζ) = Σ_i N_i(ξ, η, ζ) · X_i`, with 8 corners `X_i`. On the shell, the 8 corners are:

```
X_{dxn, dyn, dzn} = r_{dzn} · ĉ_{dxn, dyn}
```

where `ĉ_{dxn, dyn}` is the unit-sphere lateral position (4 lateral corners, same on top and bottom faces of the cell). The cell sits between two radial heights `r_0` and `r_1`.

Define:
```
r_mid    = (r_0 + r_1) / 2
half_dr  = (r_1 - r_0) / 2
ĉ_avg   = (ĉ_00 + ĉ_10 + ĉ_01 + ĉ_11) / 4    // lateral centroid
ĉ_ξ     = (-ĉ_00 + ĉ_10 - ĉ_01 + ĉ_11) / 4   // ξ-derivative of lateral
ĉ_η     = (-ĉ_00 - ĉ_10 + ĉ_01 + ĉ_11) / 4   // η-derivative of lateral
ĉ_ξη    = ( ĉ_00 - ĉ_10 - ĉ_01 + ĉ_11) / 4   // bilinear correction (unused at centroid)
```

At `(ξ, η, ζ) = (0, 0, 0)` the trilinear hex Jacobian is:

```
J(:, 0) = ∂x/∂ξ  = r_mid · ĉ_ξ
J(:, 1) = ∂x/∂η  = r_mid · ĉ_η
J(:, 2) = ∂x/∂ζ  = half_dr · ĉ_avg
```

This is exactly the standard "shell-of-revolution" Jacobian we already use for the wedge — the lateral part scales with `r_mid`, the radial part scales with `half_dr`, and the lateral centroid `ĉ_avg` is the radial direction at the cell's center.

```cpp
// Cell-wide (same on all 8 lanes)
const double r_0 = r_sh(cell_in_wave);
const double r_1 = r_sh(cell_in_wave + 1);
const double half_dr = 0.5 * (r_1 - r_0);
const double r_mid   = 0.5 * (r_0 + r_1);

// Lateral centroid and ξ/η derivatives, broadcast to all 8 lanes of the cell.
const double cx_avg = 0.25 * (coords_sh(0,0) + coords_sh(1,0) + coords_sh(2,0) + coords_sh(3,0));
const double cy_avg = 0.25 * (coords_sh(0,1) + coords_sh(1,1) + coords_sh(2,1) + coords_sh(3,1));
const double cz_avg = 0.25 * (coords_sh(0,2) + coords_sh(1,2) + coords_sh(2,2) + coords_sh(3,2));

const double cx_xi = 0.25 * (-coords_sh(0,0) + coords_sh(1,0) - coords_sh(2,0) + coords_sh(3,0));
// ... cy_xi, cz_xi, cx_eta, cy_eta, cz_eta analogous

const double J_0_0 = r_mid  * cx_xi;
const double J_0_1 = r_mid  * cx_eta;
const double J_0_2 = half_dr * cx_avg;
// ... J_1_*, J_2_* analogous, 9 entries total

const double J_det = /* 6-term expansion as in wedge code */;
const double inv_det = 1.0 / J_det;
```

Note: corners are stored in `coords_sh` in row-major order `n = dxn + 2*dyn`, so `coords_sh(0,*) = ĉ_00`, `coords_sh(1,*) = ĉ_10`, `coords_sh(2,*) = ĉ_01`, `coords_sh(3,*) = ĉ_11`.

`k_eval` (per-cell coefficient): 1-pt Gauss takes the centroid value, which we approximate as the cell-corner average:
```cpp
const int knid = (node_in_cell & 1) + 2 * ((node_in_cell >> 1) & 1);   // 0..3
const int klvl = cell_in_wave + ((node_in_cell >> 2) & 1);             // top or bottom
const double my_k = active ? k_sh(knid, klvl) : 0.0;
const double k_eval = 0.125 * wave_reduce8(my_k);                      // 8-corner average
const double kwJ = k_eval * J_det * 8.0;                               // 1-pt Gauss weight = 8 on [-1,1]³
```

---

## 4. Physical gradient per lane

```cpp
const double gx = dN_ref[node_in_cell][0];   // ±0.25
const double gy = dN_ref[node_in_cell][1];
const double gz = dN_ref[node_in_cell][2];

// J^{-T} · grad_ref  (cofactor formula, same expressions as wedge)
g0 = inv_det * (( J_1_1*J_2_2 - J_1_2*J_2_1) * gx + (-J_1_0*J_2_2 + J_1_2*J_2_0) * gy + ( J_1_0*J_2_1 - J_1_1*J_2_0) * gz);
g1 = inv_det * ((-J_0_1*J_2_2 + J_0_2*J_2_1) * gx + ( J_0_0*J_2_2 - J_0_2*J_2_0) * gy + (-J_0_0*J_2_1 + J_0_1*J_2_0) * gz);
g2 = inv_det * (( J_0_1*J_1_2 - J_0_2*J_1_1) * gx + (-J_0_0*J_1_2 + J_0_2*J_1_0) * gy + ( J_0_0*J_1_1 - J_0_1*J_1_0) * gz);
```

Identical expressions to the wedge kernel — just one of the 8 nodes per cell, not one of 6.

---

## 5. Gather + reduce + scatter

```cpp
// Gather source value at this lane's corner
const int snid = (node_in_cell & 1) + 2 * ((node_in_cell >> 1) & 1);
const int slvl = cell_in_wave + ((node_in_cell >> 2) & 1);
const double s0 = active_and_in_range ? src_sh(snid, 0, slvl) : 0.0;
const double s1 = active_and_in_range ? src_sh(snid, 1, slvl) : 0.0;
const double s2 = active_and_in_range ? src_sh(snid, 2, slvl) : 0.0;

// Per-lane partials
const double p_gu00 = g0 * s0;
const double p_gu10 = 0.5 * (g1 * s0 + g0 * s1);
const double p_gu11 = g1 * s1;
const double p_gu20 = 0.5 * (g2 * s0 + g0 * s2);
const double p_gu21 = 0.5 * (g2 * s1 + g1 * s2);
const double p_gu22 = g2 * s2;

// 6 wave_reduce8 calls — same as wedge but div_u falls out of the trace:
const double gu00 = wave_reduce8(p_gu00);
const double gu10 = wave_reduce8(p_gu10);
const double gu11 = wave_reduce8(p_gu11);
const double gu20 = wave_reduce8(p_gu20);
const double gu21 = wave_reduce8(p_gu21);
const double gu22 = wave_reduce8(p_gu22);
const double div_u = gu00 + gu11 + gu22;

// Scatter: each lane writes 3 atomic_adds to its corner
if (active_and_in_range) {
    Kokkos::atomic_add(&dst_(s, x_cell + dxn, y_cell + dyn, r_cell + dzn, 0),
                       kwJ * (2.0 * (g0*gu00 + g1*gu10 + g2*gu20) + NEG_TWO_THIRDS * g0 * div_u));
    Kokkos::atomic_add(&dst_(s, x_cell + dxn, y_cell + dyn, r_cell + dzn, 1),
                       kwJ * (2.0 * (g0*gu10 + g1*gu11 + g2*gu21) + NEG_TWO_THIRDS * g1 * div_u));
    Kokkos::atomic_add(&dst_(s, x_cell + dxn, y_cell + dyn, r_cell + dzn, 2),
                       kwJ * (2.0 * (g0*gu20 + g1*gu21 + g2*gu22) + NEG_TWO_THIRDS * g2 * div_u));
}
```

---

## 6. Boundary handling

The wedge kernel treats CMB/SURFACE by **shifting the node range** (`cmb_shift`/`surface_shift`) to exclude wedge nodes that lie on a Dirichlet boundary face. In the hex layout boundary handling is **per-corner**:

- CMB boundary applies to the 4 corners with `dzn = 0` (the cell's bottom face).
- SURFACE boundary applies to the 4 corners with `dzn = 1` (the cell's top face).

```cpp
const bool at_cmb     = cell_valid && has_flag(s, x_cell, y_cell, r_cell,     CMB);
const bool at_surface = cell_valid && has_flag(s, x_cell, y_cell, r_cell + 1, SURFACE);
const bool at_boundary = at_cmb || at_surface;

bool treat_boundary_dirichlet = false;
if (at_boundary) {
    const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
    treat_boundary_dirichlet = (get_boundary_condition_flag(bcs_, sbf) == DIRICHLET);
}

const int dzn = (node_in_cell >> 2) & 1;
const bool node_on_cmb_face     = (dzn == 0);
const bool node_on_surface_face = (dzn == 1);

// matvec contribution skipped on Dirichlet rows (consistent with wedge cmb_shift/surface_shift logic)
const bool matvec_node_in_range =
    active && cell_valid &&
    !( treat_boundary_dirichlet && !Diagonal && at_cmb     && node_on_cmb_face) &&
    !( treat_boundary_dirichlet && !Diagonal && at_surface && node_on_surface_face);

// diagonal / boundary-Dirichlet branch uses the complement (only Dirichlet rows contribute identity)
const bool diag_node_in_range =
    active && cell_valid &&
    (Diagonal ||
     ( treat_boundary_dirichlet &&
       ((at_cmb && node_on_cmb_face) || (at_surface && node_on_surface_face))));
```

Same predication shape as wedge, but the test is `dzn == 0/1` instead of `node_in_cell >= cmb_shift` / `node_in_cell < 6 - surface_shift`.

---

## 7. Per-cell op count comparison

| Op | Wedge wave path (2 wedges) | Hex 1-pt | Δ |
|---|---:|---:|---:|
| Jacobian builds | 2 (one per wedge) | 1 | -1× |
| `J_det` + `inv_det` | 2 | 1 | -1× |
| `kwJ` constructions | 2 | 1 | -1× |
| Phys-gradient `g0,g1,g2` | 2 per lane | 1 per lane | -1× |
| Per-lane partial accumulators | 6 × 2 = 12 | 6 × 1 = 6 | -1× |
| `wave_reduce8` calls | 12 | 6 | -1× |
| `atomic_add` scatters per cell | 12 | 8 | -1.5× |

Roughly halves the per-cell ALU and reduces scatter traffic by ~33 %. Memory traffic stays similar (same coords/src/k loads per cell stack), so the kernel becomes more strongly latency-bound — if anything, even more dependent on hiding LDS read latencies.

Per-cell shared-coords usage drops: lateral averages `ĉ_avg/ĉ_ξ/ĉ_η` are derived from the same 4 corner coords already in `coords_sh`. No new LDS needed.

---

## 8. Files & integration

| File | Action |
|---|---|
| `epsilon_divdiv_hex.hpp` (new) | `kHex*` constants, `team_shmem_size_dn_hex()`, `run_team_fast_dirichlet_neumann_hex<Diagonal>` member |
| `epsilon_divdiv_hex.md` (this file) | Design plan |
| `epsilon_divdiv_kerngen.hpp` | Add `KernelPath::FastDirichletNeumannHex`; extend path-name strings; add dispatch block (LB<64, 1>, vector_length=64); `#include "epsilon_divdiv_hex.hpp"` next to the wave include |

### 8.1 Constants

```cpp
static constexpr int kHexCellsPerBlock = 8;
static constexpr int kHexBlocksPerWave = 2;
static constexpr int kHexCellsPerWave  = kHexCellsPerBlock * kHexBlocksPerWave;   // 16
static constexpr int kHexLanesPerCell  = 8;
static constexpr int kHexActiveNodes   = 8;   // no padding
static constexpr int kHexAccumsPerCell = 6;   // gu00, gu10, gu11, gu20, gu21, gu22 (div_u from trace)
```

### 8.2 LDS

Reuses the exact wave LDS layout (`coords_sh[4][3]`, `src_sh[4][3][17]`, `k_sh[4][17]`, `r_sh[17]`).
`team_shmem_size_dn_hex()` returns the same byte count as `team_shmem_size_dn_wave()` (≈ 1.3 KB / team). Could alias the same function; we keep a distinct name to leave room for divergence later.

### 8.3 Dispatch in `apply_impl`

```cpp
else if (kernel_path_ == KernelPath::FastDirichletNeumannHex) {
    const int r_stacks = (hex_rad_ + kHexCellsPerWave - 1) / kHexCellsPerWave;
    const int hex_blocks = local_subdomains_ * hex_lat_ * hex_lat_ * r_stacks;
    using LB = Kokkos::LaunchBounds<64, 1>;
    Kokkos::TeamPolicy<LB> hx_policy(hex_blocks, /*team_size=*/1, /*vector_length=*/64);
    hx_policy.set_scratch_size(0, Kokkos::PerTeam(team_shmem_size_dn_hex()));
    if (diagonal_) {
        Kokkos::parallel_for(
            "epsilon_divdiv_apply_kernel_fast_dn_hex_diag", hx_policy,
            KOKKOS_CLASS_LAMBDA(const Team& team) {
                this->template run_team_fast_dirichlet_neumann_hex<true>(team);
            });
    } else {
        Kokkos::parallel_for(
            "epsilon_divdiv_apply_kernel_fast_dn_hex_matvec", hx_policy,
            KOKKOS_CLASS_LAMBDA(const Team& team) {
                this->template run_team_fast_dirichlet_neumann_hex<false>(team);
            });
    }
}
```

`update_kernel_path_flag_host_only()` is **not** changed — the hex path is opt-in only (`set_kernel_path(KernelPath::FastDirichletNeumannHex)`), same convention as the wave path. The benchmark already supports `EPSDIVDIV_HEX=1` style env-var dispatch.

---

## 9. Per-lane VGPR forecast

Predicted live state at the inner-loop peak:

- `g0, g1, g2` (3) + `gx, gy, gz` (3) = 6 doubles brief → ~12 VGPRs
- `J_0_0..J_2_2` (9 doubles, J_det, inv_det) → ~22 VGPRs (drop after gradient)
- `s0, s1, s2` (3 doubles, brief) → 6 VGPRs
- `kwJ`, `r_0`, `r_1`, `half_dr`, `r_mid` (5 doubles) → 10 VGPRs
- `gu00..gu22, div_u` (7 doubles, live during scatter) → 14 VGPRs
- 4–6 ints for `cell_in_block`, `node_in_cell`, `dxn/dyn/dzn`, `r_cell` → 6 VGPRs

**Best-case estimate: ~60 VGPRs** (no wedge loop ⇒ second Jacobian doesn't co-exist). If we hit ≤ 64 VGPRs naturally we get 8 waves/SIMD without any LaunchBounds pressure — the wave path needed `LB<64, 8>` to force this, at a 17 % perf cost. The hex path should reach 8 waves/SIMD organically.

Even with a `LaunchBounds<64, 1>` (no cap) start, the lack of wedge loop alone removes the long-lived `inv_det` that v62/v66 occupied in the wave viz.

---

## 10. Validation plan

1. **Standalone numerical sanity**: small problem (level 4, sdr 0, 1 cell laterally), DN-only BCs, compare hex path against wedge path. Expect: **NOT bitwise equal** (different operator). Sanity check: row sums of operator close to zero in interior, residual norm decreases under CG with hex operator.
2. **Manufactured solution**: choose a smooth `u = (r-r_min)(r_max-r) ê_r` and verify `Op(u)` matches the analytic ε(u):ε(v) form on a uniform mesh. Error should scale as O(h²) under refinement.
3. **A3 hackathon benchmark**: 5 timesteps with `kernel_path_ = FastDirichletNeumannHex`. Compare Nusselt-number trace against the wedge path on the same mesh. **Acceptance**: |Δ Nu / Nu| < 10⁻² over 5 timesteps on level 8.
4. **Perf measurement**: `benchmark_operators --refinement-level-subdomains 0 --min-level 8 --max-level 8 --executions 5`. Target: ≥ 15 % speedup vs wave path baseline (102 ms / 4.65 GDoFs/s); stretch 30 %.

---

## 11. Risks and mitigations

- **Accuracy:** 1-pt Gauss is only O(h²) consistent on distorted hex. Mantle elements are mildly distorted near subdomain boundaries; behavior in solver convergence needs verification before this path is used in production. **Mitigation:** treat as a separate operator; user opts in. If solver iterations grow more than ~20 %, the wall-clock win is gone and we revert.
- **div_u via trace identity:** `div_u = gu00 + gu11 + gu22` is exact at any single quadrature point. Same identity already used in wedge code — no new approximation introduced here.
- **k_eval averaging:** the wedge wave kernel computes `k_eval` as the wedge-node average (6 lanes → ×1/6). Here we use the corner average (8 lanes → ×1/8). Standard Q1 practice; matches how `k` is treated in most Q1 mantle codes.
- **Compiler may not actually drop to 64 VGPRs:** even without the wedge loop, the inv_det / cofactor matrix could persist into scatter if the compiler decides to. **Mitigation:** explicit scoping (Jacobian + cofactors in a brace block; only g0/g1/g2 leak out), same pattern as the wave kernel's `// J_*_*, J_det, ... dead` comment.
- **8 atomic_adds per cell (vs 12)** is still a lot of HBM-coherent traffic. **Mitigation:** none in this revision. A future revision could buffer the per-cell row contributions to LDS and do one global flush per wave (`accum_sh[cells_per_wave][8 corners][3]` ≈ 3 KB / team).

---

## 12. Implementation milestones

1. Write `epsilon_divdiv_hex.hpp` with constants + scratch sizing + `run_team_fast_dirichlet_neumann_hex<Diagonal>` body.
2. Add `KernelPath::FastDirichletNeumannHex` to the enum and update the two path-name string switches.
3. Add dispatch block in `apply_impl()` (alongside the existing wave block).
4. Build under `terraneo-build-therock` (RelWithDebInfo). Confirm compile success and read off VGPR/SGPR/spill from `-Rpass-analysis=kernel-resource-usage`.
5. Run benchmark_operators at level 8, sdr 0 with `kernel_path_ = FastDirichletNeumannHex` and record matvec ms / GDoFs/s.
6. If perf is on target, run rocprofv3 `--att` and rocprof-compute to confirm whether the kernel is more bandwidth-bound (success criterion: HBM utilization > 30 % vs wave's 12 %).
7. Accuracy sanity (milestones 10.1–10.2 above) **before** quoting any speed number.

If milestone 4 fails (e.g. VGPRs > 70 or spills appear), the most likely culprit is the cofactor matrix surviving into scatter — refactor to scope `J_*` + `inv_det` tighter before declaring failure.
