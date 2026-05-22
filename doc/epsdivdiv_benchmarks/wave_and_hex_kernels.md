# Wave-parallel EpsDivDiv DN kernel — explicit design plan

**Target:** rewrite `operator_fast_dirichlet_neumann_path<Diagonal>` so that **one wavefront cooperatively processes 10 cells stacked radially**. The thread-per-cell v8 path stays in place; the new path is reachable via a new `KernelPath::FastDirichletNeumannWave` enum value so we can A/B them at runtime.

---

## 0. Why this specific design

- Current kernel is at 96–114 VGPRs / 4–5 waves/SIMD, bottlenecked by per-thread live state.
- Wave-per-cell with arbitrary cell packing (a 5- or 10-cell wave with no geometric correlation) suffers from heavy lane masking and no shared work.
- **Radial-pencil packing** (10 cells with the same `(x_cell, y_cell)`, differing only in `r_cell`) is the rare case where wave-per-element actually pays off:
  - All 10 cells share lateral coordinates → LDS becomes tiny (~1.6 KB / team).
  - The Jacobian's lateral pieces are identical across the 10 cells → can be computed once per wave and reused.
  - 60 of 64 lanes active (94% SIMD utilization).

---

## 1. Lane / cell / wave mapping (the central diagram)

```
Wave (64 lanes, processes 10 cells at radial levels r0..r0+9):

  lane:  0   1   2   3   4   5  |  6   7   8   9  10  11  |  ...  | 54 55 56 57 58 59 | 60 61 62 63
  cell:  c=0                    |  c=1                    |  ...  |  c=9              |  idle
  node:  0   1   2   3   4   5  |  0   1   2   3   4   5  |       |  0  1  2  3  4  5 |
```

Lane id → `(cell_in_wave, node_in_cell)`:
```cpp
const int cell_in_wave  = lane_id / 6;       // [0, 10)
const int node_in_cell  = lane_id % 6;       // [0, 6)
const bool active       = (lane_id < 60);
```

Each cell maps to a **6-lane contiguous group**. Lanes 60–63 are masked off via predication.

Wedges (`w=0,1`) are processed **sequentially** inside this 6-lane group (Option α from the previous discussion).

---

## 2. Host-side: tile geometry, team policy, dispatch

### 2.1 Tile parameters (new branch in the host-side update_kernel_path)

When `kernel_path_ == FastDirichletNeumannWave`:
```cpp
lat_tile_     = 1;            // one cell laterally per team
r_tile_       = 10;           // ten cells radially per wave
r_passes_     = 1;            // no radial striping (one wave = one full r-stack)
team_size_    = 1;            // one wave per team (initially; tune later)
vector_length = 64;           // wave-64
// block_size = team_size_ × vector_length = 64
```

The `team_size_` member's semantics differ here vs. the thread-per-cell path:
- In thread-per-cell path: `team_size_` = lat_tile_² · r_tile_ = number of *threads* per team.
- In wave-per-cell path: `team_size_` = number of *waves* per team. Each wave handles 10 cells.

Add a flag `kernel_uses_waves_` (or just gate everything on `kernel_path_ == FastDirichletNeumannWave`) so the host-side decode logic picks the right meaning.

### 2.2 Tile loop bounds

For a problem with `hex_lat × hex_lat × hex_rad` cells per subdomain, the number of teams needed:

```cpp
lat_tiles_  = hex_lat_;                                   // 1 lat cell per team
r_tiles_    = (hex_rad_ + r_tile_ - 1) / r_tile_;         // ceil(hex_rad / 10)
blocks_     = local_subdomains_ * lat_tiles_ * lat_tiles_ * r_tiles_;
```

So one team per **(subdomain, x_cell, y_cell, r_stack)**, where `r_stack` indexes a 10-cell radial pencil.

### 2.3 TeamPolicy dispatch (new branch in `apply_impl`)

```cpp
else if (kernel_path_ == KernelPath::FastDirichletNeumannWave) {
    using LB = Kokkos::LaunchBounds<64, 1>;     // 1 wave per block, no register cap
    Kokkos::TeamPolicy<LB> policy(blocks_, /*team_size=*/1, /*vector_length=*/64);
    policy.set_scratch_size(0, Kokkos::PerTeam(team_shmem_size_dn_wave()));
    if (diagonal_) {
        Kokkos::parallel_for(
            "epsilon_divdiv_apply_kernel_fast_dn_wave_diag", policy,
            KOKKOS_CLASS_LAMBDA(const Team& team) {
                this->template run_team_fast_dirichlet_neumann_wave<true>(team);
            });
    } else {
        Kokkos::parallel_for(
            "epsilon_divdiv_apply_kernel_fast_dn_wave_matvec", policy,
            KOKKOS_CLASS_LAMBDA(const Team& team) {
                this->template run_team_fast_dirichlet_neumann_wave<false>(team);
            });
    }
}
```

`LaunchBounds<64, 1>` because:
- 1st param = max block size = 64 (one wave). Tighter compiler hint than the default 1024.
- 2nd param = min waves/SIMD = 1. We *want* the compiler to use as many VGPRs as it needs.

---

## 3. LDS layout

Per team (one wave = 10 cells, all sharing `(x_cell, y_cell)`):

| Region | Size | Purpose |
|---|---:|---|
| `coords_sh[nxy=4][3]` | 96 B | Lateral coords of the 4 hex corners (cell shares lateral verts) |
| `r_sh[nlev=11]` | 88 B | Radial heights r_0 .. r_10 (10 cells need 11 r-values) |
| `src_sh[nxy=4][3][nlev=11]` | 1056 B | Source vector at the 4 lateral × 11 radial grid points |
| `k_sh[nxy=4][nlev=11]` | 352 B | Viscosity at the same grid points |
| `accum_sh[10][7]` *(new)* | 560 B | Per-cell 7-accumulator scratchpad for sub-wave reductions (see §5.3) |
| **Total** | **~2.2 KB / team** | |

Massive headroom — LDS is 64 KB/CU, so up to 28 teams fit per CU from the LDS side. Won't be LDS-bound.

`team_shmem_size_dn_wave()` returns:
```cpp
nxy = 4; nlev = 11; cells_per_wave = 10; n_accum = 7;
nscalars = nxy*3 + nxy*3*nlev + nxy*nlev + nlev + cells_per_wave*n_accum;
//       = 12   + 132          + 44       + 11   + 70   = 269 doubles = 2152 B
return sizeof(double) * nscalars;
```

---

## 4. Cooperative LDS load (entry of the wave kernel)

`team_size = 1` wave = 64 lanes. We load the 4 lateral × 11 radial = 44 grid points into LDS, parallelized across the 64 lanes:

```cpp
// One wave (64 lanes) cooperatively loads the team's 1×1×10-cell stack
Kokkos::parallel_for(ThreadVectorRange(team, nxy), [&](int n) {
    // Each of the first 4 lanes loads one corner's coords (rest idle for this step)
    const int dxn = n % 2, dyn = n / 2;       // n ∈ [0, 4): 4 corners
    const int xi = x_cell + dxn, yi = y_cell + dyn;
    if (xi <= hex_lat_ && yi <= hex_lat_) {
        coords_sh(n, 0) = grid_(local_subdomain_id, xi, yi, 0);
        coords_sh(n, 1) = grid_(local_subdomain_id, xi, yi, 1);
        coords_sh(n, 2) = grid_(local_subdomain_id, xi, yi, 2);
    } else {
        coords_sh(n, 0) = coords_sh(n, 1) = coords_sh(n, 2) = 0.0;
    }
});

Kokkos::parallel_for(ThreadVectorRange(team, nlev), [&](int lvl) {
    const int rr = r0 + lvl;
    r_sh(lvl) = (rr <= hex_rad_) ? radii_(local_subdomain_id, rr) : 0.0;
});

// src + k load: nxy * nlev = 44 work items, perfect for 44 of 64 lanes
const int total_pairs = nxy * nlev;
Kokkos::parallel_for(ThreadVectorRange(team, total_pairs), [&](int t) {
    const int node = t / nlev;
    const int lvl  = t % nlev;
    const int dxn  = node % 2, dyn = node / 2;
    const int xi = x_cell + dxn, yi = y_cell + dyn;
    const int rr = r0 + lvl;
    if (xi <= hex_lat_ && yi <= hex_lat_ && rr <= hex_rad_) {
        k_sh(node, lvl)        = k_(local_subdomain_id, xi, yi, rr);
        src_sh(node, 0, lvl)   = src_(local_subdomain_id, xi, yi, rr, 0);
        src_sh(node, 1, lvl)   = src_(local_subdomain_id, xi, yi, rr, 1);
        src_sh(node, 2, lvl)   = src_(local_subdomain_id, xi, yi, rr, 2);
    } else {
        k_sh(node, lvl) = 0.0;
        src_sh(node, 0, lvl) = src_sh(node, 1, lvl) = src_sh(node, 2, lvl) = 0.0;
    }
});

team.team_barrier();
```

After this barrier, the wave proceeds to the cell-parallel work.

---

## 5. Per-cell compute (the hot loop)

### 5.1 Address derivation per lane

```cpp
const int lane_id      = team.team_rank() * 64 + ...;  // Kokkos doesn't expose this directly;
                                                       // use ThreadVectorRange to get a lane index
// We dispatch the work via a ThreadVectorRange:
Kokkos::parallel_for(ThreadVectorRange(team, 64), [&](int lane_id) {
    const int cell_in_wave = lane_id / 6;       // 0..9 (cell), with mask 60+ idle
    const int node_in_cell = lane_id % 6;       // 0..5 (wedge node)
    if (lane_id >= 60) return;                  // 4 idle lanes

    const int r_cell  = r0 + cell_in_wave;
    if (r_cell >= hex_rad_) return;             // boundary case

    const int x_cell_global = x_cell;           // shared across all lanes of this team
    const int y_cell_global = y_cell;
    // ...
});
```

### 5.2 Per-wedge work for one cell, in one lane

For wedge `w ∈ {0, 1}` (sequential outer loop, w=0 then w=1):

```cpp
for (int w = 0; w < 2; ++w) {
    // -- Boundary check (cell-uniform, same on all 6 lanes of this cell) --
    const bool at_cmb     = has_flag(local_subdomain_id, x_cell, y_cell, r_cell,     CMB);
    const bool at_surface = has_flag(local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE);
    // ... compute cmb_shift, surface_shift, treat_boundary_dirichlet

    // -- Wedge node indexing (per-lane; lane has its own node_in_cell) --
    const int ddx = WEDGE_NODE_OFF[w][node_in_cell][0];
    const int ddy = WEDGE_NODE_OFF[w][node_in_cell][1];
    const int ddr = WEDGE_NODE_OFF[w][node_in_cell][2];
    const int local_lvl_for_this_node = cell_in_wave + ddr;     // 0..10 inside the team's r-stack

    // -- Jacobian (redundant on all 6 lanes of this cell; cheap & avoids comms) --
    const int v0 = (w == 0) ? node_id_corner(0,0) : node_id_corner(1,1);
    const int v1 = (w == 0) ? node_id_corner(1,0) : node_id_corner(0,1);
    const int v2 = (w == 0) ? node_id_corner(0,1) : node_id_corner(1,0);
    const double r_0 = r_sh(cell_in_wave);
    const double r_1 = r_sh(cell_in_wave + 1);
    const double half_dr = 0.5 * (r_1 - r_0);
    const double r_mid   = 0.5 * (r_0 + r_1);

    // Compute J_*_*, det, inv_det, i00..i22, kwJ — same as today's code.
    // (Each lane redundantly computes; cheap given the small flop count.)
    const double i00 = ...; /* etc. */

    // -- Per-node gather: each lane handles its own wedge-node --
    const int nid = node_id_in_shmem(WEDGE_NODE_OFF[w][node_in_cell][0],
                                     WEDGE_NODE_OFF[w][node_in_cell][1]);  // 0..3
    const double s0 = src_sh(nid, 0, local_lvl_for_this_node);
    const double s1 = src_sh(nid, 1, local_lvl_for_this_node);
    const double s2 = src_sh(nid, 2, local_lvl_for_this_node);

    const double gx = dN_ref[node_in_cell][0];
    const double gy = dN_ref[node_in_cell][1];
    const double gz = dN_ref[node_in_cell][2];
    const double g0 = i00 * gx + i01 * gy + i02 * gz;
    const double g1 = i10 * gx + i11 * gy + i12 * gz;
    const double g2 = i20 * gx + i21 * gy + i22 * gz;

    // -- Per-lane partial contributions to the 7 cell-wide accumulators --
    const double p_gu00 = g0 * s0;
    const double p_gu11 = g1 * s1;
    const double p_gu22 = g2 * s2;
    const double p_gu10 = 0.5 * (g1 * s0 + g0 * s1);
    const double p_gu20 = 0.5 * (g2 * s0 + g0 * s2);
    const double p_gu21 = 0.5 * (g2 * s1 + g1 * s2);
    const double p_div  = g0 * s0 + g1 * s1 + g2 * s2;

    // -- Reduce within the cell's 6-lane group (see §5.3) --
    const double gu00  = reduce_within_cell(p_gu00, accum_sh, cell_in_wave, 0);
    /* ... 6 more ... */
    const double div_u = reduce_within_cell(p_div,  accum_sh, cell_in_wave, 6);

    // -- Per-lane scatter: each lane writes 3 atomic_adds to its node's dst position --
    // Phase 2 in the existing code, mapped onto this lane's owned node.
    if (!Diagonal) {
        const double v_x = kwJ * (2.0 * (g0 * gu00 + g1 * gu10 + g2 * gu20) + NEG_TWO_THIRDS * g0 * div_u);
        const double v_y = kwJ * (2.0 * (g0 * gu10 + g1 * gu11 + g2 * gu21) + NEG_TWO_THIRDS * g1 * div_u);
        const double v_z = kwJ * (2.0 * (g0 * gu20 + g1 * gu21 + g2 * gu22) + NEG_TWO_THIRDS * g2 * div_u);
        // 3 global atomics per lane (60 lanes total per wedge ⇒ 180 atomics per wedge per wave)
        Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 0), v_x);
        Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 1), v_y);
        Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 2), v_z);
    }
    if (Diagonal || (treat_boundary_dirichlet && at_boundary)) {
        // analogous diagonal expression
    }
}  // end wedge loop
```

### 5.3 Sub-wave reduction (the 6-lane reduce)

Reducing 6 lanes is awkward with `__shfl_xor` (it's natural for powers of 2). The clean approach is **LDS atomics on the per-cell scratchpad**:

```cpp
// accum_sh has shape [10 cells][7 accumulators]
KOKKOS_INLINE_FUNCTION
double reduce_within_cell(double partial, double* accum_sh, int cell_in_wave, int slot) {
    // Initialize once before the reduction (in node 0 of each cell-group)
    if (node_in_cell == 0) accum_sh[cell_in_wave * 7 + slot] = 0.0;
    __sync_threads_warp();   // intra-wave sync (no-op on AMD since wave is lock-step,
                             // but the compiler needs a fence). On HIP just use __builtin_amdgcn_wave_barrier()
                             // or rely on Kokkos's intra-wave ordering guarantees.
    Kokkos::atomic_add(&accum_sh[cell_in_wave * 7 + slot], partial);
    __sync_threads_warp();
    return accum_sh[cell_in_wave * 7 + slot];
}
```

LDS atomics on gfx90a are ~10 cycles each; 7 of them per cell-wedge ⇒ 70 cycles. Compared to a ~100-cycle global atomic, this is cheap.

**Alternative if LDS atomic latency hurts:** use `__shfl_xor` with strides 1, 2, 4 over an **8-lane-aligned group**. Pad to 8 lanes/cell, 8 cells/wave (64 active lanes). That wastes 2 lanes per cell (8 of 64 idle ⇒ 12.5% loss) but the reduction is 3 cheap shuffle rounds (3 × ~1 cycle = 3 cycles) instead of LDS atomics.

**Initial design uses LDS atomics** (simpler, no shuffle masking complications). Switch to padded-8 + shfl later if profiling says the reduction is hot.

---

## 6. Pseudocode of the full function

```cpp
template <bool Diagonal>
KOKKOS_INLINE_FUNCTION
void operator_fast_dirichlet_neumann_path_wave(
    const Team& team,
    const int local_subdomain_id,
    const int x_cell, const int y_cell, const int r0)
const
{
    constexpr int CELLS_PER_WAVE = 10;
    constexpr int LANES_PER_CELL = 6;
    constexpr int ACCUM_PER_CELL = 7;
    constexpr int NXY            = 4;     // 2×2 lateral corners
    constexpr int NLEV           = CELLS_PER_WAVE + 1;   // 11

    // ===== LDS layout =====
    auto* shmem = (double*) team.team_shmem().get_shmem(team_shmem_size_dn_wave());
    ScratchCoords coords_sh(shmem, NXY, 3);                 shmem += NXY * 3;
    ScratchSrc    src_sh   (shmem, NXY, 3, NLEV);           shmem += NXY * 3 * NLEV;
    ScratchK      k_sh     (shmem, NXY, NLEV);              shmem += NXY * NLEV;
    auto          r_sh     = View1D(shmem, NLEV);           shmem += NLEV;
    auto          accum_sh = View2D(shmem, CELLS_PER_WAVE, ACCUM_PER_CELL);

    // ===== Cooperative LDS load (see §4) =====
    cooperative_load_coords(team, coords_sh, x_cell, y_cell);
    cooperative_load_r     (team, r_sh,     r0);
    cooperative_load_src_k (team, src_sh, k_sh, x_cell, y_cell, r0);
    team.team_barrier();

    // ===== Wave-parallel per-cell, per-wedge compute =====
    Kokkos::parallel_for(ThreadVectorRange(team, 64), [&](int lane_id) {
        const int cell_in_wave = lane_id / LANES_PER_CELL;
        const int node_in_cell = lane_id % LANES_PER_CELL;
        const bool active      = (lane_id < CELLS_PER_WAVE * LANES_PER_CELL);

        const int r_cell = r0 + cell_in_wave;
        const bool in_bounds = active && (r_cell < hex_rad_) && (x_cell < hex_lat_) && (y_cell < hex_lat_);

        for (int w = 0; w < 2; ++w) {
            if (!in_bounds) continue;

            // (boundary check, Jacobian, gradient, partial accumulation — see §5.2)
            // (reduce_within_cell × 7  — see §5.3)
            // (scatter via 3 atomic_add per active lane)
        }
    });
}
```

---

## 7. Files

| File | Action |
|---|---|
| `epsilon_divdiv_kerngen.hpp` | Add `KernelPath::FastDirichletNeumannWave` enum, host-side dispatch, `team_shmem_size_dn_wave()`, `run_team_fast_dirichlet_neumann_wave<Diagonal>` member wrapper |
| `epsilon_divdiv_kerngen_wave.hpp` (new) | Body of `operator_fast_dirichlet_neumann_path_wave<Diagonal>` + LDS reduce helper + per-cell scatter helper |
| `epsilon_divdiv_kerngen_wave.md` (this file) | Design plan |

Selection at runtime: an env var or a member flag on `EpsilonDivDivKerngen`. E.g. set `kernel_path_ = FastDirichletNeumannWave` if `getenv("EPSDIVDIV_WAVE") != nullptr` AND the BC config matches.

---

## 8. Per-lane VGPR forecast

Live state at the inner-loop peak:
- `g0, g1, g2, s0, s1, s2, gx, gy, gz` (9 doubles, brief) ≈ 18 VGPRs
- `i00..i22` (9 doubles, live across gather + scatter) = 18 VGPRs
- `kwJ`, `r_0`, `r_1`, `half_dr`, `r_mid` (5 doubles) = 10 VGPRs
- `gu00..gu22, div_u` (7 doubles, live after reduction during scatter) = 14 VGPRs
- 4–6 ints for `cell_in_wave`, `node_in_cell`, `ddx/ddy/ddr`, `nid`, `r_cell` = 6 VGPRs

**Total ≈ 66 VGPRs** — right at the 64-VGPR / 8 waves/SIMD threshold. With careful scoping (drop `gu*` between gather and scatter and reload from `accum_sh`), should drop to <60 VGPRs and unlock 8 waves/SIMD.

---

## 9. Implementation milestones

1. **Skeleton** (no perf focus): add the enum, dispatch, empty `operator_*_wave` that just writes zeros. Verify it compiles and is selectable.
2. **Cooperative LDS load**: implement §4 in isolation. Verify the loaded data matches what the thread-per-cell path sees.
3. **Per-cell compute, gather only**: implement §5.2 up to the partial accumulators, no scatter. Verify the `accum_sh` values match the thread-per-cell intermediate `gu*, div_u`.
4. **Reduction + scatter**: implement §5.3 + the atomic scatter. Compare full output `dst` against thread-per-cell on a small problem (level 5, sdr 0). Tolerance: bit-level via deterministic atomics on a single GCD (sum order doesn't change).
5. **First perf measurement**: `benchmark_operators --refinement-level-subdomains 0 --min-level 8 --max-level 8 --executions 3` and `rocprof-compute profile`. Compare against v8 (`op_sdr0_ml8_lb128_1`).
6. **Iterate based on profile:**
   - If VGPRs > 64: drop `gu*` from registers between gather/scatter, reload from `accum_sh`.
   - If LDS-atomic reduction is hot: switch §5.3 to padded-8-lanes + `__shfl_xor` reduction.
   - If atomic scatter is hot: consider the `dst_local_hex` LDS-accumulator pattern from the deleted Laplace experiment.

---

## 10. Validation plan

1. **Numerical**: small problem (level 4, sdr 0, 1 cell laterally), both paths, compare `dst` element-wise within `eps · dofs`.
2. **A3 benchmark**: 5 timesteps with each path, compare `nu.csv` (Nusselt number trace) and final `radial_profiles`.
3. **MT sweep**: confirm strong scaling shape on MT64..MT256 doesn't regress vs v8.

---

## 11. Risks I'm betting against

- **LDS-atomic reduction overhead** swamping the SIMD-utilization gain → fallback is padded-8 + shfl_xor.
- **Redundant Jacobian compute** on all 6 lanes per cell eating most of the FLOPs win → fallback is lane-0-of-cell-group computes + broadcast via LDS (per-cell row in `accum_sh`).
- **Boundary `r_cell` at end of subdomain** causing branch divergence → mitigated by predicating the cell-mask cleanly (`active = lane_id < 60 && r_cell < hex_rad_`).
- **Address-arithmetic VGPR cost** for the per-lane scatter destination → unavoidable, plan tolerates it.

If the design works, expectations:
- 60/64 = 94% lane utilization
- ≤ 64 VGPRs / lane → 8 waves/SIMD (vs v8's 4)
- LDS dropped from 16 KB → 2.2 KB (rest is wasted headroom)
- Throughput: **+30–60% vs v8** (target ~10 GDoFs/s at level 8).


===================================================================


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
