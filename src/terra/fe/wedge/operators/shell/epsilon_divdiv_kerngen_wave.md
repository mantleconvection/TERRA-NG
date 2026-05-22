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
