#!/usr/bin/env python3
"""
Strong-scale benchmark driver for the *mantlecirculation* solver on LUMI-G.

Runs the full `mantlecirculation` app (Stokes + energy time stepping) at an
increasing MT model size. Each (MT-level, n_gpus) cell does CONSTANT solver work:
a fixed number of timesteps, each with a fixed FGMRES (Stokes) and EV (energy)
iteration count, so wall-clock differences reflect mesh/parallel scaling only --
not convergence variability.

The cell-generation logic (DEFAULT_SWEEP strong grid, --weak diagonals,
--weak-radial diagonals) is unchanged from the operator-benchmark lineage; only
the per-cell command differs:
  - 8 GCDs/node (LUMI-G: 4 MI250X x 2)
  - account project_465002367, partition standard-g
  - select_gpu ROCR wrapper + map_cpu CPU binding + MPICH_GPU_SUPPORT_ENABLED

Mesh/scaling knobs are passed as CLI overrides on top of a base config TOML
(CLI11 `--config` is overridden by later flags):
  - lateral resolution     -> --refinement-level-mesh-max <level>  (mesh-min fixed at 2)
  - subdomain refinement    -> --refinement-level-subdomains / --lat-sdr / --rad-sdr
  - radial extent           -> --radial-extra-levels  (= lat_level-1, i.e. MT/2)
Solver work per cell is pinned (see SOLVER section):
  - --max-timesteps, --stokes-krylov-{restart,max-iterations}=FGMRES,
    --energy-krylov-{restart,max-iterations}=EV, all Krylov tolerances = 0.

MT-level convention: MT(N) where N = 2^level is the LATERAL resolution; the radial
extent is MT/2: rad_level = lat_level - 1 (--radial-extra-levels -1).

For each (MT-level, n_gpus) cell:
  - emit a per-cell sbatch script under bench_mt/jobs/
  - submit via sbatch
  - the app writes its output under bench_mt/outputs/MT<...>_g<n>/ (--outdir)

Usage:
    python3 submit_bench_mt_lumi.py            # submit the default strong grid
    python3 submit_bench_mt_lumi.py --dry-run  # just print what would be submitted
    python3 submit_bench_mt_lumi.py --levels 8 9    # only those mesh levels
    python3 submit_bench_mt_lumi.py --config <base.toml> --max-timesteps 10
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

BENCH_DIR   = Path(__file__).resolve().parent
APP_DIR     = BENCH_DIR.parent
# mantlecirculation binary (LUMI PFS build). Override with $TERRANEO_BIN.
BINARY      = Path(os.environ.get(
    "TERRANEO_BIN",
    "/pfs/lustrep3/users/bohmfabi/terraneo-build/apps/mantlecirculation/mantlecirculation"))
JOB_DIR     = BENCH_DIR / "jobs"
OUTPUT_ROOT = BENCH_DIR / "outputs"

# Base config TOML supplying physics/solver settings. The mesh/scaling/solver-work
# knobs below are applied as CLI overrides on top of this. Override with --config.
DEFAULT_CONFIG = APP_DIR / "parameterfiles" / "config_fscmb_nsurf_lvl6_10steps.toml"

# MG-hierarchy coarse level (refinement-level-mesh-min). We want it as small as
# possible (deep hierarchy, tiny coarse grid), but the app requires every subdomain
# to hold >=1 cell at the coarsest level on BOTH axes:
#   lateral:  mesh_min               >= lat_sdr
#   radial:   mesh_min + radial_extra >= rad_sdr
# so the floor is per-cell, not a fixed 2. MESH_MIN_FLOOR is the smallest allowed.
MESH_MIN_FLOOR = 2

def mesh_min_for(lat_sdr: int, rad_sdr: int, radial_extra: int) -> int:
    return max(MESH_MIN_FLOOR, lat_sdr, rad_sdr - radial_extra)

# Radial extent is MT/2: rad_level = lat_level + RADIAL_EXTRA (= lat_level - 1).
RADIAL_EXTRA = -1

# (level, list of base GCD counts). Actual n_gpus = base * GCD_MULTIPLIER.
# level -> MT label = 2^level (lateral). Levels 7-11 span a wide GCD range so each
# MT model has several strong-scaling points on both sides of its sweet spot. On
# LUMI the >1024-GCD entries are skipped by MAX_SAFE_GCDS (Cray MPICH wall).
DEFAULT_SWEEP: list[tuple[int, list[int]]] = [
    # (level, [base n_gpus])      lat=2^level, rad=2^(level-1) (MT/2); actual = base*2
    (5,  [1, 2, 4, 8]),                       # MT32    actual 2..16  (+g1)
    (6,  [1, 2, 4, 8, 16]),                   # MT64    actual 2..32  (+g1)
    (7,  [2, 4, 8, 16, 32, 64, 128]),         # MT128   actual 4..256
    (8,  [8, 16, 32, 64, 128, 256, 512]),     # MT256   actual 16..1024
    (9,  [16, 32, 64, 128, 256, 512, 1024]),  # MT512   actual 32..2048 (added g32=126 M/GCD)
    (10, [64, 128, 256, 512, 1024]),          # MT1024  actual 128..2048 (added g128=126 M/GCD)
    (11, [256, 512, 1024, 2048, 4096]),       # MT2048  actual 512..8192 (added g512=252 M/GCD)
    (12, [2048, 4096]),                       # MT4096  actual 4096..8192
    # ^ MT4096 only fits at the largest GCD counts: ~126 M dofs/GCD (our proven
    #   memory ceiling, = MT2048_g1024) needs g8192 = 1024 nodes (the partition
    #   limit). g4096 (512 nodes) is ~251 M/GCD -- an OOM probe. Smaller counts OOM;
    #   larger would exceed 1024 nodes. Requires --max-gcds 8192 to clear the cap.
]

# Multiply every strong-grid GCD count by this (each cell runs at GCD_MULTIPLIER x
# the base n_gpus, i.e. less work per GCD / further-right strong scaling).
GCD_MULTIPLIER = 2

# Always include a single-GCD datapoint for these (small) levels, on top of the
# multiplied grid, so the small-model strong-scaling curves still start at 1 GCD
# even when GCD_MULTIPLIER > 1. Only the small models fit / were characterized at
# 1 GCD (the original sweep used 1 GCD only for levels 5-6).
EXTRA_1GCD_LEVELS = [5, 6]

# Cray MPICH GPU-aware MPI used to abort at >= 2048 ranks (cray_ch4_mem_utils.c:2086
# assertion + segfault in the first energy solve, from CXI MR-cache exhaustion).
# Mitigated in the generated job scripts via FI_MR_CACHE_MAX_COUNT=1048576 +
# FI_CXI_RX_MATCH_MODE=software, so >1024-GCD cells can register their halo buffers.
# Raised to the partition limit (8192 GCDs = 1024 nodes) so the large models
# (MT2048/MT4096) run on their smallest-fitting node counts.
MAX_SAFE_GCDS = 8192

# Measured owned-DOF counts per level, ISOTROPIC (lat=rad=level). With the MT/2
# radial extent the effective dofs are these x 2**RADIAL_EXTRA (i.e. halved).
# level 11 extrapolated at x7.99.
DOFS_PER_LEVEL = {
    5: 1013958, 6: 7987590, 7: 63406854, 8: 505284102,
    9: 4034399238, 10: 32243718150, 11: 257627107000,
    12: 2058460585000,   # extrapolated at x7.99 (MT4096)
}

def dofs_at(level: int) -> float | None:
    """Effective owned dofs for an MT/2-radial cell at `level` (None if unknown)."""
    iso = DOFS_PER_LEVEL.get(level)
    return iso * 2 ** RADIAL_EXTRA if iso is not None else None

# --- Weak-scaling diagonals (constant work per GCD) ---
# Hold dofs/GCD (and subdomains/GCD) fixed by stepping the model up one level
# (x8 dofs) and the GCD count up x8 together. Anchored at level 7.
#   base_gpus 1 -> ~63 M/GCD (iso), 2 -> ~32 M, 4 -> ~16 M, 8 -> ~8 M, ...
WEAK_BASE_LEVEL    = 7
WEAK_MAX_LEVEL     = 11
WEAK_DEFAULT_FILLS = [63, 32, 16, 8]     # target M-dofs/GCD diagonals to emit
WEAK_MAX_GPUS      = 4096                # skip points beyond this (512 LUMI nodes)

def weak_base_gpus(fill_mdofs: int, base_level: int = WEAK_BASE_LEVEL) -> int:
    """GCD count at the anchor level giving ~fill_mdofs dofs/GCD."""
    return max(1, round(DOFS_PER_LEVEL[base_level] / (fill_mdofs * 1e6)))

def weak_cells(fills: list[int], base_level: int, max_level: int,
               max_gpus: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for fill in fills:
        m = weak_base_gpus(fill, base_level)
        for L in range(base_level, max_level + 1):
            n = m * 8 ** (L - base_level)
            if n > max_gpus:
                break
            if (L, n) not in seen:
                seen.add((L, n))
                cells.append((L, n))
    return sorted(cells)

# --- Radial-only weak-scaling diagonals ---
# Fix the lateral mesh; grow ONLY the radial direction. Each step k adds one radial
# diamond level (rad_level = lat_level + k -> x2 dofs) AND one radial subdomain
# level (rad_sdr += 1 -> x2 subdomains), with GPUs x2. dofs/GCD and subdomains/GCD
# stay fixed, isolating radial scaling at 2x granularity.
RADIAL_LAT_LEVEL     = 8
RADIAL_LAT_SDR       = 1
# Anchor every diagonal at base_gpus = 8 = 1 LUMI node (lat_sdr 1 -> rad_sdr 1 so
# base_gpus = 4^lat_sdr * 2^rad_sdr = 4*2 = 8), varying the starting radial level.
RADIAL_DEFAULT_FILLS = [126, 63, 32, 16, 8]
RADIAL_BASE_GPUS     = 8   # 1 node on LUMI-G
RADIAL_BASE_RADSDR   = 1   # base_gpus = 4^1 * 2^1 = 8 with lat_sdr=1

def radial_start_offset(fill_mdofs: int) -> int:
    """Radial extra levels at the base anchor that hit `fill_mdofs` per GCD."""
    return round(math.log2(fill_mdofs * 1e6 * RADIAL_BASE_GPUS
                            / DOFS_PER_LEVEL[RADIAL_LAT_LEVEL]))

def radial_cells(fills: list[int], max_gpus: int) -> list[dict]:
    cells: list[dict] = []
    for fill in fills:
        rad_start_off = radial_start_offset(fill)
        k = 0
        while RADIAL_BASE_GPUS * 2 ** k <= max_gpus:
            cells.append(dict(
                lat_level=RADIAL_LAT_LEVEL, lat_sdr=RADIAL_LAT_SDR,
                rad_level=RADIAL_LAT_LEVEL + rad_start_off + k,
                rad_sdr=RADIAL_BASE_RADSDR + k,
                radial_extra=rad_start_off + k,
                n_gpus=RADIAL_BASE_GPUS * 2 ** k, fill=fill))
            k += 1
    return cells

# --- Solver work per cell (constant across the sweep) ---
# Each timestep does a fixed amount of work so wall-clock reflects scaling, not
# convergence. FGMRES = Stokes Krylov iterations, EV = energy (entropy-viscosity)
# Krylov iterations. We cap each solver's max-iterations and zero its tolerances so
# it always runs the full count. We do NOT touch the Krylov *restart*: the app sizes
# its FGMRES tmp-vector pool from the (base-config / default) restart, and forcing
# restart > that aborts with "insufficient tmp vectors". The iteration count is
# therefore reached via restart cycles (e.g. 50 = 10 cycles of restart-5).
DEFAULT_MAX_TIMESTEPS = 10
DEFAULT_FGMRES        = 10   # Stokes FGMRES iterations / timestep
DEFAULT_EV_ITERS      = 50   # energy (EV) Krylov iterations / timestep

def solver_overrides(max_timesteps: int, fgmres: int, ev_iters: int) -> str:
    # A benchmark must write NO diagnostic output and skip the diagnostic compute.
    # Every per-step output AND the Nusselt/V_rms computation is gated in the app by
    # write_output = (timestep % output_frequency == 0). The base config sets
    # output-frequency=1 (full T/u/eta XDMF + radial profiles + nu.csv + nu_h_stats +
    # timer_trees every step); at scale that dominates wall-clock and overflows the
    # disk quota. Setting output-frequency past max_timesteps makes write_output false
    # for every step, disabling all of it (radial profiles, Nusselt, nu_h dump, timer
    # trees, XDMF) and the Nusselt/V_rms compute. --no-xdmf and --no-radial-profiles
    # also kill the one-time *initial* (step-0) XDMF and radial-profile passes.
    # (The step-0 Nu_top diagnostic and the intrinsic EV nu_h field stay -- no flag.)
    # Per-timestep wall-clock still comes from the "### Timestep" log lines.
    return (
        f"--max-timesteps {max_timesteps} --no-xdmf --no-radial-profiles "
        f"--output-frequency {max_timesteps + 1} "
        f"--stokes-krylov-max-iterations {fgmres} "
        f"--stokes-krylov-relative-tolerance 0 --stokes-krylov-absolute-tolerance 0 "
        f"--energy-krylov-max-iterations {ev_iters} "
        f"--energy-krylov-relative-tolerance 0 --energy-krylov-absolute-tolerance 0"
    )

# Solver precision / restart presets (--mode):
#   low-mem : FP16 (float) Krylov basis for Stokes + energy, Stokes restart lowered to 5
#             (energy restart is already 5 by default), and a single pre/post smoothing
#             step in the velocity multigrid. Minimises both the FGMRES workspace (the
#             dominant memory term at high dofs/GCD) and the smoother cost.
#   std     : full double Krylov basis, Stokes restart 10 (= the base-config restart, so
#             the tmp-vector pool is sized for it), and the default 2 pre/post smoothing
#             steps. More memory + smoother work, stronger convergence.
# The app sizes its FGMRES tmp pool from the base-config restart (stokes=10); restart may
# only be LOWERED at runtime, so std keeps 10 and low-mem lowers Stokes to 5.
def mode_overrides(mode: str) -> str:
    if mode == "low-mem":
        return (" --stokes-float-krylov-basis --energy-float-krylov-basis"
                " --stokes-krylov-restart 5 --energy-krylov-restart 5"
                " --stokes-viscous-pc-num-smoothing-steps-prepost 1")
    return (" --stokes-krylov-restart 10"
            " --stokes-viscous-pc-num-smoothing-steps-prepost 2")

def mode_suffix(mode: str) -> str:
    # Tag every cell with its mode so the output dir, job script and SBATCH job-name all
    # carry it (e.g. MT1024_g128_lowmem vs MT1024_g128_std) -- the two modes never collide.
    return "_lowmem" if mode == "low-mem" else "_std"

# LUMI-G: each node has 8 GCDs (4 MI250X * 2). standard-g allocates GPUs; we bind
# one rank per GCD. Account / partition can be overridden via env.
GPUS_PER_NODE = 8
ACCOUNT       = os.environ.get("SLURM_ACCOUNT", "project_465002367")
PARTITION     = "standard-g"

# NUMA-aware CPU cores, one per GCD, in the canonical LUMI GCD order. For a partial
# node we take the first `tasks_per_node` entries.
CPU_BIND_ORDER = [49, 57, 17, 25, 1, 9, 33, 41]

# Walltime cap. The fixed-work run (10 steps x 10 FGMRES + 50 EV) finishes in a
# few minutes even for the heaviest cells, so 15 min is ample headroom and gives
# much better backfill priority than a multi-hour limit. Overridable with --time-limit.
def time_limit_for(level: int) -> str:
    return "00:15:00"

# Subdomain count under INDEPENDENT lateral/radial refinement:
# total subdomains = 10 * 4^lat_sdr (lateral) * 2^rad_sdr (radial). Decoupling the
# two axes lets us hold subdomains/GPU constant across the sweep, avoiding the
# over-decomposition cliffs of the old isotropic sdr (a cell with more GPUs but 4x
# the subdomains/GPU can run *slower* despite identical global DoFs).
def subdomains_lr(lat_sdr: int, rad_sdr: int) -> int:
    return 10 * (4 ** lat_sdr) * (2 ** rad_sdr)

# Hold subdomains/GPU fixed across the sweep. 5 is the smallest constant achievable
# for power-of-2 GPU counts (minimal over-decomposition that still balances); the
# single-GPU anchor falls back to 10 (the base mesh has 10 macro-subdomains).
TARGET_SUBDOM_PER_GPU = 5

def choose_lat_rad(n_gpus: int, target: int = TARGET_SUBDOM_PER_GPU) -> tuple[int, int]:
    """Pick (lat_sdr, rad_sdr) so subdomains == target * n_gpus, holding
    subdomains/GPU constant. Grows the decomposition laterally while toggling the
    radial split 0/1, keeping the radial decomposition shallow (radial extent is
    MT/2). Falls back to the smallest balanced (>= n_gpus, evenly dividing)
    decomposition when the exact target isn't reachable (e.g. n_gpus = 1)."""
    want = target * n_gpus
    if want % 10 == 0:
        q = want // 10
        if q & (q - 1) == 0:                  # q is a power of two -> exact solution
            k = q.bit_length() - 1            # k = 2*lat_sdr + rad_sdr
            lat, rad = k // 2, k % 2
            if subdomains_lr(lat, rad) % n_gpus == 0:
                return lat, rad
    best = None                               # fallback: smallest balanced split
    for lat in range(0, 7):
        for rad in range(0, 3):
            subs = subdomains_lr(lat, rad)
            if subs >= n_gpus and subs % n_gpus == 0 and (best is None or subs < best[0]):
                best = (subs, lat, rad)
    if best is None:
        raise ValueError(f"no balanced (lat_sdr, rad_sdr) found for n_gpus={n_gpus}")
    return best[1], best[2]

def choose_iso_sdr(n_gpus: int, rad_level: int):
    """Isotropic (uniform lat_sdr = rad_sdr = s) decomposition: subdomains = 10*8^s
    (emitted as the old --refinement-level-subdomains s). Smallest s that is >= n_gpus,
    evenly divides n_gpus, and keeps rad_sdr = s <= rad_level - 1 (else the EpsDivDiv
    operator hits 1 radial cell/subdomain and aborts). Returns None if no valid s, so
    the cell is skipped under --isotropic."""
    for s in range(0, 7):
        if s > rad_level - 1:
            break
        nsub = 10 * 8 ** s
        if nsub >= n_gpus and nsub % n_gpus == 0:
            return s
    return None

def gpu_to_node_layout(n_gpus: int) -> tuple[int, int]:
    """Map a target n_gpus to (#nodes, #ntasks_per_node).

    Up to 8 GCDs we pack onto a single node; beyond that we use full nodes
    (8 GCDs each).
    """
    if n_gpus <= GPUS_PER_NODE:
        return (1, n_gpus)
    if n_gpus % GPUS_PER_NODE != 0:
        raise ValueError(f"n_gpus={n_gpus} is not a multiple of {GPUS_PER_NODE} for >1 nodes")
    return (n_gpus // GPUS_PER_NODE, GPUS_PER_NODE)

def write_lumi_job(cell_tag: str, config: Path, out_path: Path, nodes: int, tpn: int,
                   time_limit: str, app_args: str, echo_line: str) -> Path:
    """Per-cell sbatch script for one mantlecirculation cell on LUMI-G."""
    job_path = JOB_DIR / f"bench_{cell_tag}.sh"
    cpu_bind = "map_cpu:" + ",".join(str(c) for c in CPU_BIND_ORDER[:tpn])
    script = f"""#!/bin/bash -l
#SBATCH --job-name=bench_{cell_tag}
#SBATCH --output=bench_{cell_tag}.o%j
#SBATCH --error=bench_{cell_tag}.e%j
#SBATCH -D {JOB_DIR}
#SBATCH --partition={PARTITION}
#SBATCH --account={ACCOUNT}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tpn}
#SBATCH --gpus-per-node={tpn}
#SBATCH --time={time_limit}

echo "{echo_line}"

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_NO_ASYNC_COPY=1
export OMP_NUM_THREADS=1
# Raise the libfabric memory-registration cache ceiling. At high rank counts the
# halo exchange opens >10000 active MRs, exhausting the CXI provider default and
# tripping a Cray MPICH internal assertion (cray_ch4_mem_utils.c) + segfault in
# the first energy solve. Lifting the cache count (and the per-region cap) lets
# the >=2048-rank cells register all their halo buffers.
export FI_MR_CACHE_MAX_COUNT=1048576
export FI_CXI_RX_MATCH_MODE=software
ulimit -c 0

# Per-GCD GPU binding wrapper (maps SLURM_LOCALID -> ROCR_VISIBLE_DEVICES).
SELECT_GPU=${{SLURM_SUBMIT_DIR}}/select_gpu_${{SLURM_JOB_ID}}.sh
cat > ${{SELECT_GPU}} << 'INNER'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
INNER
chmod +x ${{SELECT_GPU}}

# mantlecirculation writes its output tree under --outdir; cd there to capture any
# CWD-relative artifacts (timing trees) too.
mkdir -p {out_path}
cd {out_path}

srun --cpu-bind={cpu_bind} ${{SELECT_GPU}} {BINARY} --config {config} {app_args} --outdir {out_path} --outdir-overwrite

rm -f ${{SELECT_GPU}}
"""
    job_path.write_text(script)
    job_path.chmod(0o755)
    return job_path

def render_job_script(level: int, n_gpus: int, opts) -> tuple[Path, Path, dict]:
    mt_label   = 2 ** level
    rad_level  = level + RADIAL_EXTRA
    nodes, tpn = gpu_to_node_layout(n_gpus)

    if getattr(opts, "isotropic", False):
        s = choose_iso_sdr(n_gpus, rad_level)
        if s is None:
            return None, None, None        # no valid isotropic decomposition -> skip cell
        lat_sdr = rad_sdr = s
        subdomains  = subdomains_lr(s, s)
        mesh_min    = mesh_min_for(s, s, RADIAL_EXTRA)
        decomp_flag = f"--refinement-level-subdomains {s}"
        decomp_tag  = "_iso"
    else:
        lat_sdr, rad_sdr = choose_lat_rad(n_gpus)
        subdomains  = subdomains_lr(lat_sdr, rad_sdr)
        mesh_min    = mesh_min_for(lat_sdr, rad_sdr, RADIAL_EXTRA)
        decomp_flag = f"--lat-sdr {lat_sdr} --rad-sdr {rad_sdr}"
        decomp_tag  = ""

    cell_tag = f"MT{mt_label}_g{n_gpus}{decomp_tag}{mode_suffix(opts.mode)}"
    out_path = OUTPUT_ROOT / cell_tag

    info = dict(
        cell_tag=cell_tag, mt_label=mt_label,
        level=level, mesh_min=mesh_min, rad_level=rad_level,
        lat_sdr=lat_sdr, rad_sdr=rad_sdr, subdom_per_gpu=subdomains // n_gpus,
        max_timesteps=opts.max_timesteps, fgmres=opts.fgmres, ev_iters=opts.ev_iters,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
    )
    app_args = (
        f"--refinement-level-mesh-min {mesh_min} --refinement-level-mesh-max {level} "
        f"{decomp_flag} --radial-extra-levels {RADIAL_EXTRA} "
        + solver_overrides(opts.max_timesteps, opts.fgmres, opts.ev_iters)
        + mode_overrides(opts.mode)
    )
    echo_line = (f"Cell: {cell_tag}  mesh=[{mesh_min}..{level}]  rad_level={rad_level}  "
                 f"lat_sdr={lat_sdr}  rad_sdr={rad_sdr}  subdomains={subdomains}  "
                 f"subdom/GCD={subdomains // n_gpus}  steps={opts.max_timesteps}  "
                 f"fgmres={opts.fgmres}  ev={opts.ev_iters}  n_gpus={n_gpus}  "
                 f"nodes={nodes}x{tpn}  partition={PARTITION}")
    time_limit = opts.time_limit or time_limit_for(level)
    job_path = write_lumi_job(cell_tag, opts.config, out_path, nodes, tpn,
                              time_limit, app_args, echo_line)
    return job_path, out_path, info

def render_radial_job_script(cell: dict, opts) -> tuple[Path, Path, dict]:
    """One step of a radial-only weak diagonal: lateral fixed, radial level and
    radial subdomains grow by k, GPUs by 2**k. dofs and GCDs both x2 per step."""
    lat_level  = cell["lat_level"]
    lat_sdr    = cell["lat_sdr"]
    rad_level  = cell["rad_level"]
    rad_sdr    = cell["rad_sdr"]
    k          = cell["radial_extra"]
    n_gpus     = cell["n_gpus"]
    nodes, tpn = gpu_to_node_layout(n_gpus)
    subdomains = 10 * 4 ** lat_sdr * 2 ** rad_sdr
    mesh_min   = mesh_min_for(lat_sdr, rad_sdr, k)
    cell_tag   = f"RAD_l{lat_level}r{rad_level}_g{n_gpus}{mode_suffix(opts.mode)}"
    out_path   = OUTPUT_ROOT / cell_tag

    info = dict(
        cell_tag=cell_tag, lat_level=lat_level, mesh_min=mesh_min, rad_level=rad_level,
        lat_sdr=lat_sdr, rad_sdr=rad_sdr, radial_extra=k,
        max_timesteps=opts.max_timesteps, fgmres=opts.fgmres, ev_iters=opts.ev_iters,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
        subdom_per_gcd=subdomains // n_gpus,
    )
    app_args = (
        f"--refinement-level-mesh-min {mesh_min} --refinement-level-mesh-max {lat_level} "
        f"--radial-extra-levels {k} --lat-sdr {lat_sdr} --rad-sdr {rad_sdr} "
        + solver_overrides(opts.max_timesteps, opts.fgmres, opts.ev_iters)
        + mode_overrides(opts.mode)
    )
    echo_line = (f"Cell: {cell_tag}  mesh=[{mesh_min}..{lat_level}]  rad_level={rad_level}  "
                 f"lat_sdr={lat_sdr}  rad_sdr={rad_sdr}  subdomains={subdomains}  "
                 f"steps={opts.max_timesteps}  fgmres={opts.fgmres}  ev={opts.ev_iters}  "
                 f"n_gpus={n_gpus}  nodes={nodes}x{tpn}  partition={PARTITION}")
    time_limit = opts.time_limit or time_limit_for(rad_level)
    job_path = write_lumi_job(cell_tag, opts.config, out_path, nodes, tpn,
                              time_limit, app_args, echo_line)
    return job_path, out_path, info

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="only emit the job scripts; don't sbatch")
    p.add_argument("--levels", type=int, nargs="+", default=None,
                   help="override which mesh levels to submit (default: full sweep)")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                   help="only submit cells whose final n_gpus is in this list "
                        "(applied after the GCD_MULTIPLIER; strong/weak grid only)")
    p.add_argument("--max-gcds", type=int, default=MAX_SAFE_GCDS,
                   help=f"skip cells above this GCD count (default: {MAX_SAFE_GCDS}, the "
                        f"Cray MPICH MR-exhaustion wall); raise to attempt larger runs")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"base config TOML (default: {DEFAULT_CONFIG.name})")
    p.add_argument("--mode", choices=["std", "low-mem"], default="std",
                   help="solver precision/restart preset: std = double Krylov basis + "
                        "Stokes restart 10; low-mem = FP16 basis + restart 5. The cell tag "
                        "(output dir / job script / job-name) is suffixed _std or _lowmem.")
    p.add_argument("--isotropic", action="store_true",
                   help="use the uniform (isotropic) --refinement-level-subdomains s "
                        "decomposition (lat_sdr=rad_sdr=s, like the original sweep) instead "
                        "of the asymmetric lat_sdr/rad_sdr. Cells whose required s exceeds "
                        "rad_level-1 are skipped. Cell tag gets an extra _iso suffix.")
    p.add_argument("--max-timesteps", type=int, default=DEFAULT_MAX_TIMESTEPS,
                   help=f"timesteps per cell (default: {DEFAULT_MAX_TIMESTEPS})")
    p.add_argument("--fgmres", type=int, default=DEFAULT_FGMRES,
                   help=f"Stokes FGMRES iterations per timestep (default: {DEFAULT_FGMRES})")
    p.add_argument("--ev-iters", type=int, default=DEFAULT_EV_ITERS,
                   help=f"energy (EV) Krylov iterations per timestep (default: {DEFAULT_EV_ITERS})")
    p.add_argument("--time-limit", type=str, default=None,
                   help="override the SLURM walltime for every cell (e.g. 03:00:00)")
    p.add_argument("--gcd-multiplier", type=int, default=GCD_MULTIPLIER,
                   help=f"multiply every strong-grid GCD count by this "
                        f"(default: {GCD_MULTIPLIER}); use 1 for the base grid")
    p.add_argument("--weak", action="store_true",
                   help="weak-scaling diagonals at constant dofs/GCD "
                        "(n_gpus = base * 8^(level-7)) instead of the strong grid")
    p.add_argument("--weak-fills", type=int, nargs="+", default=WEAK_DEFAULT_FILLS,
                   help=f"target dofs/GCD per diagonal, in millions "
                        f"(default: {WEAK_DEFAULT_FILLS})")
    p.add_argument("--weak-max-level", type=int, default=WEAK_MAX_LEVEL,
                   help=f"top level for --weak (default: {WEAK_MAX_LEVEL}; "
                        f"each +1 level is x8 GCDs)")
    p.add_argument("--weak-max-gpus", type=int, default=WEAK_MAX_GPUS,
                   help=f"skip --weak / --weak-radial points above this many GCDs "
                        f"(default: {WEAK_MAX_GPUS} = 512 nodes)")
    p.add_argument("--weak-radial", action="store_true",
                   help="radial-only weak-scaling diagonals: lateral mesh fixed, radial "
                        "level+subdomains and GPUs x2 per step (dofs x2/step, constant dofs/GCD)")
    p.add_argument("--weak-radial-fills", type=int, nargs="+", default=RADIAL_DEFAULT_FILLS,
                   help=f"target dofs/GCD per radial diagonal, in millions "
                        f"(default: {RADIAL_DEFAULT_FILLS})")
    args = p.parse_args(argv)

    if not args.config.is_file():
        raise SystemExit(f"ERROR: base config not found: {args.config}")

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"binary: {BINARY}")
    print(f"base config: {args.config}")
    print(f"solver work/cell: {args.max_timesteps} steps x ({args.fgmres} FGMRES + "
          f"{args.ev_iters} EV), Krylov tol=0 (fixed work)")

    # Build a render list of (kind, item): 'iso' -> (level, n_gpus); 'radial' -> cell dict.
    if args.weak_radial:
        cells = radial_cells(args.weak_radial_fills, args.weak_max_gpus)
        over = [c for c in cells if c["n_gpus"] > args.max_gcds]
        if over:
            print(f"  SKIP {len(over)} radial cell(s) > {args.max_gcds} GCDs (MR-exhaustion wall)")
            cells = [c for c in cells if c["n_gpus"] <= args.max_gcds]
        render_list = [("radial", c) for c in cells]
        print(f"radial-only weak-scaling diagonals (lateral fixed at level {RADIAL_LAT_LEVEL}, "
              f"lat_sdr {RADIAL_LAT_SDR}; fills {args.weak_radial_fills} M-dofs/GCD): {len(cells)} cells")
        dpg0 = DOFS_PER_LEVEL.get(RADIAL_LAT_LEVEL)
        for c in cells:
            n = c["n_gpus"]
            nodes, _ = gpu_to_node_layout(n)
            subdom = 10 * 4 ** c["lat_sdr"] * 2 ** c["rad_sdr"]
            dpg_s = f"  dofs/GCD={dpg0 * 2**c['radial_extra'] / n / 1e6:5.1f}M" if dpg0 else ""
            print(f"  fill={c['fill']:2}M  rad_level={c['rad_level']:2}  rad_sdr={c['rad_sdr']}  "
                  f"n_gpus={n:5}  nodes={nodes:4}  subdom/GCD={subdom // n}{dpg_s}")
    else:
        if args.weak:
            cells = weak_cells(args.weak_fills, WEAK_BASE_LEVEL,
                               args.weak_max_level, args.weak_max_gpus)
        else:
            cells = [(level, n_gpus * args.gcd_multiplier)
                     for level, gpu_list in DEFAULT_SWEEP
                     for n_gpus in gpu_list]
            cells += [(lvl, 1) for lvl in EXTRA_1GCD_LEVELS]
            seen = set()
            cells = sorted(c for c in cells if not (c in seen or seen.add(c)))
        if args.levels is not None:
            cells = [(level, n_gpus) for (level, n_gpus) in cells if level in args.levels]
        if args.gpus is not None:
            cells = [(level, n_gpus) for (level, n_gpus) in cells if n_gpus in args.gpus]
        over = [(l, n) for (l, n) in cells if n > args.max_gcds]
        if over:
            print(f"  SKIP {len(over)} cell(s) > {args.max_gcds} GCDs (MR-exhaustion wall): "
                  + ", ".join(f"MT{2**l}_g{n}" for l, n in over))
            cells = [(l, n) for (l, n) in cells if n <= args.max_gcds]
        render_list = [("iso", c) for c in cells]
        if args.weak:
            print(f"weak-scaling diagonals (fills {args.weak_fills} M-dofs/GCD): {len(cells)} cells")
        else:
            print(f"strong-scaling grid (radial = MT/2, rad_level = lat_level-1): {len(cells)} cells")
        for (level, n_gpus) in cells:
            if args.isotropic:
                s = choose_iso_sdr(n_gpus, level + RADIAL_EXTRA)
                if s is None:
                    print(f"  MT{2**level:<5} n_gpus={n_gpus:5}  SKIP (no isotropic s <= rad_level-1)")
                    continue
                lat_sdr = rad_sdr = s
            else:
                lat_sdr, rad_sdr = choose_lat_rad(n_gpus)
            subs = subdomains_lr(lat_sdr, rad_sdr)
            nodes, _ = gpu_to_node_layout(n_gpus)
            dpg = dofs_at(level)
            dpg_s = f"  dofs/GCD={dpg / n_gpus / 1e6:5.1f}M" if dpg else ""
            print(f"  MT{2**level:<5} lat={level:2} rad={level+RADIAL_EXTRA:2}  n_gpus={n_gpus:5}  "
                  f"nodes={nodes:4}  lat_sdr={lat_sdr} rad_sdr={rad_sdr}  "
                  f"subdom/GCD={subs // n_gpus}{dpg_s}")

    submitted = []
    for kind, item in render_list:
        if kind == "radial":
            job_path, out_path, info = render_radial_job_script(item, args)
        else:
            level, n_gpus = item
            job_path, out_path, info = render_job_script(level, n_gpus, args)
        if job_path is None:
            print(f"  SKIP MT{2**item[0]}_g{item[1]}: no valid isotropic decomposition "
                  f"(required s > rad_level-1)")
            continue
        if args.dry_run:
            print(f"  [dry-run] would submit: {job_path}")
        else:
            res = subprocess.run(["sbatch", str(job_path)], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  FAILED to submit {job_path}: {res.stderr.strip()}", file=sys.stderr)
            else:
                jobid = res.stdout.strip().split()[-1]
                print(f"  submitted job {jobid}: {info['cell_tag']}")
                submitted.append((info, jobid))

    if submitted and not args.dry_run:
        # Persist a manifest so collect_bench_mt.py can map cells -> job ids.
        if args.weak_radial:
            manifest = JOB_DIR / "manifest_radial.txt"
            header = ("# cell_tag\tjob_id\tlat_level\trad_level\tlat_sdr\trad_sdr"
                      "\tmax_timesteps\tfgmres\tev_iters\tn_gpus\tnodes\ttasks_per_node\n")
            fields = ["lat_level", "rad_level", "lat_sdr", "rad_sdr",
                      "max_timesteps", "fgmres", "ev_iters", "n_gpus", "nodes", "tasks_per_node"]
        else:
            manifest = JOB_DIR / ("manifest_weak.txt" if args.weak else "manifest.txt")
            header = ("# cell_tag\tjob_id\tlevel\trad_level\tlat_sdr\trad_sdr\tmax_timesteps"
                      "\tfgmres\tev_iters\tn_gpus\tnodes\ttasks_per_node\n")
            fields = ["level", "rad_level", "lat_sdr", "rad_sdr", "max_timesteps", "fgmres",
                      "ev_iters", "n_gpus", "nodes", "tasks_per_node"]
        with open(manifest, "w") as f:
            f.write(header)
            for info, jobid in submitted:
                f.write("\t".join([info["cell_tag"], jobid] +
                                  [str(info[k]) for k in fields]) + "\n")
        print(f"\nWrote manifest: {manifest}")

if __name__ == "__main__":
    main(sys.argv[1:])
