#!/usr/bin/env python3
"""
Strong-scale benchmark driver for the *mantlecirculation* solver on JUWELS Booster.

JUWELS port of submit_bench_mt_lumi.py. Runs the full `mantlecirculation` app
(Stokes + energy time stepping) at an increasing MT model size. Each
(MT-level, n_gpus) cell does CONSTANT solver work: a fixed number of timesteps,
each with a fixed FGMRES (Stokes) and EV (energy) iteration count, so wall-clock
differences reflect mesh/parallel scaling only -- not convergence variability.

The cell-generation logic (DEFAULT_SWEEP strong grid, --weak diagonals,
--weak-radial diagonals) is identical to the LUMI driver; only the platform
layer differs:
  - 4 GPUs/node (JUWELS Booster: 4 A100 per node) instead of 8 GCDs (LUMI-G)
  - account walberlamovinggeo, partition booster (largebooster above 384 nodes)
  - srun --gpu-bind=closest (no ROCR select_gpu wrapper / map_cpu binding)
  - NVHPC/25.5 + OpenMPI/4.1.8 module env with UCX over CUDA IPC

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
  - emit a per-cell sbatch script under $SCRATCH/.../jobs_juwels/
  - submit via sbatch
  - the app writes its output under .../outputs_juwels/MT<...>_g<n>/ (--outdir)

Output/job trees live on scratch, NOT $HOME: $HOME hit its hard quota mid-sweep
on an earlier run and silently truncated timer-tree writes (and a crash left
~1000 core dumps that overflowed the quota). Scratch avoids both.

Usage:
    python3 submit_bench_mt_jb.py            # submit the default strong grid
    python3 submit_bench_mt_jb.py --dry-run  # just print what would be submitted
    python3 submit_bench_mt_jb.py --levels 8 9    # only those mesh levels
    python3 submit_bench_mt_jb.py --config <base.toml> --max-timesteps 10
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
# mantlecirculation binary (JUWELS Booster build). Override with $TERRANEO_BIN.
BINARY      = Path(os.environ.get(
    "TERRANEO_BIN",
    "/p/home/jusers/boehm2/juwels/terraneo-build/apps/mantlecirculation/mantlecirculation"))
# Job scripts, SLURM logs, and app outputs go to scratch (95 TB free), NOT $HOME:
# $HOME's hard quota truncated timer-tree writes mid-sweep on an earlier run.
# Override the root with $BENCH_MT_SCRATCH.
SCRATCH_ROOT = Path(os.environ.get(
    "BENCH_MT_SCRATCH",
    "/p/scratch/walberlamovinggeo/boehm2/mantlecirculation/bench_mt"))
JOB_DIR     = SCRATCH_ROOT / "jobs_juwels"
OUTPUT_ROOT = SCRATCH_ROOT / "outputs_juwels"

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

# (level, list of GPU counts to test). level -> MT label = 2^level (lateral).
DEFAULT_SWEEP: list[tuple[int, list[int]]] = [
    # (level, [n_gpus])           lat=2^level, rad=2^(level-1) (MT/2)
    (5,  [1, 2, 4, 8]),                  # MT32    (lat=32,   rad=16)
    (6,  [1, 2, 4, 8, 16]),              # MT64    (lat=64,   rad=32)
    (7,  [4, 8, 16, 32]),                # MT128   (lat=128,  rad=64)
    (8,  [16, 32, 64, 128, 256]),        # MT256   (lat=256,  rad=128)
    (9,  [64, 128, 256, 512]),           # MT512   (lat=512,  rad=256)
    (10, [256, 512, 1024, 2048]),        # MT1024  (lat=1024, rad=512)
    (11, [1024, 2048, 4096]),            # MT2048  (lat=2048, rad=1024)
]

# Multiply every strong-grid GPU count by this (each cell runs at GPU_MULTIPLIER x
# the base n_gpus, i.e. less work per GPU / further-right strong scaling).
GPU_MULTIPLIER = 2

# Always include a single-GPU datapoint for these (small) levels, on top of the
# multiplied grid, so the small-model strong-scaling curves still start at 1 GPU
# even when GPU_MULTIPLIER > 1. Only the small models fit / were characterized at
# 1 GPU (the original sweep used 1 GPU only for levels 5-6).
EXTRA_1GPU_LEVELS = [5, 6]

# Upper guard on GPU count per cell. On JUWELS (OpenMPI) there is no Cray-MPICH
# >=2048-rank crash; the limit here is the machine budget. The proven JUWELS sweep
# ran cleanly to 2048 GPUs (512 nodes, largebooster). Cells above this are skipped
# and logged. Partition auto-switches to largebooster past BOOSTER_NODE_CAP nodes.
MAX_SAFE_GPUS = 2048

# Measured owned-DOF counts per level, ISOTROPIC (lat=rad=level). With the MT/2
# radial extent the effective dofs are these x 2**RADIAL_EXTRA (i.e. halved).
# level 11 extrapolated at x7.99.
DOFS_PER_LEVEL = {
    5: 1013958, 6: 7987590, 7: 63406854, 8: 505284102,
    9: 4034399238, 10: 32243718150, 11: 257627107000,
}

def dofs_at(level: int) -> float | None:
    """Effective owned dofs for an MT/2-radial cell at `level` (None if unknown)."""
    iso = DOFS_PER_LEVEL.get(level)
    return iso * 2 ** RADIAL_EXTRA if iso is not None else None

# --- Weak-scaling diagonals (constant work per GPU) ---
# Hold dofs/GPU (and subdomains/GPU) fixed by stepping the model up one level
# (x8 dofs) and the GPU count up x8 together. Anchored at level 7.
#   base_gpus 1 -> ~63 M/GPU (iso), 2 -> ~32 M, 4 -> ~16 M, 8 -> ~8 M, ...
WEAK_BASE_LEVEL    = 7
WEAK_MAX_LEVEL     = 11
WEAK_DEFAULT_FILLS = [63, 32, 16, 8]     # target M-dofs/GPU diagonals to emit
WEAK_MAX_GPUS      = 4096                # skip points beyond this (1024 JUWELS nodes)

def weak_base_gpus(fill_mdofs: int, base_level: int = WEAK_BASE_LEVEL) -> int:
    """GPU count at the anchor level giving ~fill_mdofs dofs/GPU."""
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
# level (rad_sdr += 1 -> x2 subdomains), with GPUs x2. dofs/GPU and subdomains/GPU
# stay fixed, isolating radial scaling at 2x granularity.
RADIAL_LAT_LEVEL     = 8
RADIAL_LAT_SDR       = 1
# Anchor every diagonal at base_gpus = 8 (lat_sdr 1 -> rad_sdr 1 so
# base_gpus = 4^lat_sdr * 2^rad_sdr = 4*2 = 8), varying the starting radial level.
# 8 is set by the subdomain decomposition, not the node size: on JUWELS Booster
# that anchor spans 2 nodes (4 GPUs each).
RADIAL_DEFAULT_FILLS = [126, 63, 32, 16, 8]
RADIAL_BASE_GPUS     = 8   # = 2 JUWELS Booster nodes (subdomain-derived, not node-derived)
RADIAL_BASE_RADSDR   = 1   # base_gpus = 4^1 * 2^1 = 8 with lat_sdr=1

def radial_start_offset(fill_mdofs: int) -> int:
    """Radial extra levels at the base anchor that hit `fill_mdofs` per GPU."""
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

# JUWELS Booster: each node has 4 A100 GPUs and 48 CPU cores (2x 24-core EPYC).
# booster allocates GPUs via --gres; we bind one rank per GPU with --gpu-bind=closest
# and give each rank 12 cores. Account / partition can be overridden via env.
GPUS_PER_NODE    = 4
CPUS_PER_TASK    = 12   # 48 cores / 4 GPUs per node
ACCOUNT          = os.environ.get("SLURM_ACCOUNT", "walberlamovinggeo")
# Regular booster caps a job at 384 nodes; above that needs largebooster.
BOOSTER_NODE_CAP = 384

def partition_for(nodes: int) -> str:
    return "largebooster" if nodes > BOOSTER_NODE_CAP else "booster"

# Walltime cap. The fixed-work run (10 steps x 10 FGMRES + 50 EV) finishes in a
# few minutes even for the heaviest cells, so 15 min is ample headroom and gives
# much better backfill priority than a multi-hour limit. Overridable with --time-limit.
def time_limit_for(level: int) -> str:
    return "00:15:00"

# The app applies one subdomain-refinement level to BOTH axes:
# total subdomains = 10 * 4^sdr (lateral) * 2^sdr (radial) = 10 * 8^sdr.
def subdomains_for(sdr: int) -> int:
    return 10 * (8 ** sdr)

# Pick the smallest sdr s.t. subdomains >= n_gpus and divides n_gpus evenly
# (balanced rank distribution, least over-decomposition).
def choose_sdr(n_gpus: int) -> int:
    for sdr in range(0, 7):
        subs = subdomains_for(sdr)
        if subs >= n_gpus and subs % n_gpus == 0:
            return sdr
    raise ValueError(f"no balanced subdomain refinement found for n_gpus={n_gpus}")

def gpu_to_node_layout(n_gpus: int) -> tuple[int, int]:
    """Map a target n_gpus to (#nodes, #ntasks_per_node).

    Up to 4 GPUs we pack onto a single node; beyond that we use full nodes
    (4 GPUs each).
    """
    if n_gpus <= GPUS_PER_NODE:
        return (1, n_gpus)
    if n_gpus % GPUS_PER_NODE != 0:
        raise ValueError(f"n_gpus={n_gpus} is not a multiple of {GPUS_PER_NODE} for >1 nodes")
    return (n_gpus // GPUS_PER_NODE, GPUS_PER_NODE)

def write_juwels_job(cell_tag: str, config: Path, out_path: Path, nodes: int, tpn: int,
                     time_limit: str, app_args: str, echo_line: str) -> Path:
    """Per-cell sbatch script for one mantlecirculation cell on JUWELS Booster."""
    job_path  = JOB_DIR / f"bench_{cell_tag}.sh"
    partition = partition_for(nodes)
    script = f"""#!/bin/bash -l
#SBATCH --job-name=bench_{cell_tag}
#SBATCH --output=bench_{cell_tag}.o%j
#SBATCH --error=bench_{cell_tag}.e%j
#SBATCH -D {JOB_DIR}
#SBATCH --partition={partition}
#SBATCH --account={ACCOUNT}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tpn}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --gres=gpu:{tpn}
#SBATCH --time={time_limit}

echo "{echo_line}"

module load Stages/2025
module load CUDA/12
module load NVHPC/25.5-CUDA-12
module load OpenMPI/4.1.8

export OMPI_MCA_pml=ucx
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc
export OMP_NUM_THREADS=1
ulimit -c 0

# mantlecirculation writes its output tree under --outdir; cd there to capture any
# CWD-relative artifacts (timing trees) too.
mkdir -p {out_path}
cd {out_path}

srun --gpu-bind=closest {BINARY} --config {config} {app_args} --outdir {out_path} --outdir-overwrite
"""
    job_path.write_text(script)
    job_path.chmod(0o755)
    return job_path

def render_job_script(level: int, n_gpus: int, opts) -> tuple[Path, Path, dict]:
    mt_label   = 2 ** level
    rad_level  = level + RADIAL_EXTRA
    cell_tag   = f"MT{mt_label}_g{n_gpus}"
    out_path   = OUTPUT_ROOT / cell_tag
    nodes, tpn = gpu_to_node_layout(n_gpus)
    sdr        = choose_sdr(n_gpus)
    mesh_min   = mesh_min_for(sdr, sdr, RADIAL_EXTRA)

    info = dict(
        cell_tag=cell_tag, mt_label=mt_label,
        level=level, mesh_min=mesh_min, rad_level=rad_level, sdr=sdr,
        max_timesteps=opts.max_timesteps, fgmres=opts.fgmres, ev_iters=opts.ev_iters,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
    )
    app_args = (
        f"--refinement-level-mesh-min {mesh_min} --refinement-level-mesh-max {level} "
        f"--refinement-level-subdomains {sdr} --radial-extra-levels {RADIAL_EXTRA} "
        + solver_overrides(opts.max_timesteps, opts.fgmres, opts.ev_iters)
    )
    echo_line = (f"Cell: {cell_tag}  mesh=[{mesh_min}..{level}]  rad_level={rad_level}  sdr={sdr}  "
                 f"subdomains={subdomains_for(sdr)}  steps={opts.max_timesteps}  "
                 f"fgmres={opts.fgmres}  ev={opts.ev_iters}  n_gpus={n_gpus}  "
                 f"nodes={nodes}x{tpn}  partition={partition_for(nodes)}")
    time_limit = opts.time_limit or time_limit_for(level)
    job_path = write_juwels_job(cell_tag, opts.config, out_path, nodes, tpn,
                                time_limit, app_args, echo_line)
    return job_path, out_path, info

def render_radial_job_script(cell: dict, opts) -> tuple[Path, Path, dict]:
    """One step of a radial-only weak diagonal: lateral fixed, radial level and
    radial subdomains grow by k, GPUs by 2**k. dofs and GPUs both x2 per step."""
    lat_level  = cell["lat_level"]
    lat_sdr    = cell["lat_sdr"]
    rad_level  = cell["rad_level"]
    rad_sdr    = cell["rad_sdr"]
    k          = cell["radial_extra"]
    n_gpus     = cell["n_gpus"]
    nodes, tpn = gpu_to_node_layout(n_gpus)
    subdomains = 10 * 4 ** lat_sdr * 2 ** rad_sdr
    mesh_min   = mesh_min_for(lat_sdr, rad_sdr, k)
    cell_tag   = f"RAD_l{lat_level}r{rad_level}_g{n_gpus}"
    out_path   = OUTPUT_ROOT / cell_tag

    info = dict(
        cell_tag=cell_tag, lat_level=lat_level, mesh_min=mesh_min, rad_level=rad_level,
        lat_sdr=lat_sdr, rad_sdr=rad_sdr, radial_extra=k,
        max_timesteps=opts.max_timesteps, fgmres=opts.fgmres, ev_iters=opts.ev_iters,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
        subdom_per_gpu=subdomains // n_gpus,
    )
    app_args = (
        f"--refinement-level-mesh-min {mesh_min} --refinement-level-mesh-max {lat_level} "
        f"--radial-extra-levels {k} --lat-sdr {lat_sdr} --rad-sdr {rad_sdr} "
        + solver_overrides(opts.max_timesteps, opts.fgmres, opts.ev_iters)
    )
    echo_line = (f"Cell: {cell_tag}  mesh=[{mesh_min}..{lat_level}]  rad_level={rad_level}  "
                 f"lat_sdr={lat_sdr}  rad_sdr={rad_sdr}  subdomains={subdomains}  "
                 f"steps={opts.max_timesteps}  fgmres={opts.fgmres}  ev={opts.ev_iters}  "
                 f"n_gpus={n_gpus}  nodes={nodes}x{tpn}  partition={partition_for(nodes)}")
    time_limit = opts.time_limit or time_limit_for(rad_level)
    job_path = write_juwels_job(cell_tag, opts.config, out_path, nodes, tpn,
                                time_limit, app_args, echo_line)
    return job_path, out_path, info

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="only emit the job scripts; don't sbatch")
    p.add_argument("--levels", type=int, nargs="+", default=None,
                   help="override which mesh levels to submit (default: full sweep)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"base config TOML (default: {DEFAULT_CONFIG.name})")
    p.add_argument("--max-timesteps", type=int, default=DEFAULT_MAX_TIMESTEPS,
                   help=f"timesteps per cell (default: {DEFAULT_MAX_TIMESTEPS})")
    p.add_argument("--fgmres", type=int, default=DEFAULT_FGMRES,
                   help=f"Stokes FGMRES iterations per timestep (default: {DEFAULT_FGMRES})")
    p.add_argument("--ev-iters", type=int, default=DEFAULT_EV_ITERS,
                   help=f"energy (EV) Krylov iterations per timestep (default: {DEFAULT_EV_ITERS})")
    p.add_argument("--time-limit", type=str, default=None,
                   help="override the SLURM walltime for every cell (e.g. 03:00:00)")
    p.add_argument("--weak", action="store_true",
                   help="weak-scaling diagonals at constant dofs/GPU "
                        "(n_gpus = base * 8^(level-7)) instead of the strong grid")
    p.add_argument("--weak-fills", type=int, nargs="+", default=WEAK_DEFAULT_FILLS,
                   help=f"target dofs/GPU per diagonal, in millions "
                        f"(default: {WEAK_DEFAULT_FILLS})")
    p.add_argument("--weak-max-level", type=int, default=WEAK_MAX_LEVEL,
                   help=f"top level for --weak (default: {WEAK_MAX_LEVEL}; "
                        f"each +1 level is x8 GPUs)")
    p.add_argument("--weak-max-gpus", type=int, default=WEAK_MAX_GPUS,
                   help=f"skip --weak / --weak-radial points above this many GPUs "
                        f"(default: {WEAK_MAX_GPUS} = 1024 nodes)")
    p.add_argument("--weak-radial", action="store_true",
                   help="radial-only weak-scaling diagonals: lateral mesh fixed, radial "
                        "level+subdomains and GPUs x2 per step (dofs x2/step, constant dofs/GPU)")
    p.add_argument("--weak-radial-fills", type=int, nargs="+", default=RADIAL_DEFAULT_FILLS,
                   help=f"target dofs/GPU per radial diagonal, in millions "
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
        over = [c for c in cells if c["n_gpus"] > MAX_SAFE_GPUS]
        if over:
            print(f"  SKIP {len(over)} radial cell(s) > {MAX_SAFE_GPUS} GPUs (machine budget)")
            cells = [c for c in cells if c["n_gpus"] <= MAX_SAFE_GPUS]
        render_list = [("radial", c) for c in cells]
        print(f"radial-only weak-scaling diagonals (lateral fixed at level {RADIAL_LAT_LEVEL}, "
              f"lat_sdr {RADIAL_LAT_SDR}; fills {args.weak_radial_fills} M-dofs/GPU): {len(cells)} cells")
        dpg0 = DOFS_PER_LEVEL.get(RADIAL_LAT_LEVEL)
        for c in cells:
            n = c["n_gpus"]
            nodes, _ = gpu_to_node_layout(n)
            subdom = 10 * 4 ** c["lat_sdr"] * 2 ** c["rad_sdr"]
            dpg_s = f"  dofs/GPU={dpg0 * 2**c['radial_extra'] / n / 1e6:5.1f}M" if dpg0 else ""
            print(f"  fill={c['fill']:2}M  rad_level={c['rad_level']:2}  rad_sdr={c['rad_sdr']}  "
                  f"n_gpus={n:5}  nodes={nodes:4}  subdom/GPU={subdom // n}{dpg_s}")
    else:
        if args.weak:
            cells = weak_cells(args.weak_fills, WEAK_BASE_LEVEL,
                               args.weak_max_level, args.weak_max_gpus)
        else:
            cells = [(level, n_gpus * GPU_MULTIPLIER)
                     for level, gpu_list in DEFAULT_SWEEP
                     for n_gpus in gpu_list]
            cells += [(lvl, 1) for lvl in EXTRA_1GPU_LEVELS]
            seen = set()
            cells = sorted(c for c in cells if not (c in seen or seen.add(c)))
        if args.levels is not None:
            cells = [(level, n_gpus) for (level, n_gpus) in cells if level in args.levels]
        over = [(l, n) for (l, n) in cells if n > MAX_SAFE_GPUS]
        if over:
            print(f"  SKIP {len(over)} cell(s) > {MAX_SAFE_GPUS} GPUs (machine budget): "
                  + ", ".join(f"MT{2**l}_g{n}" for l, n in over))
            cells = [(l, n) for (l, n) in cells if n <= MAX_SAFE_GPUS]
        render_list = [("iso", c) for c in cells]
        if args.weak:
            print(f"weak-scaling diagonals (fills {args.weak_fills} M-dofs/GPU): {len(cells)} cells")
        else:
            print(f"strong-scaling grid (radial = MT/2, rad_level = lat_level-1): {len(cells)} cells")
        for (level, n_gpus) in cells:
            sdr = choose_sdr(n_gpus)
            nodes, _ = gpu_to_node_layout(n_gpus)
            dpg = dofs_at(level)
            dpg_s = f"  dofs/GPU={dpg / n_gpus / 1e6:5.1f}M" if dpg else ""
            print(f"  MT{2**level:<5} lat={level:2} rad={level+RADIAL_EXTRA:2}  n_gpus={n_gpus:5}  "
                  f"nodes={nodes:4}  sdr={sdr}  subdom/GPU={10 * 8**sdr // n_gpus}{dpg_s}")

    submitted = []
    for kind, item in render_list:
        if kind == "radial":
            job_path, out_path, info = render_radial_job_script(item, args)
        else:
            level, n_gpus = item
            job_path, out_path, info = render_job_script(level, n_gpus, args)
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
            header = ("# cell_tag\tjob_id\tlevel\trad_level\tsdr\tmax_timesteps\tfgmres\tev_iters"
                      "\tn_gpus\tnodes\ttasks_per_node\n")
            fields = ["level", "rad_level", "sdr", "max_timesteps", "fgmres", "ev_iters",
                      "n_gpus", "nodes", "tasks_per_node"]
        with open(manifest, "w") as f:
            f.write(header)
            for info, jobid in submitted:
                f.write("\t".join([info["cell_tag"], jobid] +
                                  [str(info[k]) for k in fields]) + "\n")
        print(f"\nWrote manifest: {manifest}")

if __name__ == "__main__":
    main(sys.argv[1:])
