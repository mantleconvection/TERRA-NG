#!/usr/bin/env python3
"""
Strong-scale benchmark driver for the *mantlecirculation* solver on MareNostrum 5 (ACC).

MareNostrum 5 port of submit_bench_mt_jb.py / submit_bench_mt_lumi.py. Runs the full
`mantlecirculation` app (Stokes + energy time stepping) at an increasing MT model
size. Each (MT-level, n_gpus) cell does CONSTANT solver work: a fixed number of
timesteps, each with a fixed FGMRES (Stokes) and EV (energy) iteration count, so
wall-clock differences reflect mesh/parallel scaling only -- not convergence
variability.

The cell-generation logic (DEFAULT_SWEEP strong grid, --weak diagonals,
--weak-radial diagonals) is identical to the JUWELS/LUMI drivers; only the platform
layer differs:
  - 4 GPUs/node (MareNostrum 5 ACC: 4 NVIDIA H100 64GB per node), same as JUWELS
  - account ehpc433, qos acc_ehpc (no partition); optional reservation
  - mpirun --bind-to none + a per-rank gpu_bind.sh wrapper (CUDA_VISIBLE_DEVICES),
    mirroring the run_test_*.sh launch recipe terraneo-build emits for this machine
  - compenv-acc/nvhpx23.11 module (NVHPC); SLURM_CPU_BIND=none

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

Decomposition: the default is the asymmetric lat_sdr/rad_sdr split (subdomains/GPU held
constant; see choose_lat_rad). Pass --isotropic for the original uniform single-knob
--refinement-level-subdomains s decomposition (lat_sdr=rad_sdr=s), tagged _iso.

Solver mode (--mode): std (default) runs the double-precision Krylov basis; low-mem runs
the FP16/BF16 Krylov basis + lowered restart/smoothing (needs the mc-float-krylov binary)
to cut the FGMRES workspace -- the dominant memory term at high dofs/GPU. Cells are tagged
_std / _lowmem so the two never collide.

For each (MT-level, n_gpus) cell:
  - emit a per-cell sbatch script under bench_mt/jobs_mn5/
  - submit via sbatch
  - the app writes its output under bench_mt/outputs_mn5/MT<...>_g<n>/ (--outdir)

Usage:
    python3 submit_bench_mt_mn5.py            # submit the default strong grid
    python3 submit_bench_mt_mn5.py --dry-run  # just print what would be submitted
    python3 submit_bench_mt_mn5.py --levels 8 9    # only those mesh levels
    python3 submit_bench_mt_mn5.py --isotropic --mode low-mem   # uniform sdr, FP16 low-mem
    python3 submit_bench_mt_mn5.py --config <base.toml> --max-timesteps 10
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
# mantlecirculation binary (MareNostrum 5 build). Override with $TERRANEO_BIN.
BINARY      = Path(os.environ.get(
    "TERRANEO_BIN",
    os.path.expanduser("~/terraneo-build/apps/mantlecirculation/mantlecirculation")))
# Job scripts, SLURM logs, and app outputs. Distinct from the operator-benchmark
# (submit_bench_mt.py) jobs/ and outputs/ so the two sweeps don't clobber each
# other. The fixed-work solver run disables XDMF/diagnostic output (see
# solver_overrides), so the footprint stays light and $HOME is fine; override the
# root with $BENCH_MT_ROOT.
ROOT        = Path(os.environ.get("BENCH_MT_ROOT", str(BENCH_DIR)))
JOB_DIR     = ROOT / "jobs_mn5"
OUTPUT_ROOT = ROOT / "outputs_mn5"

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

# (level, list of base GPU counts). Actual n_gpus = base * GPU_MULTIPLIER.
# level -> MT label = 2^level (lateral). Levels 7-11 span a wide GPU range so each
# MT model has several strong-scaling points on both sides of its sweet spot.
DEFAULT_SWEEP: list[tuple[int, list[int]]] = [
    # (level, [base n_gpus])      lat=2^level, rad=2^(level-1) (MT/2); actual = base*2
    (5,  [1, 2, 4, 8]),                       # MT32    actual 2..16  (+g1)
    (6,  [1, 2, 4, 8, 16]),                   # MT64    actual 2..32  (+g1)
    (7,  [1, 2, 4, 8, 16, 32, 64, 128]),      # MT128   actual 2..256   (+g1,g2 probe)
    (8,  [2, 4, 8, 16, 32, 64, 128, 256, 512]),     # MT256   actual 4..1024  (+g2,g4 probe)
    # Larger models (MT512+): one denser small-node cell prepended (mirrors the LUMI/JUWELS
    # drivers). These high-dofs/GPU points cover the smallest node count the model fits on;
    # the H100 64 GB reaches them with --mode low-mem (the FP16 Krylov basis cuts the FGMRES
    # workspace). actual n_gpus = base * 2.
    (9,  [4, 8, 16, 32, 64, 128, 256, 512, 1024]),  # MT512   actual 8..2048 (+g4,g8 probe; g32 dense)
    (10, [64, 128, 256, 512, 1024]),          # MT1024  actual 128..2048 (added g128 = 32 nodes)
    (11, [256, 512, 1024, 2048, 4096]),       # MT2048  actual 512..8192 (added g512 = 128 nodes)
    (12, [256, 512, 1024, 2048, 4096]),       # MT4096  actual 512..8192 (probes; see below)
    # ^ MT4096 (~1.03e12 effective dofs) is a heavy probe even for the 64 GB H100s: at the
    #   reachable node counts dofs/GPU stays high (g512=128 nodes ~2010 M/GPU, g1024=256
    #   nodes ~1005 M/GPU, g2048=512 nodes ~502 M/GPU). Expect these to need --mode low-mem
    #   and likely still OOM at the smaller node counts; g4096/g8192 exceed MAX_SAFE_GPUS and
    #   are skipped.
]

# Multiply every strong-grid GPU count by this (each cell runs at GPU_MULTIPLIER x
# the base n_gpus, i.e. less work per GPU / further-right strong scaling).
GPU_MULTIPLIER = 2

# Always include a single-GPU datapoint for these (small) levels, on top of the
# multiplied grid, so the small-model strong-scaling curves still start at 1 GPU
# even when GPU_MULTIPLIER > 1. Only the small models fit / were characterized at
# 1 GPU (the original sweep used 1 GPU only for levels 5-6).
EXTRA_1GPU_LEVELS = [5, 6]

# Upper guard on GPU count per cell. The limit here is the machine budget, not an
# MPI ceiling (OpenMPI/NVHPC over the SS-11 fabric scales past 2048 ranks). Cells
# above this are skipped and logged. Override with $BENCH_MAX_SAFE_GPUS.
MAX_SAFE_GPUS = int(os.environ.get("BENCH_MAX_SAFE_GPUS", "2048"))

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

# --- Weak-scaling diagonals (constant work per GPU) ---
# Hold dofs/GPU (and subdomains/GPU) fixed by stepping the model up one level
# (x8 dofs) and the GPU count up x8 together. Anchored at level 7.
#   base_gpus 1 -> ~63 M/GPU (iso), 2 -> ~32 M, 4 -> ~16 M, 8 -> ~8 M, ...
WEAK_BASE_LEVEL    = 7
WEAK_MAX_LEVEL     = 11
WEAK_DEFAULT_FILLS = [63, 32, 16, 8]     # target M-dofs/GPU diagonals to emit
WEAK_MAX_GPUS      = 4096                # skip points beyond this (1024 nodes)

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
# 8 is set by the subdomain decomposition, not the node size: on MareNostrum 5
# (4 GPUs/node) that anchor spans 2 nodes.
RADIAL_DEFAULT_FILLS = [126, 63, 32, 16, 8]
RADIAL_BASE_GPUS     = 8   # = 2 MareNostrum 5 nodes (subdomain-derived, not node-derived)
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

# Solver precision / restart presets (--mode), ported from the JUWELS/LUMI drivers.
# Requires the mc-float-krylov binary: --stokes/--energy-float-krylov-basis store the
# Krylov basis in FP16 (native BF16 since the Kokkos 5.1.0 upgrade), which removes the
# dominant memory term (the FGMRES workspace) at high dofs/GPU.
#   low-mem : FP16 (float) Krylov basis for Stokes + energy, Stokes restart lowered to 5
#             (energy restart is already 5 by default), and a single pre/post smoothing
#             step in the velocity multigrid. Minimises both the FGMRES workspace and the
#             smoother cost.
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

# MareNostrum 5 ACC: each node has 4 NVIDIA H100 64GB GPUs and 2 Intel Sapphire
# Rapids sockets (40 cores each = 80 cores/node). One rank per GPU; 20 cores/rank.
# SLURM account / qos / reservation / module mirror the run_test_*.sh scripts that
# terraneo-build emits for this machine; each is overridable via env.
GPUS_PER_NODE = 4
CPUS_PER_TASK = 20                                              # 80 cores / 4 GPUs
ACCOUNT       = os.environ.get("SLURM_ACCOUNT", "ehpc433")
QOS           = os.environ.get("SLURM_QOS", "acc_ehpc")               # production acc queue
RESERVATION   = os.environ.get("SLURM_RESERVATION", "")              # set to a reservation name to use one
MODULES       = os.environ.get("BENCH_MODULES", "compenv-acc/nvhpx23.11")

# Walltime cap. The fixed-work run (10 steps x 10 FGMRES + 50 EV) finishes in a
# few minutes even for the heaviest cells, so 15 min is ample headroom and gives
# much better backfill priority than a multi-hour limit. Overridable with --time-limit.
def time_limit_for(level: int) -> str:
    return "00:15:00"

# Subdomain decomposition with INDEPENDENT lateral/radial refinement:
# total subdomains = 10 * 4^lat_sdr * 2^rad_sdr = 10 * 2^(2*lat_sdr + rad_sdr).
# Refining the two axes separately gives x2 granularity (vs x8 for a single coupled
# level), which matches the x2 GPU steps -- so subdomains/GPU can be held CONSTANT
# across a strong-scaling sweep instead of swinging ~2.5x..20x. A coupled single
# level jumps the decomposition x8 per +1 GPU-doubling-pair, producing the
# over-decomposition cliffs (a cell with more GPUs but 4x the subdomains/GPU can run
# *slower* despite identical global DoFs).
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

    Up to 4 GPUs we pack onto a single node; beyond that we use full nodes
    (4 H100 each).
    """
    if n_gpus <= GPUS_PER_NODE:
        return (1, n_gpus)
    if n_gpus % GPUS_PER_NODE != 0:
        raise ValueError(f"n_gpus={n_gpus} is not a multiple of {GPUS_PER_NODE} for >1 nodes")
    return (n_gpus // GPUS_PER_NODE, GPUS_PER_NODE)

def write_mn5_job(cell_tag: str, config: Path, out_path: Path, nodes: int, tpn: int,
                  time_limit: str, app_args: str, echo_line: str) -> Path:
    """Per-cell sbatch script for one mantlecirculation cell on MareNostrum 5 (ACC).

    Mirrors the run_test_*.sh launch recipe from terraneo-build: nvhpc module,
    one rank per H100 via mpirun + a gpu_bind.sh wrapper, no explicit partition.
    """
    job_path     = JOB_DIR / f"bench_{cell_tag}.sh"
    ntasks       = nodes * tpn
    account_line = f"#SBATCH --account={ACCOUNT}\n" if ACCOUNT else ""
    qos_line     = f"#SBATCH --qos={QOS}\n" if QOS else ""
    resv_line    = f"#SBATCH --reservation={RESERVATION}\n" if RESERVATION else ""
    module_block = f"\nmodule load {MODULES}\n" if MODULES else ""
    script = f"""#!/bin/bash
#SBATCH --job-name=bench_{cell_tag}
#SBATCH --output=bench_{cell_tag}.o%j
#SBATCH --error=bench_{cell_tag}.e%j
#SBATCH -D {JOB_DIR}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --gres=gpu:{tpn}
#SBATCH --time={time_limit}
{account_line}{qos_line}{resv_line}
echo "{echo_line}"
{module_block}
export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=1
ulimit -c 0

# mantlecirculation writes its output tree under --outdir; cd there to capture any
# CWD-relative artifacts (timing trees) too.
mkdir -p {out_path}
cd {out_path}

# Per-GPU binding wrapper (one rank per H100 via CUDA_VISIBLE_DEVICES).
cat > ./gpu_bind.sh << 'EOF'
#!/bin/bash
local_rank=${{OMPI_COMM_WORLD_LOCAL_RANK:-${{SLURM_LOCALID}}}}
local_rank=$(( local_rank % {GPUS_PER_NODE} ))
export CUDA_VISIBLE_DEVICES="${{local_rank}}"
"$@"
EOF
chmod +x ./gpu_bind.sh

mpirun --bind-to none -np {ntasks} ./gpu_bind.sh {BINARY} --config {config} {app_args} --outdir {out_path} --outdir-overwrite

rm -f ./gpu_bind.sh
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
        mode=opts.mode,
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
                 f"subdom/GPU={subdomains // n_gpus}  mode={opts.mode}  "
                 f"steps={opts.max_timesteps}  fgmres={opts.fgmres}  ev={opts.ev_iters}  "
                 f"n_gpus={n_gpus}  nodes={nodes}x{tpn}  qos={QOS}")
    time_limit = opts.time_limit or time_limit_for(level)
    job_path = write_mn5_job(cell_tag, opts.config, out_path, nodes, tpn,
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
    cell_tag   = f"RAD_l{lat_level}r{rad_level}_g{n_gpus}{mode_suffix(opts.mode)}"
    out_path   = OUTPUT_ROOT / cell_tag

    info = dict(
        cell_tag=cell_tag, lat_level=lat_level, mesh_min=mesh_min, rad_level=rad_level,
        lat_sdr=lat_sdr, rad_sdr=rad_sdr, radial_extra=k, mode=opts.mode,
        max_timesteps=opts.max_timesteps, fgmres=opts.fgmres, ev_iters=opts.ev_iters,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
        subdom_per_gpu=subdomains // n_gpus,
    )
    app_args = (
        f"--refinement-level-mesh-min {mesh_min} --refinement-level-mesh-max {lat_level} "
        f"--radial-extra-levels {k} --lat-sdr {lat_sdr} --rad-sdr {rad_sdr} "
        + solver_overrides(opts.max_timesteps, opts.fgmres, opts.ev_iters)
        + mode_overrides(opts.mode)
    )
    echo_line = (f"Cell: {cell_tag}  mesh=[{mesh_min}..{lat_level}]  rad_level={rad_level}  "
                 f"lat_sdr={lat_sdr}  rad_sdr={rad_sdr}  subdomains={subdomains}  "
                 f"mode={opts.mode}  steps={opts.max_timesteps}  fgmres={opts.fgmres}  "
                 f"ev={opts.ev_iters}  n_gpus={n_gpus}  nodes={nodes}x{tpn}  qos={QOS}")
    time_limit = opts.time_limit or time_limit_for(rad_level)
    job_path = write_mn5_job(cell_tag, opts.config, out_path, nodes, tpn,
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
                        "(applied after the GPU_MULTIPLIER; strong/weak grid only)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                   help=f"base config TOML (default: {DEFAULT_CONFIG.name})")
    p.add_argument("--mode", choices=["std", "low-mem"], default="std",
                   help="solver precision/restart preset: std = double Krylov basis + "
                        "Stokes restart 10; low-mem = FP16 basis + restart 5 (needs the "
                        "mc-float-krylov binary). The cell tag (output dir / job script / "
                        "job-name) is suffixed _std or _lowmem.")
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
    print(f"SLURM: account={ACCOUNT}  qos={QOS}"
          + (f"  reservation={RESERVATION}" if RESERVATION else "  (no reservation)"))
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
        if args.gpus is not None:
            cells = [(level, n_gpus) for (level, n_gpus) in cells if n_gpus in args.gpus]
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
            dpg_s = f"  dofs/GPU={dpg / n_gpus / 1e6:5.1f}M" if dpg else ""
            print(f"  MT{2**level:<5} lat={level:2} rad={level+RADIAL_EXTRA:2}  n_gpus={n_gpus:5}  "
                  f"nodes={nodes:4}  lat_sdr={lat_sdr} rad_sdr={rad_sdr}  "
                  f"subdom/GPU={subs // n_gpus:2}{dpg_s}")

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
            manifest = JOB_DIR / f"manifest_radial{mode_suffix(args.mode)}.txt"
            header = ("# cell_tag\tjob_id\tlat_level\trad_level\tlat_sdr\trad_sdr\tmode"
                      "\tmax_timesteps\tfgmres\tev_iters\tn_gpus\tnodes\ttasks_per_node\n")
            fields = ["lat_level", "rad_level", "lat_sdr", "rad_sdr", "mode",
                      "max_timesteps", "fgmres", "ev_iters", "n_gpus", "nodes", "tasks_per_node"]
        else:
            manifest = JOB_DIR / (f"manifest_weak{mode_suffix(args.mode)}.txt" if args.weak
                                  else f"manifest{mode_suffix(args.mode)}.txt")
            header = ("# cell_tag\tjob_id\tlevel\trad_level\tlat_sdr\trad_sdr\tsubdom_per_gpu\tmode"
                      "\tmax_timesteps\tfgmres\tev_iters\tn_gpus\tnodes\ttasks_per_node\n")
            fields = ["level", "rad_level", "lat_sdr", "rad_sdr", "subdom_per_gpu", "mode",
                      "max_timesteps", "fgmres", "ev_iters",
                      "n_gpus", "nodes", "tasks_per_node"]
        with open(manifest, "w") as f:
            f.write(header)
            for info, jobid in submitted:
                f.write("\t".join([info["cell_tag"], jobid] +
                                  [str(info[k]) for k in fields]) + "\n")
        print(f"\nWrote manifest: {manifest}")

if __name__ == "__main__":
    main(sys.argv[1:])
