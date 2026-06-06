#!/usr/bin/env python3
"""
Strong-scale benchmark driver for the *operator* microbenchmark on JUWELS Booster.

Runs the `benchmark_operators` app (matvec throughput for the production
EpsDivDivKerngen operator) at an increasing MT model size. For each
(MT-level, n_gpus) cell in the sweep:
  - emit a per-cell sbatch script under bench_mt/jobs/
  - submit via sbatch
  - the app writes its CSV / timer-tree under bench_mt/outputs/MT<...>_g<n>/{csv,tts}/

MT-level convention: MT(N) where N = 2^level is the lateral (and, for this app,
also the radial) resolution. benchmark_operators builds the domain with
`create_uniform(level, level, ..., sdr, sdr)` so lateral and radial use the SAME
refinement level and the SAME subdomain refinement -- there is no radial-extra
offset like the full mc app has.

Usage:
    python3 submit_bench_mt.py            # submit the default sweep
    python3 submit_bench_mt.py --dry-run  # just print what would be submitted
    python3 submit_bench_mt.py --levels 8 9    # only those mesh levels
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
# benchmark_operators binary (JUWELS Booster build). Override with $BENCH_OPERATORS_BIN.
BINARY      = Path(os.environ.get(
    "BENCH_OPERATORS_BIN",
    "/p/home/jusers/boehm2/juwels/terraneo-build/apps/benchmarks/performance/benchmark_operators"))
JOB_DIR     = BENCH_DIR / "jobs"
OUTPUT_ROOT = BENCH_DIR / "outputs"

# (level, list of GPU counts to test). level -> MT label = 2^level.
DEFAULT_SWEEP: list[tuple[int, list[int]]] = [
    # (level, [n_gpus])
    (5,  [1, 2, 4, 8]),                  # MT32    (lat=rad=32)
    (6,  [1, 2, 4, 8, 16]),              # MT64    (lat=rad=64)
    (7,  [4, 8, 16, 32]),                # MT128   (lat=rad=128)
    (8,  [16, 32, 64, 128, 256]),        # MT256   (lat=rad=256)
    (9,  [64, 128, 256, 512]),           # MT512   (lat=rad=512)
    (10, [256, 512, 1024, 2048]),        # MT1024  (lat=rad=1024)
]

# Measured owned-DOF counts per level (from the strong sweep CSVs). Used to map a
# target dofs/GCD to a base GPU count and to print the achieved fill. level 11
# extrapolated at x7.99.
DOFS_PER_LEVEL = {
    5: 1013958, 6: 7987590, 7: 63406854, 8: 505284102,
    9: 4034399238, 10: 32243718150, 11: 257627107000,
}

# --- Weak-scaling diagonals (constant work per GCD) ---
# Hold dofs/GCD (and subdomains/GCD) fixed by stepping the model up one level
# (x8 dofs) and the GCD count up x8 together. Anchored at level 7: the fill is
# dofs(7)/base_gpus, and choose_sdr then keeps level - sdr and subdomains/GCD
# constant along the whole diagonal -> n_gpus = base_gpus * 8^(level-7).
#   base_gpus 1 -> ~63 M/GCD, 2 -> ~32 M, 4 -> ~16 M, 8 -> ~8 M, ...
WEAK_BASE_LEVEL    = 7
WEAK_MAX_LEVEL     = 10                  # 10 -> up to 512 GCDs at 63 M/GCD
WEAK_DEFAULT_FILLS = [63, 32, 16, 8]     # target M-dofs/GCD diagonals to emit
WEAK_MAX_GPUS      = 2048                # skip points beyond this (256 nodes)

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
# stay fixed, isolating radial scaling at 2x (vs 8x) granularity.
#
# The fill (dofs/GCD) is set by the base GPU count at the anchor (rad_level =
# lat_level): dofs(lat 8 isotropic)/base_gpus. base_gpus is snapped to a power of
# two and clamped to >= GPUS_PER_NODE so the diagonal STARTS AT ONE FULL NODE;
# base_rad_sdr is chosen so subdom/GCD = 10 is held across all fills
# (base_gpus = 4^lat_sdr * 2^rad_sdr). With lat 8 / lat_sdr 1 on JUWELS (4 GPUs/node):
#   base 4 (1 node) -> 126 M/GCD, 8 -> 63 M, 16 -> 32 M, 32 -> 16 M, ...
RADIAL_LAT_LEVEL     = 8
RADIAL_LAT_SDR       = 1
RADIAL_DEFAULT_FILLS = [126, 63, 32, 16]  # target M-dofs/GCD radial diagonals

def radial_base(fill_mdofs: int) -> tuple[int, int]:
    """(base_gpus, base_rad_sdr) at the anchor for a target dofs/GCD. base_gpus is
    clamped to >= GPUS_PER_NODE so the first datapoint is a single full node."""
    target       = DOFS_PER_LEVEL[RADIAL_LAT_LEVEL] / (fill_mdofs * 1e6)
    base_gpus    = max(GPUS_PER_NODE, 1 << max(0, round(math.log2(target))))
    base_rad_sdr = (base_gpus.bit_length() - 1) - 2 * RADIAL_LAT_SDR
    return base_gpus, max(0, base_rad_sdr)

def radial_cells(fills: list[int], max_gpus: int) -> list[dict]:
    cells: list[dict] = []
    for fill in fills:
        base_gpus, base_rad_sdr = radial_base(fill)
        k = 0
        while base_gpus * 2 ** k <= max_gpus:
            cells.append(dict(
                lat_level=RADIAL_LAT_LEVEL, lat_sdr=RADIAL_LAT_SDR,
                rad_level=RADIAL_LAT_LEVEL + k, rad_sdr=base_rad_sdr + k,
                radial_extra=k, n_gpus=base_gpus * 2 ** k, fill=fill))
            k += 1
    return cells

# Matvec repetitions per timing (matches the remeasure_history / 2gcd configs).
DEFAULT_EXECUTIONS = 5
# Untimed warmup matvecs before the timed region (amortizes launch/alloc overhead).
DEFAULT_WARMUP = 5

# JUWELS Booster: each node has 4 A100 GPUs. One rank per GPU, bound to the
# closest GPU. Account can be overridden via env.
GPUS_PER_NODE = 4
ACCOUNT       = os.environ.get("SLURM_ACCOUNT", "walberlamovinggeo")

# --partition=booster caps at 384 nodes per job; above that needs
# --partition=largebooster (same hardware, different policy).
BOOSTER_NODE_CAP = 384

# Walltime: the operator microbenchmark is a handful of matvecs, so it is fast;
# give the two largest models a little more for domain setup / allocation.
def time_limit_for(level: int) -> str:
    return "00:30:00" if level >= 9 else "00:15:00"

# benchmark_operators applies one subdomain-refinement level to BOTH axes:
# total subdomains = 10 * 4^sdr (lateral, x4/level) * 2^sdr (radial, x2/level)
#                  = 10 * 8^sdr.
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

def write_job(cell_tag: str, out_path: Path, nodes: int, tpn: int,
              time_limit: str, bench_args: str, echo_line: str) -> Path:
    """Write the JUWELS Booster sbatch script for one benchmark_operators cell."""
    job_path  = JOB_DIR / f"bench_{cell_tag}.sh"
    partition = "largebooster" if nodes > BOOSTER_NODE_CAP else "booster"
    script = f"""#!/bin/bash -l
#SBATCH --job-name=bench_{cell_tag}
#SBATCH --output=bench_{cell_tag}.o%j
#SBATCH --error=bench_{cell_tag}.e%j
#SBATCH -D {JOB_DIR}
#SBATCH --partition={partition}
#SBATCH --account={ACCOUNT}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tpn}
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:{tpn}
#SBATCH --time={time_limit}

echo "{echo_line}  partition={partition}"

module load Stages/2025
module load CUDA/12
module load NVHPC/25.5-CUDA-12
module load OpenMPI/4.1.8

export OMPI_MCA_pml=ucx
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc

# benchmark_operators writes csv/ and tts/ relative to CWD.
mkdir -p {out_path}/csv {out_path}/tts
cd {out_path}

srun --gpu-bind=closest {BINARY} {bench_args}
"""
    job_path.write_text(script)
    job_path.chmod(0o755)
    return job_path

def render_job_script(level: int, n_gpus: int, executions: int, warmup: int) -> tuple[Path, Path, dict]:
    mt_label   = 2 ** level
    cell_tag   = f"MT{mt_label}_g{n_gpus}"
    out_path   = OUTPUT_ROOT / cell_tag
    nodes, tpn = gpu_to_node_layout(n_gpus)
    sdr        = choose_sdr(n_gpus)

    info = dict(
        cell_tag=cell_tag, mt_label=mt_label,
        level=level, sdr=sdr, executions=executions,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
    )
    bench_args = (
        f"--min-level {level} --max-level {level} "
        f"--refinement-level-subdomains {sdr} --executions {executions} --warmup {warmup}"
    )
    echo_line = (f"Cell: {cell_tag}  level={level}  sdr={sdr}  subdomains={subdomains_for(sdr)}  "
                 f"executions={executions}  n_gpus={n_gpus}  nodes={nodes}x{tpn}")
    job_path = write_job(cell_tag, out_path, nodes, tpn,
                         time_limit_for(level), bench_args, echo_line)
    return job_path, out_path, info

def render_radial_job_script(cell: dict, executions: int, warmup: int) -> tuple[Path, Path, dict]:
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
    cell_tag   = f"RAD_l{lat_level}r{rad_level}_g{n_gpus}"
    out_path   = OUTPUT_ROOT / cell_tag

    info = dict(
        cell_tag=cell_tag, lat_level=lat_level, rad_level=rad_level,
        lat_sdr=lat_sdr, rad_sdr=rad_sdr, radial_extra=k, executions=executions,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
        subdom_per_gcd=subdomains // n_gpus,
    )
    bench_args = (
        f"--min-level {lat_level} --max-level {lat_level} "
        f"--radial-extra-levels {k} --lat-sdr {lat_sdr} --rad-sdr {rad_sdr} "
        f"--executions {executions} --warmup {warmup}"
    )
    echo_line = (f"Cell: {cell_tag}  lat_level={lat_level}  rad_level={rad_level}  "
                 f"lat_sdr={lat_sdr}  rad_sdr={rad_sdr}  subdomains={subdomains}  "
                 f"n_gpus={n_gpus}  nodes={nodes}x{tpn}")
    # radial mesh can be deep; give the larger radial levels the longer budget.
    job_path = write_job(cell_tag, out_path, nodes, tpn,
                         time_limit_for(rad_level), bench_args, echo_line)
    return job_path, out_path, info

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="only emit the job scripts; don't sbatch")
    p.add_argument("--levels", type=int, nargs="+", default=None,
                   help="override which mesh levels to submit (default: full sweep)")
    p.add_argument("--executions", type=int, default=DEFAULT_EXECUTIONS,
                   help=f"matvec repetitions per timing (default: {DEFAULT_EXECUTIONS})")
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                   help=f"untimed warmup matvecs before timing (default: {DEFAULT_WARMUP})")
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
                        f"(default: {WEAK_MAX_GPUS} = 256 nodes)")
    p.add_argument("--weak-radial", action="store_true",
                   help="radial-only weak-scaling diagonals: lateral mesh fixed, radial "
                        "level+subdomains and GPUs x2 per step (dofs x2/step, constant dofs/GCD)")
    p.add_argument("--weak-radial-fills", type=int, nargs="+", default=RADIAL_DEFAULT_FILLS,
                   help=f"target dofs/GCD per radial diagonal, in millions "
                        f"(default: {RADIAL_DEFAULT_FILLS})")
    args = p.parse_args(argv)

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Build a render list of (kind, item): 'iso' -> (level, n_gpus); 'radial' -> cell dict.
    if args.weak_radial:
        cells = radial_cells(args.weak_radial_fills, args.weak_max_gpus)
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
            cells = [(level, n_gpus)
                     for level, gpu_list in DEFAULT_SWEEP
                     for n_gpus in gpu_list]
        if args.levels is not None:
            cells = [(level, n_gpus) for (level, n_gpus) in cells if level in args.levels]
        render_list = [("iso", c) for c in cells]
        if args.weak:
            print(f"weak-scaling diagonals (fills {args.weak_fills} M-dofs/GCD): {len(cells)} cells")
        else:
            print(f"strong-scaling grid: {len(cells)} cells")
        for (level, n_gpus) in cells:
            sdr = choose_sdr(n_gpus)
            nodes, _ = gpu_to_node_layout(n_gpus)
            dpg = DOFS_PER_LEVEL.get(level)
            dpg_s = f"  dofs/GCD={dpg / n_gpus / 1e6:5.1f}M" if dpg else ""
            print(f"  level={level:2}  MT{2**level:<5}  n_gpus={n_gpus:5}  nodes={nodes:4}"
                  f"  sdr={sdr}  subdom/GCD={10 * 8**sdr // n_gpus}{dpg_s}")

    submitted = []
    for kind, item in render_list:
        if kind == "radial":
            job_path, out_path, info = render_radial_job_script(item, args.executions, args.warmup)
        else:
            level, n_gpus = item
            job_path, out_path, info = render_job_script(level, n_gpus, args.executions, args.warmup)
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
        # One manifest per mode so the three sweeps don't clobber each other.
        if args.weak_radial:
            manifest = JOB_DIR / "manifest_radial.txt"
            header = ("# cell_tag\tjob_id\tlat_level\trad_level\tlat_sdr\trad_sdr"
                      "\texecutions\tn_gpus\tnodes\ttasks_per_node\n")
            fields = ["lat_level", "rad_level", "lat_sdr", "rad_sdr",
                      "executions", "n_gpus", "nodes", "tasks_per_node"]
        else:
            manifest = JOB_DIR / ("manifest_weak.txt" if args.weak else "manifest.txt")
            header = "# cell_tag\tjob_id\tlevel\tsdr\texecutions\tn_gpus\tnodes\ttasks_per_node\n"
            fields = ["level", "sdr", "executions", "n_gpus", "nodes", "tasks_per_node"]
        with open(manifest, "w") as f:
            f.write(header)
            for info, jobid in submitted:
                f.write("\t".join([info["cell_tag"], jobid] +
                                  [str(info[k]) for k in fields]) + "\n")
        print(f"\nWrote manifest: {manifest}")

if __name__ == "__main__":
    main(sys.argv[1:])
