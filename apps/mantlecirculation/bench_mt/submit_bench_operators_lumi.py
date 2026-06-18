#!/usr/bin/env python3
"""
Strong-scale benchmark driver for the *operator* microbenchmark on LUMI-G.

Runs the `benchmark_operators` app (matvec throughput for the production
EpsDivDivKerngen operator). Uses SEPARATE lateral/radial refinement
(--lat-sdr / --rad-sdr / --radial-extra-levels), so the domain is no longer the
old isotropic create_uniform(level, level, sdr, sdr); lateral and radial mesh
levels and subdomain refinements are independent.

Default mode = RADIAL strong scaling: fix the global mesh (lat_level, lat_sdr,
rad_level) and scale ONLY the radial direction -- each step rad_sdr += 1 doubles
the radial subdomain count and doubles GPUs (n_gpus = 4^lat_sdr * 2^rad_sdr),
while total DoFs stay fixed (sdr does not change DoFs), so DoFs/GCD halves per
step and subdomains/GCD stays constant. Lateral refinement is never touched.
STRONG_RAD_LEVELS gives several global problem sizes (one strong-scaling curve
each). Each cell writes CSV / timer-tree under bench_mt/outputs/<tag>/{csv,tts}/.

Other modes: --weak (8x level / 8x GCD isotropic weak diagonals) and
--weak-radial (radial weak diagonals: rad_level AND rad_sdr AND GPUs x2/step).

Usage:
    python3 submit_bench_operators_lumi.py            # default radial strong-scaling sweep
    python3 submit_bench_operators_lumi.py --dry-run  # just print what would be submitted
    python3 submit_bench_operators_lumi.py --weak     # isotropic weak diagonals instead
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
# benchmark_operators binary (LUMI PFS build). Override with $BENCH_OPERATORS_BIN.
BINARY      = Path(os.environ.get(
    "BENCH_OPERATORS_BIN",
    "/pfs/lustrep3/users/bohmfabi/terraneo-build/apps/benchmarks/performance/benchmark_operators"))
JOB_DIR     = BENCH_DIR / "jobs"
OUTPUT_ROOT = BENCH_DIR / "outputs"

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
# Every diagonal STARTS AT ONE NODE (GPUS_PER_NODE GCDs) with subdom/GCD = 10; the
# requested fill is realized by the *starting radial level* rather than the GPU
# count. The 63 M anchor is isotropic (rad_level = lat_level); each halving of the
# fill starts one radial level coarser. With lat 8 / lat_sdr 1 on 8 GCDs:
#   rad_level 8 -> 63 M/GCD, 7 -> 32 M, 6 -> 16 M, 5 -> 8 M, ...
RADIAL_LAT_LEVEL     = 8
RADIAL_LAT_SDR       = 1
RADIAL_DEFAULT_FILLS = [63, 32, 16, 8]    # target M-dofs/GCD radial diagonals

def radial_anchor(fill_mdofs: int) -> tuple[int, int]:
    """Starting (rad_level, rad_sdr) so the diagonal begins at GPUS_PER_NODE GCDs
    (one node) with subdom/GCD = 10 at the requested fill."""
    rad_sdr   = (GPUS_PER_NODE.bit_length() - 1) - 2 * RADIAL_LAT_SDR   # -> subdom/GCD = 10
    fill_iso  = DOFS_PER_LEVEL[RADIAL_LAT_LEVEL] / GPUS_PER_NODE         # fill at rad_level = lat_level
    rad_level = RADIAL_LAT_LEVEL + round(math.log2(fill_mdofs * 1e6 / fill_iso))
    return rad_level, max(0, rad_sdr)

def radial_cells(fills: list[int], max_gpus: int) -> list[dict]:
    cells: list[dict] = []
    for fill in fills:
        rad_level0, rad_sdr0 = radial_anchor(fill)
        k = 0
        while GPUS_PER_NODE * 2 ** k <= max_gpus:
            rad_level = rad_level0 + k
            cells.append(dict(
                lat_level=RADIAL_LAT_LEVEL, lat_sdr=RADIAL_LAT_SDR,
                rad_level=rad_level, rad_sdr=rad_sdr0 + k,
                radial_extra=rad_level - RADIAL_LAT_LEVEL,
                n_gpus=GPUS_PER_NODE * 2 ** k, fill=fill))
            k += 1
    return cells

# --- Radial strong-scaling sweeps ---
# Fix a global mesh (lat_level, rad_level) and strong-scale it by increasing the
# radial SUBDOMAIN refinement in 2x steps: rad_sdr += 1 -> subdomains x2 -> GPUs x2,
# total dofs fixed (sdr does not change dofs) so dofs/GCD halves each step.
# subdom/GCD stays 10. Each problem starts at one node (GPUS_PER_NODE GCDs) and
# scales until rad_sdr = rad_level (one radial cell per subdomain) or the GPU cap.
# Four global problem sizes via rad_level (lat 8): total dofs ~1.0 G / 505 M / 252 M
# / 126 M, i.e. 126 / 63 / 32 / 16 M dofs/GCD at the 1-node start.
STRONG_LAT_LEVEL  = 8
STRONG_LAT_SDR    = 1
STRONG_RAD_LEVELS = [9, 8, 7, 6]

def strong_cells(rad_levels: list[int], max_gpus: int) -> list[dict]:
    cells: list[dict] = []
    rad_sdr0 = (GPUS_PER_NODE.bit_length() - 1) - 2 * STRONG_LAT_SDR   # rad_sdr at one node
    for rad_level in rad_levels:
        rad_sdr = rad_sdr0
        # stop one short of rad_sdr == rad_level: the wedge operator needs >= 2
        # radial cells per subdomain (2^(rad_level - rad_sdr) >= 2).
        while rad_sdr < rad_level:
            n_gpus = 4 ** STRONG_LAT_SDR * 2 ** rad_sdr
            if n_gpus > max_gpus:
                break
            cells.append(dict(
                cell_tag=f"STR_l{STRONG_LAT_LEVEL}r{rad_level}_g{n_gpus}",
                lat_level=STRONG_LAT_LEVEL, lat_sdr=STRONG_LAT_SDR,
                rad_level=rad_level, rad_sdr=rad_sdr,
                radial_extra=rad_level - STRONG_LAT_LEVEL,
                n_gpus=n_gpus, problem=rad_level))
            rad_sdr += 1
    return cells

# Matvec repetitions per timing (matches the remeasure_history / 2gcd configs).
DEFAULT_EXECUTIONS = 5
# Untimed warmup matvecs before the timed region (amortizes launch/alloc overhead).
DEFAULT_WARMUP = 5

# LUMI-G: each node has 8 GCDs (4 MI250X * 2). standard-g allocates GPUs; we
# bind one rank per GCD. Account / partition can be overridden via env.
GPUS_PER_NODE = 8
ACCOUNT       = os.environ.get("SLURM_ACCOUNT", "project_465002367")
PARTITION     = "standard-g"

# NUMA-aware CPU cores, one per GCD, in the canonical LUMI GCD order. For a
# partial node we take the first `tasks_per_node` entries (matches the proven
# 2-GCD config that used map_cpu:49,57).
CPU_BIND_ORDER = [49, 57, 17, 25, 1, 9, 33, 41]

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

    Up to 8 GCDs we pack onto a single node; beyond that we use full nodes
    (8 GCDs each).
    """
    if n_gpus <= GPUS_PER_NODE:
        return (1, n_gpus)
    if n_gpus % GPUS_PER_NODE != 0:
        raise ValueError(f"n_gpus={n_gpus} is not a multiple of {GPUS_PER_NODE} for >1 nodes")
    return (n_gpus // GPUS_PER_NODE, GPUS_PER_NODE)

def write_lumi_job(cell_tag: str, out_path: Path, nodes: int, tpn: int,
                   time_limit: str, bench_args: str, echo_line: str) -> Path:
    """Write the shared LUMI-G sbatch script for one benchmark_operators cell."""
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
export OMP_NUM_THREADS=1
ulimit -c 0

# Per-GCD GPU binding wrapper (maps SLURM_LOCALID -> ROCR_VISIBLE_DEVICES).
SELECT_GPU=${{SLURM_SUBMIT_DIR}}/select_gpu_${{SLURM_JOB_ID}}.sh
cat > ${{SELECT_GPU}} << 'INNER'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
INNER
chmod +x ${{SELECT_GPU}}

# benchmark_operators writes csv/ and tts/ relative to CWD.
mkdir -p {out_path}/csv {out_path}/tts
cd {out_path}

srun --cpu-bind={cpu_bind} ${{SELECT_GPU}} {BINARY} {bench_args}

rm -f ${{SELECT_GPU}}
"""
    job_path.write_text(script)
    job_path.chmod(0o755)
    return job_path

def render_job_script(level: int, n_gpus: int, executions: int, warmup: int, time_limit: str = None) -> tuple[Path, Path, dict]:
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
                 f"executions={executions}  n_gpus={n_gpus}  nodes={nodes}x{tpn}  partition={PARTITION}")
    job_path = write_lumi_job(cell_tag, out_path, nodes, tpn,
                              time_limit or time_limit_for(level), bench_args, echo_line)
    return job_path, out_path, info

def render_radial_job_script(cell: dict, executions: int, warmup: int, time_limit: str = None) -> tuple[Path, Path, dict]:
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
    cell_tag   = cell.get("cell_tag") or f"RAD_l{lat_level}r{rad_level}_g{n_gpus}"
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
                 f"n_gpus={n_gpus}  nodes={nodes}x{tpn}  partition={PARTITION}")
    # radial mesh can be deep; give the larger radial levels the longer budget.
    job_path = write_lumi_job(cell_tag, out_path, nodes, tpn,
                              time_limit or time_limit_for(rad_level), bench_args, echo_line)
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
    p.add_argument("--time-limit", type=str, default=None,
                   help="override the SLURM walltime for every cell (e.g. 00:10:00)")
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
    p.add_argument("--rad-levels", type=int, nargs="+", default=None,
                   help=f"radial problem sizes (rad_level) for the default strong sweep "
                        f"(default: {STRONG_RAD_LEVELS}); each +1 doubles total DoFs")
    p.add_argument("--weak-radial", action="store_true",
                   help="radial-only weak-scaling diagonals: lateral mesh fixed, radial "
                        "level+subdomains and GPUs x2 per step (dofs x2/step, constant dofs/GCD)")
    p.add_argument("--weak-radial-fills", type=int, nargs="+", default=RADIAL_DEFAULT_FILLS,
                   help=f"target dofs/GCD per radial diagonal, in millions "
                        f"(default: {RADIAL_DEFAULT_FILLS})")
    args = p.parse_args(argv)

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Three modes. --weak uses 'iso' cells (level, n_gpus); --weak-radial and the
    # default strong sweep use radial-style cell dicts rendered the same way.
    if args.weak:
        cells = weak_cells(args.weak_fills, WEAK_BASE_LEVEL,
                           args.weak_max_level, args.weak_max_gpus)
        if args.levels is not None:
            cells = [(level, n_gpus) for (level, n_gpus) in cells if level in args.levels]
        render_list = [("iso", c) for c in cells]
        print(f"weak-scaling diagonals (fills {args.weak_fills} M-dofs/GCD): {len(cells)} cells")
        for (level, n_gpus) in cells:
            sdr = choose_sdr(n_gpus)
            nodes, _ = gpu_to_node_layout(n_gpus)
            dpg = DOFS_PER_LEVEL.get(level)
            dpg_s = f"  dofs/GCD={dpg / n_gpus / 1e6:5.1f}M" if dpg else ""
            print(f"  level={level:2}  MT{2**level:<5}  n_gpus={n_gpus:5}  nodes={nodes:4}"
                  f"  sdr={sdr}  subdom/GCD={10 * 8**sdr // n_gpus}{dpg_s}")
    else:
        if args.weak_radial:
            cells = radial_cells(args.weak_radial_fills, args.weak_max_gpus)
            print(f"radial-only weak-scaling diagonals (lateral fixed at level {RADIAL_LAT_LEVEL}, "
                  f"lat_sdr {RADIAL_LAT_SDR}; fills {args.weak_radial_fills} M-dofs/GCD): {len(cells)} cells")
        else:
            rad_levels = args.rad_levels or STRONG_RAD_LEVELS
            cells = strong_cells(rad_levels, args.weak_max_gpus)
            print(f"radial strong-scaling sweeps (lateral fixed at level {STRONG_LAT_LEVEL}, "
                  f"lat_sdr {STRONG_LAT_SDR}; problems rad_level {rad_levels}): {len(cells)} cells")
        render_list = [("radial", c) for c in cells]
        dpg0 = DOFS_PER_LEVEL.get(RADIAL_LAT_LEVEL)
        for c in cells:
            n = c["n_gpus"]
            nodes, _ = gpu_to_node_layout(n)
            subdom = 10 * 4 ** c["lat_sdr"] * 2 ** c["rad_sdr"]
            dpg_s = f"  dofs/GCD={dpg0 * 2 ** c['radial_extra'] / n / 1e6:6.1f}M" if dpg0 else ""
            label = f"fill={c['fill']:3}M" if "fill" in c else f"prob=rad{c['rad_level']}"
            print(f"  {label}  rad_level={c['rad_level']:2}  rad_sdr={c['rad_sdr']}  "
                  f"n_gpus={n:5}  nodes={nodes:4}  subdom/GCD={subdom // n}{dpg_s}")

    submitted = []
    for kind, item in render_list:
        if kind == "radial":
            job_path, out_path, info = render_radial_job_script(item, args.executions, args.warmup, args.time_limit)
        else:
            level, n_gpus = item
            job_path, out_path, info = render_job_script(level, n_gpus, args.executions, args.warmup, args.time_limit)
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
        if args.weak:
            manifest = JOB_DIR / "manifest_weak.txt"
            header = "# cell_tag\tjob_id\tlevel\tsdr\texecutions\tn_gpus\tnodes\ttasks_per_node\n"
            fields = ["level", "sdr", "executions", "n_gpus", "nodes", "tasks_per_node"]
        else:  # radial-style: --weak-radial diagonals or the default strong sweep
            manifest = JOB_DIR / ("manifest_radial.txt" if args.weak_radial else "manifest_strong.txt")
            header = ("# cell_tag\tjob_id\tlat_level\trad_level\tlat_sdr\trad_sdr"
                      "\texecutions\tn_gpus\tnodes\ttasks_per_node\n")
            fields = ["lat_level", "rad_level", "lat_sdr", "rad_sdr",
                      "executions", "n_gpus", "nodes", "tasks_per_node"]
        with open(manifest, "w") as f:
            f.write(header)
            for info, jobid in submitted:
                f.write("\t".join([info["cell_tag"], jobid] +
                                  [str(info[k]) for k in fields]) + "\n")
        print(f"\nWrote manifest: {manifest}")

if __name__ == "__main__":
    main(sys.argv[1:])
