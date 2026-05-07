#!/usr/bin/env python3
"""
Strong-scale benchmark driver for the mantle-circulation app on JUWELS Booster.

For each (MT-level, n_gpus) cell in the sweep:
  - emit a per-cell sbatch script under bench_mt/jobs/
  - submit via sbatch
  - the resulting outdir is bench_mt/outputs/MT<...>_g<n>/

MT-level convention: MT(N) where N = 2 * number_of_radial_cells = 2^mesh_max
(implied by --radial-extra-levels=-1, so radial cells = 2^(mesh_max - 1)).

Usage:
    python3 submit_bench_mt.py            # submit the default sweep
    python3 submit_bench_mt.py --dry-run  # just print what would be submitted
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

BENCH_DIR     = Path(__file__).resolve().parent
APP_DIR       = BENCH_DIR.parent
BINARY        = Path("/p/home/jusers/boehm2/juwels/terraneo-build/apps/mantlecirculation/mantlecirculation")
BENCH_CONFIG  = BENCH_DIR / "config_bench_A3.toml"
JOB_DIR       = BENCH_DIR / "jobs"
OUTPUT_ROOT   = BENCH_DIR / "outputs"

# (mesh_max, list of GPU counts to test). mesh_max -> MT label = 2^mesh_max.
DEFAULT_SWEEP: list[tuple[int, list[int]]] = [
    # (mesh_max, [n_gpus])
    (5,  [1, 2, 4, 8]),                  # MT32    (rad=16,  lat=32)
    (6,  [1, 2, 4, 8, 16]),              # MT64    (rad=32,  lat=64)
    (7,  [4, 8, 16, 32]),                # MT128   (rad=64,  lat=128)
    (8,  [16, 32, 64, 128, 256]),        # MT256   (rad=128, lat=256)
    (9,  [64, 128, 256, 512]),           # MT512   (rad=256, lat=512)
    (10, [256, 512, 1024, 2048]),        # MT1024  (rad=512, lat=1024)
]

GPUS_PER_NODE  = 4
ACCOUNT        = "walberlamovinggeo"

# JUWELS Booster: --partition=booster caps at 384 nodes per job; >384 needs --partition=largebooster
# (same hardware, different policy). The 2048-GPU MT1024 cell is 512 nodes -> largebooster.
BOOSTER_NODE_CAP = 384

# Walltime: MT32..MT256 fit comfortably in 15 min; MT512/MT1024 cells at low
# GPU counts can take >15 min (observed: MT512_g64, MT1024_g256, MT1024_g512
# all timed out at 15 min on the first sweep), so give the two largest models
# a 30-min budget.
def time_limit_for(mesh_max: int) -> str:
    return "00:30:00" if mesh_max >= 9 else "00:15:00"

# Total subdomain count = 10 * 4^lat_sdr * 2^rad_sdr
# (lat sdr subdivides 2 axes per level -> x4; rad sdr subdivides 1 axis -> x2).
# We restrict rad_sdr in {max(0, lat_sdr-1), lat_sdr} so the radial direction
# is never refined more than the lateral. Allowing rad_sdr=lat_sdr (in addition
# to lat_sdr-1) gives a 2x granularity step alongside the 8x lat-step, which
# avoids over-decomposing just to land on a rank-balanced count.
def subdomains_for(lat_sdr: int, rad_sdr: int) -> int:
    return 10 * (4 ** lat_sdr) * (2 ** rad_sdr)

# Pick the (lat_sdr, rad_sdr) with smallest total subdomain count s.t.
# subdomains >= n_gpus and divides n_gpus evenly.
def choose_lat_rad(n_gpus: int) -> tuple[int, int]:
    candidates = []
    for lat in range(0, 5):
        for rad in {max(0, lat - 1), lat}:
            candidates.append((subdomains_for(lat, rad), lat, rad))
    candidates.sort()
    for subs, lat, rad in candidates:
        if subs >= n_gpus and subs % n_gpus == 0:
            return lat, rad
    raise ValueError(f"no balanced (lat_sdr, rad_sdr) found for n_gpus={n_gpus}")

# Validate per the runtime guards in parameters.hpp:
#   mesh_min + radial_extra_levels >= 0          -> mesh_min >= 1
#   mesh_min >= lat_sdr
#   mesh_min + radial_extra_levels >= rad_sdr    -> mesh_min >= rad_sdr + 1
# (the last with radial_extra_levels = -1). mesh_min >= 2 keeps a non-trivial
# MG hierarchy.
def choose_mesh_min(lat_sdr: int, rad_sdr: int) -> int:
    return max(2, lat_sdr, rad_sdr + 1)

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

def render_job_script(mesh_max: int, n_gpus: int) -> tuple[Path, Path, dict]:
    mt_label         = 2 ** mesh_max
    cell_tag         = f"MT{mt_label}_g{n_gpus}"
    job_path         = JOB_DIR / f"bench_{cell_tag}.sh"
    out_path         = OUTPUT_ROOT / cell_tag
    nodes, tpn   = gpu_to_node_layout(n_gpus)
    lat_sdr, rad_sdr = choose_lat_rad(n_gpus)
    mesh_min     = choose_mesh_min(lat_sdr, rad_sdr)
    radial_extra = -1

    info = dict(
        cell_tag=cell_tag, mt_label=mt_label,
        mesh_max=mesh_max, mesh_min=mesh_min,
        lat_sdr=lat_sdr, rad_sdr=rad_sdr, radial_extra=radial_extra,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
    )

    cmdline_args = (
        f"--config {BENCH_CONFIG} "
        f"--outdir {out_path} "
        f"--outdir-overwrite "
        f"--refinement-level-mesh-min {mesh_min} "
        f"--refinement-level-mesh-max {mesh_max} "
        f"--lat-sdr {lat_sdr} "
        f"--rad-sdr {rad_sdr} "
        f"--radial-extra-levels {radial_extra} "
        f"--no-xdmf --no-radial-profiles"
    )

    partition  = "largebooster" if nodes > BOOSTER_NODE_CAP else "booster"
    time_limit = time_limit_for(mesh_max)

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

echo "Cell: {cell_tag}  mesh_max={mesh_max}  mesh_min={mesh_min}  lat_sdr={lat_sdr}  rad_sdr={rad_sdr}  radial_extra={radial_extra}  n_gpus={n_gpus}  nodes={nodes}x{tpn}  partition={partition}"

module load Stages/2025
module load CUDA/12
module load NVHPC/25.5-CUDA-12
module load OpenMPI/4.1.8

export OMPI_MCA_pml=ucx
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc

srun --gpu-bind=closest {BINARY} {cmdline_args}
"""

    job_path.write_text(script)
    job_path.chmod(0o755)
    return job_path, out_path, info

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="only emit the job scripts; don't sbatch")
    p.add_argument("--levels", type=int, nargs="+", default=None,
                   help="override which mesh_max levels to submit (default: full sweep)")
    args = p.parse_args(argv)

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    cells = []
    for mesh_max, gpu_list in DEFAULT_SWEEP:
        if args.levels is not None and mesh_max not in args.levels:
            continue
        for n_gpus in gpu_list:
            cells.append((mesh_max, n_gpus))

    print(f"Sweep covers {len(cells)} cells:")
    for (mesh_max, n_gpus) in cells:
        print(f"  mesh_max={mesh_max}  MT{2**mesh_max}  n_gpus={n_gpus}")

    submitted = []
    for (mesh_max, n_gpus) in cells:
        job_path, out_path, info = render_job_script(mesh_max, n_gpus)
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
        manifest = JOB_DIR / "manifest.txt"
        with open(manifest, "w") as f:
            f.write("# cell_tag\tjob_id\tmesh_max\tlat_sdr\trad_sdr\tmesh_min\tradial_extra\tn_gpus\tnodes\ttasks_per_node\n")
            for info, jobid in submitted:
                f.write("\t".join([
                    info["cell_tag"], jobid,
                    str(info["mesh_max"]),
                    str(info["lat_sdr"]), str(info["rad_sdr"]),
                    str(info["mesh_min"]), str(info["radial_extra"]),
                    str(info["n_gpus"]), str(info["nodes"]), str(info["tasks_per_node"]),
                ]) + "\n")
        print(f"\nWrote manifest: {manifest}")

if __name__ == "__main__":
    main(sys.argv[1:])
