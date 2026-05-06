#!/usr/bin/env python3
"""
Strong-scale benchmark driver for the mantle-circulation app on Helma H100.

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
BINARY        = APP_DIR / "mantlecirculation"
BENCH_CONFIG  = BENCH_DIR / "config_bench_A3.toml"
JOB_DIR       = BENCH_DIR / "jobs"
OUTPUT_ROOT   = BENCH_DIR / "outputs"

# (mesh_max, list of GPU counts to test). mesh_max -> MT label = 2^mesh_max.
DEFAULT_SWEEP: list[tuple[int, list[int]]] = [
    # (mesh_max, [n_gpus])
    (5, [1, 2, 4, 8]),         # MT32   (rad=16, lat=32)
    (6, [1, 2, 4, 8, 16]),     # MT64   (rad=32, lat=64)
    (7, [4, 8, 16, 32]),       # MT128  (rad=64, lat=128)
    (8, [16, 32, 64]),         # MT256  (rad=128, lat=256)
    (9, [64, 128]),            # MT512  (rad=256, lat=512)
]

GPUS_PER_NODE  = 4
TIME_LIMIT     = "00:30:00"

# Pick the smallest level_subdomains s.t. 10*8^k >= n_gpus, capped at 3.
def choose_subdomains(n_gpus: int) -> int:
    for k in range(0, 4):
        if 10 * (8 ** k) >= n_gpus:
            return k
    return 3

# Validate per the runtime guards in parameters.hpp:
#   mesh_min + radial_extra_levels >= 0
#   mesh_min >= lat_sdr_eff
#   mesh_min + radial_extra_levels >= rad_sdr_eff
# We use lat_sdr_eff = rad_sdr_eff = level_subdomains and radial_extra_levels = -1.
def choose_mesh_min(level_subdomains: int) -> int:
    # mesh_min must be >= max(level_subdomains, level_subdomains + 1) = level_subdomains + 1
    # (the radial constraint is the binding one because radial_extra_levels = -1).
    # But we also want mesh_min >= 2 for a non-trivial MG hierarchy.
    return max(2, level_subdomains + 1)

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
    nodes, tpn       = gpu_to_node_layout(n_gpus)
    level_subdomains = choose_subdomains(n_gpus)
    mesh_min         = choose_mesh_min(level_subdomains)
    radial_extra     = -1

    info = dict(
        cell_tag=cell_tag, mt_label=mt_label,
        mesh_max=mesh_max, mesh_min=mesh_min,
        level_subdomains=level_subdomains, radial_extra=radial_extra,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
    )

    cmdline_args = (
        f"--config {BENCH_CONFIG} "
        f"--outdir {out_path} "
        f"--outdir-overwrite "
        f"--refinement-level-mesh-min {mesh_min} "
        f"--refinement-level-mesh-max {mesh_max} "
        f"--refinement-level-subdomains {level_subdomains} "
        f"--radial-extra-levels {radial_extra} "
        f"--no-xdmf --no-radial-profiles"
    )

    script = f"""#!/bin/bash -l
#SBATCH --job-name=bench_{cell_tag}
#SBATCH --output=bench_{cell_tag}.o%j
#SBATCH --error=bench_{cell_tag}.e%j
#SBATCH -D {JOB_DIR}
#SBATCH --partition=h100
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tpn}
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:{tpn}
#SBATCH --time={TIME_LIMIT}
#SBATCH --export=NONE

cd {APP_DIR}
echo "Running from: $(pwd)"
echo "Cell: {cell_tag}  mesh_max={mesh_max}  level_subdomains={level_subdomains}  mesh_min={mesh_min}  radial_extra={radial_extra}  n_gpus={n_gpus}  nodes={nodes}x{tpn}"

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load openmpi/5.0.5-nvhpc24.11-cuda

mpirun -np {n_gpus} {BINARY} {cmdline_args}
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
            f.write("# cell_tag\tjob_id\tmesh_max\tlevel_subdomains\tmesh_min\tradial_extra\tn_gpus\tnodes\ttasks_per_node\n")
            for info, jobid in submitted:
                f.write("\t".join([
                    info["cell_tag"], jobid,
                    str(info["mesh_max"]), str(info["level_subdomains"]), str(info["mesh_min"]),
                    str(info["radial_extra"]),
                    str(info["n_gpus"]), str(info["nodes"]), str(info["tasks_per_node"]),
                ]) + "\n")
        print(f"\nWrote manifest: {manifest}")

if __name__ == "__main__":
    main(sys.argv[1:])
