#!/usr/bin/env python3
"""
JUWELS-Booster weak-scaling sweep for benchmark_operators, radial-only.

Per step k:
  lat_level = LAT_LEVEL          (fixed)
  lat_sdr   = LAT_SDR            (fixed)
  rad_level = RAD_LEVEL_BASE + k (+1 -> DoFs x2)
  rad_sdr   = k                  (+1 -> subdomains x2)
  n_gpus    = 2**k               (DoFs and GPUs both x2 -> work/GPU constant)

Constants:
  subdomains-per-GPU = 10 * 2^rad_sdr / n_gpus = 10 throughout
  cells/subdomain    = 2^(lat_level - lat_sdr) lat x 2^(rad_level - rad_sdr) rad
                     = 2^LAT_LEVEL x 2^RAD_LEVEL_BASE  (independent of k)

So both the per-GPU subdomain count and the per-subdomain element count are
constant - this is the cleanest possible weak-scaling sweep.

Outputs/jobs under scratch jobs_bo_weak_rad/ and outputs_bo_weak_rad/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCH_DIR     = Path(__file__).resolve().parent
BINARY        = Path("/p/home/jusers/boehm2/juwels/terraneo-build/apps/benchmarks/performance/benchmark_operators")
SCRATCH_ROOT  = Path("/p/scratch/walberlamovinggeo/boehm2/mantlecirculation/bench_mt")
JOB_DIR       = SCRATCH_ROOT / "jobs_bo_weak_rad"
OUTPUT_ROOT   = SCRATCH_ROOT / "outputs_bo_weak_rad"

LAT_LEVEL        = 5
LAT_SDR          = 0
RAD_LEVEL_BASE   = 4
DEFAULT_STEPS    = list(range(0, 12))   # k = 0..11 -> n_gpus = 1..2048

GPUS_PER_NODE    = 4
ACCOUNT          = "walberlamovinggeo"
BOOSTER_NODE_CAP = 384
EXECUTIONS       = 5
WARMUP           = 5
TIME_LIMIT       = "00:10:00"

def gpu_to_node_layout(n_gpus: int) -> tuple[int, int]:
    if n_gpus <= GPUS_PER_NODE:
        return (1, n_gpus)
    if n_gpus % GPUS_PER_NODE != 0:
        raise ValueError(f"n_gpus={n_gpus} is not a multiple of {GPUS_PER_NODE} for >1 nodes")
    return (n_gpus // GPUS_PER_NODE, GPUS_PER_NODE)

def render_job_script(step: int, executions: int, warmup: int) -> tuple[Path, Path, dict]:
    rad_level  = RAD_LEVEL_BASE + step
    rad_sdr    = step
    n_gpus     = 2 ** step
    cell_tag   = f"WSrad_k{step:02d}_g{n_gpus}_rl{rad_level}"
    job_path   = JOB_DIR / f"bench_bo_wr_{cell_tag}.sh"
    out_path   = OUTPUT_ROOT / cell_tag
    nodes, tpn = gpu_to_node_layout(n_gpus)

    info = dict(
        cell_tag=cell_tag, step=step,
        lat_level=LAT_LEVEL, rad_level=rad_level,
        lat_sdr=LAT_SDR,    rad_sdr=rad_sdr,
        n_gpus=n_gpus, nodes=nodes, tasks_per_node=tpn,
    )

    cmdline_args = (
        f"--min-level {LAT_LEVEL} --max-level {LAT_LEVEL} "
        f"--radial-extra-levels {rad_level - LAT_LEVEL} "
        f"--lat-sdr {LAT_SDR} --rad-sdr {rad_sdr} "
        f"--executions {executions} "
        f"--warmup {warmup}"
    )

    partition = "largebooster" if nodes > BOOSTER_NODE_CAP else "booster"

    script = f"""#!/bin/bash -l
#SBATCH --job-name=bwr_{cell_tag}
#SBATCH --output=bench_bo_wr_{cell_tag}.o%j
#SBATCH --error=bench_bo_wr_{cell_tag}.e%j
#SBATCH -D {JOB_DIR}
#SBATCH --partition={partition}
#SBATCH --account={ACCOUNT}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={tpn}
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:{tpn}
#SBATCH --time={TIME_LIMIT}

echo "Cell: {cell_tag}  step={step}  lat={LAT_LEVEL} rad={rad_level}  lat_sdr={LAT_SDR} rad_sdr={rad_sdr}  n_gpus={n_gpus} nodes={nodes}x{tpn}  partition={partition}  executions={executions} warmup={warmup}"

module load Stages/2025
module load CUDA/12
module load NVHPC/25.5-CUDA-12
module load OpenMPI/4.1.8

export OMPI_MCA_pml=ucx
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc

mkdir -p {out_path}/csv {out_path}/tts
cd {out_path}

srun --gpu-bind=closest {BINARY} {cmdline_args}
"""

    job_path.write_text(script)
    job_path.chmod(0o755)
    return job_path, out_path, info

def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--steps", type=int, nargs="+", default=None,
                   help=f"step indices to submit (default {DEFAULT_STEPS})")
    p.add_argument("--executions", type=int, default=EXECUTIONS,
                   help=f"timed matvec count (default {EXECUTIONS})")
    p.add_argument("--warmup", type=int, default=WARMUP,
                   help=f"untimed warmup matvecs before timing (default {WARMUP})")
    args = p.parse_args(argv)

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    steps = args.steps if args.steps is not None else DEFAULT_STEPS

    print(f"Weak-scaling radial-only sweep, {len(steps)} steps "
          f"(lat={LAT_LEVEL} fixed, rad starts at {RAD_LEVEL_BASE}, executions={args.executions}, warmup={args.warmup}):")
    submitted = []
    for k in steps:
        job_path, out_path, info = render_job_script(k, args.executions, args.warmup)
        if args.dry_run:
            print(f"  [dry-run] step {k}: rad={info['rad_level']} rad_sdr={info['rad_sdr']} "
                  f"n_gpus={info['n_gpus']} nodes={info['nodes']}  -> {job_path}")
        else:
            res = subprocess.run(["sbatch", str(job_path)], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"  FAILED step {k}: {res.stderr.strip()}", file=sys.stderr)
            else:
                jobid = res.stdout.strip().split()[-1]
                print(f"  submitted job {jobid}: {info['cell_tag']}")
                submitted.append((info, jobid))

    if submitted and not args.dry_run:
        manifest = JOB_DIR / "manifest.txt"
        with open(manifest, "w") as f:
            f.write("# cell_tag\tjob_id\tstep\tlat_level\trad_level\tlat_sdr\trad_sdr\tn_gpus\tnodes\ttasks_per_node\n")
            for info, jobid in submitted:
                f.write("\t".join([
                    info["cell_tag"], jobid, str(info["step"]),
                    str(info["lat_level"]), str(info["rad_level"]),
                    str(info["lat_sdr"]), str(info["rad_sdr"]),
                    str(info["n_gpus"]), str(info["nodes"]), str(info["tasks_per_node"]),
                ]) + "\n")
        print(f"\nWrote manifest: {manifest}")

if __name__ == "__main__":
    main(sys.argv[1:])
