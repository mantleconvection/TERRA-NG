#!/usr/bin/env python3
"""C1 benchmark, level 6, pseudo-cfl=1.5. Outputs go to $WORK_pn29po."""
import os
import subprocess
from pathlib import Path

# -----------------------
# Resolve $WORK_pn29po
# -----------------------
work_root = os.environ.get("WORK_pn29po")
if not work_root:
    raise SystemExit("ERROR: $WORK_pn29po is not set in the submitting shell.")

work_base = Path(work_root) / "mantlecirculation"
work_base.mkdir(parents=True, exist_ok=True)

# -----------------------
# Configuration
# -----------------------
job_name      = "mc_C1_lvl6_cfl15"
nodes         = 2
gpus_per_node = 5
time_limit    = "24:00:00"
partition     = "general"
account       = "pn29po"

binary = "./mantlecirculation"
config = "./config_C1_lvl6_cfl15.toml"

outdir = work_base / "output_C1_lvl6_cfl15"
log_o  = work_base / f"{job_name}.o%j"
log_e  = work_base / f"{job_name}.e%j"

# -----------------------
# Generate script
# -----------------------
job_dir = Path("job_scripts")
job_dir.mkdir(exist_ok=True)
script_path = job_dir / f"{job_name}.sh"

script_content = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output={log_o}
#SBATCH --error={log_e}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --account={account}

module load slurm_setup

module sw stack/24.5.0
module load cmake gcc/14.2.0
module load intel-toolkit/2025.2.0

module list

cd ${{SLURM_SUBMIT_DIR}}

export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_RDMA=1
export I_MPI_OFFLOAD_FAST_MEMCPY_COLL=1
export PSM3_RDMA=1
export PSM3_GPUDIRECT=0

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

srun {binary} --config {config} --outdir {outdir} --outdir-overwrite
"""

with script_path.open("w") as f:
    f.write(script_content)
script_path.chmod(0o755)

try:
    subprocess.run(["sbatch", str(script_path)], check=True)
    print(f"Submitted {script_path.name}  ->  outdir={outdir}")
    script_path.unlink()
except subprocess.CalledProcessError as e:
    print(f"ERROR submitting {script_path.name}")
    print(e.stderr)

if not any(job_dir.iterdir()):
    job_dir.rmdir()
