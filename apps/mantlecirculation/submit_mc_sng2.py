#!/usr/bin/env python3
import subprocess
from pathlib import Path

# -----------------------
# Configuration
# -----------------------
job_name = "mc_C1_lvl7_sdr1"
nodes = 20
gpus_per_node = 4
ntasks = nodes * gpus_per_node
time_limit = "12:00:00"

partition = "general"
account = "pn29po"

binary = "/home/fabi/sng_mount/terraneo-build/apps/mantlecirculation/mantlecirculation"
config = "/home/fabi/sng_mount/terraneo/apps/mantlecirculation/config_C1_lvl7.toml"

# -----------------------
# Generate script
# -----------------------
job_dir = Path("job_scripts")
job_dir.mkdir(exist_ok=True)
script_path = job_dir / f"{job_name}.sh"

script_content = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}.o%j
#SBATCH --error={job_name}.e%j
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

srun {binary} --config {config}

"""

with script_path.open("w") as f:
    f.write(script_content)

script_path.chmod(0o755)

# -----------------------
# Submit
# -----------------------
try:
    result = subprocess.run(["sbatch", str(script_path)], check=True)
    print(f"Submitted {script_path.name}")
    script_path.unlink()
except subprocess.CalledProcessError as e:
    print(f"ERROR submitting {script_path.name}")
    print(e.stderr)

if not any(job_dir.iterdir()):
    job_dir.rmdir()
