#!/usr/bin/env python3
"""C1 benchmark, level 7, pseudo-cfl=1.5. JUWELS Booster. Outputs go to $SCRATCH_walberlamovinggeo."""
import os
import subprocess
from pathlib import Path

# -----------------------
# Resolve $SCRATCH_walberlamovinggeo
# -----------------------
work_root = os.environ.get("SCRATCH_walberlamovinggeo")
if not work_root:
    raise SystemExit("ERROR: $SCRATCH_walberlamovinggeo is not set in the submitting shell.")
user = os.environ.get("USER")
if not user:
    raise SystemExit("ERROR: $USER is not set in the submitting shell.")

work_base = Path(work_root) / user / "mantlecirculation"
work_base.mkdir(parents=True, exist_ok=True)

# -----------------------
# Configuration
# -----------------------
job_name      = "mc_C1_lvl7_cfl15"
nodes         = 10
gpus_per_node = 4
time_limit    = "24:00:00"
partition     = "booster"
account       = "walberlamovinggeo"

binary = "/p/home/jusers/boehm2/juwels/terraneo-build/apps/mantlecirculation/mantlecirculation"
config = "/p/home/jusers/boehm2/juwels/terraneo/config_C1_lvl7_cfl15.toml"

outdir = work_base / "output_C1_lvl7_cfl15"
log_o  = work_base / f"{job_name}.o%j"
log_e  = work_base / f"{job_name}.e%j"

# -----------------------
# Generate script
# -----------------------
job_dir = Path("job_scripts")
job_dir.mkdir(exist_ok=True)
script_path = job_dir / f"{job_name}.sh"

script_content = f"""#!/bin/bash -x
#SBATCH --job-name={job_name}
#SBATCH --output={log_o}
#SBATCH --error={log_e}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={nodes * gpus_per_node}
#SBATCH --ntasks-per-node={gpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --gpu-bind=closest
#SBATCH --time={time_limit}

source ~/.bashrc

cd ${{SLURM_SUBMIT_DIR}}

export CUDA_VISIBLE_DEVICES=0,1,2,3

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
