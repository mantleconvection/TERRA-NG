#!/usr/bin/env python3
import os, subprocess
from pathlib import Path
work_root = os.environ.get("SCRATCH_walberlamovinggeo")
if not work_root: raise SystemExit("ERROR: $SCRATCH_walberlamovinggeo not set.")
user = os.environ.get("USER")
work_base = Path(work_root) / user / "mantlecirculation"
work_base.mkdir(parents=True, exist_ok=True)
job_name = "mc_A3_lvl6_supg_cfl025"
config = "/p/home/jusers/boehm2/juwels/terraneo/apps/mantlecirculation/config_A3_lvl6_supg_cfl025.toml"
nodes, ntasks, time_limit = 3, 10, "24:00:00"
partition, account = "booster", "walberlamovinggeo"
binary = "/p/home/jusers/boehm2/juwels/terraneo-build/apps/mantlecirculation/mantlecirculation"
outdir = work_base / f"output_{job_name.replace('mc_','')}"
log_o, log_e = work_base / f"{job_name}.o%j", work_base / f"{job_name}.e%j"
job_dir = Path("job_scripts"); job_dir.mkdir(exist_ok=True)
script_path = job_dir / f"{job_name}.sh"
script_content = f"""#!/bin/bash -x
#SBATCH --job-name={job_name}
#SBATCH --output={log_o}
#SBATCH --error={log_e}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --time={time_limit}

source ~/.bashrc
cd ${{SLURM_SUBMIT_DIR}}
export CUDA_VISIBLE_DEVICES=0,1,2,3
srun {binary} --config {config} --outdir {outdir} --outdir-overwrite
"""
with script_path.open("w") as f: f.write(script_content)
script_path.chmod(0o755)
try:
    subprocess.run(["sbatch", str(script_path)], check=True)
    print(f"Submitted {script_path.name}  ->  outdir={outdir}")
    script_path.unlink()
except subprocess.CalledProcessError as e:
    print(f"ERROR: {e.stderr}")
if not any(job_dir.iterdir()): job_dir.rmdir()
