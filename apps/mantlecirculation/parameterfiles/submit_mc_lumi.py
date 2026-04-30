#!/usr/bin/env python3
"""Submit the A3, C1, C3, and C4 mantlecirculation EV-stabilized runs on LUMI-G.

Resolves each config TOML relative to this script's own directory. The mantlecirculation
binary path is taken from `$TERRANEO_BIN`; the SLURM account from `$SLURM_ACCOUNT` (default
project_465002889); the scratch root from `$SCRATCH` (or `$LUMI_SCRATCH`) so the script
does not depend on user-specific install paths.

LUMI-G layout: each node has 8 GCDs (4 MI250X * 2). The configs use
`refinement-level-subdomains=1` (10 * 2^3 = 80 subdomains), so we run 16 ranks across
2 nodes (8 ranks per node, 1 GCD per rank, 5 subdomains per GCD).
"""
import os, subprocess
from pathlib import Path

script_dir = Path(__file__).resolve().parent

binary = os.environ.get("TERRANEO_BIN")
if not binary:
    raise SystemExit("ERROR: set TERRANEO_BIN to the path of the mantlecirculation binary.")

scratch_root = os.environ.get("SCRATCH") or os.environ.get("LUMI_SCRATCH")
if not scratch_root:
    raise SystemExit("ERROR: set $SCRATCH (or $LUMI_SCRATCH) to a writable scratch directory.")
work_base = Path(scratch_root) / os.environ["USER"] / "mantlecirculation"
work_base.mkdir(parents=True, exist_ok=True)

account   = os.environ.get("SLURM_ACCOUNT", "project_465002889")
partition = "standard-g"
nodes, ntasks_per_node, time_limit = 2, 8, "24:00:00"

# (tag, config-filename) pairs — config resolved against script_dir.
RUNS = [
    ("A3_lvl6_ev_cfl025",            "config_A3_lvl6_ev_cfl025.toml"),
    ("C1_lvl6_ev_cfl025_picard2",    "config_C1_lvl6_ev_cfl025_picard2.toml"),
    ("C3_lvl6_ev_cfl025_picard2",    "config_C3_lvl6_ev_cfl025_picard2.toml"),
    ("C4_lvl6_ev_cfl025_picard2",    "config_C4_lvl6_ev_cfl025_picard2.toml"),
]

job_dir = Path("job_scripts"); job_dir.mkdir(exist_ok=True)

for tag, config_name in RUNS:
    config = script_dir / config_name
    if not config.is_file():
        print(f"SKIP {tag}: config not found at {config}")
        continue

    job_name = f"mc_{tag}"
    outdir   = work_base / f"output_{tag}"
    log_o    = work_base / f"{job_name}.o%j"
    log_e    = work_base / f"{job_name}.e%j"

    script_path = job_dir / f"{job_name}.sh"
    script_content = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output={log_o}
#SBATCH --error={log_e}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gpus-per-node={ntasks_per_node}
#SBATCH --time={time_limit}

# Per-GCD GPU binding wrapper (maps SLURM_LOCALID -> ROCR_VISIBLE_DEVICES).
SELECT_GPU=${{SLURM_SUBMIT_DIR}}/select_gpu_${{SLURM_JOB_ID}}
cat <<'EOS' > ${{SELECT_GPU}}
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec $*
EOS
chmod +x ${{SELECT_GPU}}

# NUMA-aware CPU binding for all 8 MI250X GCDs per node.
CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
export MPICH_GPU_SUPPORT_ENABLED=1

srun --cpu-bind=${{CPU_BIND}} ${{SELECT_GPU}} {binary} \\
    --config {config} --outdir {outdir} --outdir-overwrite

rm -f ${{SELECT_GPU}}
"""
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    try:
        subprocess.run(["sbatch", str(script_path)], check=True)
        print(f"Submitted {script_path.name}  ->  outdir={outdir}")
        script_path.unlink()
    except subprocess.CalledProcessError as e:
        print(f"ERROR submitting {script_path.name}: {e}")

if not any(job_dir.iterdir()): job_dir.rmdir()
