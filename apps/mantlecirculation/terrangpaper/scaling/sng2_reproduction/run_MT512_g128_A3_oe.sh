#!/bin/bash -l
#SBATCH --job-name=oe_MT512_g128_A3_oe
#SBATCH --output=%x_%j.out
#SBATCH --partition=general
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
module load slurm_setup
module sw stack/24.5.0
module load cmake gcc/14.2.0
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
ulimit -c 0
export TMPDIR=${TMPDIR:-/tmp}
# Paths. Override any of these to run from a checkout instead of the
# original scratch tree:
#   TERRANG_BIN  the mantlecirculation binary
#   TERRANG_CFG  config_scal_A3.toml (defaults to the copy next to this tree)
#   TERRANG_OUT  output directory for this point
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BIN="${TERRANG_BIN:?set TERRANG_BIN to the mantlecirculation binary}"
CFG="${TERRANG_CFG:-$HERE/../config_scal_A3.toml}"
OUT="${TERRANG_OUT:-./MT512_g128_A3_oe}"
# A relative run directory is taken relative to the submit directory.
case "$OUT" in /*) ;; *) OUT="${SLURM_SUBMIT_DIR:-$PWD}/$OUT";; esac

mkdir -p "$TMPDIR" "$OUT"
cd "$OUT"
srun --chdir="$OUT" "$BIN" --config "$CFG" --extended-parameters \
  --energy-solver ev \
  --reference-viscosity 2.459983e25 --radius-cmb 3527020 --radius-surface 6418020 \
  --temperature-surface 0 --temperature-cmb 3500 \
  --viscosity-min 1e18 --viscosity-max 1e28 \
  --refinement-level-mesh-min 3 --refinement-level-mesh-max 9 \
  --refinement-level-subdomains 2 --radial-extra-levels -1 \
  --max-timesteps 10 --no-xdmf --no-radial-profiles --output-frequency 9 --dt-min 1e-8 \
  --stokes-krylov-max-iterations 10 --stokes-krylov-restart 10 \
  --stokes-krylov-relative-tolerance 0 --stokes-krylov-absolute-tolerance 0 \
  --stokes-viscous-pc-num-smoothing-steps-prepost 2 \
  --energy-krylov-max-iterations 50 \
  --energy-krylov-relative-tolerance 0 --energy-krylov-absolute-tolerance 0 \
  --outdir "$OUT" --outdir-overwrite
