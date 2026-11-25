#!/bin/bash -l
#SBATCH --job-name=ls_MT128_g32_std
#SBATCH --output=%x_%j.out
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:40:00

echo "Cell: MT128_g32_std  mesh=[2..7]  lat_sdr=2 rad_sdr=0  steps=10  fgmres=10  ev=50  n_gcds=32  nodes=4x8"

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_NO_ASYNC_COPY=1
export OMP_NUM_THREADS=1
export FI_MR_CACHE_MAX_COUNT=1048576
export FI_CXI_RX_MATCH_MODE=software
ulimit -c 0

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BIN="${TERRANG_BIN:?set TERRANG_BIN to the mantlecirculation binary}"
CFG="${TERRANG_CFG:-$HERE/../config_scal_A3.toml}"
OUT="${TERRANG_OUT:-./MT128_g32_std}"
# A relative run directory is taken relative to the submit directory.
case "$OUT" in /*) ;; *) OUT="${SLURM_SUBMIT_DIR:-$PWD}/$OUT";; esac
mkdir -p "$OUT"

SELECT_GPU=${SLURM_SUBMIT_DIR}/select_gpu_${SLURM_JOB_ID}.sh
cat > ${SELECT_GPU} << 'INNER'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
INNER
chmod +x ${SELECT_GPU}
cd "$OUT"

srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 ${SELECT_GPU} "$BIN" --config "$CFG" --extended-parameters \
  --energy-solver ev \
  --reference-viscosity 2.459983e25 --radius-cmb 3527020 --radius-surface 6418020 \
  --temperature-surface 0 --temperature-cmb 3500 \
  --viscosity-min 1e18 --viscosity-max 1e28 \
  --refinement-level-mesh-min 2 --refinement-level-mesh-max 7 \
  --lat-sdr 2 --rad-sdr 0 --radial-extra-levels -1 \
  --max-timesteps 10 --no-xdmf --no-radial-profiles --output-frequency 9 --dt-min 1e-8 \
  --stokes-krylov-max-iterations 10 --stokes-krylov-restart 10 \
  --stokes-krylov-relative-tolerance 0 --stokes-krylov-absolute-tolerance 0 \
  --energy-krylov-max-iterations 50 \
  --energy-krylov-relative-tolerance 0 --energy-krylov-absolute-tolerance 0 \
  --stokes-viscous-pc-num-smoothing-steps-prepost 2 \
  --outdir "$OUT" --outdir-overwrite

rm -f ${SELECT_GPU}
