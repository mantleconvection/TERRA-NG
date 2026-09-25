#!/bin/bash -l
#SBATCH --job-name=terrang_hourglass
#SBATCH --output=%x_%j.out
#SBATCH --partition=standard-g
# Node and task counts are not fixed here; pass them to sbatch, see README.md.
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#
# Multigrid residual history of the one-point wedge operator with and without
# hourglass control, for one model size. Run once per MT and concatenate.

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_NO_ASYNC_COPY=1
export OMP_NUM_THREADS=1
export FI_MR_CACHE_MAX_COUNT=1048576
export FI_CXI_RX_MATCH_MODE=software
ulimit -c 0

BIN="${TERRANG_BIN:?set TERRANG_BIN to the test_epsilon_divdiv_quadrature_matrix binary}"
MT="${TERRANG_MT:?set TERRANG_MT to one of 32 64 128 256 512}"
OUT="${TERRANG_OUT:-./hourglass_MT${MT}}"
# A relative run directory is taken relative to the submit directory.
case "$OUT" in /*) ;; *) OUT="${SLURM_SUBMIT_DIR:-$PWD}/$OUT";; esac

case "$MT" in
  32) LEVEL=5;; 64) LEVEL=6;; 128) LEVEL=7;; 256) LEVEL=8;; 512) LEVEL=9;;
  *) echo "unsupported TERRANG_MT=$MT" >&2; exit 1;;
esac

# The hierarchy starts at level 1, so the subdomain refinement cannot exceed 1:
# 10 subdomains on 1 rank, 40 with lat_sdr 1, 80 with lat_sdr 1 and rad_sdr 1.
NTASKS="${SLURM_NTASKS:-1}"
case "$NTASKS" in
  1|2|5|10) export SCALE_LAT_SDR=0; export SCALE_RAD_SDR=0;;
  20)       export SCALE_LAT_SDR=0; export SCALE_RAD_SDR=1;;
  40)       export SCALE_LAT_SDR=1; export SCALE_RAD_SDR=0;;
  80)       export SCALE_LAT_SDR=1; export SCALE_RAD_SDR=1;;
  *) echo "no subdomain refinement defined for $NTASKS ranks" >&2; exit 1;;
esac

NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
case "$NTASKS_PER_NODE" in
  1) CPU_BIND=map_cpu:49;;
  2) CPU_BIND=map_cpu:49,57;;
  4) CPU_BIND=map_cpu:49,57,17,25;;
  8) CPU_BIND=map_cpu:49,57,17,25,1,9,33,41;;
  *) CPU_BIND=cores;;
esac

mkdir -p "$OUT"
cd "$OUT"

SELECT_GPU=./select_gpu_${SLURM_JOB_ID:-local}.sh
cat > ${SELECT_GPU} << 'INNER'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
INNER
chmod +x ${SELECT_GPU}

# series "1pt"  : one quadrature point, no stabilisation
# series "1ptK" : one quadrature point plus hourglass control, C = 0.3
# series "2ptK" : two radial quadrature points
RUN_SCALE=$LEVEL                                srun --cpu-bind=${CPU_BIND} ${SELECT_GPU} "$BIN" | grep -E '^(ITER|DISC),' | awk '!seen[$0]++' > iter_MT${MT}_1pt.csv
RUN_SCALE=$LEVEL STAB_C_KERNGEN=0.3             srun --cpu-bind=${CPU_BIND} ${SELECT_GPU} "$BIN" | grep -E '^(ITER|DISC),' | awk '!seen[$0]++' > iter_MT${MT}_1ptK.csv
RUN_SCALE=$LEVEL STAB_C_KERNGEN=0.3 RUN_QP=2    srun --cpu-bind=${CPU_BIND} ${SELECT_GPU} "$BIN" | grep -E '^(ITER|DISC),' | awk '!seen[$0]++' > iter_MT${MT}_2ptK.csv

cat iter_MT${MT}_1pt.csv iter_MT${MT}_1ptK.csv iter_MT${MT}_2ptK.csv > iter_MT${MT}.csv
echo "wrote $OUT/iter_MT${MT}.csv"
