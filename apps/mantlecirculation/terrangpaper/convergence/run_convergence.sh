#!/bin/bash -l
#SBATCH --job-name=terrang_conv
#SBATCH --output=%x_%j.out
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#
# Stokes convergence on the Lin (2022) and Stotz (2017) radial viscosity profiles.
# Node and task counts are not fixed here; pass them to sbatch, see README.md.

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_NO_ASYNC_COPY=1
export OMP_NUM_THREADS=1
export FI_MR_CACHE_MAX_COUNT=1048576
export FI_CXI_RX_MATCH_MODE=software
ulimit -c 0

HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BIN="${TERRANG_BIN:?set TERRANG_BIN to the test_epsilon_divdiv_ablock_mg_gca binary}"
MT="${TERRANG_MT:?set TERRANG_MT to one of 32 64 128 256 512 1024 2048}"
PROFILE="${TERRANG_PROFILE:-lin}"          # lin | stotz
MAX_CYCLES="${TERRANG_MAX_CYCLES:-100}"    # 100 -> iteration count, 10 -> residual after 10 iters
OUT="${TERRANG_OUT:-./conv_MT${MT}_${PROFILE}}"
# A relative run directory is taken relative to the submit directory.
case "$OUT" in /*) ;; *) OUT="${SLURM_SUBMIT_DIR:-$PWD}/$OUT";; esac

# The viscosity CSVs ship with the repository in data/radialprofiles. The default
# assumes the job was submitted from this directory; otherwise set TERRANG_PROFILE_DIR.
export TERRANG_PROFILE_DIR="${TERRANG_PROFILE_DIR:-$HERE/../../../../data/radialprofiles}"
if [ ! -f "$TERRANG_PROFILE_DIR/ViscosityProfile_Lin_et_al_2022.csv" ]; then
  echo "no viscosity profiles under $TERRANG_PROFILE_DIR; set TERRANG_PROFILE_DIR to <repo>/data/radialprofiles" >&2
  exit 1
fi

case "$MT" in
  32) LEVEL=5;; 64) LEVEL=6;; 128) LEVEL=7;; 256) LEVEL=8;;
  512) LEVEL=9;; 1024) LEVEL=10;; 2048) LEVEL=11;;
  *) echo "unsupported TERRANG_MT=$MT" >&2; exit 1;;
esac

case "$PROFILE" in
  lin|stotz) ;;
  *) echo "TERRANG_PROFILE must be lin or stotz" >&2; exit 1;;
esac

# The grid is split into 10 * 4^lat_sdr * 2^rad_sdr subdomains; one rank owns one
# or more of them, so the rank count fixes the refinement.
NTASKS="${SLURM_NTASKS:-1}"
case "$NTASKS" in
  1|2|5|10)  LAT_SDR=0; RAD_SDR=0;;
  20)        LAT_SDR=0; RAD_SDR=1;;
  40)        LAT_SDR=1; RAD_SDR=0;;
  80)        LAT_SDR=1; RAD_SDR=1;;
  160)       LAT_SDR=2; RAD_SDR=0;;
  320)       LAT_SDR=2; RAD_SDR=1;;
  640)       LAT_SDR=2; RAD_SDR=2;;
  1280)      LAT_SDR=3; RAD_SDR=1;;
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

echo "MT$MT  level $LEVEL  profile $PROFILE  ranks $NTASKS  lat_sdr $LAT_SDR  rad_sdr $RAD_SDR  max-cycles $MAX_CYCLES"

srun --cpu-bind=${CPU_BIND} ${SELECT_GPU} "$BIN" \
  --solve stokes --visc-profile ${PROFILE} --gca 0 \
  --min-level 2 --max-level ${LEVEL} \
  --lat-sdr ${LAT_SDR} --rad-sdr ${RAD_SDR} \
  --bc dirichlet --cheby-order 2 --cheby-prepost 3 --coarse-tol 1e-6 \
  --max-cycles ${MAX_CYCLES}
