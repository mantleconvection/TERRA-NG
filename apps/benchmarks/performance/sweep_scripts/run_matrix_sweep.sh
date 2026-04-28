#!/bin/bash -l
#SBATCH --job-name=mat_sw
#SBATCH --output=mat_sw.o%j
#SBATCH --error=mat_sw.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=0:20:00
#SBATCH --account=project_465002889

export MPICH_GPU_SUPPORT_ENABLED=1
export ROCR_VISIBLE_DEVICES=0

cd /pfs/lustrep3/users/bohmfabi/terraneo-build/apps/benchmarks/performance
BENCH=./benchmark_operators
COMMON="--min-level 8 --max-level 8 --executions 5"

# Grid of (lateral, radial) tile sizes, r_passes=1
LATERAL=(1 2 3 4 8)
RADIAL=(2 4 8 16 32 64)

echo "=== Matrix sweep lat × r (r_passes=1) ==="

for LAT in "${LATERAL[@]}"; do
    for R in "${RADIAL[@]}"; do
        TEAM=$((2 * LAT * LAT * R))
        # Skip configs with team > 1024 (hardware max)
        if [ $TEAM -gt 1024 ]; then
            echo "--- (${LAT},${R},1) team=${TEAM} SKIP (too big) ---"
            continue
        fi
        echo "--- (${LAT},${R},1) team=${TEAM} ---"
        srun --cpu-bind=map_cpu:49 ${BENCH} ${COMMON} \
            --lat-tile $LAT --r-tile $R --r-passes 1
    done
done

echo "=== Done ==="
