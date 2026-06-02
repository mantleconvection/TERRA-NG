#!/bin/bash -l
#SBATCH --job-name=bo_remeasure_hist
#SBATCH --output=bo_remeasure_hist.o%j
#SBATCH --error=bo_remeasure_hist.e%j
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --nodes=1

# Re-measure the WHOLE EpsDivDiv optimization history (v00a..v10 + current op)
# on the present H100/toolchain, at the canonical config (level 8, sdr 0,
# 505M dofs).  Produces:
#   (1) throughput for every version  -> stdout tables (parse for gdofs/s)
#   (2) ncu roofline metrics per version -> ncu_history_<jobid>.csv
unset SLURM_EXPORT_ENV
module load openmpi/5.0.5-nvhpc24.11-cuda cmake

cd /home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance
BENCH=./benchmark_operators

echo "############ STEP 1: throughput, all versions (level 8, sdr 0, exec=5) ############"
${BENCH} --min-level 8 --max-level 8 --refinement-level-subdomains 0 --executions 5

echo ""
echo "############ STEP 2: ncu roofline metrics, every EpsDivDiv matvec ############"
METRICS=dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__time_duration.sum,launch__registers_per_thread,\
launch__shared_mem_per_block_allocated,lts__t_bytes.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum

# Profile every kernel whose name contains EpsilonDivDiv (one matvec per version
# since --executions 1).  No launch-count cap -> captures all 14 versions.
ncu --metrics "${METRICS}" \
    --kernel-name-base demangled \
    --kernel-name "regex:EpsilonDivDiv" \
    --target-processes all \
    --csv --log-file ncu_history_${SLURM_JOB_ID}.csv \
    ${BENCH} --min-level 8 --max-level 8 --refinement-level-subdomains 0 --executions 1

echo ""
echo "=== wrote ncu_history_${SLURM_JOB_ID}.csv ==="
ls -lh ncu_history_${SLURM_JOB_ID}.csv
echo "=== Done ==="
