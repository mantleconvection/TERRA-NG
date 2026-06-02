#!/bin/bash -l
#SBATCH --job-name=bo_ncu_lean
#SBATCH --output=bo_ncu_lean.o%j
#SBATCH --error=bo_ncu_lean.e%j
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --time=00:40:00
#SBATCH --export=NONE
#SBATCH --nodes=1

# Lean ncu pass for the roofline: counter metrics only (no expensive SOL
# %-of-peak metrics), profiling only EpsilonDivDivKerngen* (v01..v10 + current).
# The two slow baselines (EpsilonDivDivSimple=v00a, EpsilonDivDiv=v00b) are
# excluded from profiling by the kernel-name regex (they still run unprofiled,
# ~1 min); their roofline coords are unchanged from the old ncu_data.csv.
# Throughput (gdofs/s) for ALL versions was already captured in job 467167.
unset SLURM_EXPORT_ENV
module load openmpi/5.0.5-nvhpc24.11-cuda cmake
cd /home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance
BENCH=./benchmark_operators

# Counter-only metrics (collect in ~1-2 replay passes). % of peak can be
# derived later from these + known H100 peaks.
# Cheap HW-counter metrics only (NO sass_thread_inst FLOP counts -- those force
# slow instruction-counting that takes ~18 min on the v00a/v00b baselines).
# FLOPs are deterministic on sm_90 and already known (frozen snapshots +
# run 467249 for current), so we don't re-measure them here.
METRICS=dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_bytes.sum,gpu__time_duration.sum,\
launch__registers_per_thread,launch__shared_mem_per_block_allocated,\
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,\
smsp__inst_executed_op_local_ld.sum,\
smsp__inst_executed_op_local_st.sum

echo "=== ncu lean: all EpsDivDiv matvecs (v00a, v00b, v01..v10 + current) ==="
ncu --metrics "${METRICS}" \
    --kernel-name-base demangled \
    --kernel-name "regex:EpsilonDivDiv" \
    --target-processes all \
    --csv --log-file ncu_history_lean_${SLURM_JOB_ID}.csv \
    ${BENCH} --min-level 8 --max-level 8 --refinement-level-subdomains 0 --executions 1

echo ""
echo "=== wrote ncu_history_lean_${SLURM_JOB_ID}.csv ==="
ls -lh ncu_history_lean_${SLURM_JOB_ID}.csv
echo "=== Done ==="
