#!/bin/bash -l
#SBATCH --job-name=mc_C1_lvl7_sdr1
#SBATCH --output=mc_C1_lvl7_sdr1.o%j
#SBATCH --error=mc_C1_lvl7_sdr1.e%j
#SBATCH --partition=general
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --account=pn29po

module load slurm_setup

module sw stack/24.5.0
module load cmake gcc/14.2.0
module load intel-toolkit/2025.2.0

module list

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

srun /home/fabi/sng_mount/terraneo-build/apps/mantlecirculation/mantlecirculation --config /home/fabi/sng_mount/terraneo/apps/mantlecirculation/config_C1_lvl7.toml

