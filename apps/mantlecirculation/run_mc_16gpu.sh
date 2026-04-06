#!/bin/bash -x
#SBATCH --account=walberlamovinggeo
#SBATCH --partition=booster
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --time=12:00:00
#SBATCH --output=mc_level6_16gpu_%j.out
#SBATCH --error=mc_level6_16gpu_%j.err

source ~/.bashrc

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun /p/home/jusers/boehm2/juwels/terraneo-build/apps/mantlecirculation/mantlecirculation \
    --config /p/home/jusers/boehm2/juwels/terraneo/apps/mantlecirculation/level6_16gpu.toml
