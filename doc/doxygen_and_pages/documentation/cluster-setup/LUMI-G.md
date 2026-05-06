# Lumi Supercomputer {#LUMI-G}

Access: https://docs.lumi-supercomputer.eu/firststeps/accessLUMI/

SSH Connect: https://docs.lumi-supercomputer.eu/firststeps/loggingin/

LUMI uses a combination of AMD MI250 GPUs (https://docs.lumi-supercomputer.eu/hardware/lumig/) and the Cray Programming Environment (PE) (https://docs.lumi-supercomputer.eu/runjobs/lumi_env/softwarestacks/).
No specific module files for compiler, mpi etc are required.


# programming enviroment, gpu partition on LUMI, AMD gpu support
$ module load PrgEnv-amd 
$ module load partition/G 
$ module load craype-accel-amd-gfx90a
$ module load rocm

# CMake configure:
$ cd ~/terraneo-build
$ cmake -S ~/terraneo -B ~/terraneo-build   -DCMAKE_C_COMPILER=cc   -DCMAKE_CXX_COMPILER=$(which hipcc) 
  -DCMAKE_CXX_STANDARD=20   -DCMAKE_CXX_STANDARD_REQUIRED=ON   -DMPI_C_COMPILER=cc   -DMPI_CXX_COMPILER=CC   
  -DKokkos_ENABLE_HIP=ON   -DKokkos_ARCH_VEGA90A=ON Kokkos_ARCH_AMD_GFX90A
# We explicitly set the cpp compiler to make sure it knows about gpus

# Build tests
$ cd tests
$ make -j 16


Examples for slurm job scripts can be found here: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/lumig-job/
