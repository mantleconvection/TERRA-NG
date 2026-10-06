# SuperMUC NG Phase 2 @ LRZ {#supermuc-ng-phase-2}

For login information, inspect the [operation documentation of SuperMUC-NG Phase 2](https://doku.lrz.de/pilot-operation-supermuc-ng-phase-2-403079197.html).

The setup below is inspired by the [entity toolkit](https://entity-toolkit.github.io/wiki/content/useful/cluster-setups/#__tabbed_1_12).

\note Use this as a starting point. You might need to adjust the module versions and possibly want to adapt the 
configuration for optimal performance.


## Modules

```
$ module sw stack/24.5.0
$ module load cmake gcc/14.2.0
$ module load intel-toolkit/2025.2.0

$ module list
Currently Loaded Modulefiles:
1) mpi_settings/2.0{mode=default}   3) cmake/3.30.0   5) intel/2025.2.0        7) intel-mkl/2025.2.0         9) intel-dnn/2025.2.0   11) intel-tbb/2022.2.0  13) intel-dal/2025.6.0    15) intel-dpl/2022.9.0   17) intel-toolkit/2025.2.0
2) stack/24.5.0{arch=auto}          4) gcc/14.2.0     6) intel-mpi/2021.16.0   8) intel-inspector/2024.1.0  10) intel-itac/2022.4.0  12) intel-ipp/2022.2.0  14) intel-ippcp/2025.2.0  16) intel-dpct/2025.2.0

Key:
auto-loaded  default-version  {variant=value}
```


## CMake configuration

Assuming an out-of-source build:
```
$ ls -alF
TERRA-NG/         # << src  
TERRA-NG_build/   # << build
$ cd TERRA-NG_build
```

Run CMake like this:
```
$ source /lrz/sys/intel/oneapi_2025.2.0/setvars.sh &> /dev/null
$ export CC=$(which icx)
$ export CXX=$(which icpx)
$ export FC=$(which ifx)

$ cmake ../TERRA-NG -DKokkos_ARCH_INTEL_PVC=ON -DKokkos_ARCH_SPR=ON -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_SYCL=ON -DKokkos_ENABLE_SYCL_RELOCATABLE_DEVICE_CODE=ON -DCMAKE_CXX_COMPILER=$(which mpiicpx)
```
Note that some options are likely not strictly necessary (maybe do not need CMAKE_C_COMPILER).


## Job file

\note You have to adapt this to your account/project. If unsure, please refer to the documentation by the LRZ.

```
#!/bin/bash
#SBATCH -J my-tn-benchmark
#SBATCH -o ./%N.%j.out
#SBATCH -D .
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8   # one task per tile
#SBATCH --account=<your-project>
#SBATCH --export=none
#SBATCH --time=00:05:00

module load slurm_setup

module sw stack/24.5.0
module load cmake gcc/14.2.0
module load intel-toolkit/2025.2.0

module list

export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_RDMA=1
export I_MPI_OFFLOAD_FAST_MEMCPY_COLL=1
export PSM3_RDMA=1
export PSM3_GPUDIRECT=0 # this will hopefully be fixed in the future

# this is just for debugging reasons
export I_MPI_DEBUG=5

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

mpiexec ./benchmark_baseline --kokkos-print-configuration
```

Then run `sbatch jobfile.job`.



