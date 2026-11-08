# Cachemiss @ LMU Geocomputing {#cachemiss-lmu}

```
$ module load mpi.ompi
$ module load nvidia-hpc

$ mkdir TERRA-NG_build

$ ls -alF
TERRA-NG/               # <== the cloned source code
TERRA-NG_build/

$ cd TERRA-NG_build

$ cmake ../TERRA-NG -DKokkos_ENABLE_CUDA=ON

# Build tests
$ cd tests
$ make -j16
```

Note the capitalization: it must be `Kokkos_ENABLE_CUDA=ON`, NOT `KOKKOS_ENABLE_CUDA=ON`.
