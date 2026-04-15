# Cachemiss @ LMU Geocomputing {#cachemiss-lmu}

```
$ module load mpi.ompi
$ module load nvidia-hpc

$ mkdir terraneo-build

$ ll
terraneo/               # <== the cloned source code
terraneo-build/

$ cd terraneo-build

$ cmake ../terraneo/ -DKokkos_ENABLE_CUDA=ON

# Build tests
$ cd tests
$ make -j16
```

Note the capitalization: it must be `Kokkos_ENABLE_CUDA=ON`, NOT `KOKKOS_ENABLE_CUDA=ON`.
