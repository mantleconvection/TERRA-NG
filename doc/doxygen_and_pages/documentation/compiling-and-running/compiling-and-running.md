# Compiling and running {#compiling-and-running}

We use the CMake build system, Kokkos for performance portability (compiling for CPU and GPU systems) and MPI for 
distributed memory parallelism.

## Obtaining the source code

If you just want to compile and run the code, you can simply clone the repository:
```
git clone https://github.com/mantleconvection/TERRA-NG
```

However, if you want to modify the code, you should fork the repository and clone your fork.
Have a look at the [contributing guidelines](#contributing) for more information.

## Configuring with CMake

For an out-of-source build (recommended), create a new directory (typically next to the source code) and run `cmake`
in that directory passing the source directory as an argument.

```
mkdir TERRA-NG_build
cd TERRA-NG_build
cmake ../TERRA-NG
```

This will create Makefiles in the build directory.

### Building for GPU systems

Without any additional arguments, Kokkos will be configured to compile for CPU systems only.
To compile for GPU systems, you need to pass flags like `-DKokkos_ENABLE_CUDA=ON` (for NVIDIA systems) to CMake.

\note Have a look at the hints for configuring on various systems in the 
[cluster setup recipes](#cluster-setup).

## Building

Run `make` in the build directory to build the code. You can also specify a specific target of course:
```
cd apps/mantlecirculation
make mantlecirculation
```

## Running

Tests can be run after building from the `tests/` directory by invoking `ctest`.
Apps, benchmarks, etc. should usually provide some usage instructions when passing `-h`.

### Multi-GPU systems

Run with one MPI process per GPU.


