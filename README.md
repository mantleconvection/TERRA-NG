# TERRA-NG

[![CI](https://github.com/mantleconvection/TERRA-NG/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mantleconvection/TERRA-NG/actions/workflows/ci.yml)
[![Doxygen Docs](https://github.com/mantleconvection/TERRA-NG/actions/workflows/doxygen.yml/badge.svg?branch=main)](https://github.com/mantleconvection/TERRA-NG/actions/workflows/doxygen.yml)

Extreme-scale mantle convection code for CPU and GPU systems.

> ❗️The code is early in development, but feel free to try it out!

## Documentation

📜 Check out the [documentation](https://mantleconvection.github.io/TERRA-NG/) pages or jump directly to one of the 
following sections:

* 🏃‍♀️ [Compiling and running](https://mantleconvection.github.io/TERRA-NG/compiling-and-running.html)
* 💻 [Cluster setup](https://mantleconvection.github.io/TERRA-NG/cluster-setup.html)
* 📖 [Framework documentation](https://mantleconvection.github.io/TERRA-NG/framework-documentation.html)
* 🔨 [How to contribute](https://mantleconvection.github.io/TERRA-NG/contributing.html)

## Quickstart

```bash
git clone https://github.com/mantleconvection/TERRA-NG.git
mkdir TERRA-NG-build
cd TERRA-NG-build
cmake ../TERRA-NG
cd apps/mantlecirculation
make
./mantlecirculation -h
```

## Features

TERRA-NG is a matrix-free finite element code written in modern C++ on top of [Kokkos](https://github.com/kokkos/kokkos)
mainly focused on massively parallel mantle convection simulations on GPU (and CPU) clusters.

An incomplete list of features
* Runs in massively parallel settings on CPU and GPU systems (via [Kokkos](https://github.com/kokkos/kokkos) and MPI)
* Stable discretization of the generalized, compressible Stokes equations (Q1-iso-Q2 / Q1) using spherical wedge finite-elements
* Plate boundary conditions
* Fully matrix-free
* Krylov methods and geometric multigrid preconditioners (using GCA coarse grid operators)
* Memory efficient unified visualization and checkpoint format (using XDMF)
* Tools (input and output of radial profiles, spherical harmonics)
* Written in modern C++20

## License

This project is licensed under the GNU GPLv3.

The directory `extern/` contains third-party code that is NOT covered
by this project's GPLv3 license. Each component in `extern/` retains its
original license; see the license files within each subdirectory.
