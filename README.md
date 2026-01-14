# TerraNeo

[![CI](https://github.com/mantleconvection/terraneo/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mantleconvection/terraneo/actions/workflows/ci.yml)
[![Doxygen Docs](https://github.com/mantleconvection/terraneo/actions/workflows/doxygen.yml/badge.svg?branch=main)](https://github.com/mantleconvection/terraneo/actions/workflows/doxygen.yml)

Extreme-scale mantle convection code for CPU and GPU systems. Originating from the [TerraNeo project](https://terraneo.fau.de).

> ❗️The code is early in development, and thus not yet ready for production. But feel free to try it out!

## Documentation

📜 Check out the [documentation](https://mantleconvection.github.io/terraneo/) pages or jump directly to one of the 
following sections:

* 🏃‍♀️ [Compiling and running](https://mantleconvection.github.io/terraneo/compiling-and-running.html)
* 💻 [Cluster setup](https://mantleconvection.github.io/terraneo/cluster-setup.html)
* 📖 [Framework documentation](https://mantleconvection.github.io/terraneo/framework-documentation.html)
* 🔨 [How to contribute](https://mantleconvection.github.io/terraneo/contributing.html)

## Quickstart

```bash
git clone https://github.com/mantleconvection/terraneo.git
mkdir terraneo-build
cd terraneo-build
cmake ../terraneo
cd apps/mantlecirculation
make
./mantlecirculation -h
```

## Features

TerraNeo is a matrix-free finite element code written in modern C++ on top of [Kokkos](https://github.com/kokkos/kokkos)
mainly focused on massively parallel mantle convection simulations on GPU (and CPU) clusters.

An incomplete list of features
* Runs in massively parallel settings on CPU and GPU systems (via [Kokkos](https://github.com/kokkos/kokkos) and MPI)
* Spherical wedge finite-elements
* Stable discretization of the generalized, compressible Stokes equations (Q1-iso-Q2 / Q1)
* Advection-Diffusion discretization using SUPG
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