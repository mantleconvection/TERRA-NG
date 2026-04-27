# Mantle circulation app {#mcm-app}

The core application of this framework is the simulation of global mantle circulation.
Therefore, one core application `apps/mantlecirculation/mantlecirculation.cpp` is provided that implements all relevant
features.
This page shall serve as an overview of the features and options that that application offers.

\note The app is still under development and features might change and may be briefly out of sync with this
documentation. Also, this documentation is work in progress.

## Governing equations

The app simulates global mantle circulation in the thick spherical shell, solving (an approximation to)
a coupled system consisting of the compressible generalized Stokes system and an energy conservation equation.
The governing equations are the Stokes system
\f[
\begin{aligned}

- \nabla \cdot ( \eta [ \nabla \mathbf{u} + (\nabla \mathbf{u})^T - \frac{2}{3}(\nabla \cdot \mathbf{u})\mathbf{I} ] ) +
  \nabla p &= \rho \mathbf{g} \\
  \nabla \cdot (\rho \mathbf{u}) &= 0
  \end{aligned}
  \f]
  and the energy conservation equation
  \f[
  \rho C_p (\frac{\partial T}{\partial t} + \mathbf{u} \cdot \nabla T) - \nabla \cdot (k \nabla T) = F(\mathbf{u}, p, \rho, T)
  \f]

## Domain and mesh

Domain and mesh are configured via the following parameters:

| App option                      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--radius-min`                  | Radius of the core-mantle-boundary (CMB).                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `--radius-max`                  | Radius of the surface.                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `--refinement-level-mesh-min`   | For multigrid: coarse grid refinement level applied to the original 10 diamonds (>= 0, 0: no refinement, 1: refine once, ...). When in doubt, choose as small as possibly (see below for details).                                                                                                                                                                                                                                                                                                  |
| `--refinement-level-mesh-max`   | Global mesh refinement level applied to the original 10 diamonds (>= 0, 0: no refinement, 1: refine once, ...). Controls the global number of finite elements (regardless of the number of processes).                                                                                                                                                                                                                                                                                              |
| `--refinement-level-subdomains` | Controls the number of subdomains (relevant for parallel load distribution). The underlying diamond structure ensures that we will at least have 10 subdomains (if this is set to 0). The the number of subdomains is increased in factors of 8 (level 1: 10 * 8 = 80 subdomains, level 2: 10 * 8 * 8 = 640 subdomains, ...). This parameter does not change the number of global elements. This also defines the minimum value for `--refinement-level-mesh-min` and `--refinement-level-mesh-max` |

## Time step size

We couple the Stokes and energy conservation equations by solving them in an alternating fashion:

```
Stokes solve -> energy solve -> Stokes solve -> energy solve -> ...
                \          /                    \          /
                 ----------                      ----------
                       \                            /               
                       subject to some time step size dt 
```

There is a balance to strike: a small time step size has to be chosen to ensure three things:
- a stable simulation (it does not blow up),
- a small time discretization error (accuracy of the energy discretization itself),
- a small Stokes-energy coupling error (accuracy of the whole system).

However, solving the Stokes system is expensive, to we still want to choose a time step size that is as large as possible.

In the `mantlecirculation` app, the time step size is controlled as follows:

First, an initial time step size is computed based on the fluxes (i.e., the velocity) using the function
[compute_dt_stable()](@ref terra::fv::hex::operators::compute_dt_stable). 
However, this function only yields a stable time step size and does not 
guarantee any kind of accuracy. That being said, it is likely that a much smaller time step size is required for 
accuracy reasons.

A scaling parameter `--dt-scaling` is provided to scale the estimated "stable" time step size. Typically, that one can 
be chosen as `0.1` or a similar small value.

Eventually, such a choice can, however, lead to many Stokes solves for a small simulated time.
Therefore, one can also choose to perform more than one energy solve per time step.
This is controlled by the parameter `--energy-substeps`.

The result is that between two Stokes solves, there are `energy_substeps` energy solves with a time step size of
`dt_stable * dt_scaling`. Choosing `energy_substeps > 1` does **not** reduce the time step size.
It basically just repeats energy solves.

For example:
```
[...]

Stokes solve

Energy solve   // dt = dt_stable * dt_scaling    | repeats energy_substeps times
Energy solve   // dt = dt_stable * dt_scaling    |
[...]                                            |
Energy solve   // dt = dt_stable * dt_scaling    | => simulated_time = energy_substeps * dt_stable * dt_scaling

Stokes solve

[...]
```

To get the same simulated time between two Stokes solves, the product (`dt_scaling * energy_substeps`) has to be constant.
Since the energy solves are relatively inexpensive, it might make sense to choose a large number of energy substeps and a 
small scaling value to decrease the time discretization error. If the above product is constant, that probably has
little effect on the Stokes-energy coupling error. 

### An example

\note More details in the [parallelization section](#parallelization).

Let's assume 512 MPI processes, and `--refinement-level-mesh-max 8` resulting in 167,772,160 elements globally.

We would like to have a multiple of 512 subdomains, or alternatively some number that is slightly below a multiple of
512.
Slightly above would be detrimental for performance. Consider having 513 subdomains. One process then processes 2
subdomains, all remaining processes process 1 subdomain.
The latter will have to wait until the single process has executed twice the average work and will be idle.

Let's see what we can do.

| `--refinement-level-subdomains` | Number of subdomains | Elements per subdomain |
|---------------------------------|----------------------|------------------------|
| `0`                             | 10                   | 16,777,216             |
| `1`                             | 80                   | 2,097,152              |
| `2`                             | 640                  | 262,144                |
| `3`                             | 5120                 | 32,768                 |

With `--refinement-level-subdomains 3` the workload would be balanced perfectly with 10 subdomains per process and
`10 * 32,768` elements per process.

Since each subdomain must at least have one element, we have to set `--refinement-level-mesh-min 3`.





