# Framework documentation {#framework-documentation}

Below is a list of the documentation pages for the framework introducing various concepts.

| Section                                              | Description                                                                            |
|------------------------------------------------------|----------------------------------------------------------------------------------------|
| [Model / Partial Differential Equations](#model-pde) | Core PDE formulation and governing equations.                                          |
| [Stokes Discretization and Solver](#stokes)          | FE discretization of the incompressible Stokes equations and the block-structured solver. |
| [Free-slip Stokes verification](#freeslip-stokes)    | Spherical-harmonic convergence tests and the fs/fs null-space penalty.                |
| [Advection-Diffusion](#advection-diffusion)          | Handling of the advection-diffusion equation ("energy equation")                       |
| [Entropy-viscosity stabilization](#entropy-viscosity) | Residual-driven artificial diffusion for the Q1 advection–diffusion solve.            |
| [Grid Structure](#grid-subdomains)                   | Grids, subdomains, allocation, kernels, Kokkos, etc.                                   |
| [Finite Element Discretization](#finite-elements)    | Overview of the FEM approach and element definitions.                                  |
| [Linear Algebra](#linear-algebra)                    | Matrix and vector representations, solvers, and preconditioners (including multigrid). |
| [Thick Spherical Shell](#shell)                      | Details on the thick spherical shell mesh.                                             |
| [Parallelization](#parallelization)                  | Parallel execution patterns.                                                           |
| [Communication](#communication)                      | Data exchange patterns and MPI communication strategies.                               |
| [Flag Fields and Masks](#flag-fields-and-masks)      | Use of masks and flag grids for selective operations and boundary tagging.             |
| [Boundary Conditions](#boundary-conditions)          | Definition and application of boundary conditions.                                     |
| [Input and Output](#input-output)                    | Data formats, visualization, radial profiles, checkpoints, logging, etc.               |
| [Mantle circulation app](#mcm-app)                   | Details on the mantle circulation app.                                                 |


