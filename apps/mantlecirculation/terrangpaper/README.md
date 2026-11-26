# TERRA-NG model description paper: inputs

Configurations and run scripts behind the figures of

> Böhm, Kohl, Ilangovan, Robl, Rezaei, Mohr, Schuberth, Köstler, Bunge, Rüde:
> *TERRA-NG v1.0: Extreme-Scale, GPU-accelerated Mantle Convection*,
> preprint, [arXiv:2609.21633](https://arxiv.org/abs/2609.21633) (2026).

Each subdirectory has its own README naming the driver, the command that runs
it and what comes out.

| directory | reproduces |
|---|---|
| `hourglass/` | **Fig. 2** — stand-alone multigrid residual per V-cycle on MT256 and the L² velocity error over MT32–MT512, for one-point quadrature with and without hourglass control and for the two-radial-point rule |
| `convergence/` | **Fig. 3** — the two tables beside the Lin and Stotz viscosity profiles: full-Stokes FGMRES iterations to a 10⁻⁶ relative residual and the residual after ten iterations, MT32–MT2048 |
| `benchmarks/` | **Fig. 4** and **Table 2** — steady-state radial temperature profiles and T = 0.5 isosurfaces of ten Zhong et al. (2008) benchmark cases, with their Nusselt numbers |
| `scaling/` | **Fig. 5** — strong scaling of the coupled Stokes–energy timestep, time per step against device count with one line per MT resolution, on SuperMUC-NG Phase 2 and LUMI-G |
| `production/` | **Fig. 6** — the end-to-end MT1024 run: radial-mean diagnostics, the temperature-deviation isosurface and the equatorial cross-cut |

`benchmarks/`, `production/` and `scaling/` drive the `mantlecirculation` app
and are configured through TOML files plus command-line refinement levels.
`convergence/` and `hourglass/` drive stand-alone solver tests from `tests/`.

All scripts take their paths from the environment: `TERRANG_BIN` points at the
binary and `TERRANG_OUT` at the run directory. The batch account goes on the
`sbatch` line. Nothing is hard-coded to a machine.
