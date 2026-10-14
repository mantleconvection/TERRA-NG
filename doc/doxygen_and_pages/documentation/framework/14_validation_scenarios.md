
# C1, C3 and A3 Validation Scenarios

## Overview

This document presents validation results for the TERRA-NG mantle convection code against community benchmark cases from Zhong et al. (2008) and Ratcliff et al. (1996), using published reference data from Ilangovan et al. (2026) (HYTEG, [doi:10.5194/gmd-19-1455-2026](https://gmd.copernicus.org/articles/19/1455/2026/gmd-19-1455-2026.pdf)), Euen et al. (2023), and other codes (ASPECT, CitcomS).

Three benchmark cases are considered on the thick spherical shell with \f$r_\text{min} = 1.22\f$, \f$r_\text{max} = 2.22\f$ (aspect ratio \f$\approx 0.55\f$, matching Earth's mantle geometry).

## Physical Setup

All cases solve the coupled Stokes--energy system with free-slip velocity boundary conditions on both CMB and surface, and Dirichlet temperature (\f$T_\text{CMB} = 1\f$, \f$T_\text{surface} = 0\f$).

**Viscosity law:** Frank-Kamenetskii: \f$\mu = r_\mu^{(0.5 - T)}\f$, giving a total viscosity contrast of \f$r_\mu\f$ between \f$T = 0\f$ (cold) and \f$T = 1\f$ (hot).

**Initial condition:** Conductive reference profile with a small spherical harmonic perturbation (\f$\epsilon = 0.01\f$).

## Benchmark Cases

| Case | Ra | \f$r_\mu\f$ | IC symmetry | Perturbation | Reference \f$\text{Nu}_\text{top}\f$ |
|------|-----|---------|-------------|--------------|----------------------------------|
| A3   | \f$7 \times 10^3\f$ | 20 | Tetrahedral | \f$Y_3^2\f$ | 3.14--3.19 |
| C1   | \f$1 \times 10^5\f$ | 1  | Cubic       | \f$Y_4^0 + \tfrac{5}{7} Y_4^4\f$ | 7.37--7.81 |
| C3   | \f$1 \times 10^5\f$ | 30 | Cubic       | \f$Y_4^0 + \tfrac{5}{7} Y_4^4\f$ | 6.50--6.79 |

The reference \f$\text{Nu}_\text{top}\f$ ranges are compiled from HYTEG (Ilangovan et al., 2026), ASPECT, and CitcomS results reported in Euen et al. (2023) and Davies et al. (2022).

## Numerical Method

- **Spatial discretization:** Q1 wedge finite elements on an icosahedral spherical shell mesh, refinement level 6 (\f$h \approx 1/64\f$).
- **Energy solver:** SUPG (Streamline Upwind Petrov-Galerkin) with implicit BDF1 time stepping.
- **Stokes solver:** FGMRES(10) with Chebyshev-smoothed geometric multigrid preconditioner for the viscous block (1 V-cycle, order-2 Chebyshev, 3 pre/post smoothing steps).
- **Time stepping:** Picard coupling with 2 iterations per timestep (Stokes \f$\to\f$ Energy \f$\to\f$ Stokes \f$\to\f$ Energy), ensuring tight velocity--temperature coupling at each step.
- **CFL:** Pseudo-CFL based on the advective constraint \f$\Delta t = \text{cfl} \cdot h / |\mathbf{u}|_\text{max}\f$.

### Solver parameters

| Parameter | Value |
|-----------|-------|
| Mesh refinement (min--max) | 2--6 |
| Energy solver | SUPG |
| Picard iterations | 2 |
| Stokes FGMRES restart | 10 |
| Stokes FGMRES max iterations | 10 |
| Stokes relative tolerance | \f$10^{-6}\f$ |
| Chebyshev smoother order | 2 |
| Pre/post smoothing steps | 3 |
| \f$\kappa\f$ (diffusivity) | 1 |

## Results

### Case A3: \f$\text{Ra} = 7 \times 10^3\f$, \f$r_\mu = 20\f$, tetrahedral IC

**Configuration:** SUPG, pseudo-CFL = 0.25.

**Result:** \f$\text{Nu}_\text{top} = 3.149\f$, inside the published range (3.14--3.19) .

<img width="3000" height="1500" alt="nu_and_profiles_A3_supg_cfl025" src="https://github.com/user-attachments/assets/42e3bf89-6172-4155-a85b-ddc68d09b176" />


### Case C1: \f$\text{Ra} = 10^5\f$, \f$r_\mu = 1\f$, cubic IC

**Configuration:** SUPG, pseudo-CFL = 0.5, picard = 2.

With \f$r_\mu = 1\f$ the viscosity contrast is only 1:1 (effectively isoviscous in the code's Frank-Kamenetskii formulation \f$\mu = 1^{(0.5-T)} = 1\f$).

**Result:** \f$\text{Nu}_\text{top} = 7.551\f$, inside the published range (7.37--7.81).

<img width="3000" height="1500" alt="nu_and_profiles_C1_supg_cfl05_picard2" src="https://github.com/user-attachments/assets/fabeb2de-ca30-487c-9f59-4750c45c647f" />


### Case C3: \f$\text{Ra} = 10^5\f$, \f$r_\mu = 30\f$, cubic IC

**Configuration:** SUPG, pseudo-CFL = 0.5, picard = 2.

With \f$r_\mu = 30\f$ the viscosity varies by a factor of 30 across the temperature range. 

**Result:** \f$\text{Nu}_\text{top} = 6.672\f$, inside the published range (6.50--6.79).

<img width="3000" height="1500" alt="nu_and_profiles_C3_supg_cfl05_picard2" src="https://github.com/user-attachments/assets/4f717d49-84dc-445c-af22-f6cc081c893c" />


## References

- Ilangovan, P., Kohl, N., and Mohr, M.: Highly scalable geodynamic simulations with HYTEG, Geosci. Model Dev., 19, 1455--1472, https://doi.org/10.5194/gmd-19-1455-2026, 2026.

## Computational Resources

All simulations were performed on JUWELS Booster at the Juelich Supercomputing Centre (JSC), using 3 nodes (10 A100 GPUs) per run with 24-hour walltimes. The compute budget was provided by the walberlamovinggeo project allocation.
