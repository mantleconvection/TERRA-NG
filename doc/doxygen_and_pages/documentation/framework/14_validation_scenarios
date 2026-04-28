
# C1, C3 and A3 Validation Scenarios

## Overview

This document presents validation results for the TerraNeo mantle convection code against community benchmark cases from Zhong et al. (2008) and Ratcliff et al. (1996), using published reference data from Ilangovan et al. (2026) (HYTEG, [doi:10.5194/gmd-19-1455-2026](https://gmd.copernicus.org/articles/19/1455/2026/gmd-19-1455-2026.pdf)), Euen et al. (2023), and other codes (ASPECT, CitcomS).

Three benchmark cases are considered on the thick spherical shell with $r_\text{min} = 1.22$, $r_\text{max} = 2.22$ (aspect ratio $\approx 0.55$, matching Earth's mantle geometry).

## Physical Setup

All cases solve the coupled Stokes--energy system with free-slip velocity boundary conditions on both CMB and surface, and Dirichlet temperature ($T_\text{CMB} = 1$, $T_\text{surface} = 0$).

**Viscosity law:** Frank-Kamenetskii: $\mu = r_\mu^{(0.5 - T)}$, giving a total viscosity contrast of $r_\mu$ between $T = 0$ (cold) and $T = 1$ (hot).

**Initial condition:** Conductive reference profile with a small spherical harmonic perturbation ($\epsilon = 0.01$).

## Benchmark Cases

| Case | Ra | $r_\mu$ | IC symmetry | Perturbation | Reference $\text{Nu}_\text{top}$ |
|------|-----|---------|-------------|--------------|----------------------------------|
| A3   | $7 \times 10^3$ | 20 | Tetrahedral | $Y_3^2$ | 3.14--3.19 |
| C1   | $1 \times 10^5$ | 1  | Cubic       | $Y_4^0 + \tfrac{5}{7} Y_4^4$ | 7.37--7.81 |
| C3   | $1 \times 10^5$ | 30 | Cubic       | $Y_4^0 + \tfrac{5}{7} Y_4^4$ | 6.50--6.79 |

The reference $\text{Nu}_\text{top}$ ranges are compiled from HYTEG (Ilangovan et al., 2026), ASPECT, and CitcomS results reported in Euen et al. (2023) and Davies et al. (2022).

## Numerical Method

- **Spatial discretization:** Q1 wedge finite elements on an icosahedral spherical shell mesh, refinement level 6 ($h \approx 1/64$).
- **Energy solver:** SUPG (Streamline Upwind Petrov-Galerkin) with implicit BDF1 time stepping.
- **Stokes solver:** FGMRES(10) with Chebyshev-smoothed geometric multigrid preconditioner for the viscous block (1 V-cycle, order-2 Chebyshev, 3 pre/post smoothing steps).
- **Time stepping:** Picard coupling with 2 iterations per timestep (Stokes $\to$ Energy $\to$ Stokes $\to$ Energy), ensuring tight velocity--temperature coupling at each step.
- **CFL:** Pseudo-CFL based on the advective constraint $\Delta t = \text{cfl} \cdot h / |\mathbf{u}|_\text{max}$.

### Solver parameters

| Parameter | Value |
|-----------|-------|
| Mesh refinement (min--max) | 2--6 |
| Energy solver | SUPG |
| Picard iterations | 2 |
| Stokes FGMRES restart | 10 |
| Stokes FGMRES max iterations | 10 |
| Stokes relative tolerance | $10^{-6}$ |
| Chebyshev smoother order | 2 |
| Pre/post smoothing steps | 3 |
| $\kappa$ (diffusivity) | 1 |

## Results

### Case A3: $\text{Ra} = 7 \times 10^3$, $r_\mu = 20$, tetrahedral IC

**Configuration:** SUPG, pseudo-CFL = 0.25.

**Result:** $\text{Nu}_\text{top} = 3.149$, inside the published range (3.14--3.19) .

<img width="3000" height="1500" alt="nu_and_profiles_A3_supg_cfl025" src="https://github.com/user-attachments/assets/42e3bf89-6172-4155-a85b-ddc68d09b176" />


### Case C1: $\text{Ra} = 10^5$, $r_\mu = 1$, cubic IC

**Configuration:** SUPG, pseudo-CFL = 0.5, picard = 2.

With $r_\mu = 1$ the viscosity contrast is only 1:1 (effectively isoviscous in the code's Frank-Kamenetskii formulation $\mu = 1^{(0.5-T)} = 1$).

**Result:** $\text{Nu}_\text{top} = 7.551$, inside the published range (7.37--7.81).

<img width="3000" height="1500" alt="nu_and_profiles_C1_supg_cfl05_picard2" src="https://github.com/user-attachments/assets/fabeb2de-ca30-487c-9f59-4750c45c647f" />


### Case C3: $\text{Ra} = 10^5$, $r_\mu = 30$, cubic IC

**Configuration:** SUPG, pseudo-CFL = 0.5, picard = 2.

With $r_\mu = 30$ the viscosity varies by a factor of 30 across the temperature range. 

**Result:** $\text{Nu}_\text{top} = 6.672$, inside the published range (6.50--6.79).

<img width="3000" height="1500" alt="nu_and_profiles_C3_supg_cfl05_picard2" src="https://github.com/user-attachments/assets/4f717d49-84dc-445c-af22-f6cc081c893c" />


## References

- Ilangovan, P., Kohl, N., and Mohr, M.: Highly scalable geodynamic simulations with HYTEG, Geosci. Model Dev., 19, 1455--1472, https://doi.org/10.5194/gmd-19-1455-2026, 2026.

## Computational Resources

All simulations were performed on JUWELS Booster at the Juelich Supercomputing Centre (JSC), using 3 nodes (10 A100 GPUs) per run with 24-hour walltimes. The compute budget was provided by the walberlamovinggeo project allocation.
