# Model / Partial differential equations {#model-pde}

> **Incomplete.**  This page gives an overview of the core equations.
> Several physically important terms are not yet included, and this list is not exhaustive:
> - **Energy equation**: adiabatic heating/cooling (\f$\propto D_i\,\alpha\,T\,u_r\f$),
>   viscous (shear) dissipation (\f$\propto D_i/\mathrm{Ra}\;\boldsymbol{\tau}:\boldsymbol{\varepsilon}\f$),
>   latent heat, and compositional transport.
> - **Stokes equation**: full compressibility model (anelastic liquid approximation),
>   non-linear and non-Newtonian rheology, inertia.
> - **Coupling**: the buoyancy forcing is currently purely thermal; chemical buoyancy and
>   phase transitions are absent.
> - **Non-dimensionalisation**: the choice of scales and the resulting dimensionless
>   parameters are not documented here.

---

## The mantle convection model

Mantle convection is modelled as the slow creep of a highly viscous fluid in a thick
spherical shell.  The governing equations couple an instantaneous force balance (Stokes)
to a time-dependent heat transport equation (energy / advection-diffusion).

### Stokes equations

> See the [Stokes Discretization and Solver](#stokes) chapter for the full discretization,
> solver, and parameter guidance.

The momentum balance and mass conservation read

\f[
    \begin{aligned}
    -\nabla \cdot \boldsymbol{\tau}(\mathbf{u}) + \nabla p &= \mathbf{f}, \\
    -\nabla \cdot \mathbf{u} &= g,
    \end{aligned}
\f]

where
\f[
    \boldsymbol{\tau}(\mathbf{u})
    = 2\eta\!\left(\boldsymbol{\varepsilon}(\mathbf{u})
      - \tfrac{1}{3}(\nabla\cdot\mathbf{u})\,\mathbf{I}\right), \qquad
    \boldsymbol{\varepsilon}(\mathbf{u})
    = \tfrac{1}{2}\!\left(\nabla\mathbf{u} + (\nabla\mathbf{u})^T\right).
\f]

**Variables and parameters:**

- \f$\mathbf{u}\f$ — velocity,
- \f$p\f$ — dynamic pressure,
- \f$\eta > 0\f$ — dynamic viscosity (spatially varying),
- \f$\mathbf{f}\f$ — body force.  In the **Boussinesq approximation** this is the thermal
  buoyancy \f$\mathbf{f} = \mathrm{Ra}\,T\,\hat{r}\f$, with Rayleigh number
  \f$\mathrm{Ra}\f$ and outward radial unit vector \f$\hat{r}\f$,
- \f$g\f$ — mass-conservation source term (\f$g = 0\f$ for incompressible / Boussinesq flow;
  \f$g \neq 0\f$ for anelastic extensions — see [Stokes: compressible extensions](#stokes-compressible)).

Because inertia is negligible in mantle flow (\f$\mathrm{Re} \ll 1\f$), no time
derivative of velocity appears: the Stokes system is solved quasi-statically at every
time step given the current temperature field.

### Energy equation

> See the [Advection-Diffusion](#advection-diffusion) chapter for the full discretization,
> compressibility corrections, and boundary condition details.

The temperature \f$T\f$ evolves according to the advection-diffusion equation

\f[
    \frac{\partial T}{\partial t}
    + \mathbf{u} \cdot \nabla T
    - \nabla \cdot (\kappa\,\nabla T)
    = f_T,
\f]

where \f$\kappa\f$ is the thermal diffusivity and \f$f_T\f$ collects volumetric heat
sources (e.g. radiogenic heating).  The advective form (\f$\mathbf{u}\cdot\nabla T\f$,
not \f$\nabla\cdot(\mathbf{u}T)\f$) is appropriate for a passive scalar such as
temperature; in the incompressible case the two are equivalent.

### Coupling and time integration

The two systems are coupled through the buoyancy force and are advanced in time by
operator splitting:

1. **Stokes solve** — given \f$T^n\f$, solve for \f$(\mathbf{u}^n, p^n)\f$.
2. **Energy update** — advance \f$T^{n+1}\f$ using \f$\mathbf{u}^n\f$ as the advecting
   velocity.

This decoupled, sequential approach is first-order accurate in time.  The Stokes solve
is computationally dominant; the energy step is typically cheaper due to its scalar
nature and the finite-volume discretization used.

---

## Non-dimensionalisation

> **Not yet documented.**  The key dimensionless group is the Rayleigh number
> \f$\mathrm{Ra} = \rho_0\,g_0\,\alpha\,\Delta T\,D^3 / (\eta_0\,\kappa_0)\f$,
> which controls the vigour of convection.  A full description of the non-dimensionalisation
> and the resulting parameter choices will be added here.