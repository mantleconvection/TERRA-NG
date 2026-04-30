# Entropy-viscosity stabilization {#entropy-viscosity}

Entropy viscosity (EV) is a residual-driven, isotropic artificial-diffusion stabilization
for the advection–diffusion energy equation discretised in Q1 on the wedge mesh.  It is the
alternative to SUPG: instead of an always-on streamline diffusion that smears sharp
features, EV adds a strictly local diffusion `ν_h(x)` that vanishes wherever the solution
is smooth and grows only near boundary layers and plumes.

We follow Kronbichler, Heister & Bangerth (*GJI* **191**, 12–29, 2012), which itself adapts
Guermond, Pasquetti & Popov (*JCP* **230**, 4248–4267, 2011).

## The recipe

Define an entropy
\f[
    E(T) = \tfrac{1}{2}(T - T_m)^2,\qquad T_m = \tfrac{1}{2}(T_{\min} + T_{\max}),
\f]
and the strong-form entropy residual on each element \f$K\f$
\f[
    r_E\big|_K = \big\| \partial_t E + (T - T_m)\,\big(\mathbf{u}\cdot\nabla T - \kappa\,\nabla^2 T - \gamma\big) \big\|_{\infty,\,K}.
\f]
With the global normalization
\f[
    E_\mathrm{avg} = \frac{1}{|\Omega|}\int_\Omega E\,\mathrm{d}V,
    \qquad
    D = \|E - E_\mathrm{avg}\|_{\infty,\,\Omega},
\f]
the artificial viscosity per element is capped by a first-order upwind bound:
\f[
    \nu_h\big|_K = \min\!\Big(\,\alpha_\mathrm{max}\,h_K\,\|\mathbf{u}\|_{\infty, K},\;\;
                              \alpha_E\,h_K^2\,r_E\big|_K\,/\,D\,\Big),
\f]
with \f$\alpha_\mathrm{max} = 0.026\,d\f$ (so 0.078 in 3D) and \f$\alpha_E = 1\f$ as in
ASPECT.  Where \f$r_E \to 0\f$ no diffusion is added; where it is large, the upwind cap
guarantees that EV adds at most the diffusion of a first-order monotone scheme.

The residual is **time-lagged** — \f$\partial_t E\f$ uses past states only — so \f$\nu_h\f$
is fully determined before the implicit advection–diffusion solve, which therefore stays
linear.  The EV contribution enters as an explicit RHS term
\f$-\Delta t\,\nabla\!\cdot\!(\nu_h\,\nabla T^n)\f$ on the otherwise pure-Galerkin solve.

## Where things live on the discrete grid

\f$\nu_h\f$ is **piecewise constant per wedge** — two values per hex cell — evaluated at
the same Felippa 3×2 quadrature used by every other wedge operator.  The element length
\f$h_w = V_\mathrm{wedge}^{1/3}\f$ is computed from the same quadrature.  The Laplacian
term \f$-\kappa\nabla^2 T\f$ inside the residual is projected weakly onto each wedge's own
nodes via a per-wedge lumped-mass projection
(\f$\mathrm{lap}_w = \kappa\,K_w T_w / M_{w,\mathrm{lumped}}\f$), so the whole stabilization
pipeline is single-element local — no halo exchange of the stabilization field is needed.
