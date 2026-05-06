# Free-slip Stokes verification {#freeslip-stokes}

The Stokes solver is verified against analytical spherical-shell solutions of
[assess](https://github.com/egpbos/assess) (with the fs–ns extension by
Ponsuganth Ilangovan Ponkumar Ilango).  The benchmark is a Y₂₂ poloidal
spherical-harmonic flow on the shell \f$R_m = 0.5,\ R_p = 1\f$ with \f$\nu = 1,\ g = 1\f$,
solved in three boundary-condition configurations:

- **zs/zs** — zero-slip (Dirichlet) at CMB and surface
  (\ref test_epsilon_divdiv_stokes_assess.cpp),
- **fs/zs** — free-slip at CMB, zero-slip at surface
  (\ref test_epsilon_divdiv_stokes_assess_freeslip.cpp),
- **fs/fs** — free-slip at both CMB and surface
  (\ref test_epsilon_divdiv_stokes_assess_freeslip_freeslip.cpp).

All three reach the expected \f$\mathcal{O}(h^2)\f$ convergence in velocity and
pressure up to refinement level 6, using FGMRES with order-2 Chebyshev-accelerated
multigrid (2 V-cycles).  The assess coefficients are hard-coded to avoid a
runtime Python/SciPy dependency, and Y₂₂ is evaluated in closed form.

## Why fs/fs needs a null-space penalty

With FREESLIP at both boundaries, the velocity block of the Stokes operator has
a 3-dimensional null space — rigid-body rotations \f$\hat{\mathbf{e}}_i \times \mathbf{r}\f$.
The operator is singular, which makes the multigrid PCG coarse solver and the
Chebyshev smoothers diverge.  We regularise by adding
\f[
    \varepsilon \sum_{i=1}^{3} \big(\mathbf{n}_i^\top \mathbf{x}\big)\,\mathbf{n}_i
\f]
to the operator apply, where \f$\{\mathbf{n}_i\}\f$ are the three orthonormalised
rotation modes after free-slip enforcement.  The penalty is wired directly into
the operator (no wrapper class, no type changes) and turns on only when both
boundaries are FREESLIP — zero overhead otherwise.

The penalty strength \f$\varepsilon = 1\f$ does not pollute the solution: the
target physical solution carries no angular momentum, so its projection onto the
null space is identically zero and the penalty term vanishes on it.  Per-apply
cost is three extra `MPI_Allreduce` of one double each — negligible against the
kernel cost.

## Convergence results

**zs/zs (Dirichlet / Dirichlet)** — \ref test_epsilon_divdiv_stokes_assess.cpp:

| Level | vel error | pre error | vel ratio | pre ratio |
|-------|-----------|-----------|-----------|-----------|
| 3 | 2.91e-5 | 6.11e-4 | — | — |
| 4 | 7.96e-6 | 1.53e-4 | 3.66 | 3.99 |
| 5 | 2.09e-6 | 4.07e-5 | 3.81 | 3.77 |
| 6 | 5.35e-7 | 1.10e-5 | 3.90 | 3.69 |

**fs/zs (free-slip at CMB / Dirichlet at surface)** — \ref test_epsilon_divdiv_stokes_assess_freeslip.cpp:

| Level | vel error | pre error | vel ratio | pre ratio |
|-------|-----------|-----------|-----------|-----------|
| 3 | 3.27e-5 | 5.04e-4 | — | — |
| 4 | 8.86e-6 | 1.23e-4 | 3.69 | 4.10 |
| 5 | 2.32e-6 | 3.27e-5 | 3.82 | 3.76 |
| 6 | 5.95e-7 | 9.08e-6 | 3.90 | 3.60 |

**fs/fs (free-slip at CMB / free-slip at surface)** — \ref test_epsilon_divdiv_stokes_assess_freeslip_freeslip.cpp:

| Level | vel error | pre error | vel ratio | pre ratio |
|-------|-----------|-----------|-----------|-----------|
| 3 | 3.18e-5 | 5.37e-4 | — | — |
| 4 | 8.10e-6 | 1.30e-4 | 3.93 | 4.15 |
| 5 | 2.03e-6 | 3.20e-5 | 3.98 | 4.04 |
| 6 | 5.04e-7 | 8.31e-6 | 4.03 | 3.86 |
