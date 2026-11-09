# Hourglass study operators

Variants of the EpsilonDivDiv operator used for the hourglass-mode study
(convergence of the pressure with the 1-point radial quadrature, see
`apps/mantlecirculation/terrangpaper/hourglass/`). They are not used by the
production solver.

| File | Content |
| --- | --- |
| `epsilon_divdiv_kerngen_hg.hpp` | Copy of the kerngen EpsilonDivDiv operator with the number of radial quadrature points as a template parameter and an optional hourglass stabilisation term (`STAB_C_KERNGEN`, 0 = off). The 1x2 radial rule has no hourglass modes, so the stabilisation only acts on the 1-point rule. |
| `epsilon_divdiv_simple_2pt.hpp` | Reference operator with the 2-point radial rule, straightforward per-cell assembly. |
| `epsilon_divdiv_simple_stab.hpp` | Reference operator with the 1-point rule plus the hourglass stabilisation, straightforward per-cell assembly. |

The only user is `tests/test_epsilon_divdiv_quadrature_matrix.cpp`, which checks
the kerngen variant against the two reference operators and against the stored
local matrices in `tests/matrix_rhs_s{1,2,3}.inc`.
