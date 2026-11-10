# Stokes convergence on published viscosity profiles

Produces the two tables of Fig. 3 of the preprint: preconditioned
FGMRES iterations to a relative residual of 1e-6, and the relative residual
reached after a fixed budget of 10 iterations.

The driver is `tests/test_epsilon_divdiv_ablock_mg_gca.cpp`, built as
`test_epsilon_divdiv_ablock_mg_gca`. It solves the Stokes saddle point on the
shell with the same preconditioner as the app: a matrix-free geometric
multigrid V-cycle on the velocity block, three pre- and three post-smoothing
steps of a degree-2 Chebyshev smoother, and the inverse lumped diagonal of the
1/eta-weighted pressure mass matrix as the Schur approximation. No-slip at both
boundaries, coarsest level 2.

| profile | flag | viscosity range |
|---|---|---|
| Lin et al. (2022)   | `--visc-profile lin`   | 4.7e20 - 5e23 Pa s |
| Stotz et al. (2017) | `--visc-profile stotz` | 5.8e19 - 7e23 Pa s |

Both are read from `data/radialprofiles/ViscosityProfile_*.csv` in this
repository, columns `radius_normalized_1p22_2p22` and
`viscosity_scaled_by_min`. The script finds them when submitted from this
directory. From anywhere else set `TERRANG_PROFILE_DIR` to
`<repo>/data/radialprofiles`.

## Running

`run_convergence.sh` takes `TERRANG_BIN` (required), `TERRANG_MT`
(32 ... 2048) and `TERRANG_PROFILE` (`lin` or `stotz`), plus the optional
`TERRANG_MAX_CYCLES`, `TERRANG_OUT` and `TERRANG_PROFILE_DIR`.
`TERRANG_MT` selects the fine level (MT32 is level 5,
MT2048 is level 11). The subdomain refinement follows from the rank count, so
node and task counts go on the `sbatch` line:

```
TERRANG_BIN=<build>/tests/test_epsilon_divdiv_ablock_mg_gca \
TERRANG_MT=256 TERRANG_PROFILE=lin \
  sbatch --account=<project> --nodes=1 --ntasks-per-node=8 run_convergence.sh
```

`TERRANG_MAX_CYCLES=100` (the default) gives the left table, the iteration
count in the `cycles` column of the final summary. `TERRANG_MAX_CYCLES=10`
gives the right table, `final_rel_res` in the same summary.
`collect_convergence.py <dir>` gathers both from the job outputs.

| MT | level | ranks | sbatch |
|---|---|---|---|
| MT32   | 5  | 1   | `--nodes=1 --ntasks-per-node=1` |
| MT64   | 6  | 1   | `--nodes=1 --ntasks-per-node=1` |
| MT128  | 7  | 10  | `--nodes=2 --ntasks-per-node=5` |
| MT256  | 8  | 10  | `--nodes=2 --ntasks-per-node=5` |
| MT512  | 9  | 80  | `--nodes=10 --ntasks-per-node=8` |
| MT1024 | 10 | 640 | `--nodes=80 --ntasks-per-node=8` |
| MT2048 | 11 | not rerun | |

The rank counts are memory-driven: with the 100-iteration budget the outer
FGMRES holds 104 block vectors of the velocity-pressure system (restart 50),
270 GB each at MT1024, 2 TB each at MT2048, which would need more than 5000
devices.
