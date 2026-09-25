# Verification benchmarks

Produces Fig. 4 of the preprint, the steady-state radial temperature profiles
and isosurfaces, and Table 2, the Nusselt numbers.

Ten cases of the Zhong et al. (2008) spherical-shell suite. MMOC at Courant 2.7,
MT256, free slip at both boundaries, isothermal Dirichlet temperatures.

| config | Ra | viscosity contrast | perturbation |
|---|---|---|---|
| `config_A1_mmoc.toml`     | 7e3 | 1    | Y_3^2 |
| `config_A3_mmoc.toml`     | 7e3 | 20   | Y_3^2 |
| `config_A4_mmoc.toml`     | 7e3 | 100  | Y_3^2 |
| `config_A5_mmoc.toml`     | 7e3 | 1000 | Y_3^2 |
| `config_A6_mmoc.toml`     | 7e3 | 1e4  | Y_3^2 |
| `config_A7_mmoc.toml`     | 7e3 | 1e5  | Y_3^2 |
| `config_C1_mmoc.toml`     | 1e5 | 1    | Y_4^0 + 5/7 Y_4^4 |
| `config_C3_mmoc.toml`     | 1e5 | 30   | Y_4^0 + 5/7 Y_4^4 |
| `config_C4_mmoc.toml`     | 1e5 | 100  | Y_4^0 + 5/7 Y_4^4 |
| `config_C1star_mmoc.toml` | 1e7 | 1    | Y_4^0 + 5/7 Y_4^4 |

`viscosity-rmu` is the top-to-bottom contrast, since `eta(T) = rmu^(1/2 - T)`.
Ra is set through `reference-viscosity`: 1.721988e24 at Ra = 1e5, scaling
inversely with Ra.

## Running

Mesh and subdomain levels go on the command line. MT256 is mesh level 8.

```
srun mantlecirculation --config config_C3_mmoc.toml --extended-parameters \
  --refinement-level-mesh-min 3 --refinement-level-mesh-max 8 \
  --lat-sdr 3 --rad-sdr 0 --radial-extra-levels -1 \
  --outdir <outdir> --outdir-overwrite
```

No step limit is set: run until the radial temperature profile and the surface
Nusselt number stop changing. Output is `nu.csv` (timestep, time, Nu_top, V_rms)
and `radial_profiles/`.
