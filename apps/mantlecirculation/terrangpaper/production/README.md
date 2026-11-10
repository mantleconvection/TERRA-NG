# Production run

Produces Fig. 6 of the preprint: the radial-mean diagnostics, the
temperature-deviation isosurface and the equatorial cross-cut of the MT1024 run.

`config_MT1024_C5dim_Raeff1e7_mmoc.toml` is the MT1024 end-to-end
mantle-circulation model: TALA-compressible with MMOC, Frank-Kamenetskii
contrast 100 on the Lin et al. (2022) radial viscosity profile, free slip at
both boundaries, 4200 K at the core-mantle boundary and 300 K at the surface.

## Running

64 nodes, 512 ranks:

```
srun mantlecirculation --config config_MT1024_C5dim_Raeff1e7_mmoc.toml \
  --extended-parameters \
  --refinement-level-mesh-min 4 --refinement-level-mesh-max 10 \
  --radial-extra-levels -1 --lat-sdr 4 --rad-sdr 0 \
  --outdir <outdir> --outdir-overwrite
```

To continue from a checkpoint add `--load-checkpoint 1 --checkpoint-dir
<outdir>/xdmf --checkpoint-step <step>`.
