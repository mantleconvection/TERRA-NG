# Anisotropic refinement {#anisotropic-refinement}

By default the spherical-shell mesh is refined isotropically: at multigrid (MG)
level \f$L\f$, each diamond holds \f$2^L\f$ cells in *every* axis (lateral
\f$x\f$, lateral \f$y\f$, radial \f$r\f$). This is fine for benchmarks where
all directions resolve features of comparable wavelength, but mantle-convection
runs often have very thin radial boundary layers next to wavelengths in the
lateral plane that are an order of magnitude larger. Spending equal DOFs in
both directions wastes memory and compute.

The mantle-circulation app supports **per-axis decoupling** for both the
diamond refinement level and the subdomain refinement level via three CLI flags
declared in \ref terra::mantlecirculation::MeshParameters:

| Flag                     | Default | Effect                                                                                                                                                                                       |
|--------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--radial-extra-levels`  | `0`     | Additive offset for the **radial** diamond refinement level relative to the lateral one. At every MG level \f$L\f$ the radial level becomes \f$L + \mathtt{radial\_extra\_levels}\f$.        |
| `--lat-sdr`              | `-1`    | Lateral subdomain refinement level. `-1` falls back to `--refinement-level-subdomains`.                                                                                                      |
| `--rad-sdr`              | `-1`    | Radial subdomain refinement level. `-1` falls back to `--refinement-level-subdomains`.                                                                                                       |

## Why apply the offset at every MG level

`radial_extra_levels` is a *constant* offset applied uniformly across the whole
multigrid hierarchy: at MG level \f$L\f$, the lateral level is \f$L\f$ and the
radial level is \f$L + \mathtt{extra}\f$. Both axes still halve per coarsening
step; the radial-vs-lateral aspect-ratio is preserved at every level. This is
what keeps the V-cycle consistent — the smoother on every level sees the same
element aspect ratio that the level above does.

## Cell counts per direction per diamond

For `mesh-min=2`, `mesh-max=6` (the typical \f$L=6\f$ run):

| `radial_extra_levels` | Coarse cells (level 2)      | Fine cells (level 6)        | Notes                              |
|-----------------------|-----------------------------|-----------------------------|------------------------------------|
| `+2`                  | 4 lat × 16 rad              | 64 lat × 256 rad            | radial-fine; BL focus              |
| `0`                   | 4 lat × 4 rad               | 64 lat × 64 rad             | isotropic (default)                |
| `-1`                  | 4 lat × 2 rad               | 64 lat × 32 rad             | radial-coarse                      |
| `-2`                  | 4 lat × 1 rad               | 64 lat × 16 rad             | minimum (requires `--rad-sdr=0`)   |

Negative offsets are valid when the dynamics is laterally rich but radially
smooth (e.g. quasi-2D, or runs that don't resolve a thin BL).

## Validation

The app refuses configurations that would break the MG hierarchy or cause
undefined behaviour. Three constraints, checked in
`apps/mantlecirculation/src/parameters.hpp` right after CLI parse:

1. **`mesh_min + radial_extra_levels >= 0`** — radial mesh refinement level
   must stay non-negative at the coarsest MG level (otherwise
   \f$2^{\mathrm{rad\_level}}\f$ is undefined).
2. **`mesh_min >= lat_sdr_eff`** — each lateral subdomain must hold at least
   one cell at the coarsest MG level.
3. **`mesh_min + radial_extra_levels >= rad_sdr_eff`** — same for radial
   subdomains.

`lat_sdr_eff` and `rad_sdr_eff` resolve the `-1`-means-fallback semantics: if
the per-axis override is negative, the value falls back to
`--refinement-level-subdomains`. Failing any check exits before mesh setup with
a descriptive message; the third hint also suggests the obvious remedy
("Consider lowering `--rad-sdr` or raising `--radial-extra-levels`").

## Example

```toml
# Radial-finer-than-lateral run (BL focus): 4× more radial cells than lateral.
refinement-level-mesh-min=2
refinement-level-mesh-max=6
refinement-level-subdomains=1
radial-extra-levels=2
# 64 × 64 × 256 cells per diamond at the fine level.
```

```toml
# Lateral-finer-than-radial run (smooth radial profile): 2× fewer radial cells.
refinement-level-mesh-min=2
refinement-level-mesh-max=6
refinement-level-subdomains=1
rad-sdr=0
radial-extra-levels=-1
# 64 × 64 × 32 cells per diamond at the fine level.
```
