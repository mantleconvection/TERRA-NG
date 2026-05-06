# Anisotropic refinement {#anisotropic-refinement}

By default the spherical-shell mesh is refined isotropically: at multigrid (MG)
level \f$L\f$, each diamond holds \f$2^L\f$ cells in *every* axis (lateral
\f$x\f$, lateral \f$y\f$, radial \f$r\f$). 

The mantle-circulation app supports decoupling for both the
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
`--refinement-level-subdomains`.

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

## Non-uniform radial-shell placement

`--radial-extra-levels` controls **how many** radial cells exist; the second
group of knobs controls **where** they are placed.  By default the shells are
equispaced in \f$[r_{\min}, r_{\max}]\f$.  A tanh map can be applied to
concentrate them near one or both radial boundaries.

| Flag                     | Default     | Effect                                                                                                                                                                                                                              |
|--------------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--radial-distribution`  | `uniform`   | One of `uniform`, `tanh-both`, `tanh-cmb`, `tanh-surface`.  `uniform` is the equispaced default; the three `tanh-*` variants apply the corresponding tanh map from \ref terra::grid::shell::make_tanh_boundary_cluster (and friends). |
| `--radial-cluster-k`     | `1.0`       | Cluster strength \f$k\f$ for the tanh-based variants.  \f$k \le 0\f$ collapses each variant back to uniform.  \f$k \approx 1\f$ gives mild clustering, \f$k \approx 2\f$ strong clustering.                                          |

The map \f$f : [0,1] \rightarrow [0,1]\f$ corresponding to each variant is:

| Variant         | Map \f$f(s)\f$                                                            | Clusters near                  |
|-----------------|---------------------------------------------------------------------------|--------------------------------|
| `uniform`       | \f$f(s) = s\f$                                                            | nowhere — equispaced.         |
| `tanh-both`     | \f$f(s) = \tfrac{1}{2}\!\left(\tfrac{\tanh(k(2s-1))}{\tanh(k)}+1\right)\f$ | both \f$r_{\min}\f$ and \f$r_{\max}\f$ |
| `tanh-cmb`      | \f$f(s) = 1 - \tfrac{\tanh(k(1-s))}{\tanh(k)}\f$                          | \f$r_{\min}\f$ (CMB)           |
| `tanh-surface`  | \f$f(s) = \tfrac{\tanh(k\,s)}{\tanh(k)}\f$                                | \f$r_{\max}\f$ (surface)       |

The shells are then placed at \f$r_i = r_{\min} + (r_{\max} - r_{\min})\,f(s_i)\f$
with \f$s_i = i / (N - 1)\f$.  The same map is applied at every multigrid level
(consistently with \ref anisotropic-refinement), so the V-cycle stays
well-defined.

This feature is orthogonal to `--radial-extra-levels`: clustering changes the
**positions** of the existing radial cells without changing their count.

```toml
# Concentrate radial layers near both boundaries (BL focus on CMB and surface).
radial-distribution=tanh-both
radial-cluster-k=1.5
```

```toml
# Strong CMB clustering only (relevant when only the CMB BL needs resolving).
radial-distribution=tanh-cmb
radial-cluster-k=2.0
```
