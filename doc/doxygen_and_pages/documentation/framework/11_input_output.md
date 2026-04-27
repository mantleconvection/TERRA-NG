# Input and Output {#input-output}

## Visualization and checkpoints

\note Putting the string 'VTK' here in case you are looking for it via full-text search. We are using XDMF instead.

The \ref terra::io::XDMFOutput class implements a combined format for storing simulation data that can be visualized in
Paraview,
and can also be loaded back into the simulation (i.e., it serves as a checkpoint).

Refer to the \ref terra::io::XDMFOutput class documentation for more details. It is quite exhaustively documented.

## Tabular data

Have a look at the \ref terra::util::Table class for writing all sorts of tabular data. Also, just for writing to
the console this class can be useful. Consider this for writing to CSV or JSON files.

## Radial profiles (input)

Radial profiles can be read from CSV files. Have a look at the functions
\ref terra::shell::interpolate_radial_profile_into_subdomains() and
\ref terra::shell::interpolate_radial_profile_into_subdomains_from_csv()
for more details.

A small tool for reading and visualizing radial profiles is provided in `apps/tools/visualize_radial_profiles.cpp`.

## 2D scalar lookup tables (input)

Sometimes a physical quantity depends on two independent variables that are known at every mesh node — a common example
in mantle dynamics is density (or viscosity, heat capacity, …) as a function of pressure and temperature.
Such relationships are often pre-tabulated on a regular 2D grid and need to be evaluated at every node during a
simulation.

This is different from the \ref terra::util::Table class described above, which is a generic key-value store for
logging and analysis.
The lookup table described here is a compact, device-capable structure designed for fast bilinear interpolation inside
Kokkos kernels.

The relevant types and functions live in `terra/io/lookup_table_2d_reader.hpp`:

- \ref terra::io::GridLayout2D describes the grid: axis origins (`x_min`, `y_min`), spacings (`dx`, `dy`), point counts
  (`nx`, `ny`), and how the flat file data maps to 2D indices via `stride_x` / `stride_y`.
- \ref terra::io::ScalarLookupTable2D is the resulting device-capable struct.
  Its `operator()(x, y)` performs bilinear interpolation and clamps queries that fall outside the table domain to the
  nearest boundary value.
- \ref terra::io::read_lookup_table_2d() reads a single column; \ref terra::io::read_lookup_tables_2d() reads several
  columns at once (one \ref terra::io::ScalarLookupTable2D per column).

The file format is deliberately flexible: columns may be separated by spaces, tabs, commas, or any mixture thereof, and
lines starting with `#` are treated as comments.

### Example table file

An annotated example is provided at `data/lookup_tables/density_p_T_example.dat`.
It maps pressure (3 points, 0–10 GPa) and temperature (4 points, 1000–2500 K) to density, and is laid out with
pressure as the outer loop (x-outer):

```
# Columns:  pressure[GPa]   temperature[K]   density[kg/m^3]
#
# ix=0 (p=0 GPa)
  0.0    1000.0   3300.0
  0.0    1500.0   3250.0
  0.0    2000.0   3200.0
  0.0    2500.0   3150.0
# ix=1 (p=5 GPa)
  5.0    1000.0   3400.0
  ...
```

### Code snippet

```cpp
#include "terra/io/lookup_table_2d_reader.hpp"

// Describe the table grid.
// pressure (x): 3 points starting at 0 GPa, spacing 5 GPa
// temperature (y): 4 points starting at 1000 K, spacing 500 K
// x-outer layout: outer loop over pressure → stride_x = ny, stride_y = 1
terra::io::GridLayout2D layout{
    .nx       = 3,
    .ny       = 4,
    .x_min    = 0.0,    // first pressure value [GPa]
    .y_min    = 1000.0, // first temperature value [K]
    .dx       = 5.0,    // pressure spacing [GPa]
    .dy       = 500.0,  // temperature spacing [K]
    .stride_x = 4,      // = ny  (pressure is the outer/slow index)
    .stride_y = 1,      //       (temperature is the inner/fast index)
};

// Read column 2 (density) from the file — result is a device-side view
// wrapped in a struct with a bilinear interpolation operator.
auto density_table = terra::io::read_lookup_table_2d(
    "density_p_T_example.dat", /*column_index=*/2, layout, "density");

// Use inside a Kokkos kernel.  Queries outside the table domain are
// automatically clamped to the nearest boundary value.
Kokkos::parallel_for("fill_density", some_range,
    KOKKOS_LAMBDA(int i) {
        double p   = pressure_at_node(i);   // [GPa]
        double T   = temperature_at_node(i); // [K]
        density(i) = density_table(p, T);   // bilinear interpolation
    });
```

A small visualization tool is provided in `apps/tools/visualize_lookup_table_2d.cpp`.
It reads one or more columns from a table file, maps each mesh node to a (pressure, temperature) pair via a selectable
source function, evaluates the lookup table at every node, and writes the result as XDMF.
Two source functions are available: `cartesian` (pressure linear in Cartesian x, temperature linear in Cartesian y) and
`radial` (pressure linear in radius, temperature linear in colatitude).
Run with `--help` for the full list of options.

## Radial profiles (output)

Have a look at the functions \ref terra::shell::radial_profiles() and \ref terra::shell::radial_profiles_to_table() that
compute radial profiles of the shell (min/max/avg) on the device and write them to a table (\ref terra::util::Table) if
desired. This way if can easily be written to console, JSON, or CSV files.
