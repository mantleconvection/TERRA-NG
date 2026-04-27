
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/lookup_table_2d_reader.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/cli11_helper.hpp"
#include "util/cli11_wrapper.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using terra::util::add_flag_with_default;
using terra::util::add_option_with_default;
using terra::util::logroot;

// ─────────────────────────────────────────────────────────────────────────────
// Source-function modes
// ─────────────────────────────────────────────────────────────────────────────

/// @brief How to derive (pressure, temperature) from a node's 3D position.
///
/// cartesian — pressure    = linear in Cartesian x  (range: [-r_max, +r_max])
///             temperature = linear in Cartesian y  (range: [-r_max, +r_max])
///
/// radial    — pressure    = linear in radius r     (range: [r_min, r_max])
///             temperature = linear in colatitude θ (range: [0, π])
///             where θ = arccos(z / r) is the angle from the north pole.
enum class SourceMode
{
    cartesian,
    radial
};

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────

struct Parameters
{
    // ── table file ──────────────────────────────────────────────────────────
    std::string table_file;

    // ── column selection ────────────────────────────────────────────────────
    /// Zero-based column indices to extract from the table file.
    std::vector< int > columns = { 0 };

    /// Display labels for the XDMF output fields (one per column).
    /// Defaults to "col_N" if fewer labels than columns are given.
    std::vector< std::string > column_labels = {};

    // ── lookup table grid layout ─────────────────────────────────────────────
    /// Number of grid points along the pressure (x) axis.
    int nx = 10;
    /// Number of grid points along the temperature (y) axis.
    int ny = 10;
    /// Pressure value at the first grid point (ix = 0).
    double x_min = 0.0;
    /// Temperature value at the first grid point (iy = 0).
    double y_min = 0.0;
    /// Spacing between pressure grid points.
    double dx = 1.0;
    /// Spacing between temperature grid points.
    double dy = 1.0;

    /// How the column data is linearized in the file.
    ///
    ///   "x-outer" — outer loop over pressure (x), inner loop over temperature (y):
    ///               row k → (ix = k/ny, iy = k%ny)   [stride_x=ny, stride_y=1]
    ///
    ///   "y-outer" — outer loop over temperature (y), inner loop over pressure (x):
    ///               row k → (iy = k/nx, ix = k%nx)   [stride_x=1, stride_y=nx]
    std::string stride_mode = "x-outer";

    // ── source function ──────────────────────────────────────────────────────
    /// Which mapping from 3D node position to (pressure, temperature) to use.
    /// "cartesian" or "radial" (see SourceMode).
    std::string source_mode = "cartesian";

    /// Pressure assigned to the low end of the spatial range
    ///   (cartesian: Cartesian x = -r_max  |  radial: r = r_min).
    /// NaN = use the table's own pressure minimum (x_min).
    double pressure_min = std::numeric_limits< double >::quiet_NaN();
    /// Pressure assigned to the high end of the spatial range
    ///   (cartesian: Cartesian x = +r_max  |  radial: r = r_max).
    /// NaN = use the table's own pressure maximum (x_min + (nx-1)*dx).
    double pressure_max = std::numeric_limits< double >::quiet_NaN();

    /// Temperature assigned to the low end of the spatial range
    ///   (cartesian: Cartesian y = -r_max  |  radial: colatitude θ = 0, north pole).
    /// NaN = use the table's own temperature minimum (y_min).
    double temperature_min = std::numeric_limits< double >::quiet_NaN();
    /// Temperature assigned to the high end of the spatial range
    ///   (cartesian: Cartesian y = +r_max  |  radial: colatitude θ = π, south pole).
    /// NaN = use the table's own temperature maximum (y_min + (ny-1)*dy).
    double temperature_max = std::numeric_limits< double >::quiet_NaN();

    // ── mesh ─────────────────────────────────────────────────────────────────
    double r_min                    = 0.5;
    double r_max                    = 1.0;
    int    lateral_refinement_level = 3;
    int    radial_refinement_level  = 3;

    // ── output ───────────────────────────────────────────────────────────────
    std::string output_directory = "visualize_lookup_table_2d_output";

    /// If true, also write the pressure and temperature source fields to XDMF
    /// (useful for verifying that the source function looks as expected).
    bool write_source_fields = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    const auto description =
        "Reads a 2D scalar lookup table from a columnar data file, evaluates it\n"
        "on a spherical shell mesh by mapping node positions to (pressure, temperature)\n"
        "via a chosen source function, and writes the result as XDMF.";

    CLI::App app{ description };

    Parameters p{};

    // ── table file ──────────────────────────────────────────────────────────
    app.add_option( "-f,--table-file", p.table_file, "Path to the columnar data file." )->required();

    // ── column selection ────────────────────────────────────────────────────
    add_option_with_default(
        app,
        "--columns",
        p.columns,
        "Zero-based column indices to read from the file (space-separated, e.g. --columns 2 3)." );

    add_option_with_default(
        app,
        "--column-labels",
        p.column_labels,
        "Labels for the XDMF output fields, one per column.\n"
        "Defaults to 'col_N' for each column index N." );

    // ── lookup table grid layout ─────────────────────────────────────────────
    add_option_with_default( app, "--nx", p.nx, "Number of grid points along the pressure (x) axis." );
    add_option_with_default( app, "--ny", p.ny, "Number of grid points along the temperature (y) axis." );
    add_option_with_default( app, "--x-min", p.x_min, "Pressure value at the first grid point (ix=0)." );
    add_option_with_default( app, "--y-min", p.y_min, "Temperature value at the first grid point (iy=0)." );
    add_option_with_default( app, "--dx", p.dx, "Spacing between pressure grid points." );
    add_option_with_default( app, "--dy", p.dy, "Spacing between temperature grid points." );

    add_option_with_default(
        app,
        "--stride-mode",
        p.stride_mode,
        "How each data column is linearized in the file:\n"
        "  'x-outer' — outer loop over pressure (x), inner over temperature (y):\n"
        "               file row k maps to (ix = k/ny, iy = k%ny)\n"
        "  'y-outer' — outer loop over temperature (y), inner over pressure (x):\n"
        "               file row k maps to (iy = k/nx, ix = k%nx)" )
        ->check( CLI::IsMember( { "x-outer", "y-outer" } ) );

    // ── source function ──────────────────────────────────────────────────────
    add_option_with_default(
        app,
        "--source-mode",
        p.source_mode,
        "How node position is mapped to (pressure, temperature):\n"
        "  'cartesian' — pressure linear in Cartesian x ∈ [-r_max, +r_max]\n"
        "                temperature linear in Cartesian y ∈ [-r_max, +r_max]\n"
        "  'radial'    — pressure linear in radius r ∈ [r_min, r_max]\n"
        "                temperature linear in colatitude θ ∈ [0, π]\n"
        "                (θ = arccos(z/r), 0 at north pole, π at south pole)" )
        ->check( CLI::IsMember( { "cartesian", "radial" } ) );


    // ── source function zoom window ──────────────────────────────────────────
    // These are optional. When omitted the full table axis range is used.
    app.add_option(
        "--pressure-min",
        p.pressure_min,
        "Pressure at the low spatial extreme of the source function\n"
        "  (cartesian: x = -r_max  |  radial: r = r_min).\n"
        "  Default: x_min (the table's own pressure minimum).\n"
        "  Set to a value inside the table range to zoom in." );
    app.add_option(
        "--pressure-max",
        p.pressure_max,
        "Pressure at the high spatial extreme of the source function\n"
        "  (cartesian: x = +r_max  |  radial: r = r_max).\n"
        "  Default: x_min + (nx-1)*dx (the table's own pressure maximum)." );
    app.add_option(
        "--temperature-min",
        p.temperature_min,
        "Temperature at the low spatial extreme of the source function\n"
        "  (cartesian: y = -r_max  |  radial: colatitude θ = 0, north pole).\n"
        "  Default: y_min (the table's own temperature minimum)." );
    app.add_option(
        "--temperature-max",
        p.temperature_max,
        "Temperature at the high spatial extreme of the source function\n"
        "  (cartesian: y = +r_max  |  radial: colatitude θ = π, south pole).\n"
        "  Default: y_min + (ny-1)*dy (the table's own temperature maximum)." );

    // ── mesh ─────────────────────────────────────────────────────────────────
    add_option_with_default( app, "--r-min", p.r_min, "Inner radius of the spherical shell." );
    add_option_with_default( app, "--r-max", p.r_max, "Outer radius of the spherical shell." );
    add_option_with_default(
        app, "--lateral-refinement-level", p.lateral_refinement_level, "Lateral refinement level." );
    add_option_with_default( app, "--radial-refinement-level", p.radial_refinement_level, "Radial refinement level." );

    // ── output ───────────────────────────────────────────────────────────────
    add_option_with_default( app, "--output-dir", p.output_directory, "XDMF output directory." );
    add_flag_with_default(
        app,
        "--write-source-fields",
        p.write_source_fields,
        "Also write the pressure and temperature source fields to XDMF." );

    CLI11_PARSE( app, argc, argv );

    logroot << "\n" << description << "\n\n";
    terra::util::print_cli_summary( app, logroot );
    logroot << "\n";

    // ── resolve labels ───────────────────────────────────────────────────────
    while ( static_cast< int >( p.column_labels.size() ) < static_cast< int >( p.columns.size() ) )
    {
        const int col_idx = p.columns[p.column_labels.size()];
        p.column_labels.push_back( "col_" + std::to_string( col_idx ) );
    }

    // ── resolve strides ──────────────────────────────────────────────────────
    terra::io::GridLayout2D layout{};
    layout.nx    = p.nx;
    layout.ny    = p.ny;
    layout.x_min = p.x_min;
    layout.y_min = p.y_min;
    layout.dx    = p.dx;
    layout.dy    = p.dy;

    if ( p.stride_mode == "x-outer" )
    {
        // outer loop over pressure (x), inner over temperature (y)
        layout.stride_x = p.ny; // ny steps to advance one pressure step
        layout.stride_y = 1;    // 1 step to advance one temperature step
    }
    else // "y-outer"
    {
        // outer loop over temperature (y), inner over pressure (x)
        layout.stride_x = 1;    // 1 step to advance one pressure step
        layout.stride_y = p.nx; // nx steps to advance one temperature step
    }

    // ── resolve source mode ──────────────────────────────────────────────────
    const SourceMode source_mode =
        ( p.source_mode == "radial" ) ? SourceMode::radial : SourceMode::cartesian;

    // ── read lookup tables ───────────────────────────────────────────────────
    logroot << "Reading lookup table from: " << p.table_file << "\n";
    logroot << "  Columns: ";
    for ( int c : p.columns )
        logroot << c << " ";
    logroot << "\n";
    logroot << "  Grid: nx=" << p.nx << "  ny=" << p.ny
            << "  x_min=" << p.x_min << "  dx=" << p.dx
            << "  y_min=" << p.y_min << "  dy=" << p.dy << "\n";
    logroot << "  Stride mode: " << p.stride_mode << "\n\n";

    std::vector< terra::io::ScalarLookupTable2D< double > > tables;
    try
    {
        tables = terra::io::read_lookup_tables_2d< double >(
            p.table_file, p.columns, layout, "lookup" );
    }
    catch ( const std::exception& e )
    {
        logroot << "Error reading table: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    // ── build mesh ───────────────────────────────────────────────────────────
    logroot << "Building spherical shell mesh "
            << "(lateral=" << p.lateral_refinement_level
            << ", radial=" << p.radial_refinement_level
            << ", r=[" << p.r_min << ", " << p.r_max << "])...\n";

    const auto domain = terra::grid::shell::DistributedDomain::create_uniform(
        p.lateral_refinement_level,
        p.radial_refinement_level,
        p.r_min,
        p.r_max,
        0,
        0,
        terra::grid::shell::subdomain_to_rank_iterate_diamond_subdomains );

    const auto shell_coords = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto shell_radii  = terra::grid::shell::subdomain_shell_radii< double >( domain );

    // ── precompute source-function constants ──────────────────────────────────
    // These scalars are captured by value into the KOKKOS_LAMBDA below.
    // Pressure/temperature range: use the user-supplied value if given,
    // otherwise fall back to the table's own axis extent.
    constexpr double pi   = 3.141592653589793;
    const double r_min    = p.r_min;
    const double r_max    = p.r_max;
    const double pres_min = std::isnan( p.pressure_min )    ? p.x_min                        : p.pressure_min;
    const double pres_max = std::isnan( p.pressure_max )    ? p.x_min + ( p.nx - 1 ) * p.dx : p.pressure_max;
    const double temp_min = std::isnan( p.temperature_min ) ? p.y_min                        : p.temperature_min;
    const double temp_max = std::isnan( p.temperature_max ) ? p.y_min + ( p.ny - 1 ) * p.dy : p.temperature_max;

    // ── XDMF output ──────────────────────────────────────────────────────────
    terra::io::XDMFOutput xdmf( p.output_directory, domain, shell_coords, shell_radii );

    // ── optional: write pressure and temperature source fields ────────────────
    if ( p.write_source_fields )
    {
        auto pressure_field    = terra::grid::shell::allocate_scalar_grid< double >( "pressure", domain );
        auto temperature_field = terra::grid::shell::allocate_scalar_grid< double >( "temperature", domain );

        Kokkos::parallel_for(
            "source_fields",
            terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( int sd, int x, int y, int r ) {
                const double px  = shell_coords( sd, x, y, 0 ) * shell_radii( sd, r );
                const double py  = shell_coords( sd, x, y, 1 ) * shell_radii( sd, r );
                const double pz  = shell_coords( sd, x, y, 2 ) * shell_radii( sd, r );
                const double rad = shell_radii( sd, r );

                double pres, temp;

                if ( source_mode == SourceMode::cartesian )
                {
                    // linear in Cartesian x and y; x ∈ [-r_max, +r_max]
                    pres = pres_min + ( pres_max - pres_min ) * ( px + r_max ) / ( 2.0 * r_max );
                    temp = temp_min + ( temp_max - temp_min ) * ( py + r_max ) / ( 2.0 * r_max );
                }
                else // radial
                {
                    // pressure: linear in r; temperature: linear in colatitude
                    pres            = pres_min + ( pres_max - pres_min ) * ( rad - r_min ) / ( r_max - r_min );
                    const double ct = pz / rad; // cos(colatitude)
                    // clamp to [-1,1] to guard against floating-point rounding
                    const double ct_clamped = ct < -1.0 ? -1.0 : ( ct > 1.0 ? 1.0 : ct );
                    const double theta      = Kokkos::acos( ct_clamped ); // ∈ [0, π]
                    temp                    = temp_min + ( temp_max - temp_min ) * theta / pi;
                }

                pressure_field( sd, x, y, r )    = pres;
                temperature_field( sd, x, y, r ) = temp;
            } );

        xdmf.add( pressure_field );
        xdmf.add( temperature_field );
    }

    // ── evaluate lookup tables on the mesh ────────────────────────────────────
    for ( int i = 0; i < static_cast< int >( tables.size() ); ++i )
    {
        const auto&       table = tables[i];
        const std::string label = p.column_labels[i];

        logroot << "Evaluating column " << p.columns[i] << " → field '" << label << "'...\n";

        auto field = terra::grid::shell::allocate_scalar_grid< double >( label, domain );

        Kokkos::parallel_for(
            "lookup_" + label,
            terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( int sd, int x, int y, int r ) {
                const double px  = shell_coords( sd, x, y, 0 ) * shell_radii( sd, r );
                const double py  = shell_coords( sd, x, y, 1 ) * shell_radii( sd, r );
                const double pz  = shell_coords( sd, x, y, 2 ) * shell_radii( sd, r );
                const double rad = shell_radii( sd, r );

                double pres, temp;

                if ( source_mode == SourceMode::cartesian )
                {
                    pres = pres_min + ( pres_max - pres_min ) * ( px + r_max ) / ( 2.0 * r_max );
                    temp = temp_min + ( temp_max - temp_min ) * ( py + r_max ) / ( 2.0 * r_max );
                }
                else // radial
                {
                    pres            = pres_min + ( pres_max - pres_min ) * ( rad - r_min ) / ( r_max - r_min );
                    const double ct = pz / rad;
                    const double ct_clamped = ct < -1.0 ? -1.0 : ( ct > 1.0 ? 1.0 : ct );
                    const double theta      = Kokkos::acos( ct_clamped );
                    temp                    = temp_min + ( temp_max - temp_min ) * theta / pi;
                }

                field( sd, x, y, r ) = table( pres, temp );
            } );

        xdmf.add( field );
    }

    Kokkos::fence();

    logroot << "\nWriting output to: " << p.output_directory << "\n";
    xdmf.write();

    logroot << "Bye :)\n";
    return 0;
}