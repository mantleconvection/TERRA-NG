#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Kokkos_Macros.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"

namespace terra::io {

/// @brief Describes the 2D grid layout for a linearized data column.
///
/// A single column in a data file contains @c nx * ny scalar values (one value
/// per data row), representing a 2D grid where
///   - the x-axis has @c nx equally-spaced points: x_min, x_min+dx, ..., x_min+(nx-1)*dx
///   - the y-axis has @c ny equally-spaced points: y_min, y_min+dy, ..., y_min+(ny-1)*dy
///
/// The flat file index @c k (0-based, counting only non-comment data rows) maps
/// to the grid index pair (ix, iy) via
/// @code
///   k = ix * stride_x + iy * stride_y
/// @endcode
///
/// Exactly one of @c stride_x / @c stride_y must be 1 (the "fast" / innermost
/// dimension); the other equals the size of that fast dimension.
///
/// Two common configurations (and their correct stride settings):
///
/// ---
///
/// @par Example A — x varies slowest (C row-major): stride_x = ny, stride_y = 1
///
/// Suppose the grid maps pressure (x, 2 points) × temperature (y, 3 points)
/// to density.  The data author loops over pressure in the outer loop:
///
/// @code
///   # pressure  temperature  density
///   #   x=0.0      y=200        row k=0  → (ix=0, iy=0)
///     0.0        200          1.10
///   #   x=0.0      y=300        row k=1  → (ix=0, iy=1)
///     0.0        300          1.05
///   #   x=0.0      y=400        row k=2  → (ix=0, iy=2)
///     0.0        400          0.98
///   #   x=1.0      y=200        row k=3  → (ix=1, iy=0)
///     1.0        200          1.08
///   #   x=1.0      y=300        row k=4  → (ix=1, iy=1)
///     1.0        300          1.02
///   #   x=1.0      y=400        row k=5  → (ix=1, iy=2)
///     1.0        400          0.95
/// @endcode
///
/// For the density column (index 2): nx=2, ny=3.
/// k = ix*3 + iy*1  →  stride_x=3 (=ny), stride_y=1
///
/// @code
///   GridLayout2D layout{ .nx=2, .ny=3,
///                        .x_min=0.0, .y_min=200.0,
///                        .dx=1.0,    .dy=100.0,
///                        .stride_x=3,  // ny — one full y-row per x step
///                        .stride_y=1 };
///   auto density = read_lookup_table_2d( "file.dat", 2, layout );
///   // density( 0.0, 200.0 ) == 1.10   (ix=0, iy=0)
///   // density( 1.0, 400.0 ) == 0.95   (ix=1, iy=2)
/// @endcode
///
/// ---
///
/// @par Example B — y varies slowest (column-major): stride_x = 1, stride_y = nx
///
/// Same grid, but the data author loops over temperature in the outer loop:
///
/// @code
///   # pressure  temperature  density
///   #   x=0.0      y=200        row k=0  → (ix=0, iy=0)
///     0.0        200          1.10
///   #   x=1.0      y=200        row k=1  → (ix=1, iy=0)
///     1.0        200          1.08
///   #   x=0.0      y=300        row k=2  → (ix=0, iy=1)
///     0.0        300          1.05
///   #   x=1.0      y=300        row k=3  → (ix=1, iy=1)
///     1.0        300          1.02
///   #   x=0.0      y=400        row k=4  → (ix=0, iy=2)
///     0.0        400          0.98
///   #   x=1.0      y=400        row k=5  → (ix=1, iy=2)
///     1.0        400          0.95
/// @endcode
///
/// For the density column (index 2): nx=2, ny=3.
/// k = ix*1 + iy*2  →  stride_x=1, stride_y=2 (=nx)
///
/// @code
///   GridLayout2D layout{ .nx=2, .ny=3,
///                        .x_min=0.0, .y_min=200.0,
///                        .dx=1.0,    .dy=100.0,
///                        .stride_x=1,  // 1 — consecutive rows differ by one x step
///                        .stride_y=2 };// nx — one full x-row per y step
///   auto density = read_lookup_table_2d( "file.dat", 2, layout );
///   // density( 0.0, 200.0 ) == 1.10   (ix=0, iy=0)
///   // density( 1.0, 400.0 ) == 0.95   (ix=1, iy=2)
/// @endcode
///
/// Both layouts produce the same ScalarLookupTable2D and the same interpolated
/// values — only the file row order differs.
struct GridLayout2D
{
    int    nx;       ///< Number of grid points along x
    int    ny;       ///< Number of grid points along y
    double x_min;    ///< x coordinate of the first (ix=0) grid point
    double y_min;    ///< y coordinate of the first (iy=0) grid point
    double dx;       ///< Grid spacing along x  (must be > 0)
    double dy;       ///< Grid spacing along y  (must be > 0)
    int    stride_x; ///< Flat-index step when ix increases by 1  (see above)
    int    stride_y; ///< Flat-index step when iy increases by 1  (see above)
};

/// @brief Device-capable 2D scalar lookup table with bilinear interpolation.
///
/// Holds a Kokkos::View with layout @c data(ix, iy) and scalar metadata that
/// is trivially copyable to the device.  All members are either arithmetic or
/// Kokkos::View (which is itself device-copyable via a reference-counted
/// handle), so the struct can be captured by value in a @c KOKKOS_LAMBDA.
///
/// Queries outside the table domain are @b clamped to the nearest boundary
/// value; no extrapolation is performed.
///
/// @tparam ScalarType Floating-point type of the table values (typically double)
template < typename ScalarType >
struct ScalarLookupTable2D
{
    using ViewType = Kokkos::View< ScalarType**, Kokkos::LayoutRight >;

    ViewType   data;  ///< 2D device view, indexed as data(ix, iy)
    ScalarType x_min; ///< x coordinate of grid point ix=0
    ScalarType y_min; ///< y coordinate of grid point iy=0
    ScalarType dx;    ///< Spacing between grid points along x
    ScalarType dy;    ///< Spacing between grid points along y
    int        nx;    ///< Number of grid points along x
    int        ny;    ///< Number of grid points along y

    /// @brief Bilinearly interpolated value at physical coordinates (x, y).
    ///
    /// Queries outside [x_min, x_min+(nx-1)*dx] × [y_min, y_min+(ny-1)*dy]
    /// are clamped to the table boundary before interpolation.
    ///
    /// @param x Physical x coordinate
    /// @param y Physical y coordinate
    /// @return Interpolated (or clamped boundary) scalar value
    KOKKOS_INLINE_FUNCTION
    ScalarType operator()( ScalarType x, ScalarType y ) const
    {
        // --- convert to fractional grid coordinates and clamp ---
        ScalarType fx = ( x - x_min ) / dx;
        ScalarType fy = ( y - y_min ) / dy;

        const ScalarType fx_max = static_cast< ScalarType >( nx - 1 );
        const ScalarType fy_max = static_cast< ScalarType >( ny - 1 );

        if ( fx < ScalarType( 0 ) )
        {
            fx = ScalarType( 0 );
        }

        if ( fx > fx_max )
        {
            fx = fx_max;
        }

        if ( fy < ScalarType( 0 ) )
        {
            fy = ScalarType( 0 );
        }

        if ( fy > fy_max )
        {
            fy = fy_max;
        }

        // --- handle degenerate single-point dimensions ---
        if ( nx == 1 && ny == 1 )
        {
            return data( 0, 0 );
        }

        if ( nx == 1 )
        {
            int iy = static_cast< int >( fy );
            if ( iy > ny - 2 )
            {
                iy = ny - 2;
            }

            ScalarType ty = fy - static_cast< ScalarType >( iy );
            return ( ScalarType( 1 ) - ty ) * data( 0, iy ) + ty * data( 0, iy + 1 );
        }

        if ( ny == 1 )
        {
            int ix = static_cast< int >( fx );
            if ( ix > nx - 2 )
            {
                ix = nx - 2;
            }
            ScalarType tx = fx - static_cast< ScalarType >( ix );
            return ( ScalarType( 1 ) - tx ) * data( ix, 0 ) + tx * data( ix + 1, 0 );
        }

        // --- full bilinear interpolation ---
        int ix = static_cast< int >( fx );
        int iy = static_cast< int >( fy );

        // clamp cell corner so ix+1 and iy+1 are always valid
        if ( ix > nx - 2 )
        {
            ix = nx - 2;
        }

        if ( iy > ny - 2 )
        {
            iy = ny - 2;
        }

        const ScalarType tx = fx - static_cast< ScalarType >( ix );
        const ScalarType ty = fy - static_cast< ScalarType >( iy );

        return ( ScalarType( 1 ) - tx ) * ( ScalarType( 1 ) - ty ) * data( ix, iy ) +
               tx * ( ScalarType( 1 ) - ty ) * data( ix + 1, iy ) + ( ScalarType( 1 ) - tx ) * ty * data( ix, iy + 1 ) +
               tx * ty * data( ix + 1, iy + 1 );
    }
};

namespace detail {

/// @brief Split a line on any combination of spaces, tabs, and commas.
///
/// Consecutive delimiters are collapsed (empty tokens are ignored), so both
/// "1.0, 2.0, 3.0" and "1.0  2.0  3.0" (and "1.0 ,2.0, 3.0") produce the
/// same three-token result.
inline std::vector< std::string > split_flexible( const std::string& line )
{
    std::vector< std::string > tokens;
    std::string                tok;

    for ( const char c : line )
    {
        if ( c == ' ' || c == '\t' || c == ',' )
        {
            if ( !tok.empty() )
            {
                tokens.push_back( tok );
                tok.clear();
            }
        }
        else
        {
            tok += c;
        }
    }

    if ( !tok.empty() )
        tokens.push_back( tok );

    return tokens;
}

/// @brief Given a flat index k, return (ix, iy) according to the layout strides.
///
/// Requires that exactly one of layout.stride_x / layout.stride_y equals 1.
inline void flat_to_grid( int k, const GridLayout2D& layout, int& ix, int& iy )
{
    if ( layout.stride_y == 1 )
    {
        // x is the slow (outer) dimension: k = ix * stride_x + iy
        ix = k / layout.stride_x;
        iy = k % layout.stride_x;
    }
    else
    {
        // y is the slow (outer) dimension: k = ix + iy * stride_y
        iy = k / layout.stride_y;
        ix = k % layout.stride_y;
    }
}

} // namespace detail

/// @brief Read selected columns from a delimited data file into 2D lookup tables.
///
/// @par File format
/// - Lines starting with @c # (after optional leading whitespace) are treated
///   as comments and ignored.
/// - Empty or whitespace-only lines are also skipped.
/// - All other lines are data rows.  Values may be separated by spaces, tabs,
///   commas, or any mixture thereof.
/// - The file must contain at least @c layout.nx * layout.ny data rows;
///   additional rows after that are silently ignored.
/// - Columns are 0-indexed.  Each requested column index must be present on
///   every data row.
///
/// @par Flat-to-grid mapping
/// Data row @c k (0-based, among non-comment rows) maps to grid index (ix, iy)
/// via the strides stored in @p layout (see @ref GridLayout2D for details).
/// The resulting Kokkos view is indexed as @c view(ix, iy).
///
/// @param filename        Path to the data file
/// @param column_indices  0-based column indices to extract (order is preserved)
/// @param layout          Grid dimensions, physical coordinates, and strides
/// @param label           Optional label prefix for the Kokkos views
///
/// @return One @ref ScalarLookupTable2D per requested column, in the same order
///         as @p column_indices.
///
/// @throws std::runtime_error if the file cannot be opened, if a data row has
///         fewer columns than the largest requested index, or if fewer than
///         @c nx * ny data rows are found.
template < typename ScalarType = double >
std::vector< ScalarLookupTable2D< ScalarType > > read_lookup_tables_2d(
    const std::string&        filename,
    const std::vector< int >& column_indices,
    const GridLayout2D&       layout,
    const std::string&        label = "lookup_table" )
{
    if ( layout.stride_x != 1 && layout.stride_y != 1 )
    {
        throw std::runtime_error(
            "terra::io::read_lookup_tables_2d: exactly one of stride_x / stride_y must be 1. "
            "See GridLayout2D documentation for the two supported configurations." );
    }

    const int total_rows = layout.nx * layout.ny;
    const int num_cols   = static_cast< int >( column_indices.size() );

    if ( num_cols == 0 )
        return {};

    // max column index we will need to access
    int max_col_idx = 0;
    for ( int c : column_indices )
    {
        if ( c < 0 )
            throw std::runtime_error( "terra::io::read_lookup_tables_2d: negative column index." );
        if ( c > max_col_idx )
            max_col_idx = c;
    }

    // --- open file ---
    std::ifstream file( filename );
    if ( !file.is_open() )
        throw std::runtime_error( "terra::io::read_lookup_tables_2d: cannot open file: " + filename );

    // --- accumulate flat values on host (one vector per requested column) ---
    std::vector< std::vector< ScalarType > > flat( num_cols );
    for ( auto& v : flat )
        v.reserve( static_cast< size_t >( total_rows ) );

    int         data_row = 0;
    std::string line;
    int         line_number = 0;

    while ( data_row < total_rows && std::getline( file, line ) )
    {
        ++line_number;

        // skip comments and blank lines
        {
            const auto first_nonspace = line.find_first_not_of( " \t\r" );
            if ( first_nonspace == std::string::npos )
                continue; // blank line
            if ( line[first_nonspace] == '#' )
                continue; // comment
        }

        const auto tokens = detail::split_flexible( line );

        if ( static_cast< int >( tokens.size() ) <= max_col_idx )
        {
            throw std::runtime_error(
                "terra::io::read_lookup_tables_2d: file '" + filename + "', line " + std::to_string( line_number ) +
                ": expected at least " + std::to_string( max_col_idx + 1 ) + " columns, found " +
                std::to_string( tokens.size() ) + "." );
        }

        for ( int c = 0; c < num_cols; ++c )
        {
            try
            {
                flat[c].push_back( static_cast< ScalarType >( std::stod( tokens[column_indices[c]] ) ) );
            }
            catch ( const std::exception& e )
            {
                throw std::runtime_error(
                    "terra::io::read_lookup_tables_2d: file '" + filename + "', line " + std::to_string( line_number ) +
                    ": cannot parse '" + tokens[column_indices[c]] + "' as a number." );
            }
        }

        ++data_row;
    }

    if ( data_row < total_rows )
    {
        throw std::runtime_error(
            "terra::io::read_lookup_tables_2d: file '" + filename + "': expected " + std::to_string( total_rows ) +
            " data rows (nx=" + std::to_string( layout.nx ) + " * ny=" + std::to_string( layout.ny ) +
            "), but found only " + std::to_string( data_row ) + "." );
    }

    // --- build host views and copy to device ---
    std::vector< ScalarLookupTable2D< ScalarType > > result;
    result.reserve( static_cast< size_t >( num_cols ) );

    for ( int c = 0; c < num_cols; ++c )
    {
        const std::string view_label = label + "_col" + std::to_string( column_indices[c] );

        // allocate device view and a matching host mirror
        typename ScalarLookupTable2D< ScalarType >::ViewType device_view( view_label, layout.nx, layout.ny );

        auto host_view = Kokkos::create_mirror_view( Kokkos::HostSpace{}, device_view );

        // scatter flat values into (ix, iy) positions using the stride mapping
        for ( int k = 0; k < total_rows; ++k )
        {
            int ix, iy;
            detail::flat_to_grid( k, layout, ix, iy );
            host_view( ix, iy ) = flat[c][static_cast< size_t >( k )];
        }

        Kokkos::deep_copy( device_view, host_view );

        ScalarLookupTable2D< ScalarType > table;
        table.data  = device_view;
        table.x_min = static_cast< ScalarType >( layout.x_min );
        table.y_min = static_cast< ScalarType >( layout.y_min );
        table.dx    = static_cast< ScalarType >( layout.dx );
        table.dy    = static_cast< ScalarType >( layout.dy );
        table.nx    = layout.nx;
        table.ny    = layout.ny;

        result.push_back( std::move( table ) );
    }

    return result;
}

/// @brief Convenience overload: read a single column from a data file.
///
/// Equivalent to calling @ref read_lookup_tables_2d with a one-element
/// column-index vector and returning the first (and only) result.
///
/// @param filename     Path to the data file
/// @param column_index 0-based index of the column to read
/// @param layout       Grid dimensions, physical coordinates, and strides
/// @param label        Optional label for the Kokkos view
///
/// @return A @ref ScalarLookupTable2D for the requested column
///
/// @throws std::runtime_error (same conditions as @ref read_lookup_tables_2d)
template < typename ScalarType = double >
ScalarLookupTable2D< ScalarType > read_lookup_table_2d(
    const std::string&  filename,
    int                 column_index,
    const GridLayout2D& layout,
    const std::string&  label = "lookup_table" )
{ return read_lookup_tables_2d< ScalarType >( filename, { column_index }, layout, label ).front(); }

} // namespace terra::io
