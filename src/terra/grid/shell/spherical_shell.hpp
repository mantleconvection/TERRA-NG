#pragma once

#include <cmath>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "../grid_types.hpp"
#include "../terra/kokkos/kokkos_wrapper.hpp"
#include "communication/buffer_copy_kernels.hpp"
#include "dense/vec.hpp"
#include "mpi/mpi.hpp"
#include "util/logging.hpp"

namespace terra::grid::shell {

/// @brief Computes the radial shell radii for a uniform grid.
///
/// Note that a shell is a 2D manifold in 3D space.
/// A layer is a 3D volume in 3D space - it is sandwiched by two shells (one on each side).
///
/// @param r_min Radius of the innermost shell.
/// @param r_max Radius of the outermost shell.
/// @param num_shells Number of shells.
/// @return Vector of shell radii (uniformly distributed in [r_min, r_max]).
template < std::floating_point T >
std::vector< T > uniform_shell_radii( T r_min, T r_max, int num_shells )
{
    if ( num_shells < 2 )
    {
        Kokkos::abort( "Number of shells must be at least 2." );
    }

    if ( r_min >= r_max )
    {
        Kokkos::abort( "r_min must be strictly less than r_max." );
    }

    std::vector< T > radii;
    radii.reserve( num_shells );
    const T r_step = ( r_max - r_min ) / ( num_shells - 1 );
    for ( int i = 0; i < num_shells; ++i )
    {
        radii.push_back( r_min + i * r_step );
    }

    // Set boundary exactly.
    radii[num_shells - 1] = r_max;

    return radii;
}

/// @brief Map to be used in @ref mapped_shell_radii.
///
/// Can be used to have smaller elements near the shell boundaries.
///
/// Returns a function
/// \f[
///     f(s) := \begin{cases}
///         s & k \leq 0 \\
///         \frac{1}{2} \frac{\tanh( k( 2s-1 ) )}{\tanh(k) + 1} & k > 0
///     \end{cases}
/// \f]
///
/// @param k \f$k = 0\f$ results in a uniform distribution (linear function), \f$k > 0\f$ refines at the boundary
///        (roughly: \f$k \approx 1\f$: mild clustering, \f$k \approx 2\f$: strong clustering)
template < std::floating_point T >
std::function< T( T ) > make_tanh_boundary_cluster( T k )
{
    return [k]( T s ) {
        if ( k <= 0.0 )
        {
            return s;
        }

        T x = T( 2 ) * s - T( 1 ); // map [0,1] -> [-1,1]
        return ( std::tanh( k * x ) / std::tanh( k ) + T( 1 ) ) * T( 0.5 );
    };
}

/// @brief Computes the radial shell radii for a non-uniformly distributed grid.
///
/// Note that a shell is a 2D manifold in 3D space.
/// A layer is a 3D volume in 3D space – it is sandwiched by two shells (one on each side).
///
/// The shell radii are generated from a uniform parameter (\f$N\f$ is the number of shells)
/// \f[
///     s_i = \frac{i}{N-1}, \; i = 0,\dots,N-1,
/// \f]
/// which is mapped to a redistributed parameter
/// \f[
///   t_i = f(s_i)
/// \f]
/// using a user-supplied function
/// \f[
///     f : [0,1] \rightarrow [0,1].
/// \f]
/// The physical radii are then computed as
/// \f[
///     r_i = r_{\min} + (r_{\max} - r_{\min}) \, t_i .
/// \f]
///
/// The function \f$f(s) = s\f$ therefore maps the shells uniformly.
/// In areas where the gradient of \f$f\f$ is small, shells are closer together.
/// In areas where the gradient of \f$f\f$ is large, shells are further apart.
///
/// Try @ref make_tanh_boundary_cluster which returns a parameterized function that can be passed to this function.
///
/// @tparam T Floating-point type.
///
/// @param r_min Radius of the innermost shell.
/// @param r_max Radius of the outermost shell.
/// @param num_shells Number of shells \f$ N \f$.
/// @param map Mapping function \f$ f : [0,1] \rightarrow [0,1] \f$ controlling shell distribution.
///            It should be monotone increasing and satisfy
///            \f$ f(0)=0 \f$ and \f$ f(1)=1 \f$.
///
/// @return Vector of shell radii (non-uniformly distributed in \f$[r_{\min}, r_{\max}]\f$).
template < std::floating_point T >
std::vector< T > mapped_shell_radii( T r_min, T r_max, int num_shells, const std::function< T( T ) >& map )
{
    if ( num_shells < 2 )
    {
        Kokkos::abort( "Number of shells must be at least 2." );
    }

    if ( r_min >= r_max )
    {
        Kokkos::abort( "r_min must be strictly less than r_max." );
    }

    std::vector< T > radii;
    radii.reserve( num_shells );

    const T inv = T( 1 ) / T( num_shells - 1 );

    for ( int i = 0; i < num_shells; ++i )
    {
        T s = T( i ) * inv; // uniform in [0,1]
        T t = map( s );     // redistributed in [0,1]
        radii.push_back( r_min + ( r_max - r_min ) * t );
    }

    // Enforce exact boundaries
    radii.front() = r_min;
    radii.back()  = r_max;

    return radii;
}

/// @brief Computes the min absolute distance of two entries in the passed vector of shell radii.
template < std::floating_point T >
T min_radial_h( const std::vector< T >& shell_radii )
{
    if ( shell_radii.size() < 2 )
    {
        throw std::runtime_error( " Need at least two shells to compute h. " );
    }

    T min_dist = std::numeric_limits< T >::infinity();
    for ( size_t i = 1; i < shell_radii.size(); ++i )
    {
        T d = std::abs( shell_radii[i] - shell_radii[i - 1] );
        if ( d < min_dist )
        {
            min_dist = d;
        }
    }
    return min_dist;
}

/// @brief Computes the max absolute distance of two entries in the passed vector of shell radii.
template < std::floating_point T >
T max_radial_h( const std::vector< T >& shell_radii )
{
    if ( shell_radii.size() < 2 )
    {
        throw std::runtime_error( " Need at least two shells to compute h. " );
    }

    T min_dist = 0;
    for ( size_t i = 1; i < shell_radii.size(); ++i )
    {
        T d = std::abs( shell_radii[i] - shell_radii[i - 1] );
        if ( d > min_dist )
        {
            min_dist = d;
        }
    }
    return min_dist;
}

/// Struct to hold the coordinates of the four base corners
/// and the number of intervals N = ntan - 1.
template < std::floating_point T >
struct BaseCorners
{
    using Vec3 = dense::Vec< T, 3 >;

    Vec3 p00; // Coordinates for global index (0, 0)
    Vec3 p0N; // Coordinates for global index (0, N)
    Vec3 pN0; // Coordinates for global index (N, 0)
    Vec3 pNN; // Coordinates for global index (N, N)
    int  N;   // Number of intervals = ntan - 1. Must be power of 2.

    // Constructor for convenience (optional)
    BaseCorners( Vec3 p00_ = {}, Vec3 p0N_ = {}, Vec3 pN0_ = {}, Vec3 pNN_ = {}, int N_ = 0 )
    : p00( p00_ )
    , p0N( p0N_ )
    , pN0( pN0_ )
    , pNN( pNN_ )
    , N( N_ )
    {}
};

// Memoization cache type: maps (i, j) index pair to computed coordinates
template < std::floating_point T >
using MemoizationCache = std::map< std::pair< int, int >, dense::Vec< T, 3 > >;

/// @brief Computes the coordinates for a specific node (i, j) in the final refined grid.
///       Uses recursion and memoization, sourcing base points from the BaseCorners struct.
///
/// @param i Row index (0 to corners.N).
/// @param j Column index (0 to corners.N).
/// @param corners Struct containing base corner coordinates and N = ntan - 1.
/// @param cache Cache to store/retrieve already computed nodes.
/// @return Vec3 Coordinates of the node (i, j) on the unit sphere.
///
template < std::floating_point T >
dense::Vec< T, 3 > compute_node_recursive( int i, int j, const BaseCorners< T >& corners, MemoizationCache< T >& cache )
{
    using Vec3 = dense::Vec< T, 3 >;

    // --- Get N and validate indices ---
    const int N    = corners.N;
    const int ntan = N + 1;
    if ( i < 0 || i >= ntan || j < 0 || j >= ntan )
    {
        throw std::out_of_range( "Requested node index out of range." );
    }
    if ( N <= 0 || ( N > 0 && ( N & ( N - 1 ) ) != 0 ) )
    {
        throw std::invalid_argument( "BaseCorners.N must be a positive power of 2." );
    }

    // --- 1. Check Cache ---
    auto cache_key = std::make_pair( i, j );
    auto it        = cache.find( cache_key );
    if ( it != cache.end() )
    {
        return it->second; // Already computed
    }

    // --- 2. Base Case: Use BaseCorners struct ---
    if ( i == 0 && j == 0 )
    {
        cache[cache_key] = corners.p00;
        return corners.p00;
    }
    if ( i == 0 && j == N )
    {
        cache[cache_key] = corners.p0N;
        return corners.p0N;
    }
    if ( i == N && j == 0 )
    {
        cache[cache_key] = corners.pN0;
        return corners.pN0;
    }
    if ( i == N && j == N )
    {
        cache[cache_key] = corners.pNN;
        return corners.pNN;
    }

    // --- 3. Recursive Step: Find creation level l2 and apply rules ---

    // Find the smallest half-stride l2 (power of 2, starting from 1)
    // such that (i, j) was NOT present on the grid with stride l = 2*l2.
    // A point is present if both i and j are multiples of l.
    int l2 = 1;
    int l  = 2; // l = 2*l2
    while ( l <= N )
    { // Iterate through possible creation strides l
        if ( i % l != 0 || j % l != 0 )
        {
            // Found the level 'l' where (i, j) was created.
            // l2 is l/2.
            break;
        }
        // If execution reaches here, (i, j) exists on grid with stride l.
        // Check the next finer level.
        l2 *= 2; // or l2 <<= 1;
        l = 2 * l2;
    }

    if ( l > N && l2 == N )
    { // If loop finished without breaking, l=2N, l2=N
        // This condition should only be true for the base corners already handled.
        // If we reach here for non-corner points, something is wrong.
        throw std::logic_error( "Internal logic error: Failed to find creation level for non-corner point." );
    }

    Vec3 p1, p2; // Parent points

    // Identify the rule used at creation level l=2*l2, based on relative position
    if ( i % l == 0 && j % l == l2 )
    {
        // Rule 1: Horizontal midpoint ("rows" loop)
        // i is multiple of l, j is halfway (offset l2)
        p1 = compute_node_recursive( i, j - l2, corners, cache );
        p2 = compute_node_recursive( i, j + l2, corners, cache );
    }
    else if ( i % l == l2 && j % l == 0 )
    {
        // Rule 2: Vertical midpoint ("columns" loop)
        // j is multiple of l, i is halfway (offset l2)
        p1 = compute_node_recursive( i - l2, j, corners, cache );
        p2 = compute_node_recursive( i + l2, j, corners, cache );
    }
    else if ( i % l == l2 && j % l == l2 )
    {
        // Rule 3: Diagonal midpoint ("diagonals" loop)
        // Both i and j are halfway (offset l2)
        p1 = compute_node_recursive( i - l2, j + l2, corners, cache );
        p2 = compute_node_recursive( i + l2, j - l2, corners, cache );
    }
    else
    {
        // This should not happen if the logic for finding l is correct and (i,j) is not a base corner.
        // The checks i%l and j%l should cover all non-zero remainder possibilities correctly.
        // If i%l==0 and j%l==0, the while loop should have continued.
        throw std::logic_error( "Internal logic error: Point does not match any creation rule." );
    }

    // Calculate Euclidean midpoint
    Vec3 mid = p1 + p2;

    // Normalize to project onto the unit sphere
    Vec3 result = mid.normalized();

    // --- 4. Store result in cache and return ---
    cache[cache_key] = result;
    return result;
}

/// @brief Generates coordinates for a rectangular subdomain of the refined spherical grid.
///
/// @param subdomain_coords_host a properly sized host-allocated view that is filled with the coordinates of the points
/// @param corners Struct containing the base corner points and N = ntan - 1.
/// @param i_start_incl Starting row index (inclusive) of the subdomain (global index).
/// @param i_end_incl Ending row index (inclusive) of the subdomain (global index).
/// @param j_start_incl Starting column index (inclusive) of the subdomain (global index).
/// @param j_end_incl Ending column index (inclusive) of the subdomain (global index).
///
/// @return Kokkos::View<T**[3], Kokkos::HostSpace> Host view containing coordinates
///         for the subdomain. Dimensions are ((i_end_incl - 1) - i_start, (j_end_incl - 1) - j_start).
///
template < std::floating_point T >
void compute_subdomain(
    const typename Grid3DDataVec< T, 3 >::HostMirror& subdomain_coords_host,
    int                                               subdomain_idx,
    const BaseCorners< T >&                           corners,
    int                                               i_start_incl,
    int                                               i_end_incl,
    int                                               j_start_incl,
    int                                               j_end_incl )
{
    using Vec3 = dense::Vec< T, 3 >;

    const int i_start = i_start_incl;
    const int j_start = j_start_incl;
    const int i_end   = i_end_incl + 1;
    const int j_end   = j_end_incl + 1;

    // --- Input Validation ---
    const int N    = corners.N;
    const int ntan = N + 1; // Derive ntan from N in corners struct
    if ( i_start < 0 || i_end > ntan || i_start >= i_end || j_start < 0 || j_end > ntan || j_start >= j_end )
    {
        throw std::invalid_argument( "Invalid subdomain boundaries." );
    }
    if ( N <= 0 || ( N > 0 && ( N & ( N - 1 ) ) != 0 ) )
    {
        throw std::invalid_argument( "BaseCorners.N must be a positive power of 2." );
    }

    // --- Initialization ---
    const size_t subdomain_rows = i_end - i_start;
    const size_t subdomain_cols = j_end - j_start;

    if ( subdomain_coords_host.extent( 1 ) != subdomain_rows || subdomain_coords_host.extent( 2 ) != subdomain_cols )
    {
        Kokkos::abort(
            "Invalid subdomain dimensions in compute_subdomain(). "
            "Could be due to having more subdomains than elements. "
            "But could be something else also..." );
    }

    MemoizationCache< T > cache; // Each subdomain computation gets its own cache

    // --- Compute nodes within the subdomain ---
    for ( int i = i_start; i < i_end; ++i )
    {
        for ( int j = j_start; j < j_end; ++j )
        {
            // Compute the node coordinates using the recursive function
            Vec3 coords = compute_node_recursive( i, j, corners, cache ); // Pass corners struct

            // Store in the subdomain view (adjusting indices)
            subdomain_coords_host( subdomain_idx, i - i_start, j - j_start, 0 ) = coords( 0 );
            subdomain_coords_host( subdomain_idx, i - i_start, j - j_start, 1 ) = coords( 1 );
            subdomain_coords_host( subdomain_idx, i - i_start, j - j_start, 2 ) = coords( 2 );
        }
    }
}

template < std::floating_point T >
void unit_sphere_single_shell_subdomain_coords(
    const typename Grid3DDataVec< T, 3 >::HostMirror& subdomain_coords_host,
    int                                               subdomain_idx,
    int                                               diamond_id,
    int                                               ntan,
    int                                               i_start_incl,
    int                                               i_end_incl,
    int                                               j_start_incl,
    int                                               j_end_incl )
{
    // Coordinates of the twelve icosahedral nodes of the base grid
    real_t i_node[12][3];

    // Association of the ten diamonds to the twelve icosahedral nodes
    //
    // For each diamond we store the indices of its vertices on the
    // icosahedral base grid in this map. Ordering: We start with the
    // pole and proceed in counter-clockwise fashion.
    int d_node[10][4];

    // -----------------------------------------
    //  Initialise the twelve icosahedral nodes
    // -----------------------------------------

    // the pentagonal nodes on each "ring" are given in anti-clockwise ordering
    real_t fifthpi = real_c( 0.4 * std::asin( 1.0 ) );
    real_t w       = real_c( 2.0 * std::acos( 1.0 / ( 2.0 * std::sin( fifthpi ) ) ) );
    real_t cosw    = std::cos( w );
    real_t sinw    = std::sin( w );
    real_t phi     = 0.0;

    // North Pole
    i_node[0][0] = 0.0;
    i_node[0][1] = 0.0;
    i_node[0][2] = +1.0;

    // South Pole
    i_node[11][0] = 0.0;
    i_node[11][1] = 0.0;
    i_node[11][2] = -1.0;

    // upper ring
    for ( int k = 1; k <= 5; k++ )
    {
        phi          = real_c( 2.0 ) * ( real_c( k ) - real_c( 0.5 ) ) * fifthpi;
        i_node[k][0] = sinw * std::cos( phi );
        i_node[k][1] = sinw * std::sin( phi );
        i_node[k][2] = cosw;
    }

    // lower ring
    for ( int k = 1; k <= 5; k++ )
    {
        phi              = real_c( 2.0 ) * ( real_c( k ) - 1 ) * fifthpi;
        i_node[k + 5][0] = sinw * std::cos( phi );
        i_node[k + 5][1] = sinw * std::sin( phi );
        i_node[k + 5][2] = -cosw;
    }

    // ----------------------------------------------
    // Setup internal index maps for mesh generation
    // ----------------------------------------------

    // Map icosahedral node indices to diamonds (northern hemisphere)
    d_node[0][0] = 0;
    d_node[0][1] = 5;
    d_node[0][2] = 6;
    d_node[0][3] = 1;
    d_node[1][0] = 0;
    d_node[1][1] = 1;
    d_node[1][2] = 7;
    d_node[1][3] = 2;
    d_node[2][0] = 0;
    d_node[2][1] = 2;
    d_node[2][2] = 8;
    d_node[2][3] = 3;
    d_node[3][0] = 0;
    d_node[3][1] = 3;
    d_node[3][2] = 9;
    d_node[3][3] = 4;
    d_node[4][0] = 0;
    d_node[4][1] = 4;
    d_node[4][2] = 10;
    d_node[4][3] = 5;

    // Map icosahedral node indices to diamonds (southern hemisphere)
    d_node[5][0] = 11;
    d_node[5][1] = 7;
    d_node[5][2] = 1;
    d_node[5][3] = 6;
    d_node[6][0] = 11;
    d_node[6][1] = 8;
    d_node[6][2] = 2;
    d_node[6][3] = 7;
    d_node[7][0] = 11;
    d_node[7][1] = 9;
    d_node[7][2] = 3;
    d_node[7][3] = 8;
    d_node[8][0] = 11;
    d_node[8][1] = 10;
    d_node[8][2] = 4;
    d_node[8][3] = 9;
    d_node[9][0] = 11;
    d_node[9][1] = 6;
    d_node[9][2] = 5;
    d_node[9][3] = 10;

    // ------------------------
    //  Meshing of unit sphere
    // ------------------------

    // "left" and "right" w.r.t. d_node depend on hemisphere
    int L, R;
    if ( diamond_id < 5 )
    {
        L = 1;
        R = 3;
    }
    else
    {
        R = 1;
        L = 3;
    }

    BaseCorners< T > corners;
    corners.N = ntan - 1;

    // Insert coordinates of four nodes of this icosahedral diamond for each dim.
    for ( int i = 0; i < 3; ++i )
    {
        corners.p00( i ) = i_node[d_node[diamond_id][0]][i];
        corners.pN0( i ) = i_node[d_node[diamond_id][L]][i];
        corners.pNN( i ) = i_node[d_node[diamond_id][2]][i];
        corners.p0N( i ) = i_node[d_node[diamond_id][R]][i];
    }

    return compute_subdomain(
        subdomain_coords_host, subdomain_idx, corners, i_start_incl, i_end_incl, j_start_incl, j_end_incl );
}

template < std::floating_point T >
void unit_sphere_single_shell_subdomain_coords(
    const typename Grid3DDataVec< T, 3 >::HostMirror& subdomain_coords_host,
    int                                               subdomain_idx,
    int                                               diamond_id,
    int                                               global_refinements,
    int                                               num_subdomains_per_side,
    int                                               subdomain_i,
    int                                               subdomain_j )
{
    const auto elements_per_side = 1 << global_refinements;
    const auto ntan              = elements_per_side + 1;

    const auto elements_subdomain_base = elements_per_side / num_subdomains_per_side;
    const auto elements_remainder      = elements_per_side % num_subdomains_per_side;

    const auto elements_in_subdomain_i = elements_subdomain_base + ( subdomain_i < elements_remainder ? 1 : 0 );
    const auto elements_in_subdomain_j = elements_subdomain_base + ( subdomain_j < elements_remainder ? 1 : 0 );

    const auto start_i = subdomain_i * elements_subdomain_base + std::min( subdomain_i, elements_remainder );
    const auto start_j = subdomain_j * elements_subdomain_base + std::min( subdomain_j, elements_remainder );

    const auto end_i = start_i + elements_in_subdomain_i;
    const auto end_j = start_j + elements_in_subdomain_j;

    unit_sphere_single_shell_subdomain_coords< T >(
        subdomain_coords_host, subdomain_idx, diamond_id, ntan, start_i, end_i, start_j, end_j );
}

/// @brief (Sortable) Globally unique identifier for a single subdomain of a diamond.
///
/// Carries the diamond ID, and the subdomain index (x, y, r) inside the diamond.
/// Is globally unique (particularly useful for in parallel settings).
/// Does not carry information about the refinement of a subdomain (just the index).
class SubdomainInfo
{
  public:
    /// @brief Creates invalid ID.
    SubdomainInfo()
    : diamond_id_( -1 )
    , subdomain_x_( -1 )
    , subdomain_y_( -1 )
    , subdomain_r_( -1 )
    {}

    /// @brief Creates unique subdomain ID.
    SubdomainInfo( int diamond_id, int subdomain_x, int subdomain_y, int subdomain_r )
    : diamond_id_( diamond_id )
    , subdomain_x_( subdomain_x )
    , subdomain_y_( subdomain_y )
    , subdomain_r_( subdomain_r )
    {}

    /// @brief Read from encoded 64-bit integer.
    ///
    /// See \ref global_id() for format.
    explicit SubdomainInfo( const int64_t global_id )
    : diamond_id_( static_cast< int >( ( global_id >> 57 ) ) )
    , subdomain_x_( static_cast< int >( ( global_id >> 0 ) & ( ( 1 << 19 ) - 1 ) ) )
    , subdomain_y_( static_cast< int >( ( global_id >> 19 ) & ( ( 1 << 19 ) - 1 ) ) )
    , subdomain_r_( static_cast< int >( ( global_id >> 38 ) & ( ( 1 << 19 ) - 1 ) ) )
    {
        if ( global_id != this->global_id() )
        {
            Kokkos::abort( "Invalid global ID conversion." );
        }
    }

    /// @brief Diamond that subdomain is part of.
    int diamond_id() const { return diamond_id_; }

    /// @brief Subdomain index in lateral x-direction (local to the diamond).
    int subdomain_x() const { return subdomain_x_; }

    /// @brief Subdomain index in lateral y-direction (local to the diamond).
    int subdomain_y() const { return subdomain_y_; }

    /// @brief Subdomain index in the radial direction (local to the diamond).
    int subdomain_r() const { return subdomain_r_; }

    bool operator<( const SubdomainInfo& other ) const
    {
        return std::tie( diamond_id_, subdomain_r_, subdomain_y_, subdomain_x_ ) <
               std::tie( other.diamond_id_, other.subdomain_r_, other.subdomain_y_, other.subdomain_x_ );
    }

    bool operator==( const SubdomainInfo& other ) const
    {
        return std::tie( diamond_id_, subdomain_r_, subdomain_y_, subdomain_x_ ) ==
               std::tie( other.diamond_id_, other.subdomain_r_, other.subdomain_y_, other.subdomain_x_ );
    }

    /// @brief Scrambles the four indices (diamond ID, x, y, r) into a single integer.
    ///
    /// Format
    /// @code
    ///
    /// bits (LSB)  0-18        (19 bits): subdomain_x
    /// bits       19-37        (19 bits): subdomain_y
    /// bits       38-56        (19 bits): subdomain_r
    /// bits       57-63 (MSB)  ( 7 bits): diamond_id (in [0, ..., 9])
    ///
    /// @endcode
    [[nodiscard]] int64_t global_id() const
    {
        if ( diamond_id_ >= 10 )
        {
            throw std::logic_error( "Diamond ID must be less than 10." );
        }

        if ( subdomain_x_ > ( 1 << 19 ) - 1 || subdomain_y_ > ( 1 << 19 ) - 1 || subdomain_r_ > ( 1 << 19 ) - 1 )
        {
            throw std::logic_error( "Subdomain indices too large." );
        }

        return ( static_cast< int64_t >( diamond_id_ ) << 57 ) | ( static_cast< int64_t >( subdomain_r_ ) << 38 ) |
               ( static_cast< int64_t >( subdomain_y_ ) << 19 ) | ( static_cast< int64_t >( subdomain_x_ ) );
    }

  private:
    /// Diamond that subdomain is part of.
    int diamond_id_;

    /// Subdomain index in lateral x-direction (local to the diamond).
    int subdomain_x_;

    /// Subdomain index in lateral y-direction (local to the diamond).
    int subdomain_y_;

    /// Subdomain index in radial direction.
    int subdomain_r_;
};

inline std::ostream& operator<<( std::ostream& os, const SubdomainInfo& si )
{
    os << "Diamond ID: " << si.diamond_id() << ", subdomains (" << si.subdomain_x() << ", " << si.subdomain_y() << ", "
       << si.subdomain_r() << ")";
    return os;
}

using SubdomainToRankDistributionFunction = std::function< mpi::MPIRank( const SubdomainInfo&, const int, const int ) >;

/// @brief Assigns all subdomains to root (rank 0).
inline mpi::MPIRank subdomain_to_rank_all_root(
    const SubdomainInfo& subdomain_info,
    const int            num_subdomains_per_diamond_laterally,
    const int            num_subdomains_per_diamond_radially )
{
    return 0;
}

/// @brief Distributes subdomains to ranks as evenly as possible.
///
/// Not really sophisticated, just iterates over subdomain indices.
/// Tries to concentrate ranks in as few diamonds as possible.
/// But does not optimize surface to volume ratio.
/// For that we need something like a z- or Hilbert curve.
inline mpi::MPIRank subdomain_to_rank_iterate_diamond_subdomains(
    const SubdomainInfo& subdomain_info,
    const int            num_subdomains_per_diamond_laterally,
    const int            num_subdomains_per_diamond_radially )
{
    const auto processes = mpi::num_processes();

    const auto subdomains_per_diamond = num_subdomains_per_diamond_laterally * num_subdomains_per_diamond_laterally *
                                        num_subdomains_per_diamond_radially;

    const auto subdomains = 10 * subdomains_per_diamond;

    const auto div = subdomains / processes;
    const auto rem = subdomains % processes;

    const auto subdomain_global_index =
        subdomain_info.diamond_id() * subdomains_per_diamond +
        subdomain_info.subdomain_x() * num_subdomains_per_diamond_laterally * num_subdomains_per_diamond_radially +
        subdomain_info.subdomain_y() * num_subdomains_per_diamond_radially + subdomain_info.subdomain_r();

    if ( subdomain_global_index < ( div + 1 ) * rem )
    {
        return subdomain_global_index / ( div + 1 );
    }

    return rem + ( subdomain_global_index - ( div + 1 ) * rem ) / div;
}

/// @brief Information about the thick spherical shell mesh.
///
/// @note If you want to create a domain for an application, use the \ref DistributedDomain class, which constructs an
///       instance of this class internally.
///
/// **General information**
///
/// The thick spherical shell is built from ten spherical diamonds. The diamonds are essentially curved hexahedra.
/// The number of cells in lateral directions is required to be a power of 2, the number of cells in the radial
/// direction can be chosen arbitrarily (though a power of two allows for maximally deep multigrid hierarchies).
///
/// Each diamond can be subdivided into subdomains (in all three directions) for better parallel distribution (each
/// process can only operate on one or more entire subdomains).
///
/// This class holds data such as
/// - the shell radii,
/// - the number of subdomains in each direction (on each diamond),
/// - the number of nodes per subdomain in each direction (including overlapping nodes where two or more subdomains
///   meet).
///
/// Note that all subdomains always have the same shape.
///
/// **Multigrid and coarsening**
///
/// Since the global number of cells in a diamond in lateral and radial direction does not need to match, and since
/// the number of cells in radial direction does not even need to be a power of two (although it is a good idea to
/// choose it that way), this class computes the maximum number of coarsening steps (which is equivalent to the number
/// of "refinement levels") dynamically. Thus, a bad choice for the number of radial layers may result in a mesh that
/// cannot be coarsened at all.
///
/// **Parallel distribution**
///
/// This class has no notion of parallel distribution. For that refer to the \ref DistributedDomain class.
///
class DomainInfo
{
  public:
    DomainInfo() = default;

    /// @brief Constructs a thick spherical shell with one subdomain per diamond (10 subdomains total) and shells at
    /// specific radii.
    ///
    /// @note Use `uniform_shell_radii()` to quickly construct a uniform list of radii.
    ///
    /// Note: a 'shell' is a spherical 2D manifold in 3D space (it is thin),
    ///       a 'layer' is defined as the volume between two 'shells' (it is thick)
    ///
    /// @param diamond_lateral_refinement_level number of lateral diamond refinements
    /// @param radii list of shell radii, vector must have at least 2 elements
    DomainInfo( int diamond_lateral_refinement_level, const std::vector< double >& radii )
    : DomainInfo( diamond_lateral_refinement_level, radii, 1, 1 )
    {}

    /// @brief Constructs a thick spherical shell with shells at specific radii.
    ///
    /// @note Use `uniform_shell_radii()` to quickly construct a uniform list of radii.
    ///
    /// Note: a 'shell' is a spherical 2D manifold in 3D space (it is thin),
    ///       a 'layer' is defined as the volume between two 'shells' (it is thick)
    ///
    /// @param diamond_lateral_refinement_level number of lateral diamond refinements
    /// @param radii list of shell radii, vector must have at least 2 elements
    /// @param num_subdomains_in_lateral_direction number of subdomains in lateral direction per diamond
    /// @param num_subdomains_in_radial_direction number of subdomains in radial direction per diamond
    DomainInfo(
        int                          diamond_lateral_refinement_level,
        const std::vector< double >& radii,
        int                          num_subdomains_in_lateral_direction,
        int                          num_subdomains_in_radial_direction )
    : diamond_lateral_refinement_level_( diamond_lateral_refinement_level )
    , radii_( radii )
    , num_subdomains_in_lateral_direction_( num_subdomains_in_lateral_direction )
    , num_subdomains_in_radial_direction_( num_subdomains_in_radial_direction )
    {
        const int num_layers = static_cast< int >( radii.size() ) - 1;
        if ( num_layers % num_subdomains_in_radial_direction_ != 0 )
        {
            throw std::invalid_argument(
                "Number of layers must be divisible by number of subdomains in radial direction. "
                "You have requested " +
                std::to_string( num_layers ) + " layers (" + std::to_string( radii.size() ) + " shells), and " +
                std::to_string( num_subdomains_in_radial_direction ) + " radial subdomains." );
        }
    }

    /// @brief The "maximum refinement level" of the subdomains.
    ///
    /// This (non-negative) number is essentially indicating how many times a subdomain can be uniformly coarsened.
    int subdomain_max_refinement_level() const
    {
        const auto max_refinement_level_lat =
            std::countr_zero( static_cast< unsigned >( subdomain_num_nodes_per_side_laterally() - 1 ) );
        const auto max_refinement_level_rad =
            std::countr_zero( static_cast< unsigned >( subdomain_num_nodes_radially() - 1 ) );

        return std::min( max_refinement_level_lat, max_refinement_level_rad );
    }

    int diamond_lateral_refinement_level() const { return diamond_lateral_refinement_level_; }

    const std::vector< double >& radii() const { return radii_; }

    int num_subdomains_per_diamond_side() const { return num_subdomains_in_lateral_direction_; }

    int num_subdomains_in_radial_direction() const { return num_subdomains_in_radial_direction_; }

    /// @brief Equivalent to calling subdomain_num_nodes_per_side_laterally( subdomain_refinement_level() )
    int subdomain_num_nodes_per_side_laterally() const
    {
        const int num_cells_per_diamond_side   = 1 << diamond_lateral_refinement_level();
        const int num_cells_per_subdomain_side = num_cells_per_diamond_side / num_subdomains_per_diamond_side();
        const int num_nodes_per_subdomain_side = num_cells_per_subdomain_side + 1;
        return num_nodes_per_subdomain_side;
    }

    /// @brief Equivalent to calling subdomain_num_nodes_radially( subdomain_refinement_level() )
    int subdomain_num_nodes_radially() const
    {
        const int num_layers               = radii_.size() - 1;
        const int num_layers_per_subdomain = num_layers / num_subdomains_in_radial_direction_;
        return num_layers_per_subdomain + 1;
    }

    /// @brief Number of nodes in the lateral direction of a subdomain on the passed level.
    ///
    /// The level must be non-negative. The finest level is given by subdomain_max_refinement_level().
    int subdomain_num_nodes_per_side_laterally( const int level ) const
    {
        if ( level < 0 )
        {
            throw std::invalid_argument( "Level must be non-negative." );
        }

        if ( level > subdomain_max_refinement_level() )
        {
            throw std::invalid_argument( "Level must be less than or equal to max subdomain refinement level." );
        }

        const int coarsening_steps = subdomain_max_refinement_level() - level;
        return ( ( subdomain_num_nodes_per_side_laterally() - 1 ) >> coarsening_steps ) + 1;
    }

    /// @brief Number of nodes in the radial direction of a subdomain on the passed level.
    ///
    /// The level must be non-negative. The finest level is given by subdomain_max_refinement_level().
    int subdomain_num_nodes_radially( const int level ) const
    {
        if ( level < 0 )
        {
            throw std::invalid_argument( "Level must be non-negative." );
        }

        if ( level > subdomain_max_refinement_level() )
        {
            throw std::invalid_argument( "Level must be less than or equal to subdomain refinement level." );
        }

        const int coarsening_steps = subdomain_max_refinement_level() - level;
        return ( ( subdomain_num_nodes_radially() - 1 ) >> coarsening_steps ) + 1;
    }

    std::vector< SubdomainInfo > local_subdomains( const SubdomainToRankDistributionFunction& subdomain_to_rank ) const
    {
        const auto rank = mpi::rank();

        std::vector< SubdomainInfo > subdomains;
        for ( int diamond_id = 0; diamond_id < 10; diamond_id++ )
        {
            for ( int x = 0; x < num_subdomains_per_diamond_side(); x++ )
            {
                for ( int y = 0; y < num_subdomains_per_diamond_side(); y++ )
                {
                    for ( int r = 0; r < num_subdomains_in_radial_direction_; r++ )
                    {
                        SubdomainInfo subdomain( diamond_id, x, y, r );

                        const auto target_rank = subdomain_to_rank(
                            subdomain, num_subdomains_per_diamond_side(), num_subdomains_in_radial_direction() );

                        if ( target_rank == rank )
                        {
                            subdomains.push_back( subdomain );
                        }
                    }
                }
            }
        }

        if ( subdomains.empty() )
        {
            throw std::logic_error( "No local subdomains found on rank " + std::to_string( rank ) + "." );
        }

        return subdomains;
    }

    /// @brief Number of global subdomains.
    [[nodiscard]] int num_global_subdomains() const
    {
        return 10 * num_subdomains_per_diamond_side() * num_subdomains_per_diamond_side() *
               num_subdomains_in_radial_direction();
    }

    /// @brief Number of hex cells on the finest level, per subdomain.
    [[nodiscard]] long long num_micro_hex_cells_per_subdomain() const
    {
        return static_cast< long long >( subdomain_num_nodes_per_side_laterally() - 1 ) *
               ( subdomain_num_nodes_per_side_laterally() - 1 ) * ( subdomain_num_nodes_radially() - 1 );
    }

    /// @brief Number of hex cells on the finest level, globally.
    [[nodiscard]] long long num_global_micro_hex_cells() const
    {
        return num_global_subdomains() * num_micro_hex_cells_per_subdomain();
    }

  private:
    /// Number of times each diamond is refined laterally in each direction.
    int diamond_lateral_refinement_level_{};

    /// Shell radii.
    std::vector< double > radii_;

    /// Number of subdomains per diamond (for parallel partitioning) in the lateral direction (at least 1).
    int num_subdomains_in_lateral_direction_{};

    /// Number of subdomains per diamond (for parallel partitioning) in the radial direction (at least 1).
    int num_subdomains_in_radial_direction_{};
};

/// @brief Neighborhood information of a single subdomain.
///
/// @note If you want to create a domain for an application, use the \ref terra::grid::shell::DistributedDomain class,
///       which constructs an instance of this class internally.
///
/// Holds information such as the MPI ranks of the neighboring subdomains, and their orientation.
/// Required for communication (packing, unpacking, sending, receiving 'ghost-layer' data).
///
/// **Details on communication**
///
/// Data is rotated during unpacking.
///
/// *Packing/sending*
///
/// Sender just packs data from the grid into a buffer using the (x, y, r) coordinates in order.
/// For instance: the face boundary xr is packed into a 2D buffer: buffer( x, r ), the face boundary yr is packed
/// as buffer( y, r ), always iterating locally from 0 to end.
///
/// *Unpacking*
///
/// When a packet from a certain neighbor subdomain arrives, we have the following information
/// set up in this class (for instance in the `NeighborSubdomainTupleFace` instances):
///
/// *Organization*
///
/// At each boundary face of a local subdomain we store in this class a list of tuples with entries:
///
/// SubdomainInfo:         neighboring subdomain identifier
/// BoundaryFace:          boundary face of the neighboring subdomain (from its local view)
/// UnpackingOrderingFace: information how to iterate over the buffer for each coordinate during unpacking
///
/// So e.g., the data (for some subdomain):
///
/// @code
///   neighborhood_face_[ F_X1R ] = { neighbor_subdomain_info, F_1YR, (BACKWARD, FORWARD), neighbor_rank }
/// @endcode
///
/// means that for this subdomain, the boundary face `F_X1R` interfaces with the neighbor subdomain
/// `neighbor_subdomain_info`, at its boundary `F_1YR`, that is located on rank `neighbor_rank`.
/// If we unpack data that we receive from the subdomain, we must invert the iteration over the first
/// buffer index, and move forward in the second index.
///
/// @note See \ref terra::grid::BoundaryVertex, \ref terra::grid::BoundaryEdge, \ref terra::grid::BoundaryFace,
///       \ref terra::grid::BoundaryDirection for details on the naming convention of the boundary types like `F_X1R`.
///       Roughly: `0 == start`, `1 == end`, `X == varying in x`, `Y == varying in y`, `R == varying in r`.
///
/// *Execution*
///
/// Let's assume we receive data from the neighbor specified above.
///
/// Sender side (with local boundary `F_1YR`):
///
/// @code
///   buffer( y, r ) = send_data( sender_local_subdomain_id, x_size - 1, y, r )
///                                                          ^^^^^^^^^^
///                                                           == x_end
///                                                           == const
/// @endcode
///
/// Receiver side (with local boundary `F_X1R`):
///
/// @code
///   recv_data( receiver_local_subdomain_id, x, y_size - 1, r ) = buffer( x_size - 1 - x, r )
///                                              ^^^^^^^^^^                ^^^^^^^^^^^^^^  ^
///                                               == y_end                    BACKWARD     FORWARD
///                                               == const
/// @endcode
///
/// @note It is due to the structure of the spherical shell mesh that we never have to swap the indices!
///       (For vertex boundaries this is anyway not required (0D array) and for edges neither (1D array - we only
///       have forward and backward iteration).)
///       Thus, it is enough to have the FORWARD/BACKWARD tuple! The radial direction is always radial.
///       And if we communicate in the radial direction (i.e., sending "xy-planes") then we never need to swap since
///       we are in the same diamond.
///
class SubdomainNeighborhood
{
  public:
    using UnpackOrderingEdge = BoundaryDirection;
    using UnpackOrderingFace = std::tuple< BoundaryDirection, BoundaryDirection >;

    using NeighborSubdomainTupleVertex = std::tuple< SubdomainInfo, BoundaryVertex, mpi::MPIRank >;
    using NeighborSubdomainTupleEdge   = std::tuple< SubdomainInfo, BoundaryEdge, UnpackOrderingEdge, mpi::MPIRank >;
    using NeighborSubdomainTupleFace   = std::tuple< SubdomainInfo, BoundaryFace, UnpackOrderingFace, mpi::MPIRank >;

    SubdomainNeighborhood() = default;

    SubdomainNeighborhood(
        const DomainInfo&                          domain_info,
        const SubdomainInfo&                       subdomain_info,
        const SubdomainToRankDistributionFunction& subdomain_to_rank )
    { setup_neighborhood( domain_info, subdomain_info, subdomain_to_rank ); }

    const std::map< BoundaryVertex, std::vector< NeighborSubdomainTupleVertex > >& neighborhood_vertex() const
    { return neighborhood_vertex_; }

    const std::map< BoundaryEdge, std::vector< NeighborSubdomainTupleEdge > >& neighborhood_edge() const
    { return neighborhood_edge_; }

    const std::map< BoundaryFace, NeighborSubdomainTupleFace >& neighborhood_face() const { return neighborhood_face_; }

  private:
    void setup_neighborhood_faces( const DomainInfo& domain_info, const SubdomainInfo& subdomain_info )
    {
        const mpi::MPIRank rank_will_be_overwritten_later = -1;

        const int diamond_id  = subdomain_info.diamond_id();
        const int subdomain_x = subdomain_info.subdomain_x();
        const int subdomain_y = subdomain_info.subdomain_y();
        const int subdomain_r = subdomain_info.subdomain_r();

        const int num_lateral_subdomains = domain_info.num_subdomains_per_diamond_side();
        const int num_radial_subdomains  = domain_info.num_subdomains_in_radial_direction();

        for ( const auto boundary_face : all_boundary_faces )
        {
            const bool diamond_diamond_boundary =
                ( boundary_face == BoundaryFace::F_0YR && subdomain_x == 0 ) ||
                ( boundary_face == BoundaryFace::F_1YR && subdomain_x == num_lateral_subdomains - 1 ) ||
                ( boundary_face == BoundaryFace::F_X0R && subdomain_y == 0 ) ||
                ( boundary_face == BoundaryFace::F_X1R && subdomain_y == num_lateral_subdomains - 1 );

            if ( diamond_diamond_boundary )
            {
                // Node equivalences: part one - communication between diamonds at the same poles
                // Iterating forward laterally.

                // d_0( 0, :, r ) = d_1( :, 0, r )
                // d_1( 0, :, r ) = d_2( :, 0, r )
                // d_2( 0, :, r ) = d_3( :, 0, r )
                // d_3( 0, :, r ) = d_4( :, 0, r )
                // d_4( 0, :, r ) = d_0( :, 0, r )

                // d_5( 0, :, r ) = d_6( :, 0, r )
                // d_6( 0, :, r ) = d_7( :, 0, r )
                // d_7( 0, :, r ) = d_8( :, 0, r )
                // d_8( 0, :, r ) = d_9( :, 0, r )
                // d_9( 0, :, r ) = d_5( :, 0, r )

                // Node equivalences: part two - communication between diamonds at different poles
                // Need to go backwards laterally.

                // d_0( :, end, r ) = d_5( end, :, r )
                // d_1( :, end, r ) = d_6( end, :, r )
                // d_2( :, end, r ) = d_7( end, :, r )
                // d_3( :, end, r ) = d_8( end, :, r )
                // d_4( :, end, r ) = d_9( end, :, r )

                // d_5( :, end, r ) = d_1( end, :, r )
                // d_6( :, end, r ) = d_2( end, :, r )
                // d_7( :, end, r ) = d_3( end, :, r )
                // d_8( :, end, r ) = d_4( end, :, r )
                // d_9( :, end, r ) = d_0( end, :, r )

                switch ( diamond_id )
                {
                case 0:
                case 1:
                case 2:
                case 3:
                case 4:

                    switch ( boundary_face )
                    {
                    // (north-north)
                    case BoundaryFace::F_0YR:
                        neighborhood_face_[BoundaryFace::F_0YR] = {
                            SubdomainInfo( ( diamond_id + 1 ) % 5, subdomain_y, 0, subdomain_r ),
                            BoundaryFace::F_X0R,
                            { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                        break;
                    case BoundaryFace::F_X0R:
                        neighborhood_face_[BoundaryFace::F_X0R] = {
                            SubdomainInfo( ( diamond_id + 4 ) % 5, 0, subdomain_x, subdomain_r ),
                            BoundaryFace::F_0YR,
                            { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                        break;

                    // (north-south)
                    case BoundaryFace::F_X1R:
                        neighborhood_face_[BoundaryFace::F_X1R] = {
                            SubdomainInfo(
                                diamond_id + 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - 1 - subdomain_x,
                                subdomain_r ),
                            BoundaryFace::F_1YR,
                            { BoundaryDirection::BACKWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                        break;
                    case BoundaryFace::F_1YR:
                        neighborhood_face_[BoundaryFace::F_1YR] = {
                            SubdomainInfo(
                                ( diamond_id + 4 ) % 5 + 5,
                                num_lateral_subdomains - 1 - subdomain_y,
                                num_lateral_subdomains - 1,
                                subdomain_r ),
                            BoundaryFace::F_X1R,
                            { BoundaryDirection::BACKWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                        break;

                    default:
                        Kokkos::abort( "This should not happen." );
                    }

                    break;

                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                    switch ( boundary_face )
                    {
                    // (south-south)
                    case BoundaryFace::F_0YR:
                        neighborhood_face_[BoundaryFace::F_0YR] = {
                            SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, subdomain_y, 0, subdomain_r ),
                            BoundaryFace::F_X0R,
                            { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                        break;
                    case BoundaryFace::F_X0R:
                        neighborhood_face_[BoundaryFace::F_X0R] = {
                            SubdomainInfo( ( diamond_id - 1 ) % 5 + 5, 0, subdomain_x, subdomain_r ),
                            BoundaryFace::F_0YR,
                            { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                        break;

                        // (south-north)
                    case BoundaryFace::F_X1R:
                        neighborhood_face_[BoundaryFace::F_X1R] = {
                            SubdomainInfo(
                                ( diamond_id - 4 ) % 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - 1 - subdomain_x,
                                subdomain_r ),
                            BoundaryFace::F_1YR,
                            { BoundaryDirection::BACKWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                        break;
                    case BoundaryFace::F_1YR:
                        neighborhood_face_[BoundaryFace::F_1YR] = {
                            SubdomainInfo(
                                diamond_id - 5,
                                num_lateral_subdomains - 1 - subdomain_y,
                                num_lateral_subdomains - 1,
                                subdomain_r ),
                            BoundaryFace::F_X1R,
                            { BoundaryDirection::BACKWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                        break;
                    default:
                        Kokkos::abort( "This should not happen." );
                    }
                    break;

                default:
                    Kokkos::abort( "Invalid diamond id." );
                }
            }
            else
            {
                // Same diamond.

                switch ( boundary_face )
                {
                case BoundaryFace::F_0YR:
                    neighborhood_face_[boundary_face] = {
                        SubdomainInfo( diamond_id, subdomain_x - 1, subdomain_y, subdomain_r ),
                        BoundaryFace::F_1YR,
                        { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                        rank_will_be_overwritten_later };
                    break;
                case BoundaryFace::F_1YR:
                    neighborhood_face_[boundary_face] = {
                        SubdomainInfo( diamond_id, subdomain_x + 1, subdomain_y, subdomain_r ),
                        BoundaryFace::F_0YR,
                        { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                        rank_will_be_overwritten_later };
                    break;
                case BoundaryFace::F_X0R:
                    neighborhood_face_[boundary_face] = {
                        SubdomainInfo( diamond_id, subdomain_x, subdomain_y - 1, subdomain_r ),
                        BoundaryFace::F_X1R,
                        { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                        rank_will_be_overwritten_later };
                    break;
                case BoundaryFace::F_X1R:
                    neighborhood_face_[boundary_face] = {
                        SubdomainInfo( diamond_id, subdomain_x, subdomain_y + 1, subdomain_r ),
                        BoundaryFace::F_X0R,
                        { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                        rank_will_be_overwritten_later };
                    break;
                case BoundaryFace::F_XY0:
                    if ( subdomain_r > 0 )
                    {
                        neighborhood_face_[boundary_face] = {
                            SubdomainInfo( diamond_id, subdomain_x, subdomain_y, subdomain_r - 1 ),
                            BoundaryFace::F_XY1,
                            { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                    }
                    break;
                case BoundaryFace::F_XY1:
                    if ( subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_face_[boundary_face] = {
                            SubdomainInfo( diamond_id, subdomain_x, subdomain_y, subdomain_r + 1 ),
                            BoundaryFace::F_XY0,
                            { BoundaryDirection::FORWARD, BoundaryDirection::FORWARD },
                            rank_will_be_overwritten_later };
                    }
                    break;
                }
            }
        }
    }

    void setup_neighborhood_edges( const DomainInfo& domain_info, const SubdomainInfo& subdomain_info )
    {
        const int diamond_id  = subdomain_info.diamond_id();
        const int subdomain_x = subdomain_info.subdomain_x();
        const int subdomain_y = subdomain_info.subdomain_y();
        const int subdomain_r = subdomain_info.subdomain_r();

        const int num_lateral_subdomains = domain_info.num_subdomains_per_diamond_side();
        const int num_radial_subdomains  = domain_info.num_subdomains_in_radial_direction();

        for ( const auto boundary_edge : all_boundary_edges )
        {
            // Edges in radial direction (beyond diamond boundaries).

            if ( diamond_id >= 0 && diamond_id <= 4 )
            {
                if ( boundary_edge == BoundaryEdge::E_00R )
                {
                    if ( subdomain_x == 0 && subdomain_y == 0 )
                    {
                        // North pole
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 2 ) % 5, 0, 0, subdomain_r ),
                            BoundaryEdge::E_00R,
                            BoundaryDirection::FORWARD,
                            -1 );
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 3 ) % 5, 0, 0, subdomain_r ),
                            BoundaryEdge::E_00R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                    else if ( subdomain_x == 0 && subdomain_y > 0 )
                    {
                        // Northern hemisphere to the right.
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5, subdomain_y - 1, 0, subdomain_r ),
                            BoundaryEdge::E_10R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                    else if ( subdomain_x > 0 && subdomain_y == 0 )
                    {
                        // Northern hemisphere to the left.
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 4 ) % 5, 0, subdomain_x - 1, subdomain_r ),
                            BoundaryEdge::E_01R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                }
                else if ( boundary_edge == BoundaryEdge::E_01R )
                {
                    if ( subdomain_x == 0 && subdomain_y < num_lateral_subdomains - 1 )
                    {
                        // Northern hemisphere to the right
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5, subdomain_y + 1, 0, subdomain_r ),
                            BoundaryEdge::E_00R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                    else if ( subdomain_x > 0 && subdomain_y == num_lateral_subdomains - 1 )
                    {
                        // Northern hemisphere to the right and southern
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo(
                                diamond_id + 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x,
                                subdomain_r ),
                            BoundaryEdge::E_10R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                }
                else if ( boundary_edge == BoundaryEdge::E_10R )
                {
                    if ( subdomain_y == 0 && subdomain_x < num_lateral_subdomains - 1 )
                    {
                        // Northern hemisphere to the left
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 4 ) % 5, 0, subdomain_x + 1, subdomain_r ),
                            BoundaryEdge::E_00R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                    else if ( subdomain_y > 0 && subdomain_x == num_lateral_subdomains - 1 )
                    {
                        // Northern hemisphere to the left and southern
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo(
                                ( diamond_id + 4 ) % 5 + 5,
                                num_lateral_subdomains - subdomain_y,
                                num_lateral_subdomains - 1,
                                subdomain_r ),
                            BoundaryEdge::E_01R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                }
                else if ( boundary_edge == BoundaryEdge::E_11R )
                {
                    if ( subdomain_x < num_lateral_subdomains - 1 && subdomain_y == num_lateral_subdomains - 1 )
                    {
                        // Northern hemi to southern (right)
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo(
                                diamond_id + 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x - 2,
                                subdomain_r ),
                            BoundaryEdge::E_11R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                    else if ( subdomain_y < num_lateral_subdomains - 1 && subdomain_x == num_lateral_subdomains - 1 )
                    {
                        // Northern hemi to southern (left)
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo(
                                ( diamond_id + 4 ) % 5 + 5,
                                num_lateral_subdomains - subdomain_y - 2,
                                num_lateral_subdomains - 1,
                                subdomain_r ),
                            BoundaryEdge::E_11R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                }
            }
            else if ( diamond_id >= 5 && diamond_id <= 9 )
            {
                if ( boundary_edge == BoundaryEdge::E_00R )
                {
                    if ( subdomain_x == 0 && subdomain_y == 0 )
                    {
                        // South pole
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 2 ) % 5 + 5, 0, 0, subdomain_r ),
                            BoundaryEdge::E_00R,
                            BoundaryDirection::FORWARD,
                            -1 );
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 3 ) % 5 + 5, 0, 0, subdomain_r ),
                            BoundaryEdge::E_00R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }

                    if ( subdomain_x == 0 && subdomain_y > 0 )
                    {
                        // Southern hemisphere to the right.
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, subdomain_y - 1, 0, subdomain_r ),
                            BoundaryEdge::E_10R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                    else if ( subdomain_x > 0 && subdomain_y == 0 )
                    {
                        // Southern hemisphere to the left.
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id + 4 ) % 5 + 5, 0, subdomain_x - 1, subdomain_r ),
                            BoundaryEdge::E_01R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                }

                if ( boundary_edge == BoundaryEdge::E_01R )
                {
                    if ( subdomain_x == 0 && subdomain_y < num_lateral_subdomains - 1 )
                    {
                        // Southern hemisphere to the right
                        neighborhood_edge_[boundary_edge].push_back(
                            { SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, subdomain_y + 1, 0, subdomain_r ),
                              BoundaryEdge::E_00R,
                              BoundaryDirection::FORWARD,
                              -1 } );
                    }
                    else if ( subdomain_x > 0 && subdomain_y == num_lateral_subdomains - 1 )
                    {
                        // Southern hemisphere to the right and north
                        neighborhood_edge_[boundary_edge].push_back(
                            { SubdomainInfo(
                                  ( diamond_id + 1 ) % 5,
                                  num_lateral_subdomains - 1,
                                  num_lateral_subdomains - subdomain_x,
                                  subdomain_r ),
                              BoundaryEdge::E_10R,
                              BoundaryDirection::FORWARD,
                              -1 } );
                    }
                }

                if ( boundary_edge == BoundaryEdge::E_10R )
                {
                    if ( subdomain_y == 0 && subdomain_x < num_lateral_subdomains - 1 )
                    {
                        // Southern hemisphere to the left
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo( ( diamond_id - 1 ) % 5 + 5, 0, subdomain_x + 1, subdomain_r ),
                            BoundaryEdge::E_00R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                    else if ( subdomain_y > 0 && subdomain_x == num_lateral_subdomains - 1 )
                    {
                        // Southern hemisphere to the left and northern
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo(
                                diamond_id - 5,
                                num_lateral_subdomains - subdomain_y,
                                num_lateral_subdomains - 1,
                                subdomain_r ),
                            BoundaryEdge::E_01R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                }

                if ( boundary_edge == BoundaryEdge::E_11R )
                {
                    if ( subdomain_x < num_lateral_subdomains - 1 && subdomain_y == num_lateral_subdomains - 1 )
                    {
                        // Southern hemi to northern (right)
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo(
                                ( diamond_id - 4 ) % 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x - 2,
                                subdomain_r ),
                            BoundaryEdge::E_11R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                    else if ( subdomain_y < num_lateral_subdomains - 1 && subdomain_x == num_lateral_subdomains - 1 )
                    {
                        // Southern hemi to northern (left)
                        neighborhood_edge_[boundary_edge].emplace_back(
                            SubdomainInfo(
                                diamond_id - 5,
                                num_lateral_subdomains - subdomain_y - 2,
                                num_lateral_subdomains - 1,
                                subdomain_r ),
                            BoundaryEdge::E_11R,
                            BoundaryDirection::FORWARD,
                            -1 );
                    }
                }
            }
            else
            {
                Kokkos::abort( "Invalid diamond ID." );
            }

            // Still beyond diamond boundaries but now lateral edges

            if ( diamond_id >= 0 && diamond_id <= 4 )
            {
                if ( boundary_edge == BoundaryEdge::E_0Y0 && subdomain_x == 0 && subdomain_r > 0 )
                {
                    // Northern hemi to the right + cmb
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo( ( diamond_id + 1 ) % 5, subdomain_y, 0, subdomain_r - 1 ),
                        BoundaryEdge::E_X01,
                        BoundaryDirection::FORWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_0Y1 && subdomain_x == 0 &&
                    subdomain_r < num_radial_subdomains - 1 )
                {
                    // Northern hemi to the right + surface
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo( ( diamond_id + 1 ) % 5, subdomain_y, 0, subdomain_r + 1 ),
                        BoundaryEdge::E_X00,
                        BoundaryDirection::FORWARD,
                        -1 );
                }
                else if ( boundary_edge == BoundaryEdge::E_X00 && subdomain_y == 0 && subdomain_r > 0 )
                {
                    // Northern hemi to the left + cmb
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo( ( diamond_id + 4 ) % 5, 0, subdomain_x, subdomain_r - 1 ),
                        BoundaryEdge::E_0Y1,
                        BoundaryDirection::FORWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_X01 && subdomain_y == 0 &&
                    subdomain_r < num_radial_subdomains - 1 )
                {
                    // Northern hemi to the left + surface
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo( ( diamond_id + 4 ) % 5, 0, subdomain_x, subdomain_r + 1 ),
                        BoundaryEdge::E_0Y0,
                        BoundaryDirection::FORWARD,
                        -1 );
                }

                // Northern to southern
                if ( boundary_edge == BoundaryEdge::E_X10 && subdomain_y == num_lateral_subdomains - 1 &&
                     subdomain_r > 0 )
                {
                    // Northern hemi to the bottom right + cmb
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo(
                            diamond_id + 5,
                            num_lateral_subdomains - 1,
                            num_lateral_subdomains - 1 - subdomain_x,
                            subdomain_r - 1 ),
                        BoundaryEdge::E_1Y1,
                        BoundaryDirection::BACKWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_X11 && subdomain_y == num_lateral_subdomains - 1 &&
                    subdomain_r < num_radial_subdomains - 1 )
                {
                    // Northern hemi to the bottom right + surface
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo(
                            diamond_id + 5,
                            num_lateral_subdomains - 1,
                            num_lateral_subdomains - 1 - subdomain_x,
                            subdomain_r + 1 ),
                        BoundaryEdge::E_1Y0,
                        BoundaryDirection::BACKWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_1Y0 && subdomain_x == num_lateral_subdomains - 1 &&
                    subdomain_r > 0 )
                {
                    // Northern hemi to the bottom left + cmb
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo(
                            ( diamond_id + 4 ) % 5 + 5,
                            num_lateral_subdomains - 1 - subdomain_y,
                            num_lateral_subdomains - 1,
                            subdomain_r - 1 ),
                        BoundaryEdge::E_X11,
                        BoundaryDirection::BACKWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_1Y1 && subdomain_x == num_lateral_subdomains - 1 &&
                    subdomain_r < num_radial_subdomains - 1 )
                {
                    // Northern hemi to the bottom left + surface
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo(
                            ( diamond_id + 4 ) % 5 + 5,
                            num_lateral_subdomains - 1 - subdomain_y,
                            num_lateral_subdomains - 1,
                            subdomain_r + 1 ),
                        BoundaryEdge::E_X10,
                        BoundaryDirection::BACKWARD,
                        -1 );
                }
            }
            else if ( diamond_id >= 5 && diamond_id <= 9 )
            {
                if ( boundary_edge == BoundaryEdge::E_0Y0 && subdomain_x == 0 && subdomain_r > 0 )
                {
                    // Southern hemi to the right + cmb
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, subdomain_y, 0, subdomain_r - 1 ),
                        BoundaryEdge::E_X01,
                        BoundaryDirection::FORWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_0Y1 && subdomain_x == 0 &&
                    subdomain_r < num_radial_subdomains - 1 )
                {
                    // Southern hemi to the right + surface
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, subdomain_y, 0, subdomain_r + 1 ),
                        BoundaryEdge::E_X00,
                        BoundaryDirection::FORWARD,
                        -1 );
                }
                else if ( boundary_edge == BoundaryEdge::E_X00 && subdomain_y == 0 && subdomain_r > 0 )
                {
                    // Southern hemi to the left + cmb
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo( ( diamond_id - 1 ) % 5 + 5, 0, subdomain_x, subdomain_r - 1 ),
                        BoundaryEdge::E_0Y1,
                        BoundaryDirection::FORWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_X01 && subdomain_y == 0 &&
                    subdomain_r < num_radial_subdomains - 1 )
                {
                    // Southern hemi to the left + surface
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo( ( diamond_id - 1 ) % 5 + 5, 0, subdomain_x, subdomain_r + 1 ),
                        BoundaryEdge::E_0Y0,
                        BoundaryDirection::FORWARD,
                        -1 );
                }

                // Southern to northern
                if ( boundary_edge == BoundaryEdge::E_X10 && subdomain_y == num_lateral_subdomains - 1 &&
                     subdomain_r > 0 )
                {
                    // Southern hemi to the top right + cmb
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo(
                            ( diamond_id - 4 ) % 5,
                            num_lateral_subdomains - 1,
                            num_lateral_subdomains - 1 - subdomain_x,
                            subdomain_r - 1 ),
                        BoundaryEdge::E_1Y1,
                        BoundaryDirection::BACKWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_X11 && subdomain_y == num_lateral_subdomains - 1 &&
                    subdomain_r < num_radial_subdomains - 1 )
                {
                    // Southern hemi to the top right + surface
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo(
                            ( diamond_id - 4 ) % 5,
                            num_lateral_subdomains - 1,
                            num_lateral_subdomains - 1 - subdomain_x,
                            subdomain_r + 1 ),
                        BoundaryEdge::E_1Y0,
                        BoundaryDirection::BACKWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_1Y0 && subdomain_x == num_lateral_subdomains - 1 &&
                    subdomain_r > 0 )
                {
                    // Southern hemi to the top left + cmb
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo(
                            diamond_id - 5,
                            num_lateral_subdomains - 1 - subdomain_y,
                            num_lateral_subdomains - 1,
                            subdomain_r - 1 ),
                        BoundaryEdge::E_X11,
                        BoundaryDirection::BACKWARD,
                        -1 );
                }
                else if (
                    boundary_edge == BoundaryEdge::E_1Y1 && subdomain_x == num_lateral_subdomains - 1 &&
                    subdomain_r < num_radial_subdomains - 1 )
                {
                    // Southern hemi to the top left + surface
                    neighborhood_edge_[boundary_edge].emplace_back(
                        SubdomainInfo(
                            diamond_id - 5,
                            num_lateral_subdomains - 1 - subdomain_y,
                            num_lateral_subdomains - 1,
                            subdomain_r + 1 ),
                        BoundaryEdge::E_X10,
                        BoundaryDirection::BACKWARD,
                        -1 );
                }
            }
            else
            {
                Kokkos::abort( "Invalid diamond ID" );
            }

            // Now only same diamond cases

            // Edges in radial direction

            if ( boundary_edge == BoundaryEdge::E_00R && subdomain_x > 0 && subdomain_y > 0 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x - 1, subdomain_y - 1, subdomain_r ),
                    BoundaryEdge::E_11R,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
            else if (
                boundary_edge == BoundaryEdge::E_10R && subdomain_x < num_lateral_subdomains - 1 && subdomain_y > 0 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x + 1, subdomain_y - 1, subdomain_r ),
                    BoundaryEdge::E_01R,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
            else if (
                boundary_edge == BoundaryEdge::E_01R && subdomain_x > 0 && subdomain_y < num_lateral_subdomains - 1 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x - 1, subdomain_y + 1, subdomain_r ),
                    BoundaryEdge::E_10R,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
            else if (
                boundary_edge == BoundaryEdge::E_11R && subdomain_x < num_lateral_subdomains - 1 &&
                subdomain_y < num_lateral_subdomains - 1 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x + 1, subdomain_y + 1, subdomain_r ),
                    BoundaryEdge::E_00R,
                    BoundaryDirection::FORWARD,
                    -1 );
            }

            // Edges in Y direction

            else if ( boundary_edge == BoundaryEdge::E_0Y0 && subdomain_x > 0 && subdomain_r > 0 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x - 1, subdomain_y, subdomain_r - 1 ),
                    BoundaryEdge::E_1Y1,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
            else if (
                boundary_edge == BoundaryEdge::E_1Y0 && subdomain_x < num_lateral_subdomains - 1 && subdomain_r > 0 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x + 1, subdomain_y, subdomain_r - 1 ),
                    BoundaryEdge::E_0Y1,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
            else if (
                boundary_edge == BoundaryEdge::E_0Y1 && subdomain_x > 0 && subdomain_r < num_radial_subdomains - 1 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x - 1, subdomain_y, subdomain_r + 1 ),
                    BoundaryEdge::E_1Y0,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
            else if (
                boundary_edge == BoundaryEdge::E_1Y1 && subdomain_x < num_lateral_subdomains - 1 &&
                subdomain_r < num_radial_subdomains - 1 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x + 1, subdomain_y, subdomain_r + 1 ),
                    BoundaryEdge::E_0Y0,
                    BoundaryDirection::FORWARD,
                    -1 );
            }

            // Radial (Y fixed)

            else if ( boundary_edge == BoundaryEdge::E_X00 && subdomain_y > 0 && subdomain_r > 0 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x, subdomain_y - 1, subdomain_r - 1 ),
                    BoundaryEdge::E_X11,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
            else if (
                boundary_edge == BoundaryEdge::E_X10 && subdomain_y < num_lateral_subdomains - 1 && subdomain_r > 0 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x, subdomain_y + 1, subdomain_r - 1 ),
                    BoundaryEdge::E_X01,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
            else if (
                boundary_edge == BoundaryEdge::E_X01 && subdomain_y > 0 && subdomain_r < num_radial_subdomains - 1 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x, subdomain_y - 1, subdomain_r + 1 ),
                    BoundaryEdge::E_X10,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
            else if (
                boundary_edge == BoundaryEdge::E_X11 && subdomain_y < num_lateral_subdomains - 1 &&
                subdomain_r < num_radial_subdomains - 1 )
            {
                neighborhood_edge_[boundary_edge].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x, subdomain_y + 1, subdomain_r + 1 ),
                    BoundaryEdge::E_X00,
                    BoundaryDirection::FORWARD,
                    -1 );
            }
        }
    }

    void setup_neighborhood_vertices( const DomainInfo& domain_info, const SubdomainInfo& subdomain_info )
    {
        const int diamond_id  = subdomain_info.diamond_id();
        const int subdomain_x = subdomain_info.subdomain_x();
        const int subdomain_y = subdomain_info.subdomain_y();
        const int subdomain_r = subdomain_info.subdomain_r();

        const int num_lateral_subdomains = domain_info.num_subdomains_per_diamond_side();
        const int num_radial_subdomains  = domain_info.num_subdomains_in_radial_direction();

        for ( const auto boundary_vertex : all_boundary_vertices )
        {
            // Across diamond boundaries
            if ( diamond_id >= 0 && diamond_id <= 4 )
            {
                {
                    // north pole
                    if ( boundary_vertex == BoundaryVertex::V_000 && subdomain_x == 0 && subdomain_y == 0 &&
                         subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 2 ) % 5, 0, 0, subdomain_r - 1 ), BoundaryVertex::V_001, -1 );
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 3 ) % 5, 0, 0, subdomain_r - 1 ), BoundaryVertex::V_001, -1 );
                    }

                    if ( boundary_vertex == BoundaryVertex::V_001 && subdomain_x == 0 && subdomain_y == 0 &&
                         subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 2 ) % 5, 0, 0, subdomain_r + 1 ), BoundaryVertex::V_000, -1 );
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 3 ) % 5, 0, 0, subdomain_r + 1 ), BoundaryVertex::V_000, -1 );
                    }
                }

                {
                    // Northern hemisphere to the right.

                    if ( boundary_vertex == BoundaryVertex::V_000 && subdomain_x == 0 && subdomain_y > 0 &&
                         subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5, subdomain_y - 1, 0, subdomain_r - 1 ),
                            BoundaryVertex::V_101,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_001 && subdomain_x == 0 && subdomain_y > 0 &&
                        subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5, subdomain_y - 1, 0, subdomain_r + 1 ),
                            BoundaryVertex::V_100,
                            -1 );
                    }
                }

                {
                    // Northern hemisphere to the left.

                    if ( boundary_vertex == BoundaryVertex::V_000 && subdomain_x > 0 && subdomain_y == 0 &&
                         subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 4 ) % 5, 0, subdomain_x - 1, subdomain_r - 1 ),
                            BoundaryVertex::V_011,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_001 && subdomain_x > 0 && subdomain_y == 0 &&
                        subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 4 ) % 5, 0, subdomain_x - 1, subdomain_r + 1 ),
                            BoundaryVertex::V_010,
                            -1 );
                    }
                }

                {
                    // Northern hemisphere to the right

                    if ( boundary_vertex == BoundaryVertex::V_010 && subdomain_x == 0 &&
                         subdomain_y < num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5, subdomain_y + 1, 0, subdomain_r - 1 ),
                            BoundaryVertex::V_001,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_011 && subdomain_x == 0 &&
                        subdomain_y < num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5, subdomain_y + 1, 0, subdomain_r + 1 ),
                            BoundaryVertex::V_000,
                            -1 );
                    }
                }

                {
                    // Northern hemisphere to the right and southern

                    if ( boundary_vertex == BoundaryVertex::V_010 && subdomain_x > 0 &&
                         subdomain_y == num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                diamond_id + 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x,
                                subdomain_r - 1 ),
                            BoundaryVertex::V_101,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_011 && subdomain_x > 0 &&
                        subdomain_y == num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                diamond_id + 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x,
                                subdomain_r + 1 ),
                            BoundaryVertex::V_100,
                            -1 );
                    }
                }

                {
                    // Northern hemisphere to the left

                    if ( boundary_vertex == BoundaryVertex::V_100 && subdomain_y == 0 &&
                         subdomain_x < num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 4 ) % 5, 0, subdomain_x + 1, subdomain_r - 1 ),
                            BoundaryVertex::V_001,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_101 && subdomain_y == 0 &&
                        subdomain_x < num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 4 ) % 5, 0, subdomain_x + 1, subdomain_r + 1 ),
                            BoundaryVertex::V_000,
                            -1 );
                    }
                }

                {
                    // Northern hemisphere to the left and southern

                    if ( boundary_vertex == BoundaryVertex::V_100 && subdomain_y > 0 &&
                         subdomain_x == num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                ( diamond_id + 4 ) % 5 + 5,
                                num_lateral_subdomains - subdomain_y,
                                num_lateral_subdomains - 1,
                                subdomain_r - 1 ),
                            BoundaryVertex::V_011,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_101 && subdomain_y > 0 &&
                        subdomain_x == num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                ( diamond_id + 4 ) % 5 + 5,
                                num_lateral_subdomains - subdomain_y,
                                num_lateral_subdomains - 1,
                                subdomain_r + 1 ),
                            BoundaryVertex::V_010,
                            -1 );
                    }
                }

                {
                    // Northern hemi to southern (right)

                    if ( boundary_vertex == BoundaryVertex::V_110 && subdomain_x < num_lateral_subdomains - 1 &&
                         subdomain_y == num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                diamond_id + 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x - 2,
                                subdomain_r - 1 ),
                            BoundaryVertex::V_111,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_111 && subdomain_x < num_lateral_subdomains - 1 &&
                        subdomain_y == num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                diamond_id + 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x - 2,
                                subdomain_r + 1 ),
                            BoundaryVertex::V_110,
                            -1 );
                    }
                }

                {
                    // Northern hemi to southern (left)

                    if ( boundary_vertex == BoundaryVertex::V_110 && subdomain_y < num_lateral_subdomains - 1 &&
                         subdomain_x == num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                ( diamond_id + 4 ) % 5 + 5,
                                num_lateral_subdomains - subdomain_y - 2,
                                num_lateral_subdomains - 1,
                                subdomain_r - 1 ),
                            BoundaryVertex::V_111,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_111 && subdomain_y < num_lateral_subdomains - 1 &&
                        subdomain_x == num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                ( diamond_id + 4 ) % 5 + 5,
                                num_lateral_subdomains - subdomain_y - 2,
                                num_lateral_subdomains - 1,
                                subdomain_r + 1 ),
                            BoundaryVertex::V_110,
                            -1 );
                    }
                }
            }
            else if ( diamond_id >= 5 && diamond_id <= 9 )
            {
                {
                    // south pole
                    if ( boundary_vertex == BoundaryVertex::V_000 && subdomain_x == 0 && subdomain_y == 0 &&
                         subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 2 ) % 5 + 5, 0, 0, subdomain_r - 1 ),
                            BoundaryVertex::V_001,
                            -1 );
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 3 ) % 5 + 5, 0, 0, subdomain_r - 1 ),
                            BoundaryVertex::V_001,
                            -1 );
                    }

                    if ( boundary_vertex == BoundaryVertex::V_001 && subdomain_x == 0 && subdomain_y == 0 &&
                         subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 2 ) % 5 + 5, 0, 0, subdomain_r + 1 ),
                            BoundaryVertex::V_000,
                            -1 );
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 3 ) % 5 + 5, 0, 0, subdomain_r + 1 ),
                            BoundaryVertex::V_000,
                            -1 );
                    }
                }

                {
                    // Southern hemisphere to the right.

                    if ( boundary_vertex == BoundaryVertex::V_000 && subdomain_x == 0 && subdomain_y > 0 &&
                         subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, subdomain_y - 1, 0, subdomain_r - 1 ),
                            BoundaryVertex::V_101,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_001 && subdomain_x == 0 && subdomain_y > 0 &&
                        subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, subdomain_y - 1, 0, subdomain_r + 1 ),
                            BoundaryVertex::V_100,
                            -1 );
                    }
                }

                {
                    // Southern hemisphere to the left.

                    if ( boundary_vertex == BoundaryVertex::V_000 && subdomain_x > 0 && subdomain_y == 0 &&
                         subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 4 ) % 5 + 5, 0, subdomain_x - 1, subdomain_r - 1 ),
                            BoundaryVertex::V_011,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_001 && subdomain_x > 0 && subdomain_y == 0 &&
                        subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 4 ) % 5 + 5, 0, subdomain_x - 1, subdomain_r + 1 ),
                            BoundaryVertex::V_010,
                            -1 );
                    }
                }

                {
                    // Southern hemisphere to the right

                    if ( boundary_vertex == BoundaryVertex::V_010 && subdomain_x == 0 &&
                         subdomain_y < num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, subdomain_y + 1, 0, subdomain_r - 1 ),
                            BoundaryVertex::V_001,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_011 && subdomain_x == 0 &&
                        subdomain_y < num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, subdomain_y + 1, 0, subdomain_r + 1 ),
                            BoundaryVertex::V_000,
                            -1 );
                    }
                }

                {
                    // Southern hemisphere to the right and north

                    if ( boundary_vertex == BoundaryVertex::V_010 && subdomain_x > 0 &&
                         subdomain_y == num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                ( diamond_id + 1 ) % 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x,
                                subdomain_r - 1 ),
                            BoundaryVertex::V_101,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_011 && subdomain_x > 0 &&
                        subdomain_y == num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                ( diamond_id + 1 ) % 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x,
                                subdomain_r + 1 ),
                            BoundaryVertex::V_100,
                            -1 );
                    }
                }

                {
                    // Southern hemisphere to the left

                    if ( boundary_vertex == BoundaryVertex::V_100 && subdomain_y == 0 &&
                         subdomain_x < num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id - 1 ) % 5 + 5, 0, subdomain_x + 1, subdomain_r - 1 ),
                            BoundaryVertex::V_001,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_101 && subdomain_y == 0 &&
                        subdomain_x < num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo( ( diamond_id - 1 ) % 5 + 5, 0, subdomain_x + 1, subdomain_r + 1 ),
                            BoundaryVertex::V_000,
                            -1 );
                    }
                }

                {
                    // Southern hemisphere to the left and northern

                    if ( boundary_vertex == BoundaryVertex::V_100 && subdomain_y > 0 &&
                         subdomain_x == num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                diamond_id - 5,
                                num_lateral_subdomains - subdomain_y,
                                num_lateral_subdomains - 1,
                                subdomain_r - 1 ),
                            BoundaryVertex::V_011,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_101 && subdomain_y > 0 &&
                        subdomain_x == num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                diamond_id - 5,
                                num_lateral_subdomains - subdomain_y,
                                num_lateral_subdomains - 1,
                                subdomain_r + 1 ),
                            BoundaryVertex::V_010,
                            -1 );
                    }
                }

                {
                    // Southern hemi to northern (right)

                    if ( boundary_vertex == BoundaryVertex::V_110 && subdomain_x < num_lateral_subdomains - 1 &&
                         subdomain_y == num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                ( diamond_id - 4 ) % 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x - 2,
                                subdomain_r - 1 ),
                            BoundaryVertex::V_111,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_111 && subdomain_x < num_lateral_subdomains - 1 &&
                        subdomain_y == num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                ( diamond_id - 4 ) % 5,
                                num_lateral_subdomains - 1,
                                num_lateral_subdomains - subdomain_x - 2,
                                subdomain_r + 1 ),
                            BoundaryVertex::V_110,
                            -1 );
                    }
                }

                {
                    // Southern hemi to northern (left)
                    if ( boundary_vertex == BoundaryVertex::V_110 && subdomain_y < num_lateral_subdomains - 1 &&
                         subdomain_x == num_lateral_subdomains - 1 && subdomain_r > 0 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                diamond_id - 5,
                                num_lateral_subdomains - subdomain_y - 2,
                                num_lateral_subdomains - 1,
                                subdomain_r - 1 ),
                            BoundaryVertex::V_111,
                            -1 );
                    }
                    else if (
                        boundary_vertex == BoundaryVertex::V_111 && subdomain_y < num_lateral_subdomains - 1 &&
                        subdomain_x == num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
                    {
                        neighborhood_vertex_[boundary_vertex].emplace_back(
                            SubdomainInfo(
                                diamond_id - 5,
                                num_lateral_subdomains - subdomain_y - 2,
                                num_lateral_subdomains - 1,
                                subdomain_r + 1 ),
                            BoundaryVertex::V_110,
                            -1 );
                    }
                }
            }
            else
            {
                Kokkos::abort( "Invalid diamond ID." );
            }

            // Same diamond:

            if ( boundary_vertex == BoundaryVertex::V_000 && subdomain_x > 0 && subdomain_y > 0 && subdomain_r > 0 )
            {
                neighborhood_vertex_[boundary_vertex].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x - 1, subdomain_y - 1, subdomain_r - 1 ),
                    BoundaryVertex::V_111,
                    -1 );
            }
            else if (
                boundary_vertex == BoundaryVertex::V_100 && subdomain_x < num_lateral_subdomains - 1 &&
                subdomain_y > 0 && subdomain_r > 0 )
            {
                neighborhood_vertex_[boundary_vertex].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x + 1, subdomain_y - 1, subdomain_r - 1 ),
                    BoundaryVertex::V_011,
                    -1 );
            }
            else if (
                boundary_vertex == BoundaryVertex::V_010 && subdomain_x > 0 &&
                subdomain_y < num_lateral_subdomains - 1 && subdomain_r > 0 )
            {
                neighborhood_vertex_[boundary_vertex].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x - 1, subdomain_y + 1, subdomain_r - 1 ),
                    BoundaryVertex::V_101,
                    -1 );
            }
            else if (
                boundary_vertex == BoundaryVertex::V_001 && subdomain_x > 0 && subdomain_y > 0 &&
                subdomain_r < num_radial_subdomains - 1 )
            {
                neighborhood_vertex_[boundary_vertex].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x - 1, subdomain_y - 1, subdomain_r + 1 ),
                    BoundaryVertex::V_110,
                    -1 );
            }
            else if (
                boundary_vertex == BoundaryVertex::V_110 && subdomain_x < num_lateral_subdomains - 1 &&
                subdomain_y < num_lateral_subdomains - 1 && subdomain_r > 0 )
            {
                neighborhood_vertex_[boundary_vertex].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x + 1, subdomain_y + 1, subdomain_r - 1 ),
                    BoundaryVertex::V_001,
                    -1 );
            }
            else if (
                boundary_vertex == BoundaryVertex::V_101 && subdomain_x < num_lateral_subdomains - 1 &&
                subdomain_y > 0 && subdomain_r < num_radial_subdomains - 1 )
            {
                neighborhood_vertex_[boundary_vertex].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x + 1, subdomain_y - 1, subdomain_r + 1 ),
                    BoundaryVertex::V_010,
                    -1 );
            }
            else if (
                boundary_vertex == BoundaryVertex::V_011 && subdomain_x > 0 &&
                subdomain_y < num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
            {
                neighborhood_vertex_[boundary_vertex].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x - 1, subdomain_y + 1, subdomain_r + 1 ),
                    BoundaryVertex::V_100,
                    -1 );
            }
            else if (
                boundary_vertex == BoundaryVertex::V_111 && subdomain_x < num_lateral_subdomains - 1 &&
                subdomain_y < num_lateral_subdomains - 1 && subdomain_r < num_radial_subdomains - 1 )
            {
                neighborhood_vertex_[boundary_vertex].emplace_back(
                    SubdomainInfo( diamond_id, subdomain_x + 1, subdomain_y + 1, subdomain_r + 1 ),
                    BoundaryVertex::V_000,
                    -1 );
            }
        }
    }

    void setup_neighborhood_ranks(
        const DomainInfo&                          domain_info,
        const SubdomainToRankDistributionFunction& subdomain_to_rank )
    {
        for ( auto& neighbors : neighborhood_vertex_ | std::views::values )
        {
            for ( auto& [neighbor_subdomain_info, neighbor_boundary_vertex, neighbor_rank] : neighbors )
            {
                neighbor_rank = subdomain_to_rank(
                    neighbor_subdomain_info,
                    domain_info.num_subdomains_per_diamond_side(),
                    domain_info.num_subdomains_in_radial_direction() );
            }
        }

        for ( auto& neighbors : neighborhood_edge_ | std::views::values )
        {
            for ( auto& [neighbor_subdomain_info, neighbor_boundary_edge, _, neighbor_rank] : neighbors )
            {
                neighbor_rank = subdomain_to_rank(
                    neighbor_subdomain_info,
                    domain_info.num_subdomains_per_diamond_side(),
                    domain_info.num_subdomains_in_radial_direction() );
            }
        }

        for ( auto& [neighbor_subdomain_info, neighbor_boundary_face, _, neighbor_rank] :
              neighborhood_face_ | std::views::values )
        {
            neighbor_rank = subdomain_to_rank(
                neighbor_subdomain_info,
                domain_info.num_subdomains_per_diamond_side(),
                domain_info.num_subdomains_in_radial_direction() );
        }
    }

    void setup_neighborhood(
        const DomainInfo&                          domain_info,
        const SubdomainInfo&                       subdomain_info,
        const SubdomainToRankDistributionFunction& subdomain_to_rank )
    {
        setup_neighborhood_faces( domain_info, subdomain_info );
        setup_neighborhood_edges( domain_info, subdomain_info );
        setup_neighborhood_vertices( domain_info, subdomain_info );
        setup_neighborhood_ranks( domain_info, subdomain_to_rank );
    }

    std::map< BoundaryVertex, std::vector< NeighborSubdomainTupleVertex > > neighborhood_vertex_;
    std::map< BoundaryEdge, std::vector< NeighborSubdomainTupleEdge > >     neighborhood_edge_;
    std::map< BoundaryFace, NeighborSubdomainTupleFace >                    neighborhood_face_;
};

/// @brief Parallel data structure organizing the thick spherical shell metadata for distributed (MPI parallel)
///        simulations.
///
/// This is essentially a wrapper for the \ref DomainInfo and the neighborhood information (\ref SubdomainNeighborhood)
/// for all process-local subdomains.
class DistributedDomain
{
  public:
    DistributedDomain() = default;

    using LocalSubdomainIdx = int;

    /// @brief Creates a \ref DistributedDomain with a single subdomain per diamond and initializes all the subdomain
    ///        neighborhoods.
    static DistributedDomain create_uniform_single_subdomain_per_diamond(
        const int                                  lateral_diamond_refinement_level,
        const int                                  radial_diamond_refinement_level,
        const real_t                               r_min,
        const real_t                               r_max,
        const SubdomainToRankDistributionFunction& subdomain_to_rank = subdomain_to_rank_iterate_diamond_subdomains )
    {
        return create_uniform(
            lateral_diamond_refinement_level,
            uniform_shell_radii( r_min, r_max, ( 1 << radial_diamond_refinement_level ) + 1 ),
            0,
            0,
            subdomain_to_rank );
    }

    /// @brief Creates a \ref DistributedDomain with a single subdomain per diamond and initializes all the subdomain
    ///        neighborhoods.
    static DistributedDomain create_uniform_single_subdomain_per_diamond(
        const int                                  lateral_diamond_refinement_level,
        const std::vector< double >&               radii,
        const SubdomainToRankDistributionFunction& subdomain_to_rank = subdomain_to_rank_iterate_diamond_subdomains )
    {
        return create_uniform( lateral_diamond_refinement_level, radii, 0, 0, subdomain_to_rank );
    }

    /// @brief Creates a \ref DistributedDomain with a single subdomain per diamond and initializes all the subdomain
    ///        neighborhoods.
    static DistributedDomain create_uniform(
        const int                                  lateral_diamond_refinement_level,
        const int                                  radial_diamond_refinement_level,
        const real_t                               r_min,
        const real_t                               r_max,
        const int                                  lateral_subdomain_refinement_level,
        const int                                  radial_subdomain_refinement_level,
        const SubdomainToRankDistributionFunction& subdomain_to_rank = subdomain_to_rank_iterate_diamond_subdomains )
    {
        return create_uniform(
            lateral_diamond_refinement_level,
            uniform_shell_radii( r_min, r_max, ( 1 << radial_diamond_refinement_level ) + 1 ),
            lateral_subdomain_refinement_level,
            radial_subdomain_refinement_level,
            subdomain_to_rank );
    }

    /// @brief Creates a \ref DistributedDomain with a single subdomain per diamond and initializes all the subdomain
    ///        neighborhoods.
    static DistributedDomain create_uniform(
        const int                                  lateral_diamond_refinement_level,
        const std::vector< double >&               radii,
        const int                                  lateral_subdomain_refinement_level,
        const int                                  radial_subdomain_refinement_level,
        const SubdomainToRankDistributionFunction& subdomain_to_rank = subdomain_to_rank_iterate_diamond_subdomains )
    {
        DistributedDomain domain;
        domain.domain_info_ = DomainInfo(
            lateral_diamond_refinement_level,
            radii,
            ( 1 << lateral_subdomain_refinement_level ),
            ( 1 << radial_subdomain_refinement_level ) );
        int idx = 0;
        for ( const auto& subdomain : domain.domain_info_.local_subdomains( subdomain_to_rank ) )
        {
            domain.subdomains_[subdomain] = {
                idx, SubdomainNeighborhood( domain.domain_info_, subdomain, subdomain_to_rank ) };
            domain.local_subdomain_index_to_subdomain_info_[idx] = subdomain;
            idx++;
        }
        return domain;
    }

    /// @brief Returns a const reference
    [[nodiscard]] const DomainInfo& domain_info() const { return domain_info_; }

    [[nodiscard]] const std::map< SubdomainInfo, std::tuple< LocalSubdomainIdx, SubdomainNeighborhood > >&
        subdomains() const
    {
        return subdomains_;
    }

    [[nodiscard]] const SubdomainInfo& subdomain_info_from_local_idx( const LocalSubdomainIdx subdomain_idx ) const
    {
        return local_subdomain_index_to_subdomain_info_.at( subdomain_idx );
    }

  private:
    DomainInfo                                                                        domain_info_;
    std::map< SubdomainInfo, std::tuple< LocalSubdomainIdx, SubdomainNeighborhood > > subdomains_;
    std::map< LocalSubdomainIdx, SubdomainInfo > local_subdomain_index_to_subdomain_info_;
};

struct SubdomainDistribution
{
    int    total;
    int    min;
    int    max;
    double avg;
};

inline SubdomainDistribution subdomain_distribution( const DistributedDomain& domain )
{
    const auto num_local_subdomains = static_cast< int >( domain.subdomains().size() );
    int        total                = num_local_subdomains;
    int        min                  = num_local_subdomains;
    int        max                  = num_local_subdomains;

    MPI_Reduce( &num_local_subdomains, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
    MPI_Reduce( &num_local_subdomains, &min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD );
    MPI_Reduce( &num_local_subdomains, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD );

    const SubdomainDistribution result{
        .total = total, .min = min, .max = max, .avg = static_cast< double >( total ) / mpi::num_processes() };
    return result;
}

template < typename ValueType >
inline Grid4DDataScalar< ValueType >
    allocate_scalar_grid( const std::string label, const DistributedDomain& distributed_domain )
{
    return Grid4DDataScalar< ValueType >(
        label,
        distributed_domain.subdomains().size(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_radially() );
}

template < typename ValueType >
inline Grid4DDataScalar< ValueType >
    allocate_scalar_grid( const std::string label, const DistributedDomain& distributed_domain, const int level )
{
    return Grid4DDataScalar< ValueType >(
        label,
        distributed_domain.subdomains().size(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally( level ),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally( level ),
        distributed_domain.domain_info().subdomain_num_nodes_radially( level ) );
}

inline Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >
    local_domain_md_range_policy_nodes( const DistributedDomain& distributed_domain )
{
    return Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
        { 0, 0, 0, 0 },
        { static_cast< long long >( distributed_domain.subdomains().size() ),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
          distributed_domain.domain_info().subdomain_num_nodes_radially() } );
}

// loop only lateral dimensions of each subdomain. Used in the precomputation of lateral parts of the
// Jacobian (-> Oliver)
inline Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >
    local_domain_md_range_policy_cells_lateral( const DistributedDomain& distributed_domain )
{
    return Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
        { 0, 0, 0 },
        { static_cast< long long >( distributed_domain.subdomains().size() ),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1 } );
}

inline Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >
    local_domain_md_range_policy_cells( const DistributedDomain& distributed_domain )
{
    return Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
        { 0, 0, 0, 0 },
        { static_cast< long long >( distributed_domain.subdomains().size() ),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
          distributed_domain.domain_info().subdomain_num_nodes_radially() - 1 } );
}

// linearized Range instaed of MDRange to loop lateral and radial dimension,
// potentially yields a performance advantage.
inline Kokkos::RangePolicy<>
    local_domain_md_range_policy_cells_linearized( const DistributedDomain& distributed_domain )
{
    return Kokkos::RangePolicy<>(
        0,
        static_cast< long long >( distributed_domain.subdomains().size() ) *
            ( distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1 ) *
            ( distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1 ) *
            ( distributed_domain.domain_info().subdomain_num_nodes_radially() - 1 ) );
}

inline Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >
    local_domain_md_range_policy_cells_fv_skip_ghost_layers( const DistributedDomain& distributed_domain )
{
    return Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
        { 0, 1, 1, 1 },
        { static_cast< long long >( distributed_domain.subdomains().size() ),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1 - 1,
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1 - 1,
          distributed_domain.domain_info().subdomain_num_nodes_radially() + 1 - 1 } );
}

/// @brief Returns an initialized grid with the coordinates of all subdomains' nodes projected to the unit sphere.
///
/// The layout is
///
///     grid( local_subdomain_id, x_idx, y_idx, node_coord )
///
/// where node_coord is in {0, 1, 2} and refers to the cartesian coordinate of the point.
template < std::floating_point T >
Grid3DDataVec< T, 3 > subdomain_unit_sphere_single_shell_coords( const DistributedDomain& domain )
{
    Grid3DDataVec< T, 3 > subdomain_coords(
        "subdomain_unit_sphere_coords",
        domain.subdomains().size(),
        domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        domain.domain_info().subdomain_num_nodes_per_side_laterally() );

    typename Grid3DDataVec< T, 3 >::HostMirror subdomain_coords_host = Kokkos::create_mirror_view( subdomain_coords );

    for ( const auto& [subdomain_info, data] : domain.subdomains() )
    {
        const auto& [subdomain_idx, neighborhood] = data;

        unit_sphere_single_shell_subdomain_coords< T >(
            subdomain_coords_host,
            subdomain_idx,
            subdomain_info.diamond_id(),
            domain.domain_info().diamond_lateral_refinement_level(),
            domain.domain_info().num_subdomains_per_diamond_side(),
            subdomain_info.subdomain_x(),
            subdomain_info.subdomain_y() );
    }

    Kokkos::deep_copy( subdomain_coords, subdomain_coords_host );
    return subdomain_coords;
}

/// @brief Returns an initialized grid with the radii of all subdomain nodes.
///
/// The layout is
///
///     grid( local_subdomain_id, r_idx ) = radius
///
template < std::floating_point T >
Grid2DDataScalar< T > subdomain_shell_radii( const DistributedDomain& domain )
{
    const int shells_per_subdomain = domain.domain_info().subdomain_num_nodes_radially();
    const int layers_per_subdomain = shells_per_subdomain - 1;

    Grid2DDataScalar< T > radii_device( "subdomain_shell_radii", domain.subdomains().size(), shells_per_subdomain );
    typename Grid2DDataScalar< T >::HostMirror radii_host = Kokkos::create_mirror_view( radii_device );

    for ( const auto& [subdomain_info, data] : domain.subdomains() )
    {
        const auto& [subdomain_idx, neighborhood] = data;

        const int subdomain_innermost_node_idx = subdomain_info.subdomain_r() * layers_per_subdomain;
        const int subdomain_outermost_node_idx = subdomain_innermost_node_idx + layers_per_subdomain;

        int j = 0;
        for ( int node_idx = subdomain_innermost_node_idx; node_idx <= subdomain_outermost_node_idx; node_idx++ )
        {
            radii_host( subdomain_idx, j ) = domain.domain_info().radii()[node_idx];
            j++;
        }
    }

    Kokkos::deep_copy( radii_device, radii_host );
    return radii_device;
}

/// @brief Returns an initialized grid with the shell index of all subdomain nodes.
///
/// The layout is
///
///     grid( local_subdomain_id, r_idx ) = global_shell_idx
///
inline Grid2DDataScalar< int > subdomain_shell_idx( const DistributedDomain& domain )
{
    const int shells_per_subdomain = domain.domain_info().subdomain_num_nodes_radially();
    const int layers_per_subdomain = shells_per_subdomain - 1;

    Grid2DDataScalar< int > shell_idx_device( "subdomain_shell_idx", domain.subdomains().size(), shells_per_subdomain );
    Grid2DDataScalar< int >::HostMirror shell_idx_host = Kokkos::create_mirror_view( shell_idx_device );

    for ( const auto& [subdomain_info, data] : domain.subdomains() )
    {
        const auto& [subdomain_idx, neighborhood] = data;
        const int subdomain_innermost_node_idx    = subdomain_info.subdomain_r() * layers_per_subdomain;
        for ( int j = 0; j < shells_per_subdomain; j++ )
        {
            shell_idx_host( subdomain_idx, j ) = subdomain_innermost_node_idx + j;
        }
    }
    Kokkos::deep_copy( shell_idx_device, shell_idx_host );
    return shell_idx_device;
}

template < typename CoordsShellType, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION dense::Vec< typename CoordsShellType::value_type, 3 > coords(
    const int              subdomain,
    const int              x,
    const int              y,
    const int              r,
    const CoordsShellType& coords_shell,
    const CoordsRadiiType& coords_radii )
{
    using T = CoordsShellType::value_type;
    static_assert( std::is_same_v< T, typename CoordsRadiiType::value_type > );

    static_assert(
        std::is_same_v< CoordsShellType, Grid3DDataVec< T, 3 > > ||
        std::is_same_v< CoordsShellType, typename Grid3DDataVec< T, 3 >::HostMirror > );

    static_assert(
        std::is_same_v< CoordsRadiiType, Grid2DDataScalar< T > > ||
        std::is_same_v< CoordsRadiiType, typename Grid2DDataScalar< T >::HostMirror > );

    dense::Vec< T, 3 > coords;
    coords( 0 ) = coords_shell( subdomain, x, y, 0 );
    coords( 1 ) = coords_shell( subdomain, x, y, 1 );
    coords( 2 ) = coords_shell( subdomain, x, y, 2 );
    return coords * coords_radii( subdomain, r );
}

template < typename CoordsShellType, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION dense::Vec< typename CoordsShellType::value_type, 3 > coords(
    const dense::Vec< int, 4 > subdomain_x_y_r,
    const CoordsShellType&     coords_shell,
    const CoordsRadiiType&     coords_radii )
{
    using T = CoordsShellType::value_type;
    static_assert( std::is_same_v< T, typename CoordsRadiiType::value_type > );

    static_assert(
        std::is_same_v< CoordsShellType, Grid3DDataVec< T, 3 > > ||
        std::is_same_v< CoordsShellType, typename Grid3DDataVec< T, 3 >::HostMirror > );

    static_assert(
        std::is_same_v< CoordsRadiiType, Grid2DDataScalar< T > > ||
        std::is_same_v< CoordsRadiiType, typename Grid2DDataScalar< T >::HostMirror > );

    return coords(
        subdomain_x_y_r( 0 ),
        subdomain_x_y_r( 1 ),
        subdomain_x_y_r( 2 ),
        subdomain_x_y_r( 3 ),
        coords_shell,
        coords_radii );
}

} // namespace terra::grid::shell
