#pragma once

#include "dense/mat.hpp"
#include "dense/vec.hpp"
#include "grid/grid_types.hpp"
#include "kokkos/kokkos_wrapper.hpp"

#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"

/// @file
///
/// Point location and evaluation of Q1 wedge fields at arbitrary physical points.
///
/// Two evaluators are offered once a point has been located: \ref evaluate_q1_scalar / \ref evaluate_q1_vec use
/// the Q1 wedge shape functions of the containing cell and are second order, and \ref evaluate_cubic_scalar /
/// \ref evaluate_cubic_vec reconstruct the field with a tensor product of Hermite cubics over
/// the structured index stencil around the cell, which is third order. A semi-Lagrangian scheme commits its
/// interpolation error afresh at every timestep, where it shows up as numerical diffusion, so the cubic is what
/// the transport uses.
///
/// This is the geometric core of the semi-Lagrangian (MMOC) transport scheme. It exploits the fact that the
/// shell mesh is a tensor product of a laterally triangulated sphere and a set of radial shells, so that the
/// wedge geometric map
///
/// \f[
///    x(\xi, \eta, \zeta) = r(\zeta) \cdot \big[ (1 - \xi - \eta) p_1 + \xi p_2 + \eta p_3 \big],
///    \qquad r(\zeta) = r_1 + \tfrac{1}{2} (r_2 - r_1)(1 + \zeta)
/// \f]
///
/// (see \ref terra::fe::wedge::forward_map) can be inverted in **closed form**: with
/// \f$ A = [p_1 \, p_2 \, p_3] \f$ and \f$ \mu = A^{-1} x \f$ we get \f$ \rho = \mu_1 + \mu_2 + \mu_3 \f$ and
/// the barycentric coordinates \f$ \lambda = \mu / \rho \f$. The point lies in the wedge iff all
/// \f$ \lambda_i \geq 0 \f$ and \f$ r_1 \leq \rho \leq r_2 \f$.
///
/// Note that \f$ \lambda \f$ and \f$ \rho \f$ are **independent of the radial cell index**: the lateral search
/// and the radial search fully decouple. We therefore first walk laterally until the barycentric coordinates
/// are non-negative, then locate the radial cell by a search over the (monotone) shell radii.
///
/// @note \f$ \rho \neq |x| \f$ in general: the flat triangle spanned by \f$ p_1, p_2, p_3 \f$ is inscribed in
///       the unit sphere, so \f$ \rho = |x| (1 + \mathcal{O}(h^2)) \f$. The radial search must use \f$ \rho \f$.

namespace terra::fe::wedge::sl {

/// @brief A wedge cell inside one subdomain's node index space.
///
/// `(x, y)` is the hex cell index (spanning nodes `x, x+1` and `y, y+1`), `r` the radial cell index (spanning
/// nodes `r` and `r+1`), and `w` selects one of the two triangles the hex cell is split into:
///
/// \code
///   2--3
///   |\ |     w = 0: (p1, p2, p3) = ( (x,y), (x+1,y), (x,y+1) )
///   | \|     w = 1: (p1, p2, p3) = ( (x+1,y+1), (x,y+1), (x+1,y) )
///   0--1
/// \endcode
///
/// This matches the ordering produced by \ref terra::fe::wedge::wedge_surface_physical_coords.
struct WedgeCell
{
    int x = 0;
    int y = 0;
    int r = 0;
    int w = 0;
};

/// @brief Lateral node indices of the three triangle vertices of a wedge cell.
KOKKOS_INLINE_FUNCTION void wedge_lateral_node_indices( const WedgeCell& c, int ( &nx )[3], int ( &ny )[3] )
{
    if ( c.w == 0 )
    {
        nx[0] = c.x;     ny[0] = c.y;
        nx[1] = c.x + 1; ny[1] = c.y;
        nx[2] = c.x;     ny[2] = c.y + 1;
    }
    else
    {
        nx[0] = c.x + 1; ny[0] = c.y + 1;
        nx[1] = c.x;     ny[1] = c.y + 1;
        nx[2] = c.x + 1; ny[2] = c.y;
    }
}

/// @brief Moves to the wedge cell sharing the edge opposite to local vertex `v` (0, 1 or 2).
///
/// Every edge crossing flips the triangle parity. From `w == 0` the hex cell offsets are
/// `(0,0)`, `(-1,0)`, `(0,-1)`; from `w == 1` they are `(0,0)`, `(+1,0)`, `(0,+1)`.
KOKKOS_INLINE_FUNCTION WedgeCell wedge_edge_neighbor( const WedgeCell& c, const int v )
{
    WedgeCell n = c;
    n.w         = 1 - c.w;

    if ( c.w == 0 )
    {
        // v == 0: opposite edge is the hex diagonal -> other triangle of the same hex cell.
        if ( v == 1 )
            n.x = c.x - 1;
        else if ( v == 2 )
            n.y = c.y - 1;
    }
    else
    {
        if ( v == 1 )
            n.x = c.x + 1;
        else if ( v == 2 )
            n.y = c.y + 1;
    }

    return n;
}

/// @brief Unit-sphere position of one lateral node.
template < typename T, typename CoordsShellType >
KOKKOS_INLINE_FUNCTION dense::Vec< T, 3 > lateral_node_position(
    const int              subdomain,
    const int              x,
    const int              y,
    const CoordsShellType& coords_shell )
{
    dense::Vec< T, 3 > p;
    for ( int d = 0; d < 3; ++d )
    {
        p( d ) = coords_shell( subdomain, x, y, d );
    }
    return p;
}

/// @brief Cone coordinates of a physical point w.r.t. an arbitrary triangle of lateral nodes.
///
/// Solves \f$ A \mu = X \f$ with \f$ A = [p_1 \, p_2 \, p_3] \f$, so that \f$ \mu \f$ is the unique coefficient
/// vector with \f$ X = \mu_1 p_1 + \mu_2 p_2 + \mu_3 p_3 \f$ and `mu >= 0` is the cone containment test,
/// irrespective of how the triangle is wound.
template < typename T, typename CoordsShellType >
KOKKOS_INLINE_FUNCTION void triangle_cone_coords(
    const dense::Vec< T, 3 >& X,
    const int                 subdomain,
    const int ( &nx )[3],
    const int ( &ny )[3],
    const CoordsShellType& coords_shell,
    dense::Vec< T, 3 >&    mu )
{
    dense::Vec< T, 3 > p[3];
    for ( int v = 0; v < 3; ++v )
    {
        p[v] = lateral_node_position< T >( subdomain, nx[v], ny[v], coords_shell );
    }

    mu = dense::Mat< T, 3, 3 >::from_col_vecs( p[0], p[1], p[2] ).inv() * X;
}

/// @brief Cone coordinates of a physical point w.r.t. the triangle of a wedge cell.
///
/// Solves \f$ A \mu = X \f$ with \f$ A = [p_1 \, p_2 \, p_3] \f$ (unit-sphere vertices).
///
/// The **unnormalised** \f$ \mu \f$ is what the point location works with: the cone spanned by the triangle
/// is exactly \f$ \{ x : A^{-1} x \geq 0 \} \f$, so `mu >= 0` componentwise is the containment test, and the
/// most negative component identifies the edge that was crossed. By Cramer's rule
/// \f$ \mu_i \propto \det( \ldots, X, \ldots ) \f$, i.e. the signs of \f$ \mu \f$ are exactly the
/// great-circle side tests of the spherical triangle.
///
/// @warning Do **not** normalise to \f$ \lambda = \mu / \rho \f$ before the containment test. For triangles
///          more than 90 degrees away from the direction of \f$ X \f$ the cone radius
///          \f$ \rho = \sum_i \mu_i \f$ is negative, and dividing flips all three signs — a point outside
///          the triangle then looks like it is inside. A single icosahedral diamond spans roughly 127 degrees,
///          so this happens well within one subdomain. Normalise only after the walk has converged.
///
/// The result depends only on the *lateral* indices of `cell`; `cell.r` is ignored.
template < typename T, typename CoordsShellType >
KOKKOS_INLINE_FUNCTION void wedge_lateral_cone_coords(
    const dense::Vec< T, 3 >& X,
    const int                 subdomain,
    const WedgeCell&          cell,
    const CoordsShellType&    coords_shell,
    dense::Vec< T, 3 >&       mu )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    triangle_cone_coords( X, subdomain, nx, ny, coords_shell, mu );
}

/// @brief Adapts a `(subdomain, x, y, r, d)` coordinate view to the lateral `(subdomain, x, y, d)` accessor
///        expected here, by pinning the radial index.
///
/// The unit-sphere node directions do not depend on the radial index, but the ghosted geometry stores them in a
/// four-dimensional (radially replicated) view so that the lateral ghost exchange can transport them. Pin any
/// owned radial layer.
template < typename ViewType >
struct RadialSliceCoords
{
    ViewType view;
    int      r = 0;

    KOKKOS_INLINE_FUNCTION auto operator()( const int subdomain, const int x, const int y, const int d ) const
    {
        return view( subdomain, x, y, r, d );
    }
};

/// @brief True if all three lateral nodes of a wedge cell carry usable geometry.
///
/// See \ref terra::fe::wedge::sl::ghosted_lateral_validity: a handful of diagonal ghost corners are degenerate
/// at the pentagonal points of the icosahedral grid, and the wedges touching them must not be used.
template < typename LateralValidityType >
KOKKOS_INLINE_FUNCTION bool wedge_lateral_nodes_valid(
    const WedgeCell&           cell,
    const int                  subdomain,
    const LateralValidityType& lateral_valid )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    for ( int v = 0; v < 3; ++v )
    {
        if ( lateral_valid( subdomain, nx[v], ny[v] ) == 0 )
            return false;
    }
    return true;
}

/// @brief Bounds of the node index space a walk may visit.
///
/// For a plain (non-ghosted) subdomain field these are the subdomain node extents. For a ghosted field they
/// are the extents of the ghosted view, so that the walk may leave the owned region by up to the ghost width.
struct IndexBounds
{
    int num_nodes_x = 0;
    int num_nodes_y = 0;
    int num_nodes_r = 0;
};

/// @brief Outcome of a point location.
template < typename T >
struct LocateResult
{
    WedgeCell cell;
    T         xi   = 0;
    T         eta  = 0;
    T         zeta = 0;

    /// Point was located inside the index space.
    bool found = false;
    /// The lateral walk left the index space (departure point outside the local + ghost region).
    bool escaped_laterally = false;
    /// The point is radially outside the shell; `zeta` has been clamped to the outermost/innermost cell.
    bool clamped_radially = false;
    /// The walk reached a wedge touching a degenerate ghost node and stopped there.
    bool hit_invalid_node = false;
};

/// @brief Reference coordinates of the point of a wedge cell closest to `X`, for use when `X` itself could not
///        be located.
///
/// Projects onto the cell by clamping the barycentric coordinates onto the simplex and the cone radius onto the
/// cell's radial extent. Used as the fallback for a departure point that left the ghosted region: interpolating
/// at the nearest representable point is a bounded, first-order error, whereas skipping the advection entirely
/// leaves the node a whole timestep behind.
template < typename T, typename CoordsShellType, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION void clamp_to_wedge(
    const dense::Vec< T, 3 >& X,
    const int                 subdomain,
    const WedgeCell&          cell,
    const CoordsShellType&    coords_shell,
    const CoordsRadiiType&    coords_radii,
    T&                        xi,
    T&                        eta,
    T&                        zeta )
{
    dense::Vec< T, 3 > mu;
    wedge_lateral_cone_coords( X, subdomain, cell, coords_shell, mu );

    T sum = T( 0 );
    for ( int v = 0; v < 3; ++v )
    {
        mu( v ) = Kokkos::max( mu( v ), T( 0 ) );
        sum += mu( v );
    }
    if ( sum <= T( 0 ) )
    {
        xi = eta = T( 1 ) / T( 3 );
        zeta     = T( 0 );
        return;
    }

    const T rho = mu( 0 ) + mu( 1 ) + mu( 2 );
    xi          = mu( 1 ) / sum;
    eta         = mu( 2 ) / sum;

    const T r1 = coords_radii( subdomain, cell.r );
    const T r2 = coords_radii( subdomain, cell.r + 1 );
    zeta       = Kokkos::clamp( T( 2 ) * ( rho - r1 ) / ( r2 - r1 ) - T( 1 ), T( -1 ), T( 1 ) );
}

/// @brief Locates a physical point in the wedge mesh of one subdomain, starting the search at `seed`.
///
/// Walks laterally across triangle edges towards the most negative barycentric coordinate, then locates the
/// radial cell. `max_lateral_steps` bounds the walk; it should be chosen as roughly `ceil(CFL) + 2`.
///
/// Points outside `[rho_clamp_min, rho_clamp_max]` are, if `clamp_radially` is set, pulled back onto that
/// interval (the physically correct closure for a no-penetration boundary) and `clamped_radially` is reported;
/// otherwise they are reported as not found. Pass the *physical* shell radii here -- for a ghosted radii array
/// these are not the array ends.
template < typename T, typename CoordsShellType, typename CoordsRadiiType, typename LateralValidityType >
KOKKOS_INLINE_FUNCTION LocateResult< T > locate_point(
    const dense::Vec< T, 3 >& X,
    const int                 subdomain,
    const WedgeCell&          seed,
    const CoordsShellType&    coords_shell,
    const CoordsRadiiType&    coords_radii,
    const IndexBounds&        bounds,
    const int                 max_lateral_steps,
    const T                   eps,
    const bool                clamp_radially,
    const T                   rho_clamp_min,
    const T                   rho_clamp_max,
    const LateralValidityType& lateral_valid )
{
    LocateResult< T > result;

    WedgeCell          cell = seed;
    dense::Vec< T, 3 > mu;
    T                  rho = T( 0 );

    // ---- lateral walk -------------------------------------------------------------------------------------
    // Visibility walk on the spherical triangulation: step across the edge opposite the most negative cone
    // coordinate until all three are non-negative.
    bool lateral_found = false;

    const T scale = X.norm();

    // The seed must be usable; callers are expected to seed with a wholly owned cell.
    if ( !wedge_lateral_nodes_valid( cell, subdomain, lateral_valid ) )
    {
        result.cell              = cell;
        result.escaped_laterally = true;
        result.hit_invalid_node  = true;
        return result;
    }

    for ( int step = 0; step <= max_lateral_steps; ++step )
    {
        wedge_lateral_cone_coords( X, subdomain, cell, coords_shell, mu );

        // Order the three candidate edge crossings by how far outside the triangle the point is.
        int order[3] = { 0, 1, 2 };
        for ( int a = 0; a < 2; ++a )
        {
            for ( int b = a + 1; b < 3; ++b )
            {
                if ( mu( order[b] ) < mu( order[a] ) )
                {
                    const int t = order[a];
                    order[a]    = order[b];
                    order[b]    = t;
                }
            }
        }

        if ( mu( order[0] ) >= -eps * scale )
        {
            rho           = mu( 0 ) + mu( 1 ) + mu( 2 );
            lateral_found = true;
            break;
        }

        // Prefer the edge we are furthest outside of, but fall back to the others rather than giving up: a
        // neighbour may be out of bounds or touch a degenerate ghost corner while another still leads to the
        // wedge that contains the point.
        bool moved = false;
        for ( int k = 0; k < 3 && !moved; ++k )
        {
            if ( mu( order[k] ) >= -eps * scale )
                break; // not outside this edge; crossing it would move away from the point

            const WedgeCell next = wedge_edge_neighbor( cell, order[k] );

            // The hex cell (x, y) spans nodes x..x+1 and y..y+1, so the last cell index is num_nodes - 2.
            if ( next.x < 0 || next.y < 0 || next.x > bounds.num_nodes_x - 2 || next.y > bounds.num_nodes_y - 2 )
                continue;

            if ( !wedge_lateral_nodes_valid( next, subdomain, lateral_valid ) )
            {
                result.hit_invalid_node = true;
                continue;
            }

            cell  = next;
            moved = true;
        }

        if ( !moved )
        {
            result.cell              = cell;
            result.escaped_laterally = true;
            return result;
        }
    }

    if ( !lateral_found )
    {
        result.cell              = cell;
        result.escaped_laterally = true;
        return result;
    }

    // ---- radial search ------------------------------------------------------------------------------------
    // coords_radii( subdomain, . ) is monotonically increasing.
    const int num_cells_r = bounds.num_nodes_r - 1;

    // The clamp bounds are the *physical* shell radii, which for a ghosted radii array differ from the array
    // ends: radial ghosts beyond the CMB / surface are extrapolated so that the array stays monotone, but a
    // departure point must never be interpolated there.
    const T r_inner = rho_clamp_min;
    const T r_outer = rho_clamp_max;

    if ( rho < r_inner )
    {
        if ( !clamp_radially )
        {
            result.cell = cell;
            return result;
        }
        rho                      = r_inner;
        result.clamped_radially = true;
    }
    else if ( rho > r_outer )
    {
        if ( !clamp_radially )
        {
            result.cell = cell;
            return result;
        }
        rho                      = r_outer;
        result.clamped_radially = true;
    }

    int lo = 0;
    int hi = num_cells_r - 1;
    while ( lo < hi )
    {
        const int mid = ( lo + hi + 1 ) / 2;
        if ( coords_radii( subdomain, mid ) <= rho )
            lo = mid;
        else
            hi = mid - 1;
    }
    cell.r = lo;

    const T r1 = coords_radii( subdomain, cell.r );
    const T r2 = coords_radii( subdomain, cell.r + 1 );

    const T inv_rho = T( 1 ) / ( mu( 0 ) + mu( 1 ) + mu( 2 ) );

    result.cell  = cell;
    result.xi    = mu( 1 ) * inv_rho;
    result.eta   = mu( 2 ) * inv_rho;
    result.zeta  = T( 2 ) * ( rho - r1 ) / ( r2 - r1 ) - T( 1 );
    result.found = true;

    return result;
}

/// @brief Radial cell and reference coordinate of a point at *true* radius `radius`.
///
/// \ref locate_point reports its radial coordinate as \f$ \rho = \sum_k \mu_k \f$, the radius of the point in
/// the wedge's own parametrisation. That is the exact inverse of the geometric map, but it is not the point's
/// physical radius: the map sends \f$ (\xi, \eta) \f$ onto the *chord* triangle spanned by three unit nodes, so
/// \f$ |x| = \rho \, c \f$ with \f$ c = |\sum_k \lambda_k p_k| \le 1 \f$, and \f$ \rho \ge |x| \f$ with equality
/// only at a node.
///
/// The nodal values a semi-Lagrangian step interpolates between sit on true spheres, so evaluating the radial
/// direction at \f$ \rho \f$ samples the field *outside* the point. The bias is one-signed, and a scheme that
/// commits it once per timestep marches a radially stratified field inwards: on the level-5 rotation test,
/// where a rigid rotation preserves every radius exactly, \f$ \langle 1/c - 1 \rangle = 1.6 \times 10^{-4} \f$
/// over the cell interiors, and the cone's mass-weighted mean radius fell from 0.745 to 0.675 over one
/// revolution -- with Q1 evaluation and with the cubic alike, since both take their radial coordinate from
/// \ref locate_point. Re-deriving the radial cell and \f$ \zeta \f$ from \f$ |x| \f$ removes it.
///
/// The velocity is deliberately left on the parametric coordinate: the wedge map is linear in \f$ x \f$, so Q1
/// reproduces a linear velocity field there to round-off, and the trajectory is what that accuracy buys.
template < typename T, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION void radial_coords_from_radius(
    const int              subdomain,
    const T                radius,
    const CoordsRadiiType& coords_radii,
    const int              num_cells_r,
    const T                r_min,
    const T                r_max,
    int&                   cell_r,
    T&                     zeta )
{
    const T r = Kokkos::clamp( radius, r_min, r_max );

    int lo = 0;
    int hi = num_cells_r - 1;
    while ( lo < hi )
    {
        const int mid = ( lo + hi + 1 ) / 2;
        if ( coords_radii( subdomain, mid ) <= r )
            lo = mid;
        else
            hi = mid - 1;
    }
    cell_r = lo;

    const T r1 = coords_radii( subdomain, cell_r );
    const T r2 = coords_radii( subdomain, cell_r + 1 );
    zeta       = T( 2 ) * ( r - r1 ) / ( r2 - r1 ) - T( 1 );
}

/// @brief Lateral index rectangle whose four corner nodes span the spherical quad used by the O(1) predictor.
///
/// The corners are the nodes `(x0, y0)`, `(x1, y0)`, `(x0, y1)` and `(x1, y1)`. They must carry valid geometry,
/// so for a ghosted field pass the *owned* corners rather than the corners of the ghosted view: the predictor
/// extrapolates happily into the ghost layer, but the degenerate diagonal ghost corners would poison the map.
struct LateralCornerBox
{
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;
    int y1 = 0;
};

/// @brief Corner box of a plain (non-ghosted) index space, i.e. the whole subdomain.
KOKKOS_INLINE_FUNCTION LateralCornerBox corner_box_from_bounds( const IndexBounds& bounds )
{
    return LateralCornerBox{ 0, 0, bounds.num_nodes_x - 1, bounds.num_nodes_y - 1 };
}

/// @brief Hex cell and triangle containing the continuous lateral index coordinates `(u, v)`.
///
/// `(u, v)` is clamped into the index space, so the result is always a representable cell. The triangle is
/// picked by the same anti-diagonal split the mesh uses inside a hex cell: `w = 0` for the lower-left half.
template < typename T >
KOKKOS_INLINE_FUNCTION WedgeCell lateral_cell_from_index_coords( const T u, const T v, const IndexBounds& bounds )
{
    // The hex cell (x, y) spans nodes x..x+1, so the last cell index is num_nodes - 2.
    const int max_cx = bounds.num_nodes_x - 2;
    const int max_cy = bounds.num_nodes_y - 2;

    WedgeCell cell;
    cell.x = Kokkos::clamp( static_cast< int >( Kokkos::floor( u ) ), 0, max_cx );
    cell.y = Kokkos::clamp( static_cast< int >( Kokkos::floor( v ) ), 0, max_cy );
    cell.r = 0;

    // Where (u, v) was clamped the fractional parts leave [0, 1]; the comparison then simply picks the triangle
    // of the boundary cell that faces the point, which is the closest representable answer.
    const T fx = u - static_cast< T >( cell.x );
    const T fy = v - static_cast< T >( cell.y );
    cell.w     = ( fx + fy <= T( 1 ) ) ? 0 : 1;

    return cell;
}

/// @brief Outcome of the O(1) index prediction.
template < typename T >
struct CellPrediction
{
    /// Continuous lateral index coordinates of the point (not clamped).
    T u = 0;
    T v = 0;
    /// `(u, v)` clamped into the index space and turned into a cell.
    WedgeCell cell;
    /// The point lies in neither half of the corner quad, so `cell` is only a nearest-corner guess.
    bool outside_quad = false;
};

/// @brief O(1) estimate of the lateral index coordinates of a physical point, without any walk.
///
/// The quad spanned by `box` (a whole diamond, in the one-subdomain-per-diamond case) is split along its
/// anti-diagonal into the two spherical triangles \f$ \{ c_{00}, c_{10}, c_{01} \} \f$ and
/// \f$ \{ c_{11}, c_{01}, c_{10} \} \f$ — the same split orientation the mesh uses inside every hex cell. The
/// cone coordinates w.r.t. those corners say which half the point is in (all components non-negative, see the
/// warning on \ref wedge_lateral_cone_coords about testing before normalising) and, once normalised to
/// barycentric coordinates \f$ \lambda \f$, where in that half:
///
/// \f[
///     u = x_0 + \lambda_1 (x_1 - x_0), \qquad v = y_0 + \lambda_2 (y_1 - y_0)
/// \f]
///
/// for the first triangle, and the mirrored expression \f$ u = x_1 - \lambda_1 (x_1 - x_0) \f$,
/// \f$ v = y_1 - \lambda_2 (y_1 - y_0) \f$ for the second. The cost is one or two 3x3 solves and does not
/// depend on the refinement level.
///
/// @note This is an **estimate**, not the answer. Mapping barycentric coordinates linearly onto index space is
///       exact only for a grid obtained by gnomonic (central) projection of a uniformly subdivided planar
///       triangle. The shell grid is instead built by recursive normalised bisection (see
///       \ref terra::grid::shell::compute_node_recursive), which places nodes at equal *angles* along the
///       diamond edges. Both agree at the corners and at the edge midpoints and disagree in between: along an
///       edge of angular width \f$ \Theta \f$ the barycentric coordinate of the node at angle \f$ \theta \f$ is
///       \f$ \sin\theta / (\sin\theta + \sin(\Theta - \theta)) \f$ rather than \f$ \theta / \Theta \f$, which
///       peaks at about 2% of the edge — a fraction of a cell at low refinement, but \f$ 0.02 N \f$ cells at
///       production resolutions. \ref locate_point_direct corrects for this against the real node positions.
template < typename T, typename CoordsShellType >
KOKKOS_INLINE_FUNCTION CellPrediction< T > predict_lateral_cell(
    const dense::Vec< T, 3 >& X,
    const int                 subdomain,
    const CoordsShellType&    coords_shell,
    const LateralCornerBox&   box,
    const IndexBounds&        bounds,
    const T                   eps )
{
    const T tol = eps * X.norm();

    // Triangle A = ( c00, c10, c01 ), triangle B = ( c11, c01, c10 ).
    const int ax[3] = { box.x0, box.x1, box.x0 };
    const int ay[3] = { box.y0, box.y0, box.y1 };
    const int bx[3] = { box.x1, box.x0, box.x1 };
    const int by[3] = { box.y1, box.y1, box.y0 };

    dense::Vec< T, 3 > mu_a;
    triangle_cone_coords( X, subdomain, ax, ay, coords_shell, mu_a );
    const T min_a = Kokkos::min( mu_a( 0 ), Kokkos::min( mu_a( 1 ), mu_a( 2 ) ) );

    CellPrediction< T > pred;

    bool               in_a = min_a >= -tol;
    dense::Vec< T, 3 > mu   = mu_a;

    if ( !in_a )
    {
        dense::Vec< T, 3 > mu_b;
        triangle_cone_coords( X, subdomain, bx, by, coords_shell, mu_b );
        const T min_b = Kokkos::min( mu_b( 0 ), Kokkos::min( mu_b( 1 ), mu_b( 2 ) ) );

        if ( min_b >= -tol )
        {
            mu = mu_b;
        }
        else
        {
            // Outside the quad altogether. Fall back to the triangle the point is least far outside of and let
            // the caller's clamping produce a nearest-corner guess.
            pred.outside_quad = true;
            in_a              = min_a >= min_b;
            mu                = in_a ? mu_a : mu_b;
        }
    }

    const T rho = mu( 0 ) + mu( 1 ) + mu( 2 );
    const T dx  = static_cast< T >( box.x1 - box.x0 );
    const T dy  = static_cast< T >( box.y1 - box.y0 );

    if ( rho <= T( 0 ) )
    {
        // More than 90 degrees away from the triangle: the barycentric coordinates are meaningless (see the
        // warning on wedge_lateral_cone_coords). Aim at the middle of the box instead.
        pred.outside_quad = true;
        pred.u            = T( 0.5 ) * static_cast< T >( box.x0 + box.x1 );
        pred.v            = T( 0.5 ) * static_cast< T >( box.y0 + box.y1 );
    }
    else
    {
        const T inv = T( 1 ) / rho;
        const T l1  = mu( 1 ) * inv;
        const T l2  = mu( 2 ) * inv;

        pred.u = in_a ? static_cast< T >( box.x0 ) + l1 * dx : static_cast< T >( box.x1 ) - l1 * dx;
        pred.v = in_a ? static_cast< T >( box.y0 ) + l2 * dy : static_cast< T >( box.y1 ) - l2 * dy;
    }

    pred.cell = lateral_cell_from_index_coords( pred.u, pred.v, bounds );
    return pred;
}

/// @brief Locates a physical point without a long walk: O(1) barycentric prediction plus affine refinement.
///
/// A drop-in alternative to \ref locate_point that needs no seed and whose cost does not grow with the
/// refinement level. Three stages:
///
///   1. \ref predict_lateral_cell gives a cell in O(1) from the corner quad.
///   2. That estimate is refined against the actual mesh. At the current cell the barycentric coordinates of
///      the point form an affine model of the index-space map, and evaluating that model *outside* the cell
///      extrapolates straight at the cell the point should be in:
///      \f$ (u, v) = \sum_k \lambda_k \, (n^x_k, n^y_k) \f$ over the cell's three lateral nodes. Since the map
///      from index space to the sphere is smooth, the extrapolation error over \f$ \Delta \f$ cells is
///      \f$ \mathcal{O}( \Delta^2 \Theta / N ) \f$ cells, so a single jump absorbs essentially all of the
///      bisection-versus-gnomonic mismatch of stage 1 and `max_refinements = 2` is ample.
///   3. \ref locate_point finishes from that cell with a small step budget. Going through the walk (rather
///      than trusting stage 2) is what makes the result *identical* to a full walk, including the escape,
///      radial clamping and node-validity reporting.
///
/// `max_walk_steps` therefore only has to absorb the residual of stage 2 — a cell or two — instead of the
/// diameter of the subdomain. All other arguments have the same meaning as in \ref locate_point.
template < typename T, typename CoordsShellType, typename CoordsRadiiType, typename LateralValidityType >
KOKKOS_INLINE_FUNCTION LocateResult< T > locate_point_direct(
    const dense::Vec< T, 3 >& X,
    const int                 subdomain,
    const CoordsShellType&    coords_shell,
    const CoordsRadiiType&    coords_radii,
    const LateralCornerBox&   box,
    const IndexBounds&        bounds,
    const int                 max_refinements,
    const int                 max_walk_steps,
    const T                   eps,
    const bool                clamp_radially,
    const T                   rho_clamp_min,
    const T                   rho_clamp_max,
    const LateralValidityType& lateral_valid )
{
    const T tol = eps * X.norm();

    WedgeCell cell = predict_lateral_cell( X, subdomain, coords_shell, box, bounds, eps ).cell;

    // locate_point() refuses to start from a cell touching a degenerate ghost corner, so retreat to the middle
    // of the corner box, which is owned geometry by construction.
    if ( !wedge_lateral_nodes_valid( cell, subdomain, lateral_valid ) )
    {
        cell = lateral_cell_from_index_coords(
            T( 0.5 ) * static_cast< T >( box.x0 + box.x1 ), T( 0.5 ) * static_cast< T >( box.y0 + box.y1 ), bounds );
    }

    for ( int it = 0; it < max_refinements; ++it )
    {
        dense::Vec< T, 3 > mu;
        wedge_lateral_cone_coords( X, subdomain, cell, coords_shell, mu );

        if ( Kokkos::min( mu( 0 ), Kokkos::min( mu( 1 ), mu( 2 ) ) ) >= -tol )
            break; // already the containing cell; the walk below will confirm it in one step

        const T rho = mu( 0 ) + mu( 1 ) + mu( 2 );
        if ( rho <= T( 0 ) )
            break; // barycentric coordinates unusable here; leave it to the walk

        int nx[3], ny[3];
        wedge_lateral_node_indices( cell, nx, ny );

        const T inv = T( 1 ) / rho;
        T       u   = T( 0 );
        T       v   = T( 0 );
        for ( int k = 0; k < 3; ++k )
        {
            const T lambda = mu( k ) * inv;
            u += lambda * static_cast< T >( nx[k] );
            v += lambda * static_cast< T >( ny[k] );
        }

        const WedgeCell next = lateral_cell_from_index_coords( u, v, bounds );
        if ( next.x == cell.x && next.y == cell.y && next.w == cell.w )
            break; // fixed point: the remaining error is below one cell

        if ( !wedge_lateral_nodes_valid( next, subdomain, lateral_valid ) )
            break; // keep the last usable cell and let the walk deal with the neighbourhood

        cell = next;
    }

    return locate_point(
        X, subdomain, cell, coords_shell, coords_radii, bounds, max_walk_steps, eps, clamp_radially,
        rho_clamp_min, rho_clamp_max, lateral_valid );
}

/// @brief Evaluates a Q1 scalar wedge field at reference coordinates inside a wedge cell.
template < typename T, typename FieldViewType >
KOKKOS_INLINE_FUNCTION T evaluate_q1_scalar(
    const FieldViewType& field,
    const int            subdomain,
    const WedgeCell&     cell,
    const T              xi,
    const T              eta,
    const T              zeta )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    T value = T( 0 );
    for ( int j = 0; j < num_nodes_per_wedge; ++j )
    {
        const int lateral = j % 3;
        const int radial  = j / 3;
        value += field( subdomain, nx[lateral], ny[lateral], cell.r + radial ) * shape_lat( j, xi, eta ) *
                 shape_rad( j, zeta );
    }
    return value;
}

/// @brief Evaluates a Q1 vector wedge field at reference coordinates inside a wedge cell.
template < typename T, int VecDim, typename FieldViewType >
KOKKOS_INLINE_FUNCTION dense::Vec< T, VecDim > evaluate_q1_vec(
    const FieldViewType& field,
    const int            subdomain,
    const WedgeCell&     cell,
    const T              xi,
    const T              eta,
    const T              zeta )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    dense::Vec< T, VecDim > value;
    for ( int d = 0; d < VecDim; ++d )
        value( d ) = T( 0 );

    for ( int j = 0; j < num_nodes_per_wedge; ++j )
    {
        const int lateral = j % 3;
        const int radial  = j / 3;
        const T   weight  = shape_lat( j, xi, eta ) * shape_rad( j, zeta );

        for ( int d = 0; d < VecDim; ++d )
        {
            value( d ) += field( subdomain, nx[lateral], ny[lateral], cell.r + radial, d ) * weight;
        }
    }
    return value;
}

/// @brief Number of nodes in a full cubic interpolation stencil.
inline constexpr int cubic_stencil_size = 4;

/// @brief Number of nodes in a full quintic interpolation stencil.
inline constexpr int quintic_stencil_size = 6;

/// @brief Widest stencil any evaluator may ask for; sizes the per-thread scratch arrays.
inline constexpr int max_stencil_size = quintic_stencil_size;

/// @brief Inclusive node index range that an interpolation stencil may read from, in one direction.
struct StencilRange
{
    int lo = 0;
    int hi = 0;
};

/// @brief Node index ranges the cubic interpolation stencil may read from.
///
/// Both lateral ranges should be the **owned** block, not the ghosted index space. A lateral ghost row does
/// carry a neighbour's real values, but it belongs to a different diamond of the icosahedral grid, and the map
/// from index space to the sphere has a kink at that seam: the neighbour's index directions meet ours at an
/// angle. A stencil straddling the seam therefore estimates its slopes across a corner of the parametrisation
/// and is not merely less accurate but inconsistent — measured at level 4 it costs about 1e-1 on a field the
/// Q1 evaluation reproduces to round-off. Confining the stencil to one diamond keeps the parametrisation
/// smooth; foot points that land outside the owned block fall back to Q1.
///
/// The radial range must be trimmed for a different reason: the radial ghost layers outside the CMB and the
/// surface hold extrapolated *radii* and no field data at all (see
/// \ref terra::fe::wedge::sl::ghosted_shell_radii), and reading them would poison the interpolation of every
/// node in the boundary layer.
struct StencilBounds
{
    StencilRange x;
    StencilRange y;
    StencilRange r;
};

/// @brief Stencil bounds spanning a whole index space, with no direction trimmed.
KOKKOS_INLINE_FUNCTION StencilBounds full_stencil_bounds( const IndexBounds& bounds )
{
    return StencilBounds{ { 0, bounds.num_nodes_x - 1 },
                          { 0, bounds.num_nodes_y - 1 },
                          { 0, bounds.num_nodes_r - 1 } };
}

/// @brief Places an interpolation stencil of up to \ref cubic_stencil_size nodes around a cell.
///
/// The cell spans nodes `cell_index` and `cell_index + 1`, both of which are assumed to lie in `range`. The
/// stencil is centred on the cell (`base = cell_index - 1`) wherever there is room, and slid inwards where
/// there is not — near a boundary the interpolation becomes one-sided rather than reaching outside, which is
/// what keeps the evaluation point inside the stencil's span (no extrapolation) at the price of a larger error
/// constant. If fewer than four nodes are available the stencil shrinks to three (quadratic) or two (linear).
///
/// @return the number of stencil nodes, `base .. base + n - 1`.
KOKKOS_INLINE_FUNCTION int cubic_stencil_window( const int cell_index, const StencilRange& range, int& base )
{
    const int available = range.hi - range.lo + 1;
    // Compared rather than passed to Kokkos::min: binding a namespace-scope constexpr to a reference odr-uses
    // it, which nvcc rejects in device code.
    const int n         = ( available < cubic_stencil_size ) ? available : cubic_stencil_size;

    base = Kokkos::clamp( ( n >= 3 ) ? cell_index - 1 : cell_index, range.lo, range.hi - n + 1 );
    return n;
}

/// @brief Links from a subdomain's lateral ghost ring to the neighbour that *owns* that region.
///
/// Entry `(s, f)` is the local index of the subdomain across face `f` of subdomain `s`, in the order
/// `-x, +x, -y, +y`, or -1 where there is no usable link. See \ref canonicalise_lateral_cell for which
/// neighbours qualify.
struct LateralSubdomainLinks
{
    Kokkos::View< int* [4] > link;              ///< (local subdomain, face) -> local subdomain, or -1 if not resident
    Kokkos::View< int* [4] > same_diamond;      ///< (local subdomain, face) -> 1 if the neighbour shares the diamond
    int                      owned_nodes = 0;   ///< owned nodes per lateral side, i.e. num_nodes_ghosted - 2*ghost
    int                      ghost       = 1;   ///< ghost layer width the exchange fills

    KOKKOS_INLINE_FUNCTION bool valid() const { return link.extent( 0 ) > 0; }
};

/// @brief Re-expresses a located cell in the subdomain that owns it, when the walk ended in the ghost ring.
///
/// \ref locate_point never changes `subdomain`: it walks within one subdomain's ghosted index space, so a foot
/// point that crossed a seam is found in the *ghost ring*, where the neighbour's values were copied before the
/// kernel ran. That is enough to locate the point, but not to reconstruct well: the stencil is clamped to the
/// owned block, so the evaluation there is one-sided and lower order, while for the neighbour -- for whom the
/// same point is interior -- it would be centred. The two sides then disagree about a shared interface node,
/// and \ref MMOCTransport::step has to pick one.
///
/// Where the neighbour is resident on this device *and* belongs to the same diamond, the fix is a translation:
/// subdomains tile a diamond in a regular lateral grid with the seam nodes shared, so the index axes are
/// aligned and only the cell index shifts, by the owned side length minus one. The reference coordinates
/// \f$ \xi, \eta \f$ and the triangle index carry over untouched.
///
/// @note Three cases are deliberately **not** handled and leave the cell where it is, which reproduces the
///       behaviour this function was added to improve rather than anything worse:
///       - **Across a diamond seam.** The neighbour's index axes meet ours at an angle, so the remap is a
///         rotation or reflection rather than a translation. The transform is already known to the ghost
///         exchange -- `SubdomainNeighborhood::neighborhood_face()` yields it as `ordering` -- but it is
///         ambiguous at the twelve pentagonal points, where five diamonds meet and there is no consistent
///         four-neighbour continuation. Those corners are exactly the ones
///         \ref ghosted_lateral_validity already marks invalid.
///       - **A ghost corner at a diamond edge**, where the diagonal hop would leave the diamond on one of its
///         two legs. A corner interior to a diamond is handled: call this twice and it hops x then y.
///       - **A neighbour on another rank**, whose interior is not in this device's memory at all. No walk can
///         reach it; only a wider ghost layer would.
template < typename T >
KOKKOS_INLINE_FUNCTION bool canonicalise_lateral_cell( int&                         subdomain,
                                                       WedgeCell&                   cell,
                                                       const LateralSubdomainLinks& links,
                                                       const IndexBounds&           bounds )
{
    if ( !links.valid() )
        return false;

    const int shift  = links.owned_nodes - 1;
    const int last_x = bounds.num_nodes_x - 2;
    const int last_y = bounds.num_nodes_y - 2;

    const bool low_x  = cell.x < links.ghost;
    const bool high_x = cell.x > last_x - links.ghost;
    const bool low_y  = cell.y < links.ghost;
    const bool high_y = cell.y > last_y - links.ghost;

    if ( !low_x && !high_x && !low_y && !high_y )
        return false; // interior: nothing to canonicalise

    // One axis per call. A ghost *corner* sits in the ring in both directions at once and needs the diagonal
    // neighbour; calling this twice walks there in two hops, x first, provided both links exist.
    const int face      = low_x ? 0 : ( high_x ? 1 : ( low_y ? 2 : 3 ) );
    const int neighbour = links.link( subdomain, face );
    if ( neighbour < 0 )
        return false; // across a diamond seam, or not resident on this device

    if ( low_x )
        cell.x += shift;
    else if ( high_x )
        cell.x -= shift;
    else if ( low_y )
        cell.y += shift;
    else
        cell.y -= shift;

    subdomain = neighbour;
    return true;
}

/// @brief Places the widest *centred* stencil of at most `preferred` nodes around a cell.
///
/// \ref cubic_stencil_window slides its window inwards near a boundary, which keeps the evaluation inside the
/// stencil's span at the price of a one-sided cubic. That trade stops paying at higher order: a one-sided
/// quintic oscillates far more than a centred cubic costs, so above four nodes the window is never slid. The
/// width simply steps down -- six nodes where a centred six fits, four where a centred four fits, and below
/// that the sliding cubic/quadratic/linear ladder of \ref cubic_stencil_window, which is well behaved
/// one-sided.
///
/// On the icosahedral grid the reduced-width band is structural, not a shortage of data. The lateral ranges
/// are the owned block, because the index parametrisation kinks at the diamond seam and a stencil straddling
/// it is inconsistent (see \ref StencilBounds); widening the ghost layer would supply values across the seam
/// but not remove the kink. So the outer two rings of every diamond stay cubic: about 23% of the nodes at
/// level 5, 12% at level 6, plus the outermost two radial layers.
///
/// @return the number of stencil nodes, `base .. base + n - 1`.
KOKKOS_INLINE_FUNCTION int stencil_window( const int cell_index, const StencilRange& range, const int preferred,
                                           int& base )
{
    for ( int n = preferred; n > cubic_stencil_size; n -= 2 )
    {
        const int b = cell_index - ( n / 2 - 1 );
        if ( b >= range.lo && b + n - 1 <= range.hi )
        {
            base = b;
            return n;
        }
    }
    return cubic_stencil_window( cell_index, range, base );
}

/// @brief Widens lateral stencil bounds into the ghost ring wherever the neighbour is a same-diamond link.
///
/// The lateral bounds are the owned block because the index parametrisation kinks at a *diamond* seam. An
/// internal seam between two subdomains of the same diamond has no such kink -- it is an artefact of the
/// decomposition -- so a stencil may cross it, and with the ghost layer filled the nodes are there to read.
/// Without this the reduced-order band scales with the number of subdomains rather than with the geometry,
/// which is what made the result depend on how the mesh was partitioned.
KOKKOS_INLINE_FUNCTION StencilBounds widen_across_internal_seams( StencilBounds                bounds,
                                                                  const int                    subdomain,
                                                                  const LateralSubdomainLinks& links )
{
    if ( !links.valid() )
        return bounds;

    // Residency is irrelevant here: the ghost exchange fills the ring from whichever rank owns the neighbour,
    // so the nodes are present either way. What matters is only that there is no parametrisation kink to cross,
    // which is what `same_diamond` records. (canonicalise_lateral_cell *does* need residency, because it
    // indexes the neighbour's own array.)
    if ( links.same_diamond( subdomain, 0 ) )
        bounds.x.lo -= links.ghost;
    if ( links.same_diamond( subdomain, 1 ) )
        bounds.x.hi += links.ghost;
    if ( links.same_diamond( subdomain, 2 ) )
        bounds.y.lo -= links.ghost;
    if ( links.same_diamond( subdomain, 3 ) )
        bounds.y.hi += links.ghost;

    return bounds;
}

/// @brief Lagrange interpolation through `n` nodes, evaluated by Neville's algorithm.
///
/// The unique polynomial of degree `n - 1` through the stencil, so it reproduces any polynomial of that degree
/// in the interpolation variable exactly. Neville is used rather than barycentric weights because it needs no
/// precomputation and handles the non-uniform radial abscissae and the uniform lateral ones with the same
/// code; at these widths the O(n^2) recurrence is a handful of flops.
///
/// Unlike \ref pchip_interpolate_1d this is a single polynomial over the whole stencil rather than a Hermite
/// piece on the bracketing interval, so as the foot point crosses a cell boundary and the window shifts the
/// reconstruction changes: both windows pass through the shared node, so the result is continuous, but its
/// derivative is not. That is the usual trade for semi-Lagrangian transport, where the reconstruction is
/// consumed once per step, and boundedness comes from the caller's clip to the containing cell rather than
/// from the interpolant.
template < typename T >
KOKKOS_INLINE_FUNCTION T lagrange_interpolate_1d( const T* xs, const T* fs, const int n, const T xq )
{
    T p[max_stencil_size];
    for ( int i = 0; i < n; ++i )
        p[i] = fs[i];

    for ( int k = 1; k < n; ++k )
        for ( int i = 0; i < n - k; ++i )
            p[i] = ( ( xq - xs[i + k] ) * p[i] + ( xs[i] - xq ) * p[i + 1] ) / ( xs[i] - xs[i + k] );

    return p[0];
}

/// @brief Piecewise cubic Hermite interpolation through up to four nodes, optionally PCHIP-limited.
///
/// The nodal derivatives are the three-point parabolic estimates (the centred difference for equally spaced
/// nodes) with the usual one-sided parabola at the two ends, which makes the interpolant \f$ C^1 \f$ and third
/// order accurate — a decisive improvement over the second-order multilinear evaluation for a semi-Lagrangian
/// scheme, where the interpolation error is applied afresh every timestep and shows up as numerical diffusion.
///
/// With `monotone` set, the derivatives are then passed through the Fritsch–Carlson limiter: zero the slope at
/// a local extremum of the data, and otherwise cap it at three times the smaller neighbouring secant. That is
/// the PCHIP construction, and it makes the interpolant monotone on every interval where the data is, so a
/// sharp jump — a thermal boundary layer, a plume front — produces no over- or undershoot. The price is the
/// usual one: a smooth extremum is also flattened, so the order drops to two in the few cells around a peak.
/// Leave it off for fields that are smooth by construction (a Stokes velocity), keep it on for the transported
/// scalar.
///
/// `xs` must be strictly increasing and `xq` should lie within `xs[0] .. xs[n - 1]`; a query outside is
/// evaluated on the nearest interval's cubic, i.e. extrapolated.
template < typename T >
KOKKOS_INLINE_FUNCTION T pchip_interpolate_1d( const T* xs, const T* fs, const int n, const T xq, const bool monotone )
{
    if ( n <= 1 )
        return fs[0];

    T h[max_stencil_size - 1];
    T d[max_stencil_size - 1];
    for ( int k = 0; k < n - 1; ++k )
    {
        h[k] = xs[k + 1] - xs[k];
        d[k] = ( fs[k + 1] - fs[k] ) / h[k];
    }

    T m[max_stencil_size];
    if ( n == 2 )
    {
        m[0] = d[0];
        m[1] = d[0];
    }
    else
    {
        for ( int k = 1; k < n - 1; ++k )
            m[k] = ( d[k - 1] * h[k] + d[k] * h[k - 1] ) / ( h[k - 1] + h[k] );

        // One-sided ends: the slope of the parabola through the first (last) three nodes.
        m[0]     = ( d[0] * ( T( 2 ) * h[0] + h[1] ) - d[1] * h[0] ) / ( h[0] + h[1] );
        m[n - 1] = ( d[n - 2] * ( T( 2 ) * h[n - 2] + h[n - 3] ) - d[n - 3] * h[n - 2] ) / ( h[n - 2] + h[n - 3] );
    }

    if ( monotone )
    {
        for ( int k = 0; k < n; ++k )
        {
            // At the ends both neighbouring secants collapse onto the single adjacent one, which reduces the
            // rule below to the standard PCHIP endpoint limiter.
            const T dl = d[( k > 0 ) ? k - 1 : 0];
            const T dr = d[( k < n - 1 ) ? k : n - 2];

            if ( dl * dr <= T( 0 ) )
            {
                m[k] = T( 0 );
            }
            else if ( m[k] * dl < T( 0 ) )
            {
                m[k] = T( 0 );
            }
            else
            {
                // dl and dr share a sign here, and m[k] now shares it too, so one clamp suffices.
                const T limit = T( 3 ) * Kokkos::min( Kokkos::abs( dl ), Kokkos::abs( dr ) );
                m[k] = ( dl > T( 0 ) ) ? Kokkos::min( m[k], limit ) : Kokkos::max( m[k], -limit );
            }
        }
    }

    int k = 0;
    while ( k < n - 2 && xq > xs[k + 1] )
        ++k;

    const T hk = h[k];
    const T s  = ( xq - xs[k] ) / hk;
    const T s2 = s * s;
    const T s3 = s2 * s;

    return ( T( 2 ) * s3 - T( 3 ) * s2 + T( 1 ) ) * fs[k] + ( s3 - T( 2 ) * s2 + s ) * hk * m[k] +
           ( -T( 2 ) * s3 + T( 3 ) * s2 ) * fs[k + 1] + ( s3 - s2 ) * hk * m[k + 1];
}

/// @brief Continuous lateral index coordinates of a point given by wedge reference coordinates.
///
/// The barycentric coordinates of either triangle of a hex cell map onto the cell's index square exactly:
/// with \f$ \lambda = (1 - \xi - \eta, \xi, \eta) \f$ and the node indices of \ref wedge_lateral_node_indices,
/// \f$ (u, v) = \sum_k \lambda_k (n^x_k, n^y_k) \f$ collapses to \f$ (x + \xi, y + \eta) \f$ for `w == 0` and to
/// \f$ (x + 1 - \xi, y + 1 - \eta) \f$ for `w == 1`. Both land in \f$ [x, x+1] \times [y, y+1] \f$, so the
/// triangulation inside the hex cell is irrelevant to where the point is — which is exactly what lets a
/// structured stencil be laid over the lateral index space.
template < typename T >
KOKKOS_INLINE_FUNCTION void wedge_lateral_index_coords(
    const WedgeCell& cell,
    const T          xi,
    const T          eta,
    T&               u,
    T&               v )
{
    u = ( cell.w == 0 ) ? static_cast< T >( cell.x ) + xi : static_cast< T >( cell.x + 1 ) - xi;
    v = ( cell.w == 0 ) ? static_cast< T >( cell.y ) + eta : static_cast< T >( cell.y + 1 ) - eta;
}

/// @brief Cone radius of a point given by its radial cell and reference coordinate.
template < typename T, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION T wedge_radius_from_zeta(
    const int              subdomain,
    const WedgeCell&       cell,
    const CoordsRadiiType& coords_radii,
    const T                zeta )
{
    const T r1 = coords_radii( subdomain, cell.r );
    const T r2 = coords_radii( subdomain, cell.r + 1 );
    return r1 + T( 0.5 ) * ( zeta + T( 1 ) ) * ( r2 - r1 );
}

/// @brief True if the cell spanning nodes `cell_index, cell_index + 1` lies wholly inside `range`.
///
/// Guards against evaluating the stencil polynomial outside its own span: a cubic extrapolated even one cell is
/// far worse than a linear evaluation inside the cell, so a point outside the stencil's range is handed back to
/// the Q1 evaluator instead.
KOKKOS_INLINE_FUNCTION bool cell_inside_stencil_range( const int cell_index, const StencilRange& range )
{
    return cell_index >= range.lo && cell_index + 1 <= range.hi;
}

/// @brief True if every lateral node of a `nu x nv` stencil anchored at `(bx, by)` carries usable geometry.
template < typename LateralValidityType >
KOKKOS_INLINE_FUNCTION bool cubic_stencil_lateral_valid(
    const int                  subdomain,
    const int                  bx,
    const int                  by,
    const int                  nu,
    const int                  nv,
    const LateralValidityType& lateral_valid )
{
    for ( int j = 0; j < nv; ++j )
        for ( int i = 0; i < nu; ++i )
            if ( lateral_valid( subdomain, bx + i, by + j ) == 0 )
                return false;
    return true;
}

/// @brief Range of a scalar field over the eight nodes of one wedge cell's hex cell.
///
/// The local bound for a semi-Lagrangian limiter: the departure point lies inside this cell, so a transported
/// value outside the range of its corners is a new extremum invented by the reconstruction.
template < typename T, typename FieldViewType >
KOKKOS_INLINE_FUNCTION void cell_value_range(
    const FieldViewType& field,
    const int            subdomain,
    const WedgeCell&     cell,
    T&                   lo,
    T&                   hi )
{
    lo = field( subdomain, cell.x, cell.y, cell.r );
    hi = lo;

    for ( int k = 0; k < 2; ++k )
        for ( int j = 0; j < 2; ++j )
            for ( int i = 0; i < 2; ++i )
            {
                const T v = field( subdomain, cell.x + i, cell.y + j, cell.r + k );
                lo        = Kokkos::min( lo, v );
                hi        = Kokkos::max( hi, v );
            }
}

/// @brief Evaluates a scalar wedge field by monotone cubic interpolation on the structured index stencil.
///
/// A drop-in, higher-order replacement for \ref evaluate_q1_scalar. The multilinear Q1 evaluation is only
/// second order, and in a semi-Lagrangian scheme that error is committed once per timestep and accumulates as
/// numerical diffusion; this instead reconstructs the field with a tensor product of the \f$ C^1 \f$,
/// PCHIP-limited cubics of \ref pchip_interpolate_1d, laterally over the \f$ (x, y) \f$ index grid and radially
/// over the shell radii.
///
/// Laterally the stencil is uniform in index space (the map from index space to the sphere is smooth, so the
/// field is a smooth function of the indices and interpolating there is legitimate); radially it uses the
/// actual node radii, so non-uniform layer thicknesses are handled exactly.
///
/// **Boundaries.** `stencil` states which nodes may be read. Where the four-node window does not fit — against
/// the radial ends of the shell, or against the edge of the ghost layer — it is slid inwards instead of being
/// allowed to reach outside, so the interpolation stays one-sided but never extrapolates. Where the window
/// would cover one of the degenerate diagonal ghost corners of the icosahedral grid (see
/// \ref terra::fe::wedge::sl::ghosted_lateral_validity) there is no usable structured neighbourhood at all, and
/// the evaluation falls back to \ref evaluate_q1_scalar on the located wedge, whose three lateral nodes the
/// point location has already vetted. That affects at most the four hex cells at each corner of a subdomain.
///
/// **Bounding the result.** `clip_to_cell` clips the reconstruction into the range of the eight nodes of the
/// cell that contains the point — the property Q1 has for free, since it is a convex combination of exactly
/// those values, and the one a semi-Lagrangian step needs: a value outside that range is a new extremum the
/// transport then has to carry. This is the Bermejo-Staniforth filter, and it is what the transport relies on.
/// It matters most for a foot point that has drifted out of its cell: \ref locate_point reports
/// \f$ \xi, \eta, \zeta \f$ straight from the closed-form inverse and does not clamp them onto the reference
/// wedge, so a departure point can arrive here sitting in a neighbouring cell of the same stencil, where the
/// tensor-product sweeps are bounded by *that* cell's nodes instead.
///
/// `limit_slopes` additionally PCHIP-limits the nodal derivatives inside each one-dimensional sweep. **Leave it
/// off.** It is off by default because applying it here does not give a monotone three-dimensional
/// reconstruction: the second and third sweeps limit values produced by the first, not nodal data, so the
/// flattening each sweep applies at an extremum compounds across the three. In a semi-Lagrangian scheme that
/// error is committed afresh every step and ratchets. Measured on `test_mmoc_rotation` at level 5, one full
/// revolution of the cone, with the limiter on: the peak is held at 0.644 of the exact 1.0, but the field
/// spreads into a plateau — 185 nodes sit above 90% of the peak where the exact field has 1 — and the total
/// mass grows monotonically to 3.0x its initial value, which is what drives an L2 error of 1.79. With the
/// limiter off and only `clip_to_cell` in force the peak is lower at 0.355, but the plateau is gone (37 nodes
/// above 90% of the peak), mass ends within 12% of exact, and the L2 error falls to 0.845 -- better than the
/// 0.896 of the Q1 evaluation this replaces, and inside the test's tolerance where 1.79 was not. The high peak
/// the limiter appears to buy is mass it invented, not a peak it preserved.
///
template < typename T, typename FieldViewType, typename CoordsRadiiType, typename LateralValidityType >
KOKKOS_INLINE_FUNCTION T evaluate_cubic_scalar(
    const FieldViewType&       field,
    const int                  subdomain,
    const WedgeCell&           cell,
    const T                    xi,
    const T                    eta,
    const T                    zeta,
    const CoordsRadiiType&     coords_radii,
    const StencilBounds&       stencil,
    const LateralValidityType& lateral_valid,
    const bool                 clip_to_cell = true,
    const bool                 limit_slopes = false,
    const int                  preferred_width = cubic_stencil_size )
{
    // A width of two asks for the Q1 wedge evaluation itself, which is what the stencil ladder bottoms out at
    // anyway. Having it selectable makes the interpolation order a runtime choice for the whole ladder.
    if ( preferred_width <= 2 || !cell_inside_stencil_range( cell.x, stencil.x ) ||
         !cell_inside_stencil_range( cell.y, stencil.y ) || !cell_inside_stencil_range( cell.r, stencil.r ) )
        return evaluate_q1_scalar( field, subdomain, cell, xi, eta, zeta );

    int       bx = 0, by = 0, br = 0;
    const int nu = stencil_window( cell.x, stencil.x, preferred_width, bx );
    const int nv = stencil_window( cell.y, stencil.y, preferred_width, by );
    const int nr = stencil_window( cell.r, stencil.r, preferred_width, br );

    if ( !cubic_stencil_lateral_valid( subdomain, bx, by, nu, nv, lateral_valid ) )
        return evaluate_q1_scalar( field, subdomain, cell, xi, eta, zeta );

    T u = T( 0 ), v = T( 0 );
    wedge_lateral_index_coords( cell, xi, eta, u, v );
    const T rho = wedge_radius_from_zeta( subdomain, cell, coords_radii, zeta );
    
    T us[max_stencil_size], vs[max_stencil_size], rs[max_stencil_size];
    for ( int i = 0; i < nu; ++i )
        us[i] = static_cast< T >( bx + i );
    for ( int j = 0; j < nv; ++j )
        vs[j] = static_cast< T >( by + j );
    for ( int k = 0; k < nr; ++k )
        rs[k] = coords_radii( subdomain, br + k );

    T f_r[max_stencil_size];
    for ( int k = 0; k < nr; ++k )
    {
        T f_v[max_stencil_size];
        for ( int j = 0; j < nv; ++j )
        {
            T f_u[max_stencil_size];
            for ( int i = 0; i < nu; ++i )
                f_u[i] = field( subdomain, bx + i, by + j, br + k );

            f_v[j] = ( nu > cubic_stencil_size ) ? lagrange_interpolate_1d( us, f_u, nu, u )
                                                : pchip_interpolate_1d( us, f_u, nu, u, limit_slopes );
        }
        f_r[k] = ( nv > cubic_stencil_size ) ? lagrange_interpolate_1d( vs, f_v, nv, v )
                                             : pchip_interpolate_1d( vs, f_v, nv, v, limit_slopes );
    }

    const T value = ( nr > cubic_stencil_size ) ? lagrange_interpolate_1d( rs, f_r, nr, rho )
                                                : pchip_interpolate_1d( rs, f_r, nr, rho, limit_slopes );

    if ( !clip_to_cell )
        return value;

    T lo = T( 0 ), hi = T( 0 );
    cell_value_range( field, subdomain, cell, lo, hi );
    return Kokkos::clamp( value, lo, hi );
}

/// @brief Evaluates a vector wedge field by monotone cubic interpolation on the structured index stencil.
///
/// Componentwise \ref evaluate_cubic_scalar; see there for the stencil and boundary handling. For a velocity
/// this wants `monotone = false`: a velocity is smooth by construction, so the limiter can only cost accuracy.
///
/// @note Whether a velocity is worth reconstructing this way is not obvious and should be measured. The MMOC
///       transport keeps its velocity on \ref evaluate_q1_vec: in the rotation test the two are
///       indistinguishable (the rigid-rotation velocity is linear in \f$ x \f$, which Q1 reproduces exactly
///       while this reconstruction does not), and Q1 is an order of magnitude cheaper.
template < typename T, int VecDim, typename FieldViewType, typename CoordsRadiiType, typename LateralValidityType >
KOKKOS_INLINE_FUNCTION dense::Vec< T, VecDim > evaluate_cubic_vec(
    const FieldViewType&       field,
    const int                  subdomain,
    const WedgeCell&           cell,
    const T                    xi,
    const T                    eta,
    const T                    zeta,
    const CoordsRadiiType&     coords_radii,
    const StencilBounds&       stencil,
    const LateralValidityType& lateral_valid,
    const bool                 monotone = false )
{
    if ( !cell_inside_stencil_range( cell.x, stencil.x ) || !cell_inside_stencil_range( cell.y, stencil.y ) ||
         !cell_inside_stencil_range( cell.r, stencil.r ) )
        return evaluate_q1_vec< T, VecDim >( field, subdomain, cell, xi, eta, zeta );

    int       bx = 0, by = 0, br = 0;
    const int nu = cubic_stencil_window( cell.x, stencil.x, bx );
    const int nv = cubic_stencil_window( cell.y, stencil.y, by );
    const int nr = cubic_stencil_window( cell.r, stencil.r, br );

    if ( !cubic_stencil_lateral_valid( subdomain, bx, by, nu, nv, lateral_valid ) )
        return evaluate_q1_vec< T, VecDim >( field, subdomain, cell, xi, eta, zeta );

    T u = T( 0 ), v = T( 0 );
    wedge_lateral_index_coords( cell, xi, eta, u, v );
    const T rho = wedge_radius_from_zeta( subdomain, cell, coords_radii, zeta );

    T us[max_stencil_size], vs[max_stencil_size], rs[max_stencil_size];
    for ( int i = 0; i < nu; ++i )
        us[i] = static_cast< T >( bx + i );
    for ( int j = 0; j < nv; ++j )
        vs[j] = static_cast< T >( by + j );
    for ( int k = 0; k < nr; ++k )
        rs[k] = coords_radii( subdomain, br + k );

    dense::Vec< T, VecDim > value;
    for ( int d = 0; d < VecDim; ++d )
    {
        T f_r[max_stencil_size];
        for ( int k = 0; k < nr; ++k )
        {
            T f_v[max_stencil_size];
            for ( int j = 0; j < nv; ++j )
            {
                T f_u[max_stencil_size];
                for ( int i = 0; i < nu; ++i )
                    f_u[i] = field( subdomain, bx + i, by + j, br + k, d );

                f_v[j] = pchip_interpolate_1d( us, f_u, nu, u, monotone );
            }
            f_r[k] = pchip_interpolate_1d( vs, f_v, nv, v, monotone );
        }
        value( d ) = pchip_interpolate_1d( rs, f_r, nr, rho, monotone );
    }
    return value;
}

/// @brief Forward map of a wedge cell: reference coordinates -> physical point.
///
/// Provided here so that tests can round-trip against \ref locate_point without pulling in the kernel helpers.
template < typename T, typename CoordsShellType, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION dense::Vec< T, 3 > wedge_forward_map(
    const int              subdomain,
    const WedgeCell&       cell,
    const CoordsShellType& coords_shell,
    const CoordsRadiiType& coords_radii,
    const T                xi,
    const T                eta,
    const T                zeta )
{
    int nx[3], ny[3];
    wedge_lateral_node_indices( cell, nx, ny );

    dense::Vec< T, 3 > p[3];
    for ( int v = 0; v < 3; ++v )
    {
        for ( int d = 0; d < 3; ++d )
        {
            p[v]( d ) = coords_shell( subdomain, nx[v], ny[v], d );
        }
    }

    return forward_map(
        p[0],
        p[1],
        p[2],
        coords_radii( subdomain, cell.r ),
        coords_radii( subdomain, cell.r + 1 ),
        xi,
        eta,
        zeta );
}

} // namespace terra::fe::wedge::sl
