
#pragma once

#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"

/*! @namespace terra::fe::wedge
    @brief Features for wedge elements.
   
    \section geometry Geometry
   
      \f$ \xi, \eta \in [0, 1] \f$ (lateral reference coords)
   
      \f$ \zeta \in [-1, 1] \f$   (radial reference coords)
   
      \f$ 0 \leq \xi + \eta \leq 1 \f$
   
    \section enumeration Node enumeration
   
      \code
   
      r_node_idx = r_cell_idx + 1 (outer):
   
      5
      |\
      | \
      3--4
      \endcode
   
      \code
   
      r_node_idx = r_cell_idx (inner):
   
      2
      |\
      | \
      0--1
      \endcode
   
    \section shape Shape functions
   
    Lateral:
   
      \f[
      \begin{align}
        N^\mathrm{lat}_0 = N^\mathrm{lat}_3 &= 1 - \xi - \eta \\
        N^\mathrm{lat}_1 = N^\mathrm{lat}_4 &= \xi \\
        N^\mathrm{lat}_2 = N^\mathrm{lat}_5 &= \eta
      \end{align}
      \f]
  
    Radial:
   
      \f[
      \begin{align}
        N^\mathrm{rad}_0 = N^\mathrm{rad}_1 = N^\mathrm{rad}_2 &= \frac{1}{2} ( 1 - \zeta ) \\
        N^\mathrm{rad}_3 = N^\mathrm{rad}_4 = N^\mathrm{rad}_5 &= \frac{1}{2} ( 1 + \zeta ) \\
      \end{align}
      \f]
  
    Full:
   
      \f[
      N_i = N^\mathrm{lat}_i * N^\mathrm{rad}_i
      \f]
   
   
    \section phy Physical coordinates
   
      \code
      r_1, r_2                     radii of bottom and top (r_1 < r_2)
      p1_phy, p2_phy, p3_phy       coords of triangle on unit sphere
      \endcode
*/
namespace terra::fe::wedge {

/// @brief Radial shape function.
/*!
      \f[
      \begin{align}
        N^\mathrm{rad}_0 = N^\mathrm{rad}_1 = N^\mathrm{rad}_2 &= \frac{1}{2} ( 1 - \zeta ) \\
        N^\mathrm{rad}_3 = N^\mathrm{rad}_4 = N^\mathrm{rad}_5 &= \frac{1}{2} ( 1 + \zeta ) \\
      \end{align}
      \f]
*/
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T shape_rad( const int node_idx, const T zeta )
{
    const T N_rad[2] = { static_cast< T >( 0.5 ) * ( 1 - zeta ), static_cast< T >( 0.5 ) * ( 1 + zeta ) };
    return N_rad[node_idx / 3];
}

/// @brief Radial shape function.
/*!
      \f[
      \begin{align}
        N^\mathrm{rad}_0 = N^\mathrm{rad}_1 = N^\mathrm{rad}_2 &= \frac{1}{2} ( 1 - \zeta ) \\
        N^\mathrm{rad}_3 = N^\mathrm{rad}_4 = N^\mathrm{rad}_5 &= \frac{1}{2} ( 1 + \zeta ) \\
      \end{align}
      \f]
*/   
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T shape_rad( const int node_idx, const dense::Vec< T, 3 >& xi_eta_zeta )
{
    return shape_rad( node_idx, xi_eta_zeta( 2 ) );
}

/// @brief Lateral shape function.
/*!
      \f[
      \begin{align}
        N^\mathrm{lat}_0 = N^\mathrm{lat}_3 &= 1 - \xi - \eta \\
        N^\mathrm{lat}_1 = N^\mathrm{lat}_4 &= \xi \\
        N^\mathrm{lat}_2 = N^\mathrm{lat}_5 &= \eta
      \end{align}
      \f]
*/
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T shape_lat( const int node_idx, const T xi, const T eta )
{
    const T N_lat[3] = { static_cast< T >( 1.0 ) - xi - eta, xi, eta };
    return N_lat[node_idx % 3];
}

/// @brief Lateral shape function.
/*!
      \f[
      \begin{align}
        N^\mathrm{lat}_0 = N^\mathrm{lat}_3 &= 1 - \xi - \eta \\
        N^\mathrm{lat}_1 = N^\mathrm{lat}_4 &= \xi \\
        N^\mathrm{lat}_2 = N^\mathrm{lat}_5 &= \eta
      \end{align}
      \f]
*/
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T shape_lat( const int node_idx, const dense::Vec< T, 3 >& xi_eta_zeta )
{
    return shape_lat( node_idx, xi_eta_zeta( 0 ), xi_eta_zeta( 1 ) );
}

/// @brief (Tensor-product) Shape function.
/// \f[
///   N_i = N^\mathrm{lat}_i * N^\mathrm{rad}_i
/// \f]
///
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T shape( const int node_idx, const T xi, const T eta, const T zeta )
{
    return shape_lat( node_idx, xi, eta ) * shape_rad( node_idx, zeta );
}

/// @brief (Tensor-product) Shape function.
/// \f[
///   N_i = N^\mathrm{lat}_i * N^\mathrm{rad}_i
/// \f]
///
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T shape( const int node_idx, const dense::Vec< T, 3 >& xi_eta_zeta )
{
    return shape_lat( node_idx, xi_eta_zeta ) * shape_rad( node_idx, xi_eta_zeta );
}

/// @brief Gradient of the radial part of the shape function, in the radial direction
/// \f$ \frac{\partial}{\partial \zeta} N^\mathrm{rad}_j \f$
///
/// This is different from the radial part of the gradient of the full shape function!
///
/// That would be
///
/// \f$ \frac{\partial}{\partial \zeta} N_j = N^\mathrm{lat}_j \frac{\partial}{\partial \zeta} N^\mathrm{rad}_j \f$
///
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T grad_shape_rad( const int node_idx )
{
    constexpr T grad_N_rad[2] = { -0.5, 0.5 };
    return grad_N_rad[node_idx / 3];
}

/// @brief Gradient of the lateral part of the shape function, in xi direction
/// \f$ \frac{\partial}{\partial \xi} N^\mathrm{lat}_j \f$
///
/// This is different from the \f$ \xi \f$ part (first entry) of the gradient of the full shape function!
///
/// That would be
///
/// \f$ \frac{\partial}{\partial \xi} N_j = N^\mathrm{rad}_j \frac{\partial}{\partial \xi} N^\mathrm{lat}_j \f$
///
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T grad_shape_lat_xi( const int node_idx )
{
    constexpr T grad_N_lat_xi[3] = { -1.0, 1.0, 0.0 };
    return grad_N_lat_xi[node_idx % 3];
}

/// @brief Gradient of the lateral part of the shape function, in eta direction
/// \f$ \frac{\partial}{\partial \eta} N^\mathrm{lat}_j \f$
///
/// This is different from the \f$ \eta \f$ part (second entry) of the gradient of the full shape function!
///
/// That would be
///
/// \f$ \frac{\partial}{\partial \eta} N_j = N^\mathrm{rad}_j \frac{\partial}{\partial \eta} N^\mathrm{lat}_j \f$
///
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T grad_shape_lat_eta( const int node_idx )
{
    constexpr T grad_N_lat_eta[3] = { -1.0, 0.0, 1.0 };
    return grad_N_lat_eta[node_idx % 3];
}

/// @brief Gradient of the full shape function:
/*!
    \f[
    \nabla N_j =
    \begin{bmatrix}
        \frac{\partial}{\partial \xi} N_j \\
        \frac{\partial}{\partial \eta} N_j \\
        \frac{\partial}{\partial \zeta} N_j
    \end{bmatrix}
    =
    \begin{bmatrix}
        N^\mathrm{rad}_j \frac{\partial}{\partial \xi} N^\mathrm{lat}_j \\
        N^\mathrm{rad}_j \frac{\partial}{\partial \eta} N^\mathrm{lat}_j \\
        N^\mathrm{lat}_j \frac{\partial}{\partial \zeta} N^\mathrm{rad}_j
    \end{bmatrix}
    \f]
*/
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 >
    grad_shape( const int node_idx, const T xi, const T eta, const T zeta )
{
    dense::Vec< T, 3 > grad_N;
    grad_N( 0 ) = grad_shape_lat_xi< T >( node_idx ) * shape_rad( node_idx, zeta );
    grad_N( 1 ) = grad_shape_lat_eta< T >( node_idx ) * shape_rad( node_idx, zeta );
    grad_N( 2 ) = shape_lat( node_idx, xi, eta ) * grad_shape_rad< T >( node_idx );
    return grad_N;
}

/// @brief Gradient of the full shape function:
/*!
    \f[
    \nabla N_j =
    \begin{bmatrix}
        \frac{\partial}{\partial \xi} N_j \\
        \frac{\partial}{\partial \eta} N_j \\
        \frac{\partial}{\partial \zeta} N_j
    \end{bmatrix}
    =
    \begin{bmatrix}
        N^\mathrm{rad}_j \frac{\partial}{\partial \xi} N^\mathrm{lat}_j \\
        N^\mathrm{rad}_j \frac{\partial}{\partial \eta} N^\mathrm{lat}_j \\
        N^\mathrm{lat}_j \frac{\partial}{\partial \zeta} N^\mathrm{rad}_j
    \end{bmatrix}
    \f]
*/
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 >
    grad_shape( const int node_idx, const dense::Vec< T, 3 >& xi_eta_zeta )
{
    return grad_shape( node_idx, xi_eta_zeta( 0 ), xi_eta_zeta( 1 ), xi_eta_zeta( 2 ) );
}

/// @brief Returns the coarse grid radial shape function evaluated at a point of the reference fine grid wedge.
///
/// @param coarse_node_idx       wedge node index of the coarse grid shape function ("which coarse grid shape function to
///                              evaluate") (in {0, ..., 5})
/// @param fine_radial_wedge_idx 0 for inner fine wedge, 1 for outer fine wedge
/// @param zeta_fine             coordinate in the reference fine-wedge (in [-1, 1])
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T
    shape_rad_coarse( const int coarse_node_idx, const int fine_radial_wedge_idx, const T zeta_fine )
{
    switch ( coarse_node_idx / 3 )
    {
    case 0:
        switch ( fine_radial_wedge_idx )
        {
        case 0:
            // coarse node at bottom (in {0, 1, 2}), evaluated in bottom fine wedge
            // 0.5 at 1
            // 1 at -1,
            return 0.25 * ( 3 - zeta_fine );
        case 1:
            // coarse node at bottom (in {0, 1, 2}), evaluated in top fine wedge
            // 0 at 1
            // 0.5 at -1,
            return 0.25 * ( 1 - zeta_fine );
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_radial_wedge_idx )
        {
        case 0:
            // coarse node at top (in {3, 4, 5}), evaluated in bottom fine wedge
            // 0.5 at 1
            // 0 at -1,
            return 0.25 * ( 1 + zeta_fine );
        case 1:
            // coarse node at top (in {3, 4, 5}), evaluated in top fine wedge
            // 1 at 1
            // 0.5 at -1
            return 0.25 * ( 3 + zeta_fine );
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}

/// @brief Returns the coarse grid lateral shape function evaluated at a point of the reference fine grid wedge.
///
/// @param coarse_node_idx        wedge node index of the coarse grid shape function ("which coarse grid shape function
///                               to evaluate") (in {0, ..., 5})
/// @param fine_lateral_wedge_idx index of the lateral wedge index in a (once) refined coarse mesh, in {0, 1, 2, 3}
///                               0: bottom left triangle (orientation up)
///                               1: bottom right triangle (orientation: up)
///                               2: top triangle (orientation: up)
///                               3: center triangle (orientation: down)
/// @param xi_fine                xi-coordinate in the reference fine-wedge (in [0, 1])
/// @param eta_fine               eta-coordinate in the reference fine-wedge (in [0, 1])
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T
    shape_lat_coarse( const int coarse_node_idx, const int fine_lateral_wedge_idx, const T xi_fine, const T eta_fine )
{
    switch ( coarse_node_idx % 3 )
    {
    case 0:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return -0.5 * eta_fine - 0.5 * xi_fine + 1.0;
        case 1:
            return -0.5 * eta_fine - 0.5 * xi_fine + 0.5;
        case 2:
            return -0.5 * eta_fine - 0.5 * xi_fine + 0.5;
        case 3:
            return 0.5 * eta_fine + 0.5 * xi_fine;
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0.5 * xi_fine;
        case 1:
            return 0.5 * xi_fine + 0.5;
        case 2:
            return 0.5 * xi_fine;
        case 3:
            return 0.5 - 0.5 * xi_fine;
        default:
            return 0.0;
        }
    case 2:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0.5 * eta_fine;
        case 1:
            return 0.5 * eta_fine;
        case 2:
            return 0.5 * eta_fine + 0.5;
        case 3:
            return 0.5 - 0.5 * eta_fine;
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T shape_coarse(
    const int coarse_node_idx,
    const int fine_radial_wedge_idx,
    const int fine_lateral_wedge_idx,
    const T   xi_fine,
    const T   eta_fine,
    const T   zeta_fine )
{
    return shape_lat_coarse( coarse_node_idx, fine_lateral_wedge_idx, xi_fine, eta_fine ) *
           shape_rad_coarse( coarse_node_idx, fine_radial_wedge_idx, zeta_fine );
}
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T shape_coarse(
    const int                 coarse_node_idx,
    const int                 fine_radial_wedge_idx,
    const int                 fine_lateral_wedge_idx,
    const dense::Vec< T, 3 >& xi_eta_zeta_fine )
{
    return shape_lat_coarse( coarse_node_idx, fine_lateral_wedge_idx, xi_eta_zeta_fine( 0 ), xi_eta_zeta_fine( 1 ) ) *
           shape_rad_coarse( coarse_node_idx, fine_radial_wedge_idx, xi_eta_zeta_fine( 2 ) );
}
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T grad_shape_rad_coarse( const int coarse_node_idx, const int fine_radial_wedge_idx )
{
    switch ( coarse_node_idx / 3 )
    {
    case 0:
        switch ( fine_radial_wedge_idx )
        {
        case 0:
            // coarse node at bottom (in {0, 1, 2}), evaluated in bottom fine wedge
            // 0.5 at 1
            // 1 at -1,
            return -0.25;
        case 1:
            // coarse node at bottom (in {0, 1, 2}), evaluated in top fine wedge
            // 0 at 1
            // 0.5 at -1,
            return -0.25;
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_radial_wedge_idx )
        {
        case 0:
            // coarse node at top (in {3, 4, 5}), evaluated in bottom fine wedge
            // 0.5 at 1
            // 0 at -1,
            return 0.25;
        case 1:
            // coarse node at top (in {3, 4, 5}), evaluated in top fine wedge
            // 1 at 1
            // 0.5 at -1
            return 0.25;
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T
    grad_shape_lat_coarse_xi( const int coarse_node_idx, const int fine_lateral_wedge_idx )
{
    // derivatives in xi_fine direction
    switch ( coarse_node_idx % 3 )
    {
    case 0:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return -0.5;
        case 1:
            return -0.5;
        case 2:
            return -0.5;
        case 3:
            return 0.5;
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0.5;
        case 1:
            return 0.5;
        case 2:
            return 0.5;
        case 3:
            return -0.5;
        default:
            return 0.0;
        }
    case 2:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0;
        case 1:
            return 0;
        case 2:
            return 0;
        case 3:
            return 0;
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T
    grad_shape_lat_coarse_eta( const int coarse_node_idx, const int fine_lateral_wedge_idx )
{
    // derivatives in eta_fine direction
    switch ( coarse_node_idx % 3 )
    {
    case 0:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return -0.5;
        case 1:
            return -0.5;
        case 2:
            return -0.5;
        case 3:
            return 0.5;
        default:
            return 0.0;
        }
    case 1:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0;
        case 1:
            return 0;
        case 2:
            return 0;
        case 3:
            return 0;
        default:
            return 0.0;
        }
    case 2:
        switch ( fine_lateral_wedge_idx )
        {
        case 0:
            return 0.5;
        case 1:
            return 0.5;
        case 2:
            return 0.5;
        case 3:
            return -0.5;
        default:
            return 0.0;
        }
    default:
        return 0.0;
    }
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 > grad_shape_coarse(
    const int node_idx,
    const int fine_radial_wedge_idx,
    const int fine_lateral_wedge_idx,
    const T   xi,
    const T   eta,
    const T   zeta )
{
    dense::Vec< T, 3 > grad_N;
    grad_N( 0 ) = grad_shape_lat_coarse_xi< T >( node_idx, fine_lateral_wedge_idx ) *
                  shape_rad_coarse( node_idx, fine_radial_wedge_idx, zeta );
    grad_N( 1 ) = grad_shape_lat_coarse_eta< T >( node_idx, fine_lateral_wedge_idx ) *
                  shape_rad_coarse( node_idx, fine_radial_wedge_idx, zeta );
    grad_N( 2 ) = shape_lat_coarse( node_idx, fine_lateral_wedge_idx, xi, eta ) *
                  grad_shape_rad_coarse< T >( node_idx, fine_radial_wedge_idx );
    return grad_N;
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 > grad_shape_coarse(
    const int                 node_idx,
    const int                 fine_radial_wedge_idx,
    const int                 fine_lateral_wedge_idx,
    const dense::Vec< T, 3 >& xi_eta_zeta_fine )
{
    return grad_shape_coarse(
        node_idx,
        fine_radial_wedge_idx,
        fine_lateral_wedge_idx,
        xi_eta_zeta_fine( 0 ),
        xi_eta_zeta_fine( 1 ),
        xi_eta_zeta_fine( 2 ) );
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T forward_map_rad( const T r_1, const T r_2, const T zeta )
{
    return r_1 + 0.5 * ( r_2 - r_1 ) * ( 1 + zeta );
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 > forward_map_lat(
    const dense::Vec< T, 3 >& p1_phy,
    const dense::Vec< T, 3 >& p2_phy,
    const dense::Vec< T, 3 >& p3_phy,
    const T                   xi,
    const T                   eta )
{
    return ( 1 - xi - eta ) * p1_phy + xi * p2_phy + eta * p3_phy;
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 > forward_map(
    const dense::Vec< T, 3 >& p1_phy,
    const dense::Vec< T, 3 >& p2_phy,
    const dense::Vec< T, 3 >& p3_phy,
    const T                   r_1,
    const T                   r_2,
    const T                   xi,
    const T                   eta,
    const T                   zeta )
{
    return forward_map_rad( r_1, r_2, zeta ) * forward_map_lat( p1_phy, p2_phy, p3_phy, xi, eta );
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr T grad_forward_map_rad( const T r_1, const T r_2 )
{
    return 0.5 * ( r_2 - r_1 );
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 > grad_forward_map_lat_xi(
    const dense::Vec< T, 3 >& p1_phy,
    const dense::Vec< T, 3 >& p2_phy,
    const dense::Vec< T, 3 >& /* p3_phy */ )
{
    return p2_phy - p1_phy;
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 > grad_forward_map_lat_eta(
    const dense::Vec< T, 3 >& p1_phy,
    const dense::Vec< T, 3 >& /* p2_phy */,
    const dense::Vec< T, 3 >& p3_phy )
{
    return p3_phy - p1_phy;
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Mat< T, 3, 3 > jac_lat(
    const dense::Vec< T, 3 >& p1_phy,
    const dense::Vec< T, 3 >& p2_phy,
    const dense::Vec< T, 3 >& p3_phy,
    const T                   xi,
    const T                   eta )
{
    const auto col_0 = grad_forward_map_lat_xi( p1_phy, p2_phy, p3_phy );
    const auto col_1 = grad_forward_map_lat_eta( p1_phy, p2_phy, p3_phy );
    const auto col_2 = forward_map_lat( p1_phy, p2_phy, p3_phy, xi, eta );
    return dense::Mat< T, 3, 3 >::from_col_vecs( col_0, col_1, col_2 );
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 > jac_rad( const T r_1, const T r_2, const T zeta )
{
    const auto r      = forward_map_rad( r_1, r_2, zeta );
    const auto grad_r = grad_forward_map_rad( r_1, r_2 );
    return dense::Vec< T, 3 >{ r, r, grad_r };
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Mat< T, 3, 3 >
    jac( const dense::Vec< T, 3 >& p1_phy,
         const dense::Vec< T, 3 >& p2_phy,
         const dense::Vec< T, 3 >& p3_phy,
         const T                   r_1,
         const T                   r_2,
         const T                   xi,
         const T                   eta,
         const T                   zeta )
{
    return jac_lat( p1_phy, p2_phy, p3_phy, xi, eta ) *
           dense::Mat< T, 3, 3 >::diagonal_from_vec( jac_rad( r_1, r_2, zeta ) );
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Mat< T, 3, 3 >
    jac( const dense::Vec< T, 3 > p_phy[3], const T r_1, const T r_2, const dense::Vec< T, 3 >& xi_eta_zeta_fine )
{
    return jac(
        p_phy[0], p_phy[1], p_phy[2], r_1, r_2, xi_eta_zeta_fine( 0 ), xi_eta_zeta_fine( 1 ), xi_eta_zeta_fine( 2 ) );
}

/// @brief Returns the symmetric gradient of the shape function of a dof at a quadrature point.
///
/// @param J_inv_transposed       inverse transposed jacobian to map to physical element
/// @param quad_point             quadrature point on the reference element
/// @param dof                    local index of the shape function a wedge
/// @param dim                    dimension of the vectorial shape function, of which the gradient should be computed
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr dense::Mat< T, 3, 3 > symmetric_grad(
    const dense::Mat< T, 3, 3 >& J_inv_transposed,
    const dense::Vec< T, 3 >&    quad_point,
    const int                    dof,
    const int                    dim )
{
    dense::Mat< T, 3, 3 > grad =
        J_inv_transposed * dense::Mat< T, 3, 3 >::from_single_col_vec( grad_shape( dof, quad_point ), dim );
    return ( grad + grad.transposed() ) * 0.5;
}

} // namespace terra::fe::wedge
