
#pragma once

#include "terra/dense/vec.hpp"

/// @namespace terra::fe::wedge::quadrature
/// @brief Quadrature rules for the reference wedge.
///
/// Taken from https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_wedge/quadrature_rules_wedge.html
///
/// Reference:
/// \code
///   Carlos Felippa,
///   A compendium of FEM integration formulas for symbolic work,
///   Engineering Computation,
///   Volume 21, Number 8, 2004, pages 867-890.
/// \endcode
///
/// Reference wedge:
/*!
    \f[
    \begin{align}
      0 \leq X \\
      0 \leq Y \\
      X + Y \leq 1 \\
      -1 \leq Z \leq 1
    \end{align}
    \f]
*/
namespace terra::fe::wedge::quadrature {

constexpr int quad_felippa_1x1_num_quad_points = 1;

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void
    quad_felippa_1x1_quad_points( dense::Vec< T, 3 > ( &quad_points )[quad_felippa_1x1_num_quad_points] )
{
    quad_points[0] = { 1.0 / 3.0, 1.0 / 3.0, 0.0 };
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void
    quad_felippa_1x1_quad_weights( T ( &quad_weights )[quad_felippa_1x1_num_quad_points] )
{
    quad_weights[0] = 1.0;
}

constexpr int quad_felippa_1x2_num_quad_points = 2;

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void
    quad_felippa_1x2_quad_points( dense::Vec< T, 3 > ( &quad_points )[quad_felippa_1x2_num_quad_points] )
{
    quad_points[0] = { 1.0 / 3.0, 1.0 / 3.0, 1.0/Kokkos::sqrt(3) };
    quad_points[1] = { 1.0 / 3.0, 1.0 / 3.0, -1.0/Kokkos::sqrt(3) };
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void
    quad_felippa_1x2_quad_weights( T ( &quad_weights )[quad_felippa_1x2_num_quad_points] )
{
    quad_weights[0] = 0.5;
    quad_weights[1] = 0.5;
}

constexpr int quad_felippa_3x2_num_quad_points = 6;

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void
    quad_felippa_3x2_quad_points( dense::Vec< T, 3 > ( &quad_points )[quad_felippa_3x2_num_quad_points] )
{
    quad_points[0] = { 0.6666666666666666, 0.1666666666666667, -0.5773502691896257 };
    quad_points[1] = { 0.1666666666666667, 0.6666666666666666, -0.5773502691896257 };
    quad_points[2] = { 0.1666666666666667, 0.1666666666666667, -0.5773502691896257 };
    quad_points[3] = { 0.6666666666666666, 0.1666666666666667, 0.5773502691896257 };
    quad_points[4] = { 0.1666666666666667, 0.6666666666666666, 0.5773502691896257 };
    quad_points[5] = { 0.1666666666666667, 0.1666666666666667, 0.5773502691896257 };
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void quad_felippa_3x2_quad_points_ptr( dense::Vec< T, 3 >* quad_points )
{
    quad_points[0] = { 0.6666666666666666, 0.1666666666666667, -0.5773502691896257 };
    quad_points[1] = { 0.1666666666666667, 0.6666666666666666, -0.5773502691896257 };
    quad_points[2] = { 0.1666666666666667, 0.1666666666666667, -0.5773502691896257 };
    quad_points[3] = { 0.6666666666666666, 0.1666666666666667, 0.5773502691896257 };
    quad_points[4] = { 0.1666666666666667, 0.6666666666666666, 0.5773502691896257 };
    quad_points[5] = { 0.1666666666666667, 0.1666666666666667, 0.5773502691896257 };
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void
    quad_felippa_3x2_quad_weights( T ( &quad_weights )[quad_felippa_3x2_num_quad_points] )
{
    quad_weights[0] = 0.1666666666666667;
    quad_weights[1] = 0.1666666666666667;
    quad_weights[2] = 0.1666666666666667;
    quad_weights[3] = 0.1666666666666667;
    quad_weights[4] = 0.1666666666666667;
    quad_weights[5] = 0.1666666666666667;
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void quad_felippa_3x2_quad_weights_ptr( T* quad_weights )
{
    quad_weights[0] = 0.1666666666666667;
    quad_weights[1] = 0.1666666666666667;
    quad_weights[2] = 0.1666666666666667;
    quad_weights[3] = 0.1666666666666667;
    quad_weights[4] = 0.1666666666666667;
    quad_weights[5] = 0.1666666666666667;
}

} // namespace terra::fe::wedge::quadrature
