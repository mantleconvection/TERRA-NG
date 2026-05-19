
#pragma once

#include "terra/dense/vec.hpp"

namespace terra::fe::wedge::quadrature {

template < typename ScalarT >
struct QuadRuleWedge1Point
{
    static constexpr int num_quad_points = 1;

    static constexpr ScalarT single_quad_weight = 1.0;

    KOKKOS_INLINE_FUNCTION static constexpr void get_quad_point( int index, ScalarT& xi, ScalarT& eta, ScalarT& zeta )
    {
        xi   = 1.0 / 3.0;
        eta  = 1.0 / 3.0;
        zeta = 0.0;
    }
};

template < typename ScalarT >
struct QuadRuleWedge6Points
{
    static constexpr int num_quad_points = 6;

    static constexpr ScalarT single_quad_weight = 0.1666666666666667;

    KOKKOS_INLINE_FUNCTION static constexpr void get_quad_point( int index, ScalarT& xi, ScalarT& eta, ScalarT& zeta )
    {
        zeta  = static_cast< ScalarT >( ( index / 3 ) * 2 - 1 ) * 0.5773502691896257;
        int i = index % 3;
        xi    = i == 0 ? 2.0 / 3.0 : 1.0 / 6.0;
        eta   = i == 1 ? 2.0 / 3.0 : 1.0 / 6.0;
    }
};

} // namespace terra::fe::wedge::quadrature
