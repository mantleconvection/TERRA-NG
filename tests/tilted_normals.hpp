#pragma once

/// @file  tilted_normals.hpp
/// @brief User-set free-slip normals for operator tests: r̂ tilted by a smooth perturbation and
///        renormalised to unit length. `phase` changes the perturbation pattern, so the CMB and
///        surface fields can be made different (an operator that swaps them is then detected).

#include "terra/grid/grid_types.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"

namespace terra::test {

/// Lateral unit-normal field shaped like `coords` (unit-sphere coords): n = (r̂ + eps·t) / |r̂ + eps·t|.
/// eps = 0.1 tilts the normals by up to about 6 degrees.
inline grid::Grid3DDataVec< double, 3 >
    tilted_normals( const grid::Grid3DDataVec< double, 3 >& coords, const double eps, const double phase )
{
    grid::Grid3DDataVec< double, 3 > n( "tilted_normals", coords.extent( 0 ), coords.extent( 1 ), coords.extent( 2 ) );
    Kokkos::parallel_for(
        "tilted_normals",
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
            { 0, 0, 0 }, { coords.extent( 0 ), coords.extent( 1 ), coords.extent( 2 ) } ),
        KOKKOS_LAMBDA( const int s, const int x, const int y ) {
            const double rx  = coords( s, x, y, 0 );
            const double ry  = coords( s, x, y, 1 );
            const double rz  = coords( s, x, y, 2 );
            const double vx  = rx + eps * Kokkos::sin( 3.0 * ry + phase );
            const double vy  = ry + eps * Kokkos::cos( 2.0 * rz + phase );
            const double vz  = rz + eps * Kokkos::sin( 2.0 * rx + phase );
            const double inv = 1.0 / Kokkos::sqrt( vx * vx + vy * vy + vz * vz );
            n( s, x, y, 0 )  = inv * vx;
            n( s, x, y, 1 )  = inv * vy;
            n( s, x, y, 2 )  = inv * vz;
        } );
    Kokkos::fence();
    return n;
}

} // namespace terra::test
