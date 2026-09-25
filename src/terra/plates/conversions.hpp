/*
 * Copyright (c) 2022 Berta Vilacis, Marcus Mohr.
 *
 * This file is part of HyTeG
 * (see https://i10git.cs.fau.de/hyteg/hyteg).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

// #include "core/math/Constants.h"

// #include "terraneo/helpers/typeAliases.hpp"
#include "terra/dense/vec.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"

using vec3D = terra::dense::Vec< double, 3 >; 
namespace terra{
namespace plates::conversions {

inline constexpr double pi = 3.14159265358979323846;

/// Transforms angle from degrees to radians
KOKKOS_INLINE_FUNCTION double degToRad( double degree )
{
   return ( degree * ( pi / static_cast< double >( 180 ) ) );
}

/// Transforms vector of angles componentwise from degrees to radians
KOKKOS_INLINE_FUNCTION vec3D degToRad( vec3D degree )
{
   return ( degree * ( pi / static_cast< double >( 180 ) ) );
}

/// Transforms angle from radians to degrees
KOKKOS_INLINE_FUNCTION double radToDeg( double radian )
{
   return ( radian * ( static_cast< double >( 180 ) / pi ) );
}

/// Transforms vector of angles componentwise from radians to degrees
KOKKOS_INLINE_FUNCTION vec3D radToDeg( vec3D radian )
{
   return ( radian * ( static_cast< double >( 180 ) / pi ) );
}

/// Transform 3D vector from spherical to cartesian coordintates
///
/// Transform 3D vector from spherical coordinates (lon, lat, rad) to cartesian
/// ones (x,y,z), is radius is not given assume unit point on sphere.
inline vec3D sph2cart( const std::vector< double >& lonlat, const double radius = static_cast< double >( 1 ) )
{
   vec3D xyz;
   xyz(0) = radius * cos( degToRad( lonlat[1] ) ) * cos( degToRad( lonlat[0] ) );
   xyz(1) = radius * cos( degToRad( lonlat[1] ) ) * sin( degToRad( lonlat[0] ) );
   xyz(2) = radius * sin( degToRad( lonlat[1] ) );
   return xyz;
}

/// Same, from lon/lat given as scalars.
///
/// Device-callable counterpart of the std::vector overload above: the plate velocity kernels need this inside
/// a Kokkos parallel region, where std::vector is not available.
KOKKOS_INLINE_FUNCTION vec3D
    sph2cart( const double lon, const double lat, const double radius = static_cast< double >( 1 ) )
{
   vec3D xyz;
   xyz( 0 ) = radius * Kokkos::cos( degToRad( lat ) ) * Kokkos::cos( degToRad( lon ) );
   xyz( 1 ) = radius * Kokkos::cos( degToRad( lat ) ) * Kokkos::sin( degToRad( lon ) );
   xyz( 2 ) = radius * Kokkos::sin( degToRad( lat ) );
   return xyz;
}

/// Transform 3D vector from cartesian (x, y, z) to spherical coordinates (lon, lat, rad)
KOKKOS_INLINE_FUNCTION vec3D cart2sph( const vec3D& xyz )
{
   vec3D lonlatrad;
   lonlatrad(0) = radToDeg( Kokkos::atan2( xyz(1), xyz(0) ) );
   lonlatrad(1) = radToDeg( Kokkos::atan2( xyz(2), Kokkos::sqrt( xyz(0) * xyz(0) + xyz(1) * xyz(1) ) ) );
   lonlatrad(2) = xyz.norm();

   return lonlatrad;
}

} // namespace plates::conversions
} // namespace terra 