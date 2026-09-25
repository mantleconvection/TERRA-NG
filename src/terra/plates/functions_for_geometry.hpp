/*
 * Copyright (c) 2022 Berta Vilacis, Marcus Mohr, Fatemeh Rezaei. 
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

// #include "terraneo/helpers/conversions.hpp"

#include <cassert>
#include "terra/plates/types.hpp"
#include "terra/plates/conversions.hpp"

namespace terra{
namespace plates {


/// Given two points on a sphere with the radius of the Earth, it returs the
/// distance between them in km using the Haversine formula
inline double distancePointPoint( const vec3D& lonlat1, const vec3D& lonlat2 )
{
   double phi1 = conversions::degToRad( lonlat1(1) );
   double phi2 = conversions::degToRad( lonlat2(1) );
   double dphi = conversions::degToRad( lonlat2(1) - lonlat1(1) );
   double dlam = conversions::degToRad( lonlat2(0) - lonlat1(0) );

   double a = std::sin( dphi * static_cast< double >( 0.5 ) ) * std::sin( dphi * static_cast< double >( 0.5 ) ) +
              std::cos( phi1 ) * std::cos( phi2 ) * std::sin( dlam * static_cast< double >( 0.5 ) ) * std::sin( dlam * static_cast< double >( 0.5 ) );
   double c = static_cast< double >( 2 ) * std::atan2( std::sqrt( a ), std::sqrt( static_cast< double >( 1 ) - a ) );
   return c * plates::constants::earthRadiusInKm;
}

vec3D cart2sph( const vec3D& xyz )
{
   vec3D lonlatrad;
   lonlatrad(0) = conversions::radToDeg( atan2( xyz(1), xyz(0) ) );
   lonlatrad(1) = conversions::radToDeg( atan2( xyz(2), sqrt( xyz(0) * xyz(0) + xyz(1) * xyz(1) ) ) );
   lonlatrad(2) = xyz.norm();

   return lonlatrad;
}

/// The following function that calculates the intersection between a point
/// and a line is inspired by the Python code from geopy, see:
/// https://github.com/geopy/geopy/blob/master/geopy/distance.py
inline vec3D intersectPointWithLine( const vec3D& point, const vec3D& lineStart, const vec3D& lineEnd )
{
   assert( lineStart(0) != lineEnd(0) && lineStart(1) != lineEnd(1) && lineStart(2) != lineEnd(2) );

   std::cout << "Line in intersectPointWithLine() should not be degenerate.\n";

   // find the intersect point (the projected point) on the line
   vec3D  lineVec    = ( lineEnd - lineStart );
   double lineLength = lineVec.norm();

   // calculate the magnitude of the interesection point
   double mag = 0.0;
   mag        = ( point - lineStart ).dot( lineVec );
   mag        = mag / ( lineLength * lineLength );
   vec3D intersection;

   // if closest point does not fall within the line segment, take the shorter distance to an endpoint
   double eps = static_cast< double >( 1e-5 );
   if ( mag < eps || mag > static_cast< double >( 1 ) )
   {
      double ix = ( point - lineStart ).norm();
      double iy = ( point - lineEnd ).norm();
      if ( ix > iy )
      {
         intersection = lineEnd;
      }
      else
      {
         intersection = lineStart;
      }
   }
   else
   {
      intersection = lineStart + mag * lineVec;
   }

   return intersection;
}

/// Calls all the functions to return the distance between the point and the line segment that
/// constructs the polygon, the returned value is in km
inline double getDistanceLinePoint( const vec3D& point, const vec3D& pini, const vec3D& pend )
{
   vec3D intersectPoint = intersectPointWithLine( point, pini, pend );
   vec3D pintersect     = conversions::cart2sph( intersectPoint );
   vec3D ppoint         = conversions::cart2sph( point );
   return distancePointPoint( ppoint, pintersect );
}

/// Given some normal (cartesian) vector on the surface of a sphere, computes two normalized (cartesian) vectors that span the
/// tangential space.
inline std::pair< vec3D, vec3D > findOrthogonalVectorsCart( const vec3D& normalCart )
{
   // Choose an arbitrary vector (not parallel) to normal
   double smallest = std::fabs( normalCart(0) );
   vec3D  u1{ 1, 0, 0 };

   if ( std::fabs( normalCart(1) ) < smallest )
   {
      smallest = std::fabs( normalCart(1) );
      u1       = vec3D{ 0, 1, 0 };
   }

   if ( std::fabs( normalCart(2) ) < smallest )
   {
      u1 = vec3D{ 0, 0, 1 };
   }

   // Compute the first orthogonal vector
   u1 = normalCart.cross( u1 ).normalized();

   // Compute the second orthogonal vector
   vec3D u2 = normalCart.cross( u1 ).normalized();

   return { u1, u2 };
}

} // namespace plates
} // namespace terra