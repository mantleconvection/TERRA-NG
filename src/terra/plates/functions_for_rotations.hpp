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

// #include "terraneo/helpers/conversions.hpp"
// #include <Eigen/Array>
#include "terra/plates/types.hpp"

namespace terra {
namespace plates {

using plates::FiniteRotation;
using plates::rotIter_t;

/// Computes a matrix for rotation around a given axis vector by given angle
inline mat3D xyzRotationMatrix( const vec3D& vector, double angle )
{
   mat3D  rotMat;
   double x       = vector( 0 );
   double y       = vector( 1 );
   double z       = vector( 2 );
   rotMat( 0, 0 ) = std::cos( angle ) + x * x * ( static_cast< double >( static_cast< double >( 1 ) ) - std::cos( angle ) );
   rotMat( 0, 1 ) = x * y * ( static_cast< double >( 1 ) - std::cos( angle ) ) - z * std::sin( angle );
   rotMat( 0, 2 ) = x * z * ( static_cast< double >( 1 ) - std::cos( angle ) ) + y * std::sin( angle );
   rotMat( 1, 0 ) = y * x * ( static_cast< double >( 1 ) - std::cos( angle ) ) + z * std::sin( angle );
   rotMat( 1, 1 ) = std::cos( angle ) + y * y * ( static_cast< double >( 1 ) - std::cos( angle ) );
   rotMat( 1, 2 ) = y * z * ( static_cast< double >( 1 ) - std::cos( angle ) ) - x * std::sin( angle );
   rotMat( 2, 0 ) = z * x * ( static_cast< double >( 1 ) - std::cos( angle ) ) - y * std::sin( angle );
   rotMat( 2, 1 ) = z * y * ( static_cast< double >( 1 ) - std::cos( angle ) ) + x * std::sin( angle );
   rotMat( 2, 2 ) = std::cos( angle ) + z * z * ( static_cast< double >( 1 ) - cos( angle ) );
   return rotMat;
}

/// Obtains the rotation matrix to move a plate polygon with a center to (0,0,0)
inline mat3D getRotationMatrixPolygon( const vec3D& axis )
{
   double xm    = axis( 0 );
   double ym    = axis( 1 );
   double zm    = axis( 2 );
   double norm  = std::sqrt( xm * xm + ym * ym );
   double angle = std::atan2( norm, zm );
   return xyzRotationMatrix( { ym / norm, -xm / norm, static_cast< double >( 0 ) }, angle );
}

/// Determine rotation matrix for a finite rotation
inline mat3D rotationMatrix( const double lon, const double lat, const double angleInDegree )
{
   double angleInRadians = conversions::degToRad( angleInDegree );
   vec3D  eXYZ           = conversions::sph2cart( { lon, lat } );
   return xyzRotationMatrix( eXYZ, angleInRadians );
}

/// Calculate longitude, latitude and angle of rotation for the reconstruction path from the data of the rotational file
inline vec3D rotMatrix2LonLatW( const mat3D& Rot )
{
   vec3D  lonlatang;
   double sqrtRot = sqrt( ( Rot( 2, 1 ) - Rot( 1, 2 ) ) * ( Rot( 2, 1 ) - Rot( 1, 2 ) ) +
                          ( Rot( 0, 2 ) - Rot( 2, 0 ) ) * ( Rot( 0, 2 ) - Rot( 2, 0 ) ) +
                          ( Rot( 1, 0 ) - Rot( 0, 1 ) ) * ( Rot( 1, 0 ) - Rot( 0, 1 ) ) );
   // latitude
   if ( sqrtRot == static_cast< double >( 0 ) ) // <- direct comparision of FP value for zero?
                                 // should probably add a tolerance
   {
      lonlatang(1) = static_cast< double >( 0 );
   }
   else
   {
      lonlatang(1) = asin( ( Rot( 1, 0 ) - Rot( 0, 1 ) ) / sqrtRot );
   }

   // longitude
   lonlatang(0) = atan2( Rot( 0, 2 ) - Rot( 2, 0 ), Rot( 2, 1 ) - Rot( 1, 2 ) );

   // angle
   lonlatang(2) = atan2( sqrtRot, Rot( 0, 0 ) + Rot( 1, 1 ) + Rot( 2, 2 ) - static_cast< double >( 1 ) );
   lonlatang    = conversions::radToDeg( lonlatang );

   return lonlatang;
}

/// Calculate stage Pole for a pair of rotations
inline vec3D stagePoleF( const vec3D& rot1, const vec3D& rot2 )
{
   mat3D R1     = rotationMatrix( rot1(0), rot1(1), rot1(2) );
   mat3D R2     = rotationMatrix( rot2(0), rot2(1), rot2(2) * static_cast< double >( -1 ) );
   mat3D stageR = R1 * R2;
   return rotMatrix2LonLatW( stageR );
}

/// Get the intermediate rotations of the reconstruction tree
inline vec3D
    intermediateRotations( const plates::RotationInfo& rot1, const plates::RotationInfo& rot2, double time )
{
   double dT        = ( rot2.time - time ) / ( rot2.time - rot1.time );
   mat3D  R2        = rotationMatrix( rot2.longitude, rot2.latitude, rot2.angle );
   vec3D  lonlatang = stagePoleF( { rot1.longitude, rot1.latitude, rot1.angle }, { rot2.longitude, rot2.latitude, rot2.angle } );
   mat3D  stgR      = rotationMatrix( lonlatang(0), lonlatang(1), lonlatang(2) * dT );
   mat3D  intR      = stgR * R2;

   return rotMatrix2LonLatW( intR );
}

/// Given the input data from the rotational find the lines of the specific
/// time asked and get the rotations
inline int determineSeriesOfFiniteRotations( const rotIter_t&               rangeBegin,
                                             const rotIter_t&               rangeEnd,
                                             const std::array< double, 2 >& time,
                                             std::vector< FiniteRotation >& finRot )
{
   std::vector< plates::RotationInfo > fin0;
   int                                           pID = -1;

   plates::RotationInfo aux;
   aux.time        = static_cast< double >( 0 );
   aux.longitude   = static_cast< double >( 0 );
   aux.latitude    = static_cast< double >( 90 );
   aux.angle       = static_cast< double >( 0 );
   aux.plateID     = 0;
   aux.conjugateID = 0;
   fin0.push_back( aux );

   for ( rotIter_t rot = rangeBegin; rot != rangeEnd; ++rot )
   {
      fin0.push_back( *rot );
   }

   for ( int ii = 0; ii < time.size(); ++ii )
   {
      double t = time[ii];
      // keep it flexible for possible for finite rotation ending with the age we are interested in.
      // thus age time+1 will not exist, and we won't be able to calculate the velocity. 
      // fix take the approach of [time-1, time]
      if (time[time.size()-1] > fin0[fin0.size()-1].time)
      {
         t = time[ii]-1;
      }
      
      // determine in which position of the asked time to grab the corresponding lines
      //
      // NOTE: if loop does not return a hit, the 0 remains unchanged!
      int jj = 0;
      for ( int idx = 0; idx < fin0.size(); ++idx )
      {
         if ( fin0[idx].time - t >= static_cast< double >( 0 ) )
         {
            jj = idx;
            break;
         }
      }

      if ( fin0.size() - 2 > jj )
      {
         if ( int( time[0] ) == int( fin0[jj].time ) && fin0[jj].time == fin0[jj + 1].time )
            jj = jj + 2;
      }

      vec3D lonlatAng = intermediateRotations( fin0[jj - 1], fin0[jj], t );
      finRot.push_back( { t, lonlatAng } );
      if (ii == 0){
         pID = fin0[jj-1].conjugateID;
      }
   }

   return pID;
}

/// Combine a sequence of finite rotations
inline std::array< FiniteRotation, 2 > combineSeriesOfFiniteRotations( const std::vector< FiniteRotation >& FinRot )
{
   std::array< FiniteRotation, 2 > absFin;

   for ( int jj = 0; jj < 2; ++jj )
   {
      absFin[jj] = { static_cast< double >( 0 ), { static_cast< double >( 0 ), static_cast< double >( 90 ), static_cast< double >( 0 ) } };

      for ( int ii = jj; ii < FinRot.size(); ii = ii + 2 )
      {
         mat3D Rot1           = rotationMatrix( absFin[jj].lonLatAng(0), absFin[jj].lonLatAng(1), absFin[jj].lonLatAng(2) );
         mat3D Rot2           = rotationMatrix( FinRot[ii].lonLatAng(0), FinRot[ii].lonLatAng(1), FinRot[ii].lonLatAng(2) );
         Rot1                 = Rot2 * Rot1;
         absFin[jj].lonLatAng = rotMatrix2LonLatW( Rot1 );
         absFin[jj].time      = FinRot[ii].time;
      }
   }

   return absFin;
}

} // namespace plates
} // namespace terra
