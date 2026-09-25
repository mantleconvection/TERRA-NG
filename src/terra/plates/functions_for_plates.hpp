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

#include <limits>
#include <vector>
#include <array>
#include <stdexcept>
#include <string>

#include "terra/dense/vec.hpp"
#include "terra/plates/conversions.hpp"
#include "terra/plates/functions_for_geometry.hpp"
#include "terra/plates/functions_for_rotations.hpp"
#include "terra/plates/types.hpp"
#include "terra/plates/plate_storage.hpp"
#include "terra/plates/spherical_predicates.hpp"

namespace terra {
namespace plates {

inline vec3D sph2cart( const std::vector< double >& lonlat, const double radius = static_cast< double >( 1 ) )
{
   vec3D xyz;
   xyz(0) = radius * cos( conversions::degToRad( lonlat[1] ) ) * cos( conversions::degToRad( lonlat[0] ) );
   xyz(1) = radius * cos( conversions::degToRad( lonlat[1] ) ) * sin( conversions::degToRad( lonlat[0] ) );
   xyz(2) = radius * sin( conversions::degToRad( lonlat[1] ) );
   return xyz;
}

/// Determine to which plate a point belongs
///
/// The function returns a bool to indicate whether any plate matched, the plate's ID and
/// the distance from this plate's boundary in km. The first matching plate wins.
///
/// \param point longitude, latitude and radius, with the angles in degrees
inline std::tuple< bool, uint_t, double >
    findPlateAndDistance( const double age, const PlateStorage& plateStore, const vec3D& point, uint_t idWhenNoPlateFound )
{
   // query all plates for given age stage
   const auto& plates = plateStore.getPlatesForStage( std::ceil( age ) );

   const geometry::UnitVec pointCart = geometry::lonLatDegToUnit( point( 0 ), point( 1 ) );

   for ( const auto& currentPlate : plates )
   {
      const Polygon& bdrPolygon = currentPlate.boundary;

      // The polygon is stored as (longitude, latitude, 0) per vertex; convert on the fly here.
      // Phase 3 of the port replaces this with unit cartesian vertices held in a flat buffer, at
      // which point the trigonometry leaves the inner loop entirely.
      const auto result = geometry::pointInSphericalPolygon(
          pointCart,
          static_cast< int >( bdrPolygon.size() ),
          [&bdrPolygon]( const int index ) {
             return geometry::lonLatDegToUnit( bdrPolygon[index]( 0 ), bdrPolygon[index]( 1 ) );
          } );

      if ( result.inside )
      {
         return std::make_tuple( true, currentPlate.id, result.distanceRad * plates::constants::earthRadiusInKm );
      }
   }

   return std::make_tuple( false, idWhenNoPlateFound, std::numeric_limits< double >::max() );
}

/// From the Euler vector compute the surface velocity in xyz
KOKKOS_INLINE_FUNCTION vec3D eulerVectorToVelocity( const vec3D& point, const vec3D& wXYZ, const double smoothing )
{
   double earthRadius = plates::constants::earthRadiusInKm * static_cast< double >( 1e3 );
   double toms        = static_cast< double >( 3600 * 24 * 365 ); // conversions factor cm/yr -> m/s

   vec3D eVector;
   vec3D pxyz;

   eVector = conversions::degToRad( wXYZ ) * static_cast< double >( 1e-6 );

   // Transform to the point to the xyz in a sphere of earthRadius;
   pxyz    = conversions::sph2cart( point( 0 ), point( 1 ), earthRadius );
   vec3D v = eVector.cross( pxyz );

   v(0) *= smoothing / toms;
   v(1) *= smoothing / toms;
   v(2) *= smoothing / toms;

   return v;
}

/// Euler vector (cartesian, degrees per Ma) of a plate at a given age stage
///
/// This walks the reconstruction tree from the plate up to the reference frame and forms the
/// stage pole over [age, age+1]. It depends only on the plate and the age -- the point does not
/// enter until eulerVectorToVelocity(). Split out of computeCartesianVelocityVector() so the
/// tree walk can be hoisted out of the per-point loop; see
/// PlateVelocityProvider::prepareEulerVectors().
inline vec3D computeEulerVector( const PlateRotationProvider& rotData, const int plateID, const double age )
{
   // age of the euler pole is defined by ((age1 + age2)/2)
   // This is valid when the velocities are calculated every 1 Myrs. 
   // Otherwise this needs to be changed to the desired time step
   // taking into conserdation the time resolution of the plate boundaries available.
   std::array< double, 2 >     time{ age, age + 1 };
   std::vector< RotationInfo > recTree;

   std::vector< FiniteRotation >      FinRot;
   const std::vector< RotationInfo >& rotations = rotData.getRotations();

   int pID = plateID;

   using rotIter_t = std::vector< RotationInfo >::const_iterator;

   // Walk the reconstruction circuit from this plate down to the anchor (id 0), accumulating one series of
   // finite rotations per hop.
   //
   // Two ways this can fail to terminate on its own, both of which are data problems rather than programming
   // errors, and both of which used to hang here rather than report anything:
   //
   //   * no rotation entry carries the current id -- the search below matches nothing, `pID` is left unchanged
   //     and the loop spins. This is what happens when a topology file names plates that the rotation file does
   //     not describe, i.e. when the two files come from different reconstructions.
   //   * the circuit contains a cycle, so the walk revisits an id it has already left.
   //
   // Both are caught: the first directly, the second by bounding the number of hops, since a well-formed
   // circuit can visit each plate at most once.
   const size_t max_hops = rotations.size() + 1;
   size_t       hops     = 0;

   while ( pID != 0 )
   {
      bool found = false;

      for ( rotIter_t it = rotations.begin(); it != rotations.end(); ++it )
      {
         if ( it->plateID == pID )
         {
            rotIter_t rangeBegin = it;
            rotIter_t rangeEnd   = rangeBegin + 1;

            // Bound the scan by end(): a matching block that runs to the last entry would otherwise read past
            // the end of the vector.
            while ( rangeEnd != rotations.end() && rangeEnd->plateID == pID )
            {
               ++rangeEnd;
            }

            // append to list of finite rotations
            pID   = plates::determineSeriesOfFiniteRotations( rangeBegin, rangeEnd, time, FinRot );
            found = true;
            break;
         }
      }

      if ( !found )
      {
         throw std::runtime_error(
             "computeEulerVector: no rotation data for plate id " + std::to_string( pID ) +
             " while building the reconstruction circuit of plate " + std::to_string( plateID ) + " at age " +
             std::to_string( age ) +
             " Ma. The topology and rotation files most likely come from different reconstructions." );
      }

      if ( ++hops > max_hops )
      {
         throw std::runtime_error(
             "computeEulerVector: the reconstruction circuit of plate " + std::to_string( plateID ) + " at age " +
             std::to_string( age ) + " Ma does not terminate at the anchor plate; the rotation data contains a cycle." );
      }
   }

   std::array< FiniteRotation, 2 > finNahs = plates::combineSeriesOfFiniteRotations( FinRot );

   // compute Euler Vector
   vec3D lonlatang = plates::stagePoleF( finNahs[0].lonLatAng, finNahs[1].lonLatAng );
   lonlatang(2)    = lonlatang(2) / ( finNahs[1].time - finNahs[0].time );

   return conversions::sph2cart( { lonlatang(0), lonlatang(1) }, lonlatang(2) );
}

/// Get the velocity in given the plate id, create the reconstruction path, get
/// the rotations and calculate the velocity
///
/// Recomputes the Euler vector on every call. Prefer computeEulerVector() once per
/// (plate, age) plus eulerVectorToVelocity() per point.
inline vec3D computeCartesianVelocityVector( const PlateRotationProvider& rotData,
                                             const int                    plateID,
                                             const double                 age,
                                             const vec3D&                 point,
                                             const double                 smoothing )
{
   return eulerVectorToVelocity( point, computeEulerVector( rotData, plateID, age ), smoothing );
}

} // namespace plates
} // namespace terra
