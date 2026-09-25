/*
 * Copyright (c) 2022-2025 Berta Vilacis, Marcus Mohr, Nils Kohl.
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

namespace terra{
namespace plates {

/// \defgroup SmoothingStrategies SmoothingStrategies
///
/// In order to avoid problems with discontinuous Dirichlet boundary conditions
/// we can "smooth" plate velocities such that the become zero at the interfaces
/// between plates.
/// @{

/// Does not apply any smoothing.
class NoSmoothing
{
 public:
   double operator()( const double distanceFromBoundary ) const { return 1.0; }
};

/// Linearly decrease smoothing factor from 1.0 to 0.0
///
/// The velocity of a point is scaled by a smoothing factor that linearly
/// increases with distance of the point from the boundary from 0.0 with
/// the given slope and is capped at 1.0.
class LinearDistanceSmoother
{
 public:
   LinearDistanceSmoother( double slope )
   : slope_( slope ) {};

   double operator()( const double distanceFromBoundary ) const
   {
      double smoothing = distanceFromBoundary * slope_;
      return smoothing > static_cast< double >( 1 ) ? static_cast< double >( 1 ) : smoothing;
   }

 private:
   double slope_{ static_cast< double >( 0 ) };
};

/// @}
} // namespace plates
} // namespace terra