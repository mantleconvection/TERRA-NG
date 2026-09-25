/*
 * Copyright (c) 2022-2025 Berta Vilacis, Marcus Mohr, Nils Kohl, Andreas Burkhart, Fatemeh Rezaei.
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

#include <set>

#include "terra/plates/plate_not_found_handlers.hpp"
#include "terra/plates/plate_rotation_provider.hpp"
#include "terra/plates/plate_stage_data.hpp"
#include "terra/plates/plate_storage.hpp"
#include "terra/plates/smoothing_strategies.hpp"
#include "terra/plates/conversions.hpp"

// preserve ordering of includes
#include "terra/plates/file_io.hpp"
#include "terra/plates/local_averaging_point_weight_provider.hpp"
#include "terra/plates/functions_for_plates.hpp"

namespace terra {
namespace plates {

/// API class for computation of velocities from plate reconstructions
class PlateVelocityProvider
{
  public:
    PlateVelocityProvider( std::string nameOfTopologiesFile, std::string nameOfRotationsFile )
    : plateTopologies_(
          nameOfTopologiesFile,
          static_cast< double >( 1 ),
          []( const std::string& filename ) { return io::readJsonFile( filename ); } )
    , plateRotations_( nameOfRotationsFile, []( const std::string& filename ) {
        return io::readRotationsFile( filename );
    } )
    {}

    /// Returns the plate ID for a point at a given age stage
    ///
    /// This is basically an auxilliary function for testing plate detection and allows to
    /// generate data to visualise plate movement.
    uint_t findPlateID( const vec3D& point, const double age ) const
    {
        uint_t plateID{ idWhenNoPlateFound };
        bool   plateFound{ false };
        double distance{ static_cast< double >( -1 ) };

        // Transform the point to Lon, Lat, Radius - to perform all caculations
        // We use the Lon, Lat coordinates
        const vec3D pointLonLat = conversions::cart2sph( point );

        std::tie( plateFound, plateID, distance ) =
            findPlate( pointLonLat, age );
        return plateID;
    }

    // This function checks if a point is on/very close to the boundary at a certain age
    // Primarily written to be used in the viscosity function of a Circulation Model
    bool findPlateBoundaries( const vec3D& point, const double age )
    {
        double eps = static_cast< double >( 1e-2 ); // 3e-2
        uint_t plateID{ 0 };
        bool   plateFound{ false };
        double distance{ static_cast< double >( -1 ) };

        // Transform the point to Lon, Lat, Radius - to perform all calculations
        // We use the Lon, Lat coordinates
        const vec3D pointLonLat = conversions::cart2sph( point );

        std::tie( plateFound, plateID, distance ) =
            findPlate( pointLonLat, age );
        distance /= plates::constants::earthRadiusInKm;

        if ( distance < eps )
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /// Returns velocity vector for a point determined from the velocity of the associated plate at given age stage
    ///
    /// This is the convenience (non-expert) version of the method which uses a
    /// - LinearDistanceSmoother{ 0.015 }
    /// - DefaultPlateNotFoundHandler{}
    vec3D getPointVelocity( const vec3D& point, const double age )
    { return getPointVelocity( point, age, LinearDistanceSmoother{ 0.015 }, DefaultPlateNotFoundHandler{} ); }

    /// Alternative: Interpolated linearly in time between the current and next plate age stage. Defaults to the boundaries of
    /// plateTopologies_.getListOfPlateStages() if age lies out of bounds of plateTopologies_.getListOfPlateStages().
    vec3D getPointVelocityInterpolatedInTime( const vec3D& point, const double age )
    {
        return getPointVelocityInterpolatedInTime(
            point, age, LinearDistanceSmoother{ 0.015 }, DefaultPlateNotFoundHandler{} );
    }

    /// Returns velocity vector for a point determined from the velocity of the associated plate at given age stage
    ///
    /// This is the expert version of the method which allows to explicitly set a SmoothingStrategy and
    /// a PlateNotFoundStrategy.
    template < typename SmoothingStrategy, typename PlateNotFoundStrategy >
    vec3D getPointVelocity(
        const vec3D&            point,
        const double            age,
        SmoothingStrategy       computeSmoothing,
        PlateNotFoundStrategy&& errorHandler )
    {
        uint_t plateID{ 0 };
        bool   plateFound{ false };
        double distance{ static_cast< double >( -1 ) };

        // Transform the point to Lon, Lat, Radius - to perform all calculations
        // We use the Lon, Lat coordinates
        vec3D pointLonLat = conversions::cart2sph( point );

        std::tie( plateFound, plateID, distance ) =
            findPlate( pointLonLat, age );

        if ( !plateFound )
        {
            return errorHandler( point, age );
        }
        // else
        // {
        //    WALBERLA_LOG_DETAIL_ON_ROOT( "Point found on plate with ID = " << plateID << ", distance to boundary = " << distance );
        // }

        const double smoothingFactor = computeSmoothing( distance );
        // if ( mpi::rank == 0 )
        // {
        logroot << "Smoothing Factor: " << smoothingFactor << "\n"
                << "Plate ID: " << plateID << std::endl;
        // }

        return eulerVectorToVelocity( pointLonLat, eulerVectorFor( plateID, age ), smoothingFactor );
    }

    /// Find the age in the plateStages list and return the surrounding ages in the list
    /// Crop to plateStages if age lies outside.
    ///
    /// Returns a double tuple s.t. tuple[0] <= age <= tuple[1].
    /// tuple[2] is a linear interpolation factor s.t. age = (1-tuple[2])*tuple[0] + tuple[2]*tuple[1]
    std::tuple< double, double, double > getSurroundingAges( const double age )
    {
        std::vector< double > plateStages = plateTopologies_.getListOfPlateStages();
        // find the age in the plateStages list, crop to plateStages if age lies outside
        uint_t indexFloor;
        uint_t indexCeil;
        double interpolationFactor;
        if ( age >= plateStages.back() )
        {
            indexFloor          = plateStages.size() - 1;
            indexCeil           = indexFloor;
            interpolationFactor = static_cast< double >( 1.0 );
        }
        else if ( age <= plateStages.front() )
        {
            indexFloor          = 0;
            indexCeil           = 0;
            interpolationFactor = static_cast< double >( 1.0 );
        }
        else
        {
            auto iteratorLowerBound = std::lower_bound( plateStages.begin(), plateStages.end(), age );
            indexCeil               = static_cast< uint_t >( std::distance( plateStages.begin(), iteratorLowerBound ) );
            indexFloor              = indexCeil - 1;
            interpolationFactor =
                ( age - plateStages.at( indexFloor ) ) / ( plateStages.at( indexCeil ) - plateStages.at( indexFloor ) );
        }

        return { plateStages.at( indexFloor ), plateStages.at( indexCeil ), interpolationFactor };
    }

    /// Returns velocity vector for a point determined from the velocity of the associated plate at given age stage
    ///
    /// Interpolated linearly in time between the current and next plate age stage. Defaults to the boundaries of
    /// plateTopologies_.getListOfPlateStages() if age lies out of bounds of plateTopologies_.getListOfPlateStages().
    ///
    /// This is the version using a SmoothingStrategy like LinearDistanceSmoother.
    template < typename SmoothingStrategy, typename PlateNotFoundStrategy >
    vec3D getPointVelocityInterpolatedInTime(
        const vec3D&            point,
        const double            age,
        SmoothingStrategy       computeSmoothing,
        PlateNotFoundStrategy&& errorHandler )
    {
        // find surrounding plate ages and interpolation factor
        double ageFloor;
        double ageCeil;
        double interpolationFactor;
        std::tie( ageFloor, ageCeil, interpolationFactor ) = getSurroundingAges( age );

        // call getPointVelocity twice
        vec3D vecFloor = getPointVelocity( point, ageFloor, computeSmoothing, errorHandler );
        vec3D vecCeil  = getPointVelocity( point, ageCeil, computeSmoothing, errorHandler );

        // linear interpolation in time
        return vecFloor + interpolationFactor * ( vecCeil - vecFloor );
    }

    /// Computes a weighted average of the velocity around the given point using the provided point and weights.
    ///
    /// Averaging is only applied if at least one of the provided points is located on a different plate.
    template < typename PlateNotFoundStrategy >
    vec3D getLocallyAveragedPointVelocity(
        const vec3D&                             point,
        const double                             age,
        const LocalAveragingPointWeightProvider& pointWeightProvider,
        PlateNotFoundStrategy&&                  errorHandler )
    {
        uint_t plateID{ 0 };
        bool   plateFound{ false };
        double distance{ static_cast< double >( -1 ) };

        // Transform the point to Lon, Lat, Radius - to perform all calculations
        // We use the Lon, Lat coordinates
        const vec3D pointLonLat = conversions::cart2sph( point );

        std::tie( plateFound, plateID, distance ) =
            findPlate( pointLonLat, age );

        if ( !plateFound )
        {
            return errorHandler( point, age );
        }

        vec3D  avgVelCart{ 0, 0, 0 };
        double weightSum = 0;

        if ( pointWeightProvider.maxDistance( pointLonLat ) < distance )
        {
            // We do not apply averaging since all points that would be used for averaging are on the same plate.
            // if ( mpi::rank() == 0 )
            // {
            // logroot << "No averaging.\n" << "Plate ID: " << plateID << std::endl;
            // }

            return eulerVectorToVelocity( pointLonLat, eulerVectorFor( plateID, age ), 1.0 );
        }

        const auto pointsAndWeights = pointWeightProvider.samplePointsAndWeightsLonLat( pointLonLat );

        // We average since at least some of the samples are possibly on at least one other plate.
        for ( const auto& [samplePointSphLonLat, weight] : pointsAndWeights )
        {
            uint_t avgPointPlateID{ 0 };
            bool   avgPointPlateFound{ false };
            double avgPointDistance{ static_cast< double >( -1 ) };

            std::tie( avgPointPlateFound, avgPointPlateID, avgPointDistance ) =
                findPlate( samplePointSphLonLat, age );

            if ( avgPointPlateFound )
            {
                // This is possibly slightly inaccurate since we are averaging over the cartesian velocity vectors and then projecting
                // out the normal component. It would be better to average in the "lonlat-space" and then convert and return the
                // cartesian vector. On the other hand, averaging the plate velocities is already a somewhat arbitrary and physically
                // meaningless approximation in the first place, so this might just work.
                const vec3D sampleVel =
                    eulerVectorToVelocity( samplePointSphLonLat, eulerVectorFor( avgPointPlateID, age ), 1.0 );

                for ( int d = 0; d < 3; ++d )
                {
                    avgVelCart( d ) += weight * sampleVel( d );
                }
                weightSum += weight;
            }
        }

        avgVelCart( 0 ) /= weightSum;
        avgVelCart( 1 ) /= weightSum;
        avgVelCart( 2 ) /= weightSum;

        const auto n                = point.normalized();
        const auto dot              = n.dot( avgVelCart );
        const auto normalComponent  = dot * n;
        const auto tangentComponent = avgVelCart - normalComponent;

        return tangentComponent;
    }

    /// Returns velocity vector for a point determined from the velocity of the associated plate at given age stage
    ///
    /// Interpolated linearly in time between the current and next plate age stage. Defaults to the boundaries of
    /// plateTopologies_.getListOfPlateStages() if age lies out of bounds of plateTopologies_.getListOfPlateStages().
    ///
    /// This is the version using a weighted average of the velocity around the given point given a LocalAveragingPointWeightProvider.
    template < typename PlateNotFoundStrategy >
    vec3D getLocallyAveragedPointVelocityInterpolatedInTime(
        const vec3D&                             point,
        const double                             age,
        const LocalAveragingPointWeightProvider& pointWeightProvider,
        PlateNotFoundStrategy&&                  errorHandler )
    {
        // find surrounding plate ages and interpolation factor
        double ageFloor;
        double ageCeil;
        double interpolationFactor;
        std::tie( ageFloor, ageCeil, interpolationFactor ) = getSurroundingAges( age );

        // call getLocallyAveragedPointVelocity twice
        vec3D vecFloor = getLocallyAveragedPointVelocity( point, ageFloor, pointWeightProvider, errorHandler );
        vec3D vecCeil  = getLocallyAveragedPointVelocity( point, ageCeil, pointWeightProvider, errorHandler );

        // linear interpolation in time
        return vecFloor + interpolationFactor * ( vecCeil - vecFloor );
    }

    /// Query function to obtain a vector of plate stages available in the datafiles
    const std::vector< double >& getListOfPlateStages() const { return plateTopologies_.getListOfPlateStages(); }

    /// Plate ID to be used when no associated plate was found for a point
    const uint_t idWhenNoPlateFound{ 0 };

    double getMinAge() const { return plateTopologies_.getMinAge(); }
    double getMaxAge() const { return plateTopologies_.getMaxAge(); }

    /// Precomputes the Euler vector of every plate of the stage containing \p age
    ///
    /// computeEulerVector() walks the whole reconstruction tree and depends only on the plate and
    /// the age, but the velocity queries need it per sample point. With O(50) plates per stage
    /// against O(10^5) surface points -- times seven sample points, times two bracketing stages --
    /// hoisting it here turns the dominant cost of the velocity path into a handful of evaluations.
    ///
    /// Calling this is optional: eulerVectorFor() falls back to computing on the fly. It is not
    /// optional for correctness under a threaded host backend, though -- the fallback is const and
    /// race-free, but only a prepared table keeps the per-point path from being slow.
    void prepareEulerVectors( const double age )
    {
        if ( stages_.find( age ) == stages_.end() )
        {
            stages_.emplace( age, PlateStageData( plateTopologies_, plateRotations_, age ) );

            for ( const auto id : stages_.at( age ).platesWithoutRotations() )
            {
                if ( platesWithoutRotations_.insert( id ).second )
                {
                    util::logroot << "WARNING: no usable rotation data for plate " << id
                                  << "; points on it will be reported as having no plate." << std::endl;
                }
            }
        }
        evictDistantStages( age );

        for ( const auto& plate : plateTopologies_.getPlatesForStage( std::ceil( age ) ) )
        {
            const auto key = std::make_pair( plate.id, age );
            if ( eulerVectors_.find( key ) == eulerVectors_.end() )
            {
                // A topology file may name plates that the rotation file does not describe -- the two are
                // often taken from different reconstructions. Such a plate has no Euler vector and therefore
                // no velocity, so record it and let the queries treat points on it as "no plate found", which
                // routes them through the caller's PlateNotFoundHandler like any other uncovered point.
                // Without this the failure would surface once per sample point, deep inside a parallel loop.
                try
                {
                    eulerVectors_.emplace(
                        key, computeEulerVector( plateRotations_, static_cast< int >( plate.id ), age ) );
                }
                catch ( const std::runtime_error& e )
                {
                    if ( platesWithoutRotations_.insert( plate.id ).second )
                    {
                        util::logroot << "WARNING: no usable rotation data for plate " << plate.id
                                      << "; points on it will be reported as having no plate. (" << e.what() << ")"
                                      << std::endl;
                    }
                }
            }
        }
    }

    /// Same, for both stages bracketing \p age, as used by the *InterpolatedInTime queries
    void prepareEulerVectorsInterpolatedInTime( const double age )
    {
        const auto [ageFloor, ageCeil, interpolationFactor] = getSurroundingAges( age );
        prepareEulerVectors( ageFloor );
        prepareEulerVectors( ageCeil );
    }

    /// Plates that the topologies name but the rotation data does not describe, as found by the most recent
    /// prepareEulerVectors(). Non-empty means the two input files disagree and some surface points will have
    /// no velocity; callers that care should report it rather than silently accepting the gap.
    const std::set< uint_t >& platesWithoutRotations() const { return platesWithoutRotations_; }

    /// Packed data for one prepared age stage, including the device views.
    ///
    /// Exposed so that callers can evaluate velocities inside a Kokkos kernel instead of going through the
    /// host-only query path; see terra/plates/plate_velocity_device.hpp. The stage must have been prepared
    /// first with prepareEulerVectors(), which is also what fills in its per-plate Euler vectors.
    const PlateStageData& stageFor( const double age ) const { return stages_.at( age ); }

    /// Whether a stage has been prepared for this age.
    bool hasStageFor( const double age ) const { return stages_.find( age ) != stages_.end(); }

    /// Access to the raw topology store, e.g. to pack the plate polygons into flat device buffers.
    PlateStorage&       plateTopologies() { return plateTopologies_; }
    const PlateStorage& plateTopologies() const { return plateTopologies_; }

  private:
    /// Locates the plate under a point, preferring the packed stage data
    ///
    /// Same contract as findPlateAndDistance(): first matching plate wins, distance in km, and
    /// std::numeric_limits<double>::max() when nothing matched. The packed path additionally
    /// rejects candidates by bounding cap and by longitude/latitude bin, which the polygon-list
    /// path cannot do. Falls back to the unpacked search when the stage was not prepared.
    std::tuple< bool, uint_t, double > findPlate( const vec3D& pointLonLat, const double age ) const
    {
        const auto stage = stages_.find( age );
        if ( stage == stages_.end() )
        {
            const auto unpacked = findPlateAndDistance( age, plateTopologies_, pointLonLat, idWhenNoPlateFound );
            return usableResult( unpacked );
        }

        const auto p = geometry::lonLatDegToUnit( pointLonLat( 0 ), pointLonLat( 1 ) );
        const auto r = findPlateInStage( stage->second.host(), p );

        if ( !r.found )
        {
            return std::make_tuple( false, idWhenNoPlateFound, std::numeric_limits< double >::max() );
        }

        return usableResult(
            std::make_tuple( true, r.plateId, r.distanceRad * plates::constants::earthRadiusInKm ) );
    }

    /// Drops packed stages far from \p age
    ///
    /// A stage costs a few hundred kB of host and device memory, and a run walking 400 Ma would
    /// otherwise accumulate every one of them. At most two are live at a time -- the pair
    /// bracketing the current age -- so a small window is plenty, and findPlate() falls back to
    /// the unpacked search if a stage is ever missing, which makes eviction safe by construction.
    void evictDistantStages( const double age )
    {
        constexpr std::size_t maxStages = 4;

        while ( stages_.size() > maxStages )
        {
            auto furthest = stages_.begin();
            for ( auto it = stages_.begin(); it != stages_.end(); ++it )
            {
                if ( std::abs( it->first - age ) > std::abs( furthest->first - age ) )
                {
                    furthest = it;
                }
            }
            stages_.erase( furthest );
        }
    }

    /// Euler vector for a plate at an age stage, from the prepared table where possible
    ///
    /// const, and therefore safe to call concurrently: a miss recomputes rather than memoising.
    /// Demotes a hit on a plate with no usable rotation data to a miss, so that every query path funnels such
    /// points into the caller's PlateNotFoundHandler rather than failing to produce a velocity for them.
    std::tuple< bool, uint_t, double > usableResult( std::tuple< bool, uint_t, double > result ) const
    {
        if ( std::get< 0 >( result ) && platesWithoutRotations_.count( std::get< 1 >( result ) ) > 0 )
        {
            return std::make_tuple( false, idWhenNoPlateFound, std::numeric_limits< double >::max() );
        }
        return result;
    }

    vec3D eulerVectorFor( const uint_t plateID, const double age ) const
    {
        const auto it = eulerVectors_.find( std::make_pair( plateID, age ) );
        if ( it != eulerVectors_.end() )
        {
            return it->second;
        }
        return computeEulerVector( plateRotations_, static_cast< int >( plateID ), age );
    }

    PlateStorage          plateTopologies_;
    PlateRotationProvider plateRotations_;

    /// Plates named by the topologies whose reconstruction circuit cannot be built from the rotation data.
    /// Populated by prepareEulerVectors(); see \ref platesWithoutRotations.
    std::set< uint_t > platesWithoutRotations_;

    /// Euler vectors keyed by (plate ID, age). Bounded by #stages x #plates, so a few thousand
    /// entries at most over a whole run.
    std::map< std::pair< uint_t, double >, vec3D > eulerVectors_;

    /// Packed, device-ready polygons per prepared age stage
    std::map< double, PlateStageData > stages_;
};

} // namespace plates
} // namespace terra
