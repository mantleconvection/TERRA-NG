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

#include <algorithm>

#include "terra/plates/json.hpp" // question : 
                                 // should we include this file or is there an internal JSON handler included already ?
                                 // For now I will include it in the same folder till we know :)

// #include "terraneo/dataimport/file_io.hpp"
#include "terra/plates/conversions.hpp"
// #include "terraneo/helpers/typeAliases.hpp"
#include "terra/plates/functions_for_rotations.hpp"
#include "terra/plates/types.hpp"
#include "terra/plates/utilities.hpp"
#include "util/logging.hpp"

using terra::util::logroot;
using terra::util::logall;

using uint_t = unsigned int;

namespace terra {
namespace plates {

/// Class for managing plate topology information
class PlateStorage
{
  public:
    /// Information describing a single plate at a certain age
    struct PlateInfo
    {
        /// ID of the plate
        uint_t id{ 0 };

        /// Nodes describing the boundary of the plate
        Polygon boundary;

        /// Textual name of plate
        std::string name;
    };

    using plateVec_t       = std::vector< PlateInfo >;
    using ageToPlatesMap_t = std::map< std::string, plateVec_t >;

    template < typename ImportStrategy >
    PlateStorage( std::string nameOfPlateTopologiesFile, double sphereRadius, ImportStrategy readJsonfile )
    : srcFile_( nameOfPlateTopologiesFile )
    {
        // import topology data
        nlohmann::json rootNode = readJsonfile( nameOfPlateTopologiesFile )["ages"];

        // convert required data into our internal format
        extractPlateInfo( rootNode );

        // for testing
        printStatistics();
    }

    /// Report statistics on what an object of this class stores
    void printStatistics()
    {
        uint_t nPlates{ 0 };
        for ( auto entry : ageToPlatesMap_ )
        {
            nPlates += entry.second.size();
        }
        // if ( mpi::rank == 0 )
        // {
            logroot << "PlateStorage object:\n"
                      << " - stores plates for " << ageToPlatesMap_.size() << " age stages\n"
                      << " - stores a total of " << nPlates << " plates\n"
                      << " - age stages range from " << *listOfPlateStages_.begin() << " to "
                      << listOfPlateStages_.back() << " Ma\n"
                      << " - data was obtained from file = '" << srcFile_ << "'" << std::endl;
        // }
    }

    double getMinAge() const { return *listOfPlateStages_.begin(); }

    double getMaxAge() const { return listOfPlateStages_.back(); }

    /// Plates of the age stage \p age
    ///
    /// Looked up by numeric age rather than by the formatted stage name. The name-based lookup
    /// this replaced built a std::stringstream and formatted "topology_%.4fMa_polygon" on every
    /// call -- and it is called once per plate per sample point, so it was a measurable part of
    /// the plate search rather than a setup cost.
    ///
    /// The comparison is exact, which is what the string lookup effectively was too. Stage ages
    /// come from the datafile as integers ("0.0000", "1.0000", ...) and callers pass
    /// std::ceil( age ), so both sides are exactly representable and the match is unambiguous.
    plateVec_t& getPlatesForStage( double age )
    {
        return const_cast< plateVec_t& >( const_cast< const PlateStorage* >( this )->getPlatesForStage( age ) );
    }

    const plateVec_t& getPlatesForStage( double age ) const
    {
        const auto iter = ageToPlates_.find( age );
        if ( iter == ageToPlates_.end() )
        {
            std::cerr << "No plates found for " << ageToKeyStr( age ) << std::endl;
            std::abort();
        }

        return *iter->second;
    }

    const std::vector< double >& getListOfPlateStages() const { return listOfPlateStages_; }

  private:
    /// method for data preparation
    ///
    /// this method will convert the imported data into a format more suitable for
    /// using it in the computations; it will convert the plate polygons into
    /// cartesian coordinates, compute the barycenter of each plate and group
    /// plates into vectors depending on their age stage
    void extractPlateInfo( const nlohmann::json& rootNode )
    {
        // we need to fill the list of age stages and do a reservation here
        listOfPlateStages_.clear();
        listOfPlateStages_.reserve( rootNode.size() );

        for ( uint_t idx = 0; idx < rootNode.size(); ++idx )
        {
            // prepare key for map entry and vector to hold plates for this age stage
            std::string ageName = rootNode[idx]["name"];
            uint_t      nPlates = rootNode[idx]["features"].size();
            ageToPlatesMap_.emplace( ageName, plateVec_t( nPlates ) );

            plateVec_t& plates = ageToPlatesMap_[ageName];

            // convert ageName to numeric age value and insert it
            listOfPlateStages_.push_back( keyStrToAge( ageName ) );

            // extract plates for this age stage and put into vector of plates
            for ( uint_t k = 0; k < nPlates; ++k )
            {
                // get ID and name
                plates[k].id   = rootNode[idx]["features"][k]["properties"]["PLATEID1"];
                plates[k].name = static_cast< std::string >( rootNode[idx]["features"][k]["properties"]["NAME"] );

                // extract coordinates of polygon (order: longitude, latitude)
                const nlohmann::json coordinates = rootNode[idx]["features"][k]["geometry"]["coordinates"][0];
                for ( auto& element : coordinates )
                {
                    plates[k].boundary.push_back( { element[0], element[1], static_cast< double >( 0 ) } );
                }
            }
        }

        // sort the list of plate stages (should be sorted in data-file, but hey,
        // better safe than sorry
        std::sort( listOfPlateStages_.begin(), listOfPlateStages_.end() );

        // Index the stages by numeric age. References into a std::map's mapped values are stable,
        // and nothing is inserted into ageToPlatesMap_ after this point.
        for ( auto& [ageName, plates] : ageToPlatesMap_ )
        {
            ageToPlates_[keyStrToAge( ageName )] = &plates;
        }
    };

    /// name of datafile from which object obtained information
    std::string srcFile_;

    /// map to allow retrieving all plates of a certain age stage by giving that age
    ageToPlatesMap_t ageToPlatesMap_;

    /// a vector containing a sorted list of the ages of the plate stages found in data-file
    std::vector< double > listOfPlateStages_;

    /// numeric age -> plates of that stage, pointing into ageToPlatesMap_
    std::map< double, plateVec_t* > ageToPlates_;
};

} // namespace plates
} // namespace terra
