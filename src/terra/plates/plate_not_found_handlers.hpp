/*
 * Copyright (c) 2022 Marcus Mohr, Fatemeh Rezaei. 
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
#include <iomanip>
#include <iostream>
#include <set>
#include <vector>
#include <map>
#include <numeric>
#include <string>

#include "mpi/mpi.hpp"
#include "terra/plates/utilities.hpp"

namespace terra {
namespace plates {

/// \defgroup PlateNotFoundHandlers PlateNotFoundHandlers
///
/// Functors for handling the case the we fail to find a plate associated with a point.
/// These will be called by PlateVelocityProvider::getPointVelocity() in this case
/// and get passed the point as vec3D object and the age as realt.
/// @{

/// Default strategy
///
/// The default strategy when we find no plate for a point is to set the velocity to zero.
class DefaultPlateNotFoundHandler
{
  public:
    vec3D operator()( [[maybe_unused]] const vec3D& point, [[maybe_unused]] const double age )
    {
        // WALBERLA_UNUSED( point ); // replaced with [[maybe_unused]] c++17 standard
        // WALBERLA_UNUSED( age );   // replaced with [[maybe_unused]] c++17 standard
        return { static_cast< double >( 0 ), static_cast< double >( 0 ), static_cast< double >( 0 ) };
    }
};

/// Statistics strategy
///
/// The statistics strategy will record the points and ages for which we fail to find plates
/// and can be queried for a report.
class StatisticsPlateNotFoundHandler
{
  public:
    using map_t = std::map< std::string, std::set< vec3D > >;

    vec3D operator()( const vec3D& point, const double age )
    {
        pointsNotFound_[plates::ageToKeyStr( age )].insert( point );
        return { static_cast< double >( 0 ), static_cast< double >( 0 ), static_cast< double >( 0 ) };
    }

    void generateReport()
    {
        // remove duplicates from list of points:
        // we can only compare vec3D for equality, but for sorting we'd need "<"
        // std::sort( pointsNotFound_.begin(), pointsNotFound_.end() );
        // auto last = std::unique( pointsNotFound_.begin(), pointsNotFound_.end() );
        // pointsNotFound_.erase( last, pointsNotFound_.end() );

        map_t pointsNotFoundGlobal = collectDataOnRoot();

        if ( mpi::rank() == 0 )
        {
            for ( const auto& elem : pointsNotFoundGlobal )
            {
                std::string              keyStr       = elem.first;
                const std::set< vec3D >& pointsForAge = elem.second;
                std::cout << "List of points w/o plateID for age = " << keyStrToAge( keyStr ) << ":" << std::endl;
                uint_t k = 1;
                for ( const auto& point : pointsForAge )
                {
                    std::cout << "" << std::setw( 3 ) << k++ << ": (" << std::scientific << std::showpos << point( 0 )
                              << ", " << point( 1 ) << ", " << point( 2 ) << ")" << std::endl;
                }
            }
        }
    }

  private:
    /// Gather the maps of all processes on root and merge them (other ranks get an empty map).
    ///
    /// Each point is sent as a tuple ( age, x, y, z ) of doubles, so a single MPI_Gatherv suffices.
    map_t collectDataOnRoot()
    {
        constexpr int tupleSize = 4;

        // flatten local map; parse the full age from the key, as keyStrToAge() truncates to 6 characters
        std::vector< double > sendBuf;
        for ( const auto& [key, points] : pointsNotFound_ )
        {
            const double age = std::stod( key.substr( 9 ) );
            for ( const auto& p : points )
            {
                sendBuf.insert( sendBuf.end(), { age, p( 0 ), p( 1 ), p( 2 ) } );
            }
        }

        const int myRank    = mpi::rank();
        const int nProcs    = mpi::num_processes();
        const int sendCount = static_cast< int >( sendBuf.size() );

        std::vector< int > counts( myRank == 0 ? nProcs : 0 );
        std::vector< int > displs( myRank == 0 ? nProcs : 0 );
        MPI_Gather( &sendCount, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD );

        std::vector< double > recvBuf;
        if ( myRank == 0 )
        {
            std::exclusive_scan( counts.begin(), counts.end(), displs.begin(), 0 );
            recvBuf.resize( displs.back() + counts.back() );
        }

        MPI_Gatherv( sendBuf.data(),
                     sendCount,
                     mpi::mpi_datatype< double >(),
                     recvBuf.data(),
                     counts.data(),
                     displs.data(),
                     mpi::mpi_datatype< double >(),
                     0,
                     MPI_COMM_WORLD );

        // rebuild the map; std::set takes care of merging duplicates across ranks
        map_t globalMap;
        for ( std::size_t i = 0; i < recvBuf.size(); i += tupleSize )
        {
            globalMap[ageToKeyStr( recvBuf[i] )].insert( vec3D{ recvBuf[i + 1], recvBuf[i + 2], recvBuf[i + 3] } );
        }

        return globalMap;
    }

    map_t pointsNotFound_;
};

/// @}
} // namespace plates
} // namespace terra
