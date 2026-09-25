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

#include <csignal>
#include <fstream>

#include "terra/plates/json.hpp"
#include "terra/plates/types.hpp"
#include "util/logging.hpp"

namespace terra {

using util::logall;
using util::logroot;

namespace io {

// using walberla::real_c;
// using walberla::real_t;
// using walberla::uint_t;

/// Auxilliary function for opening an input file

/// \param fileName  name (including potentially a path) of the input file
/// \return ifstream for the file
inline std::ifstream openFileForReading( std::string& fileName )
{
    std::ifstream infile( fileName, std::ios::in );
    if ( !infile.is_open() )
    {
        logall << "Failed to open file '" << fileName << "' for reading!" << std::endl;
        std::abort();
    }
    return infile;
}

/// Imports a file in JSON format
inline nlohmann::json readJsonFile( std::string filename )
{
    if ( mpi::rank() == 0 )
    {
        logroot << "Starting to read datafile '" << filename << "' \n";
    }
    std::ifstream  infile{ openFileForReading( filename ) };
    nlohmann::json inobj;
    infile >> inobj;
    if ( mpi::rank() == 0 )
    {
        logroot << "Finished reading datafile '" << filename << "' \n";
    }
    return inobj;
}

/// Imports data from a text file with rotation information
inline std::vector< plates::RotationInfo > readRotationsFile( std::string filename )
{
    // if ( mpi::rank() == 0 )
    // {
    logroot << "Starting to read datafile '" << filename << "' \n";
    // }
    std::ifstream file{ openFileForReading( filename ) };

    // Determine number of lines in file for reserving space
    std::string line;
    uint_t      numRotations{ 0 };
    while ( std::getline( file, line ) )
    {
        numRotations++;
    }
    file.clear();    // clear EOF
    file.seekg( 0 ); // rewind stream

    // if ( mpi::rank == 0 )
    // {
    logroot << "Found " << numRotations << " rotations in data-file \n";
    // }
    std::vector< plates::RotationInfo > rotations( numRotations, plates::RotationInfo() );

    std::size_t charsUsed;
    uint_t      k{ 0 };

// this was primarily introduced for the regression testing during the
// refactoring of the original coding attempt; the accuracy of the data
// in the input file is smaller than even binary32 ;-)
// #ifdef WALBERLA_DOUBLE_ACCURACY
#define PLATES_IO_STR_TO_FP std::stod
    // #else
    // #define PLATES_IO_STR_TO_FP std::stof
    // #endif

    while ( std::getline( file, line ) )
    {
        // position inside line-string
        std::size_t pos{ 0 };

        rotations[k].plateID = std::stoul( line, &charsUsed );

        pos += charsUsed;
        rotations[k].time = static_cast< double >( PLATES_IO_STR_TO_FP( line.substr( pos ), &charsUsed ) );

        pos += charsUsed;
        rotations[k].latitude = static_cast< double >( PLATES_IO_STR_TO_FP( line.substr( pos ), &charsUsed ) );

        pos += charsUsed;
        rotations[k].longitude = static_cast< double >( PLATES_IO_STR_TO_FP( line.substr( pos ), &charsUsed ) );

        pos += charsUsed;
        rotations[k].angle = static_cast< double >( PLATES_IO_STR_TO_FP( line.substr( pos ), &charsUsed ) );

        pos += charsUsed;
        rotations[k].conjugateID = std::stoul( line.substr( pos ), &charsUsed );

        // don't forget to increment index into vector
        ++k;
    }

    // Safety check
    if ( k != numRotations )
    {
        // if ( mpi::rank == 0 )
        // {
        logroot << " Imported " << k << " rotations,\n"
                << " but should have been " << numRotations << std::endl;
        // }
        logall << "Data import error" << std::endl;
        std::abort();
    }

    // if ( mpi::rank == 0 )
    // {
    logroot << "Finished reading datafile '" << filename << "' \n";
    // }

    return rotations;
}

} // namespace io
} // namespace terra
