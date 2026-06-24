#pragma once

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <string>

#include "util/logging.hpp"

namespace terra::mantlecirculation {

/// Query device HBM in use (total - free) on the calling rank's GCD and log the
/// max/min across ranks. Used to profile where GPU memory goes; the delta
/// between two probes attributes memory to the work done in between.
inline void log_hbm( const std::string& label )
{
    size_t freeB = 0, totalB = 0;
    if ( hipMemGetInfo( &freeB, &totalB ) != hipSuccess )
        return;

    double used = double( totalB - freeB ) / ( 1024.0 * 1024.0 * 1024.0 );
    double used_max = used, used_min = used;
    MPI_Allreduce( MPI_IN_PLACE, &used_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &used_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD );

    util::logroot << "[HBM] " << label << ": " << used_max << " GB/GCD (max over ranks), " << used_min
                  << " GB/GCD (min)" << std::endl;
}

} // namespace terra::mantlecirculation
