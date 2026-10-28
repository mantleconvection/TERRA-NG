/// A simple MPI ring bandwidth benchmark with optional CUDA support.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <optional>
#include <thread>
#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/table.hpp"
#include <mpi.h>

#include "communication/shell/communication.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "terra/grid/bit_masks.hpp"
#include "terra/grid/shell/bit_masks.hpp"
#include "terra/linalg/vector.hpp"
#include <kernels/common/grid_operations.hpp>

#include "fe/wedge/operators/shell/epsilon_divdiv.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_w1.hpp"
#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/table.hpp"



using namespace terra;
using linalg::apply;
using linalg::DstOf;
using linalg::OperatorLike;
using linalg::SrcOf;
using linalg::VectorW1IsoW2W1;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;
using terra::grid::shell::BoundaryConditions;
using util::logroot;

struct Parameters
{
    double interval  = 1.0;
    int msg_size = 10;
};

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    util::print_general_info( argc, argv );

    const auto description =
        "Communication benchmark. ";
    CLI::App app{ description };

    // parse params
    Parameters parameters{};
    util::add_option_with_default( app, "--interval", parameters.interval, "Message sending interval." );
    util::add_option_with_default( app, "--msg-size", parameters.msg_size, "Number of data points to be sent." );

    CLI11_PARSE( app, argc, argv );
    logroot << "Ring comm benchmark." << std::endl;
    logroot << "Interval:       " << parameters.interval << " seconds." << std::endl;
    logroot << "Message size:       " << parameters.msg_size << " Byte." << std::endl;
   
    const auto num_processes = mpi::num_processes();
    const int rank = mpi::rank();
    const int next = ( rank + 1 ) % num_processes;
    const int prev = ( rank - 1 + num_processes ) % num_processes;

 
    // send/receive buffers
    grid::Grid1DDataScalar< unsigned char > send_data(
            "send",
            parameters.msg_size
           );
    
     grid::Grid1DDataScalar< unsigned char > receive_data(
            "receive",
            parameters.msg_size);
    

    const int buffer_size = receive_data.span();
    logroot << "Buffer_size = " << buffer_size << std::endl;

    // send data in ring
    while ( true )
    {
        if ( parameters.interval > 0 )
        {
            std::this_thread::sleep_for( std::chrono::duration< double >( parameters.interval ) );
        }

        mpi::barrier();

        auto t0 = std::chrono::steady_clock::now();

        
        MPI_Sendrecv(
                send_data.data(),
                buffer_size,
                MPI_UNSIGNED_CHAR,
                next,
                0,
                receive_data.data(),
                buffer_size,
                MPI_UNSIGNED_CHAR,
                prev,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE 
        );
        

        auto t1 = std::chrono::steady_clock::now();

        const double dt = std::chrono::duration< double >( t1 - t0 ).count();
        const double bw = static_cast< double >( 2 * buffer_size ) / dt / 1e9; // GB/s

        // One sample only: local stats = the single sample
        double local_min_bw = bw;
        double local_max_bw = bw;
        double local_avg_bw = bw;

        double global_min_bw, global_max_bw, global_sum_bw;
        MPI_Reduce( &local_min_bw, &global_min_bw, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD );
        MPI_Reduce( &local_max_bw, &global_max_bw, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        MPI_Reduce( &local_avg_bw, &global_sum_bw, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );

        const double global_avg_bw = global_sum_bw / num_processes;

        // One sample only: local stats = the single sample
        double local_min_dt = dt;
        double local_max_dt = dt;
        double local_avg_dt = dt;

        double global_min_dt, global_max_dt, global_sum_dt;
        MPI_Reduce( &local_min_dt, &global_min_dt, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD );
        MPI_Reduce( &local_max_dt, &global_max_dt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD );
        MPI_Reduce( &local_avg_dt, &global_sum_dt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );

        const double global_avg_dt = global_sum_dt / num_processes;

        
            logroot << std::fixed << std::setprecision( 3 ) << "Bandwidth (send + recv): min = " << std::setw( 10 )
                      << global_min_bw << " GB/s | max = " << std::setw( 10 ) << global_max_bw
                      << " GB/s | avg = " << std::setw( 10 ) << global_avg_bw
                      << " GB/s || Duration (send + recv): min = " << std::setw( 10 ) << global_min_dt * 1e3
                      << " ms | max = " << std::setw( 10 ) << global_max_dt * 1e3 << " ms | avg = " << std::setw( 10 )
                      << global_avg_dt * 1e3 << " ms" << std::endl;
        
    }
}