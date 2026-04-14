/**
 * Measures attained memory bandwidth using raw HIP allocations (no Kokkos Views).
 * STREAM-triad: A[i] = B[i] + scalar * C[i]
 * Bandwidth = 3 * N * sizeof(double) / time  (2 reads + 1 write)
 */
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>

#define HIP_CHECK( call )                                                         \
    do                                                                            \
    {                                                                             \
        hipError_t err = call;                                                    \
        if ( err != hipSuccess )                                                  \
        {                                                                         \
            std::cerr << "HIP error: " << hipGetErrorString( err ) << " at "      \
                      << __FILE__ << ":" << __LINE__ << "\n";                     \
            std::exit( 1 );                                                       \
        }                                                                         \
    } while ( 0 )

__global__ void triad_kernel( double* __restrict__ A,
                              const double* __restrict__ B,
                              const double* __restrict__ C,
                              const double scalar,
                              const size_t N )
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < N )
    {
        A[i] = B[i] + scalar * C[i];
    }
}

int main()
{
    const int warmup     = 10;
    const int iterations = 50;
    const double scalar  = 3.14159;
    const int block_size = 256;

    // Data sizes from 1 KB to 1 GB
    std::vector< size_t > sizes_bytes;
    for ( size_t s = 1024; s <= 16ULL * 1024 * 1024 * 1024; s *= 2 )
    {
        sizes_bytes.push_back( s );
    }

    std::cout << std::setw( 14 ) << "Size"
              << std::setw( 14 ) << "Size (KB)"
              << std::setw( 14 ) << "Size (MB)"
              << std::setw( 16 ) << "BW (GB/s)"
              << std::setw( 14 ) << "Time (us)"
              << "\n";
    std::cout << std::string( 72, '-' ) << "\n";

    std::cerr << "size_bytes,size_kb,size_mb,bw_gbs,time_us\n";

    hipEvent_t start, stop;
    HIP_CHECK( hipEventCreate( &start ) );
    HIP_CHECK( hipEventCreate( &stop ) );

    for ( size_t total_bytes : sizes_bytes )
    {
        const size_t N     = total_bytes / sizeof( double );
        const size_t bytes = N * sizeof( double );
        const int grid     = ( N + block_size - 1 ) / block_size;

        double* d_A = nullptr;
        double* d_B = nullptr;
        double* d_C = nullptr;
        HIP_CHECK( hipMalloc( &d_A, bytes ) );
        HIP_CHECK( hipMalloc( &d_B, bytes ) );
        HIP_CHECK( hipMalloc( &d_C, bytes ) );

        // Initialize on device
        HIP_CHECK( hipMemset( d_A, 0, bytes ) );
        HIP_CHECK( hipMemset( d_B, 1, bytes ) );
        HIP_CHECK( hipMemset( d_C, 2, bytes ) );
        HIP_CHECK( hipDeviceSynchronize() );

        // Warmup
        for ( int w = 0; w < warmup; ++w )
        {
            hipLaunchKernelGGL( triad_kernel, dim3( grid ), dim3( block_size ), 0, 0,
                                d_A, d_B, d_C, scalar, N );
        }
        HIP_CHECK( hipDeviceSynchronize() );

        // Timed runs
        HIP_CHECK( hipEventRecord( start ) );
        for ( int it = 0; it < iterations; ++it )
        {
            hipLaunchKernelGGL( triad_kernel, dim3( grid ), dim3( block_size ), 0, 0,
                                d_A, d_B, d_C, scalar, N );
        }
        HIP_CHECK( hipEventRecord( stop ) );
        HIP_CHECK( hipEventSynchronize( stop ) );

        float ms = 0;
        HIP_CHECK( hipEventElapsedTime( &ms, start, stop ) );
        double elapsed = ( ms / 1000.0 ) / iterations;

        double bytes_moved = 3.0 * N * sizeof( double );
        double bw_gbs      = bytes_moved / elapsed / 1e9;
        double time_us     = elapsed * 1e6;
        double size_kb     = total_bytes / 1024.0;
        double size_mb     = total_bytes / ( 1024.0 * 1024.0 );

        std::cout << std::setw( 14 ) << total_bytes
                  << std::setw( 14 ) << std::fixed << std::setprecision( 1 ) << size_kb
                  << std::setw( 14 ) << std::fixed << std::setprecision( 2 ) << size_mb
                  << std::setw( 16 ) << std::fixed << std::setprecision( 1 ) << bw_gbs
                  << std::setw( 14 ) << std::fixed << std::setprecision( 1 ) << time_us
                  << "\n";

        std::cerr << total_bytes << "," << size_kb << "," << size_mb << ","
                  << bw_gbs << "," << time_us << "\n";

        HIP_CHECK( hipFree( d_A ) );
        HIP_CHECK( hipFree( d_B ) );
        HIP_CHECK( hipFree( d_C ) );
    }

    HIP_CHECK( hipEventDestroy( start ) );
    HIP_CHECK( hipEventDestroy( stop ) );

    return 0;
}
