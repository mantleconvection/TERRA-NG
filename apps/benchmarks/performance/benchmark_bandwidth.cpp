/**
 * Measures attained memory bandwidth on AMD MI250X comparing:
 *   - Raw HIP (hipMalloc + __global__ kernel)
 *   - Kokkos 1D View<double*>
 *   - Kokkos 2D View<double**>
 *   - Kokkos 3D View<double***>
 *   - Kokkos 4D View<double****>
 *
 * All use the same total data size. STREAM-triad: A[i] = B[i] + scalar * C[i]
 * Bandwidth = 3 * N * sizeof(double) / time  (2 reads + 1 write)
 */
#include <Kokkos_Core.hpp>
#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string>

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

__global__ void triad_raw( double* __restrict__ A,
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

struct Result
{
    double bw_gbs;
    double time_us;
};

// ---------- Raw HIP ----------
Result bench_raw( size_t N, int warmup, int iterations, double scalar )
{
    const int block_size = 256;
    const size_t bytes   = N * sizeof( double );
    const int grid       = ( N + block_size - 1 ) / block_size;

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    HIP_CHECK( hipMalloc( &d_A, bytes ) );
    HIP_CHECK( hipMalloc( &d_B, bytes ) );
    HIP_CHECK( hipMalloc( &d_C, bytes ) );
    HIP_CHECK( hipMemset( d_A, 0, bytes ) );
    HIP_CHECK( hipMemset( d_B, 1, bytes ) );
    HIP_CHECK( hipMemset( d_C, 2, bytes ) );
    HIP_CHECK( hipDeviceSynchronize() );

    for ( int w = 0; w < warmup; ++w )
        hipLaunchKernelGGL( triad_raw, dim3( grid ), dim3( block_size ), 0, 0, d_A, d_B, d_C, scalar, N );
    HIP_CHECK( hipDeviceSynchronize() );

    hipEvent_t start, stop;
    HIP_CHECK( hipEventCreate( &start ) );
    HIP_CHECK( hipEventCreate( &stop ) );
    HIP_CHECK( hipEventRecord( start ) );
    for ( int it = 0; it < iterations; ++it )
        hipLaunchKernelGGL( triad_raw, dim3( grid ), dim3( block_size ), 0, 0, d_A, d_B, d_C, scalar, N );
    HIP_CHECK( hipEventRecord( stop ) );
    HIP_CHECK( hipEventSynchronize( stop ) );

    float ms = 0;
    HIP_CHECK( hipEventElapsedTime( &ms, start, stop ) );
    double elapsed    = ( ms / 1000.0 ) / iterations;
    double bytes_moved = 3.0 * N * sizeof( double );

    HIP_CHECK( hipEventDestroy( start ) );
    HIP_CHECK( hipEventDestroy( stop ) );
    HIP_CHECK( hipFree( d_A ) );
    HIP_CHECK( hipFree( d_B ) );
    HIP_CHECK( hipFree( d_C ) );

    return { bytes_moved / elapsed / 1e9, elapsed * 1e6 };
}

// ---------- Kokkos 1D ----------
Result bench_kokkos_1d( size_t N, int warmup, int iterations, double scalar )
{
    Kokkos::View< double* > A( "A", N );
    Kokkos::View< double* > B( "B", N );
    Kokkos::View< double* > C( "C", N );

    Kokkos::parallel_for( "init", N, KOKKOS_LAMBDA( const size_t i ) {
        A( i ) = 0.0; B( i ) = 1.0; C( i ) = 2.0;
    } );
    Kokkos::fence();

    for ( int w = 0; w < warmup; ++w )
        Kokkos::parallel_for( "warmup", N, KOKKOS_LAMBDA( const size_t i ) { A( i ) = B( i ) + scalar * C( i ); } );
    Kokkos::fence();

    Kokkos::Timer timer;
    for ( int it = 0; it < iterations; ++it )
        Kokkos::parallel_for( "triad_1d", N, KOKKOS_LAMBDA( const size_t i ) { A( i ) = B( i ) + scalar * C( i ); } );
    Kokkos::fence();
    double elapsed     = timer.seconds() / iterations;
    double bytes_moved = 3.0 * N * sizeof( double );
    return { bytes_moved / elapsed / 1e9, elapsed * 1e6 };
}

// ---------- Kokkos 2D ----------
template < typename ScheduleT = Kokkos::Schedule< Kokkos::Dynamic > >
Result bench_kokkos_2d( size_t N, int warmup, int iterations, double scalar )
{
    const size_t d0 = N / 4;
    const size_t d1 = 4;

    Kokkos::View< double** > A( "A", d0, d1 );
    Kokkos::View< double** > B( "B", d0, d1 );
    Kokkos::View< double** > C( "C", d0, d1 );

    Kokkos::parallel_for( "init", Kokkos::MDRangePolicy< Kokkos::Rank< 2 > >( { 0, 0 }, { d0, d1 } ),
        KOKKOS_LAMBDA( const size_t i, const size_t j ) {
            A( i, j ) = 0.0; B( i, j ) = 1.0; C( i, j ) = 2.0;
        } );
    Kokkos::fence();

    auto policy = Kokkos::MDRangePolicy< Kokkos::Rank< 2 >, ScheduleT >( { 0, 0 }, { d0, d1 } );
    for ( int w = 0; w < warmup; ++w )
        Kokkos::parallel_for( "warmup", policy,
            KOKKOS_LAMBDA( const size_t i, const size_t j ) { A( i, j ) = B( i, j ) + scalar * C( i, j ); } );
    Kokkos::fence();

    Kokkos::Timer timer;
    for ( int it = 0; it < iterations; ++it )
        Kokkos::parallel_for( "triad_2d", policy,
            KOKKOS_LAMBDA( const size_t i, const size_t j ) { A( i, j ) = B( i, j ) + scalar * C( i, j ); } );
    Kokkos::fence();
    double elapsed     = timer.seconds() / iterations;
    double bytes_moved = 3.0 * N * sizeof( double );
    return { bytes_moved / elapsed / 1e9, elapsed * 1e6 };
}

// ---------- Kokkos 3D ----------
template < typename ScheduleT = Kokkos::Schedule< Kokkos::Dynamic > >
Result bench_kokkos_3d( size_t N, int warmup, int iterations, double scalar )
{
    const size_t d0 = N / 16;
    const size_t d1 = 4;
    const size_t d2 = 4;

    Kokkos::View< double*** > A( "A", d0, d1, d2 );
    Kokkos::View< double*** > B( "B", d0, d1, d2 );
    Kokkos::View< double*** > C( "C", d0, d1, d2 );

    Kokkos::parallel_for( "init", Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { d0, d1, d2 } ),
        KOKKOS_LAMBDA( const size_t i, const size_t j, const size_t k ) {
            A( i, j, k ) = 0.0; B( i, j, k ) = 1.0; C( i, j, k ) = 2.0;
        } );
    Kokkos::fence();

    auto policy = Kokkos::MDRangePolicy< Kokkos::Rank< 3 >, ScheduleT >( { 0, 0, 0 }, { d0, d1, d2 } );
    for ( int w = 0; w < warmup; ++w )
        Kokkos::parallel_for( "warmup", policy,
            KOKKOS_LAMBDA( const size_t i, const size_t j, const size_t k ) {
                A( i, j, k ) = B( i, j, k ) + scalar * C( i, j, k );
            } );
    Kokkos::fence();

    Kokkos::Timer timer;
    for ( int it = 0; it < iterations; ++it )
        Kokkos::parallel_for( "triad_3d", policy,
            KOKKOS_LAMBDA( const size_t i, const size_t j, const size_t k ) {
                A( i, j, k ) = B( i, j, k ) + scalar * C( i, j, k );
            } );
    Kokkos::fence();
    double elapsed     = timer.seconds() / iterations;
    double bytes_moved = 3.0 * N * sizeof( double );
    return { bytes_moved / elapsed / 1e9, elapsed * 1e6 };
}

// ---------- Kokkos 4D ----------
template < typename ScheduleT = Kokkos::Schedule< Kokkos::Dynamic > >
Result bench_kokkos_4d( size_t N, int warmup, int iterations, double scalar )
{
    const size_t d0 = N / 64;
    const size_t d1 = 4;
    const size_t d2 = 4;
    const size_t d3 = 4;

    Kokkos::View< double**** > A( "A", d0, d1, d2, d3 );
    Kokkos::View< double**** > B( "B", d0, d1, d2, d3 );
    Kokkos::View< double**** > C( "C", d0, d1, d2, d3 );

    Kokkos::parallel_for( "init", Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { d0, d1, d2, d3 } ),
        KOKKOS_LAMBDA( const size_t i, const size_t j, const size_t k, const size_t l ) {
            A( i, j, k, l ) = 0.0; B( i, j, k, l ) = 1.0; C( i, j, k, l ) = 2.0;
        } );
    Kokkos::fence();

    auto policy = Kokkos::MDRangePolicy< Kokkos::Rank< 4 >, ScheduleT >( { 0, 0, 0, 0 }, { d0, d1, d2, d3 } );
    for ( int w = 0; w < warmup; ++w )
        Kokkos::parallel_for( "warmup", policy,
            KOKKOS_LAMBDA( const size_t i, const size_t j, const size_t k, const size_t l ) {
                A( i, j, k, l ) = B( i, j, k, l ) + scalar * C( i, j, k, l );
            } );
    Kokkos::fence();

    Kokkos::Timer timer;
    for ( int it = 0; it < iterations; ++it )
        Kokkos::parallel_for( "triad_4d", policy,
            KOKKOS_LAMBDA( const size_t i, const size_t j, const size_t k, const size_t l ) {
                A( i, j, k, l ) = B( i, j, k, l ) + scalar * C( i, j, k, l );
            } );
    Kokkos::fence();
    double elapsed     = timer.seconds() / iterations;
    double bytes_moved = 3.0 * N * sizeof( double );
    return { bytes_moved / elapsed / 1e9, elapsed * 1e6 };
}

int main( int argc, char* argv[] )
{
    Kokkos::ScopeGuard scope_guard( argc, argv );

    const int warmup     = 10;
    const int iterations = 50;
    const double scalar  = 3.14159;

    // Sizes from 1 KB to 16 GB (must be divisible by 64 for 4D split)
    std::vector< size_t > sizes_bytes;
    for ( size_t s = 1024; s <= 16ULL * 1024 * 1024 * 1024; s *= 2 )
    {
        sizes_bytes.push_back( s );
    }

    using Static  = Kokkos::Schedule< Kokkos::Static >;
    using Dynamic = Kokkos::Schedule< Kokkos::Dynamic >;

    // Header
    std::cout << std::setw( 12 ) << "Size (MB)"
              << std::setw( 12 ) << "Raw HIP"
              << std::setw( 12 ) << "K 1D"
              << std::setw( 12 ) << "K 2D"
              << std::setw( 12 ) << "K 2D/S"
              << std::setw( 12 ) << "K 3D"
              << std::setw( 12 ) << "K 3D/S"
              << std::setw( 12 ) << "K 4D"
              << std::setw( 12 ) << "K 4D/S"
              << "   (all GB/s)\n";
    std::cout << std::string( 108, '-' ) << "\n";

    // CSV header on stderr
    std::cerr << "size_bytes,size_mb,raw_hip,kokkos_1d,kokkos_2d,kokkos_2d_static,kokkos_3d,kokkos_3d_static,kokkos_4d,kokkos_4d_static\n";

    for ( size_t total_bytes : sizes_bytes )
    {
        const size_t N  = total_bytes / sizeof( double );
        double size_mb  = total_bytes / ( 1024.0 * 1024.0 );

        // N must be divisible by 64 for the 4D case
        if ( N < 64 )
            continue;

        auto r_raw  = bench_raw( N, warmup, iterations, scalar );
        auto r_1d   = bench_kokkos_1d( N, warmup, iterations, scalar );
        auto r_2d   = bench_kokkos_2d< Dynamic >( N, warmup, iterations, scalar );
        auto r_2ds  = bench_kokkos_2d< Static >( N, warmup, iterations, scalar );
        auto r_3d   = bench_kokkos_3d< Dynamic >( N, warmup, iterations, scalar );
        auto r_3ds  = bench_kokkos_3d< Static >( N, warmup, iterations, scalar );
        auto r_4d   = bench_kokkos_4d< Dynamic >( N, warmup, iterations, scalar );
        auto r_4ds  = bench_kokkos_4d< Static >( N, warmup, iterations, scalar );

        std::cout << std::setw( 12 ) << std::fixed << std::setprecision( 2 ) << size_mb
                  << std::setw( 12 ) << std::fixed << std::setprecision( 1 ) << r_raw.bw_gbs
                  << std::setw( 12 ) << std::fixed << std::setprecision( 1 ) << r_1d.bw_gbs
                  << std::setw( 12 ) << std::fixed << std::setprecision( 1 ) << r_2d.bw_gbs
                  << std::setw( 12 ) << std::fixed << std::setprecision( 1 ) << r_2ds.bw_gbs
                  << std::setw( 12 ) << std::fixed << std::setprecision( 1 ) << r_3d.bw_gbs
                  << std::setw( 12 ) << std::fixed << std::setprecision( 1 ) << r_3ds.bw_gbs
                  << std::setw( 12 ) << std::fixed << std::setprecision( 1 ) << r_4d.bw_gbs
                  << std::setw( 12 ) << std::fixed << std::setprecision( 1 ) << r_4ds.bw_gbs
                  << "\n";

        std::cerr << total_bytes << "," << size_mb << ","
                  << r_raw.bw_gbs << "," << r_1d.bw_gbs << ","
                  << r_2d.bw_gbs << "," << r_2ds.bw_gbs << ","
                  << r_3d.bw_gbs << "," << r_3ds.bw_gbs << ","
                  << r_4d.bw_gbs << "," << r_4ds.bw_gbs << "\n";
    }

    return 0;
}
