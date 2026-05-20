// File: kokkos_lincomb_benchmark.cpp
// Single-file benchmark harness for comparing lincomb implementations in Kokkos
// Variants: MDRange (straight), RangePolicy (flat), MDRange tiled
// Usage: ./lincomb_bench [N0 N1 N2 N3] [niter] [variant]
//   variant: 0=all, 1=mdrange, 2=flat, 3=tiled

#include <Kokkos_Core.hpp>
#include <iostream>
#include <limits>
#include <vector>

using Scalar = float; // change to float if you want
// using View4D = Kokkos::View< Scalar****, Kokkos::LayoutRight, Kokkos::DefaultExecutionSpace >;
// using View4D = Kokkos::View< Scalar****, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace >;
using View4D = Kokkos::View< Scalar**** >;

// Simple fill helper
void fill_view( View4D v, Scalar val )
{
    const int N0 = v.extent( 0 );
    const int N1 = v.extent( 1 );
    const int N2 = v.extent( 2 );
    const int N3 = v.extent( 3 );
    Kokkos::parallel_for(
        "fill_view",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >( { 0, 0, 0, 0 }, { N0, N1, N2, N3 } ),
        KOKKOS_LAMBDA( int a, int i, int j, int k ) { v( a, i, j, k ) = val; } );
    Kokkos::fence();
}

// Variant A: MDRange straightforward
void lincomb_mdrange( const View4D& y, Scalar c0, Scalar c1, const View4D& x1 )
{
    const int N0 = y.extent( 0 );
    const int N1 = y.extent( 1 );
    const int N2 = y.extent( 2 );
    const int N3 = y.extent( 3 );
    Kokkos::parallel_for(
        "lincomb_mdrange",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >( { 0, 0, 0, 0 }, { N0, N1, N2, N3 } ),
        KOKKOS_LAMBDA( int a, int i, int j, int k ) { y( a, i, j, k ) = c0 + c1 * x1( a, i, j, k ); } );
}

// Variant B: flat RangePolicy (1D) with index unravel
void lincomb_flat( const View4D& y, Scalar c0, Scalar c1, const View4D& x1 )
{
    const std::size_t N0 = y.extent( 0 );
    const std::size_t N1 = y.extent( 1 );
    const std::size_t N2 = y.extent( 2 );
    const std::size_t N3 = y.extent( 3 );
    const std::size_t NE = N0 * N1 * N2 * N3;

    Kokkos::parallel_for(
        "lincomb_flat", Kokkos::RangePolicy< std::size_t >( 0, NE ), KOKKOS_LAMBDA( std::size_t idx ) {
            std::size_t r = idx;
            const int   k = r % N3;
            r /= N3;
            const int j = r % N2;
            r /= N2;
            const int i = r % N1;
            r /= N1;
            const int a     = r;
            y( a, i, j, k ) = c0 + c1 * x1( a, i, j, k );
        } );
}

// Variant C: MDRange with tiling (adjust tile sizes for your hardware)
void lincomb_tiled(
    const View4D& y,
    Scalar        c0,
    Scalar        c1,
    const View4D& x1,
    int           t0 = 1,
    int           t1 = 8,
    int           t2 = 8,
    int           t3 = 32 )
{
    const int N0 = y.extent( 0 );
    const int N1 = y.extent( 1 );
    const int N2 = y.extent( 2 );
    const int N3 = y.extent( 3 );

    using Policy = Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right >, Kokkos::IndexType< int > >;
    Policy policy( { 0, 0, 0, 0 }, { N0, N1, N2, N3 }, { t0, t1, t2, t3 } );

    Kokkos::parallel_for(
        "lincomb_tiled", policy, KOKKOS_LAMBDA( int a, int i, int j, int k ) {
            y( a, i, j, k ) = c0 + c1 * x1( a, i, j, k );
        } );
}

struct Result
{
    double time;
};

// Run benchmark for one callable niter times (with warmup) and return best/avg
Result run_benchmark( std::function< void() > fn, int niter = 10, int nwarm = 2 )
{
    Kokkos::fence();
    // warmup
    for ( int i = 0; i < nwarm; ++i )
    {
        fn();
        Kokkos::fence();
    }

    std::vector< double > times;
    times.reserve( niter );
    for ( int it = 0; it < niter; ++it )
    {
        Kokkos::Timer t;
        fn();
        Kokkos::fence();
        const double elapsed = t.seconds();
        times.push_back( elapsed );
    }
    double sum = 0, best = std::numeric_limits< double >::max();
    for ( double v : times )
    {
        sum += v;
        if ( v < best )
            best = v;
    }
    return { sum / times.size() };
}

int main( int argc, char** argv )
{
    Kokkos::initialize( argc, argv );
    {
        int N0 = 8, N1 = 256, N2 = 256, N3 = 64; // default sizes - tune for your machine
        int niter   = 8;
        int variant = 0;
        if ( argc >= 5 )
        {
            N0 = atoi( argv[1] );
            N1 = atoi( argv[2] );
            N2 = atoi( argv[3] );
            N3 = atoi( argv[4] );
        }
        if ( argc >= 6 )
            niter = atoi( argv[5] );
        if ( argc >= 7 )
            variant = atoi( argv[6] );

        const std::size_t NE = std::size_t( N0 ) * N1 * N2 * N3;
        std::cout << "Problem: " << N0 << " x " << N1 << " x " << N2 << " x " << N3 << " = " << NE << " elements\n";

        View4D x1( "x1", N0, N1, N2, N3 );
        View4D y( "y", N0, N1, N2, N3 );

        fill_view( x1, 1.2345 );
        fill_view( y, 0.0 );

        const Scalar c0 = 0.1;
        const Scalar c1 = 2.0;

        const double bytes_per_element = double( sizeof( Scalar ) ) * 2.0; // read x1, write y
        const double flops_per_element = 2.0;                              // c0 + c1*x -> 1 mul + 1 add

        auto run_and_report = [&]( const std::string& name, std::function< void() > fn ) {
            if ( variant != 0 )
            {
                // run only the requested one
            }
            std::cout << "Running: " << name << " ...\n";
            auto   res     = run_benchmark( fn, niter );
            double time    = res.time;
            double gbytes  = ( bytes_per_element * double( NE ) ) / ( time * 1e9 );
            double gflops  = ( flops_per_element * double( NE ) ) / ( time * 1e9 );
            double gpoints = double( NE ) / ( time * 1e9 ); // gridpoints updated per second in billions
            std::cout << "  avg time (s): " << time << "\n";
            std::cout << "  throughput: " << gbytes << " GB/s, " << gflops << " GFLOP/s, " << gpoints << " Gpoints/s\n";
            std::cout << "\n";
        };

        // decide which to run
        if ( variant == 0 || variant == 1 )
        {
            run_and_report( "MDRange (straight)", [&] { lincomb_mdrange( y, c0, c1, x1 ); } );
        }
        if ( variant == 0 || variant == 2 )
        {
            run_and_report( "RangePolicy (flat)", [&] { lincomb_flat( y, c0, c1, x1 ); } );
        }
        if ( variant == 0 || variant == 3 )
        {
            // tile sizes: you can tweak these when testing
            run_and_report( "MDRange tiled", [&] { lincomb_tiled( y, c0, c1, x1, 1, 4, 4, 32 ); } );
        }
    }
    Kokkos::finalize();
    return 0;
}
