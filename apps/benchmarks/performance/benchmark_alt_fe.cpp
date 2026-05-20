#include <Kokkos_Core.hpp>
#include <iostream>

using Scalar = float;
// using FieldView = Kokkos::View< Scalar****, Kokkos::LayoutRight >;
// using FieldView = Kokkos::View< Scalar****, Kokkos::LayoutLeft >;
using FieldView = Kokkos::View< Scalar**** >;

struct BenchmarkResult
{
    double avg_time;
    double gnode_updates_per_sec;
};

template < typename Kernel >
BenchmarkResult run_benchmark( Kernel kernel, int niter, int nnodes )
{
    Kokkos::Timer timer;
    for ( int it = 0; it < niter; ++it )
    {
        kernel();
    }
    Kokkos::fence();
    double total         = timer.seconds();
    double avg           = total / niter;
    double gnode_updates = ( nnodes ) / ( avg * 1e9 );
    return { avg, gnode_updates };
}

// ---------------- Kernels ----------------

void fe_kernel_range( FieldView out, const FieldView in )
{
    int A  = out.extent( 0 );
    int NX = out.extent( 1 );
    int NY = out.extent( 2 );
    int NZ = out.extent( 3 );

    int ncells = A * ( NX - 1 ) * ( NY - 1 ) * ( NZ - 1 );

    Kokkos::parallel_for(
        "fe_kernel_range", Kokkos::RangePolicy<>( 0, ncells ), KOKKOS_LAMBDA( int idx ) {
            int k = idx % ( NZ - 1 );
            int j = ( idx / ( NZ - 1 ) ) % ( NY - 1 );
            int i = ( idx / ( ( NZ - 1 ) * ( NY - 1 ) ) ) % ( NX - 1 );
            int a = idx / ( ( NZ - 1 ) * ( NY - 1 ) * ( NX - 1 ) );

            for ( int di = 0; di <= 1; ++di )
                for ( int dj = 0; dj <= 1; ++dj )
                    for ( int dk = 0; dk <= 1; ++dk )
                    {
                        int ni = i + di, nj = j + dj, nk = k + dk;
                        Kokkos::atomic_add( &out( a, ni, nj, nk ), 0.125 * in( a, ni, nj, nk ) );
                    }
        } );
}

void fe_kernel_mdrange( FieldView out, const FieldView in )
{
    int A  = out.extent( 0 );
    int NX = out.extent( 1 );
    int NY = out.extent( 2 );
    int NZ = out.extent( 3 );

    Kokkos::parallel_for(
        "fe_kernel_mdrange",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >( { 0, 0, 0, 0 }, { A, NX - 1, NY - 1, NZ - 1 } ),
        KOKKOS_LAMBDA( int a, int i, int j, int k ) {
            for ( int di = 0; di <= 1; ++di )
                for ( int dj = 0; dj <= 1; ++dj )
                    for ( int dk = 0; dk <= 1; ++dk )
                    {
                        int ni = i + di, nj = j + dj, nk = k + dk;
                        Kokkos::atomic_add( &out( a, ni, nj, nk ), 0.125 * in( a, ni, nj, nk ) );
                    }
        } );
}

int main( int argc, char* argv[] )
{
    Kokkos::initialize( argc, argv );
    {
        int       A = 10, NX = 129, NY = 129, NZ = 129;
        FieldView in( "in", A, NX, NY, NZ );
        FieldView out( "out", A, NX, NY, NZ );

        // fill input with 1.0
        auto h_in = Kokkos::create_mirror_view( in );
        for ( int a = 0; a < A; a++ )
            for ( int i = 0; i < NX; i++ )
                for ( int j = 0; j < NY; j++ )
                    for ( int k = 0; k < NZ; k++ )
                        h_in( a, i, j, k ) = 1.0;
        Kokkos::deep_copy( in, h_in );

        int nnodes = A * NX * NY * NZ;
        int niter  = 5;

        auto res1 = run_benchmark( [&]() { fe_kernel_range( out, in ); }, niter, nnodes );
        auto res2 = run_benchmark( [&]() { fe_kernel_mdrange( out, in ); }, niter, nnodes );

        std::cout << "FE kernel RangePolicy: " << res1.gnode_updates_per_sec << " Gnode/s\n";
        std::cout << "FE kernel MDRangePolicy: " << res2.gnode_updates_per_sec << " Gnode/s\n";
    }
    Kokkos::finalize();
}
