
#include <kernels/common/grid_operations.hpp>

#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/info.hpp"
#include "util/table.hpp"

using terra::grid::shell::allocate_scalar_grid;
using terra::grid::shell::DistributedDomain;
using terra::kernels::common::lincomb;
using terra::kernels::common::set_constant;

enum class BenchmarkType : int
{
    LINCOMB_1 = 0,
    LINCOMB_2,
    STENCIL_INNER_CONSTANT_7,
    ELEMENTWISE_CONSTANT_MATVEC,
};

constexpr auto all_benchmark_types = {
    BenchmarkType::LINCOMB_1,
    BenchmarkType::LINCOMB_2,
    BenchmarkType::STENCIL_INNER_CONSTANT_7,
    BenchmarkType::ELEMENTWISE_CONSTANT_MATVEC };

const std::map< BenchmarkType, std::string > benchmark_description = {
    { BenchmarkType::LINCOMB_1, "Simple y <- c0 * x0 (c scalar, y, x vector)" },
    { BenchmarkType::LINCOMB_2, "Simple y <- c0 * x0 + c1 * x1 (c scalar, y, x vector)" },
    { BenchmarkType::STENCIL_INNER_CONSTANT_7,
      "Constant 7-point stencil applied to the inner nodes of each subdomain (y <- Ax)." },
    { BenchmarkType::ELEMENTWISE_CONSTANT_MATVEC,
      "Constant elementwise operator to all elements of each subdomain (bilinear elements, y <- Ax)." },
};

struct BenchmarkData
{
    int    lateral_refinement_level;
    int    radial_refinement_level;
    int    dofs;
    double duration;
};

template < std::floating_point T >
struct ConstantStencilInner
{
    terra::grid::Grid4DDataScalar< T > src;
    terra::grid::Grid4DDataScalar< T > dst;

    ConstantStencilInner( terra::grid::Grid4DDataScalar< T > _src, terra::grid::Grid4DDataScalar< T > _dst )
    : src( _src )
    , dst( _dst )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int subdomain_idx, const int x, const int y, const int r ) const
    {
        dst( subdomain_idx, x, y, r ) = 4.0 * src( subdomain_idx, x, y, r ) -
                                        ( src( subdomain_idx, x + 1, y, r ) + src( subdomain_idx, x - 1, y, r ) +
                                          src( subdomain_idx, x, y + 1, r ) + src( subdomain_idx, x, y - 1, r ) +
                                          src( subdomain_idx, x, y, r + 1 ) + +src( subdomain_idx, x, y, r - 1 ) );
    }
};

template < std::floating_point T >
struct ConstantElementwise
{
    terra::grid::Grid4DDataScalar< T > src;
    terra::grid::Grid4DDataScalar< T > dst;

    ConstantElementwise( terra::grid::Grid4DDataScalar< T > _src, terra::grid::Grid4DDataScalar< T > _dst )
    : src( _src )
    , dst( _dst )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int subdomain_idx, const int x_cell, const int y_cell, const int r_cell ) const
    {
        terra::dense::Vec< T, 8 >    src_local;
        terra::dense::Mat< T, 8, 8 > mat;

        mat.fill( 0.1 );

        for ( int x = x_cell; x < x_cell + 1; x++ )
        {
            for ( int y = y_cell; y < y_cell + 1; y++ )
            {
                for ( int r = r_cell; r < r_cell + 1; r++ )
                {
                    src_local( 4 * ( r - r_cell ) + 2 * ( y - y_cell ) + ( x - x_cell ) ) =
                        src( subdomain_idx, x, y, r );
                }
            }
        }

        terra::dense::Vec< T, 8 > dst_local = mat * src_local;

        for ( int x = x_cell; x < x_cell + 1; x++ )
        {
            for ( int y = y_cell; y < y_cell + 1; y++ )
            {
                for ( int r = r_cell; r < r_cell + 1; r++ )
                {
                    Kokkos::atomic_add(
                        &dst( subdomain_idx, x, y, r ),
                        dst_local( 4 * ( r - r_cell ) + 2 * ( y - y_cell ) + ( x - x_cell ) ) );
                }
            }
        }
    }
};

template < std::floating_point T >
BenchmarkData
    run( const BenchmarkType benchmark,
         const int           lateral_refinement_level,
         const int           radial_refinement_level,
         const int           executions )
{
    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond(
        lateral_refinement_level, radial_refinement_level, 0.5, 1.0 );

    const auto y  = allocate_scalar_grid< T >( "y", domain );
    const auto x0 = allocate_scalar_grid< T >( "x0", domain );
    const auto x1 = allocate_scalar_grid< T >( "x1", domain );

    set_constant( x0, T( 1.0 ) );
    set_constant( x1, T( 1.0 ) );

    if ( !y.span_is_contiguous() )
    {
        std::cout << "Span is not contiguous!" << std::endl;
    }
    const int dofs = y.span();

    Kokkos::Timer timer;

    Kokkos::fence();
    timer.reset();

    for ( int i = 0; i < executions; ++i )
    {
        if ( benchmark == BenchmarkType::LINCOMB_1 )
        {
            lincomb( y, T( 0.0 ), T( 42.0 ), x0 );
        }
        else if ( benchmark == BenchmarkType::LINCOMB_2 )
        {
            lincomb( y, T( 0.0 ), T( 42.0 ), x0, T( 4711.0 ), x1 );
        }
        else if ( benchmark == BenchmarkType::STENCIL_INNER_CONSTANT_7 )
        {
            Kokkos::parallel_for(
                "stencil",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                    { 0, 1, 1, 1 }, { y.extent( 0 ), y.extent( 1 ) - 1, y.extent( 2 ) - 1, y.extent( 3 ) - 1 } ),
                ConstantStencilInner< T >( x0, y ) );
        }
        else if ( benchmark == BenchmarkType::ELEMENTWISE_CONSTANT_MATVEC )
        {
            Kokkos::parallel_for(
                "elementwise",
                terra::grid::shell::local_domain_md_range_policy_cells( domain ),
                ConstantElementwise< T >( x0, y ) );
        }
    }

    Kokkos::fence();
    const double duration = timer.seconds() / executions;

    // This does not seem to be necessary but leaving it here just in case.
    const bool print_derived_value = false;
    if ( print_derived_value )
    {
        const auto mm = terra::kernels::common::max_abs_entry( y );
        std::cout << "Printing some derived value to ensure nothing is optimized out: " << mm << std::endl;
    }

    return BenchmarkData{ lateral_refinement_level, radial_refinement_level, dofs, duration };
}

void run_all()
{
    const int min_level  = 2;
    const int max_level  = 7;
    const int executions = 5;

    for ( auto benchmark : all_benchmark_types )
    {
        std::cout << benchmark_description.at( benchmark ) << std::endl;

        terra::util::Table table;

        for ( int i = min_level; i <= max_level; ++i )
        {
            BenchmarkData data_float  = run< float >( benchmark, i, i, executions );
            BenchmarkData data_double = run< double >( benchmark, i, i, executions );

            if ( data_double.dofs != data_float.dofs )
            {
                Kokkos::abort( "Dofs should be equal." );
            }

            table.add_row(
                { { "lateral level", i },
                  { "radial level", i },
                  { "dofs", data_double.dofs },
                  { "[float] duration (s)", data_float.duration },
                  { "[float] updated dofs/sec", data_float.dofs / data_float.duration },
                  { "[double] duration (s)", data_double.duration },
                  { "[double] updated dofs/sec", data_double.dofs / data_double.duration } } );
        }

        table.print_pretty();

        std::cout << std::endl;
        std::cout << std::endl;
    }
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    if ( terra::mpi::num_processes() != 1 )
    {
        Kokkos::abort( "Benchmark not set up for more than 1 MPI process yet." );
    }

    terra::util::print_general_info( argc, argv );

    run_all();

    MPI_Finalize();
}