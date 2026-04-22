#pragma once

#include "grid/grid_types.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "mpi/mpi.hpp"
#include "terra/grid/bit_masks.hpp"
#include "terra/util/bit_masking.hpp"

namespace terra::kernels::common {

template < typename ScalarType >
void set_constant( const grid::Grid2DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::deep_copy( x, value );
}

template < typename ScalarType >
void set_constant( const grid::Grid3DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::deep_copy( x, value );
}

template < typename ScalarType >
void set_constant( const grid::Grid4DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::deep_copy( x, value );
}

template < typename ScalarType, int VecDim >
void set_constant( const grid::Grid4DDataVec< ScalarType, VecDim >& x, ScalarType value )
{
    for ( int d = 0; d < VecDim; ++d )
        set_constant( x.comp_[d], value );
}

template < typename ScalarType >
void set_constant( const grid::Grid5DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::deep_copy( x, value );
}

template < typename ScalarType >
void scale( const grid::Grid4DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::parallel_for(
        "scale (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) { x( local_subdomain, i, j, k ) *= value; } );

    Kokkos::fence();
}

template < typename ScalarType, util::FlagLike FlagType >
void assign_masked_else_keep_old(
    const grid::Grid4DDataScalar< ScalarType >& dst,
    const ScalarType&                           value,
    const grid::Grid4DDataScalar< FlagType >&   mask_grid,
    const FlagType                              mask_value )
{
    Kokkos::parallel_for(
        "assign_masked",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { dst.extent( 0 ), dst.extent( 1 ), dst.extent( 2 ), dst.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            const ScalarType mask_val = util::has_flag( mask_grid( local_subdomain, i, j, k ), mask_value ) ? 1.0 : 0.0;
            dst( local_subdomain, i, j, k ) = mask_val * value + ( 1.0 - mask_val ) * dst( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType, util::FlagLike FlagType >
void assign_masked_else_keep_old(
    const grid::Grid4DDataScalar< ScalarType >& dst,
    const grid::Grid4DDataScalar< ScalarType >& src,
    const grid::Grid4DDataScalar< FlagType >&   mask_grid,
    const FlagType                              mask_value )
{
    Kokkos::parallel_for(
        "assign_masked",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { dst.extent( 0 ), dst.extent( 1 ), dst.extent( 2 ), dst.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            const ScalarType mask_val = util::has_flag( mask_grid( local_subdomain, i, j, k ), mask_value ) ? 1.0 : 0.0;
            dst( local_subdomain, i, j, k ) =
                mask_val * src( local_subdomain, i, j, k ) + ( 1.0 - mask_val ) * dst( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim, util::FlagLike FlagType >
void assign_masked_else_keep_old(
    const grid::Grid4DDataVec< ScalarType, VecDim >& dst,
    const ScalarType&                                value,
    const grid::Grid4DDataScalar< FlagType >&        mask_grid,
    const FlagType                                   mask_value )
{
    for ( int d = 0; d < VecDim; ++d )
        assign_masked_else_keep_old( dst.comp_[d], value, mask_grid, mask_value );
}

template < typename ScalarType, int VecDim, util::FlagLike FlagType >
void assign_masked_else_keep_old(
    const grid::Grid4DDataVec< ScalarType, VecDim >& dst,
    const grid::Grid4DDataVec< ScalarType, VecDim >& src,
    const grid::Grid4DDataScalar< FlagType >&        mask_grid,
    const FlagType                                   mask_value )
{
    for ( int d = 0; d < VecDim; ++d )
        assign_masked_else_keep_old( dst.comp_[d], src.comp_[d], mask_grid, mask_value );
}

template < typename ScalarType, int VecDim, util::FlagLike FlagType >
void assign_masked_else_keep_old(
    const grid::Grid4DDataVec< ScalarType, VecDim >& dst,
    const ScalarType&                                value,
    const grid::Grid4DDataScalar< FlagType >&        mask_grid,
    const FlagType                                   mask_value,
    const int                                        vector_component )
{
    Kokkos::parallel_for(
        "assign_masked",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { dst.extent( 0 ), dst.extent( 1 ), dst.extent( 2 ), dst.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            const ScalarType mask_val = util::has_flag( mask_grid( local_subdomain, i, j, k ), mask_value ) ? 1.0 : 0.0;
            dst( local_subdomain, i, j, k, vector_component ) =
                mask_val * value + ( 1.0 - mask_val ) * dst( local_subdomain, i, j, k, vector_component );
        } );

    Kokkos::fence();
}

template < typename ScalarType >
void lincomb(
    const grid::Grid4DDataScalar< ScalarType >& y,
    ScalarType                                  c_0,
    ScalarType                                  c_1,
    const grid::Grid4DDataScalar< ScalarType >& x_1 )
{
    Kokkos::parallel_for(
        "lincomb 1 arg (Grid4DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) = c_0 + c_1 * x_1( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType >
void lincomb(
    const grid::Grid4DDataScalar< ScalarType >& y,
    ScalarType                                  c_0,
    ScalarType                                  c_1,
    const grid::Grid4DDataScalar< ScalarType >& x_1,
    ScalarType                                  c_2,
    const grid::Grid4DDataScalar< ScalarType >& x_2 )
{
    Kokkos::parallel_for(
        "lincomb 2 args (Grid4DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) =
                c_0 + c_1 * x_1( local_subdomain, i, j, k ) + c_2 * x_2( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType >
void lincomb(
    const grid::Grid4DDataScalar< ScalarType >& y,
    ScalarType                                  c_0,
    ScalarType                                  c_1,
    const grid::Grid4DDataScalar< ScalarType >& x_1,
    ScalarType                                  c_2,
    const grid::Grid4DDataScalar< ScalarType >& x_2,
    ScalarType                                  c_3,
    const grid::Grid4DDataScalar< ScalarType >& x_3 )
{
    Kokkos::parallel_for(
        "lincomb 3 args (Grid4DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) = c_0 + c_1 * x_1( local_subdomain, i, j, k ) +
                                            c_2 * x_2( local_subdomain, i, j, k ) +
                                            c_3 * x_3( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim >
void lincomb(
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    ScalarType                                       c_0,
    ScalarType                                       c_1,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_1 )
{
    for ( int d = 0; d < VecDim; ++d )
        lincomb( y.comp_[d], c_0, c_1, x_1.comp_[d] );
}

template < typename ScalarType, int VecDim >
void lincomb(
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    ScalarType                                       c_0,
    ScalarType                                       c_1,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_1,
    ScalarType                                       c_2,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_2 )
{
    for ( int d = 0; d < VecDim; ++d )
        lincomb( y.comp_[d], c_0, c_1, x_1.comp_[d], c_2, x_2.comp_[d] );
}

template < typename ScalarType, int VecDim >
void lincomb(
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    ScalarType                                       c_0,
    ScalarType                                       c_1,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_1,
    ScalarType                                       c_2,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_2,
    ScalarType                                       c_3,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_3 )
{
    for ( int d = 0; d < VecDim; ++d )
        lincomb( y.comp_[d], c_0, c_1, x_1.comp_[d], c_2, x_2.comp_[d], c_3, x_3.comp_[d] );
}

template < typename ScalarType >
void invert_inplace( const grid::Grid4DDataScalar< ScalarType >& y )
{
    Kokkos::parallel_for(
        "invert",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) = 1.0 / y( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim >
void invert_inplace( const grid::Grid4DDataVec< ScalarType, VecDim >& y )
{
    for ( int d = 0; d < VecDim; ++d )
        invert_inplace( y.comp_[d] );
}

template < typename ScalarType >
void mult_elementwise_inplace(
    const grid::Grid4DDataScalar< ScalarType >& y,
    const grid::Grid4DDataScalar< ScalarType >& x )
{
    Kokkos::parallel_for(
        "mult_elementwise_inplace",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) *= x( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim >
void mult_elementwise_inplace(
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x )
{
    for ( int d = 0; d < VecDim; ++d )
        mult_elementwise_inplace( y.comp_[d], x.comp_[d] );
}

template < typename ScalarType >
ScalarType min_entry( const grid::Grid4DDataScalar< ScalarType >& x )
{
    ScalarType min_val = 0.0;
    Kokkos::parallel_reduce(
        "min_entry",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_min ) {
            ScalarType val = x( local_subdomain, i, j, k );
            local_min      = Kokkos::min( local_min, val );
        },
        Kokkos::Min< ScalarType >( min_val ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &min_val, 1, mpi::mpi_datatype< ScalarType >(), MPI_MIN, MPI_COMM_WORLD );

    return min_val;
}

template < typename ScalarType >
ScalarType min_abs_entry( const grid::Grid4DDataScalar< ScalarType >& x )
{
    ScalarType min_mag = 0.0;
    Kokkos::parallel_reduce(
        "min_abs_entry",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_min ) {
            ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k ) );
            local_min      = Kokkos::min( local_min, val );
        },
        Kokkos::Min< ScalarType >( min_mag ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &min_mag, 1, mpi::mpi_datatype< ScalarType >(), MPI_MIN, MPI_COMM_WORLD );

    return min_mag;
}

template < typename ScalarType >
ScalarType max_abs_entry( const grid::Grid4DDataScalar< ScalarType >& x )
{
    ScalarType max_mag = 0.0;
    Kokkos::parallel_reduce(
        "max_abs_entry",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_max ) {
            ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k ) );
            local_max      = Kokkos::max( local_max, val );
        },
        Kokkos::Max< ScalarType >( max_mag ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &max_mag, 1, mpi::mpi_datatype< ScalarType >(), MPI_MAX, MPI_COMM_WORLD );

    return max_mag;
}

template < typename ScalarType, int VecDim >
ScalarType max_abs_entry( const grid::Grid4DDataVec< ScalarType, VecDim >& x )
{
    ScalarType max_mag = 0.0;
    for ( int d = 0; d < VecDim; ++d )
    {
        // max_abs_entry for scalar internally does MPI_Allreduce,
        // so we call per-component and take the max on the host.
        ScalarType comp_max = max_abs_entry( x.comp_[d] );
        max_mag             = std::max( max_mag, comp_max );
    }
    return max_mag;
}

template < typename ScalarType, util::FlagLike FlagType >
ScalarType max_abs_entry(
    const grid::Grid4DDataScalar< ScalarType >& x,
    const grid::Grid4DDataScalar< FlagType >&   mask,
    const FlagType&                             mask_value )
{
    ScalarType max_mag = 0.0;
    Kokkos::parallel_reduce(
        "max_abs_entry",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_max ) {
            if ( util::has_flag( mask( local_subdomain, i, j, k ), mask_value ) )
            {
                ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k ) );
                local_max      = Kokkos::max( local_max, val );
            }
        },
        Kokkos::Max< ScalarType >( max_mag ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &max_mag, 1, mpi::mpi_datatype< ScalarType >(), MPI_MAX, MPI_COMM_WORLD );

    return max_mag;
}

template < typename ScalarType >
ScalarType max_abs_entry_subset(
    const grid::Grid4DDataScalar< ScalarType >& x,
    dense::Vec< int, 4 >                        start,
    dense::Vec< int, 4 >                        end_excl )
{
    ScalarType max_mag = 0.0;
    Kokkos::parallel_reduce(
        "max_abs_entry",
        Kokkos::MDRangePolicy(
            { start( 0 ), start( 1 ), start( 2 ), start( 3 ) },
            { end_excl( 0 ), end_excl( 1 ), end_excl( 2 ), end_excl( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_max ) {
            ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k ) );
            local_max      = Kokkos::max( local_max, val );
        },
        Kokkos::Max< ScalarType >( max_mag ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &max_mag, 1, mpi::mpi_datatype< ScalarType >(), MPI_MAX, MPI_COMM_WORLD );

    return max_mag;
}

template < typename ScalarType, int VecDim >
ScalarType max_vector_magnitude( const grid::Grid4DDataVec< ScalarType, VecDim >& x )
{
    ScalarType max_mag = 0.0;
    Kokkos::parallel_reduce(
        "max_vector_magnitude",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_max ) {
            ScalarType val = 0;
            for ( int d = 0; d < VecDim; ++d )
            {
                val += x( local_subdomain, i, j, k, d ) * x( local_subdomain, i, j, k, d );
            }
            val       = Kokkos::sqrt( val );
            local_max = Kokkos::max( local_max, val );
        },
        Kokkos::Max< ScalarType >( max_mag ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &max_mag, 1, mpi::mpi_datatype< ScalarType >(), MPI_MAX, MPI_COMM_WORLD );

    return max_mag;
}

template < typename ScalarType, int VecDim >
void vector_magnitude(
    grid::Grid4DDataScalar< ScalarType >&            magnitude_out,
    const grid::Grid4DDataVec< ScalarType, VecDim >& vectorial_data_in )
{
    Kokkos::parallel_for(
        "vector_magnitude",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0 },
            { vectorial_data_in.extent( 0 ),
              vectorial_data_in.extent( 1 ),
              vectorial_data_in.extent( 2 ),
              vectorial_data_in.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            ScalarType val = 0;
            for ( int d = 0; d < VecDim; ++d )
            {
                val +=
                    vectorial_data_in( local_subdomain, i, j, k, d ) * vectorial_data_in( local_subdomain, i, j, k, d );
            }
            magnitude_out( local_subdomain, i, j, k ) = Kokkos::sqrt( val );
        } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim >
void extract_vector_component(
    grid::Grid4DDataScalar< ScalarType >&            component_out,
    const grid::Grid4DDataVec< ScalarType, VecDim >& vectorial_data_in,
    const int                                        component )
{
    if ( component < 0 || component >= VecDim )
    {
        Kokkos::abort( "Vector component invalid." );
    }

    Kokkos::parallel_for(
        "extract_vector_component",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0 },
            { vectorial_data_in.extent( 0 ),
              vectorial_data_in.extent( 1 ),
              vectorial_data_in.extent( 2 ),
              vectorial_data_in.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            component_out( local_subdomain, i, j, k ) = vectorial_data_in( local_subdomain, i, j, k, component );
        } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim >
void set_vector_component(
    grid::Grid4DDataVec< ScalarType, VecDim >& vectorial_data,
    const int                                  component,
    const ScalarType                           constant )
{
    if ( component < 0 || component >= VecDim )
    {
        Kokkos::abort( "Vector component invalid." );
    }

    Kokkos::parallel_for(
        "set_vector_component",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0 },
            { vectorial_data.extent( 0 ),
              vectorial_data.extent( 1 ),
              vectorial_data.extent( 2 ),
              vectorial_data.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            vectorial_data( local_subdomain, i, j, k, component ) = constant;
        } );

    Kokkos::fence();
}

template < typename ScalarType >
ScalarType sum_of_absolutes( const grid::Grid4DDataScalar< ScalarType >& x )
{
    ScalarType sum_abs = 0.0;
    Kokkos::parallel_reduce(
        "sum_of_absolutes",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_sum_abs ) {
            ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k ) );
            local_sum_abs  = local_sum_abs + val;
        },
        Kokkos::Sum< ScalarType >( sum_abs ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &sum_abs, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return sum_abs;
}

template < typename ScalarType, util::FlagLike FlagType >
ScalarType count_masked( const grid::Grid4DDataScalar< FlagType >& mask, const FlagType& mask_value )
{
    auto count = static_cast< ScalarType >( 0 );

    Kokkos::parallel_reduce(
        "masked_sum",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0 }, { mask.extent( 0 ), mask.extent( 1 ), mask.extent( 2 ), mask.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_sum ) {
            const ScalarType mask_val = util::has_flag( mask( local_subdomain, i, j, k ), mask_value ) ?
                                            static_cast< ScalarType >( 1 ) :
                                            static_cast< ScalarType >( 0 );
            local_sum                 = local_sum + mask_val;
        },
        Kokkos::Sum< ScalarType >( count ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &count, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return count;
}

template < typename ScalarType, util::FlagLike FlagType >
ScalarType masked_sum(
    const grid::Grid4DDataScalar< ScalarType >& x,
    const grid::Grid4DDataScalar< FlagType >&   mask,
    const FlagType&                             mask_value )
{
    ScalarType sum = 0.0;

    Kokkos::parallel_reduce(
        "masked_sum",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_sum ) {
            const ScalarType mask_val = util::has_flag( mask( local_subdomain, i, j, k ), mask_value ) ? 1.0 : 0.0;
            ScalarType       val      = x( local_subdomain, i, j, k ) * mask_val;
            local_sum                 = local_sum + val;
        },
        Kokkos::Sum< ScalarType >( sum ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &sum, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return sum;
}

template < typename ScalarType, util::FlagLike FlagType0, util::FlagLike FlagType1 >
ScalarType masked_sum(
    const grid::Grid4DDataScalar< ScalarType >& x,
    const grid::Grid4DDataScalar< FlagType0 >&  mask0,
    const grid::Grid4DDataScalar< FlagType1 >&  mask1,
    const FlagType0&                            mask0_value,
    const FlagType1&                            mask1_value )
{
    ScalarType sum = 0.0;

    Kokkos::parallel_reduce(
        "masked_sum",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_sum ) {
            ScalarType mask_val = 1.0;
            mask_val *= util::has_flag( mask0( local_subdomain, i, j, k ), mask0_value ) ? 1.0 : 0.0;
            mask_val *= util::has_flag( mask1( local_subdomain, i, j, k ), mask1_value ) ? 1.0 : 0.0;
            ScalarType val = x( local_subdomain, i, j, k ) * mask_val;
            local_sum      = local_sum + val;
        },
        Kokkos::Sum< ScalarType >( sum ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &sum, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return sum;
}

template < typename ScalarType >
ScalarType dot_product( const grid::Grid4DDataScalar< ScalarType >& x, const grid::Grid4DDataScalar< ScalarType >& y )
{
    ScalarType dot_prod = 0.0;

    Kokkos::parallel_reduce(
        "dot_product",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_dot_prod ) {
            ScalarType val = x( local_subdomain, i, j, k ) * y( local_subdomain, i, j, k );
            local_dot_prod = local_dot_prod + val;
        },
        Kokkos::Sum< ScalarType >( dot_prod ) );

    Kokkos::fence( "dot_product" );

    MPI_Allreduce( MPI_IN_PLACE, &dot_prod, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return dot_prod;
}

template < typename ScalarType, util::FlagLike FlagType >
ScalarType masked_dot_product(
    const grid::Grid4DDataScalar< ScalarType >& x,
    const grid::Grid4DDataScalar< ScalarType >& y,
    const grid::Grid4DDataScalar< FlagType >&   mask,
    const FlagType&                             mask_value )
{
    ScalarType dot_prod = 0.0;

    Kokkos::parallel_reduce(
        "masked_dot_product",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_dot_prod ) {
            const ScalarType mask_val = util::has_flag( mask( local_subdomain, i, j, k ), mask_value ) ? 1.0 : 0.0;
            ScalarType       val      = x( local_subdomain, i, j, k ) * y( local_subdomain, i, j, k ) * mask_val;
            local_dot_prod            = local_dot_prod + val;
        },
        Kokkos::Sum< ScalarType >( dot_prod ) );

    Kokkos::fence( "masked_dot_product" );

    MPI_Allreduce( MPI_IN_PLACE, &dot_prod, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return dot_prod;
}

template < typename ScalarType, util::FlagLike FlagType, int VecDim >
ScalarType masked_dot_product(
    const grid::Grid4DDataVec< ScalarType, VecDim >& x,
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    const grid::Grid4DDataScalar< FlagType >&        mask,
    const FlagType&                                  mask_value )
{
    ScalarType dot_prod = 0.0;
    for ( int d = 0; d < VecDim; ++d )
    {
        // masked_dot_product for scalar internally does MPI_Allreduce,
        // so we accumulate the per-component results.
        dot_prod += masked_dot_product( x.comp_[d], y.comp_[d], mask, mask_value );
    }
    return dot_prod;
}

template < typename ScalarType >
ScalarType dot_product_subset(
    const grid::Grid4DDataScalar< ScalarType >& x,
    const grid::Grid4DDataScalar< ScalarType >& y,
    dense::Vec< int, 4 >                        start,
    dense::Vec< int, 4 >                        end_excl )
{
    ScalarType dot_prod = 0.0;

    Kokkos::parallel_reduce(
        "masked_dot_product",
        Kokkos::MDRangePolicy(
            { start( 0 ), start( 1 ), start( 2 ), start( 3 ) },
            { end_excl( 0 ), end_excl( 1 ), end_excl( 2 ), end_excl( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_dot_prod ) {
            ScalarType val = x( local_subdomain, i, j, k ) * y( local_subdomain, i, j, k );
            local_dot_prod = local_dot_prod + val;
        },
        Kokkos::Sum< ScalarType >( dot_prod ) );

    Kokkos::fence( "masked_dot_product" );

    MPI_Allreduce( MPI_IN_PLACE, &dot_prod, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

    return dot_prod;
}

template < typename ScalarType >
bool has_nan_or_inf( const grid::Grid4DDataScalar< ScalarType >& x )
{
    bool has_nan_or_inf = false;

    Kokkos::parallel_reduce(
        "masked_dot_product",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, bool& local_has_nan_or_inf ) {
            local_has_nan_or_inf = local_has_nan_or_inf || ( Kokkos::isnan( x( local_subdomain, i, j, k ) ) ||
                                                             Kokkos::isinf( x( local_subdomain, i, j, k ) ) );
        },
        Kokkos::LOr< bool >( has_nan_or_inf ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &has_nan_or_inf, 1, mpi::mpi_datatype< bool >(), MPI_LOR, MPI_COMM_WORLD );

    return has_nan_or_inf;
}

template < typename ScalarType, int VecDim >
bool has_nan_or_inf( const grid::Grid4DDataVec< ScalarType, VecDim >& x )
{
    for ( int d = 0; d < VecDim; ++d )
    {
        if ( has_nan_or_inf( x.comp_[d] ) )
            return true;
    }
    return false;
}

template < typename ScalarTypeDst, typename ScalarTypeSrc >
void cast( const grid::Grid4DDataScalar< ScalarTypeDst >& dst, const grid::Grid4DDataScalar< ScalarTypeSrc >& src )
{
    Kokkos::parallel_for(
        "cast",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { dst.extent( 0 ), dst.extent( 1 ), dst.extent( 2 ), dst.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            dst( local_subdomain, i, j, k ) = static_cast< ScalarTypeDst >( src( local_subdomain, i, j, k ) );
        } );

    Kokkos::fence();
}

template < typename ScalarTypeDst >
void rand( const grid::Grid4DDataScalar< ScalarTypeDst >& dst )
{
    static_assert(
        std::is_same_v< ScalarTypeDst, double > || std::is_same_v< ScalarTypeDst, float >,
        "Random integers not implemented. But can be done easily below." );

    Kokkos::Random_XorShift64_Pool<> random_pool( /*seed=*/12345 );
    Kokkos::parallel_for(
        "rand",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { dst.extent( 0 ), dst.extent( 1 ), dst.extent( 2 ), dst.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            auto generator                  = random_pool.get_state();
            dst( local_subdomain, i, j, k ) = static_cast< ScalarTypeDst >( generator.drand() );
            random_pool.free_state( generator );
        } );

    Kokkos::fence();
}

template < typename ScalarTypeDst, int VecDim >
void rand( const grid::Grid4DDataVec< ScalarTypeDst, VecDim >& dst )
{
    for ( int d = 0; d < VecDim; ++d )
        rand( dst.comp_[d] );
}

} // namespace terra::kernels::common
