
#pragma once

#include <type_traits>

#include "linalg/vector_q1.hpp"
#include "terra/eigen/eigen_wrapper.hpp"
#include "terra/linalg/operator.hpp"

namespace terra::linalg::util {

/// @brief Assembles a VectorQ1Scalar into a Eigen::SparseVector. Likely not efficient - only use for debugging.
/// @note Only works in the serial case and on host space (only for debugging!).
template < typename ScalarType >
Eigen::SparseVector< ScalarType > debug_sparse_assembly_vector_vec_q1_scalar( const VectorQ1Scalar< double >& vec )
{
    if ( mpi::num_processes() != 1 )
    {
        throw std::runtime_error( "debug_sparse_assembly_vector_vec_q1_scalar: only works in serial case" );
    }

    if ( !Kokkos::SpaceAccessibility< Kokkos::HostSpace,
                                      typename std::decay_t< decltype( vec.grid_data() ) >::memory_space >::accessible )
    {
        throw std::runtime_error( "debug_sparse_assembly_vector_vec_q1_scalar: vec must be on host space" );
    }

    const auto rows = kernels::common::count_masked< long >( vec.mask_data(), grid::NodeOwnershipFlag::OWNED );

    Eigen::SparseVector< ScalarType > result( rows );

    int row = 0;

    for ( int idx_0 = 0; idx_0 < vec.grid_data().extent( 0 ); ++idx_0 )
    {
        for ( int idx_1 = 0; idx_1 < vec.grid_data().extent( 1 ); ++idx_1 )
        {
            for ( int idx_2 = 0; idx_2 < vec.grid_data().extent( 2 ); ++idx_2 )
            {
                for ( int idx_3 = 0; idx_3 < vec.grid_data().extent( 3 ); ++idx_3 )
                {
                    if ( !terra::util::has_flag(
                             vec.mask_data()( idx_0, idx_1, idx_2, idx_3 ), grid::NodeOwnershipFlag::OWNED ) )
                    {
                        continue;
                    }

                    const auto value = vec.grid_data()( idx_0, idx_1, idx_2, idx_3 );
                    if ( value != 0.0 )
                    {
                        result.insert( row ) = value;
                    }

                    row++;
                }
            }
        }
    }

    return result;
}

/// @brief Brute force sparse matrix assembly for debugging purposes.
///
/// Let the operator be representable by a nxm matrix. The assembly involves m matrix-vector multiplications, one with
/// each of the standard basis vectors {e^1, ..., e^m} of R^m (per matrix-vector multiplication one column of the
/// operator is 'assembled').
///
/// Needless to say, this is a very inefficient but flexible approach as any kind of operator can be assembled easily,
/// but it involves many (n) matrix-vector multiplications.
///
/// @note ONLY FOR TESTING/DEBUGGING PURPOSES - EXPECTED TO BE EXTREMELY SLOW
/// @note Only works when running in host space, in serial.
template < OperatorLike Operator >
Eigen::SparseMatrix< double > debug_sparse_assembly_operator_vec_q1_scalar(
    const grid::shell::DistributedDomain& domain_src,
    Operator&                             A,
    VectorQ1Scalar< double >&             tmp_src,
    VectorQ1Scalar< double >&             tmp_dst )
{
    static_assert( std::is_same_v< SrcOf< Operator >, VectorQ1Scalar< double > > );
    static_assert( std::is_same_v< DstOf< Operator >, VectorQ1Scalar< double > > );

    if ( mpi::num_processes() != 1 )
    {
        throw std::runtime_error( "debug_sparse_assembly_operator_vec_q1_scalar: only works in serial case" );
    }

    if ( !Kokkos::SpaceAccessibility< Kokkos::HostSpace,
                                      typename std::decay_t< decltype( tmp_src.grid_data() ) >::memory_space >::accessible )
    {
        throw std::runtime_error( "debug_sparse_assembly_vec_q1_scalar: tmp_src must be on host space" );
    }

    const auto rows = kernels::common::count_masked< long >( tmp_dst.mask_data(), grid::NodeOwnershipFlag::OWNED );
    const auto cols = kernels::common::count_masked< long >( tmp_src.mask_data(), grid::NodeOwnershipFlag::OWNED );

    Eigen::SparseMatrix< double > mat( rows, cols );

    assign( tmp_src, 0.0 );

    int A_col = 0;

    for ( int idx_0 = 0; idx_0 < tmp_src.grid_data().extent( 0 ); ++idx_0 )
    {
        for ( int idx_1 = 0; idx_1 < tmp_src.grid_data().extent( 1 ); ++idx_1 )
        {
            for ( int idx_2 = 0; idx_2 < tmp_src.grid_data().extent( 2 ); ++idx_2 )
            {
                for ( int idx_3 = 0; idx_3 < tmp_src.grid_data().extent( 3 ); ++idx_3 )
                {
                    if ( !terra::util::has_flag(
                             tmp_src.mask_data()( idx_0, idx_1, idx_2, idx_3 ), grid::NodeOwnershipFlag::OWNED ) )
                    {
                        continue;
                    }

                    assign( tmp_src, 0.0 );
                    tmp_src.grid_data()( idx_0, idx_1, idx_2, idx_3 ) = 1.0;

                    communication::shell::send_recv(
                        domain_src, tmp_src.grid_data(), communication::CommunicationReduction::MAX );

                    linalg::apply( A, tmp_src, tmp_dst );

                    int A_row = 0;

                    for ( int iidx_0 = 0; iidx_0 < tmp_dst.grid_data().extent( 0 ); ++iidx_0 )
                    {
                        for ( int iidx_1 = 0; iidx_1 < tmp_dst.grid_data().extent( 1 ); ++iidx_1 )
                        {
                            for ( int iidx_2 = 0; iidx_2 < tmp_dst.grid_data().extent( 2 ); ++iidx_2 )
                            {
                                for ( int iidx_3 = 0; iidx_3 < tmp_dst.grid_data().extent( 3 ); ++iidx_3 )
                                {
                                    if ( !terra::util::has_flag(
                                             tmp_dst.mask_data()( iidx_0, iidx_1, iidx_2, iidx_3 ),
                                             grid::NodeOwnershipFlag::OWNED ) )
                                    {
                                        continue;
                                    }

                                    const auto value = tmp_dst.grid_data()( iidx_0, iidx_1, iidx_2, iidx_3 );
                                    if ( value != 0.0 )
                                    {
                                        mat.insert( A_row, A_col ) =
                                            tmp_dst.grid_data()( iidx_0, iidx_1, iidx_2, iidx_3 );
                                    }

                                    A_row++;
                                }
                            }
                        }
                    }

                    A_col++;
                }
            }
        }
    }

    mat.makeCompressed();
    return mat;
}

} // namespace terra::linalg::util