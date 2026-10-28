
#pragma once

#include <Kokkos_UnorderedMap.hpp>

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_w1.hpp"

using terra::fe::wedge::num_nodes_per_wedge;
using terra::fe::wedge::num_wedges_per_hex_cell;

namespace terra::linalg::solvers {

struct SelectiveStorageKey
{
    int local_subdomain_id, x_cell, y_cell, r_cell, wedge;
};

///
template < typename ScalarT, int LocalMatrixDim >
class LocalMatrixStorage
{
  public:
    using ScalarType = ScalarT;

    using LocalMatrixType = terra::dense::Mat< ScalarType, LocalMatrixDim, LocalMatrixDim >;

  private:
    // coarsest grid boolean field for elements on which a GCA hierarchy has to be built
    int                                                                              level_range_;
    grid::Grid4DDataScalar< ScalarType >                                             GCAElements_;
    terra::grid::Grid4DDataMatrices< ScalarType, LocalMatrixDim, LocalMatrixDim, 2 > local_matrices_full_;
    Kokkos::View< dense::Mat< ScalarType, LocalMatrixDim, LocalMatrixDim >*, terra::grid::Layout >
                                     local_matrices_selective_;
    grid::Grid5DDataScalar< int >    indices_;
    linalg::OperatorStoredMatrixMode operator_stored_matrix_mode_;
    Kokkos::View< int >              nMatrices_;
    int                              capacity_;

  public:
    // default initialize storage without memory in GCAElements_ or local_matrices_
    // necessary as optional stuff is not available or complicated on GPU
    // (e.g. mem allocation for pointers, std::optional)
    LocalMatrixStorage() {}

    // ctor to be called in operators
    LocalMatrixStorage(
        // domain required for full storage size
        grid::shell::DistributedDomain domain,
        // mode required to determine which storage type to read/write/allocate
        linalg::OperatorStoredMatrixMode operator_stored_matrix_mode,

        // only required for selective storage: level range for mapping from marked coarse elements
        // to domain at hand
        int  level_range,
        // marked coarse elements
        grid::Grid4DDataScalar< ScalarType >  GCAElements )
    : operator_stored_matrix_mode_( operator_stored_matrix_mode )
    {
        if ( operator_stored_matrix_mode == linalg::OperatorStoredMatrixMode::Selective )
        {
            level_range_ = level_range;
            GCAElements_ = GCAElements;

            // compute required capacity of map/selective storage
            int nGCAElements = kernels::common::dot_product( GCAElements_, GCAElements_ );
            capacity_        = 2 * nGCAElements * Kokkos::pow( 2, ( 3 * level_range_ ) );
            std::cout << "Number of GCA coarse elements: " << nGCAElements << "/"
                      << GCAElements_.extent( 0 ) * ( GCAElements_.extent( 1 ) - 1 ) *
                             ( GCAElements_.extent( 2 ) - 1 ) * ( GCAElements_.extent( 2 ) - 1 )
                      << ", capacity: " << capacity_ << std::endl;
            local_matrices_selective_ =
                Kokkos::View< dense::Mat< ScalarType, LocalMatrixDim, LocalMatrixDim >*, terra::grid::Layout >(
                    "local_matrices_selective_", capacity_ );

            // initialize indexing field
            indices_ = terra::grid::Grid5DDataScalar< int >(
                "indices_",
                domain.subdomains().size(),
                domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain.domain_info().subdomain_num_nodes_radially() - 1,
                2 );

            kernels::common::set_constant( indices_, -1 );

            // init matrix counter
            nMatrices_ = Kokkos::View< int >( "nMatrices_" );
            Kokkos::deep_copy( nMatrices_, 0 );

            //local_matrices_selective_ = Kokkos::UnorderedMap< SelectiveStorageKey, LocalMatrixType >( 10 * capacity );
        }
        else if ( operator_stored_matrix_mode == linalg::OperatorStoredMatrixMode::Full )
        {
            local_matrices_full_ = terra::grid::Grid4DDataMatrices< ScalarType, LocalMatrixDim, LocalMatrixDim, 2 >(
                "local_matrices_full",
                domain.subdomains().size(),
                domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain.domain_info().subdomain_num_nodes_radially() - 1 );
        }
        else
        {
            Kokkos::abort( "LocalMatrixStorage() not implemented." );
        }
    }

    /// @brief Set the local matrix stored in the operator
    KOKKOS_INLINE_FUNCTION
    void set_matrix(
        const int                                             local_subdomain_id,
        const int                                             x_cell,
        const int                                             y_cell,
        const int                                             r_cell,
        const int                                             wedge,
        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > mat ) const
    {
        if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Full )
        {
            for ( int i = 0; i < LocalMatrixDim; ++i )
            {
                for ( int j = 0; j < LocalMatrixDim; ++j )
                {
                    local_matrices_full_( local_subdomain_id, x_cell, y_cell, r_cell, wedge )( i, j ) = mat( i, j );
                }
            }
        }
        else if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Selective )
        {
            // access map
            if ( indices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) == -1 )
            {
                // matrix at that spatial index not written:
                // assign current linear index + increment linear index atomically
                indices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) =
                    Kokkos::atomic_fetch_add( &nMatrices_(), 1 );
            }
            else
            {
                Kokkos::abort( "Trying to write matrix that is already present." );
            }
            if ( nMatrices_() >= capacity_ + 1 )
            {
                Kokkos::abort( "Too many matrices to store." );
            }

            for ( int i = 0; i < LocalMatrixDim; ++i )
            {
                for ( int j = 0; j < LocalMatrixDim; ++j )
                {
                    // write matrix to linearized position in local_matrices_selective_
                    local_matrices_selective_( indices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) )( i, j ) =
                        mat( i, j );
                }
            }
        }
        else
        {
            Kokkos::abort( "set_matrix() not implemented." );
        }
    }

    /// @brief Retrives the local matrix
    /// if there is stored local matrices, the desired local matrix is loaded and returned
    /// if not, the local matrix is assembled on-the-fly
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > get_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > ijslice;
        if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Full )
        {
            for ( int i = 0; i < LocalMatrixDim; ++i )
            {
                for ( int j = 0; j < LocalMatrixDim; ++j )
                {
                    ijslice( i, j ) = local_matrices_full_( local_subdomain_id, x_cell, y_cell, r_cell, wedge )( i, j );
                }
            }
        }
        else if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Selective )
        {
            // access map
            if ( indices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) == -1 )
            {
                Kokkos::abort( "Matrix not found." );
            }
            if ( indices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) >= capacity_ )
            {
                Kokkos::abort( "Too many matrices." );
            }

            for ( int i = 0; i < LocalMatrixDim; ++i )
            {
                for ( int j = 0; j < LocalMatrixDim; ++j )
                {
                    ijslice( i, j ) = local_matrices_selective_(
                        indices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) )( i, j );
                }
            }
        }
        else
        {
            Kokkos::abort( "get_matrix() not implemented." );
        }
        return ijslice;
    }

    /// @brief Checks for presence of a local matrix for a certain element
    KOKKOS_INLINE_FUNCTION
    bool has_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Full )
        {
            return true;
        }
        else if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Selective )
        {
            return indices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) != -1;
        }
        else
        {
            Kokkos::abort( "This should not happen." );
        }
    }
};
} // namespace terra::linalg::solvers