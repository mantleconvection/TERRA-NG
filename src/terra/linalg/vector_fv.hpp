#pragma once
#include <mpi.h>

#include "communication/shell/communication.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "terra/grid/bit_masks.hpp"
#include "terra/grid/shell/bit_masks.hpp"
#include "vector.hpp"

namespace terra::linalg {

/// @brief Finite volume vector on distributed shell grid with one DoF per hex (merging 2 wedges) and ghost-layer in all
/// directions.
///
/// Only non-ghost-layer cells can be expected to be up-to-date. Unlike nodal grids, communication has to be executed
/// first in general, before running kernels (if ghost-layer data is required).
///
/// Satisfies the VectorLike concept (see vector.hpp).
/// Provides operations for scalar fields.
template < typename ScalarT >
class VectorFVScalar
{
  public:
    /// @brief Scalar type of the vector.
    using ScalarType = ScalarT;

    /// @brief Default constructor.
    VectorFVScalar() = default;

    /// @brief Construct a scalar finite volume vector with label and domain.
    /// @param label Name for the vector.
    /// @param distributed_domain Distributed shell domain.
    VectorFVScalar( const std::string& label, const grid::shell::DistributedDomain& distributed_domain )
    : comm_( distributed_domain.comm() )
    {
        grid::Grid4DDataScalar< ScalarType > grid_data(
            label,
            distributed_domain.subdomains().size(),
            distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
            distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
            distributed_domain.domain_info().subdomain_num_nodes_radially() + 1 );

        grid_data_ = grid_data;
    }

    /// @brief Linear combination implementation for VectorLike concept.
    /// Computes: \f$ y = c_0 + \sum_i c_i x_i \f$
    /// @param c Coefficients.
    /// @param x Input vectors.
    /// @param c0 Scalar to add.
    void lincomb_impl( const std::vector< ScalarType >& c, const std::vector< VectorFVScalar >& x, const ScalarType c0 )
    {
        if ( c.size() != x.size() )
        {
            throw std::runtime_error( "VectorQ1Scalar::lincomb: c and x must have the same size" );
        }

        if ( x.size() == 0 )
        {
            kernels::common::set_constant( grid_data_, c0 );
        }
        else if ( x.size() == 1 )
        {
            kernels::common::lincomb( grid_data_, c0, c[0], x[0].grid_data() );
        }
        else if ( x.size() == 2 )
        {
            kernels::common::lincomb( grid_data_, c0, c[0], x[0].grid_data(), c[1], x[1].grid_data() );
        }
        else if ( x.size() == 3 )
        {
            kernels::common::lincomb(
                grid_data_, c0, c[0], x[0].grid_data(), c[1], x[1].grid_data(), c[2], x[2].grid_data() );
        }
        else
        {
            throw std::runtime_error( "VectorFVScalar::lincomb: not implemented" );
        }
    }

    /// @brief Dot product implementation for VectorLike concept.
    /// Computes: \f$ \sum_{i} y_i \cdot x_i \f$ over owned nodes.
    /// @param x Other vector.
    /// @return Dot product value.
    ScalarType dot_impl( const VectorFVScalar& x ) const
    {
        return kernels::common::dot_product_subset(
            grid_data_,
            x.grid_data(),
            dense::Vec< int, 4 >{ 0, 1, 1, 1 },
            dense::Vec< int, 4 >{
                static_cast< int >( grid_data_.extent( 0 ) ),
                static_cast< int >( grid_data_.extent( 1 ) - 1 ),
                static_cast< int >( grid_data_.extent( 2 ) - 1 ),
                static_cast< int >( grid_data_.extent( 3 ) - 1 ) },
            comm_ );
    }

    /// @brief Invert entries implementation for VectorLike concept.
    /// Computes: \f$ y_i = 1 / y_i \f$
    void invert_entries_impl() { kernels::common::invert_inplace( grid_data_ ); }

    /// @brief Elementwise scaling implementation for VectorLike concept.
    /// Computes: \f$ y_i = y_i \cdot x_i \f$
    /// @param x Scaling vector.
    void scale_with_vector_impl( const VectorFVScalar& x )
    { kernels::common::mult_elementwise_inplace( grid_data_, x.grid_data() ); }

    /// @brief Randomize entries implementation for VectorLike concept.
    /// Sets each entry of grid_data to a random value.
    void randomize_impl() { return kernels::common::rand( grid_data_ ); }

    /// @brief Max absolute entry implementation for VectorLike concept.
    /// Computes: \f$ \max_i |y_i| \f$
    /// @return Maximum absolute value.
    ScalarType max_abs_entry_impl() const
    {
        return kernels::common::max_abs_entry_subset(
            grid_data_,
            dense::Vec< int, 4 >{ 0, 1, 1, 1 },
            dense::Vec< int, 4 >{
                grid_data_.extent( 0 ),
                grid_data_.extent( 1 ) - 1,
                grid_data_.extent( 2 ) - 1,
                grid_data_.extent( 3 ) - 1 },
            comm_ );
    }

    /// @brief NaN/Inf check implementation for VectorLike concept.
    /// Returns true if any entry of grid_data is NaN or inf.
    /// @return True if NaN or inf is present.
    bool has_nan_or_inf_impl() const { return kernels::common::has_nan_or_inf( grid_data_, comm_ ); }

    /// @brief Swap implementation for VectorLike concept.
    /// Exchanges grid_data and mask_data with another vector.
    /// @param other Other vector.
    void swap_impl( VectorFVScalar& other )
    {
        std::swap( grid_data_, other.grid_data_ );
        std::swap( comm_, other.comm_ );
    }

    /// @brief Get const reference to grid data.
    const grid::Grid4DDataScalar< ScalarType >& grid_data() const { return grid_data_; }
    /// @brief Get mutable reference to grid data.
    grid::Grid4DDataScalar< ScalarType >& grid_data() { return grid_data_; }

    /// @brief MPI communicator this vector's reductions run on.
    MPI_Comm comm() const { return comm_; }

  private:
    grid::Grid4DDataScalar< ScalarType > grid_data_;
    MPI_Comm                             comm_ = MPI_COMM_WORLD;
};

/// @brief Static assertion: VectorQ1Scalar satisfies VectorLike concept.
static_assert( VectorLike< VectorFVScalar< double > > );

/// @brief Finite volume vector on distributed shell grid with one DoF per hex (merging 2 wedges) and ghost-layer in all
/// directions.
///
/// Only non-ghost-layer cells can be expected to be up-to-date. Unlike nodal grids, communication has to be executed
/// first in general, before running kernels (if ghost-layer data is required).
///
/// Satisfies the VectorLike concept (see vector.hpp).
/// Provides operations for vector fields.
template < typename ScalarT, int VecDim = 3 >
class VectorFVVec
{
  public:
    /// @brief Scalar type of the vector.
    using ScalarType = ScalarT;

    /// @brief Default constructor.
    VectorFVVec() = default;

    /// @brief Construct a scalar finite volume vector with label and domain.
    /// @param label Name for the vector.
    /// @param distributed_domain Distributed shell domain.
    VectorFVVec( const std::string& label, const grid::shell::DistributedDomain& distributed_domain )
    : comm_( distributed_domain.comm() )
    {
        grid::Grid4DDataVec< ScalarType, VecDim > grid_data(
            label,
            distributed_domain.subdomains().size(),
            distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
            distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
            distributed_domain.domain_info().subdomain_num_nodes_radially() + 1 );

        grid_data_ = grid_data;
    }

    /// @brief Linear combination implementation for VectorLike concept.
    /// Computes: \f$ y = c_0 + \sum_i c_i x_i \f$
    /// @param c Coefficients.
    /// @param x Input vectors.
    /// @param c0 Scalar to add.
    void lincomb_impl( const std::vector< ScalarType >& c, const std::vector< VectorFVVec >& x, const ScalarType c0 )
    {
        if ( c.size() != x.size() )
        {
            throw std::runtime_error( "VectorQ1Scalar::lincomb: c and x must have the same size" );
        }

        if ( x.size() == 0 )
        {
            kernels::common::set_constant( grid_data_, c0 );
        }
        else if ( x.size() == 1 )
        {
            kernels::common::lincomb( grid_data_, c0, c[0], x[0].grid_data() );
        }
        else if ( x.size() == 2 )
        {
            kernels::common::lincomb( grid_data_, c0, c[0], x[0].grid_data(), c[1], x[1].grid_data() );
        }
        else if ( x.size() == 3 )
        {
            kernels::common::lincomb(
                grid_data_, c0, c[0], x[0].grid_data(), c[1], x[1].grid_data(), c[2], x[2].grid_data() );
        }
        else
        {
            throw std::runtime_error( "VectorFVScalar::lincomb: not implemented" );
        }
    }

    /// @brief Dot product implementation for VectorLike concept.
    /// Computes: \f$ \sum_{i} y_i \cdot x_i \f$ over owned nodes.
    /// @param x Other vector.
    /// @return Dot product value.
    ScalarType dot_impl( const VectorFVVec& x ) const
    {
        return kernels::common::dot_product_subset(
            grid_data_,
            x.grid_data(),
            dense::Vec< int, 5 >{ 0, 1, 1, 1, 0 },
            dense::Vec< int, 5 >{
                static_cast< int >( grid_data_.extent( 0 ) ),
                static_cast< int >( grid_data_.extent( 1 ) - 1 ),
                static_cast< int >( grid_data_.extent( 2 ) - 1 ),
                static_cast< int >( grid_data_.extent( 3 ) - 1 ),
                VecDim } );
    }

    /// @brief Invert entries implementation for VectorLike concept.
    /// Computes: \f$ y_i = 1 / y_i \f$
    void invert_entries_impl() { kernels::common::invert_inplace( grid_data_ ); }

    /// @brief Elementwise scaling implementation for VectorLike concept.
    /// Computes: \f$ y_i = y_i \cdot x_i \f$
    /// @param x Scaling vector.
    void scale_with_vector_impl( const VectorFVVec& x )
    { kernels::common::mult_elementwise_inplace( grid_data_, x.grid_data() ); }

    /// @brief Randomize entries implementation for VectorLike concept.
    /// Sets each entry of grid_data to a random value.
    void randomize_impl() { return kernels::common::rand( grid_data_ ); }

    /// @brief Max absolute entry implementation for VectorLike concept.
    /// Computes: \f$ \max_i |y_i| \f$
    /// @return Maximum absolute value.
    ScalarType max_abs_entry_impl() const
    {
        return kernels::common::max_abs_entry_subset(
            grid_data_,
            dense::Vec< int, 5 >{ 0, 1, 1, 1, 0 },
            dense::Vec< int, 5 >{
                grid_data_.extent( 0 ),
                grid_data_.extent( 1 ) - 1,
                grid_data_.extent( 2 ) - 1,
                grid_data_.extent( 3 ) - 1,
                VecDim } );
    }

    /// @brief NaN/Inf check implementation for VectorLike concept.
    /// Returns true if any entry of grid_data is NaN or inf.
    /// @return True if NaN or inf is present.
    bool has_nan_or_inf_impl() const { return kernels::common::has_nan_or_inf( grid_data_, comm_ ); }

    /// @brief Swap implementation for VectorLike concept.
    /// Exchanges grid_data and mask_data with another vector.
    /// @param other Other vector.
    void swap_impl( VectorFVVec& other )
    {
        std::swap( grid_data_, other.grid_data_ );
        std::swap( comm_, other.comm_ );
    }

    /// @brief Get const reference to grid data.
    const grid::Grid4DDataVec< ScalarType, VecDim >& grid_data() const { return grid_data_; }
    /// @brief Get mutable reference to grid data.
    grid::Grid4DDataVec< ScalarType, VecDim >& grid_data() { return grid_data_; }

    /// @brief MPI communicator this vector's reductions run on.
    MPI_Comm comm() const { return comm_; }

  private:
    grid::Grid4DDataVec< ScalarType, VecDim > grid_data_;
    MPI_Comm                                  comm_ = MPI_COMM_WORLD;
};

/// @brief Static assertion: VectorQ1Scalar satisfies VectorLike concept.
static_assert( VectorLike< VectorFVVec< double > > );

} // namespace terra::linalg