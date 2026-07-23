#pragma once
#include <mpi.h>

#include "communication/shell/communication.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "terra/grid/bit_masks.hpp"
#include "terra/grid/shell/bit_masks.hpp"
#include "vector.hpp"

namespace terra::linalg {

/// @brief Q1 scalar finite element vector on a distributed shell grid.
///
/// Same layout as required for tensor-product wedge elements.
///
/// Satisfies the VectorLike concept (see vector.hpp).
/// Provides masked grid data and operations for scalar fields.
template < typename ScalarT >
class VectorQ1Scalar
{
  public:
    /// @brief Scalar type of the vector.
    using ScalarType = ScalarT;

    /// @brief Default constructor.
    VectorQ1Scalar() = default;

    /// @brief Construct a Q1 scalar vector with label, domain, and mask data.
    /// @param label Name for the vector.
    /// @param distributed_domain Distributed shell domain.
    /// @param mask_data Mask data grid.
    VectorQ1Scalar(
        const std::string&                                       label,
        const grid::shell::DistributedDomain&                    distributed_domain,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_data )
    : mask_data_( mask_data )
    , comm_( distributed_domain.comm() )
    {
        grid::Grid4DDataScalar< ScalarType > grid_data(
            label,
            distributed_domain.subdomains().size(),
            distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
            distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
            distributed_domain.domain_info().subdomain_num_nodes_radially() );

        grid_data_ = grid_data;

        if ( mask_data_.extent( 0 ) != grid_data_.extent( 0 ) || mask_data_.extent( 1 ) != grid_data_.extent( 1 ) ||
             mask_data_.extent( 2 ) != grid_data_.extent( 2 ) || mask_data_.extent( 3 ) != grid_data_.extent( 3 ) )
        {
            throw std::runtime_error(
                "VectorQ1Scalar::VectorQ1Scalar: mask_data and grid_data must have the same size" );
        }
    }

    /// @brief Linear combination implementation for VectorLike concept.
    /// Computes: \f$ y = c_0 + \sum_i c_i x_i \f$
    /// @param c Coefficients.
    /// @param x Input vectors.
    /// @param c0 Scalar to add.
    void lincomb_impl( const std::vector< ScalarType >& c, const std::vector< VectorQ1Scalar >& x, const ScalarType c0 )
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
            throw std::runtime_error( "VectorQ1Scalar::lincomb: not implemented" );
        }
    }

    /// @brief Dot product implementation for VectorLike concept.
    /// Computes: \f$ \sum_{i} y_i \cdot x_i \f$ over owned nodes.
    /// @param x Other vector.
    /// @return Dot product value.
    ScalarType dot_impl( const VectorQ1Scalar& x ) const
    {
        return kernels::common::masked_dot_product(
            grid_data_, x.grid_data(), mask_data(), grid::NodeOwnershipFlag::OWNED, comm_ );
    }

    /// @brief Invert entries implementation for VectorLike concept.
    /// Computes: \f$ y_i = 1 / y_i \f$
    void invert_entries_impl() { kernels::common::invert_inplace( grid_data_ ); }

    /// @brief Elementwise scaling implementation for VectorLike concept.
    /// Computes: \f$ y_i = y_i \cdot x_i \f$
    /// @param x Scaling vector.
    void scale_with_vector_impl( const VectorQ1Scalar& x )
    {
        kernels::common::mult_elementwise_inplace( grid_data_, x.grid_data() );
    }

    /// @brief Randomize entries implementation for VectorLike concept.
    /// Sets each entry of grid_data to a random value.
    void randomize_impl() { return kernels::common::rand( grid_data_ ); }

    /// @brief Min entry implementation for VectorLike concept.
    /// Computes: \f$ \min_i y_i \f$
    /// @return Minimum value.
    ScalarType min_entry_impl() const { return kernels::common::min_entry( grid_data_ ); }

    /// @brief Min absolute entry implementation for VectorLike concept.
    /// Computes: \f$ \min_i |y_i| \f$
    /// @return Minimum absolute value.
    ScalarType min_abs_entry_impl() const { return kernels::common::min_abs_entry( grid_data_ ); }

    /// @brief Max entry implementation for VectorLike concept.
    /// Computes: \f$ \max_i y_i \f$
    /// @return Maximum value.
    ScalarType max_entry_impl() const { return kernels::common::max_entry( grid_data_ ); }

    /// @brief Max absolute entry implementation for VectorLike concept.
    /// Computes: \f$ \max_i |y_i| \f$
    /// @return Maximum absolute value.
    ScalarType max_abs_entry_impl() const { return kernels::common::max_abs_entry( grid_data_, comm_ ); }

    /// @brief NaN/inf check implementation for VectorLike concept.
    /// Returns true if any entry of grid_data is NaN/inf.
    /// @return True if NaN/inf is present.
    bool has_nan_or_inf_impl() const { return kernels::common::has_nan_or_inf( grid_data_, comm_ ); }

    /// @brief Swap implementation for VectorLike concept.
    /// Exchanges grid_data and mask_data with another vector.
    /// @param other Other vector.
    void swap_impl( VectorQ1Scalar& other )
    {
        std::swap( grid_data_, other.grid_data_ );
        std::swap( mask_data_, other.mask_data_ );
        std::swap( comm_, other.comm_ );
    }

    /// @brief Get const reference to grid data.
    const grid::Grid4DDataScalar< ScalarType >& grid_data() const { return grid_data_; }
    /// @brief Get mutable reference to grid data.
    grid::Grid4DDataScalar< ScalarType >& grid_data() { return grid_data_; }

    /// @brief Get const reference to mask data.
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_data() const { return mask_data_; }
    /// @brief Get mutable reference to mask data.
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_data() { return mask_data_; }

    /// @brief MPI communicator this vector's reductions run on.
    MPI_Comm comm() const { return comm_; }

  private:
    grid::Grid4DDataScalar< ScalarType >              grid_data_;
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > mask_data_;
    MPI_Comm                                          comm_ = MPI_COMM_WORLD;
};

/// @brief Static assertion: VectorQ1Scalar satisfies VectorLike concept.
static_assert( VectorLike< VectorQ1Scalar< double > > );

/// @brief Q1 vector finite element vector on a distributed shell grid.
///
/// Same layout as required for tensor-product wedge elements.
///
/// Satisfies the VectorLike concept (see vector.hpp).
/// Provides masked grid data and operations for vector fields.
template < typename ScalarT, int VecDim = 3 >
class VectorQ1Vec
{
  public:
    /// @brief Default constructor.
    VectorQ1Vec() = default;

    /// @brief Construct a Q1 vector with label, domain, and mask data.
    /// @param label Name for the vector.
    /// @param distributed_domain Distributed shell domain.
    /// @param mask_data Mask data grid.
    VectorQ1Vec(
        const std::string&                                       label,
        const grid::shell::DistributedDomain&                    distributed_domain,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_data )
    : mask_data_( mask_data )
    , comm_( distributed_domain.comm() )
    {
        grid::Grid4DDataVec< ScalarType, VecDim > grid_data(
            label,
            distributed_domain.subdomains().size(),
            distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
            distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
            distributed_domain.domain_info().subdomain_num_nodes_radially() );

        grid_data_ = grid_data;

        if ( mask_data_.extent( 0 ) != grid_data_.extent( 0 ) || mask_data_.extent( 1 ) != grid_data_.extent( 1 ) ||
             mask_data_.extent( 2 ) != grid_data_.extent( 2 ) || mask_data_.extent( 3 ) != grid_data_.extent( 3 ) )
        {
            throw std::runtime_error(
                "VectorQ1Scalar::VectorQ1Scalar: mask_data and grid_data must have the same size" );
        }
    }

    /// @brief Scalar type of the vector.
    using ScalarType = ScalarT;
    /// @brief Dimension of the vector field.
    const static int Dim = VecDim;

    /// @brief Linear combination implementation for VectorLike concept.
    /// Computes: \f$ y = c_0 + \sum_i c_i x_i \f$
    /// @param c Coefficients.
    /// @param x Input vectors.
    /// @param c0 Scalar to add.
    void lincomb_impl( const std::vector< ScalarType >& c, const std::vector< VectorQ1Vec >& x, const ScalarType c0 )
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
            throw std::runtime_error( "VectorQ1Scalar::lincomb: not implemented" );
        }
    }

    /// @brief Dot product implementation for VectorLike concept.
    /// Computes: \f$ \sum_{i} y_i \cdot x_i \f$ over owned nodes.
    /// @param x Other vector.
    /// @return Dot product value.
    ScalarType dot_impl( const VectorQ1Vec& x ) const
    {
        return kernels::common::masked_dot_product(
            grid_data_, x.grid_data(), mask_data_, grid::NodeOwnershipFlag::OWNED, comm_ );
    }

    /// @brief Invert entries implementation for VectorLike concept.
    /// Computes: \f$ y_i = 1 / y_i \f$
    void invert_entries_impl() { kernels::common::invert_inplace( grid_data_ ); }

    /// @brief Elementwise scaling implementation for VectorLike concept.
    /// Computes: \f$ y_i = y_i \cdot x_i \f$
    /// @param x Scaling vector.
    void scale_with_vector_impl( const VectorQ1Vec& x )
    {
        kernels::common::mult_elementwise_inplace( grid_data_, x.grid_data() );
    }

    /// @brief Randomize entries implementation for VectorLike concept.
    /// Sets each entry of grid_data to a random value.
    void randomize_impl() { return kernels::common::rand( grid_data_ ); }

    /// @brief Min entry implementation for VectorLike concept.
    /// Computes: \f$ \min_i y_i \f$
    /// @return Minimum value.
    ScalarType min_entry_impl() const { return kernels::common::min_entry( grid_data_ ); }

    /// @brief Min absolute entry implementation for VectorLike concept.
    /// Computes: \f$ \min_i |y_i| \f$
    /// @return Minimum absolute value.
    ScalarType min_abs_entry_impl() const { return kernels::common::min_abs_entry( grid_data_ ); }

    /// @brief Max entry implementation for VectorLike concept.
    /// Computes: \f$ \max_i y_i \f$
    /// @return Maximum value.
    ScalarType max_entry_impl() const { return kernels::common::max_entry( grid_data_ ); }

    /// @brief Max absolute entry implementation for VectorLike concept.
    /// Computes: \f$ \max_i |y_i| \f$
    /// @return Maximum absolute value.
    ScalarType max_abs_entry_impl() const { return kernels::common::max_abs_entry( grid_data_, comm_ ); }

    /// @brief NaN/inf check implementation for VectorLike concept.
    /// Returns true if any entry of grid_data is NaN/inf.
    /// @return True if NaN/inf is present.
    bool has_nan_or_inf_impl() const { return kernels::common::has_nan_or_inf( grid_data_, comm_ ); }

    /// @brief Swap implementation for VectorLike concept.
    /// Exchanges grid_data and mask_data with another vector.
    /// @param other Other vector.
    void swap_impl( VectorQ1Vec& other )
    {
        std::swap( grid_data_, other.grid_data_ );
        std::swap( mask_data_, other.mask_data_ );
        std::swap( comm_, other.comm_ );
    }

    /// @brief Get const reference to grid data.
    const grid::Grid4DDataVec< ScalarType, VecDim >& grid_data() const { return grid_data_; }
    /// @brief Get mutable reference to grid data.
    grid::Grid4DDataVec< ScalarType, VecDim >& grid_data() { return grid_data_; }

    /// @brief Get const reference to mask data.
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_data() const { return mask_data_; }
    /// @brief Get mutable reference to mask data.
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_data() { return mask_data_; }

    /// @brief MPI communicator this vector's reductions run on.
    MPI_Comm comm() const { return comm_; }

  private:
    grid::Grid4DDataVec< ScalarType, VecDim >         grid_data_;
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > mask_data_;
    MPI_Comm                                          comm_ = MPI_COMM_WORLD;
};

/// @brief Static assertion: VectorQ1Vec satisfies VectorLike concept.
static_assert( VectorLike< VectorQ1Vec< double, 3 > > );

} // namespace terra::linalg
