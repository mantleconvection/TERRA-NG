#pragma once
#include <mpi.h>

#include "communication/shell/communication.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "vector.hpp"
#include "vector_q1.hpp"

namespace terra::linalg {

/// @brief Block vector consisting of a Q1 vector and a Q1 scalar vector on distributed shell grids.
///
/// Same layout as required for tensor-product wedge elements.
///
/// Satisfies the Block2VectorLike concept (see vector.hpp).
/// Used for mixed finite element methods (e.g., Q1isoQ2-Q1).
template < typename ScalarT, int VecDim = 3 >
class VectorQ1IsoQ2Q1
{
  public:
    /// @brief Scalar type of the block vector.
    using ScalarType = ScalarT;

    /// @brief Type of the first block (vector field).
    using Block1Type = VectorQ1Vec< ScalarType, VecDim >;
    /// @brief Type of the second block (scalar field).
    using Block2Type = VectorQ1Scalar< ScalarType >;

    /// @brief Default constructor.
    VectorQ1IsoQ2Q1() = default;

    /// @brief Construct a block vector with labels, domains, and mask data for both blocks.
    /// @param label Name for the vector.
    /// @param distributed_domain_fine Distributed shell domain for the vector block.
    /// @param distributed_domain_coarse Distributed shell domain for the scalar block.
    /// @param mask_data_fine Mask data for the vector block.
    /// @param mask_data_coarse Mask data for the scalar block.
    VectorQ1IsoQ2Q1(
        const std::string&                                       label,
        const grid::shell::DistributedDomain&                    distributed_domain_fine,
        const grid::shell::DistributedDomain&                    distributed_domain_coarse,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_data_fine,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask_data_coarse )
    : u_( label + "_u", distributed_domain_fine, mask_data_fine )
    , p_( label + "_p", distributed_domain_coarse, mask_data_coarse )
    {}

    /// @brief Linear combination implementation for Block2VectorLike concept.
    /// Computes: \( \text{block}_1 = c_0 + \sum_i c_i x_i.\text{block}_1 \), \( \text{block}_2 = c_0 + \sum_i c_i x_i.\text{block}_2 \)
    /// @param c Coefficients.
    /// @param x Input block vectors.
    /// @param c0 Scalar to add.
    void
        lincomb_impl( const std::vector< ScalarType >& c, const std::vector< VectorQ1IsoQ2Q1 >& x, const ScalarType c0 )
    {
        std::vector< Block1Type > us;
        std::vector< Block2Type > ps;

        for ( const auto& xx : x )
        {
            us.emplace_back( xx.block_1() );
            ps.emplace_back( xx.block_2() );
        }

        u_.lincomb_impl( c, us, c0 );
        p_.lincomb_impl( c, ps, c0 );
    }

    /// @brief Dot product implementation for Block2VectorLike concept.
    /// Computes: \( \text{block}_1 \cdot x.\text{block}_1 + \text{block}_2 \cdot x.\text{block}_2 \)
    /// @param x Other block vector.
    /// @return Dot product value.
    ScalarType dot_impl( const VectorQ1IsoQ2Q1& x ) const
    {
        return x.block_1().dot_impl( u_ ) + x.block_2().dot_impl( p_ );
    }

    /// @brief Invert entries implementation for Block2VectorLike concept.
    /// Computes: \( \text{block}_1 = 1 / \text{block}_1 \), \( \text{block}_2 = 1 / \text{block}_2 \)
    void invert_entries_impl()
    {
        block_1().invert_entries_impl();
        block_2().invert_entries_impl();
    }

    /// @brief Elementwise scaling implementation for Block2VectorLike concept.
    /// Computes: \( \text{block}_1 = \text{block}_1 \cdot x.\text{block}_1 \), \( \text{block}_2 = \text{block}_2 \cdot x.\text{block}_2 \)
    /// @param x Scaling block vector.
    void scale_with_vector_impl( const VectorQ1IsoQ2Q1& x )
    {
        block_1().scale_with_vector_impl( x.block_1() );
        block_2().scale_with_vector_impl( x.block_2() );
    }

    /// @brief Randomize entries implementation for Block2VectorLike concept.
    /// Sets each entry of both blocks to a random value.
    void randomize_impl()
    {
        block_1().randomize_impl();
        block_2().randomize_impl();
    }

    /// @brief Max absolute entry implementation for Block2VectorLike concept.
    /// Computes: \( \max( \max_i |\text{block}_1|, \max_j |\text{block}_2| ) \)
    /// @return Maximum absolute value across both blocks.
    ScalarType max_abs_entry_impl() const
    {
        return std::max( block_1().max_abs_entry_impl(), block_2().max_abs_entry_impl() );
    }

    /// @brief NaN/inf check implementation for Block2VectorLike concept.
    /// Returns true if any entry of either block is NaN/inf.
    /// @return True if NaN/inf is present.
    bool has_nan_or_inf_impl() const { return block_1().has_nan_or_inf_impl() || block_2().has_nan_or_inf_impl(); }

    /// @brief Swap implementation for Block2VectorLike concept.
    /// Exchanges the contents of both blocks with another block vector.
    /// @param other Other block vector.
    void swap_impl( VectorQ1IsoQ2Q1& other )
    {
        u_.swap_impl( other.u_ );
        p_.swap_impl( other.p_ );
    }

    /// @brief Add mask data to both blocks.
    /// @param mask_data_block_1 Mask data for block 1.
    /// @param mask_data_block_2 Mask data for block 2.
    void add_mask_data(
        const grid::Grid4DDataScalar< unsigned char >& mask_data_block_1,
        const grid::Grid4DDataScalar< unsigned char >& mask_data_block_2 )
    {
        block_1().mask_data() = mask_data_block_1;
        block_2().mask_data() = mask_data_block_2;
    }

    /// @brief Get const reference to block 1 (vector field).
    const Block1Type& block_1() const { return u_; }
    /// @brief Get const reference to block 2 (scalar field).
    const Block2Type& block_2() const { return p_; }

    /// @brief Get mutable reference to block 1 (vector field).
    Block1Type& block_1() { return u_; }
    /// @brief Get mutable reference to block 2 (scalar field).
    Block2Type& block_2() { return p_; }

    /// @brief MPI communicator the blocks are distributed over (same on both).
    MPI_Comm comm() const { return u_.comm(); }

  private:
    Block1Type u_; ///< Vector block (velocity, etc.)
    Block2Type p_; ///< Scalar block (pressure, etc.)
};

/// @brief Static assertion: VectorQ1IsoQ2Q1 satisfies Block2VectorLike concept.
static_assert( Block2VectorLike< VectorQ1IsoQ2Q1< double > > );

/// @brief Allocate a VectorQ1IsoQ2Q1 block vector with grid data for both blocks.
/// @param label Name for the vector.
/// @param distributed_domain_fine Distributed shell domain for the vector block.
/// @param distributed_domain_coarse Distributed shell domain for the scalar block.
/// @param level Multigrid level.
/// @return Allocated VectorQ1IsoQ2Q1 block vector.
template < typename ValueType, int VecDim = 3 >
VectorQ1IsoQ2Q1< ValueType, VecDim > allocate_vector_q1isoq2_q1(
    const std::string                     label,
    const grid::shell::DistributedDomain& distributed_domain_fine,
    const grid::shell::DistributedDomain& distributed_domain_coarse,
    const int                             level )
{
    grid::Grid4DDataVec< ValueType, VecDim > grid_data_fine(
        label,
        distributed_domain_fine.subdomains().size(),
        distributed_domain_fine.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain_fine.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain_fine.domain_info().subdomain_num_nodes_radially() );

    grid::Grid4DDataScalar< ValueType > grid_data_coarse(
        label,
        distributed_domain_coarse.subdomains().size(),
        distributed_domain_coarse.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain_coarse.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain_coarse.domain_info().subdomain_num_nodes_radially() );

    VectorQ1IsoQ2Q1< ValueType, VecDim > vector_q1isoq2_q1;
    vector_q1isoq2_q1.block_1().add_grid_data( grid_data_fine, level );
    vector_q1isoq2_q1.block_2().add_grid_data( grid_data_coarse, level );
    return vector_q1isoq2_q1;
}

} // namespace terra::linalg