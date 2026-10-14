#pragma once
#include "linalg.hpp"
#include "vector.hpp"

namespace terra::linalg {

/// @brief Modes for applying an operator to a vector.
///
/// @note It is important to pair the modes correctly.
///       **If unsure, use temporary vectors, apply each operator individually (typically defaulting to modes 'Replace'
///       and 'CommunicateAdditively'), and add the results in the end.**
///
/// Elementwise operators have to communicate after all local values have been updated.
/// The general approach is to communicate and sum up values of overlapping nodes.
/// If the 'ApplyMode' is set to 'Add' and additive communication follows that, then also previous values will
/// be included in the sum.
///
/// The modes are, for instance, useful for block matrices like the Stokes operator.
/// If you want to execute, e.g., \f$ y \gets Au + B^T p \f$ then you could execute this as follows:
/*!
    \f[
        \begin{aligned}
        y &\gets Au && \qquad \text{replace and skip communication} \\
        y &\gets y + B^Tp && \qquad \text{add and communicate additively} \\
        \end{aligned}
    \f]
*/
/// Note that if communication is not skipped in the first step in this example, the subdomain boundary data will be
/// wrong.
///
enum class OperatorApplyMode
{
    Replace, ///< Overwrite the destination vector.
    Add,     ///< Add to the destination vector.
};

/// @brief Modes for communication during operator application.
///
/// @note See @ref OperatorApplyMode for more details.
///
enum class OperatorCommunicationMode
{
    SkipCommunication,     ///< Do not communicate.
    CommunicateAdditively, ///< Communicate and add results.
};

/// @brief Modes for applying stored matrices.
enum class OperatorStoredMatrixMode
{
    Off,       ///< Do not use stored matrices.
    Full,      ///< Use stored matrices on all elements.
    Selective, ///< Use stored matrices on selected, marked elements only, assemble on all others.
};

/// @brief Concept for types that behave like linear operators.
///
/// Requires vector types, matvec implementation, and compatibility with VectorLike.
template < typename T >
concept OperatorLike = requires(
    const T&                         self_const,
    T&                               self,
    const typename T::SrcVectorType& src,
    typename T::DstVectorType&       dst,
    dense::Vec< int, 2 >             block ) {
    // Requires exposing the vector types.
    typename T::SrcVectorType;
    typename T::DstVectorType;

    // Require that Src and Dst vector types satisfy VectorLike
    requires VectorLike< typename T::SrcVectorType >;
    requires VectorLike< typename T::DstVectorType >;

    // Required matvec implementation
    // TODO: T& self is not const because having apply_impl const is not convenient - mostly because
    //       it is handy to reuse send/recv buffers that are members of an operator implementation.
    //       To modify these members (i.e., to communicate) we cannot be const :(
    { self.apply_impl( src, dst ) } -> std::same_as< void >;
};

/// @brief Concept for types that can be used as Galerkin coarse-grid operators in a multigrid hierarchy.
/// See galerkin_coarsening_linear.hpp for more information on GCA.
template < typename Op >
concept GCACapable = requires(
    // dummy variables to define requirements
    const Op& self_const,
    const int local_subdomain_id,
    const int x_cell,
    const int y_cell,
    const int r_cell,
    // 0 or 1 wedge within the hex cell on which GCA operates
    const int                                                                            wedge,
    const dense::Mat< typename Op::ScalarType, Op::LocalMatrixDim, Op::LocalMatrixDim >& mat ) {
    // Require that the argument to be a linear operator
    requires OperatorLike< Op >;

    // The operator must show the dimensions of the local matrix (might be 18 for vectorial diffusion ops in 3D like EpsilonDivDiv,
    //  or 6 for a simple scalar div-k-grad)
    { Op::LocalMatrixDim } -> std::convertible_to< int >;

    // The operator must allow read/write access to his local matrices in order to GCA them.
    // It is indeed weird that the write access set_local_matrix() shall be const.
    // However, the View access implementation of Kokkos and the requirement of kernels being const
    // make this possible and necessary.
    {
        self_const.get_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge )
    } -> std::same_as< dense::Mat< typename Op::ScalarType, Op::LocalMatrixDim, Op::LocalMatrixDim > >;

    { self_const.set_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge, mat ) } -> std::same_as< void >;

    // Since TwoGridGCA works with multiple operators, and their respective domains (coarse-fine nested loop),
    // it is required to have access to geometric information stored in the operators.
    { self_const.get_domain() } -> std::same_as< const grid::shell::DistributedDomain& >;
    { self_const.get_radii() } -> std::same_as< grid::Grid2DDataScalar< typename Op::ScalarType > >;
};

/// @brief Concept for types that behave like block 2x2 operators.
/// Extends OperatorLike and requires block types and accessors.
template < typename T >
concept Block2x2OperatorLike = OperatorLike< T > && requires( const T& self_const, T& self ) {
    // Require that Src and Dst vector types satisfy VectorLike
    requires Block2VectorLike< typename T::SrcVectorType >;
    requires Block2VectorLike< typename T::DstVectorType >;

    typename T::Block11Type;
    typename T::Block12Type;
    typename T::Block21Type;
    typename T::Block22Type;

    requires OperatorLike< typename T::Block11Type >;
    requires OperatorLike< typename T::Block12Type >;
    requires OperatorLike< typename T::Block21Type >;
    requires OperatorLike< typename T::Block22Type >;

    { self_const.block_11() } -> std::same_as< const typename T::Block11Type& >;
    { self_const.block_12() } -> std::same_as< const typename T::Block12Type& >;
    { self_const.block_21() } -> std::same_as< const typename T::Block21Type& >;
    { self_const.block_22() } -> std::same_as< const typename T::Block22Type& >;

    { self.block_11() } -> std::same_as< typename T::Block11Type& >;
    { self.block_12() } -> std::same_as< typename T::Block12Type& >;
    { self.block_21() } -> std::same_as< typename T::Block21Type& >;
    { self.block_22() } -> std::same_as< typename T::Block22Type& >;
};

/// @brief Alias for the source vector type of an operator.
template < OperatorLike Operator >
using SrcOf = Operator::SrcVectorType;

/// @brief Alias for the destination vector type of an operator.
template < OperatorLike Operator >
using DstOf = Operator::DstVectorType;

/// @brief Apply an operator to a source vector and write to a destination vector.
/// @param A Operator to apply.
/// @param src Source vector.
/// @param dst Destination vector.
template < OperatorLike Operator >
void apply( Operator& A, const SrcOf< Operator >& src, DstOf< Operator >& dst )
{
    A.apply_impl( src, dst );
}

namespace detail {

/// @brief Dummy operator for testing concepts.
/// Implements apply_impl as a no-op.
template < VectorLike SrcVectorT, VectorLike DstVectorT >
class DummyOperator
{
  public:
    using SrcVectorType = SrcVectorT;
    using DstVectorType = DstVectorT;

    /// @brief Dummy apply_impl, does nothing.
    /// @param src Source vector.
    /// @param dst Destination vector.
    void apply_impl( const SrcVectorType& src, DstVectorType& dst ) const
    {
        (void) src;
        (void) dst;
    }
};

/// @brief Dummy concrete operator for concept checks.
/// Uses DummyVector as vector types.
class DummyConcreteOperator
{
  public:
    using SrcVectorType = DummyVector< double >;
    using DstVectorType = DummyVector< double >;

    /// @brief Dummy apply_impl, does nothing.
    /// @param src Source vector.
    /// @param dst Destination vector.
    void apply_impl( const SrcVectorType& src, DstVectorType& dst ) const
    {
        (void) src;
        (void) dst;
    }
};

/// @brief Dummy block 2x2 operator for concept checks.
/// Contains four DummyConcreteOperator blocks.
class DummyConcreteBlock2x2Operator
{
  public:
    using SrcVectorType = DummyBlock2Vector< double >;
    using DstVectorType = DummyBlock2Vector< double >;

    using Block11Type = DummyConcreteOperator;
    using Block12Type = DummyConcreteOperator;
    using Block21Type = DummyConcreteOperator;
    using Block22Type = DummyConcreteOperator;

    /// @brief Dummy apply_impl, does nothing.
    /// @param src Source block vector.
    /// @param dst Destination block vector.
    void apply_impl( const SrcVectorType& src, DstVectorType& dst ) const
    {
        (void) src;
        (void) dst;
    }

    /// @brief Get const reference to block 11.
    const Block11Type& block_11() const { return block_11_; }
    /// @brief Get const reference to block 12.
    const Block12Type& block_12() const { return block_12_; }
    /// @brief Get const reference to block 21.
    const Block21Type& block_21() const { return block_21_; }
    /// @brief Get const reference to block 22.
    const Block22Type& block_22() const { return block_22_; }

    /// @brief Get mutable reference to block 11.
    Block11Type& block_11() { return block_11_; }
    /// @brief Get mutable reference to block 12.
    Block12Type& block_12() { return block_12_; }
    /// @brief Get mutable reference to block 21.
    Block21Type& block_21() { return block_21_; }
    /// @brief Get mutable reference to block 22.
    Block22Type& block_22() { return block_22_; }

  private:
    Block11Type block_11_;
    Block12Type block_12_;
    Block21Type block_21_;
    Block22Type block_22_;
};

/// @brief Static assertions to check concepts for dummy operators.
static_assert( OperatorLike< DummyOperator< DummyVector< double >, DummyVector< double > > > );
static_assert( OperatorLike< DummyConcreteOperator > );
static_assert( Block2x2OperatorLike< DummyConcreteBlock2x2Operator > );

} // namespace detail

} // namespace terra::linalg
