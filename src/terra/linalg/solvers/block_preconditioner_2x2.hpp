#pragma once

#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::linalg::solvers {

/// @brief Block-diagonal preconditioner for 2x2 block operators.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Applies separate preconditioners to the (1,1) and (2,2) blocks.
/// The block-diagonal preconditioner solves:
/*!
    \f[
    \begin{pmatrix}
    P_{11} & 0 \\
    0 & P_{22}
    \end{pmatrix}
    \begin{pmatrix}
    x_1 \\ x_2
    \end{pmatrix}
    =
    \begin{pmatrix}
    b_1 \\ b_2
    \end{pmatrix}
    \f]
*/
/// where \f$ P_{11} \f$ and \f$ P_{22} \f$ are preconditioners for the (1,1) and (2,2) blocks, respectively.
/// @tparam OperatorT Operator type (must satisfy Block2x2OperatorLike).
/// @tparam Block11T Type of the (1,1) operator of the preconditioner to be (approximately) inverted
/// @tparam Block22T Type of the (2,2) operator of the preconditioner to be (approximately) inverted
/// @tparam Block11Preconditioner Preconditioner for the (1,1) block (must satisfy SolverLike).
/// @tparam Block22Preconditioner Preconditioner for the (2,2) block (must satisfy SolverLike).
template <
    Block2x2OperatorLike OperatorT,
    OperatorLike         Block11T,
    OperatorLike         Block22T,
    SolverLike           Block11Preconditioner,
    SolverLike           Block22Preconditioner >
class BlockDiagonalPreconditioner2x2
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;
    /// @brief Solution vector type (must be Block2VectorLike).
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type (must be Block2VectorLike).
    using RHSVectorType = DstOf< OperatorType >;

    /// @brief Static assertions to ensure block vector types.
    static_assert(
        Block2VectorLike< SolutionVectorType >,
        "The solution vector of the BlockPreconditioner2x2 must be Block2VectorLike." );
    static_assert(
        Block2VectorLike< RHSVectorType >,
        "The RHS vector of the BlockPreconditioner2x2 must be Block2VectorLike." );

    static_assert( std::is_same_v< SrcOf< Block11T >, typename SrcOf< OperatorT >::Block1Type > );
    static_assert( std::is_same_v< SrcOf< Block22T >, typename SrcOf< OperatorT >::Block2Type > );

    static_assert( std::is_same_v< DstOf< Block11T >, typename DstOf< OperatorT >::Block1Type > );
    static_assert( std::is_same_v< DstOf< Block22T >, typename DstOf< OperatorT >::Block2Type > );

    /// @brief Construct a block-diagonal preconditioner with given block preconditioners.
    ///
    /// When calling solve( A, x, b ) with this preconditioner, the two passed solvers are applied to the two block
    /// passed in the constructor here. This does NOT use the (1, 1) and (2, 2) blocks of the A block in the solve()
    /// call.
    ///
    /// @param block11 The (1, 1) block to approximate the inverse to.
    /// @param block22 The (2, 2) block to approximate the inverse to.
    /// @param block11_preconditioner Preconditioner for the (1,1) block.
    /// @param block22_preconditioner Preconditioner for the (2,2) block.
    BlockDiagonalPreconditioner2x2(
        const Block11T&              block11,
        const Block22T&              block22,
        const Block11Preconditioner& block11_preconditioner,
        const Block22Preconditioner& block22_preconditioner )
    : block11_( block11 )
    , block22_( block22 )
    , block11_preconditioner_( block11_preconditioner )
    , block22_preconditioner_( block22_preconditioner )
    {}

    /// @brief Solve the block-diagonal preconditioner system.
    ///
    /// Applies the block11 and block22 preconditioners to the respective blocks.
    ///
    /// @param A Is ignored. The two solvers are applied to the operators passed in the constructor.
    /// @param x Solution block vector (output).
    /// @param b Right-hand side block vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        solve( block11_preconditioner_, block11_, x.block_1(), b.block_1() );
        solve( block22_preconditioner_, block22_, x.block_2(), b.block_2() );
    }

  private:
    Block11T block11_; ///< (1,1) block of operator.
    Block22T block22_; ///< (2,2) block of operator.

    Block11Preconditioner block11_preconditioner_; ///< Preconditioner for (1,1) block.
    Block22Preconditioner block22_preconditioner_; ///< Preconditioner for (2,2) block.
};

/// @brief Static assertion: BlockDiagonalPreconditioner2x2 satisfies SolverLike concept.
static_assert( SolverLike< BlockDiagonalPreconditioner2x2<
                   linalg::detail::DummyConcreteBlock2x2Operator,
                   linalg::detail::DummyConcreteBlock2x2Operator::Block11Type,
                   linalg::detail::DummyConcreteBlock2x2Operator::Block22Type,
                   detail::DummySolver< linalg::detail::DummyConcreteBlock2x2Operator::Block11Type >,
                   detail::DummySolver< linalg::detail::DummyConcreteBlock2x2Operator::Block22Type > > > );

/// @brief Block-triangular preconditioner for 2x2 block operators.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Applies separate preconditioners to the (1,1) and (2,2) blocks.
/// The block-triangular preconditioner solves:
/*!
    \f[
    \begin{pmatrix}
    P_{11} & B^T \\
    0 & P_{22}
    \end{pmatrix}
    \begin{pmatrix}
    x_1 \\ x_2
    \end{pmatrix}
    =
    \begin{pmatrix}
    b_1 \\ b_2
    \end{pmatrix}
    \f]
*/
/// where \f$ P_{11} \f$ and \f$ P_{22} \f$ are preconditioners for the (1,1) and (2,2) blocks, respectively.
/// @tparam OperatorT Operator type (must satisfy Block2x2OperatorLike).
/// @tparam Block11T Type of the (1,1) operator of the preconditioner to be (approximately) inverted
/// @tparam Block22T Type of the (2,2) operator of the preconditioner to be (approximately) inverted
/// @tparam Block12T Type of the (1,2) operator of the preconditioner/Stokes operator, the gradient block
/// @tparam Block11Preconditioner Preconditioner for the (1,1) block (must satisfy SolverLike).
/// @tparam Block22Preconditioner Preconditioner for the (2,2) block (must satisfy SolverLike).
template <
    Block2x2OperatorLike OperatorT,
    OperatorLike         Block11T,
    OperatorLike         Block22T,
    OperatorLike         Block12T,
    SolverLike           Block11Preconditioner,
    SolverLike           Block22Preconditioner >
class BlockTriangularPreconditioner2x2
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;
    /// @brief Solution vector type (must be Block2VectorLike).
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type (must be Block2VectorLike).
    using RHSVectorType = DstOf< OperatorType >;

    /// @brief Static assertions to ensure block vector types.
    static_assert(
        Block2VectorLike< SolutionVectorType >,
        "The solution vector of the BlockPreconditioner2x2 must be Block2VectorLike." );
    static_assert(
        Block2VectorLike< RHSVectorType >,
        "The RHS vector of the BlockPreconditioner2x2 must be Block2VectorLike." );

    static_assert( std::is_same_v< SrcOf< Block11T >, typename SrcOf< OperatorT >::Block1Type > );
    static_assert( std::is_same_v< SrcOf< Block22T >, typename SrcOf< OperatorT >::Block2Type > );

    static_assert( std::is_same_v< DstOf< Block11T >, typename DstOf< OperatorT >::Block1Type > );
    static_assert( std::is_same_v< DstOf< Block22T >, typename DstOf< OperatorT >::Block2Type > );

    /// @brief Construct a block-triangular preconditioner with given block preconditioners and the gradient block.
    ///
    /// When calling solve( A, x, b ) with this preconditioner, the two passed solvers are applied to the two block
    /// passed in the constructor here. This does NOT use the (1, 1) and (2, 2) blocks of the A block in the solve()
    /// call.
    ///
    /// @param block11 The (1, 1) block to approximate the inverse to.
    /// @param block22 The (2, 2) block to approximate the inverse to.
    /// @param block12 The (1, 2) gradient block.
    /// @param block11_preconditioner Preconditioner for the (1,1) block.
    /// @param block22_preconditioner Preconditioner for the (2,2) block.
    BlockTriangularPreconditioner2x2(
        const Block11T&              block11,
        const Block22T&              block22,
        const Block12T&              block12,
        SolutionVectorType&          tmp,
        const Block11Preconditioner& block11_preconditioner,
        const Block22Preconditioner& block22_preconditioner )
    : block11_( block11 )
    , block22_( block22 )
    , block12_( block12 )
    , tmp_( tmp )
    , block11_preconditioner_( block11_preconditioner )
    , block22_preconditioner_( block22_preconditioner )
    {}

    /// @brief Solve the block-triangular preconditioner system by block-backward substitution.
    ///
    /// First, solve for the pressure block (second block row) by applying the schur preconditioner.
    /// Backward substitute the result into the first block row, multiply by 12/gradient block.
    /// Obtain the velocity solution by solving the 11 block/applying the velocity preconditioner.
    /// The Schur complement has a negative sign which is implemented in between.
    ///
    /// @param A Is ignored. The two solvers are applied to the operators passed in the constructor.
    /// @param x Solution block vector (output).
    /// @param b Right-hand side block vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        // solve schur op
        solve( block22_preconditioner_, block22_, x.block_2(), b.block_2() );

        // apply gradient op / 12Block to schur-preconditioned pressure
        apply( block12_, x.block_2(), tmp_.block_1() );

        // compute velocity residual
        lincomb( tmp_.block_1(), { 1, 1 }, { tmp_.block_1(), b.block_1() } );

        // schur comp has negative sign
        lincomb( x.block_2(), { -1 }, { x.block_2() } );

        // apply velocity preconditioner
        solve( block11_preconditioner_, block11_, x.block_1(), tmp_.block_1() );
    }

  private:
    Block11T block11_; ///< (1,1) block of operator.
    Block22T block22_; ///< (2,2) block of operator.
    Block12T block12_; ///< (1,2) block of operator.

    SolutionVectorType tmp_;

    Block11Preconditioner block11_preconditioner_; ///< Preconditioner for (1,1) block.
    Block22Preconditioner block22_preconditioner_; ///< Preconditioner for (2,2) block.
};

/// @brief Static assertion: BlockDiagonalPreconditioner2x2 satisfies SolverLike concept.
static_assert( SolverLike< BlockTriangularPreconditioner2x2<
                   linalg::detail::DummyConcreteBlock2x2Operator,
                   linalg::detail::DummyConcreteBlock2x2Operator::Block11Type,
                   linalg::detail::DummyConcreteBlock2x2Operator::Block22Type,
                   linalg::detail::DummyConcreteBlock2x2Operator::Block12Type,
                   detail::DummySolver< linalg::detail::DummyConcreteBlock2x2Operator::Block11Type >,
                   detail::DummySolver< linalg::detail::DummyConcreteBlock2x2Operator::Block22Type > > > );
} // namespace terra::linalg::solvers
