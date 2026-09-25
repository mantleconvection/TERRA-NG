#pragma once

#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::linalg::solvers {

/// @brief "Diagonal solver" for linear systems.
///
/// Implements a diagonal solve operation by inverting a given diagonal
/// in the constructor and assigning a scaled rhs to the solution for the
/// solve: x \gets D^{-1}b \f$.
/// Satisfies the SolverLike concept (see solver.hpp).
///
/// OperatorT is essentially ignored (does not need to be the identity operator).
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
template < OperatorLike OperatorT >
class DiagonalSolver
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType = DstOf< OperatorType >;

  private:
    SolutionVectorType& inv_diagonal_;

  public:
    DiagonalSolver( SolutionVectorType& diagonal )
    : inv_diagonal_( diagonal )
    {
        linalg::invert_entries( inv_diagonal_ );
    }

    /// @brief Solve the diagonal linear system by just scaling the rhs with the inverse diagonal.
    /// Implements \f$ x = b \f$.
    /// @param A Operator (matrix), unused.
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    template < typename OpT >
    void solve_impl( OpT& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        assign( x, b );
        scale_in_place( x, inv_diagonal_ );
    }
};

/// @brief Static assertion: IdentitySolver satisfies SolverLike concept.
static_assert( SolverLike< DiagonalSolver< linalg::detail::DummyOperator<
                   linalg::detail::DummyVector< double >,
                   linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers