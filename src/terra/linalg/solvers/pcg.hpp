#pragma once

#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"
#include "util/table.hpp"

#include <cmath>
#include <limits>

namespace terra::linalg::solvers {

/// @brief Preconditioned Conjugate Gradient (PCG) iterative solver for symmetric positive definite linear systems.
/// 
/// See, e.g., 
/// @code
/// Elman, H. C., Silvester, D. J., & Wathen, A. J. (2014). 
/// Finite elements and fast iterative solvers: with applications in incompressible fluid dynamics. 
/// Oxford university press. 
/// @endcode
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Supports optional preconditioning.
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam PreconditionerT Preconditioner type (must satisfy SolverLike, defaults to IdentitySolver).
template < OperatorLike OperatorT, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class PCG
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType       = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType      = DstOf< OperatorType >;
    /// @brief Scalar type for computations.
    using ScalarType         = typename SolutionVectorType::ScalarType;

    /// @brief Construct a PCG solver with default identity preconditioner.
    /// @param params Iterative solver parameters.
    /// @param statistics Shared pointer to statistics table.
    /// @param tmps Temporary vectors for workspace. (At least 4 vectors are required.)
    PCG( const IterativeSolverParameters&         params,
         const std::shared_ptr< util::Table >&    statistics,
         const std::vector< SolutionVectorType >& tmps )
    : PCG( params, statistics, tmps, IdentitySolver< OperatorT >() )
    {}

    /// @brief Construct a PCG solver with a custom preconditioner.
    /// @param params Iterative solver parameters.
    /// @param statistics Shared pointer to statistics table.
    /// @param tmps Temporary vectors for workspace. (At least 4 vectors are required.)
    /// @param preconditioner Preconditioner solver.
    PCG( const IterativeSolverParameters&         params,
         const std::shared_ptr< util::Table >&    statistics,
         const std::vector< SolutionVectorType >& tmps,
         const PreconditionerT                    preconditioner )
    : tag_( "pcg_solver" )
    , params_( params )
    , statistics_( statistics )
    , tmps_( tmps )
    , preconditioner_( preconditioner )
    {
        if ( tmps.size() < 4 )
        {
            throw std::runtime_error( "PCG: tmps.size() < 4. Need at least 4 tmp vectors." );
        }
    }

    /// @brief Set a tag string for statistics output.
    /// @param tag Tag string.
    void set_tag( const std::string& tag ) { tag_ = tag; }

    /// @brief Solve the linear system \f$ Ax = b \f$ using PCG.
    /// Calls the iterative solver and updates statistics.
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        auto& r_  = tmps_[0]; ///< Residual vector.
        auto& p_  = tmps_[1]; ///< Search direction vector.
        auto& ap_ = tmps_[2]; ///< Temporary vector for A*p.
        auto& z_  = tmps_[3]; ///< Preconditioned residual vector.

        apply( A, x, r_ );

        lincomb( r_, { 1.0, -1.0 }, { b, r_ } );

        solve( preconditioner_, A, z_, r_ );

        assign( p_, z_ );

        // TODO: should this be dot(z, z) instead or dot(r, r)?
        const ScalarType initial_residual = std::sqrt( dot( r_, r_ ) );

        if ( statistics_ )
        {
            statistics_->add_row(
                { { "tag", tag_ },
                  { "iteration", 0 },
                  { "relative_residual", 1.0 },
                  { "absolute_residual", initial_residual } } );
        }

        if ( initial_residual < params_.absolute_residual_tolerance() )
        {
            return;
        }

        for ( int iteration = 1; iteration <= params_.max_iterations(); ++iteration )
        {
            const ScalarType alpha_num = dot( z_, r_ );

            apply( A, p_, ap_ );
            const ScalarType alpha_den = dot( ap_, p_ );

            if ( std::abs( alpha_den ) < std::numeric_limits< ScalarType >::min() )
            {
                // Search direction is in or near the null space of A.
                // Cannot make progress — break to avoid inf/NaN.
                return;
            }

            const ScalarType alpha = alpha_num / alpha_den;

            lincomb( x, { 1.0, alpha }, { x, p_ } );
            lincomb( r_, { 1.0, -alpha }, { r_, ap_ } );

            // TODO: is this the correct term for the residual check?
            const ScalarType absolute_residual = std::sqrt( dot( r_, r_ ) );

            const ScalarType relative_residual = absolute_residual / initial_residual;

            if ( statistics_ )
            {
                statistics_->add_row(
                    { { "tag", tag_ },
                      { "iteration", iteration },
                      { "relative_residual", relative_residual },
                      { "absolute_residual", absolute_residual } } );
            }

            if ( relative_residual <= params_.relative_residual_tolerance() )
            {
                return;
            }

            if ( absolute_residual < params_.absolute_residual_tolerance() )
            {
                return;
            }

            solve( preconditioner_, A, z_, r_ );

            const ScalarType beta_num = dot( z_, r_ );

            if ( std::abs( alpha_num ) < std::numeric_limits< ScalarType >::min() )
            {
                return;
            }

            const ScalarType beta = beta_num / alpha_num;

            lincomb( p_, { 1.0, beta }, { z_, p_ } );
        }
    }

  private:
    std::string tag_; ///< Tag for statistics output.

    IterativeSolverParameters params_; ///< Solver parameters.

    std::shared_ptr< util::Table > statistics_; ///< Statistics table.

    std::vector< SolutionVectorType > tmps_; ///< Temporary workspace vectors.

    PreconditionerT preconditioner_; ///< Preconditioner solver.
};

/// @brief Static assertion: PCG satisfies SolverLike concept.
static_assert(
    SolverLike<
        PCG< linalg::detail::
                 DummyOperator< linalg::detail::DummyVector< double >, linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers