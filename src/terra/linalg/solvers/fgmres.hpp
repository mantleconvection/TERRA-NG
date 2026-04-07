#pragma once

#include "eigen/eigen_wrapper.hpp"
#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"
#include "util/table.hpp"

namespace terra::linalg::solvers {

template < typename ScalarType >
struct FGMRESOptions
{
    /// @brief Number of inner iterations before restart (FGMRES(m)).
    int restart = 30;

    /// @brief Relative residual tolerance for convergence.
    ScalarType relative_residual_tolerance = 1e-6;

    /// @brief Absolute residual tolerance for convergence.
    ScalarType absolute_residual_tolerance = 1e-6;

    /// @brief Maximum number of total iterations (across all restarts).
    int max_iterations = 100;
};

/// @brief Flexible GMRES (FGMRES) iterative solver for nonsymmetric linear systems.
///
/// FGMRES allows for varying (flexible) preconditioning at each iteration,
/// making it suitable for use with inexact or variable preconditioners.
///
/// Reference:
/// @code
/// Saad, Y. (1993).
/// A flexible inner-outer preconditioned GMRES algorithm.
/// SIAM Journal on Scientific Computing, 14(2), 461-469.
/// @endcode
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Supports optional right preconditioning via the flexible framework.
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam PreconditionerT Preconditioner type (must satisfy SolverLike, defaults to IdentitySolver).
template < OperatorLike OperatorT, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class FGMRES
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;

    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;

    /// @brief Right-hand side vector type.
    using RHSVectorType = DstOf< OperatorType >;

    /// @brief Scalar type for computations.
    using ScalarType = typename SolutionVectorType::ScalarType;

    /// @brief Construct an FGMRES solver with a custom preconditioner.
    ///
    /// @param tmp Temporary vectors for workspace. Must contain at least 2*restart + 4 vectors:
    ///            - tmp[0]: residual vector r
    ///            - tmp[1 .. restart+1]: Arnoldi basis vectors V_0..V_restart
    ///            - tmp[restart+2 .. 2*restart+1]: preconditioned directions Z_0..Z_{restart-1}
    ///            - tmp[2*restart+2]: work vector w for A*z
    ///            - tmp[2*restart+3]: accumulator vector for solution update
    /// @param options FGMRES solver parameters (restart, tolerances, max iterations).
    /// @param statistics Shared pointer to statistics table for logging iteration progress (optional).
    /// @param preconditioner Preconditioner solver (defaults to identity).
    FGMRES(
        const std::vector< SolutionVectorType >& tmp,
        const FGMRESOptions< ScalarType >&       options        = {},
        const std::shared_ptr< util::Table >&    statistics     = nullptr,
        const PreconditionerT                    preconditioner = IdentitySolver< OperatorT >() )
    : tag_( "fgmres_solver" )
    , tmp_( tmp )
    , options_( options )
    , statistics_( statistics )
    , preconditioner_( preconditioner )
    , skip_preconditioner_in_case_of_nan_or_infs_( true )
    {}

    /// @brief Set a tag string for statistics output identification.
    /// @param tag Tag string to identify this solver instance in statistics.
    void set_tag( const std::string& tag ) { tag_ = tag; }

    /// @brief Set the number of inner iterations before restart (FGMRES(m)).
    /// @param m Restart parameter (must be at least 1).
    void set_restart( int m ) { options_.restart = std::max( 1, m ); }

    /// @brief Solve the linear system \f$ Ax = b \f$ using flexible GMRES with restarts.
    ///
    /// Uses right preconditioning: solves \f$ A M^{-1} y = b \f$ where \f$ x = M^{-1} y \f$.
    /// The preconditioner M can vary at each iteration (flexibility).
    ///
    /// The method builds an Arnoldi basis via modified Gram-Schmidt orthogonalization,
    /// applies Givens rotations to minimize the least-squares residual in the Krylov subspace,
    /// and updates the solution. If convergence is not achieved within the restart window,
    /// the process restarts with the updated solution until max_iterations is reached or
    /// convergence criteria are satisfied.
    ///
    /// @param A Operator (matrix) to solve with.
    /// @param x Solution vector (input: initial guess, output: final solution).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        // Compute initial residual r = b - A*x
        auto& r = tmp_[0];
        apply( A, x, r );
        lincomb( r, { 1.0, -1.0 }, { b, r } );

        ScalarType       beta0            = std::sqrt( dot( r, r ) );
        const ScalarType initial_residual = beta0;

        if ( statistics_ )
        {
            statistics_->add_row(
                { { "tag", tag_ }, { "iteration", 0 }, { "relative_residual", 1.0 }, { "absolute_residual", beta0 } } );
        }

        if ( beta0 <= options_.absolute_residual_tolerance )
        {
            return;
        }

        // Validate workspace size.
        // Required: 1 (r) + (m+1) (V) + m (Z) + 1 (w) + 1 (acc) = 2m + 4
        const int nTmp   = static_cast< int >( tmp_.size() );
        const int m_req  = options_.restart;
        const int needed = 2 * m_req + 4;
        if ( nTmp < needed )
        {
            std::cerr << "FGMRES: insufficient tmp vectors. Provided " << nTmp
                      << ", required at least (2*restart + 4) = " << needed << " for restart m = " << m_req
                      << std::endl;
            Kokkos::abort( "FGMRES: insufficient tmp vectors" );
        }

        // Workspace layout offsets:
        const int offV   = 1;                    // V_0 .. V_m stored in tmp[1 .. m+1]
        const int offZ   = offV + ( m_req + 1 ); // Z_0 .. Z_{m-1} stored in tmp[offV + m+1 .. offV + 2m]
        const int idxW   = offZ + m_req;         // w = A*z_j
        const int idxAcc = idxW + 1;             // accumulator for solution update

        // Allocate small dense arrays for Hessenberg matrix and Givens rotations.
        Eigen::Matrix< ScalarType, Eigen::Dynamic, Eigen::Dynamic > H( m_req + 1, m_req );
        Eigen::Matrix< ScalarType, Eigen::Dynamic, 1 >              g( m_req + 1 );
        std::vector< ScalarType >                                   cs( m_req + 1, 0 ), sn( m_req + 1, 0 );

        int total_iters = 0;

        // Outer restart loop
        while ( total_iters < options_.max_iterations )
        {
            // Initialize Arnoldi: V_0 = r / ||r||
            auto& V0 = tmp_[offV + 0];
            lincomb( V0, { 1.0 / beta0 }, { r } );

            // Reset dense workspace for this restart cycle
            H.setZero();
            g.setZero();
            std::fill( cs.begin(), cs.end(), ScalarType( 0 ) );
            std::fill( sn.begin(), sn.end(), ScalarType( 0 ) );
            g( 0 ) = beta0;

            int inner_its = 0;

            // Inner Arnoldi iteration (build Krylov subspace up to dimension m_req)
            for ( int j = 0; j < m_req && total_iters < options_.max_iterations; ++j, ++total_iters )
            {
                // Apply preconditioner: z_j = M^{-1} v_j
                auto& vj = tmp_[offV + j];
                auto& zj = tmp_[offZ + j];
                assign( zj, 0 );
                solve( preconditioner_, A, zj, vj );

                const bool preconditioner_result_contains_nan_or_inf = has_nan_or_inf( zj );

                if ( skip_preconditioner_in_case_of_nan_or_infs_ && preconditioner_result_contains_nan_or_inf )
                {
                    util::logroot
                        << "FGMRES: The preconditioner appears to have produced a vector that has NaN or Inf entries.\n"
                           "        This may be a result of the preconditioner or of the input provided by FGMRES.\n"
                           "        To at least provide a fix for the first case, we keep the input to the preconditioner\n"
                           "        and write it to the output (equivalent of skipping the preconditioner in this iteration).\n"
                           "        (Details: total_iters = "
                        << total_iters << ", j = " << j << ")" << std::endl;

                    assign( zj, vj );
                }

                // Apply operator: w = A * z_j
                auto& w = tmp_[idxW];
                apply( A, zj, w );

                // Modified Gram-Schmidt orthogonalization against V_0..V_j
                for ( int i = 0; i <= j; ++i )
                {
                    auto&            vi  = tmp_[offV + i];
                    const ScalarType hij = dot( w, vi );
                    H( i, j )            = hij;
                    lincomb( w, { 1.0, -hij }, { w, vi } );
                }

                // Compute h_{j+1,j} and normalize to get v_{j+1}
                const ScalarType h_jp1j = std::sqrt( dot( w, w ) );

                // SAFE BREAK 1: Arnoldi Breakdown
                if ( h_jp1j < std::numeric_limits< ScalarType >::epsilon() * initial_residual )
                {
                    H( j + 1, j ) = 0;
                    inner_its     = j + 1;

                    util::logroot << "FGMRES: Arnoldi breakdown. Restarting.\n"
                                     "        (Details: total_iters = "
                                  << total_iters << ", j = " << j << ")" << std::endl;

                    break; // Subspace is "full" or converged; exit to solve current least squares
                }

                H( j + 1, j ) = h_jp1j;

                if ( h_jp1j > ScalarType( 0 ) )
                {
                    auto& vjp1 = tmp_[offV + j + 1];
                    lincomb( vjp1, { 1.0 / h_jp1j }, { w } );
                }

                // Apply previous Givens rotations to column j of H
                for ( int i = 0; i < j; ++i )
                {
                    const ScalarType temp = cs[i] * H( i, j ) + sn[i] * H( i + 1, j );
                    H( i + 1, j )         = -sn[i] * H( i, j ) + cs[i] * H( i + 1, j );
                    H( i, j )             = temp;
                }

                // Compute new Givens rotation to zero H(j+1, j)
                {
                    const ScalarType a    = H( j, j );
                    const ScalarType b    = H( j + 1, j );
                    const ScalarType r_ab = std::hypot( a, b );
                    cs[j]                 = ( r_ab == ScalarType( 0 ) ) ? ScalarType( 1 ) : a / r_ab;
                    sn[j]                 = ( r_ab == ScalarType( 0 ) ) ? ScalarType( 0 ) : b / r_ab;
                    H( j, j )             = r_ab;
                    H( j + 1, j )         = 0;

                    // SAFE BREAK 2: Stagnation / Singularity
                    if ( std::abs( H( j, j ) ) < std::numeric_limits< ScalarType >::epsilon() )
                    {
                        inner_its = j; // Don't include this column in the back-solve

                        util::logroot << "FGMRES: Stagnation after Givens rotation. Restarting.\n"
                                         "        (Details: total_iters = "
                                      << total_iters << ", j = " << j << ")" << std::endl;

                        break;
                    }

                    // Apply new Givens rotation to g
                    const ScalarType gj  = g( j );
                    const ScalarType gj1 = g( j + 1 );
                    g( j )               = cs[j] * gj + sn[j] * gj1;
                    g( j + 1 )           = -sn[j] * gj + cs[j] * gj1;
                }

                // Estimate residual norm from least-squares problem
                const ScalarType abs_res = std::abs( g( j + 1 ) );
                const ScalarType rel_res = abs_res / initial_residual;

                if ( statistics_ )
                {
                    statistics_->add_row(
                        { { "tag", tag_ },
                          { "iteration", total_iters + 1 },
                          { "relative_residual", rel_res },
                          { "absolute_residual", abs_res } } );
                }

                // SAFE BREAK 3: NaN detected in residual estimate.
                // Discard this iteration and return the current best solution.
                if ( !std::isfinite( abs_res ) )
                {
                    util::logroot
                        << "FGMRES: NaN or Inf detected in residual estimate. "
                           "Returning current solution without this iteration's update.\n"
                           "        (Details: total_iters = "
                        << total_iters << ", j = " << j << ")" << std::endl;

                    inner_its = j; // exclude the corrupted iteration from the back-solve
                    ++total_iters;
                    break;
                }

                inner_its = j + 1;

                // Check for convergence
                if ( rel_res <= options_.relative_residual_tolerance || abs_res < options_.absolute_residual_tolerance )
                {
                    break;
                }
            }

            // Solve upper-triangular system R*y = g(0..inner_its-1) via back-substitution
            Eigen::Matrix< ScalarType, Eigen::Dynamic, 1 > y( inner_its );
            y.setZero();
            for ( int row = inner_its - 1; row >= 0; --row )
            {
                ScalarType sum = g( row );
                for ( int col = row + 1; col < inner_its; ++col )
                    sum -= H( row, col ) * y( col );
                y( row ) = sum / H( row, row );
            }

            // Update solution: x += sum_{i=0}^{inner_its-1} y_i * z_i
            auto& acc = tmp_[idxAcc];
            assign( acc, 0 );
            for ( int i = 0; i < inner_its; ++i )
            {
                auto& zi = tmp_[offZ + i];
                lincomb( acc, { 1.0, y( i ) }, { acc, zi } );
            }
            lincomb( x, { 1.0, 1.0 }, { x, acc } );

            // Recompute residual r = b - A*x for next restart
            apply( A, x, r );
            lincomb( r, { 1.0, -1.0 }, { b, r } );
            beta0 = std::sqrt( dot( r, r ) );

            // Check for final convergence or abort on NaN
            if ( !std::isfinite( beta0 ) )
            {
                util::logroot << "FGMRES: Residual is NaN/Inf after restart. Aborting solve.\n"
                                 "        (Details: beta0 = "
                              << beta0 << ", total_iters = " << total_iters << ")" << std::endl;
                return;
            }
            if ( beta0 <= options_.absolute_residual_tolerance ||
                 beta0 / initial_residual <= options_.relative_residual_tolerance )
            {
                return;
            }
        }
    }

  private:
    std::string tag_; ///< Tag for statistics output identification.

    std::vector< SolutionVectorType > tmp_; ///< Temporary workspace vectors.

    FGMRESOptions< ScalarType > options_; ///< Solver parameters (restart, tolerances, max_iterations).

    std::shared_ptr< util::Table > statistics_; ///< Statistics table for iteration logging (optional).

    PreconditionerT preconditioner_; ///< Preconditioner solver.

    bool skip_preconditioner_in_case_of_nan_or_infs_;
};

/// @brief Static assertion: FGMRES satisfies SolverLike concept.
static_assert(
    SolverLike<
        FGMRES< linalg::detail::
                    DummyOperator< linalg::detail::DummyVector< double >, linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers