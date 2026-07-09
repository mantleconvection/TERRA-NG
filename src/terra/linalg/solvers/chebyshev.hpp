#pragma once

#include "linalg/diagonally_scaled_operator.hpp"
#include "linalg/operator.hpp"
#include "power_iteration.hpp"
#include "solver.hpp"

#include <cmath>

namespace terra::linalg::solvers {

/// @brief Chebyshev accelerated Jacobi iterative solver for linear systems.
///
/// Computes (given an order \f$p \geq 1, \ p \in \mathbb{N}\f$)
/// \f[
///   x^{k+1} = x^k + \alpha_k \, D^{-1}(b - A x^k) + \beta_k \, d^k, \quad
///   d^0 = 0
/// \f]
/// for \f$k = 0\f$ up to \f$k+1 = p\f$, where \f$D = \mathrm{diag}(A)\f$, and the coefficients \f$ \alpha_k \f$ and
/// \f$ \beta_k \f$ are computed recursively from
/// \f[
///   \begin{aligned}
///   \rho_0 &= \frac{\delta}{\theta}, \quad \\
///   \rho_{k+1} &= (2(\theta/\delta) - \rho_k)^{-1}, \quad  \\
///   \alpha_0 &= \theta^{-1}, \\
///   \alpha_k &= \frac{2 \, \rho_{k+1}}{\delta}, \quad  \\
///   \beta_0 &= 0, \\
///   \beta_k &= \rho_{k+1} \rho_k, \\
///   d_0 &= 0, \\
///   d_k &= x^{k+1} - x^{k} = \alpha_k D^{-1}(b - A x^k) + \beta_k d_{k-1}
///   \end{aligned}
/// \f]
/// with \f$ \theta = (\lambda_{\max} + \lambda_{\min})/2 \f$ and \f$ \delta = (\lambda_{\max} - \lambda_{\min})/2\f$ ,
/// and \f$ \lambda_{\min}, \lambda_{\max} \f$ the eigenvalue bounds of \f$ D^{-1}A \f$.
///
/// The max eigenvalue \f$\lambda_{\max}\f$ is estimated before the first solve (or if requested via
/// `refresh_max_eigenvalue_estimate_in_next_solve()`) applying a specified number of power iterations. A safety margin
/// is added to hopefully ensure that the estimate is not too far off. The minimum eigenvalue is the obtained by scaling
/// the maximum eigenvalue.
///
/// Each iteration executes \f$p\f$ matrix-vector products (plus some vector-vector operations).
/// Setting the order \f$p = 1\f$ recovers the standard weighted Jacobi relaxation.
/// If unsure, try low-order settings with \f$p = 2\f$ or \f$p = 3\f$.
///
/// The number of iterations repeats the entire process. For instance, setting the iterations \f$\text{it} = 10\f$ will
/// perform \f$\text{it} \times p\f$ updates.
///
/// Satisfies the SolverLike concept (see solver.hpp).
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
template < OperatorLike OperatorT >
class Chebyshev
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType = DstOf< OperatorType >;

    /// @brief Scalar type for computations.
    using ScalarType = SolutionVectorType::ScalarType;

    /// @brief Construct a Chebyshev accelerated Jacobi solver.
    /// @param order Chebyshev order \f$p >= 1\f$. Equivalent to weighted Jacobi for \f$p = 1\f$.
    /// @param inverse_diagonal Inverse of the diagonal of the operator (D^-1).
    /// @param tmps Temporary vectors for workspace. We need 2 here.
    /// @param iterations Number of Chebyshev iterations to perform.
    /// @param max_ev_power_iterations Number of power iterations executed to estimate the max eigenvalue of ((D^-1)A).
    ///                                An additional safety margin is added internally during the setup of the Chebyshev
    ///                                solver.
    Chebyshev(
        const int                                order,
        const SolutionVectorType&                inverse_diagonal,
        const std::vector< SolutionVectorType >& tmps,
        const int                                iterations              = 1,
        const int                                max_ev_power_iterations = 50 )
    : order_( order )
    , inverse_diagonal_( inverse_diagonal )
    , iterations_( iterations )
    , tmps_( tmps )
    , max_ev_power_iterations_( max_ev_power_iterations )
    , need_max_ev_estimation_( true )
    {
        if ( order_ < 1 )
        {
            Kokkos::abort( "Chebyshev order must be at least 1." );
        }

        if ( tmps_.size() < 2 )
        {
            Kokkos::abort( "Chebyshev requires at least 2 tmp vectors." );
        }
    }

    /// @brief Solve the linear system using the Chebyshev iteration.
    /// Applies the update rule for the specified number of iterations.
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        if ( need_max_ev_estimation_ )
        {
            estimate_max_eigenvalues( A );
            need_max_ev_estimation_ = false;
        }

        auto& d = tmps_[0];
        auto& z = tmps_[1];

        // Guard against degenerate eigenvalue estimates (e.g. from near-singular operators).
        if ( !std::isfinite( max_ev_estimate_ ) || max_ev_estimate_ <= ScalarType( 0 ) )
        {
            max_ev_estimate_ = ScalarType( 1 );
        }

        const auto lambda_max = 1.5 * max_ev_estimate_;
        const auto lambda_min = 0.1 * max_ev_estimate_;

        const auto theta = 0.5 * ( lambda_max + lambda_min );
        const auto delta = 0.5 * ( lambda_max - lambda_min );

        const auto sigma = theta / delta;
        auto       rho   = 1.0 / sigma;

        for ( int iteration = 0; iteration < iterations_; ++iteration )
        {
            // r = b - A(x);
            // z = M_inv(r);

            apply( A, x, z );
            // Fused tail (z holds A*x): z = D^-1(b - A x); d = (1/theta) z; x = x + d.
            // First Chebyshev direction => beta = 0 (old d not read).
            chebyshev_fused_update( x, d, z, b, inverse_diagonal_, ScalarType( 1.0 / theta ), ScalarType( 0 ) );

            // cheby recurrence

            for ( int i = 2; i <= order_; ++i )
            {
                const auto rho_new = 1.0 / ( 2.0 * sigma - rho );

                const auto beta  = rho_new * rho;
                const auto alpha = 2.0 * rho_new / delta;

                apply( A, x, z );
                // Fused tail (z holds A*x): z = D^-1(b - A x); d = alpha z + beta d; x = x + d.
                chebyshev_fused_update( x, d, z, b, inverse_diagonal_, ScalarType( alpha ), ScalarType( beta ) );

                rho = rho_new;
            }
        }
    }

    SolutionVectorType& get_inverse_diagonal() { return inverse_diagonal_; }

    void refresh_max_eigenvalue_estimate_in_next_solve() { need_max_ev_estimation_ = true; }

  private:
    void estimate_max_eigenvalues( OperatorType& A )
    {
        DiagonallyScaledOperator< OperatorType > inv_diag_A( A, inverse_diagonal_ );
        max_ev_estimate_ = solvers::power_iteration( inv_diag_A, tmps_[0], tmps_[1], max_ev_power_iterations_ );
    }

    int                               order_;            ///< Order of the Chebyshev smoother.
    SolutionVectorType                inverse_diagonal_; ///< Inverse diagonal vector.
    int                               iterations_;       ///< Number of iterations.
    std::vector< SolutionVectorType > tmps_;             ///< Temporary workspace vectors.
    int        max_ev_power_iterations_;                 ///< Number of power iterations to estimate the max eigenvalue.
    bool       need_max_ev_estimation_;                  ///< If true, max eigenvalue is estimated in next solve call.
    ScalarType max_ev_estimate_;                         ///< Estimate of the max eigenvalue of ((D^-1)A).
};

/// @brief Static assertion: Chebyshev satisfies SolverLike concept.
static_assert( SolverLike< Chebyshev< linalg::detail::DummyConcreteOperator > > );

} // namespace terra::linalg::solvers