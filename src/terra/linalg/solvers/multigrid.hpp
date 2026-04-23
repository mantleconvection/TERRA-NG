#pragma once

#include "solver.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

namespace terra::linalg::solvers {

/// @brief Multigrid solver for linear systems.
/// 
/// Satisfies the SolverLike concept (see solver.hpp).
/// Supports arbitrary operators, prolongation/restriction, smoothers, and coarse grid solvers.
/// Implements recursive V-cycle multigrid.
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam ProlongationT Prolongation operator type (must satisfy OperatorLike).
/// @tparam RestrictionT Restriction operator type (must satisfy OperatorLike).
/// @tparam SmootherT Smoother type (must satisfy SolverLike).
/// @tparam CoarseGridSolverT Coarse grid solver type (must satisfy SolverLike).
template <
    OperatorLike OperatorT,
    OperatorLike ProlongationT,
    OperatorLike RestrictionT,
    SolverLike   SmootherT,
    SolverLike   CoarseGridSolverT >
class Multigrid
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType         = OperatorT;
    /// @brief Prolongation operator type.
    using ProlongationType     = ProlongationT;
    /// @brief Restriction operator type.
    using RestrictionType      = RestrictionT;
    /// @brief Smoother type.
    using SmootherType         = SmootherT;
    /// @brief Coarse grid solver type.
    using CoarseGridSolverType = CoarseGridSolverT;

    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType      = DstOf< OperatorType >;

    /// @brief Scalar type for computations.
    using ScalarType = SolutionVectorType::ScalarType;

  private:
    std::vector< ProlongationType >   P_additive_; ///< Prolongation operators for each level.
    std::vector< RestrictionType >    R_;          ///< Restriction operators for each level.
    std::vector< OperatorT >          A_c_;        ///< Coarse grid operators for each level.
    std::vector< SolutionVectorType > tmp_r_;      ///< Temporary residual vectors for each level.
    std::vector< SolutionVectorType > tmp_e_;      ///< Temporary error vectors for each level.
    std::vector< SolutionVectorType > tmp_;        ///< Temporary workspace vectors for each level.
    std::vector< SmootherType >       smoothers_pre_;  ///< Pre-smoothers for each level.
    std::vector< SmootherType >       smoothers_post_; ///< Post-smoothers for each level.
    CoarseGridSolverType              coarse_grid_solver_; ///< Coarse grid solver.

    int        num_cycles_;                   ///< Number of multigrid cycles to perform.
    ScalarType relative_residual_threshold_;  ///< Relative residual threshold for stopping.

    std::shared_ptr< util::Table > statistics_; ///< Statistics table.
    std::string                    tag_ = "multigrid"; ///< Tag for statistics output.

  public:
    /// @brief Construct a multigrid solver.
    ///
    /// Vector ordering of arguments always goes from the coarsest level (index 0) to the finest.
    ///
    /// @param P_additive Prolongation operators for each coarse level. 
    ///                   Size must match the number of levels - 1.
    ///                   Must be additive prolongation operators, i.e., @code apply( P, x, y ) @endcode computes \f$ y = y + P x \f$.
    /// @param R Restriction operators for each coarse level.
    /// @param A_c Coarse grid operators for each coarse level.
    /// @param tmp_r Temporary residual vectors for each coarse level.
    /// @param tmp_e Temporary error vectors for each coarse level.
    /// @param tmp Temporary workspace vectors for each level (including the finest level).
    /// @param smoothers_pre Pre-smoothers for each level (including the finest level).
    /// @param smoothers_post Post-smoothers for each level (including the finest level).
    /// @param coarse_grid_solver Coarse grid solver.
    /// @param num_cycles Number of multigrid cycles to perform.
    /// @param relative_residual_threshold Relative residual threshold for stopping.
    Multigrid(
        const std::vector< ProlongationType >&   P_additive,
        const std::vector< RestrictionType >&    R,
        const std::vector< OperatorT >&          A_c,
        const std::vector< SolutionVectorType >& tmp_r,
        const std::vector< SolutionVectorType >& tmp_e,
        const std::vector< SolutionVectorType >& tmp,
        const std::vector< SmootherType >&       smoothers_pre,
        const std::vector< SmootherType >&       smoothers_post,
        const CoarseGridSolverType&              coarse_grid_solver,
        int                                      num_cycles,
        ScalarType                               relative_residual_threshold )
    : P_additive_( P_additive )
    , R_( R )
    , A_c_( A_c )
    , tmp_r_( tmp_r )
    , tmp_e_( tmp_e )
    , tmp_( tmp )
    , smoothers_pre_( smoothers_pre )
    , smoothers_post_( smoothers_post )
    , coarse_grid_solver_( coarse_grid_solver )
    , num_cycles_( num_cycles )
    , relative_residual_threshold_( relative_residual_threshold )
    {}

    /// @brief Set a tag string for statistics output.
    /// @param tag Tag string.
    void set_tag( const std::string& tag ) { tag_ = tag; }

    /// @brief Collect statistics in a shared table.
    /// @param statistics Shared pointer to statistics table.
    void collect_statistics( const std::shared_ptr< util::Table >& statistics ) { statistics_ = statistics; }

    /// @brief Solve the linear system using multigrid cycles.
    /// Calls the recursive V-cycle and updates statistics.
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        util::Timer timer_mg_solve( "mg_solve" );

        if ( P_additive_.size() != A_c_.size() || R_.size() != A_c_.size() || tmp_r_.size() != A_c_.size() ||
             tmp_e_.size() != A_c_.size() || tmp_.size() != A_c_.size() + 1 )
        {
            throw std::runtime_error(
                "Multigrid: P_additive, R, A_c, tmp_e, and tmp_r must be available for all coarse levels. tmp "
                "requires the finest grid allocated, too." );
        }

        const int max_level = P_additive_.size();

        ScalarType initial_residual = 0.0;

        if ( statistics_ )
        {
            util::Timer t_res( "mg_residual_compute" );
            apply( A, x, tmp_[max_level] );
            lincomb( tmp_[max_level], { 1.0, -1.0 }, { b, tmp_[max_level] } );
            initial_residual = norm_2( tmp_[max_level] );

            statistics_->add_row(
                { { "tag", tag_ },
                  { "cycle", 0 },
                  { "relative_residual", 1.0 },
                  { "absolute_residual", initial_residual },
                  { "residual_convergence_rate", 1.0 } } );
        }

        ScalarType previous_absolut_residual = initial_residual;

        for ( int cycle = 1; cycle <= num_cycles_; ++cycle )
        {
            {
                util::Timer t_cyc( "mg_vcycle" );
                solve_recursive( A, x, b, max_level );
            }

            if ( statistics_ )
            {
                util::Timer t_res( "mg_residual_compute" );
                apply( A, x, tmp_[max_level] );
                lincomb( tmp_[max_level], { 1.0, -1.0 }, { b, tmp_[max_level] } );
                const auto absolute_residual = norm_2( tmp_[max_level] );

                const auto relative_residual = absolute_residual / initial_residual;

                statistics_->add_row(
                    { { "tag", tag_ },
                      { "cycle", cycle },
                      { "relative_residual", relative_residual },
                      { "absolute_residual", absolute_residual },
                      { "residual_convergence_rate", absolute_residual / previous_absolut_residual } } );

                if ( relative_residual <= relative_residual_threshold_ )
                {
                    return;
                }

                previous_absolut_residual = absolute_residual;
            }
        }
    }

  private:
    /// @brief Recursive V-cycle multigrid solver.
    /// @param A Operator (matrix) at current level.
    /// @param x Solution vector (output) at current level.
    /// @param b Right-hand side vector (input) at current level.
    /// @param level Current multigrid level.
    void solve_recursive( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b, int level )
    {
        if ( level == 0 )
        {
            util::Timer t_cs( "mg_coarse_solve" );
            solve( coarse_grid_solver_, A, tmp_e_[0], tmp_r_[0] );
            return;
        }

        // relax on Ax = b
        {
            util::Timer t_sp( "mg_smoother_pre" );
            solve( smoothers_pre_[level], A, x, b );
        }

        // compute the residual r = b - Ax and restrict it to the coarse level
        {
            util::Timer t_rr( "mg_residual_and_restrict" );
            apply( A, x, tmp_[level] );
            lincomb( tmp_[level], { 1.0, -1.0 }, { b, tmp_[level] } );
            apply( R_[level - 1], tmp_[level], tmp_r_[level - 1] );
        }

        // solve (recursively) A_c e_c = r_c
        assign( tmp_e_[level - 1], 0.0 );
        solve_recursive( A_c_[level - 1], tmp_e_[level - 1], tmp_r_[level - 1], level - 1 );

        // apply the coarse grid correction x = x + P e_c
        {
            util::Timer t_p( "mg_prolongate_correct" );
            apply( P_additive_[level - 1], tmp_e_[level - 1], x );
        }

        // relax on A x = b
        {
            util::Timer t_sP( "mg_smoother_post" );
            solve( smoothers_post_[level], A, x, b );
        }
    }
};

} // namespace terra::linalg::solvers
