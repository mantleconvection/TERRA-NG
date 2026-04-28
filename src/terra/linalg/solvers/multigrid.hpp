#pragma once

#include <mpi.h>
#include <type_traits>

#include "solver.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

namespace terra::linalg::solvers {

/// @brief Placeholder redistribute type — methods never called when
/// redistribute_down_ is empty. Exists so Multigrid's default template parameter
/// has a complete, default-constructible type for the no-agglomeration path.
struct NoRedistribute
{
    template < typename V >
    void apply( const V&, V& ) const {}
    template < typename V >
    void apply_transpose( const V&, V& ) const {}
};

/// @brief Multigrid solver for linear systems.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Supports arbitrary operators, prolongation/restriction, smoothers, and coarse grid solvers.
/// Implements recursive V-cycle multigrid.
///
/// Optionally supports hierarchical agglomeration: per level, a Redistribute
/// operator moves the restricted residual from the fine level's communicator to
/// a (smaller) coarse level communicator, and the coarse correction back up.
/// Inactive coarse ranks skip the recursive descent but still participate in
/// the collectives inside Redistribute::apply / apply_transpose.
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam ProlongationT Prolongation operator type (must satisfy OperatorLike).
/// @tparam RestrictionT Restriction operator type (must satisfy OperatorLike).
/// @tparam SmootherT Smoother type (must satisfy SolverLike).
/// @tparam CoarseGridSolverT Coarse grid solver type (must satisfy SolverLike).
/// @tparam RedistributeT Optional; defaults to NoRedistribute. Pass the concrete
///         Redistribute<GridDataType> type when using agglomeration.
template <
    OperatorLike OperatorT,
    OperatorLike ProlongationT,
    OperatorLike RestrictionT,
    SolverLike   SmootherT,
    SolverLike   CoarseGridSolverT,
    class        RedistributeT = NoRedistribute >
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

    // ---------------------------------------------------------------------------
    // Optional agglomeration plumbing. All three vectors are either empty (= no
    // agglomeration, classical V-cycle) or sized A_c_.size() with one entry per
    // coarse descent step.
    //
    // When configured:
    //   - R_[level-1] writes its output into tmp_r_fine_[level-1] (which lives on
    //     the *fine* level's communicator), and the plan moves that vector's
    //     contents onto tmp_r_[level-1] (which lives on the *coarse* level's
    //     sub-comm). Mirrored for prolongation with tmp_e_fine_.
    // ---------------------------------------------------------------------------
    std::vector< RedistributeT >      redistribute_down_;
    std::vector< SolutionVectorType > tmp_r_fine_;
    std::vector< SolutionVectorType > tmp_e_fine_;

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

    /// @brief Construct an agglomerated multigrid solver.
    ///
    /// Same as the non-agglomerated constructor, plus:
    ///   - @p redistribute_down: Redistribute operators, one per coarse descent
    ///     step. redistribute_down[L-1] moves a vector from level L's comm to
    ///     level L-1's comm. apply_transpose is used on the way back up.
    ///   - @p tmp_r_fine, tmp_e_fine: per-coarse-level scratch vectors allocated
    ///     on the *fine* level's communicator. Restriction output goes into
    ///     tmp_r_fine; prolongation input is read from tmp_e_fine.
    /// @p tmp_r, tmp_e must carry the coarse level's sub-comm; on ranks not
    ///     part of the sub-comm they are empty (zero-subdomain) vectors.
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
        ScalarType                               relative_residual_threshold,
        std::vector< RedistributeT >             redistribute_down,
        std::vector< SolutionVectorType >        tmp_r_fine,
        std::vector< SolutionVectorType >        tmp_e_fine )
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
    , redistribute_down_( std::move( redistribute_down ) )
    , tmp_r_fine_( std::move( tmp_r_fine ) )
    , tmp_e_fine_( std::move( tmp_e_fine ) )
    {
        // Allow two modes for the three agglomeration vectors:
        //   all empty         = agglomeration disabled at runtime
        //                       (solve_recursive's `agglomerated = !redistribute_down_.empty()` sees false);
        //   all sized A_c     = agglomeration enabled, one entry per descent.
        const bool all_empty = redistribute_down_.empty() && tmp_r_fine_.empty() && tmp_e_fine_.empty();
        const bool all_sized = redistribute_down_.size() == A_c.size()
                                && tmp_r_fine_.size()   == A_c.size()
                                && tmp_e_fine_.size()   == A_c.size();
        if ( !all_empty && !all_sized )
        {
            throw std::runtime_error(
                "Multigrid (agglomerated): redistribute_down, tmp_r_fine, and tmp_e_fine must either all be empty "
                "(agglomeration disabled) or each have one entry per coarse level." );
        }
    }

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

        const std::string level_suffix = "_L" + std::to_string( level );

        // Agglomeration is only available when RedistributeT is a concrete
        // redistribute type (not the NoRedistribute placeholder) AND the caller
        // provided the per-level operators/buffers. The if constexpr below
        // ensures the agglomerated branches — which require .grid_data() on the
        // vector type and valid per-level buffers — aren't instantiated when
        // RedistributeT is NoRedistribute.
        constexpr bool can_agglomerate = !std::is_same_v< RedistributeT, NoRedistribute >;

        // relax on Ax = b
        {
            util::Timer t_sp( "mg_smoother_pre" + level_suffix );
            solve( smoothers_pre_[level], A, x, b );
        }

        bool active_at_coarse = true;

#ifdef TERRA_MG_AGGLOM_DEBUG
        auto dbg = [&]( const char* phase ) {
            int r; MPI_Comm_rank( MPI_COMM_WORLD, &r );
            if ( r == 0 )
                std::cout << "[MG_DBG] level=" << level << " phase=" << phase << std::endl << std::flush;
        };
#else
        auto dbg = []( const char* ) {};
#endif

        if constexpr ( can_agglomerate )
        {
            // Runtime-selected path. When redistribute_down_ is empty this
            // collapses to the classical V-cycle (no agglomeration), which lets
            // callers instantiate Multigrid with a real Redistribute type but
            // leave the extra vectors empty to disable agglomeration at runtime.
            if ( redistribute_down_.empty() )
            {
                // Classical V-cycle.
                {
                    util::Timer t_rr( "mg_residual_and_restrict" + level_suffix );
                    apply( A, x, tmp_[level] );
                    lincomb( tmp_[level], { 1.0, -1.0 }, { b, tmp_[level] } );
                    apply( R_[level - 1], tmp_[level], tmp_r_[level - 1] );
                }

                assign( tmp_e_[level - 1], 0.0 );
                solve_recursive( A_c_[level - 1], tmp_e_[level - 1], tmp_r_[level - 1], level - 1 );

                {
                    util::Timer t_p( "mg_prolongate_correct" + level_suffix );
                    apply( P_additive_[level - 1], tmp_e_[level - 1], x );
                }
            }
            else
            {
                // Agglomerated path. If this descent's Redistribute is the
                // identity (factor=1: upper and lower comms + owner maps match
                // exactly), there's no hop to do — treat it as a classical
                // descent: restrict straight into tmp_r_[L-1], no apply, no
                // apply_transpose, prolongate straight from tmp_e_[L-1].
                const bool identity_descent = redistribute_down_[level - 1].is_identity();

                dbg( "pre_smooth_done" );
                {
                    util::Timer t_rr( "mg_residual_and_restrict" + level_suffix );
                    apply( A, x, tmp_[level] );
                    dbg( "residual_applied" );
                    lincomb( tmp_[level], { 1.0, -1.0 }, { b, tmp_[level] } );
                    apply( R_[level - 1],
                            tmp_[level],
                            identity_descent ? tmp_r_[level - 1] : tmp_r_fine_[level - 1] );
                    dbg( "restriction_done" );
                }

                if ( !identity_descent )
                {
                    util::Timer t_rd( "mg_redistribute_down" + level_suffix );
                    redistribute_down_[level - 1].apply( tmp_r_fine_[level - 1].grid_data(),
                                                          tmp_r_[level - 1].grid_data() );
                    dbg( "redistribute_down_done" );
                }
                active_at_coarse = ( tmp_r_[level - 1].comm() != MPI_COMM_NULL );

                if ( active_at_coarse )
                {
                    assign( tmp_e_[level - 1], 0.0 );
                    dbg( "entering_coarse_recursion" );
                    solve_recursive( A_c_[level - 1], tmp_e_[level - 1], tmp_r_[level - 1], level - 1 );
                    dbg( "returned_from_coarse" );
                }
                else
                {
                    dbg( "skipped_coarse_recursion" );
                }

                if ( !identity_descent )
                {
                    util::Timer t_ru( "mg_redistribute_up" + level_suffix );
                    redistribute_down_[level - 1].apply_transpose( tmp_e_[level - 1].grid_data(),
                                                                    tmp_e_fine_[level - 1].grid_data() );
                    dbg( "redistribute_up_done" );
                }

                {
                    util::Timer t_p( "mg_prolongate_correct" + level_suffix );
                    apply( P_additive_[level - 1],
                            identity_descent ? tmp_e_[level - 1] : tmp_e_fine_[level - 1],
                            x );
                    dbg( "prolongation_done" );
                }
            }
        }
        else
        {
            // Classical V-cycle path.
            {
                util::Timer t_rr( "mg_residual_and_restrict" + level_suffix );
                apply( A, x, tmp_[level] );
                lincomb( tmp_[level], { 1.0, -1.0 }, { b, tmp_[level] } );
                apply( R_[level - 1], tmp_[level], tmp_r_[level - 1] );
            }

            assign( tmp_e_[level - 1], 0.0 );
            solve_recursive( A_c_[level - 1], tmp_e_[level - 1], tmp_r_[level - 1], level - 1 );

            {
                util::Timer t_p( "mg_prolongate_correct" + level_suffix );
                apply( P_additive_[level - 1], tmp_e_[level - 1], x );
            }
        }

        // relax on A x = b
        {
            util::Timer t_sP( "mg_smoother_post" + level_suffix );
            solve( smoothers_post_[level], A, x, b );
        }
    }
};

} // namespace terra::linalg::solvers
