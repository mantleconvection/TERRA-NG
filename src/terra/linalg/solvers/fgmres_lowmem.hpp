#pragma once

#include "eigen/eigen_wrapper.hpp"
#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

namespace terra::linalg::solvers {

/// @brief Flexible GMRES with a reduced-precision Krylov basis ("low-memory FGMRES").
///
/// Identical algorithm to @ref FGMRES, but the Arnoldi basis V_0..V_m and the
/// preconditioned directions Z_0..Z_{m-1} are *stored* in @p BasisVectorType
/// (typically a single-precision version of the solution vector), while the
/// operator, preconditioner, residual, orthogonalization and dense Hessenberg
/// arithmetic all run in the full solver precision (@p SolutionVectorType).
///
/// Basis vectors are converted to/from a double-precision scratch vector at the
/// few points where they participate in arithmetic (apply, preconditioner,
/// dot, lincomb). This halves the memory of the dominant FGMRES workspace (the
/// 2*restart+1 basis vectors) without exposing the ill-conditioned Stokes
/// operator to single precision — so convergence matches the full-precision
/// solver while the basis storage is roughly halved.
///
/// @tparam OperatorT      Operator type (OperatorLike); runs in full precision.
/// @tparam BasisVectorType Storage type for the Krylov basis (e.g. float block vector).
/// @tparam PreconditionerT Preconditioner (SolverLike); runs in full precision.
template < OperatorLike OperatorT, typename BasisVectorType, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class FGMRESLowMem
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;
    using ScalarType         = typename SolutionVectorType::ScalarType;

    /// @param work  Full-precision scratch vectors. Must contain at least 4:
    ///              [0]=r (residual), [1]=w (=A*z_j), [2]=v_scratch, [3]=z_scratch.
    ///              v_scratch/z_scratch must be distinct (the preconditioner solve
    ///              z = M^{-1} v reads v while writing z).
    /// @param basis Reduced-precision basis vectors. Must contain at least
    ///              2*restart + 1: V_0..V_restart and Z_0..Z_{restart-1}.
    FGMRESLowMem(
        const std::vector< SolutionVectorType >& work,
        const std::vector< BasisVectorType >&    basis,
        const FGMRESOptions< ScalarType >&       options        = {},
        const std::shared_ptr< util::Table >&    statistics     = nullptr,
        const PreconditionerT                    preconditioner = IdentitySolver< OperatorT >() )
    : tag_( "fgmres_solver" )
    , work_( work )
    , basis_( basis )
    , options_( options )
    , statistics_( statistics )
    , preconditioner_( preconditioner )
    , skip_preconditioner_in_case_of_nan_or_infs_( true )
    {}

    void set_tag( const std::string& tag ) { tag_ = tag; }
    void set_restart( int m ) { options_.restart = std::max( 1, m ); }

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        util::Timer timer_fgmres_solve( "fgmres_solve" );

        auto& r      = work_[0];
        auto& w      = work_[1];
        auto& v_work = work_[2]; // double scratch for a basis vector in arithmetic
        auto& z_work = work_[3]; // double scratch for the current preconditioned direction

        {
            util::Timer t_mv( "fgmres_matvec" );
            apply( A, x, r );
        }
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

        const int m_req = options_.restart;

        if ( static_cast< int >( work_.size() ) < 4 )
        {
            std::cerr << "FGMRESLowMem: need >= 4 work vectors, got " << work_.size() << std::endl;
            Kokkos::abort( "FGMRESLowMem: insufficient work vectors" );
        }
        if ( static_cast< int >( basis_.size() ) < 2 * m_req + 1 )
        {
            std::cerr << "FGMRESLowMem: need >= 2*restart+1 = " << ( 2 * m_req + 1 ) << " basis vectors, got "
                      << basis_.size() << std::endl;
            Kokkos::abort( "FGMRESLowMem: insufficient basis vectors" );
        }

        const int offV = 0;             // V_0 .. V_m   in basis_[0 .. m]
        const int offZ = offV + m_req + 1; // Z_0 .. Z_{m-1} in basis_[m+1 .. 2m]

        Eigen::Matrix< ScalarType, Eigen::Dynamic, Eigen::Dynamic > H( m_req + 1, m_req );
        Eigen::Matrix< ScalarType, Eigen::Dynamic, 1 >              g( m_req + 1 );
        std::vector< ScalarType >                                   cs( m_req + 1, 0 ), sn( m_req + 1, 0 );

        int total_iters = 0;

        while ( total_iters < options_.max_iterations )
        {
            // V_0 = r / ||r||   (compute in double, store as reduced precision)
            lincomb( v_work, { 1.0 / beta0 }, { r } );
            convert( v_work, basis_[offV + 0] );

            H.setZero();
            g.setZero();
            std::fill( cs.begin(), cs.end(), ScalarType( 0 ) );
            std::fill( sn.begin(), sn.end(), ScalarType( 0 ) );
            g( 0 ) = beta0;

            int inner_its = 0;

            for ( int j = 0; j < m_req && total_iters < options_.max_iterations; ++j, ++total_iters )
            {
                // z_j = M^{-1} v_j   (v_j up-converted to double scratch)
                convert( basis_[offV + j], v_work );
                assign( z_work, 0 );
                {
                    util::Timer t_pc( "fgmres_precondition" );
                    solve( preconditioner_, A, z_work, v_work );
                }

                if ( skip_preconditioner_in_case_of_nan_or_infs_ && has_nan_or_inf( z_work ) )
                {
                    util::logroot << "FGMRESLowMem: preconditioner produced NaN/Inf; skipping it this iteration. "
                                     "(total_iters = "
                                  << total_iters << ", j = " << j << ")" << std::endl;
                    assign( z_work, v_work );
                }

                // store Z_j in reduced precision
                convert( z_work, basis_[offZ + j] );

                // w = A z_j   (z_work still holds z_j in double)
                {
                    util::Timer t_mv( "fgmres_matvec" );
                    apply( A, z_work, w );
                }

                ScalarType h_jp1j;
                {
                    util::Timer t_gs( "fgmres_gram_schmidt" );
                    for ( int i = 0; i <= j; ++i )
                    {
                        convert( basis_[offV + i], v_work ); // V_i up to double
                        const ScalarType hij = dot( w, v_work );
                        H( i, j )            = hij;
                        lincomb( w, { 1.0, -hij }, { w, v_work } );
                    }
                    h_jp1j = std::sqrt( dot( w, w ) );
                }

                if ( h_jp1j < std::numeric_limits< ScalarType >::epsilon() * initial_residual )
                {
                    H( j + 1, j ) = 0;
                    inner_its     = j + 1;
                    util::logroot << "FGMRESLowMem: Arnoldi breakdown. Restarting. (total_iters = " << total_iters
                                  << ", j = " << j << ")" << std::endl;
                    break;
                }

                H( j + 1, j ) = h_jp1j;

                if ( h_jp1j > ScalarType( 0 ) )
                {
                    util::Timer t_gs( "fgmres_gram_schmidt" );
                    lincomb( v_work, { 1.0 / h_jp1j }, { w } ); // v_{j+1} in double
                    convert( v_work, basis_[offV + j + 1] );    // store reduced precision
                }

                util::Timer t_hess( "fgmres_hessenberg" );

                for ( int i = 0; i < j; ++i )
                {
                    const ScalarType temp = cs[i] * H( i, j ) + sn[i] * H( i + 1, j );
                    H( i + 1, j )         = -sn[i] * H( i, j ) + cs[i] * H( i + 1, j );
                    H( i, j )             = temp;
                }

                {
                    const ScalarType a    = H( j, j );
                    const ScalarType bb   = H( j + 1, j );
                    const ScalarType r_ab = std::hypot( a, bb );
                    cs[j]                 = ( r_ab == ScalarType( 0 ) ) ? ScalarType( 1 ) : a / r_ab;
                    sn[j]                 = ( r_ab == ScalarType( 0 ) ) ? ScalarType( 0 ) : bb / r_ab;
                    H( j, j )             = r_ab;
                    H( j + 1, j )         = 0;

                    if ( std::abs( H( j, j ) ) < std::numeric_limits< ScalarType >::epsilon() )
                    {
                        inner_its = j;
                        util::logroot << "FGMRESLowMem: Stagnation after Givens. Restarting. (total_iters = "
                                      << total_iters << ", j = " << j << ")" << std::endl;
                        break;
                    }

                    const ScalarType gj  = g( j );
                    const ScalarType gj1 = g( j + 1 );
                    g( j )               = cs[j] * gj + sn[j] * gj1;
                    g( j + 1 )           = -sn[j] * gj + cs[j] * gj1;
                }

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

                if ( !std::isfinite( abs_res ) )
                {
                    util::logroot << "FGMRESLowMem: NaN/Inf in residual estimate; returning current solution. "
                                     "(total_iters = "
                                  << total_iters << ", j = " << j << ")" << std::endl;
                    inner_its = j;
                    ++total_iters;
                    break;
                }

                inner_its = j + 1;

                if ( rel_res <= options_.relative_residual_tolerance ||
                     abs_res < options_.absolute_residual_tolerance )
                {
                    break;
                }
            }

            Eigen::Matrix< ScalarType, Eigen::Dynamic, 1 > y( inner_its );
            y.setZero();
            {
                util::Timer t_bs( "fgmres_hessenberg" );
                for ( int row = inner_its - 1; row >= 0; --row )
                {
                    ScalarType sum = g( row );
                    for ( int col = row + 1; col < inner_its; ++col )
                        sum -= H( row, col ) * y( col );
                    y( row ) = sum / H( row, row );
                }
            }

            // x += sum_i y_i Z_i   (Z_i down-converted from reduced precision,
            // accumulated straight into x -- no separate accumulator needed)
            {
                util::Timer t_upd( "fgmres_restart_update" );
                for ( int i = 0; i < inner_its; ++i )
                {
                    convert( basis_[offZ + i], z_work );
                    lincomb( x, { 1.0, y( i ) }, { x, z_work } );
                }
            }

            {
                util::Timer t_mv( "fgmres_matvec" );
                apply( A, x, r );
            }
            lincomb( r, { 1.0, -1.0 }, { b, r } );
            beta0 = std::sqrt( dot( r, r ) );

            if ( !std::isfinite( beta0 ) )
            {
                util::logroot << "FGMRESLowMem: Residual NaN/Inf after restart. Aborting. (beta0 = " << beta0
                              << ", total_iters = " << total_iters << ")" << std::endl;
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
    std::string                       tag_;
    std::vector< SolutionVectorType > work_;  ///< full-precision scratch (>= 5)
    std::vector< BasisVectorType >    basis_; ///< reduced-precision Krylov basis (>= 2*restart+1)
    FGMRESOptions< ScalarType >       options_;
    std::shared_ptr< util::Table >    statistics_;
    PreconditionerT                   preconditioner_;
    bool                              skip_preconditioner_in_case_of_nan_or_infs_;
};

} // namespace terra::linalg::solvers
