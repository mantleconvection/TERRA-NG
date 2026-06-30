#pragma once

#include <memory>
#include <vector>

#include "grid/bit_masks.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/iterative_solver_info.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/table.hpp"

#include "wbfbt_pressure_poisson.hpp"

namespace terra::mantlecirculation {

/// @brief Literal composed operator K_w = B C_w^-1 B^T on the pressure space.
///
/// `apply_impl(p, q)` computes q = B (C_w^-1 (B^T p)) using a single velocity-
/// space temporary.  B and B^T are caller-provided gradient/divergence
/// operators; for K_w we pass in versions constructed with Dirichlet=0 BCs on
/// all physical boundaries (regardless of the outer Stokes BC type), so the
/// operators themselves enforce a symmetric BC pair and K_w is symmetric.
///
/// C_w^-1 is a velocity-space inverse diagonal (either 1/diag(A) or the lumped
/// inverse mass) supplied by the caller as a shallow Kokkos view, so external
/// viscosity refreshes propagate without action here.
template < typename ScalarType, linalg::OperatorLike GradientOp, linalg::OperatorLike DivergenceOp >
class ExplicitKwOperator
{
  public:
    using PressureVector = linalg::VectorQ1Scalar< ScalarType >;
    using VelocityVector = linalg::SrcOf< DivergenceOp >;

    using SrcVectorType = PressureVector;
    using DstVectorType = PressureVector;

    ExplicitKwOperator(
        GradientOp            gradient,
        DivergenceOp          divergence,
        const VelocityVector& c_w_inv_diag,
        VelocityVector        tmp_velocity )
    : B_T_( std::move( gradient ) )
    , B_( std::move( divergence ) )
    , c_w_inv_diag_( c_w_inv_diag )
    , tmp_v_( std::move( tmp_velocity ) )
    {}

    void apply_impl( const PressureVector& src, PressureVector& dst )
    {
        linalg::apply( B_T_, src, tmp_v_ );
        linalg::scale_in_place( tmp_v_, c_w_inv_diag_ );
        linalg::apply( B_, tmp_v_, dst );
    }

  private:
    GradientOp     B_T_;
    DivergenceOp   B_;
    VelocityVector c_w_inv_diag_;
    VelocityVector tmp_v_;
};

/// @brief K_w solver: invert the literal K_w = B C_w^-1 B^T via FGMRES with
///        identity preconditioner.  No rediscretization, no MG hierarchy.
template < typename ScalarType, linalg::OperatorLike GradientOp, linalg::OperatorLike DivergenceOp >
class ExplicitKwPressurePoissonSolver final : public WBFBTPressurePoissonSolver< ScalarType >
{
  public:
    using PressureVector = linalg::VectorQ1Scalar< ScalarType >;
    using VelocityVector = linalg::SrcOf< DivergenceOp >;
    using KwOp           = ExplicitKwOperator< ScalarType, GradientOp, DivergenceOp >;
    using KwFGMRES       = linalg::solvers::FGMRES< KwOp >;

    static constexpr int kFGMRESRestart = 30;

    ExplicitKwPressurePoissonSolver(
        GradientOp                                               gradient,
        DivergenceOp                                             divergence,
        const VelocityVector&                                    c_w_inv_diag,
        const grid::shell::DistributedDomain&                    velocity_domain,
        const grid::shell::DistributedDomain&                    pressure_domain,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& velocity_ownership_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& pressure_ownership_mask,
        int                                                      max_iterations,
        ScalarType                                               relative_tol,
        std::shared_ptr< util::Table >                           table )
    : pressure_ownership_mask_( pressure_ownership_mask )
    , rhs_proj_( "explicit_kw_rhs_proj", pressure_domain, pressure_ownership_mask )
    , tmps_( make_fgmres_tmps_( pressure_domain, pressure_ownership_mask ) )
    , kw_op_(
          std::move( gradient ),
          std::move( divergence ),
          c_w_inv_diag,
          VelocityVector( "explicit_kw_tmp_v", velocity_domain, velocity_ownership_mask ) )
    , fgmres_(
          tmps_,
          linalg::solvers::FGMRESOptions< ScalarType >{
              kFGMRESRestart,
              relative_tol,
              static_cast< ScalarType >( 1e-30 ),
              max_iterations },
          table ? table : std::make_shared< util::Table >() )
    , table_( table )
    {
        fgmres_.set_tag( "wbfbt_kw_fgmres" );

        num_dofs_finest_ = kernels::common::count_masked< long >(
            pressure_ownership_mask, grid::NodeOwnershipFlag::OWNED );
    }

    void solve( const PressureVector& rhs, PressureVector& sol ) override
    {
        linalg::assign( rhs_proj_, rhs );
        project_out_constant_( rhs_proj_ );

        linalg::assign( sol, ScalarType( 0 ) );

        fgmres_.solve_impl( kw_op_, sol, rhs_proj_ );

        project_out_constant_( sol );
    }

    void refresh( const PressureVector& /*sqrt_eta_pressure_finest*/ ) override {}

  private:
    static std::vector< PressureVector > make_fgmres_tmps_(
        const grid::shell::DistributedDomain&                    pressure_domain,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& pressure_ownership_mask )
    {
        constexpr int kNTmps = 2 * kFGMRESRestart + 4;
        std::vector< PressureVector > v;
        v.reserve( kNTmps );
        for ( int i = 0; i < kNTmps; ++i )
        {
            v.emplace_back(
                "explicit_kw_fgmres_tmp_" + std::to_string( i ),
                pressure_domain,
                pressure_ownership_mask );
        }
        return v;
    }

    void project_out_constant_( PressureVector& v )
    {
        const ScalarType s    = kernels::common::masked_sum(
            v.grid_data(), v.mask_data(), grid::NodeOwnershipFlag::OWNED );
        const ScalarType mean = s / static_cast< ScalarType >( num_dofs_finest_ );
        linalg::lincomb( v, { ScalarType( 1 ) }, { v }, -mean );
    }

    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > pressure_ownership_mask_;
    long                                              num_dofs_finest_{ 0 };

    PressureVector                rhs_proj_;
    std::vector< PressureVector > tmps_;
    KwOp                          kw_op_;
    KwFGMRES                      fgmres_;

    std::shared_ptr< util::Table > table_;
};

} // namespace terra::mantlecirculation
