#pragma once

#include <memory>
#include <vector>

#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "linalg/diagonally_scaled_operator.hpp"
#include "linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/power_iteration.hpp"
#include "linalg/solvers/velocity_prec_handle.hpp"
#include "linalg/vector_q1.hpp"

#include "parameters.hpp"

namespace terra::mantlecirculation {

/// @brief A velocity-block multigrid V-cycle whose entire hierarchy (operators,
/// smoothers, coarse solver, transfers, level vectors, viscosity, coordinates) is
/// stored and worked in @p MGScalar, exposed to the double block-triangular
/// preconditioner via the type-erased VelocityPrecHandle. The double residual is
/// converted down to MGScalar, the V-cycle runs entirely in MGScalar, and the
/// correction is converted back up.
///
/// Classical (non-agglomerated, gca==0) path only. The geometry (coordinates) is
/// stored in MGScalar too, so MGScalar must hold enough precision for the Jacobian
/// (float is fine at production refinements; half degrades the geometry).
template < typename MGScalar, typename ViscousDouble >
class LowPrecVCycle : public linalg::solvers::VelocityPrecHandle< ViscousDouble >::Impl
{
    // Two-scalar viscous operator: src/dst/coefficient + matvec I/O in MGScalar (low,
    // for bandwidth/storage), but stored coordinates in DOUBLE so the Jacobian stays
    // accurate. Avoids both the geometry tax (no float coord copies — the original
    // double coords are shared) and the low-precision cancellation in the Jacobian.
    using Viscous      = fe::wedge::operators::shell::EpsilonDivDivKerngen< MGScalar, 3, double >;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< MGScalar >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< MGScalar >;
    using Smoother     = linalg::solvers::Chebyshev< Viscous >;
    using CoarseSolver = linalg::solvers::PCG< Viscous >;
    using MG = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseSolver >;

    using VelVec = linalg::VectorQ1Vec< MGScalar, 3 >;

    using SrcVecD = linalg::SrcOf< ViscousDouble >;
    using DstVecD = linalg::DstOf< ViscousDouble >;

  public:
    LowPrecVCycle(
        const std::vector< std::shared_ptr< grid::shell::DistributedDomain > >& domains,
        const std::vector< grid::Grid3DDataVec< double, 3 > >&                   coords_shell,
        const std::vector< grid::Grid2DDataScalar< double > >&                  coords_radii,
        const std::vector< grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > >& boundary_mask,
        const std::vector< grid::Grid4DDataScalar< grid::NodeOwnershipFlag > >&        ownership_mask,
        const std::vector< linalg::VectorQ1Scalar< double > >&                  eta,
        grid::shell::BoundaryConditions&                                        bcs,
        const Parameters&                                                       prm,
        std::shared_ptr< util::Table >                                          table )
    : num_levels_( static_cast< int >( domains.size() ) )
    {
        util::logroot << "Setting up reduced-precision (" << ( sizeof( MGScalar ) ) << "-byte) MG V-cycle ..."
                      << std::endl;

        // --- Convert per-level viscosity coefficient to MGScalar ---
        // Coordinates are NOT converted: the two-scalar operator reads the original
        // double coords directly (shared, ref-counted — no duplication, no geometry
        // tax) and keeps the Jacobian in double.
        for ( int l = 0; l < num_levels_; ++l )
        {
            eta_.emplace_back( "mg_eta_" + std::to_string( l ), *domains[l], ownership_mask[l] );
            kernels::common::copy_convert( eta[l].grid_data(), eta_.back().grid_data() );
        }

        // --- Operators: MGScalar storage/work, double coords (re-discretised per level) ---
        for ( int l = 0; l < num_levels_; ++l )
        {
            A_.emplace_back(
                *domains[l], coords_shell[l], coords_radii[l], boundary_mask[l], eta_[l].grid_data(), bcs, false );
        }
        for ( int l = 0; l < num_levels_ - 1; ++l )
        {
            P_.emplace_back( linalg::OperatorApplyMode::Add );
            R_.emplace_back( *domains[l] );
        }

        // --- Coarse operators view (levels 0 .. num-2) for the MG ---
        // A_ holds every level; the MG takes the coarse ones, finest is passed at solve.

        // --- Tmp vectors ---
        for ( int l = 0; l < num_levels_; ++l )
        {
            tmp_mg_.emplace_back( "mg_tmp_" + std::to_string( l ), *domains[l], ownership_mask[l] );
            tmp_mg_2_.emplace_back( "mg_tmp2_" + std::to_string( l ), *domains[l], ownership_mask[l] );
            if ( l < num_levels_ - 1 )
            {
                tmp_mg_r_.emplace_back( "mg_tmp_r_" + std::to_string( l ), *domains[l], ownership_mask[l] );
                tmp_mg_e_.emplace_back( "mg_tmp_e_" + std::to_string( l ), *domains[l], ownership_mask[l] );
            }
        }

        // --- Inverse diagonals + Chebyshev smoothers ---
        for ( int l = 0; l < num_levels_; ++l )
        {
            inverse_diagonals_.emplace_back( "mg_inv_diag_" + std::to_string( l ), *domains[l], ownership_mask[l] );
            VelVec ones( "mg_inv_diag_tmp_" + std::to_string( l ), *domains[l], ownership_mask[l] );
            linalg::assign( ones, MGScalar( 1 ) );
            A_[l].set_diagonal( true );
            linalg::apply( A_[l], ones, inverse_diagonals_.back() );
            A_[l].set_diagonal( false );
            linalg::invert_entries( inverse_diagonals_.back() );
        }
        for ( int l = 0; l < num_levels_; ++l )
        {
            std::vector< VelVec > smoother_tmps{ tmp_mg_[l], tmp_mg_2_[l] };
            smoothers_.emplace_back(
                prm.stokes_solver_parameters.viscous_pc_chebyshev_order,
                inverse_diagonals_[l],
                smoother_tmps,
                prm.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost,
                prm.stokes_solver_parameters.viscous_pc_num_power_iterations );
        }

        // --- Coarse grid solver (PCG on level 0) ---
        for ( int i = 0; i < 4; ++i )
            coarse_tmps_.emplace_back( "mg_coarse_tmp_" + std::to_string( i ), *domains[0], ownership_mask[0] );
        coarse_solver_ = std::make_unique< CoarseSolver >(
            linalg::solvers::IterativeSolverParameters{ 50, 1e-6, 1e-16 }, table, coarse_tmps_ );
        coarse_solver_->set_tag( "mg_coarse_pcg" );

        // --- The MG (classical V-cycle; coarse operators only) ---
        std::vector< Viscous > A_coarse;
        for ( int l = 0; l < num_levels_ - 1; ++l )
            A_coarse.push_back( A_[l] );

        mg_ = std::make_unique< MG >(
            P_, R_, A_coarse, tmp_mg_r_, tmp_mg_e_, tmp_mg_, smoothers_, smoothers_, *coarse_solver_,
            prm.stokes_solver_parameters.viscous_pc_num_vcycles, MGScalar( 1e-6 ) );

        // --- Boundary scratch (double <-> MGScalar) ---
        r_ = std::make_unique< VelVec >( "mg_r", *domains[num_levels_ - 1], ownership_mask[num_levels_ - 1] );
        z_ = std::make_unique< VelVec >( "mg_z", *domains[num_levels_ - 1], ownership_mask[num_levels_ - 1] );
    }

    /// Double interface: convert RHS down, run the MGScalar V-cycle, convert up.
    void solve_impl( ViscousDouble& /*A_double*/, SrcVecD& x, const DstVecD& b ) override
    {
        convert( b, *r_ );
        linalg::assign( *z_, MGScalar( 0 ) );
        linalg::solvers::solve( *mg_, A_[num_levels_ - 1], *z_, *r_ );
        convert( *z_, x );
    }

  private:
    int num_levels_;

    std::vector< linalg::VectorQ1Scalar< MGScalar > > eta_;

    std::vector< Viscous >      A_;
    std::vector< Prolongation > P_;
    std::vector< Restriction >  R_;

    std::vector< VelVec > tmp_mg_, tmp_mg_2_, tmp_mg_r_, tmp_mg_e_, coarse_tmps_;
    std::vector< VelVec > inverse_diagonals_;
    std::vector< Smoother > smoothers_;

    std::unique_ptr< CoarseSolver > coarse_solver_;
    std::unique_ptr< MG >           mg_;
    std::unique_ptr< VelVec >       r_, z_;
};

} // namespace terra::mantlecirculation
