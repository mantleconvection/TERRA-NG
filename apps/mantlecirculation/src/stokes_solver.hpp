#pragma once

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "build_radii.hpp"
#include "communication/shell/redistribute.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/strong_algebraic_freeslip_enforcement.hpp"
#include "fe/wedge/linearforms/shell/inv_rho_grad_rho_dot_u.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "geophysics/viscosity/viscosity_interpolation.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/agglomerated_distribution.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "hbm_probe.hpp"
#include "interpolators.hpp"
#include "kernels/common/grid_operations.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/diagonally_scaled_operator.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/diagonal_solver.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/fgmres_lowmem.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/power_iteration.hpp"
#include "linalg/solvers/velocity_prec_handle.hpp"
#include "linalg/vector_q1.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "low_prec_vcycle.hpp"
#include "mpi/level_comms.hpp"
#include "mpi/mpi.hpp"
#include "parameters.hpp"
#include "shell/spherical_harmonics.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

#include "mpi/level_comms.hpp"

#include "build_radii.hpp"
#include "hbm_probe.hpp"
#include "interpolators.hpp"
#include "low_prec_vcycle.hpp"
#ifdef TERRA_ENABLE_PYTHON
#include "ml/neural_solver.hpp"
#endif
#include "parameters.hpp"

namespace terra::mantlecirculation {

/// MG-level communicator + subdomain-to-rank ladder, derived once from the
/// agglomeration factors.  Built before any DistributedDomain because the
/// domain loop needs the per-level comm + subdomain distribution to put each
/// level on the correct sub-communicator.  StokesContext consumes the same
/// object so all per-level state (eta, A_c, smoothers, inverse_diagonals,
/// tmp_mg_*, coarse_grid_solver, redistribute_down) lives on the matching
/// comm without any knowledge of the ladder.
class MGAgglomeration
{
  public:
    explicit MGAgglomeration( const Parameters& prm, MPI_Comm world = MPI_COMM_WORLD )
    : num_mg_levels_(
          prm.mesh_parameters.refinement_level_mesh_max - prm.mesh_parameters.refinement_level_mesh_min + 1 )
    , factors_( prm.stokes_solver_parameters.viscous_pc_agglom_factors )
    , world_comm_( world )
    {
        // When the user didn't specify factors, leave factors_ empty so the
        // StokesContext skips the agglomeration code path entirely and uses
        // the classical multigrid (no Redistribute, no upper-comm meshes).
        // (This is a debugging short-circuit while we hunt the GPU memory
        // fault that appears when the all-1s agglom path runs.)
        if ( factors_.empty() )
            return;

        if ( static_cast< int >( factors_.size() ) != num_mg_levels_ - 1 )
        {
            throw std::runtime_error(
                "viscous_pc_agglom_factors length (" + std::to_string( factors_.size() ) +
                ") must equal num_mg_levels - 1 (" + std::to_string( num_mg_levels_ - 1 ) + ")" );
        }

        level_comms_ = mpi::build_level_comms( world, factors_ );
        cum_factors_.push_back( 1 );
        for ( int f : factors_ )
            cum_factors_.push_back( cum_factors_.back() * f );

        util::logroot << "MG agglomeration factors = {";
        for ( size_t i = 0; i < factors_.size(); ++i )
            util::logroot << ( i ? ", " : "" ) << factors_[i];
        util::logroot << "}" << std::endl;
    }

    int                       num_mg_levels() const { return num_mg_levels_; }
    const std::vector< int >& factors() const { return factors_; }

    /// Sub-comm for MG level L (0 = coarsest, num_mg_levels-1 = finest).
    /// Returns the world communicator when no agglomeration factors are set.
    MPI_Comm comm( int L ) const
    {
        if ( factors_.empty() )
            return world_comm_;
        return level_comms_[( num_mg_levels_ - 1 ) - L];
    }

    /// Cumulative agglomeration factor at level L (= world / comm-size at L).
    int cum_factor( int L ) const
    {
        if ( factors_.empty() )
            return 1;
        return cum_factors_[( num_mg_levels_ - 1 ) - L];
    }

    /// Subdomain → rank distribution function at level L, accounting for the
    /// cumulative agglomeration factor.
    grid::shell::SubdomainToRankDistributionFunction subdomain_fn( int L ) const
    {
        if ( factors_.empty() )
            return grid::shell::subdomain_to_rank_iterate_diamond_subdomains;

        const int cf = cum_factor( L );
        if ( cf == 1 )
            return grid::shell::subdomain_to_rank_iterate_diamond_subdomains;
        return grid::shell::agglomerated_subdomain_to_rank(
            grid::shell::subdomain_to_rank_iterate_diamond_subdomains, cf );
    }

  private:
    int                     num_mg_levels_;
    std::vector< int >      factors_;
    MPI_Comm                world_comm_;
    std::vector< MPI_Comm > level_comms_;
    std::vector< int >      cum_factors_;
};

/// All Stokes-system state: viscosity hierarchy, GCA elements, fine-/coarse-
/// level operators, multigrid V-cycle (with optional comm-aware
/// agglomeration), Schur preconditioner, and the outer FGMRES.  Owns the
/// Stokes block vectors `u`, `f`, and a temporary used during RHS assembly.
///
/// The constructor takes the same shared_ptr<domain> + const-ref deps style
/// as the energy solvers; the MG ladder helpers come in as `std::function`s
/// so this class doesn't need to know how mc.cpp built them.
template < typename ScalarType >
class StokesContext
{
    using Stokes           = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous          = typename Stokes::Block11Type;
    using Gradient         = typename Stokes::Block12Type;
    using ViscousMass      = fe::wedge::operators::shell::VectorMass< ScalarType >;
    using Prolongation     = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction      = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;
    using RestrictionScalar = fe::wedge::operators::shell::RestrictionConstant< ScalarType >;
    using PressureMass     = fe::wedge::operators::shell::KMass< ScalarType >;
    using Smoother         = linalg::solvers::Chebyshev< Viscous >;
    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;
    using VelGridData      = grid::Grid4DDataVec< ScalarType, 3 >;
    using Redistribute     = communication::shell::Redistribute< VelGridData >;
    using PrecVisc =
        linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver, Redistribute >;
    using PrecSchur = linalg::solvers::DiagonalSolver< PressureMass >;
    // The (1,1) velocity preconditioner is type-erased so its internal precision
    // (the MG V-cycle precision) can be chosen at runtime via --stokes-mg-precision.
    using VelPrecHandle = linalg::solvers::VelocityPrecHandle< Viscous >;
    using PrecStokes    = linalg::solvers::
        BlockTriangularPreconditioner2x2< Stokes, Viscous, PressureMass, Gradient, VelPrecHandle, PrecSchur >;
    // Outer solver: either the standard double FGMRES or the low-memory variant
    // that stores the Krylov basis in single precision (operator/preconditioner/
    // orthogonalization stay in ScalarType). Selected at runtime via
    // --stokes-float-krylov-basis.
    // FP16 Krylov-basis storage: native __half on HIP (Kokkos 4.6+), genuine 2 B/dof
    // (4x vs double). Validated to match the double residual curve to ~4 sig figs
    // with 0 NaN. The basis is store-only + convert (never operated on directly), so
    // no half arithmetic is instantiated; entries (~6e-5) are covered by FP16 denorms.
    // (BF16 would need Kokkos >= 5.1.0 and gives no benefit here -- fewer mantissa
    //  bits, and FP16's range proved sufficient.)
    using BasisVectorType = linalg::VectorQ1IsoQ2Q1< Kokkos::Experimental::bhalf_t, 3 >;
    using FGMRESDouble    = linalg::solvers::FGMRES< Stokes, PrecStokes >;

#ifdef TERRA_ENABLE_PYTHON
    /// Copyable view onto a NeuralSolver so it fits FGMRES's by-value
    /// preconditioner slot (NeuralSolver itself owns Python state and is
    /// non-copyable).
    struct NeuralPrecRef
    {
        using OperatorType = Stokes;
        ml::NeuralSolver< Stokes >* impl = nullptr;
        void solve_impl( OperatorType& A,
                         typename OperatorType::SrcVectorType& x,
                         const typename OperatorType::DstVectorType& b )
        {
            impl->solve_impl( A, x, b );
        }
    };
    using FGMRESNeural = linalg::solvers::FGMRES< Stokes, NeuralPrecRef >;
#endif
    using FGMRESFloat     = linalg::solvers::FGMRESLowMem< Stokes, BasisVectorType, PrecStokes >;

  public:
    StokesContext(
        const std::vector< std::shared_ptr< grid::shell::DistributedDomain > >&        domains,
        const std::vector< grid::Grid3DDataVec< ScalarType, 3 > >&                     coords_shell,
        const std::vector< grid::Grid2DDataScalar< ScalarType > >&                     coords_radii,
        const std::vector< grid::Grid4DDataScalar< grid::NodeOwnershipFlag > >&        ownership_mask,
        const std::vector< grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > >& boundary_mask,
        grid::shell::BoundaryConditions&                                               bcs,
        const MGAgglomeration&                                                         agglom,
        const Parameters&                                                              prm,
        std::shared_ptr< util::Table >                                                 table )
    : domains_( domains )
    , coords_shell_( coords_shell )
    , coords_radii_( coords_radii )
    , ownership_mask_( ownership_mask )
    , boundary_mask_( boundary_mask )
    , prm_( prm )
    , table_( std::move( table ) )
    , num_levels_( static_cast< int >( domains.size() ) )
    , velocity_level_( num_levels_ - 1 )
    , pressure_level_( num_levels_ - 2 )
    {
        // Element-wise copy of the BCs C-array (reference assignment is forbidden).
        bcs_[0]                    = bcs[0];
        bcs_[1]                    = bcs[1];
        const int   lat_sdr        = ( prm.mesh_parameters.lat_sdr >= 0 ) ? prm.mesh_parameters.lat_sdr :
                                                                            prm.mesh_parameters.refinement_level_subdomains;
        const int   rad_sdr        = ( prm.mesh_parameters.rad_sdr >= 0 ) ? prm.mesh_parameters.rad_sdr :
                                                                            prm.mesh_parameters.refinement_level_subdomains;
        const auto& agglom_factors = agglom.factors();
        using grid::shell::DistributedDomain;
        using linalg::VectorQ1IsoQ2Q1;
        using linalg::VectorQ1Scalar;
        using linalg::VectorQ1Vec;
        using util::logroot;

        num_dofs_pressure_ =
            kernels::common::count_masked< long >( ownership_mask_[pressure_level_], grid::NodeOwnershipFlag::OWNED );

        // ---------------- Stokes block vectors ----------------
        // "tmp" was a dedicated full Stokes vector used only as RHS-assembly
        // scratch; we reuse the block preconditioner's triangular_prec_tmp_
        // (idle until the solve) instead, saving one velocity-sized vector.
        // "u_prev" is required by the compressible (TALA) RHS: the (1/rho)grad(rho).u
        // continuity term is lagged on the previous velocity iterate.
        // "u_prev2..4": deeper solution history for the polynomial time-extrapolation
        // initial guess (--stokes-guess-extrap).
        std::vector< std::string > stok_vec_names = { "u", "f", "u_prev", "u_prev2", "u_prev3", "u_prev4", "u_guess" };
        for ( const auto& name : stok_vec_names )
        {
            stok_vecs_[name] = VectorQ1IsoQ2Q1< ScalarType >(
                name,
                *domains_[velocity_level_],
                *domains_[pressure_level_],
                ownership_mask_[velocity_level_],
                ownership_mask_[pressure_level_] );
        }

        // Density is owned by the caller now (upstream's layout) and passed into
        // solve(); it is initialised there, not here.

        // ---------------- Viscosity ----------------
        // Radial profile (constant or CSV-driven), then projected into Q1 on every level.
        std::vector< grid::Grid2DDataScalar< ScalarType > > radial_viscosity_profile;
        radial_viscosity_profile.reserve( num_levels_ );
        if ( prm_.physics_parameters.viscosity_parameters.viscosity_profile_csv_path.empty() )
        {
            logroot << "Using constant viscosity profile." << std::endl;
            for ( int level = 0; level < num_levels_; level++ )
            {
                radial_viscosity_profile.push_back(
                    shell::interpolate_constant_radial_profile( coords_radii_[level], ScalarType( 1 ) ) );
            }
        }
        else
        {
            logroot << "Using radially varying viscosity profile." << std::endl;
            for ( int level = 0; level < num_levels_; level++ )
            {
                radial_viscosity_profile.push_back( shell::interpolate_radial_profile_into_subdomains_from_csv(
                    prm_.physics_parameters.viscosity_parameters.viscosity_profile_csv_path,
                    prm_.physics_parameters.radial_profiles_radii_key,
                    prm_.physics_parameters.viscosity_parameters.viscosity_profile_value_key,
                    coords_radii_[level],
                    ScalarType( 1 ) / prm_.mesh_parameters.mantle_thickness_m,
                    ScalarType( 1 ) / prm_.physics_parameters.viscosity_parameters.reference_viscosity ) );
            }
        }

        eta_.reserve( num_levels_ );
        for ( int level = 0; level < num_levels_; level++ )
        {
            const std::string name = ( level == num_levels_ - 1 ) ?
                                         std::string( "eta" ) :
                                         std::string( "eta_level_" ) + std::to_string( level );
            eta_.emplace_back( name, *domains_[level], ownership_mask_[level] );
        }
        for ( int level = 0; level < num_levels_; level++ )
        {
            // GCA still needs an approximation of viscosity on coarse grids
            // for the weighting of the mass matrix.
            geophysics::viscosity::RadialProfileViscosityInterpolator viscosity_interpolator(
                radial_viscosity_profile[level],
                ScalarType( 1 ),
                prm_.physics_parameters.viscosity_parameters.min_viscosity,
                prm_.physics_parameters.viscosity_parameters.max_viscosity );
            viscosity_interpolator.interpolate( eta_[level].grid_data() );
        }

        // Assign radial_viscosity_profile at highest level to class member eta_profile_, since we need the reference profile for viscosity laws
        eta_profile_ = radial_viscosity_profile[velocity_level_];

        // ---------------- GCA element selection ----------------
        GCAElements_  = VectorQ1Scalar< ScalarType >( "GCAElements", *domains_[0], ownership_mask_[0] );
        const int gca = prm_.stokes_solver_parameters.gca;
        if ( gca > 0 && std::any_of( agglom_factors.begin(), agglom_factors.end(), []( int f ) { return f > 1; } ) )
        {
            throw std::runtime_error(
                "MG agglomeration (--stokes-viscous-pc-agglom-factors) is not yet compatible with GCA (gca > 0). "
                "TwoGridGCA's element-matrix transfer between consecutive MG levels currently assumes both levels "
                "share a communicator, which is violated when the coarse level has been agglomerated onto a sub-comm. "
                "Either disable agglomeration or keep gca = 0 until the GCA assembly is extended to bridge "
                "sub-comms via Redistribute." );
        }
        if ( gca == 2 )
        {
            linalg::assign( GCAElements_, 0 );
            logroot << "Adaptive GCA: determining GCA elements on level " << velocity_level_ << std::endl;
            terra::linalg::solvers::GCAElementsCollector< ScalarType >(
                *domains_[velocity_level_],
                eta_[velocity_level_].grid_data(),
                velocity_level_,
                GCAElements_.grid_data() );
        }
        else if ( gca == 1 )
        {
            logroot << "GCA on all elements " << std::endl;
            assign( GCAElements_, 1 );
        }

        // FGMRES workspace selector. The workspace (Krylov basis + scratch) is the
        // single largest allocation, so we DEFER allocating it until just before the
        // FGMRES objects are built (after the MG hierarchy + the Chebyshev eigenvalue
        // estimate). Otherwise its ~20+ GB coincides with the estimate's transient
        // power-iteration temps and inflates the setup-time HBM peak.
        use_float_basis_ = prm_.stokes_solver_parameters.float_krylov_basis;

        // ---------------- Multigrid tmp vectors ----------------
        for ( int level = 0; level < num_levels_; level++ )
        {
            tmp_mg_.emplace_back( "tmp_mg_" + std::to_string( level ), *domains_[level], ownership_mask_[level] );
            tmp_mg_2_.emplace_back( "tmp_mg_2_" + std::to_string( level ), *domains_[level], ownership_mask_[level] );
            if ( level < num_levels_ - 1 )
            {
                tmp_mg_r_.emplace_back(
                    "tmp_mg_r_" + std::to_string( level ), *domains_[level], ownership_mask_[level] );
                tmp_mg_e_.emplace_back(
                    "tmp_mg_e_" + std::to_string( level ), *domains_[level], ownership_mask_[level] );
            }
        }

        // ---------------- Stokes operators ----------------
        grid::shell::BoundaryConditions bcs_neumann = {
            { grid::shell::ShellBoundaryFlag::CMB, grid::shell::BoundaryConditionFlag::NEUMANN },
            { grid::shell::ShellBoundaryFlag::SURFACE, grid::shell::BoundaryConditionFlag::NEUMANN },
        };

        K_ = std::make_unique< Stokes >(
            *domains_[velocity_level_],
            *domains_[pressure_level_],
            coords_shell_[velocity_level_],
            coords_radii_[velocity_level_],
            boundary_mask_[velocity_level_],
            eta_[velocity_level_].grid_data(),
            bcs_,
            false );
        K_->block_11().set_penalty_epsilon(
            static_cast< ScalarType >( prm_.stokes_solver_parameters.penalty_epsilon ) );

        K_neumann_ = std::make_unique< Stokes >(
            *domains_[velocity_level_],
            *domains_[pressure_level_],
            coords_shell_[velocity_level_],
            coords_radii_[velocity_level_],
            boundary_mask_[velocity_level_],
            eta_[velocity_level_].grid_data(),
            bcs_neumann,
            false );

        // Diagonal-only twin of K_neumann_, needed to lift an inhomogeneous Dirichlet
        // velocity (plate velocities) onto the right-hand side.
        K_neumann_diag_ = std::make_unique< Stokes >(
            *domains_[velocity_level_],
            *domains_[pressure_level_],
            coords_shell_[velocity_level_],
            coords_radii_[velocity_level_],
            boundary_mask_[velocity_level_],
            eta_[velocity_level_].grid_data(),
            bcs_neumann,
            true );

        M_ = std::make_unique< ViscousMass >(
            *domains_[velocity_level_], coords_shell_[velocity_level_], coords_radii_[velocity_level_], false );

        // The double-precision velocity multigrid (coarse operators, GCA, smoothers,
        // coarse solver, prec_11_) is only built when --stokes-mg-precision=double.
        // For float, LowPrecVCycle builds its own hierarchy, so this is skipped and
        // its memory is never allocated.
        const bool build_double_mg = ( prm_.stokes_solver_parameters.mg_precision == MGPrecision::DOUBLE );
        if ( build_double_mg )
        {
            // ---------------- Coarse grid operators / transfer ----------------
            logroot << "Setting up Stokes solver and preconditioners ..." << std::endl;

            for ( int level = 0; level < num_levels_ - 1; level++ )
            {
                A_c_.emplace_back(
                    *domains_[level],
                    coords_shell_[level],
                    coords_radii_[level],
                    boundary_mask_[level],
                    eta_[level].grid_data(),
                    bcs_,
                    false );
                // The free-slip rotation penalty must match the fine operator on every
                // coarse level, or the Chebyshev spectrum estimate is polluted there.
                A_c_.back().set_penalty_epsilon(
                    static_cast< ScalarType >( prm_.stokes_solver_parameters.penalty_epsilon ) );
                if ( gca == 2 )
                {
                    A_c_.back().set_stored_matrix_mode(
                        linalg::OperatorStoredMatrixMode::Selective, level, GCAElements_.grid_data() );
                }
                else if ( gca == 1 )
                {
                    A_c_.back().set_stored_matrix_mode(
                        linalg::OperatorStoredMatrixMode::Full, level, GCAElements_.grid_data() );
                }
                P_.emplace_back( linalg::OperatorApplyMode::Add );
                R_.emplace_back( *domains_[level] );
            }

            // GCA assembly (top-down)
            if ( gca > 0 )
            {
                for ( int level = num_levels_ - 2; level >= 0; level-- )
                {
                    logroot << "Assembling GCA on level " << prm_.mesh_parameters.refinement_level_mesh_min + level
                            << std::endl;
                    linalg::solvers::TwoGridGCA< ScalarType, Viscous >(
                        ( level == num_levels_ - 2 ) ? K_neumann_->block_11() : A_c_[level + 1],
                        A_c_[level],
                        level,
                        GCAElements_.grid_data() );
                }
            }

            // ---------------- Inverse diagonals ----------------
            for ( int level = 0; level < num_levels_; level++ )
            {
                inverse_diagonals_.emplace_back(
                    "inverse_diagonal_" + std::to_string( level ), *domains_[level], ownership_mask_[level] );

                if ( domains_[level]->comm() == MPI_COMM_NULL )
                    continue;

                VectorQ1Vec< ScalarType > tmp(
                    "inverse_diagonal_tmp" + std::to_string( level ), *domains_[level], ownership_mask_[level] );
                linalg::assign( tmp, ScalarType( 1 ) );
                if ( level == num_levels_ - 1 )
                {
                    K_->block_11().set_diagonal( true );
                    linalg::apply( K_->block_11(), tmp, inverse_diagonals_.back() );
                    K_->block_11().set_diagonal( false );
                }
                else
                {
                    A_c_[level].set_diagonal( true );
                    linalg::apply( A_c_[level], tmp, inverse_diagonals_.back() );
                    A_c_[level].set_diagonal( false );
                }
                linalg::invert_entries( inverse_diagonals_.back() );
            }

            // ---------------- Smoothers (Chebyshev) ----------------
            logroot << "Setting up multigrid smoother ..." << std::endl;
            smoothers_.reserve( num_levels_ );
            for ( int level = 0; level < num_levels_; level++ )
            {
                std::vector< VectorQ1Vec< ScalarType > > smoother_tmps;
                smoother_tmps.push_back( tmp_mg_[level] );
                smoother_tmps.push_back( tmp_mg_2_[level] );

                smoothers_.emplace_back(
                    prm_.stokes_solver_parameters.viscous_pc_chebyshev_order,
                    inverse_diagonals_[level],
                    smoother_tmps,
                    prm_.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost,
                    prm_.stokes_solver_parameters.viscous_pc_num_power_iterations );
            }

            // Diagnostic: estimate Chebyshev spectrum per level (mirrors the
            // estimate Chebyshev does internally on first solve).
            if ( prm_.devel_parameters.extended_diagnostics )
            {
                for ( int level = 0; level < num_levels_; level++ )
                {
                    if ( domains_[level]->comm() == MPI_COMM_NULL )
                        continue;

                    VectorQ1Vec< ScalarType > tmp_pi_it( "cheby_est_tmpIt", *domains_[level], ownership_mask_[level] );
                    VectorQ1Vec< ScalarType > tmp_pi_aux(
                        "cheby_est_tmpAux", *domains_[level], ownership_mask_[level] );
                    const auto log_level = prm_.mesh_parameters.refinement_level_mesh_min + level;
                    auto&      A_lvl     = ( level == num_levels_ - 1 ) ? K_->block_11() : A_c_[level];
                    linalg::DiagonallyScaledOperator< Viscous > inv_diag_A( A_lvl, inverse_diagonals_[level] );
                    const double                                lmax_est = linalg::solvers::power_iteration(
                        inv_diag_A,
                        tmp_pi_it,
                        tmp_pi_aux,
                        prm_.stokes_solver_parameters.viscous_pc_num_power_iterations );
                    logroot << "[Cheby estimate] level " << log_level << ": lambda_max(D^-1 A_viscous) ~ " << lmax_est
                            << "  => lambda_max_cheby = " << 1.5 * lmax_est << ", lambda_min_cheby = " << 0.1 * lmax_est
                            << std::endl;
                }
            }

            // ---------------- Coarse grid solver ----------------
            logroot << "Setting up multigrid coarse grid solver ..." << std::endl;
            coarse_grid_tmps_.reserve( 4 );
            for ( int i = 0; i < 4; i++ )
            {
                coarse_grid_tmps_.emplace_back( "tmp_coarse_grid", *domains_[0], ownership_mask_[0] );
            }
            coarse_grid_solver_ = std::make_unique< CoarseGridSolver >(
                linalg::solvers::IterativeSolverParameters{ 50, 1e-6, 1e-16 }, table_, coarse_grid_tmps_ );
            coarse_grid_solver_->set_tag( "coarse_grid_pcg" );

            // ---------------- Multigrid preconditioner (with optional agglomeration) ----------------
            logroot << "Setting up multigrid preconditioner ..." << std::endl;

            const int num_mg_levels = num_levels_;

            std::vector< Redistribute >              redistribute_down;
            std::vector< VectorQ1Vec< ScalarType > > tmp_mg_r_fine;
            std::vector< VectorQ1Vec< ScalarType > > tmp_mg_e_fine;

            if ( !agglom_factors.empty() )
            {
                redistribute_down.reserve( num_mg_levels - 1 );
                tmp_mg_r_fine.reserve( num_mg_levels - 1 );
                tmp_mg_e_fine.reserve( num_mg_levels - 1 );
                domains_upper_.reserve( num_mg_levels - 1 );
                mask_upper_.reserve( num_mg_levels - 1 );

                const auto orig_subdomain_to_rank = grid::shell::subdomain_to_rank_iterate_diamond_subdomains;

                for ( int L = 0; L < num_mg_levels - 1; ++L )
                {
                    const int lat_level = prm_.mesh_parameters.refinement_level_mesh_min + L;
                    const int rad_level = lat_level + prm_.mesh_parameters.radial_extra_levels;

                    const MPI_Comm upper_comm    = agglom.comm( L + 1 );
                    const int      upper_cf      = agglom.cum_factor( L + 1 );
                    const bool     same_as_lower = ( upper_comm == agglom.comm( L ) );

                    if ( same_as_lower )
                    {
                        domains_upper_.push_back( domains_[L] );
                        mask_upper_.push_back( ownership_mask_[L] );
                    }
                    else
                    {
                        DistributedDomain dom_up = DistributedDomain::create_uniform_on_comm(
                            upper_comm,
                            lat_level,
                            build_shell_radii< double >( prm_.mesh_parameters, ( 1 << rad_level ) + 1 ),
                            lat_sdr,
                            rad_sdr,
                            ( upper_cf == 1 ) ?
                                orig_subdomain_to_rank :
                                grid::shell::agglomerated_subdomain_to_rank( orig_subdomain_to_rank, upper_cf ) );
                        mask_upper_.push_back( grid::setup_node_ownership_mask_data( dom_up ) );
                        domains_upper_.push_back( std::make_shared< DistributedDomain >( std::move( dom_up ) ) );
                    }

                    tmp_mg_r_fine.emplace_back(
                        "tmp_r_fine_L" + std::to_string( L ), *domains_upper_.back(), mask_upper_.back() );
                    tmp_mg_e_fine.emplace_back(
                        "tmp_e_fine_L" + std::to_string( L ), *domains_upper_.back(), mask_upper_.back() );

                    redistribute_down.emplace_back(
                        *domains_upper_.back(),
                        *domains_[L],
                        ( upper_cf == 1 ) ?
                            orig_subdomain_to_rank :
                            grid::shell::agglomerated_subdomain_to_rank( orig_subdomain_to_rank, upper_cf ),
                        agglom.subdomain_fn( L ) );
                }

                // Restrictions are halo'd on the upper comm under agglomeration.
                R_.clear();
                R_.reserve( num_mg_levels - 1 );
                for ( int L = 0; L < num_mg_levels - 1; ++L )
                    R_.emplace_back( *domains_upper_[L] );
            }

            // Zero the restricted residual on Dirichlet velocity boundary shells so
            // the viscous v-cycle preserves the imposed boundary value exactly:
            // restriction otherwise smears interior residual into the eliminated
            // coarse boundary rows, the smoother updates them unopposed, and
            // prolongation leaks a solver-tolerance-sized velocity back onto the
            // fine boundary. Selection is by the CMB / SURFACE flag, correct under
            // lateral and radial subdomain decomposition.
            {
                const bool zero_cmb =
                    grid::shell::get_boundary_condition_flag( bcs_, grid::shell::ShellBoundaryFlag::CMB ) ==
                    grid::shell::BoundaryConditionFlag::DIRICHLET;
                const bool zero_surface =
                    grid::shell::get_boundary_condition_flag( bcs_, grid::shell::ShellBoundaryFlag::SURFACE ) ==
                    grid::shell::BoundaryConditionFlag::DIRICHLET;
                for ( auto& restriction : R_ )
                    restriction.set_dirichlet_boundary_zeroing( zero_cmb, zero_surface );
            }

            prec_11_ = std::make_unique< PrecVisc >(
                P_,
                R_,
                A_c_,
                tmp_mg_r_,
                tmp_mg_e_,
                tmp_mg_,
                smoothers_,
                smoothers_,
                *coarse_grid_solver_,
                prm_.stokes_solver_parameters.viscous_pc_num_vcycles,
                1e-6,
                std::move( redistribute_down ),
                std::move( tmp_mg_r_fine ),
                std::move( tmp_mg_e_fine ) );
        } // end if ( build_double_mg )

        // ---------------- Schur preconditioner ----------------
        logroot << "Setting up Schur complement preconditioner ..." << std::endl;
        k_pm_ = VectorQ1Scalar< ScalarType >( "k_pm", *domains_[pressure_level_], ownership_mask_[pressure_level_] );
        assign( k_pm_, eta_[pressure_level_] );
        linalg::invert_entries( k_pm_ );

        pmass_ = std::make_unique< PressureMass >(
            *domains_[pressure_level_],
            coords_shell_[pressure_level_],
            coords_radii_[pressure_level_],
            k_pm_.grid_data(),
            false );
        pmass_->set_lumped_diagonal( true );

        lumped_diagonal_pmass_ = VectorQ1Scalar< ScalarType >(
            "lumped_diagonal_pmass", *domains_[pressure_level_], ownership_mask_[pressure_level_] );
        {
            VectorQ1Scalar< ScalarType > tmp(
                "inverse_diagonal_tmp" + std::to_string( pressure_level_ ),
                *domains_[pressure_level_],
                ownership_mask_[pressure_level_] );
            linalg::assign( tmp, ScalarType( 1 ) );
            linalg::apply( *pmass_, tmp, lumped_diagonal_pmass_ );
        }
        // Schur relaxation (HyTeG Uzawa relaxParamSchur analogue): the DiagonalSolver
        // applies D^{-1}, so to get the effective preconditioner relax · Ŝ^{-1} we scale
        // the lumped diagonal D by 1/relax before it is inverted.
        const ScalarType schur_relax = static_cast< ScalarType >( prm_.stokes_solver_parameters.schur_relaxation );
        if ( schur_relax > ScalarType( 0 ) && schur_relax != ScalarType( 1 ) )
        {
            linalg::lincomb(
                lumped_diagonal_pmass_, { ScalarType( 1 ) / schur_relax }, { lumped_diagonal_pmass_ } );
        }
        inv_lumped_pmass_ = std::make_unique< PrecSchur >( lumped_diagonal_pmass_ );

        // ---------------- Outer block-triangular preconditioner ----------------
        logroot << "Setting up outer block-preconditioner ..." << std::endl;
        triangular_prec_tmp_ = VectorQ1IsoQ2Q1< ScalarType >(
            "triangular_prec_tmp",
            *domains_[velocity_level_],
            *domains_[pressure_level_],
            ownership_mask_[velocity_level_],
            ownership_mask_[pressure_level_] );

        // Select the velocity-preconditioner precision at runtime (--stokes-mg-precision).
        // double -> forward to the existing double multigrid; float/half -> a V-cycle
        // whose whole hierarchy runs in that precision (LowPrecVCycle).
        std::shared_ptr< typename VelPrecHandle::Impl > vel_impl;
        switch ( prm_.stokes_solver_parameters.mg_precision )
        {
        case MGPrecision::FLOAT:
            vel_impl = std::make_shared< LowPrecVCycle< float, Viscous > >(
                domains_, coords_shell_, coords_radii_, boundary_mask_, ownership_mask_, eta_, bcs_, prm_, table_ );
            break;
        case MGPrecision::HALF:
            // The EpsDivDiv operator does not compile/run with half-precision geometry
            // (coordinate differencing in the Jacobian needs >= float). A half *V-cycle*
            // would require keeping the operator/coords >= float (vectors-only half),
            // which is a separate mixed-precision-within-the-operator change.
            logroot << "ERROR: --stokes-mg-precision half is not supported (the velocity "
                       "operator needs >= float geometry). Use 'float'."
                    << std::endl;
            Kokkos::abort( "stokes-mg-precision: half unsupported" );
            break;
        case MGPrecision::DOUBLE:
        default:
            vel_impl = std::make_shared< linalg::solvers::ForwardingPrecImpl< Viscous, PrecVisc > >( *prec_11_ );
            break;
        }
        VelPrecHandle vel_prec( vel_impl );
        prec_stokes_ = std::make_unique< PrecStokes >(
            K_->block_11(), *pmass_, K_->block_12(), triangular_prec_tmp_, vel_prec, *inv_lumped_pmass_ );

        // ---------------- Outer FGMRES ----------------
        logroot << "Setting up FGMRES ... (Krylov basis precision: " << ( use_float_basis_ ? "single" : "double" )
                << ")" << std::endl;

        // ---------------- FGMRES (Stokes) workspace (deferred — see note above) ----------------
        // Allocated here, AFTER the MG hierarchy + Chebyshev eigenvalue estimate, so the
        // estimate's transient temps do not coincide with this (the largest) allocation.
        //   double path: 2*restart+4 full-precision vectors.
        //   float-basis path: 3 full-precision scratch (r/w aliased, v, z) + 2*restart+1 basis.
        if ( prm_.devel_parameters.extended_diagnostics )
            log_hbm( "stokes: before FGMRES workspace" );
        if ( use_float_basis_ )
        {
            constexpr int kNumStokesWork = 3; // FGMRESLowMem aliases r/w
            stokes_work_fgmres_.reserve( kNumStokesWork );
            for ( int i = 0; i < kNumStokesWork; i++ )
            {
                stokes_work_fgmres_.emplace_back(
                    "stokes_work_fgmres",
                    *domains_[velocity_level_],
                    *domains_[pressure_level_],
                    ownership_mask_[velocity_level_],
                    ownership_mask_[pressure_level_] );
            }
            const int num_stokes_basis = 2 * prm_.stokes_solver_parameters.krylov_restart + 1;
            stokes_basis_fgmres_.reserve( num_stokes_basis );
            for ( int i = 0; i < num_stokes_basis; i++ )
            {
                stokes_basis_fgmres_.emplace_back(
                    "stokes_basis_fgmres",
                    *domains_[velocity_level_],
                    *domains_[pressure_level_],
                    ownership_mask_[velocity_level_],
                    ownership_mask_[pressure_level_] );
            }
        }
        else
        {
            const int num_stokes_fgmres_tmps = 2 * prm_.stokes_solver_parameters.krylov_restart + 4;
            stokes_tmp_fgmres_.reserve( num_stokes_fgmres_tmps );
            for ( int i = 0; i < num_stokes_fgmres_tmps; i++ )
            {
                stokes_tmp_fgmres_.emplace_back(
                    "stokes_tmp_fgmres",
                    *domains_[velocity_level_],
                    *domains_[pressure_level_],
                    ownership_mask_[velocity_level_],
                    ownership_mask_[pressure_level_] );
            }
        }
        if ( prm_.devel_parameters.extended_diagnostics )
            log_hbm( "stokes: after FGMRES workspace (delta = Krylov basis+scratch)" );

        const linalg::solvers::FGMRESOptions< ScalarType > stokes_fgmres_opts{
            .restart                     = prm_.stokes_solver_parameters.krylov_restart,
            .relative_residual_tolerance = prm_.stokes_solver_parameters.krylov_relative_tolerance,
            .absolute_residual_tolerance = prm_.stokes_solver_parameters.krylov_absolute_tolerance,
            .max_iterations              = prm_.stokes_solver_parameters.krylov_max_iterations };
        if ( use_float_basis_ )
        {
            stokes_fgmres_float_ = std::make_unique< FGMRESFloat >(
                stokes_work_fgmres_, stokes_basis_fgmres_, stokes_fgmres_opts, table_, *prec_stokes_ );
            stokes_fgmres_float_->set_tag( "stokes_fgmres" );
        }
        else
        {
            stokes_fgmres_double_ = std::make_unique< FGMRESDouble >(
                stokes_tmp_fgmres_, stokes_fgmres_opts, table_, *prec_stokes_ );
            stokes_fgmres_double_->set_tag( "stokes_fgmres" );
        }

        if ( !prm_.stokes_solver_parameters.neural_precon.empty() )
        {
#ifdef TERRA_ENABLE_PYTHON
            if ( use_float_basis_ )
                throw std::runtime_error( "--stokes-neural-precon is not wired for the float-basis FGMRES" );
            logroot << "Setting up neural Stokes preconditioner ('"
                    << prm_.stokes_solver_parameters.neural_precon << "') ..." << std::endl;
            ml::NeuralSolverOptions nopt;
            nopt.model        = prm_.stokes_solver_parameters.neural_precon;
            nopt.log_residual = false;
            neural_prec_ = std::make_unique< ml::NeuralSolver< Stokes > >(
                nopt, *domains_[velocity_level_], *domains_[pressure_level_], triangular_prec_tmp_ );
            neural_prec_->set_eta( eta_[velocity_level_] ); // variable-viscosity models read it
            stokes_fgmres_neural_ = std::make_unique< FGMRESNeural >(
                stokes_tmp_fgmres_, stokes_fgmres_opts, table_, NeuralPrecRef{ neural_prec_.get() } );
            stokes_fgmres_neural_->set_tag( "stokes_fgmres" );
#else
            throw std::runtime_error( "--stokes-neural-precon needs a build with -DTERRA_ENABLE_PYTHON=ON" );
#endif
        }

        if ( !prm_.stokes_solver_parameters.neural_guess.empty() )
        {
#ifdef TERRA_ENABLE_PYTHON
            logroot << "Setting up neural Stokes initial guess ('"
                    << prm_.stokes_solver_parameters.neural_guess << "') ..." << std::endl;
            ml::NeuralSolverOptions gopt;
            gopt.model        = prm_.stokes_solver_parameters.neural_guess;
            gopt.log_residual = true; // prints ||b - Ax|| before/after = the guess quality
            neural_guess_ = std::make_unique< ml::NeuralSolver< Stokes > >(
                gopt, *domains_[velocity_level_], *domains_[pressure_level_], triangular_prec_tmp_ );
            neural_guess_->set_eta( eta_[velocity_level_] );
#else
            throw std::runtime_error( "--stokes-neural-guess needs a build with -DTERRA_ENABLE_PYTHON=ON" );
#endif
        }

        log_hbm( "stokes: ctor end (delta = MG hierarchy + operators + coarse + preconditioner)" );

        // Helper objects for the compressible (TALA) rhs grid transfer: the
        // (1/rho)grad(rho).u term is assembled on the velocity level and restricted
        // to the pressure level. Allocated regardless of --compressible (cheap).
        R_scalar_ =
            std::make_unique< RestrictionScalar >( *domains_[pressure_level_], linalg::OperatorApplyMode::Replace );
        tala_rhs_tmp_ = linalg::VectorQ1Scalar< ScalarType >(
            "tala_rhs_tmp", *domains_[velocity_level_], ownership_mask_[velocity_level_] );
    }

    // Public accessors needed by the rest of the app.

    linalg::VectorQ1IsoQ2Q1< ScalarType >& solution() { return stok_vecs_["u"]; }
    linalg::VectorQ1Scalar< ScalarType >&  eta_fine() { return eta_[velocity_level_]; }
    long                                   num_dofs_pressure() const { return num_dofs_pressure_; }
    const grid::shell::BoundaryConditions& boundary_conditions() const { return bcs_; }

    /// Update the fine-level viscosity from the current temperature using the
    /// configured viscosity law. No-op for ViscosityLaw::CONSTANT. Coarse-
    /// level eta (used by MG smoothing/coarse solves) is intentionally not
    /// touched, matching the pre-refactor behavior.
    void update_viscosity( const linalg::VectorQ1Scalar< ScalarType >& T )
    {
        if ( prm_.physics_parameters.viscosity_parameters.law == ViscosityLaw::CONSTANT )
            return;

        util::Timer timer_visc_update( "viscosity_update" );
        Kokkos::parallel_for(
            "viscosity_from_temperature",
            grid::shell::local_domain_md_range_policy_nodes( *domains_[velocity_level_] ),
            ViscosityFromTemperature{
                prm_.physics_parameters.viscosity_parameters.law,
                eta_[velocity_level_].grid_data(),
                T.grid_data(),
                eta_profile_,
                coords_radii_[velocity_level_],
                prm_.physics_parameters.viscosity_parameters.activation_energy,
                prm_.physics_parameters.viscosity_parameters.activation_volume,
                prm_.mesh_parameters.radius_max,
                prm_.physics_parameters.viscosity_parameters.min_viscosity,
                prm_.physics_parameters.viscosity_parameters.max_viscosity } );
        Kokkos::fence();

        // The fine viscous operator is matrix-free and now sees the new eta, but the
        // A-block MG smoother's cached D^-1 and Chebyshev eigenvalue interval were
        // estimated from the *initial* viscosity. Left stale, the smoother is
        // mistuned after the viscosity evolves and the MG stalls at high contrast
        // (HyTeG refreshes these on every viscosity update). Refresh the fine level.
        if ( prm_.stokes_solver_parameters.refresh_viscous_pc )
            refresh_viscous_smoother();
    }

    /// Refresh the A-block MG viscous preconditioner after a viscosity update.
    /// Always refreshes the fine-level D^-1 + Chebyshev bounds. When
    /// --stokes-refresh-coarse-viscosity is set, it ALSO restricts the (evolving,
    /// T-dependent) fine viscosity down the MG hierarchy (weighted average, non-GCA)
    /// so the coarse operators track the current viscosity, and re-tunes every level's
    /// smoother — the fix for the weak coarse correction at high viscosity contrast.
    void refresh_viscous_smoother()
    {
        if ( smoothers_.empty() )
            return;

        const bool coarse = prm_.stokes_solver_parameters.refresh_coarse_viscosity;

        // 1) Coarsen the fine viscosity down the hierarchy: eta_[L] = R(eta_[L+1]) / R(1).
        if ( coarse )
        {
            for ( int level = num_levels_ - 2; level >= 0; --level )
            {
                if ( domains_[level]->comm() == MPI_COMM_NULL || !eta_restr_[level] )
                    continue;
                linalg::apply( *eta_restr_[level], eta_[level + 1], eta_[level] ); // R(eta)
                auto ed = eta_[level].grid_data();
                auto nd = eta_restr_inv_norm_[level].grid_data();
                Kokkos::parallel_for(
                    "eta_coarsen_average",
                    grid::shell::local_domain_md_range_policy_nodes( *domains_[level] ),
                    KOKKOS_LAMBDA( const int id, const int x, const int y, const int r ) {
                        ed( id, x, y, r ) *= nd( id, x, y, r );
                    } );
                Kokkos::fence();
            }
        }

        // 2) Recompute D^-1 and re-tune the Chebyshev smoother. Fine level always;
        //    all levels when the coarse viscosity was refreshed above.
        const int first = coarse ? 0 : ( num_levels_ - 1 );
        for ( int level = first; level < num_levels_; ++level )
        {
            if ( domains_[level]->comm() == MPI_COMM_NULL )
                continue;
            VectorQ1Vec< ScalarType > tmp( "invdiag_refresh_tmp", *domains_[level], ownership_mask_[level] );
            linalg::assign( tmp, ScalarType( 1 ) );
            auto& A = ( level == num_levels_ - 1 ) ? K_->block_11() : A_c_[level];
            A.set_diagonal( true );
            linalg::apply( A, tmp, inverse_diagonals_[level] );
            A.set_diagonal( false );
            linalg::invert_entries( inverse_diagonals_[level] );
            linalg::assign( smoothers_[level].get_inverse_diagonal(), inverse_diagonals_[level] );
            smoothers_[level].refresh_max_eigenvalue_estimate_in_next_solve();
        }
    }

    /// Solve  K · u = f(T_for_buoyancy, rho, alpha)  with the configured FGMRES +
    /// MG/Schur preconditioner.  `u_dirichlet`, when set, is an inhomogeneous
    /// Dirichlet velocity at the surface (assimilated plate velocities).
    /// When `log_convergence` is true, the per-step Stokes and coarse-grid PCG
    /// tables are printed; in either case the table is cleared at the end.
    template < typename RhoFieldType >
    void solve( const linalg::VectorQ1Scalar< ScalarType >&                   T_for_buoyancy,
                const std::optional< linalg::VectorQ1IsoQ2Q1< ScalarType > >& u_dirichlet,
                const RhoFieldType&                                           rho,
                const grid::Grid2DDataScalar< ScalarType >&                   alpha,
                bool                                                          compressible,
                bool                                                          log_convergence )
    {
        util::Timer timer_stokes( "stokes" );

        util::logroot << "Setting up Stokes rhs ..." << std::endl;

        Kokkos::parallel_for(
            "Stokes rhs interpolation",
            grid::shell::local_domain_md_range_policy_nodes( *domains_[velocity_level_] ),
            BuoyancyForceAssembly(
                coords_shell_[velocity_level_],
                coords_radii_[velocity_level_],
                triangular_prec_tmp_.block_1().grid_data(),
                T_for_buoyancy.grid_data(),
                rho,
                alpha,
                prm_.physics_parameters.rayleigh_number,
                1.0 ) );

        linalg::apply( *M_, triangular_prec_tmp_.block_1(), stok_vecs_["f"].block_1() );

        // Strong enforcement of the velocity BCs on the RHS, applied per boundary.
        // NOTE: get_shell_boundary_flag( bcs_, FLAG ) only returns the FIRST boundary
        // carrying FLAG, so when both boundaries share a BC type (e.g. no-slip/no-slip
        // => both DIRICHLET, or free-slip/free-slip => both FREESLIP) the second boundary
        // would be left completely unenforced. Loop over both boundaries and dispatch on
        // each one's own flag instead.
        for ( const auto sbf : { grid::shell::ShellBoundaryFlag::CMB, grid::shell::ShellBoundaryFlag::SURFACE } )
        {
            const auto bcf = grid::shell::get_boundary_condition_flag( bcs_, sbf );

            if ( bcf == grid::shell::BoundaryConditionFlag::DIRICHLET )
            {
                // Plate velocities enter here: an inhomogeneous Dirichlet value at the surface.
                if ( sbf == grid::shell::ShellBoundaryFlag::SURFACE && u_dirichlet.has_value() )
                {
                    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
                        *K_neumann_,
                        *K_neumann_diag_,
                        *u_dirichlet,
                        triangular_prec_tmp_ /*tmp_vec*/,
                        stok_vecs_["f"],
                        boundary_mask_[velocity_level_],
                        sbf );
                }
                else
                {
                    fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
                        stok_vecs_["f"], boundary_mask_[velocity_level_], sbf );
                }
            }
            else if ( bcf == grid::shell::BoundaryConditionFlag::FREESLIP )
            {
                fe::strong_algebraic_freeslip_enforcement_in_place(
                    stok_vecs_["f"], coords_shell_[velocity_level_], boundary_mask_[velocity_level_], sbf );
            }
        }

        // Apply TALA RHS to mass equation if needed...
        if ( compressible )
        {
            using MassRHS = fe::wedge::linearforms::shell::InvRhoGradRhoDotU< ScalarType, RhoFieldType >;

            MassRHS mass_rhs(
                *domains_[pressure_level_],
                *domains_[velocity_level_],
                coords_shell_[pressure_level_],
                coords_shell_[velocity_level_],
                coords_radii_[pressure_level_],
                coords_radii_[velocity_level_],
                rho,
                stok_vecs_["u_prev"].block_1() );

            linalg::apply( mass_rhs, stok_vecs_["f"].block_2() );
        }

        util::logroot << "Solving Stokes ..." << std::endl;

        // Initial-guess policy for the warm-start benchmark. Default (neither flag):
        // persistence, i.e. u carries over from the previous timestep.
        // --stokes-guess-zero: cold start (the baseline).
        // --stokes-guess-extrap N: polynomial time-extrapolation from the stored
        // history, N = 1 linear (2u1-u2), 2 quadratic, 3 cubic; falls back to
        // persistence until enough history exists.
        {
            auto&     u  = stok_vecs_["u"];
            const int ne = prm_.stokes_solver_parameters.guess_extrap;
            if ( prm_.stokes_solver_parameters.guess_zero )
            {
                linalg::assign( u, static_cast< ScalarType >( 0 ) );
                util::logroot << "Initial guess: zero" << std::endl;
            }
            else if ( const int kp = prm_.stokes_solver_parameters.guess_proj; kp > 0 && n_hist_ >= kp )
            {
                // PROJECTION warm start: the residual-optimal combination of the
                // last kp solutions for the CURRENT operator and rhs,
                //   u0 = argmin_{u in span(u_prev..)} ||f - K u||,
                // i.e. c = G^-1 g with G_ij = <K u_i, K u_j>, g_i = <K u_i, f>.
                // kp matvecs + a kp x kp solve. Optimal by construction and
                // robust to non-uniform dt, unlike fixed Taylor coefficients.
                static const char* names[4] = { "u_prev", "u_prev2", "u_prev3", "u_prev4" };
                const int          k        = std::min( kp, 4 );
                std::vector< std::vector< ScalarType > > G( k, std::vector< ScalarType >( k, 0 ) );
                std::vector< ScalarType >                g( k, 0 );
                // Two scratch vectors only (w, and u itself which is overwritten at
                // the end). Per j: w = K u_j gives g_j and G_jj; the off-diagonals
                // use the symmetry of K: <K u_i, K u_j> = <u_i, K (K u_j)>, one more
                // matvec into u. Total k + k(k-1)/2 matvecs.
                auto& w = triangular_prec_tmp_;
                for ( int j = 0; j < k; ++j )
                {
                    linalg::apply( *K_, stok_vecs_[names[j]], w );
                    g[j]    = linalg::dot( w, stok_vecs_["f"] );
                    G[j][j] = linalg::dot( w, w );
                    if ( j > 0 )
                    {
                        linalg::apply( *K_, w, u ); // u = K K u_j
                        for ( int i = 0; i < j; ++i )
                            G[i][j] = G[j][i] = linalg::dot( stok_vecs_[names[i]], u );
                    }
                }
                // Solve G c = g (tiny dense system, Gaussian elimination with pivoting).
                std::vector< ScalarType > c( k, 0 );
                {
                    auto A = G;
                    auto b = g;
                    for ( int p = 0; p < k; ++p )
                    {
                        int piv = p;
                        for ( int r = p + 1; r < k; ++r )
                            if ( std::abs( A[r][p] ) > std::abs( A[piv][p] ) )
                                piv = r;
                        std::swap( A[p], A[piv] );
                        std::swap( b[p], b[piv] );
                        if ( std::abs( A[p][p] ) < 1e-300 )
                            continue;
                        for ( int r = p + 1; r < k; ++r )
                        {
                            const ScalarType fct = A[r][p] / A[p][p];
                            for ( int q = p; q < k; ++q )
                                A[r][q] -= fct * A[p][q];
                            b[r] -= fct * b[p];
                        }
                    }
                    for ( int p = k - 1; p >= 0; --p )
                    {
                        ScalarType s = b[p];
                        for ( int q = p + 1; q < k; ++q )
                            s -= A[p][q] * c[q];
                        c[p] = std::abs( A[p][p] ) < 1e-300 ? 0 : s / A[p][p];
                    }
                }
                // u = sum_i c_i u_i, accumulated 2-3 vectors at a time (kernel limit 3).
                linalg::lincomb( u, { c[0] }, { stok_vecs_[names[0]] } );
                for ( int i = 1; i < k; ++i )
                    linalg::lincomb( u, { 1.0, c[i] }, { u, stok_vecs_[names[i]] } );
                util::logroot << "Initial guess: projection onto last " << k << " solutions, c =";
                for ( int i = 0; i < k; ++i )
                    util::logroot << " " << c[i];
                util::logroot << std::endl;
            }
            else if ( ne > 0 && n_hist_ >= ne + 1 )
            {
                // Lagrange extrapolation weights to the current solve time. The
                // uniform-spacing stencils (2,-1 / 3,-3,1 / 4,-6,4,-1) are only
                // correct for constant dt; with the CFL ramp dt grows ~1.5x per
                // step and the uniform cubic amplifies the history error.
                const int                 n = std::min( ne, 3 ) + 1;
                std::array< ScalarType, 4 > w{};
                const bool                nonuniform = t_now_set_ && n_thist_ >= n;
                if ( nonuniform )
                {
                    for ( int i = 0; i < n; i++ )
                    {
                        w[i] = 1;
                        for ( int j = 0; j < n; j++ )
                            if ( j != i )
                                w[i] *= ( t_now_ - t_hist_[j] ) / ( t_hist_[i] - t_hist_[j] );
                    }
                }
                else
                {
                    static const ScalarType uni[3][4] = { { 2, -1, 0, 0 }, { 3, -3, 1, 0 }, { 4, -6, 4, -1 } };
                    for ( int i = 0; i < 4; i++ )
                        w[i] = uni[n - 2][i];
                }
                if ( n == 2 )
                    linalg::lincomb( u, { w[0], w[1] }, { stok_vecs_["u_prev"], stok_vecs_["u_prev2"] } );
                else if ( n == 3 )
                    linalg::lincomb(
                        u, { w[0], w[1], w[2] }, { stok_vecs_["u_prev"], stok_vecs_["u_prev2"], stok_vecs_["u_prev3"] } );
                else
                {
                    // lincomb kernels take at most 3 inputs: cubic in two stages
                    linalg::lincomb(
                        u, { w[0], w[1], w[2] }, { stok_vecs_["u_prev"], stok_vecs_["u_prev2"], stok_vecs_["u_prev3"] } );
                    linalg::lincomb( u, { 1.0, w[3] }, { u, stok_vecs_["u_prev4"] } );
                }
                util::logroot << "Initial guess: time-extrapolation order " << n - 1
                              << ( nonuniform ? " (non-uniform dt, weights" : " (uniform weights" );
                for ( int i = 0; i < n; i++ )
                    util::logroot << " " << w[i];
                util::logroot << ")" << std::endl;
            }
        }

        if ( prm_.stokes_solver_parameters.tolerance_relative_to_rhs )
        {
            const auto&      f     = stok_vecs_["f"];
            const ScalarType tol_f = prm_.stokes_solver_parameters.krylov_relative_tolerance *
                                     std::sqrt( linalg::dot( f, f ) );
            const ScalarType tol = std::max( tol_f, prm_.stokes_solver_parameters.krylov_absolute_tolerance );
            if ( stokes_fgmres_float_ )
                stokes_fgmres_float_->set_absolute_tolerance( tol );
            if ( stokes_fgmres_double_ )
                stokes_fgmres_double_->set_absolute_tolerance( tol );
#ifdef TERRA_ENABLE_PYTHON
            if ( stokes_fgmres_neural_ )
                stokes_fgmres_neural_->set_absolute_tolerance( tol );
#endif
            util::logroot << "Stokes tolerance relative to ||f||: abs target = " << tol << std::endl;
        }

        // Warm-start training pairs in double precision: r = f - K u_guess now,
        // e = u - u_guess after the solve (xdmf is float, and K amplifies float
        // rounding of u to the size of a cubic-extrapolation residual).
        const bool dump_pairs = !prm_.stokes_solver_parameters.dump_pairs_dir.empty();
        if ( dump_pairs )
        {
            const auto& dir = prm_.stokes_solver_parameters.dump_pairs_dir;
            std::filesystem::create_directories( dir );
            auto& w = triangular_prec_tmp_;
            linalg::assign( stok_vecs_["u_guess"], stok_vecs_["u"] );
            linalg::apply( *K_, stok_vecs_["u"], w );
            linalg::lincomb( w, { -1.0, 1.0 }, { w, stok_vecs_["f"] } );
            char tag[32];
            std::snprintf( tag, sizeof( tag ), "%04d", dump_count_ );
            dump_stokes_vec_f64_( dir + "/pair_" + tag + "_r.bin", w );
            {
                std::ofstream os( dir + "/pair_" + tag + "_eta.bin", std::ios::binary );
                dump_view_f64_( os, eta_[velocity_level_].grid_data() );
            }
            util::logroot << "Dumped guess residual pair " << tag << ": ||f - K u_guess|| = "
                          << std::sqrt( linalg::dot( w, w ) ) << ", ||f|| = "
                          << std::sqrt( linalg::dot( stok_vecs_["f"], stok_vecs_["f"] ) ) << std::endl;
        }

#ifdef TERRA_ENABLE_PYTHON
        // Overwrites the initial guess (including the previous-timestep warm start)
        // with one application of the rhs-trained operator to the physical rhs --
        // the deployment that operator is actually in-distribution for.
        if ( neural_guess_ )
        {
            util::logroot << "Neural initial guess ..." << std::endl;
            neural_guess_->solve_impl( *K_, stok_vecs_["u"], stok_vecs_["f"] );
            // Residual-optimal rescaling of the guess, alpha = <f, K z> / <K z, K z>
            // (one matvec): absorbs any global amplitude mismatch between the
            // model's training units and the app's rhs, so what remains measures
            // the guess DIRECTION. alpha ~ 1 means the scaling was right.
            {
                auto& z = stok_vecs_["u"];
                auto& w = triangular_prec_tmp_; // idle scratch until the solve
                linalg::apply( *K_, z, w );
                const ScalarType num   = linalg::dot( stok_vecs_["f"], w );
                const ScalarType den   = linalg::dot( w, w );
                const ScalarType alpha = den > 0 ? num / den : ScalarType( 0 );
                linalg::lincomb( z, { alpha }, { z } );
                linalg::apply( *K_, z, w );
                linalg::lincomb( w, { -1.0, 1.0 }, { w, stok_vecs_["f"] } );
                util::logroot << "Neural guess linesearch: alpha = " << alpha
                              << ", ||f - K(alpha z)|| = " << std::sqrt( linalg::dot( w, w ) ) << std::endl;
            }
        }

        if ( prm_.stokes_solver_parameters.iterative_refinement )
        {
            if ( use_float_basis_ )
                throw std::runtime_error( "--stokes-iterative-refinement is not wired for the float-basis solver" );
            solve_ir_();
        }
        else if ( stokes_fgmres_neural_ )
            ::terra::linalg::solvers::solve( *stokes_fgmres_neural_, *K_, stok_vecs_["u"], stok_vecs_["f"] );
        else
#else
        if ( prm_.stokes_solver_parameters.iterative_refinement )
        {
            if ( use_float_basis_ )
                throw std::runtime_error( "--stokes-iterative-refinement is not wired for the float-basis solver" );
            solve_ir_();
        }
        else
#endif
        if ( use_float_basis_ )
            ::terra::linalg::solvers::solve( *stokes_fgmres_float_, *K_, stok_vecs_["u"], stok_vecs_["f"] );
        else
            ::terra::linalg::solvers::solve( *stokes_fgmres_double_, *K_, stok_vecs_["u"], stok_vecs_["f"] );

        if ( dump_pairs )
        {
            auto& e = stok_vecs_["u_guess"];
            linalg::lincomb( e, { 1.0, -1.0 }, { stok_vecs_["u"], e } );
            char tag[32];
            std::snprintf( tag, sizeof( tag ), "%04d", dump_count_ );
            dump_stokes_vec_f64_( prm_.stokes_solver_parameters.dump_pairs_dir + "/pair_" + tag + "_e.bin", e );
            // achieved accuracy of the "true" solution: the converter drops pairs whose
            // final residual is not small against the guess residual
            auto& w = triangular_prec_tmp_;
            linalg::apply( *K_, stok_vecs_["u"], w );
            linalg::lincomb( w, { -1.0, 1.0 }, { w, stok_vecs_["f"] } );
            util::logroot << "Dumped pair " << tag << " final ||f - K u|| = " << std::sqrt( linalg::dot( w, w ) )
                          << std::endl;
            ++dump_count_;
        }

        // Block-wise final residual r = f - K u: says whether the remaining
        // residual lives in the momentum or the continuity block. (Rank-local
        // dot; the neural preconditioner demos run single-rank.)
        {
            linalg::apply( *K_, stok_vecs_["u"], triangular_prec_tmp_ );
            linalg::lincomb( triangular_prec_tmp_, { 1.0, -1.0 }, { stok_vecs_["f"], triangular_prec_tmp_ } );
            const auto r_u = std::sqrt( triangular_prec_tmp_.block_1().dot_impl( triangular_prec_tmp_.block_1() ) );
            const auto r_p = std::sqrt( triangular_prec_tmp_.block_2().dot_impl( triangular_prec_tmp_.block_2() ) );
            const auto f_u = std::sqrt( stok_vecs_["f"].block_1().dot_impl( stok_vecs_["f"].block_1() ) );
            const auto f_p = std::sqrt( stok_vecs_["f"].block_2().dot_impl( stok_vecs_["f"].block_2() ) );
            util::logroot << "Stokes residual blocks: ||r_u|| = " << r_u << ", ||r_p|| = " << r_p
                          << "  (||f_u|| = " << f_u << ", ||f_p|| = " << f_p << ")" << std::endl;
        }

        if ( log_convergence )
        {
            table_->query_rows_equals( "tag", "stokes_fgmres" ).print_pretty();
            //table_->query_rows_equals( "tag", "coarse_grid_pcg" ).print_pretty();
        }
        table_->clear();

        // "Normalize" pressure (subtract average).
        auto&            p = stok_vecs_["u"].block_2();
        const ScalarType avg_pressure_approximation =
            kernels::common::masked_sum( p.grid_data(), p.mask_data(), grid::NodeOwnershipFlag::OWNED ) /
            static_cast< ScalarType >( num_dofs_pressure_ );
        linalg::lincomb( p, { 1.0 }, { p }, -avg_pressure_approximation );

        // Shift the solution history and store u_prev for the next timestep
        linalg::assign( stok_vecs_["u_prev4"], stok_vecs_["u_prev3"] );
        linalg::assign( stok_vecs_["u_prev3"], stok_vecs_["u_prev2"] );
        linalg::assign( stok_vecs_["u_prev2"], stok_vecs_["u_prev"] );
        linalg::assign( stok_vecs_["u_prev"], stok_vecs_["u"] );
        ++n_hist_;
        if ( t_now_set_ )
        {
            for ( int i = 3; i > 0; i-- )
                t_hist_[i] = t_hist_[i - 1];
            t_hist_[0] = t_now_;
            n_thist_   = std::min( n_thist_ + 1, 4 );
        }
    }

    template < typename ViewT >
    static void dump_view_f64_( std::ofstream& os, const ViewT& dev )
    {
        auto host = Kokkos::create_mirror( Kokkos::HostSpace{}, dev );
        Kokkos::deep_copy( host, dev );
        for ( std::size_t s = 0; s < host.extent( 0 ); ++s )
            for ( std::size_t i = 0; i < host.extent( 1 ); ++i )
                for ( std::size_t j = 0; j < host.extent( 2 ); ++j )
                    for ( std::size_t k = 0; k < host.extent( 3 ); ++k )
                    {
                        const double v = static_cast< double >( host( s, i, j, k ) );
                        os.write( reinterpret_cast< const char* >( &v ), sizeof( double ) );
                    }
    }

    /// Raw f64 layout: velocity component-planar [3][subdomain][i][j][k] on the
    /// velocity grid, then pressure [subdomain][i][j][k] on the pressure grid.
    void dump_stokes_vec_f64_( const std::string& path, const linalg::VectorQ1IsoQ2Q1< ScalarType >& v )
    {
        std::ofstream os( path, std::ios::binary );
        for ( int d = 0; d < 3; ++d )
            dump_view_f64_( os, v.block_1().grid_data().comp_[d] );
        dump_view_f64_( os, v.block_2().grid_data() );
    }

    /// Simulated time the next solve() belongs to (end of the current timestep).
    /// Enables non-uniform-dt extrapolation of the initial guess.
    void set_solve_time( ScalarType t )
    {
        t_now_     = t;
        t_now_set_ = true;
    }

    /// Damped iterative refinement  x <- x + omega * M (f - K x)  with the
    /// configured preconditioner M (neural model if --stokes-neural-precon is
    /// set, block MG/Schur otherwise). No Krylov acceleration: the printed
    /// per-step contraction IS ||I - omega M K|| along the current error, and
    /// the block norms say in which block M fails. Reuses the FGMRES scratch
    /// vectors; iteration count / tolerance come from the krylov settings.
    /// (Rank-local norms; the neural demos run single-rank.)
    void solve_ir_()
    {
        auto&        x     = stok_vecs_["u"];
        auto&        f     = stok_vecs_["f"];
        auto&        r     = stokes_tmp_fgmres_[0];
        auto&        z     = stokes_tmp_fgmres_[1];
        auto&        xprev = stokes_tmp_fgmres_[3];
        auto&        xcur  = stokes_tmp_fgmres_[4];
        const double beta  = prm_.stokes_solver_parameters.ir_momentum;
        linalg::assign( xprev, x );
        const double omega = prm_.stokes_solver_parameters.ir_damping;
        const int    n_it  = prm_.stokes_solver_parameters.krylov_max_iterations;
        const double tol   = prm_.stokes_solver_parameters.krylov_relative_tolerance;

        util::logroot << "Iterative refinement, damping " << omega << " ..." << std::endl;
        const bool auto_mom = prm_.stokes_solver_parameters.ir_auto_momentum;
        double     rho_ema  = 0.0;
        double     beta_eff = beta;
        double     rel_prev = 1.0, rel_pp = 1.0, rel_ppp = 1.0;
        double     f0       = 1.0;
        for ( int it = 0; it <= n_it; ++it )
        {
            linalg::apply( *K_, x, r );
            linalg::lincomb( r, { 1.0, -1.0 }, { f, r } );
            const double r_u = std::sqrt( r.block_1().dot_impl( r.block_1() ) );
            const double r_p = std::sqrt( r.block_2().dot_impl( r.block_2() ) );
            const double rn  = std::sqrt( r_u * r_u + r_p * r_p );
            if ( it == 0 )
                f0 = rn > 0.0 ? rn : 1.0;
            const double rel = rn / f0;
            if ( auto_mom )
            {
                if ( it >= 2 && rel_prev > 0.0 )
                {
                    const double ratio = std::min( rel / rel_prev, 0.999 );
                    rho_ema = ( rho_ema == 0.0 ) ? ratio : 0.85 * rho_ema + 0.15 * ratio;
                }
                if ( it >= 6 && rho_ema > 0.0 )
                {
                    const double s = std::sqrt( std::max( 1.0 - rho_ema, 1e-4 ) );
                    beta_eff      = std::min( 0.85, std::pow( ( 1.0 - s ) / ( 1.0 + s ), 2.0 ) );
                }
                if ( it >= 3 && rel > rel_ppp )
                {
                    // residual rose over 3 steps: kill the momentum memory
                    linalg::assign( xprev, x );
                    beta_eff *= 0.7;
                }
                util::logroot << "stokes_ir |   auto beta " << beta_eff << " (rho " << rho_ema << ")"
                              << std::endl;
            }
            rel_ppp = rel_pp; rel_pp = rel_prev; rel_prev = rel;
            util::logroot << "stokes_ir | it " << it << " | rel " << rel << " | ||r_u|| " << r_u
                          << " | ||r_p|| " << r_p << std::endl;
            if ( rel < tol || it == n_it )
                break;
            linalg::assign( z, 0 );
#ifdef TERRA_ENABLE_PYTHON
            // hybrid: even steps neural, odd steps block MG/Schur
            if ( neural_prec_ && ( !prm_.stokes_solver_parameters.ir_hybrid || it % 2 == 0 ) )
                neural_prec_->solve_impl( *K_, z, r );
            else
#endif
                prec_stokes_->solve_impl( *K_, z, r );
            if ( prm_.stokes_solver_parameters.ir_linesearch )
            {
                // residual minimiser along z: one matvec, no basis
                auto& w = stokes_tmp_fgmres_[2];
                linalg::apply( *K_, z, w );
                const double den   = w.dot_impl( w );
                const double alpha = den > 0.0 ? w.dot_impl( r ) / den : 0.0;
                util::logroot << "stokes_ir |   alpha " << alpha << std::endl;
                linalg::lincomb( x, { 1.0, omega * alpha }, { x, z } );
            }
            else if ( beta_eff > 0.0 )
            {
                linalg::assign( xcur, x );
                linalg::lincomb( x, { 1.0 + beta_eff, omega, -beta_eff }, { x, z, xprev } );
                linalg::assign( xprev, xcur );
            }
            else
                linalg::lincomb( x, { 1.0, omega }, { x, z } );
        }
    }

  private:
    // Inputs stored BY VALUE so this context owns its dependencies and isn't
    // tied to the lifetime/identity of the mc.cpp locals.  Vectors-of-views
    // are cheap to copy (Kokkos::View handles are refcounted), and the BC
    // C-array is two structs.
    std::vector< std::shared_ptr< grid::shell::DistributedDomain > >        domains_;
    std::vector< grid::Grid3DDataVec< ScalarType, 3 > >                     coords_shell_;
    std::vector< grid::Grid2DDataScalar< ScalarType > >                     coords_radii_;
    std::vector< grid::Grid4DDataScalar< grid::NodeOwnershipFlag > >        ownership_mask_;
    std::vector< grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_;
    grid::Grid2DDataScalar< ScalarType >                                    eta_profile_;
    grid::shell::BoundaryConditions                                         bcs_{}; // velocity BC set (owned copy)
    const Parameters&                                                       prm_;
    std::shared_ptr< util::Table >                                          table_;

    int  num_levels_;
    int  velocity_level_;
    int  pressure_level_;
    long num_dofs_pressure_ = 0;

    // Owned per-level state. Order matters: data members declared earlier are
    // destroyed later, so things that other members hold by-reference (eta_,
    // stok_vecs_, *_tmp_*) must come first.
    std::vector< linalg::VectorQ1Scalar< ScalarType > >            eta_;
    linalg::VectorQ1Scalar< ScalarType >                           GCAElements_;
    std::map< std::string, linalg::VectorQ1IsoQ2Q1< ScalarType > > stok_vecs_;
    std::vector< linalg::VectorQ1IsoQ2Q1< ScalarType > >   stokes_tmp_fgmres_;   // double path
    std::vector< linalg::VectorQ1IsoQ2Q1< ScalarType > >   stokes_work_fgmres_;  // float-basis path: scratch
    std::vector< BasisVectorType >                         stokes_basis_fgmres_; // float-basis path: basis
    linalg::VectorQ1Scalar< ScalarType >                   tala_rhs_tmp_;        // compressible (TALA) rhs scratch
    std::vector< linalg::VectorQ1Vec< ScalarType > >       tmp_mg_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       tmp_mg_2_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       tmp_mg_r_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       tmp_mg_e_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       inverse_diagonals_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       coarse_grid_tmps_;

    // Heavy operators / solvers held via unique_ptr so we can construct in
    // body order (rather than fighting member-init order).
    std::unique_ptr< Stokes >            K_;
    std::unique_ptr< Stokes >            K_neumann_;
    std::unique_ptr< Stokes >            K_neumann_diag_;
    std::unique_ptr< ViscousMass >       M_;
    std::vector< Viscous >               A_c_;
    std::vector< Prolongation >          P_;
    std::vector< Restriction >           R_;
    std::unique_ptr< RestrictionScalar > R_scalar_;
    // Scalar-restriction hierarchy for coarsening the T-dependent viscosity down the
    // MG levels on refresh (non-GCA path). eta_restr_[L] restricts level L+1 -> L;
    // eta_restr_inv_norm_[L] = 1/R_L(1) turns the functional restriction into a
    // weighted average (coarse eta = R(eta)/R(1)).
    std::vector< std::unique_ptr< RestrictionScalar > > eta_restr_;
    std::vector< VectorQ1Scalar< ScalarType > >         eta_restr_inv_norm_;
    std::vector< Smoother >              smoothers_;
    std::unique_ptr< CoarseGridSolver >  coarse_grid_solver_;

    // Comm-aware MG agglomeration (empty/no-op when agglom_factors is all 1s).
    std::vector< std::shared_ptr< grid::shell::DistributedDomain > > domains_upper_;
    std::vector< grid::Grid4DDataScalar< grid::NodeOwnershipFlag > > mask_upper_;

    std::unique_ptr< PrecVisc > prec_11_;

    // Schur preconditioner pieces.
    linalg::VectorQ1Scalar< ScalarType > k_pm_;
    std::unique_ptr< PressureMass >      pmass_;
    linalg::VectorQ1Scalar< ScalarType > lumped_diagonal_pmass_;
    std::unique_ptr< PrecSchur >         inv_lumped_pmass_;

    // Outer Stokes preconditioner / solver. Exactly one of the two FGMRES variants
    // is allocated, selected by use_float_basis_ (--stokes-float-krylov-basis).
    linalg::VectorQ1IsoQ2Q1< ScalarType >                  triangular_prec_tmp_;
    std::unique_ptr< PrecStokes >                          prec_stokes_;
    bool                                                   use_float_basis_ = false;
    std::unique_ptr< FGMRESDouble >                        stokes_fgmres_double_;
    std::unique_ptr< FGMRESFloat >                         stokes_fgmres_float_;
#ifdef TERRA_ENABLE_PYTHON
    std::unique_ptr< ml::NeuralSolver< Stokes > > neural_prec_;
    std::unique_ptr< FGMRESNeural >               stokes_fgmres_neural_;
    std::unique_ptr< ml::NeuralSolver< Stokes > > neural_guess_;
#endif
    int n_hist_ = 0; ///< solutions stored so far (gates the extrapolation order)
    ScalarType                  t_now_     = 0;
    bool                        t_now_set_ = false;
    std::array< ScalarType, 4 > t_hist_{}; ///< solve times of u_prev..u_prev4
    int                         n_thist_ = 0;
    int                         dump_count_ = 0;
};

} // namespace terra::mantlecirculation
