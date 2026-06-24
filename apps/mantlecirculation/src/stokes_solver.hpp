#pragma once

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "communication/shell/redistribute.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/strong_algebraic_freeslip_enforcement.hpp"
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
#include "mpi/mpi.hpp"
#include "shell/spherical_harmonics.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

#include "mpi/level_comms.hpp"

#include "build_radii.hpp"
#include "hbm_probe.hpp"
#include "interpolators.hpp"
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
    : num_mg_levels_( prm.mesh_parameters.refinement_level_mesh_max
                      - prm.mesh_parameters.refinement_level_mesh_min + 1 )
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

    int                     num_mg_levels() const { return num_mg_levels_; }
    const std::vector< int >& factors()      const { return factors_; }

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
    int                       num_mg_levels_;
    std::vector< int >        factors_;
    MPI_Comm                  world_comm_;
    std::vector< MPI_Comm >   level_comms_;
    std::vector< int >        cum_factors_;
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
    using Stokes       = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous      = typename Stokes::Block11Type;
    using Gradient     = typename Stokes::Block12Type;
    using ViscousMass  = fe::wedge::operators::shell::VectorMass< ScalarType >;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;
    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;
    using Smoother     = linalg::solvers::Chebyshev< Viscous >;
    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;
    using VelGridData      = grid::Grid4DDataVec< ScalarType, 3 >;
    using Redistribute     = communication::shell::Redistribute< VelGridData >;
    using PrecVisc         = linalg::solvers::Multigrid< Viscous,
                                                          Prolongation,
                                                          Restriction,
                                                          Smoother,
                                                          CoarseGridSolver,
                                                          Redistribute >;
    using PrecSchur  = linalg::solvers::DiagonalSolver< PressureMass >;
    // The (1,1) velocity preconditioner is type-erased so its internal precision
    // (the MG V-cycle precision) can be chosen at runtime via --stokes-mg-precision.
    using VelPrecHandle = linalg::solvers::VelocityPrecHandle< Viscous >;
    using PrecStokes     = linalg::solvers::
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
    using BasisVectorType = linalg::VectorQ1IsoQ2Q1< Kokkos::Experimental::half_t, 3 >;
    using FGMRESDouble    = linalg::solvers::FGMRES< Stokes, PrecStokes >;
    using FGMRESFloat     = linalg::solvers::FGMRESLowMem< Stokes, BasisVectorType, PrecStokes >;

  public:
    StokesContext(
        const std::vector< std::shared_ptr< grid::shell::DistributedDomain > >&         domains,
        const std::vector< grid::Grid3DDataVec< ScalarType, 3 > >&                      coords_shell,
        const std::vector< grid::Grid2DDataScalar< ScalarType > >&                      coords_radii,
        const std::vector< grid::Grid4DDataScalar< grid::NodeOwnershipFlag > >&         ownership_mask,
        const std::vector< grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > >&  boundary_mask,
        grid::shell::BoundaryConditions&                                                bcs,
        const MGAgglomeration&                                                          agglom,
        const Parameters&                                                               prm,
        std::shared_ptr< util::Table >                                                  table )
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
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];
        const int lat_sdr = ( prm.mesh_parameters.lat_sdr >= 0 ) ? prm.mesh_parameters.lat_sdr
                                                                 : prm.mesh_parameters.refinement_level_subdomains;
        const int rad_sdr = ( prm.mesh_parameters.rad_sdr >= 0 ) ? prm.mesh_parameters.rad_sdr
                                                                 : prm.mesh_parameters.refinement_level_subdomains;
        const auto& agglom_factors = agglom.factors();
        using grid::shell::DistributedDomain;
        using linalg::VectorQ1IsoQ2Q1;
        using linalg::VectorQ1Scalar;
        using linalg::VectorQ1Vec;
        using util::logroot;

        num_dofs_pressure_ = kernels::common::count_masked< long >(
            ownership_mask_[pressure_level_], grid::NodeOwnershipFlag::OWNED );

        // ---------------- Stokes block vectors ----------------
        std::vector< std::string > stok_vec_names = { "u", "f", "tmp" };
        for ( const auto& name : stok_vec_names )
        {
            stok_vecs_[name] = VectorQ1IsoQ2Q1< ScalarType >(
                name,
                *domains_[velocity_level_],
                *domains_[pressure_level_],
                ownership_mask_[velocity_level_],
                ownership_mask_[pressure_level_] );
        }

        // ---------------- Viscosity ----------------
        // Radial profile (constant or CSV-driven), then projected into Q1 on every level.
        std::vector< grid::Grid2DDataScalar< ScalarType > > radial_viscosity_profile;
        radial_viscosity_profile.reserve( num_levels_ );
        if ( !prm_.physics_parameters.viscosity_parameters.radial_profile_enabled )
        {
            logroot << "Using constant viscosity profile." << std::endl;
            for ( int level = 0; level < num_levels_; level++ )
            {
                radial_viscosity_profile.push_back(
                    shell::interpolate_constant_radial_profile( coords_radii_[level], 1.0 ) );
            }
        }
        else
        {
            logroot << "Using radially varying viscosity profile." << std::endl;
            for ( int level = 0; level < num_levels_; level++ )
            {
                radial_viscosity_profile.push_back(
                    shell::interpolate_radial_profile_into_subdomains_from_csv(
                        prm_.physics_parameters.viscosity_parameters.radial_profile_csv_filename,
                        prm_.physics_parameters.viscosity_parameters.radial_profile_radii_key,
                        prm_.physics_parameters.viscosity_parameters.radial_profile_viscosity_key,
                        coords_radii_[level] ) );
            }
        }

        eta_.reserve( num_levels_ );
        for ( int level = 0; level < num_levels_; level++ )
        {
            const std::string name = ( level == num_levels_ - 1 )
                                         ? std::string( "eta" )
                                         : std::string( "eta_level_" ) + std::to_string( level );
            eta_.emplace_back( name, *domains_[level], ownership_mask_[level] );
        }
        for ( int level = 0; level < num_levels_; level++ )
        {
            // GCA still needs an approximation of viscosity on coarse grids
            // for the weighting of the mass matrix.
            geophysics::viscosity::RadialProfileViscosityInterpolator viscosity_interpolator(
                radial_viscosity_profile[level],
                prm_.physics_parameters.viscosity_parameters.reference_viscosity );
            viscosity_interpolator.interpolate( eta_[level].grid_data() );
        }

        // ---------------- GCA element selection ----------------
        GCAElements_ = VectorQ1Scalar< ScalarType >(
            "GCAElements", *domains_[0], ownership_mask_[0] );
        const int gca = prm_.stokes_solver_parameters.gca;
        if ( gca > 0 &&
             std::any_of( agglom_factors.begin(), agglom_factors.end(), []( int f ) { return f > 1; } ) )
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

        log_hbm( "stokes: before FGMRES workspace" );
        // ---------------- FGMRES (Stokes) workspace ----------------
        // Two layouts, selected by --stokes-float-krylov-basis:
        //   double path: 2*restart+4 full-precision vectors.
        //   float-basis path: 4 full-precision scratch (r, w, v_scratch, z_scratch)
        //                     + a single-precision Krylov basis of 2*restart+1.
        // Only the active layout is allocated.
        use_float_basis_ = prm_.stokes_solver_parameters.float_krylov_basis;
        if ( use_float_basis_ )
        {
            constexpr int kNumStokesWork = 4;
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

        log_hbm( "stokes: after FGMRES workspace (delta = Krylov basis+scratch)" );

        // ---------------- Multigrid tmp vectors ----------------
        for ( int level = 0; level < num_levels_; level++ )
        {
            tmp_mg_.emplace_back( "tmp_mg_" + std::to_string( level ), *domains_[level], ownership_mask_[level] );
            tmp_mg_2_.emplace_back( "tmp_mg_2_" + std::to_string( level ), *domains_[level], ownership_mask_[level] );
            if ( level < num_levels_ - 1 )
            {
                tmp_mg_r_.emplace_back( "tmp_mg_r_" + std::to_string( level ), *domains_[level], ownership_mask_[level] );
                tmp_mg_e_.emplace_back( "tmp_mg_e_" + std::to_string( level ), *domains_[level], ownership_mask_[level] );
            }
        }

        // ---------------- Stokes operators ----------------
        grid::shell::BoundaryConditions bcs_neumann = {
            { grid::shell::ShellBoundaryFlag::CMB,     grid::shell::BoundaryConditionFlag::NEUMANN },
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

        K_neumann_ = std::make_unique< Stokes >(
            *domains_[velocity_level_],
            *domains_[pressure_level_],
            coords_shell_[velocity_level_],
            coords_radii_[velocity_level_],
            boundary_mask_[velocity_level_],
            eta_[velocity_level_].grid_data(),
            bcs_neumann,
            false );

        M_ = std::make_unique< ViscousMass >(
            *domains_[velocity_level_], coords_shell_[velocity_level_], coords_radii_[velocity_level_], false );

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
                logroot << "Assembling GCA on level "
                        << prm_.mesh_parameters.refinement_level_mesh_min + level << std::endl;
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
                "inverse_diagonal_tmp" + std::to_string( level ),
                *domains_[level],
                ownership_mask_[level] );
            linalg::assign( tmp, 1.0 );
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
        for ( int level = 0; level < num_levels_; level++ )
        {
            if ( domains_[level]->comm() == MPI_COMM_NULL )
                continue;

            VectorQ1Vec< ScalarType > tmp_pi_it(
                "cheby_est_tmpIt", *domains_[level], ownership_mask_[level] );
            VectorQ1Vec< ScalarType > tmp_pi_aux(
                "cheby_est_tmpAux", *domains_[level], ownership_mask_[level] );
            const auto log_level = prm_.mesh_parameters.refinement_level_mesh_min + level;
            auto&      A_lvl     = ( level == num_levels_ - 1 ) ? K_->block_11() : A_c_[level];
            linalg::DiagonallyScaledOperator< Viscous > inv_diag_A( A_lvl, inverse_diagonals_[level] );
            const double lmax_est = linalg::solvers::power_iteration(
                inv_diag_A,
                tmp_pi_it,
                tmp_pi_aux,
                prm_.stokes_solver_parameters.viscous_pc_num_power_iterations );
            logroot << "[Cheby estimate] level " << log_level
                    << ": lambda_max(D^-1 A_viscous) ~ " << lmax_est
                    << "  => lambda_max_cheby = " << 1.5 * lmax_est
                    << ", lambda_min_cheby = " << 0.1 * lmax_est << std::endl;
        }

        // ---------------- Coarse grid solver ----------------
        logroot << "Setting up multigrid coarse grid solver ..." << std::endl;
        coarse_grid_tmps_.reserve( 4 );
        for ( int i = 0; i < 4; i++ )
        {
            coarse_grid_tmps_.emplace_back(
                "tmp_coarse_grid", *domains_[0], ownership_mask_[0] );
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

            const auto orig_subdomain_to_rank =
                grid::shell::subdomain_to_rank_iterate_diamond_subdomains;

            for ( int L = 0; L < num_mg_levels - 1; ++L )
            {
                const int lat_level = prm_.mesh_parameters.refinement_level_mesh_min + L;
                const int rad_level = lat_level + prm_.mesh_parameters.radial_extra_levels;

                const MPI_Comm upper_comm   = agglom.comm( L + 1 );
                const int      upper_cf     = agglom.cum_factor( L + 1 );
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
                        ( upper_cf == 1 )
                            ? orig_subdomain_to_rank
                            : grid::shell::agglomerated_subdomain_to_rank( orig_subdomain_to_rank, upper_cf ) );
                    mask_upper_.push_back( grid::setup_node_ownership_mask_data( dom_up ) );
                    domains_upper_.push_back( std::make_shared< DistributedDomain >( std::move( dom_up ) ) );
                }

                tmp_mg_r_fine.emplace_back( "tmp_r_fine_L" + std::to_string( L ),
                                             *domains_upper_.back(), mask_upper_.back() );
                tmp_mg_e_fine.emplace_back( "tmp_e_fine_L" + std::to_string( L ),
                                             *domains_upper_.back(), mask_upper_.back() );

                redistribute_down.emplace_back(
                    *domains_upper_.back(),
                    *domains_[L],
                    ( upper_cf == 1 )
                        ? orig_subdomain_to_rank
                        : grid::shell::agglomerated_subdomain_to_rank( orig_subdomain_to_rank, upper_cf ),
                    agglom.subdomain_fn( L ) );
            }

            // Restrictions are halo'd on the upper comm under agglomeration.
            R_.clear();
            R_.reserve( num_mg_levels - 1 );
            for ( int L = 0; L < num_mg_levels - 1; ++L )
                R_.emplace_back( *domains_upper_[L] );
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

        // ---------------- Schur preconditioner ----------------
        logroot << "Setting up Schur complement preconditioner ..." << std::endl;
        k_pm_ = VectorQ1Scalar< ScalarType >(
            "k_pm", *domains_[pressure_level_], ownership_mask_[pressure_level_] );
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
            linalg::assign( tmp, 1.0 );
            linalg::apply( *pmass_, tmp, lumped_diagonal_pmass_ );
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

        // Wrap the velocity preconditioner in the type-erased handle. For now this
        // forwards to the double multigrid (*prec_11_); --stokes-mg-precision will
        // swap in a reduced-precision V-cycle impl here.
        VelPrecHandle vel_prec(
            std::make_shared< linalg::solvers::ForwardingPrecImpl< Viscous, PrecVisc > >( *prec_11_ ) );
        prec_stokes_ = std::make_unique< PrecStokes >(
            K_->block_11(), *pmass_, K_->block_12(), triangular_prec_tmp_, vel_prec, *inv_lumped_pmass_ );

        // ---------------- Outer FGMRES ----------------
        logroot << "Setting up FGMRES ... (Krylov basis precision: "
                << ( use_float_basis_ ? "single" : "double" ) << ")" << std::endl;
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

        log_hbm( "stokes: ctor end (delta = MG hierarchy + operators + coarse + preconditioner)" );
    }

    // Public accessors needed by the rest of the app.

    linalg::VectorQ1IsoQ2Q1< ScalarType >& solution() { return stok_vecs_["u"]; }
    linalg::VectorQ1Scalar< ScalarType >&  eta_fine() { return eta_[velocity_level_]; }
    long                                   num_dofs_pressure() const { return num_dofs_pressure_; }
    const grid::shell::BoundaryConditions& boundary_conditions() const { return bcs_; }

    /// Update the fine-level viscosity from the current temperature using the
    /// configured viscosity law.  No-op for ViscosityLaw::CONSTANT.  Coarse-
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
                prm_.physics_parameters.viscosity_parameters.rmu,
                eta_[velocity_level_].grid_data(),
                T.grid_data() } );
        Kokkos::fence();
    }

    /// Solve  K · u = f(T_for_buoyancy)  with the configured FGMRES + MG/Schur
    /// preconditioner.  When `log_convergence` is true, the per-step Stokes
    /// and coarse-grid PCG tables are printed; in either case the table is
    /// cleared at the end of the call.
    void solve( const linalg::VectorQ1Scalar< ScalarType >& T_for_buoyancy, bool log_convergence )
    {
        util::Timer timer_stokes( "stokes" );

        util::logroot << "Setting up Stokes rhs ..." << std::endl;

        Kokkos::parallel_for(
            "Stokes rhs interpolation",
            grid::shell::local_domain_md_range_policy_nodes( *domains_[velocity_level_] ),
            RHSVelocityInterpolator(
                coords_shell_[velocity_level_],
                coords_radii_[velocity_level_],
                stok_vecs_["tmp"].block_1().grid_data(),
                T_for_buoyancy.grid_data(),
                prm_.physics_parameters.rayleigh_number ) );

        linalg::apply( *M_, stok_vecs_["tmp"].block_1(), stok_vecs_["f"].block_1() );

        fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
            stok_vecs_["f"],
            boundary_mask_[velocity_level_],
            grid::shell::get_shell_boundary_flag( bcs_, grid::shell::BoundaryConditionFlag::DIRICHLET ) );

        fe::strong_algebraic_freeslip_enforcement_in_place(
            stok_vecs_["f"],
            coords_shell_[velocity_level_],
            boundary_mask_[velocity_level_],
            grid::shell::get_shell_boundary_flag( bcs_, grid::shell::BoundaryConditionFlag::FREESLIP ) );

        util::logroot << "Solving Stokes ..." << std::endl;

        if ( use_float_basis_ )
            ::terra::linalg::solvers::solve( *stokes_fgmres_float_, *K_, stok_vecs_["u"], stok_vecs_["f"] );
        else
            ::terra::linalg::solvers::solve( *stokes_fgmres_double_, *K_, stok_vecs_["u"], stok_vecs_["f"] );

        if ( log_convergence )
        {
            table_->query_rows_equals( "tag", "stokes_fgmres" ).print_pretty();
            table_->query_rows_equals( "tag", "coarse_grid_pcg" ).print_pretty();
        }
        table_->clear();

        // "Normalize" pressure (subtract average).
        auto& p = stok_vecs_["u"].block_2();
        const ScalarType avg_pressure_approximation =
            kernels::common::masked_sum(
                p.grid_data(), p.mask_data(), grid::NodeOwnershipFlag::OWNED ) /
            static_cast< ScalarType >( num_dofs_pressure_ );
        linalg::lincomb( p, { 1.0 }, { p }, -avg_pressure_approximation );
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
    grid::shell::BoundaryConditions                                         bcs_{};  // velocity BC set (owned copy)
    const Parameters&                                                       prm_;
    std::shared_ptr< util::Table >                                          table_;

    int  num_levels_;
    int  velocity_level_;
    int  pressure_level_;
    long num_dofs_pressure_ = 0;

    // Owned per-level state. Order matters: data members declared earlier are
    // destroyed later, so things that other members hold by-reference (eta_,
    // stok_vecs_, *_tmp_*) must come first.
    std::vector< linalg::VectorQ1Scalar< ScalarType > >    eta_;
    linalg::VectorQ1Scalar< ScalarType >                   GCAElements_;
    std::map< std::string, linalg::VectorQ1IsoQ2Q1< ScalarType > > stok_vecs_;
    std::vector< linalg::VectorQ1IsoQ2Q1< ScalarType > >   stokes_tmp_fgmres_;   // double path
    std::vector< linalg::VectorQ1IsoQ2Q1< ScalarType > >   stokes_work_fgmres_;  // float-basis path: scratch
    std::vector< BasisVectorType >                         stokes_basis_fgmres_; // float-basis path: basis
    std::vector< linalg::VectorQ1Vec< ScalarType > >       tmp_mg_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       tmp_mg_2_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       tmp_mg_r_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       tmp_mg_e_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       inverse_diagonals_;
    std::vector< linalg::VectorQ1Vec< ScalarType > >       coarse_grid_tmps_;

    // Heavy operators / solvers held via unique_ptr so we can construct in
    // body order (rather than fighting member-init order).
    std::unique_ptr< Stokes >                              K_;
    std::unique_ptr< Stokes >                              K_neumann_;
    std::unique_ptr< ViscousMass >                         M_;
    std::vector< Viscous >                                 A_c_;
    std::vector< Prolongation >                            P_;
    std::vector< Restriction >                             R_;
    std::vector< Smoother >                                smoothers_;
    std::unique_ptr< CoarseGridSolver >                    coarse_grid_solver_;

    // Comm-aware MG agglomeration (empty/no-op when agglom_factors is all 1s).
    std::vector< std::shared_ptr< grid::shell::DistributedDomain > >       domains_upper_;
    std::vector< grid::Grid4DDataScalar< grid::NodeOwnershipFlag > >       mask_upper_;

    std::unique_ptr< PrecVisc >                            prec_11_;

    // Schur preconditioner pieces.
    linalg::VectorQ1Scalar< ScalarType >                   k_pm_;
    std::unique_ptr< PressureMass >                        pmass_;
    linalg::VectorQ1Scalar< ScalarType >                   lumped_diagonal_pmass_;
    std::unique_ptr< PrecSchur >                           inv_lumped_pmass_;

    // Outer Stokes preconditioner / solver. Exactly one of the two FGMRES variants
    // is allocated, selected by use_float_basis_ (--stokes-float-krylov-basis).
    linalg::VectorQ1IsoQ2Q1< ScalarType >                  triangular_prec_tmp_;
    std::unique_ptr< PrecStokes >                          prec_stokes_;
    bool                                                   use_float_basis_ = false;
    std::unique_ptr< FGMRESDouble >                        stokes_fgmres_double_;
    std::unique_ptr< FGMRESFloat >                         stokes_fgmres_float_;
};

} // namespace terra::mantlecirculation
