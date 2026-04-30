#pragma once

#include <memory>
#include <vector>

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/entropy_viscosity.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg_kerngen.hpp"
#include "fe/wedge/operators/shell/wedge_constant_div_k_grad.hpp"
#include "fe/wedge/operators/shell/wedge_lumped_lap_projector.hpp"
#include "fv/hex/conversion.hpp"
#include "fv/hex/operators/fct_advection_diffusion.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/solvers/diagonal_solver.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

#include "parameters.hpp"

namespace terra::mantlecirculation {

/// Abstract energy-equation solver: one step advances the temperature state
/// from t to t + dt using a scheme-specific update.  Concrete subclasses own
/// all of their scheme-specific state (operators, solver, scratch); the call
/// site only needs `compute_dt` and `step`.
template < typename ScalarType >
class EnergySolver
{
  public:
    virtual ~EnergySolver() = default;

    /// CFL/stability-bound dt for the scheme at the current velocity field.
    virtual ScalarType compute_dt() = 0;

    /// Take one timestep.  `print_convergence` controls whether per-step
    /// solver tables are printed (typically only on the final Picard pass).
    virtual void step( ScalarType dt, bool print_convergence ) = 0;

    /// Save start-of-timestep state so subsequent Picard iterations can
    /// re-do the energy update from the same starting point.  No-op for
    /// schemes whose `step` doesn't mutate prognostic state outside `T`.
    virtual void snapshot_for_picard() {}

    /// Restore to the snapshotted state.  Called before each Picard
    /// iteration > 0.  No-op for schemes that don't need it.
    virtual void restore_for_picard() {}
};

/// Implicit Galerkin SUPG advection-diffusion energy solve.
///
/// Operator: A = M + dt · (K_diff + K_adv + K_supg), Dirichlet rows treated
/// strongly.  Inverse diagonal recomputed each step (dt changes); the solver
/// is FGMRES with a Jacobi preconditioner.
template < typename ScalarType >
class SUPGSolver : public EnergySolver< ScalarType >
{
    using AD       = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;
    using TempMass = fe::wedge::operators::shell::Mass< ScalarType >;
    using DiagSolverT = linalg::solvers::DiagonalSolver< AD >;
    using FGMRESType  = linalg::solvers::FGMRES< AD, DiagSolverT >;

  public:
    SUPGSolver(
        const std::shared_ptr< grid::shell::DistributedDomain >&            domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&                         coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&                         coords_radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&     boundary_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&            ownership_mask,
        const linalg::VectorQ1Vec< ScalarType, 3 >&                         velocity,
        linalg::VectorQ1Scalar< ScalarType >&                               T,
        ScalarType                                                          h,
        const Parameters&                                                   prm,
        std::shared_ptr< util::Table >                                      table )
    : domain_( domain )
    , coords_shell_( coords_shell )
    , coords_radii_( coords_radii )
    , boundary_mask_( boundary_mask )
    , ownership_mask_( ownership_mask )
    , velocity_( velocity )
    , T_( T )
    , h_( h )
    , prm_( prm )
    , table_( std::move( table ) )
    , g_( "supg_g", *domain_, ownership_mask_ )
    , tmp_( "supg_tmp", *domain_, ownership_mask_ )
    , q_( "supg_q", *domain_, ownership_mask_ )
    , diag_( "supg_diag", *domain_, ownership_mask_ )
    , T_backup_( "supg_T_backup", *domain_, ownership_mask_ )
    {
        util::logroot << "Setting up SUPG energy solver ..." << std::endl;

        A_ = std::make_unique< AD >(
            *domain_, coords_shell_, coords_radii_, boundary_mask_, velocity_,
            prm_.physics_parameters.diffusivity, ScalarType( 0 ), /*treat_boundary=*/true );

        A_neumann_ = std::make_unique< AD >(
            *domain_, coords_shell_, coords_radii_, boundary_mask_, velocity_,
            prm_.physics_parameters.diffusivity, ScalarType( 0 ), /*treat_boundary=*/false );

        A_neumann_diag_ = std::make_unique< AD >(
            *domain_, coords_shell_, coords_radii_, boundary_mask_, velocity_,
            prm_.physics_parameters.diffusivity, ScalarType( 0 ), /*treat_boundary=*/false, /*diagonal=*/true );

        M_ = std::make_unique< TempMass >( *domain_, coords_shell_, coords_radii_, false );

        // Diagonal of the SUPG operator at a representative dt; recomputed every step.
        A_neumann_diag_->dt() = ScalarType( 1e-4 );
        linalg::assign( diag_, ScalarType( 0 ) );
        {
            linalg::VectorQ1Scalar< ScalarType > ones( "ones", *domain_, ownership_mask_ );
            linalg::assign( ones, ScalarType( 1 ) );
            linalg::apply( *A_neumann_diag_, ones, diag_ );
        }

        constexpr int num_gmres_tmps = 14;
        tmp_gmres_.reserve( num_gmres_tmps );
        for ( int i = 0; i < num_gmres_tmps; ++i )
        {
            tmp_gmres_.emplace_back( "tmp_energy_gmres", *domain_, ownership_mask_ );
        }

        solver_ = std::make_unique< FGMRESType >(
            tmp_gmres_,
            linalg::solvers::FGMRESOptions{
                .restart                     = prm_.energy_solver_parameters.krylov_restart,
                .relative_residual_tolerance = prm_.energy_solver_parameters.krylov_relative_tolerance,
                .absolute_residual_tolerance = prm_.energy_solver_parameters.krylov_absolute_tolerance,
                .max_iterations              = prm_.energy_solver_parameters.krylov_max_iterations },
            table_,
            DiagSolverT( diag_ ) );

        util::logroot << "SUPG energy solver ready." << std::endl;
    }

    ScalarType compute_dt() override
    {
        // SUPG: implicit diffusion, dt only constrained by advection CFL.
        const auto max_vel = kernels::common::max_vector_magnitude( velocity_.grid_data() );
        const auto dt_advection = ( max_vel > ScalarType( 1e-12 ) ) ? ( h_ / max_vel ) : ScalarType( 1e-3 );
        const auto dt = prm_.time_stepping_parameters.dt_scaling * dt_advection;

        util::logroot << "Computing dt (SUPG advection CFL) ..." << std::endl;
        util::logroot << "    max_vel:                       " << max_vel << std::endl;
        util::logroot << "    h:                             " << h_ << std::endl;
        util::logroot << "=>  dt (= dt_scaling * h/v_max):   " << dt << std::endl;
        return dt;
    }

    void snapshot_for_picard() override
    {
        Kokkos::deep_copy( T_backup_.grid_data(), T_.grid_data() );
    }

    void restore_for_picard() override
    {
        Kokkos::deep_copy( T_.grid_data(), T_backup_.grid_data() );
    }

    void step( ScalarType dt, bool print_convergence ) override
    {
        util::Timer timer_energy( "energy" );
        util::logroot << "Setting up energy solve ..." << std::endl;

        A_->dt()              = dt;
        A_neumann_->dt()      = dt;
        A_neumann_diag_->dt() = dt;

        // Update inverse diagonal each step (dt changed).
        {
            linalg::VectorQ1Scalar< ScalarType > ones( "ones", *domain_, ownership_mask_ );
            linalg::assign( ones, ScalarType( 1 ) );
            linalg::apply( *A_neumann_diag_, ones, diag_ );
            linalg::invert_entries( diag_ );
        }

        for ( int i = 0; i < prm_.time_stepping_parameters.energy_substeps; ++i )
        {
            util::logroot << "Solving energy (SUPG, substep " << i << ") ..." << std::endl;

            // RHS: q = M · T^n.
            linalg::apply( *M_, T_, q_ );

            // Dirichlet BC vector g.
            linalg::assign( g_, ScalarType( 0 ) );
            {
                auto       g_grid    = g_.grid_data();
                auto       mask      = boundary_mask_;
                const auto T_cmb_val = static_cast< ScalarType >( prm_.boundary_conditions_parameters.temperature_cmb );
                const auto T_top_val = static_cast< ScalarType >( prm_.boundary_conditions_parameters.temperature_surface );
                Kokkos::parallel_for(
                    "supg_dirichlet_g",
                    grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                    KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                        const auto flag = mask( sd, x, y, r );
                        if ( flag == grid::shell::ShellBoundaryFlag::CMB )
                            g_grid( sd, x, y, r ) = T_cmb_val;
                        else if ( flag == grid::shell::ShellBoundaryFlag::SURFACE )
                            g_grid( sd, x, y, r ) = T_top_val;
                    } );
                Kokkos::fence();
            }

            // Eliminate Dirichlet BCs from RHS.
            fe::strong_algebraic_dirichlet_enforcement_poisson_like(
                *A_neumann_, *A_neumann_diag_, g_, tmp_, q_,
                boundary_mask_, grid::shell::ShellBoundaryFlag::BOUNDARY );

            // Solve (M + dt · A) T^{n+1} = q.
            solve( *solver_, *A_, T_, q_ );

            if ( print_convergence )
            {
                table_->query_rows_equals( "tag", "fgmres_solver" ).print_pretty();
            }
            table_->clear();
        }
    }

  private:
    // Borrowed inputs.
    std::shared_ptr< grid::shell::DistributedDomain >                       domain_;
    const grid::Grid3DDataVec< ScalarType, 3 >&                             coords_shell_;
    const grid::Grid2DDataScalar< ScalarType >&                             coords_radii_;
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&         boundary_mask_;
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&                ownership_mask_;
    const linalg::VectorQ1Vec< ScalarType, 3 >&                             velocity_;
    linalg::VectorQ1Scalar< ScalarType >&                                   T_;
    ScalarType                                                              h_;
    const Parameters&                                                       prm_;
    std::shared_ptr< util::Table >                                          table_;

    // Owned state.
    std::unique_ptr< AD >                                                   A_, A_neumann_, A_neumann_diag_;
    std::unique_ptr< TempMass >                                             M_;
    std::unique_ptr< FGMRESType >                                           solver_;
    linalg::VectorQ1Scalar< ScalarType >                                    g_, tmp_, q_, diag_;
    linalg::VectorQ1Scalar< ScalarType >                                    T_backup_;
    std::vector< linalg::VectorQ1Scalar< ScalarType > >                     tmp_gmres_;
};

/// Implicit Galerkin energy solve with explicit lagged entropy-viscosity
/// stabilization (KHB / ASPECT recipe).  LHS is pure-Galerkin AD (SUPG OFF);
/// stabilization is added to the RHS as `-dt · DivKGrad(ν_h) · T^n`.
template < typename ScalarType >
class EVSolver : public EnergySolver< ScalarType >
{
    using AD_EV     = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;
    using TempMass  = fe::wedge::operators::shell::Mass< ScalarType >;
    using EVDiffOp  = fe::wedge::operators::shell::WedgeConstantDivKGrad< ScalarType >;
    using LapProj   = fe::wedge::operators::shell::WedgeLumpedLapProjector< ScalarType >;
    using DiagSolverT = linalg::solvers::DiagonalSolver< AD_EV >;
    using FGMRESType  = linalg::solvers::FGMRES< AD_EV, DiagSolverT >;

  public:
    EVSolver(
        const std::shared_ptr< grid::shell::DistributedDomain >&            domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&                         coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&                         coords_radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&     boundary_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&            ownership_mask,
        const linalg::VectorQ1Vec< ScalarType, 3 >&                         velocity,
        linalg::VectorQ1Scalar< ScalarType >&                               T,
        ScalarType                                                          h,
        const Parameters&                                                   prm,
        std::shared_ptr< util::Table >                                      table )
    : domain_( domain )
    , coords_shell_( coords_shell )
    , coords_radii_( coords_radii )
    , boundary_mask_( boundary_mask )
    , ownership_mask_( ownership_mask )
    , velocity_( velocity )
    , T_( T )
    , h_( h )
    , prm_( prm )
    , table_( std::move( table ) )
    , g_(           "ev_g",          *domain_, ownership_mask_ )
    , tmp_(         "ev_tmp",        *domain_, ownership_mask_ )
    , q_(           "ev_q",          *domain_, ownership_mask_ )
    , diag_(        "ev_diag",       *domain_, ownership_mask_ )
    , T_prev_(      "T_prev",        *domain_, ownership_mask_ )
    , rhs_ev_(      "rhs_ev",        *domain_, ownership_mask_ )
    , T_backup_(      "ev_T_backup",      *domain_, ownership_mask_ )
    , T_prev_backup_( "ev_T_prev_backup", *domain_, ownership_mask_ )
    {
        util::logroot << "Setting up entropy-viscosity (EV) energy solver ..." << std::endl;

        // Per-wedge ν_h field: extents (#subdomains, N-1, N-1, N_r-1, num_wedges).
        const auto num_sub = static_cast< long long >( domain_->subdomains().size() );
        const auto nx_c    = domain_->domain_info().subdomain_num_nodes_per_side_laterally() - 1;
        const auto nr_c    = domain_->domain_info().subdomain_num_nodes_radially() - 1;
        nu_h_wedge_ = grid::Grid5DDataScalar< ScalarType >(
            "nu_h_wedge", num_sub, nx_c, nx_c, nr_c, fe::wedge::num_wedges_per_hex_cell );

        A_ = std::make_unique< AD_EV >(
            *domain_, coords_shell_, coords_radii_, boundary_mask_, velocity_,
            prm_.physics_parameters.diffusivity, ScalarType( 0 ), /*treat_boundary=*/true );
        A_->set_supg_enabled( false );

        A_neumann_ = std::make_unique< AD_EV >(
            *domain_, coords_shell_, coords_radii_, boundary_mask_, velocity_,
            prm_.physics_parameters.diffusivity, ScalarType( 0 ), /*treat_boundary=*/false );
        A_neumann_->set_supg_enabled( false );

        A_neumann_diag_ = std::make_unique< AD_EV >(
            *domain_, coords_shell_, coords_radii_, boundary_mask_, velocity_,
            prm_.physics_parameters.diffusivity, ScalarType( 0 ), /*treat_boundary=*/false, /*diagonal=*/true );
        A_neumann_diag_->set_supg_enabled( false );

        M_ = std::make_unique< TempMass >( *domain_, coords_shell_, coords_radii_, false );

        // Per-wedge lumped-mass projection of κ·∇²T (κ folded into the
        // precomputed projector at construction time).
        lap_projector_ = std::make_unique< LapProj >(
            *domain_, coords_shell_, coords_radii_, prm_.physics_parameters.diffusivity );
        lap_w_ = lap_projector_->make_lap_storage( "ev_lap_w" );

        // ν_h read by reference; the underlying view is updated in place
        // each step by compute_nu_h.
        A_evdiff_ = std::make_unique< EVDiffOp >(
            *domain_, coords_shell_, coords_radii_, nu_h_wedge_ );

        A_neumann_diag_->dt() = ScalarType( 1e-4 );
        linalg::assign( diag_, ScalarType( 0 ) );
        {
            linalg::VectorQ1Scalar< ScalarType > ones( "ev_setup_ones", *domain_, ownership_mask_ );
            linalg::assign( ones, ScalarType( 1 ) );
            linalg::apply( *A_neumann_diag_, ones, diag_ );
        }

        constexpr int num_gmres_tmps = 14;
        tmp_gmres_.reserve( num_gmres_tmps );
        for ( int i = 0; i < num_gmres_tmps; ++i )
        {
            tmp_gmres_.emplace_back( "tmp_ev_gmres", *domain_, ownership_mask_ );
        }

        solver_ = std::make_unique< FGMRESType >(
            tmp_gmres_,
            linalg::solvers::FGMRESOptions{
                .restart                     = prm_.energy_solver_parameters.krylov_restart,
                .relative_residual_tolerance = prm_.energy_solver_parameters.krylov_relative_tolerance,
                .absolute_residual_tolerance = prm_.energy_solver_parameters.krylov_absolute_tolerance,
                .max_iterations              = prm_.energy_solver_parameters.krylov_max_iterations },
            table_,
            DiagSolverT( diag_ ) );

        // Bootstrap T_prev = T so ∂_t E = 0 on step 1.
        Kokkos::deep_copy( T_prev_.grid_data(), T_.grid_data() );

        util::logroot << "EV energy solver ready." << std::endl;
    }

    ScalarType compute_dt() override
    {
        const auto max_vel = kernels::common::max_vector_magnitude( velocity_.grid_data() );
        const auto dt_advection = ( max_vel > ScalarType( 1e-12 ) ) ? ( h_ / max_vel ) : ScalarType( 1e-3 );
        const auto dt = prm_.time_stepping_parameters.dt_scaling * dt_advection;

        util::logroot << "Computing dt (EV advection CFL) ..." << std::endl;
        util::logroot << "    max_vel:                       " << max_vel << std::endl;
        util::logroot << "    h:                             " << h_ << std::endl;
        util::logroot << "=>  dt (= dt_scaling * h/v_max):   " << dt << std::endl;
        return dt;
    }

    void snapshot_for_picard() override
    {
        // Both T and T_prev mutate inside step() (the latter via the BDF1
        // history rotation), so we snapshot the (T, T_prev) pair.
        Kokkos::deep_copy( T_backup_.grid_data(),      T_.grid_data() );
        Kokkos::deep_copy( T_prev_backup_.grid_data(), T_prev_.grid_data() );
        // Mark ν_h stale at the start of a new timestep; the first Picard
        // iteration's substep-0 will compute it.  Subsequent Picard
        // iterations of the same timestep reuse it so the explicit-lagged
        // stabilization stays consistent across the (T, u) Picard fixed
        // point.  Substeps > 0 always recompute (T evolves between them).
        nu_h_locked_for_step_ = false;
    }

    void restore_for_picard() override
    {
        Kokkos::deep_copy( T_.grid_data(),      T_backup_.grid_data() );
        Kokkos::deep_copy( T_prev_.grid_data(), T_prev_backup_.grid_data() );
    }

    void step( ScalarType dt, bool print_convergence ) override
    {
        util::Timer timer_energy( "energy" );
        util::logroot << "Setting up energy solve ..." << std::endl;

        A_->dt()              = dt;
        A_neumann_->dt()      = dt;
        A_neumann_diag_->dt() = dt;

        {
            linalg::VectorQ1Scalar< ScalarType > ones( "ev_step_ones", *domain_, ownership_mask_ );
            linalg::assign( ones, ScalarType( 1 ) );
            linalg::assign( diag_, ScalarType( 0 ) );
            linalg::apply( *A_neumann_diag_, ones, diag_ );
            linalg::invert_entries( diag_ );
        }

        const ScalarType gamma =
            prm_.physics_parameters.constant_internal_heating
                ? static_cast< ScalarType >( prm_.physics_parameters.constant_internal_heating_value )
                : ScalarType( 0 );

        for ( int i = 0; i < prm_.time_stepping_parameters.energy_substeps; ++i )
        {
            util::logroot << "Solving energy (EV, substep " << i << ") ..." << std::endl;

            // 1+2) per-wedge lap projection and ν_h.  Skipped on Picard
            // iterations > 0 of the first substep so all Picard sweeps see
            // the same explicit-lagged stabilization field; substeps beyond
            // the first always recompute since T evolves between them.
            const bool need_nu_h = ( i > 0 ) || !nu_h_locked_for_step_;
            if ( need_nu_h )
            {
                // 1) lap_w[wedge](j) = κ · (K_w · T_w / M_w_lumped)(j)
                //    via precomputed per-wedge projector — strictly local.
                lap_projector_->apply( T_, lap_w_ );

                // 2) Entropy stats (volume-weighted E_avg) and per-wedge ν_h.
                const auto stats = fe::wedge::operators::shell::compute_entropy_stats(
                    T_, ownership_mask_, *domain_, coords_shell_, coords_radii_, ev_params_ );
                fe::wedge::operators::shell::compute_nu_h(
                    nu_h_wedge_, T_, T_prev_, velocity_, lap_w_,
                    *domain_, coords_shell_, coords_radii_,
                    dt, stats, ev_params_, gamma );
            }
            if ( i == 0 )
            {
                nu_h_locked_for_step_ = true;
            }

            // 3) Explicit EV diffusion contribution: rhs_ev = ∫ ν_h ∇T · ∇φ_i.
            linalg::apply( *A_evdiff_, T_, rhs_ev_ );

            // 4) RHS:  q = M·T^n  -  dt · rhs_ev.
            linalg::apply( *M_, T_, q_ );
            linalg::lincomb( q_, { ScalarType( 1 ), -dt }, { q_, rhs_ev_ } );

            // 5) History rotation BEFORE the solve overwrites T.
            Kokkos::deep_copy( T_prev_.grid_data(), T_.grid_data() );

            // 6) Dirichlet BC vector + elimination from RHS.
            linalg::assign( g_, ScalarType( 0 ) );
            {
                auto       g_grid    = g_.grid_data();
                auto       mask      = boundary_mask_;
                const auto T_cmb_val = static_cast< ScalarType >( prm_.boundary_conditions_parameters.temperature_cmb );
                const auto T_top_val = static_cast< ScalarType >( prm_.boundary_conditions_parameters.temperature_surface );
                Kokkos::parallel_for(
                    "ev_dirichlet_g",
                    grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                    KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                        const auto flag = mask( sd, x, y, r );
                        if ( flag == grid::shell::ShellBoundaryFlag::CMB )
                            g_grid( sd, x, y, r ) = T_cmb_val;
                        else if ( flag == grid::shell::ShellBoundaryFlag::SURFACE )
                            g_grid( sd, x, y, r ) = T_top_val;
                    } );
                Kokkos::fence();
            }

            fe::strong_algebraic_dirichlet_enforcement_poisson_like(
                *A_neumann_, *A_neumann_diag_, g_, tmp_, q_,
                boundary_mask_, grid::shell::ShellBoundaryFlag::BOUNDARY );

            // 7) Solve (M + dt · A_galerkin) T^{n+1} = q.
            solve( *solver_, *A_, T_, q_ );

            if ( print_convergence )
            {
                table_->query_rows_equals( "tag", "fgmres_solver" ).print_pretty();
            }
            table_->clear();
        }
    }

  private:
    std::shared_ptr< grid::shell::DistributedDomain >                       domain_;
    const grid::Grid3DDataVec< ScalarType, 3 >&                             coords_shell_;
    const grid::Grid2DDataScalar< ScalarType >&                             coords_radii_;
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&         boundary_mask_;
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&                ownership_mask_;
    const linalg::VectorQ1Vec< ScalarType, 3 >&                             velocity_;
    linalg::VectorQ1Scalar< ScalarType >&                                   T_;
    ScalarType                                                              h_;
    const Parameters&                                                       prm_;
    std::shared_ptr< util::Table >                                          table_;

    std::unique_ptr< AD_EV >                                                A_, A_neumann_, A_neumann_diag_;
    std::unique_ptr< TempMass >                                             M_;
    std::unique_ptr< EVDiffOp >                                             A_evdiff_;
    std::unique_ptr< LapProj >                                              lap_projector_;
    std::unique_ptr< FGMRESType >                                           solver_;

    linalg::VectorQ1Scalar< ScalarType >                                    g_, tmp_, q_, diag_;
    linalg::VectorQ1Scalar< ScalarType >                                    T_prev_;
    linalg::VectorQ1Scalar< ScalarType >                                    rhs_ev_;
    linalg::VectorQ1Scalar< ScalarType >                                    T_backup_;
    linalg::VectorQ1Scalar< ScalarType >                                    T_prev_backup_;
    grid::Grid5DDataScalar< ScalarType >                                    nu_h_wedge_;
    typename LapProj::LapStorage                                            lap_w_;
    fe::wedge::operators::shell::EntropyViscosityParameters< ScalarType >   ev_params_{};
    std::vector< linalg::VectorQ1Scalar< ScalarType > >                     tmp_gmres_;

    // Locked-by-Picard flag: false at the start of each timestep (set by
    // snapshot_for_picard), set to true once substep-0 has computed ν_h so
    // subsequent Picard iterations of the same step skip the recompute.
    bool                                                                    nu_h_locked_for_step_ = false;
};

/// Explicit FCT energy update on the FV mesh, with L2 projection onto Q1 at
/// the end of each timestep so downstream consumers (Stokes buoyancy, Nu) see
/// the updated Q1 temperature.
template < typename ScalarType >
class FCTSolver : public EnergySolver< ScalarType >
{
  public:
    FCTSolver(
        const std::shared_ptr< grid::shell::DistributedDomain >&                domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&                             coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&                             coords_radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&         boundary_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&                ownership_mask,
        const linalg::VectorQ1Vec< ScalarType, 3 >&                             velocity,
        linalg::VectorQ1Scalar< ScalarType >&                                   T,
        linalg::VectorFVScalar< ScalarType >&                                   T_fct,
        const linalg::VectorFVVec< ScalarType, 3 >&                             fv_cell_centers,
        const fv::hex::DirichletBCs< ScalarType >&                              fct_bcs,
        const Parameters&                                                       prm,
        std::shared_ptr< util::Table >                                          table )
    : domain_( domain )
    , coords_shell_( coords_shell )
    , coords_radii_( coords_radii )
    , boundary_mask_( boundary_mask )
    , velocity_( velocity )
    , T_( T )
    , T_fct_( T_fct )
    , fv_cell_centers_( fv_cell_centers )
    , fct_bcs_( fct_bcs )
    , prm_( prm )
    , table_( std::move( table ) )
    , T_source_( "T_source", *domain_ )
    , T_fct_backup_( "T_fct_backup", *domain_ )
    , fv_fct_bufs_( *domain_ )
    {
        linalg::assign( T_source_, ScalarType( 0 ) );

        // l2_project_fv_to_fe needs at least 5 Q1 scalar temporaries.
        constexpr int num_l2_proj_tmps = 5;
        l2_proj_tmps_.reserve( num_l2_proj_tmps );
        for ( int i = 0; i < num_l2_proj_tmps; ++i )
        {
            l2_proj_tmps_.emplace_back(
                "fct_l2_proj_tmp_" + std::to_string( i ), *domain_, ownership_mask );
        }
    }

    void snapshot_for_picard() override
    {
        Kokkos::deep_copy( T_fct_backup_.grid_data(), T_fct_.grid_data() );
    }

    void restore_for_picard() override
    {
        Kokkos::deep_copy( T_fct_.grid_data(), T_fct_backup_.grid_data() );
    }

    ScalarType compute_dt() override
    {
        const auto dt_stable = fv::hex::operators::compute_dt_stable(
            *domain_, velocity_, fv_cell_centers_.grid_data(),
            coords_shell_, coords_radii_, prm_.physics_parameters.diffusivity );
        const auto dt = prm_.time_stepping_parameters.dt_scaling * dt_stable;

        util::logroot << "Computing dt (FCT stable) ..." << std::endl;
        util::logroot << "    dt_stable:                     " << dt_stable << std::endl;
        util::logroot << "=>  dt (= dt_stable * dt_scaling): " << dt << std::endl;
        return dt;
    }

    void step( ScalarType dt, bool /*print_convergence*/ ) override
    {
        util::Timer timer_energy( "energy" );
        util::logroot << "Setting up energy solve ..." << std::endl;

        {
            util::Timer timer_fct_substeps( "fct_substeps" );

            for ( int i = 0; i < prm_.time_stepping_parameters.energy_substeps; ++i )
            {
                util::logroot << "Solving energy (FCT, substep " << i << ") ..." << std::endl;

                {
                    util::Timer timer_fct_source_step( "fct_explicit_step_updating_source_term" );
                    if ( prm_.physics_parameters.constant_internal_heating )
                    {
                        linalg::assign( T_source_, prm_.physics_parameters.constant_internal_heating_value );
                    }
                    timer_fct_source_step.stop();

                    util::Timer timer_fct_step( "fct_explicit_step" );
                    fv::hex::operators::fct_explicit_step(
                        *domain_, T_fct_, velocity_, fv_cell_centers_.grid_data(),
                        coords_shell_, coords_radii_, dt, fv_fct_bufs_,
                        prm_.physics_parameters.diffusivity, T_source_.grid_data(),
                        /*subtract_divergence=*/true,
                        boundary_mask_, fct_bcs_ );
                    timer_fct_step.stop();
                }

                fv::hex::apply_dirichlet_bcs( T_fct_, boundary_mask_, fct_bcs_, *domain_ );
            }

            timer_fct_substeps.stop();
        }

        // Project T_fct -> Q1 T once after all substeps.
        {
            util::Timer timer_fct_projection( "fct_l2_projection" );
            fv::hex::l2_project_fv_to_fe(
                T_, T_fct_, *domain_, coords_shell_, coords_radii_, l2_proj_tmps_ );

            // Enforce Dirichlet BCs on the Q1 temperature.
            auto       T_grid    = T_.grid_data();
            auto       mask      = boundary_mask_;
            const auto T_cmb_val = static_cast< ScalarType >( prm_.boundary_conditions_parameters.temperature_cmb );
            const auto T_top_val = static_cast< ScalarType >( prm_.boundary_conditions_parameters.temperature_surface );
            Kokkos::parallel_for(
                "enforce_T_dirichlet_bcs",
                grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                    const auto flag = mask( sd, x, y, r );
                    if ( flag == grid::shell::ShellBoundaryFlag::CMB )
                        T_grid( sd, x, y, r ) = T_cmb_val;
                    else if ( flag == grid::shell::ShellBoundaryFlag::SURFACE )
                        T_grid( sd, x, y, r ) = T_top_val;
                } );
            Kokkos::fence();
        }
    }

  private:
    std::shared_ptr< grid::shell::DistributedDomain >                       domain_;
    const grid::Grid3DDataVec< ScalarType, 3 >&                             coords_shell_;
    const grid::Grid2DDataScalar< ScalarType >&                             coords_radii_;
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&         boundary_mask_;
    const linalg::VectorQ1Vec< ScalarType, 3 >&                             velocity_;
    linalg::VectorQ1Scalar< ScalarType >&                                   T_;
    linalg::VectorFVScalar< ScalarType >&                                   T_fct_;
    const linalg::VectorFVVec< ScalarType, 3 >&                             fv_cell_centers_;
    const fv::hex::DirichletBCs< ScalarType >&                              fct_bcs_;
    const Parameters&                                                       prm_;
    std::shared_ptr< util::Table >                                          table_;

    // Owned scratch.
    linalg::VectorFVScalar< ScalarType >                                    T_source_;
    linalg::VectorFVScalar< ScalarType >                                    T_fct_backup_;
    fv::hex::operators::FVFCTBuffers< ScalarType >                          fv_fct_bufs_;
    std::vector< linalg::VectorQ1Scalar< ScalarType > >                     l2_proj_tmps_;
};

} // namespace terra::mantlecirculation
