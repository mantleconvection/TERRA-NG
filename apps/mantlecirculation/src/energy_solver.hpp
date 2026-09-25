#pragma once

#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/entropy_viscosity.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/mmoc_transport.hpp"
#include "fe/wedge/operators/shell/shear_heating_kerngen.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg_kerngen.hpp"
#include "fe/wedge/operators/shell/wedge_constant_div_k_grad.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/solvers/diagonal_solver.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/fgmres_lowmem.hpp"
#include "linalg/vector_q1.hpp"
#include "linalg/vector_fv.hpp"
#include "parameters.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

#include "hbm_probe.hpp"
#include "interpolators.hpp"
#include "parameters.hpp"

namespace terra::mantlecirculation {

/// Radial (nondimensional) material profiles the energy equation needs, all normalised by
/// their reference values so an incompressible run has every one of them identically 1:
///   rho = rho_bar/rho_0, alpha = alpha/alpha_0, cp = cp_bar/cp_0,
///   kappa = k_hat/(rho*cp)  (the diffusion coefficient shape; k_hat is 1 while the code
///                            treats conductivity as constant).
/// Dividing the energy equation by rho_bar*cp_bar puts these into every term:
///   diffusion  kappa(r)/Pe        internal  H/cp(r)
///   adiabatic  Di*alpha(r)/cp(r)  shear     (Di*Pe/Ra)/(rho(r)*cp(r))
template < typename ScalarType >
struct RadialProfiles
{
    grid::Grid2DDataScalar< ScalarType > rho, alpha, cp, kappa;
    bool                                 valid = false;
};

/// Abstract energy-equation solver: one step advances the temperature state
/// from t to t + dt using a scheme-specific update.  Concrete subclasses own
/// all of their scheme-specific state (operators, solver, scratch); the call
/// site only needs `compute_dt` and `step`.
template < typename ScalarType >
class EnergySolver
{
  public:
    virtual ~EnergySolver() = default;

    /// Supply the radial material profiles. Called once after construction; solvers that
    /// do not use them ignore the call. Must come before the first step().
    virtual void set_radial_profiles( const RadialProfiles< ScalarType >& ) {}

    /// CFL/stability-bound dt for the scheme at the current velocity field.
    virtual ScalarType compute_dt( const int timestep ) = 0;

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

    /// Optional per-step diagnostics dump (called from the main loop at
    /// `output_frequency`).  Default is no-op.
    virtual void dump_diagnostics( int /*timestep*/, const std::string& /*outdir*/ ) {}

    /// Optional Q1-nodal diagnostic field (e.g., per-wedge ν_h projected to
    /// nodes for XDMF visualisation).  Returns nullptr if the scheme has no
    /// such field or it has not been enabled.  When non-null, the main loop
    /// registers it with XDMFOutput before the first write().
    virtual linalg::VectorQ1Scalar< ScalarType >* nu_h_nodal_view() { return nullptr; }

    /// Optional Q1-nodal projection of (K · T) / M_lumped — the lumped-mass
    /// Galerkin Laplacian feed to the EV residual.  Exposed so the main loop
    /// can dump it via radial_profiles for hypothesis-3 diagnosis (boundary
    /// flux dropped → wall-row |Lap| should be small relative to the first
    /// interior shell).  Returns nullptr unless the scheme allocates it.
    virtual linalg::VectorQ1Scalar< ScalarType >* lap_diag_view() { return nullptr; }

    /// Optional Q1-nodal projection of the per-wedge characteristic length
    /// h_w = V_wedge^{1/3} (cell-constant; scattered to nodes with
    /// count-normalisation).  Exposed for hypothesis-2 diagnosis (radial
    /// h_w over dr ≫ 1 → BL-normal over-damping).  Returns nullptr unless
    /// the scheme allocates it.
    virtual linalg::VectorQ1Scalar< ScalarType >* h_w_diag_view() { return nullptr; }
};

template < typename ScalarType >
ScalarType ramp_dt( const ScalarType dt, const int timestep, const int ramp_steps )
{
    constexpr ScalarType ramp_scale_start = 1e-4;
    constexpr ScalarType ramp_scale_end   = 0.5;

    if ( timestep > ramp_steps )
        return dt;

    const ScalarType scale =
        ramp_scale_start *
        std::pow( ramp_scale_end / ramp_scale_start, static_cast< ScalarType >( timestep - 1 ) / ( ramp_steps - 1 ) );

    return scale * dt;
}

/// Implicit Galerkin SUPG advection-diffusion energy solve.
///
/// Operator: A = M + dt · (K_diff + K_adv + K_supg), Dirichlet rows treated
/// strongly.  Inverse diagonal recomputed each step (dt changes); the solver
/// is FGMRES with a Jacobi preconditioner.
template < typename ScalarType >
void log_timestep_info(
    const Parameters& prm,
    const int         timestep,
    ScalarType        max_velocity,
    ScalarType        max_radial_h,
    ScalarType        dt_cfl,
    ScalarType        dt )
{
    const bool log_dimensional = prm.devel_parameters.output_dimensional;

    const ScalarType vel_scale  = log_dimensional ? prm.physics_parameters.calc_cm_per_year : ScalarType( 1 );
    const ScalarType grid_scale = log_dimensional ? prm.mesh_parameters.mantle_thickness_m : ScalarType( 1 );
    const ScalarType time_scale = log_dimensional ? prm.physics_parameters.calc_time_Ma : ScalarType( 1 );

    // Compute dimensional values if required
    max_velocity *= vel_scale;
    max_radial_h *= grid_scale;
    dt_cfl *= time_scale;
    dt *= time_scale;

    const ScalarType dt_max =
        log_dimensional ? prm.time_stepping_parameters.dt_max_Ma : prm.time_stepping_parameters.dt_max;
    const ScalarType dt_min =
        log_dimensional ? prm.time_stepping_parameters.dt_min_Ma : prm.time_stepping_parameters.dt_min;

    // Logging body
    util::logroot << "    max_vel" << ( log_dimensional ? " (cm/a) :               " : " :                      " )
                  << max_velocity << std::endl;
    util::logroot << "    h"
                  << ( log_dimensional ? " (m)   :                      " : " :                            " )
                  << max_radial_h << std::endl;
    util::logroot << "    cfl timestep size (= dt_scaling * h/v_max): " << dt_cfl << ( log_dimensional ? " Ma" : "" )
                  << std::endl;
    if ( dt_cfl > dt_max )
    {
        util::logroot << "....limiting maximum timestep size to " << dt_max
                      << ( log_dimensional ? " Ma....." : " ....." ) << std::endl;
    }
    else if ( dt_cfl < dt_min )
    {
        util::logroot << "....limiting minimum timestep size to " << dt_min
                      << ( log_dimensional ? " Ma....." : " ....." ) << std::endl;
    }
    if ( timestep <= prm.time_stepping_parameters.initial_dt_ramp_steps )
    {
        util::logroot << "....enforcing exponential ramp-up in first "
                      << prm.time_stepping_parameters.initial_dt_ramp_steps << " timesteps....." << std::endl;
    }
    util::logroot << "-------------------------------------------------" << std::endl;
    util::logroot << "=>   dt: " << dt << ( log_dimensional ? " Ma.\n" : ".\n" ) << std::endl;
}

/// Implicit Galerkin SUPG advection-diffusion energy solve.
///
/// Operator: A = M + dt · (K_diff + K_adv + K_supg), Dirichlet rows treated
/// strongly.  Inverse diagonal recomputed each step (dt changes); the solver
/// is FGMRES with a Jacobi preconditioner.

template < typename ScalarType >
class SUPGSolver : public EnergySolver< ScalarType >
{
    using AD          = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;
    using TempMass    = fe::wedge::operators::shell::Mass< ScalarType >;
    using DiagSolverT = linalg::solvers::DiagonalSolver< AD >;
    using FGMRESType  = linalg::solvers::FGMRES< AD, DiagSolverT >;

  public:
    /// Radial profiles: kappa goes into the three AD operators (they multiply the scalar
    /// diffusivity by it); the rest are read by the heating terms.
    void set_radial_profiles( const RadialProfiles< ScalarType >& p ) override
    {
        profiles_ = p;
        if ( !p.valid )
            return;
        for ( auto* op : { A_.get(), A_neumann_.get(), A_neumann_diag_.get() } )
            if ( op != nullptr )
                op->set_kappa_profile( p.kappa );
    }
    SUPGSolver(
        const std::shared_ptr< grid::shell::DistributedDomain >&        domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
        const linalg::VectorQ1Vec< ScalarType, 3 >&                     velocity,
        linalg::VectorQ1Scalar< ScalarType >&                           T,
        ScalarType                                                      h,
        const Parameters&                                               prm,
        std::shared_ptr< util::Table >                                  table )
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
    {
        // T_backup_ is only touched when there is more than one Picard iteration;
        // leave it empty (unallocated) otherwise to save a Q1-scalar field.
        if ( prm_.time_stepping_parameters.picard_iterations > 1 )
            T_backup_ = linalg::VectorQ1Scalar< ScalarType >( "supg_T_backup", *domain_, ownership_mask_ );

        util::logroot << "Setting up SUPG energy solver ..." << std::endl;

        A_ = std::make_unique< AD >(
            *domain_,
            coords_shell_,
            coords_radii_,
            boundary_mask_,
            velocity_,
            prm_.physics_parameters.thermal_diffusivity_nondim,
            ScalarType( 0 ),
            /*treat_boundary=*/true );

        A_neumann_ = std::make_unique< AD >(
            *domain_,
            coords_shell_,
            coords_radii_,
            boundary_mask_,
            velocity_,
            prm_.physics_parameters.thermal_diffusivity_nondim,
            ScalarType( 0 ),
            /*treat_boundary=*/false );

        A_neumann_diag_ = std::make_unique< AD >(
            *domain_,
            coords_shell_,
            coords_radii_,
            boundary_mask_,
            velocity_,
            prm_.physics_parameters.thermal_diffusivity_nondim,
            ScalarType( 0 ),
            /*treat_boundary=*/false,
            /*diagonal=*/true );

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

    ScalarType compute_dt( const int timestep ) override
    {
        // SUPG: implicit diffusion, dt only constrained by advection CFL.
        const auto max_vel      = kernels::common::max_vector_magnitude( velocity_.grid_data() );
        const auto dt_advection = h_ / max_vel;
        const auto dt_cfl       = prm_.time_stepping_parameters.dt_scaling * dt_advection;
        // The dt_min floor must never raise dt above the CFL-stable step: fast flows
        // (e.g. isoviscous Ra >= ~5e6) need dt << dt_min and blow up otherwise.
        const auto dt_min_eff = std::min( prm_.time_stepping_parameters.dt_min, dt_cfl );
        const auto dt         = std::clamp(
            ramp_dt( dt_cfl, timestep, prm_.time_stepping_parameters.initial_dt_ramp_steps ),
            dt_min_eff,
            prm_.time_stepping_parameters.dt_max );

        util::logroot << "Computing dt (SUPG advection CFL) ..." << std::endl;
        log_timestep_info( prm_, timestep, max_vel, h_, dt_cfl, dt );

        return dt;
    }

    void snapshot_for_picard() override
    {
        if ( prm_.time_stepping_parameters.picard_iterations > 1 )
            Kokkos::deep_copy( T_backup_.grid_data(), T_.grid_data() );
    }

    void restore_for_picard() override { Kokkos::deep_copy( T_.grid_data(), T_backup_.grid_data() ); }

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
                const auto T_cmb_val = static_cast< ScalarType >( prm_.boundary_parameters.temperature_max );
                const auto T_top_val = static_cast< ScalarType >( prm_.boundary_parameters.temperature_min );
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
                *A_neumann_, *A_neumann_diag_, g_, tmp_, q_, boundary_mask_, grid::shell::ShellBoundaryFlag::BOUNDARY );

            // Solve (M + dt · A) T^{n+1} = q.
            solve( *solver_, *A_, T_, q_ );

            if ( print_convergence )
            {
                util::logroot << "[SUPG energy FGMRES convergence]" << std::endl;
                table_->query_rows_equals( "tag", "fgmres_solver" ).print_pretty();
            }
            table_->clear();
        }
    }

  private:
    // Borrowed inputs.
    std::shared_ptr< grid::shell::DistributedDomain >               domain_;
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell_;
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii_;
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_;
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask_;
    const linalg::VectorQ1Vec< ScalarType, 3 >&                     velocity_;
    linalg::VectorQ1Scalar< ScalarType >&                           T_;
    ScalarType                                                      h_;
    const Parameters&                                               prm_;
    std::shared_ptr< util::Table >                                  table_;

    // Owned state.
    RadialProfiles< ScalarType > profiles_;
    std::unique_ptr< AD >                               A_, A_neumann_, A_neumann_diag_;
    std::unique_ptr< TempMass >                         M_;
    std::unique_ptr< FGMRESType >                       solver_;
    linalg::VectorQ1Scalar< ScalarType >                g_, tmp_, q_, diag_;
    linalg::VectorQ1Scalar< ScalarType >                T_backup_;
    std::vector< linalg::VectorQ1Scalar< ScalarType > > tmp_gmres_;
};

/// Implicit Galerkin energy solve with explicit lagged entropy-viscosity
/// stabilization (KHB / ASPECT recipe).  LHS is pure-Galerkin AD (SUPG OFF);
/// stabilization is added to the RHS as `-dt · DivKGrad(ν_h) · T^n`.
/// Composite LHS operator making the entropy-viscosity artificial diffusion
/// IMPLICIT:   C*x = A_*x  +  dt * P_int( A_evdiff_*x ),
/// where P_int zeros the nu_h contribution at Dirichlet boundary nodes so the
/// composite keeps A_'s diagonalized boundary rows.  Mirrors ASPECT (LHS assembly)
/// instead of the explicit RHS lag.  Enabled via --ev-implicit-nu-h.
template < typename ScalarT, typename BaseOp, typename EVOp >
class ImplicitEVAdvDiffOperator
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

    ImplicitEVAdvDiffOperator(
        BaseOp&                                                         base,
        EVOp&                                                           ev,
        const grid::shell::DistributedDomain&                          domain,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        SrcVectorType&                                                  scratch )
    : base_( base ), ev_( ev ), domain_( domain ), boundary_mask_( boundary_mask ), scratch_( scratch )
    {
    }

    ScalarT& dt() { return dt_; }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        linalg::apply( base_, src, dst );      // dst     = A_*src
        linalg::apply( ev_, src, scratch_ );   // scratch = A_evdiff_*src

        // Zero the nu_h contribution at Dirichlet boundary node rows so the
        // composite keeps A_'s diagonalized boundary rows (A_evdiff_ has none).
        {
            auto       s    = scratch_.grid_data();
            const auto mask = boundary_mask_;
            Kokkos::parallel_for(
                "impl_ev_zero_boundary_rows",
                grid::shell::local_domain_md_range_policy_nodes( domain_ ),
                KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                    const auto f = mask( sd, x, y, r );
                    if ( f == grid::shell::ShellBoundaryFlag::CMB ||
                         f == grid::shell::ShellBoundaryFlag::SURFACE )
                        s( sd, x, y, r ) = ScalarT( 0 );
                } );
            Kokkos::fence();
        }

        linalg::lincomb( dst, { ScalarT( 1 ), dt_ }, { dst, scratch_ } );  // dst += dt*scratch
    }

  private:
    BaseOp&                                                         base_;
    EVOp&                                                           ev_;
    const grid::shell::DistributedDomain&                          domain_;
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_;
    SrcVectorType&                                                 scratch_;
    ScalarT                                                        dt_ = ScalarT( 1 );
};

template < typename ScalarType >
class EVSolver : public EnergySolver< ScalarType >
{
    using AD_EV       = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;
    using TempMass    = fe::wedge::operators::shell::Mass< ScalarType >;
    using EVDiffOp    = fe::wedge::operators::shell::WedgeConstantDivKGrad< ScalarType >;
    using DiagSolverT = linalg::solvers::DiagonalSolver< AD_EV >;
    using FGMRESDouble = linalg::solvers::FGMRES< AD_EV, DiagSolverT >;
    // Reduced-precision Krylov basis variant (operator stays double). FP16 storage
    // (native __half on HIP): basis is store-only + convert, so no half arithmetic.
    using BasisVecT    = linalg::VectorQ1Scalar< Kokkos::Experimental::bhalf_t >;
    using FGMRESFloat  = linalg::solvers::FGMRESLowMem< AD_EV, BasisVecT, DiagSolverT >;

  public:
    /// Radial profiles: kappa goes into the three AD operators (they multiply the scalar
    /// diffusivity by it); the rest are read by the heating terms.
    void set_radial_profiles( const RadialProfiles< ScalarType >& p ) override
    {
        profiles_ = p;
        if ( !p.valid )
            return;
        for ( auto* op : { A_.get(), A_neumann_.get(), A_neumann_diag_.get() } )
            if ( op != nullptr )
                op->set_kappa_profile( p.kappa );
    }
    EVSolver(
        const std::shared_ptr< grid::shell::DistributedDomain >&        domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
        const linalg::VectorQ1Vec< ScalarType, 3 >&                     velocity,
        linalg::VectorQ1Scalar< ScalarType >&                           T,
        ScalarType                                                      h,
        const Parameters&                                               prm,
        std::shared_ptr< util::Table >                                  table,
        const grid::Grid4DDataScalar< ScalarType >&                     viscosity = {},
        const grid::Grid4DDataScalar< ScalarType >&                     rho       = {} )
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
    , viscosity_( viscosity )
    , rho_( rho )
    , tmp_(         "ev_tmp",        *domain_, ownership_mask_ )
    , q_(           "ev_q",          *domain_, ownership_mask_ )
    , diag_(        "ev_diag",       *domain_, ownership_mask_ )
    , T_prev_(      "T_prev",        *domain_, ownership_mask_ )
    , lap_T_(       "ev_lap_T",      *domain_, ownership_mask_ )
    , M_lumped_(    "ev_M_lumped",   *domain_, ownership_mask_ )
    {
        // lap_T_, rhs_ev_, g_ have strictly sequential, non-overlapping lifetimes
        // within step() (lap_T_ done at the lumped-mass divide, then rhs_ev_ for the
        // EV diffusion RHS, then g_ for the Dirichlet vector), so they share a single
        // allocation. Kokkos Views are ref-counted, so these shallow copies alias the
        // same storage; step() keeps using the three semantic names. Saves 2 Q1 fields.
        rhs_ev_ = lap_T_;
        g_      = lap_T_;

        // Picard backups are only touched when iterating; leave them empty
        // (unallocated) otherwise to save two Q1-scalar fields.
        if ( prm_.time_stepping_parameters.picard_iterations > 1 )
        {
            T_backup_      = linalg::VectorQ1Scalar< ScalarType >( "ev_T_backup",      *domain_, ownership_mask_ );
            T_prev_backup_ = linalg::VectorQ1Scalar< ScalarType >( "ev_T_prev_backup", *domain_, ownership_mask_ );
        }

        util::logroot << "Setting up entropy-viscosity (EV) energy solver ..." << std::endl;
        if ( prm_.devel_parameters.extended_diagnostics )
            log_hbm( "EV: after Q1 scalar fields (T_prev/rhs/lap/M_lumped/backups/g/tmp/q/diag)" );

        // Per-wedge ν_h field: extents (#subdomains, N-1, N-1, N_r-1, num_wedges).
        const auto num_sub = static_cast< long long >( domain_->subdomains().size() );
        const auto nx_c    = domain_->domain_info().subdomain_num_nodes_per_side_laterally() - 1;
        const auto nr_c    = domain_->domain_info().subdomain_num_nodes_radially() - 1;
        nu_h_wedge_        = grid::Grid5DDataScalar< ScalarType >(
            "nu_h_wedge", num_sub, nx_c, nx_c, nr_c, fe::wedge::num_wedges_per_hex_cell );

        A_ = std::make_unique< AD_EV >(
            *domain_,
            coords_shell_,
            coords_radii_,
            boundary_mask_,
            velocity_,
            prm_.physics_parameters.thermal_diffusivity_nondim,
            ScalarType( 0 ),
            /*treat_boundary=*/true );
        A_->set_supg_enabled( false );

        A_neumann_ = std::make_unique< AD_EV >(
            *domain_,
            coords_shell_,
            coords_radii_,
            boundary_mask_,
            velocity_,
            prm_.physics_parameters.thermal_diffusivity_nondim,
            ScalarType( 0 ),
            /*treat_boundary=*/false );
        A_neumann_->set_supg_enabled( false );

        A_neumann_diag_ = std::make_unique< AD_EV >(
            *domain_,
            coords_shell_,
            coords_radii_,
            boundary_mask_,
            velocity_,
            prm_.physics_parameters.thermal_diffusivity_nondim,
            ScalarType( 0 ),
            /*treat_boundary=*/false,
            /*diagonal=*/true );
        A_neumann_diag_->set_supg_enabled( false );

        M_ = std::make_unique< TempMass >( *domain_, coords_shell_, coords_radii_, false );

        // Global Galerkin Laplacian for κ∇²T projection.  κ is spatially uniform
        // (a single physics parameter), so we use the constant-coefficient
        // overload of the ∇·(ν ∇·) operator — no per-wedge Grid5D κ field is
        // stored — giving the standard ∫ κ ∇φ_i · ∇φ_j with additive halo exchange.
        if ( prm_.devel_parameters.extended_diagnostics )
            log_hbm( "EV: + nu_h_wedge (1 Grid5D per-wedge field; kappa is a scalar)" );
        A_kappa_ = std::make_unique< EVDiffOp >(
            *domain_, coords_shell_, coords_radii_,
            static_cast< ScalarType >( prm_.physics_parameters.thermal_diffusivity_nondim ) );

        // Global lumped mass M_lumped = M · 1, used to invert the global
        // Galerkin K·T into a Q1-nodal lap field per timestep:
        //   lap_T = (K · T) / M_lumped  ≈  −κ∇²T (Q1-nodal).
        {
            linalg::VectorQ1Scalar< ScalarType > ones( "ev_setup_ones", *domain_, ownership_mask_ );
            linalg::assign( ones, ScalarType( 1 ) );
            linalg::assign( M_lumped_, ScalarType( 0 ) );
            linalg::apply( *M_, ones, M_lumped_ );
        }

        // ν_h read by reference; the underlying view is updated in place
        // each step by compute_nu_h.
        A_evdiff_ = std::make_unique< EVDiffOp >( *domain_, coords_shell_, coords_radii_, nu_h_wedge_ );

        // OPT: fold nu_h implicitly into the fused kerngen advection-diffusion operator
        // (ASPECT-style) instead of a separate WedgeConstantDivKGrad matvec per Krylov
        // iteration. Wiring it into all three instances also puts nu_h into the
        // preconditioner diagonal and the Dirichlet lift. Enabled by --ev-implicit-nu-h.
        if ( prm_.energy_solver_parameters.implicit_nu_h )
        {
            A_->set_nu_h_field( nu_h_wedge_ );
            A_neumann_->set_nu_h_field( nu_h_wedge_ );
            A_neumann_diag_->set_nu_h_field( nu_h_wedge_ );
        }

        A_neumann_diag_->dt() = ScalarType( 1e-4 );
        linalg::assign( diag_, ScalarType( 0 ) );
        {
            linalg::VectorQ1Scalar< ScalarType > ones( "ev_setup_ones", *domain_, ownership_mask_ );
            linalg::assign( ones, ScalarType( 1 ) );
            linalg::apply( *A_neumann_diag_, ones, diag_ );
        }

        use_float_basis_ = prm_.energy_solver_parameters.float_krylov_basis;
        const linalg::solvers::FGMRESOptions< ScalarType > ev_fgmres_opts{
            .restart                     = prm_.energy_solver_parameters.krylov_restart,
            .relative_residual_tolerance = prm_.energy_solver_parameters.krylov_relative_tolerance,
            .absolute_residual_tolerance = prm_.energy_solver_parameters.krylov_absolute_tolerance,
            .max_iterations              = prm_.energy_solver_parameters.krylov_max_iterations };
        if ( use_float_basis_ )
        {
            // 4 double scratch + single-precision basis (2*restart+1).
            constexpr int kNumWork = 3; // FGMRESLowMem aliases r/w
            tmp_gmres_.reserve( kNumWork );
            for ( int i = 0; i < kNumWork; ++i )
                tmp_gmres_.emplace_back( "tmp_ev_gmres_work", *domain_, ownership_mask_ );
            const int num_basis = 2 * prm_.energy_solver_parameters.krylov_restart + 1;
            basis_gmres_.reserve( num_basis );
            for ( int i = 0; i < num_basis; ++i )
                basis_gmres_.emplace_back( "tmp_ev_gmres_basis", *domain_, ownership_mask_ );
            solver_float_ = std::make_unique< FGMRESFloat >(
                tmp_gmres_, basis_gmres_, ev_fgmres_opts, table_, DiagSolverT( diag_ ) );
        }
        else
        {
            constexpr int num_gmres_tmps = 14;
            tmp_gmres_.reserve( num_gmres_tmps );
            for ( int i = 0; i < num_gmres_tmps; ++i )
                tmp_gmres_.emplace_back( "tmp_ev_gmres", *domain_, ownership_mask_ );
            solver_double_ = std::make_unique< FGMRESDouble >(
                tmp_gmres_, ev_fgmres_opts, table_, DiagSolverT( diag_ ) );
        }

        // Bootstrap T_prev = T so ∂_t E = 0 on step 1.
        Kokkos::deep_copy( T_prev_.grid_data(), T_.grid_data() );

        // Compressible (TALA) heating setup: a total nodal source field, a
        // scratch field, and the kerngen shear-heating operator reading the
        // (in-place-updated) viscosity field. Only when compressible.
        if ( prm_.physics_parameters.compressible )
        {
            heating_source_  = linalg::VectorQ1Scalar< ScalarType >( "ev_heating_source", *domain_, ownership_mask_ );
            heating_scratch_ = linalg::VectorQ1Scalar< ScalarType >( "ev_heating_scratch", *domain_, ownership_mask_ );
            heating_base_    = linalg::VectorQ1Scalar< ScalarType >( "ev_heating_base", *domain_, ownership_mask_ );
            diag_ones_       = linalg::VectorQ1Scalar< ScalarType >( "ev_diag_ones", *domain_, ownership_mask_ );
            linalg::assign( diag_ones_, ScalarType( 1 ) );
            shear_op_        = std::make_unique< fe::wedge::operators::shell::ShearHeatingKerngen< ScalarType > >(
                *domain_, coords_shell_, coords_radii_, viscosity_ );
            const ScalarType Di = static_cast< ScalarType >( prm_.physics_parameters.dissipation_number );
            const ScalarType Ra = static_cast< ScalarType >( prm_.physics_parameters.rayleigh_number );
            shear_op_->set_scale( Ra != ScalarType( 0 ) ? Di / Ra : ScalarType( 0 ) );
            if ( prm_.devel_parameters.extended_diagnostics )
                log_hbm( "EV: + compressible heating source fields (2 Q1)" );
        }

        // Apply runtime EV parameter overrides from the CLI.
        ev_params_.alpha_max = static_cast< ScalarType >( prm_.energy_solver_parameters.ev_alpha_max );
        ev_params_.alpha_E   = static_cast< ScalarType >( prm_.energy_solver_parameters.ev_alpha_E );

        util::logroot << "EV energy solver ready  (α_max=" << ev_params_.alpha_max << ", α_E=" << ev_params_.alpha_E
                      << ")" << std::endl;
    }

    linalg::VectorQ1Scalar< ScalarType >* nu_h_nodal_view() override { return nu_h_nodal_diag_.get(); }
    linalg::VectorQ1Scalar< ScalarType >* lap_diag_view() override { return lap_diag_.get(); }
    linalg::VectorQ1Scalar< ScalarType >* h_w_diag_view() override { return h_w_nodal_diag_.get(); }

    ScalarType compute_dt( const int timestep ) override
    {
        const auto max_vel      = kernels::common::max_vector_magnitude( velocity_.grid_data() );
        const auto dt_advection = h_ / max_vel;
        const auto dt_cfl       = prm_.time_stepping_parameters.dt_scaling * dt_advection;
        // The dt_min floor must never raise dt above the CFL-stable step: fast flows
        // (e.g. isoviscous Ra >= ~5e6) need dt << dt_min and blow up otherwise.
        const auto dt_min_eff = std::min( prm_.time_stepping_parameters.dt_min, dt_cfl );
        const auto dt         = std::clamp(
            ramp_dt( dt_cfl, timestep, prm_.time_stepping_parameters.initial_dt_ramp_steps ),
            dt_min_eff,
            prm_.time_stepping_parameters.dt_max );

        util::logroot << "Computing dt (EV advection CFL) ..." << std::endl;
        log_timestep_info( prm_, timestep, max_vel, h_, dt_cfl, dt );

        return dt;
    }

    void snapshot_for_picard() override
    {
        // Both T and T_prev mutate inside step() (the latter via the BDF1
        // history rotation), so we snapshot the (T, T_prev) pair — but only when
        // iterating; the backups are unallocated for a single Picard sweep.
        if ( prm_.time_stepping_parameters.picard_iterations > 1 )
        {
            Kokkos::deep_copy( T_backup_.grid_data(),      T_.grid_data() );
            Kokkos::deep_copy( T_prev_backup_.grid_data(), T_prev_.grid_data() );
        }
        // Mark ν_h stale at the start of a new timestep (always, independent of Picard);
        // the first Picard
        // iteration's substep-0 will compute it.  Subsequent Picard
        // iterations of the same timestep reuse it so the explicit-lagged
        // stabilization stays consistent across the (T, u) Picard fixed
        // point.  Substeps > 0 always recompute (T evolves between them).
        nu_h_locked_for_step_ = false;
    }

    void restore_for_picard() override
    {
        Kokkos::deep_copy( T_.grid_data(), T_backup_.grid_data() );
        Kokkos::deep_copy( T_prev_.grid_data(), T_prev_backup_.grid_data() );
    }

    void dump_diagnostics( int timestep, const std::string& outdir ) override
    {
        // TALA energy-consistency check (dissipation theorem). Runs here — after
        // the timestep's final Stokes solve — so velocity and temperature are a
        // mutually consistent (u solves Stokes for this T) pair; the balance is a
        // Stokes identity and only holds for such a pair. Independent of the ν_h
        // dump below.
        if ( prm_.physics_parameters.compressible )
        {
            log_dissipation_balance();
        }

        if ( !prm_.energy_solver_parameters.ev_dump_nu_h )
            return;

        // Reduce min/max/mean of nu_h_wedge_ over locally-owned cells.  Cells
        // are not shared between MPI ranks, so a global all-reduce on the
        // local sums/extrema is exact (no double-counting).
        ScalarType local_min = std::numeric_limits< ScalarType >::max();
        ScalarType local_max = std::numeric_limits< ScalarType >::lowest();
        ScalarType local_sum = 0;
        long long  local_n   = 0;

        const auto nu = nu_h_wedge_;
        Kokkos::parallel_reduce(
            "ev_nu_h_stats",
            Kokkos::MDRangePolicy< Kokkos::Rank< 5, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 0, 0, 0, 0 },
                { nu.extent( 0 ), nu.extent( 1 ), nu.extent( 2 ), nu.extent( 3 ), nu.extent( 4 ) } ),
            KOKKOS_LAMBDA( int s, int x, int y, int r, int w,
                           ScalarType& mn, ScalarType& mx, ScalarType& sm, long long& cnt ) {
                const ScalarType v = nu( s, x, y, r, w );
                if ( v < mn )
                    mn = v;
                if ( v > mx )
                    mx = v;
                sm += v;
                cnt += 1;
            },
            Kokkos::Min< ScalarType >( local_min ),
            Kokkos::Max< ScalarType >( local_max ),
            local_sum,
            local_n );
        Kokkos::fence();

        ScalarType g_min = local_min, g_max = local_max, g_sum = local_sum;
        long long  g_n = local_n;
        MPI_Allreduce( MPI_IN_PLACE, &g_min, 1, mpi::mpi_datatype< ScalarType >(), MPI_MIN, MPI_COMM_WORLD );
        MPI_Allreduce( MPI_IN_PLACE, &g_max, 1, mpi::mpi_datatype< ScalarType >(), MPI_MAX, MPI_COMM_WORLD );
        MPI_Allreduce( MPI_IN_PLACE, &g_sum, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );
        MPI_Allreduce( MPI_IN_PLACE, &g_n, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );

        const ScalarType g_mean = ( g_n > 0 ) ? ( g_sum / static_cast< ScalarType >( g_n ) ) : ScalarType( 0 );

        int rank = 0;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        if ( rank == 0 )
        {
            const std::string path = outdir + "/nu_h_stats.csv";
            std::ofstream     out( path, std::ios::app );
            if ( out.tellp() == 0 )
            {
                out << "timestep,nu_h_min,nu_h_max,nu_h_mean,n_wedges\n";
            }
            out << timestep << "," << g_min << "," << g_max << "," << g_mean << "," << g_n << "\n";
        }

        // Populate the Q1-nodal diagnostic field for XDMF visualisation.
        // Per cell: cell_avg = (ν_h_w0 + ν_h_w1) / 2.  Splat that scalar to
        // all 8 corners with atomic_add into nu_h_nodal_diag_ + count, then
        // additive halo exchange + pointwise divide.
        if ( nu_h_nodal_diag_ )
        {
            linalg::assign( *nu_h_nodal_diag_, ScalarType( 0 ) );
            linalg::assign( *nu_h_count_diag_, ScalarType( 0 ) );

            auto       diag_v  = nu_h_nodal_diag_->grid_data();
            auto       count_v = nu_h_count_diag_->grid_data();
            const auto nu      = nu_h_wedge_;

            Kokkos::parallel_for(
                "ev_nu_h_diag_scatter",
                grid::shell::local_domain_md_range_policy_cells( *domain_ ),
                KOKKOS_LAMBDA( int s, int xc, int yc, int rc ) {
                    constexpr int    hex_off_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
                    constexpr int    hex_off_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
                    constexpr int    hex_off_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
                    const ScalarType cell_avg = ScalarType( 0.5 ) * ( nu( s, xc, yc, rc, 0 ) + nu( s, xc, yc, rc, 1 ) );
                    for ( int k = 0; k < 8; ++k )
                    {
                        Kokkos::atomic_add(
                            &diag_v( s, xc + hex_off_x[k], yc + hex_off_y[k], rc + hex_off_r[k] ), cell_avg );
                        Kokkos::atomic_add(
                            &count_v( s, xc + hex_off_x[k], yc + hex_off_y[k], rc + hex_off_r[k] ), ScalarType( 1 ) );
                    }
                } );
            Kokkos::fence();

            // Additive halo exchange so seam nodes get contributions from all
            // ranks that touch them.
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                *domain_, diag_v, *diag_send_, *diag_recv_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( *domain_, diag_v, *diag_recv_ );
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                *domain_, count_v, *diag_send_, *diag_recv_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( *domain_, count_v, *diag_recv_ );

            // Pointwise divide.
            Kokkos::parallel_for(
                "ev_nu_h_diag_divide",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                    { 0, 0, 0, 0 },
                    { diag_v.extent( 0 ), diag_v.extent( 1 ), diag_v.extent( 2 ), diag_v.extent( 3 ) } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                    const ScalarType c = count_v( s, x, y, r );
                    if ( c > ScalarType( 0 ) )
                        diag_v( s, x, y, r ) /= c;
                } );
            Kokkos::fence();
        }

        // -- Hypothesis-3 diagnostic: lap_T_ → lap_diag_, plus per-shell |lap|
        //    max strata (CMB row vs first interior row, surface row vs first
        //    interior).  If the boundary-flux integral is dropped from the
        //    lumped-mass projection, |lap| at r=0 should be << |lap| at r=1.
        if ( lap_diag_ )
        {
            // Refresh the diagnostic copy with the most recent lap_T_.
            Kokkos::deep_copy( lap_diag_->grid_data(), lap_T_.grid_data() );

            const auto lap_v   = lap_diag_->grid_data();
            const auto own_v   = ownership_mask_;
            const int  N_r     = static_cast< int >( lap_v.extent( 3 ) );
            const int  surf_r  = N_r - 1;
            const int  surf_r1 = N_r - 2;

            auto reduce_max_at_r = [&]( int r_target ) -> ScalarType {
                ScalarType local_max = 0;
                Kokkos::parallel_reduce(
                    "ev_lap_max_at_r",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 3, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                        { 0, 0, 0 }, { lap_v.extent( 0 ), lap_v.extent( 1 ), lap_v.extent( 2 ) } ),
                    KOKKOS_LAMBDA( int s, int x, int y, ScalarType& m ) {
                        if ( util::has_flag( own_v( s, x, y, r_target ), grid::NodeOwnershipFlag::OWNED ) )
                        {
                            const ScalarType a = Kokkos::abs( lap_v( s, x, y, r_target ) );
                            if ( a > m )
                                m = a;
                        }
                    },
                    Kokkos::Max< ScalarType >( local_max ) );
                Kokkos::fence();
                MPI_Allreduce(
                    MPI_IN_PLACE, &local_max, 1, mpi::mpi_datatype< ScalarType >(), MPI_MAX, MPI_COMM_WORLD );
                return local_max;
            };

            const ScalarType lap_max_cmb     = reduce_max_at_r( 0 );
            const ScalarType lap_max_cmb_p1  = reduce_max_at_r( 1 );
            const ScalarType lap_max_surf    = reduce_max_at_r( surf_r );
            const ScalarType lap_max_surf_m1 = reduce_max_at_r( surf_r1 );

            if ( rank == 0 )
            {
                const std::string path = outdir + "/lap_stats.csv";
                std::ofstream     out( path, std::ios::app );
                if ( out.tellp() == 0 )
                {
                    out << "timestep,lap_max_cmb,lap_max_cmb_plus1,ratio_cmb,"
                           "lap_max_surf,lap_max_surf_minus1,ratio_surf\n";
                }
                const ScalarType r_cmb = ( lap_max_cmb_p1 > 0 ) ? ( lap_max_cmb / lap_max_cmb_p1 ) : ScalarType( 0 );
                const ScalarType r_surf =
                    ( lap_max_surf_m1 > 0 ) ? ( lap_max_surf / lap_max_surf_m1 ) : ScalarType( 0 );
                out << timestep << "," << lap_max_cmb << "," << lap_max_cmb_p1 << "," << r_cmb << "," << lap_max_surf
                    << "," << lap_max_surf_m1 << "," << r_surf << "\n";
            }
        }

        // -- Hypothesis-2 diagnostic: one-shot geom_stats.csv with global h_w
        //    and dr extrema/means and the headline ratio h_w_mean/dr_mean.
        //    Confirmation criterion: ratio ≳ 1.3 → h_w inflates BL-normal
        //    diffusion (ν_E ∝ h_w² overdamps).  < 1.1 → hypothesis dead.
        if ( h_w_nodal_diag_ && !geom_stats_written_ )
        {
            // Per-wedge h_w stats over locally-owned cells.  Cells aren't
            // shared across ranks → MPI sums on local sums/extrema are exact.
            ScalarType hw_min = std::numeric_limits< ScalarType >::max();
            ScalarType hw_max = std::numeric_limits< ScalarType >::lowest();
            ScalarType hw_sum = 0;
            long long  hw_n   = 0;
            const auto h_w_v  = h_w_wedge_;
            Kokkos::parallel_reduce(
                "ev_h_w_stats",
                Kokkos::MDRangePolicy< Kokkos::Rank< 5, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                    { 0, 0, 0, 0, 0 },
                    { h_w_v.extent( 0 ), h_w_v.extent( 1 ), h_w_v.extent( 2 ), h_w_v.extent( 3 ), h_w_v.extent( 4 ) } ),
                KOKKOS_LAMBDA(
                    int         s,
                    int         x,
                    int         y,
                    int         r,
                    int         w,
                    ScalarType& mn,
                    ScalarType& mx,
                    ScalarType& sm,
                    long long&  cnt ) {
                    const ScalarType v = h_w_v( s, x, y, r, w );
                    if ( v < mn )
                        mn = v;
                    if ( v > mx )
                        mx = v;
                    sm += v;
                    cnt += 1;
                },
                Kokkos::Min< ScalarType >( hw_min ),
                Kokkos::Max< ScalarType >( hw_max ),
                hw_sum,
                hw_n );
            Kokkos::fence();
            MPI_Allreduce( MPI_IN_PLACE, &hw_min, 1, mpi::mpi_datatype< ScalarType >(), MPI_MIN, MPI_COMM_WORLD );
            MPI_Allreduce( MPI_IN_PLACE, &hw_max, 1, mpi::mpi_datatype< ScalarType >(), MPI_MAX, MPI_COMM_WORLD );
            MPI_Allreduce( MPI_IN_PLACE, &hw_sum, 1, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );
            MPI_Allreduce( MPI_IN_PLACE, &hw_n, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD );
            const ScalarType hw_mean = ( hw_n > 0 ) ? ( hw_sum / static_cast< ScalarType >( hw_n ) ) : ScalarType( 0 );

            // Per-radial-cell dr stats.  radii_v(s, i) gives shell-boundary
            // radii; dr(s, i) = radii(s, i+1) - radii(s, i).  Same min/max/mean
            // semantics across owned subdomain cells.
            ScalarType dr_min    = std::numeric_limits< ScalarType >::max();
            ScalarType dr_max    = std::numeric_limits< ScalarType >::lowest();
            ScalarType dr_sum    = 0;
            long long  dr_n      = 0;
            const auto radii_v   = coords_radii_;
            const int  n_r_cells = static_cast< int >( radii_v.extent( 1 ) ) - 1;
            const int  n_sub     = static_cast< int >( radii_v.extent( 0 ) );
            Kokkos::parallel_reduce(
                "ev_dr_stats",
                Kokkos::MDRangePolicy< Kokkos::Rank< 2, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >( { 0, 0 }, { n_sub, n_r_cells } ),
                KOKKOS_LAMBDA( int s, int i, ScalarType& mn, ScalarType& mx, ScalarType& sm, long long& cnt ) {
                    const ScalarType v = radii_v( s, i + 1 ) - radii_v( s, i );
                    if ( v < mn )
                        mn = v;
                    if ( v > mx )
                        mx = v;
                    sm += v;
                    cnt += 1;
                },
                Kokkos::Min< ScalarType >( dr_min ),
                Kokkos::Max< ScalarType >( dr_max ),
                dr_sum,
                dr_n );
            Kokkos::fence();
            // dr is replicated across subdomains (same radii grid), so we don't
            // MPI-sum n / sum across ranks — every rank already has the full
            // 1D radial grid for its own subdomains.  Just rank-0 writes.

            const ScalarType dr_mean = ( dr_n > 0 ) ? ( dr_sum / static_cast< ScalarType >( dr_n ) ) : ScalarType( 0 );

            if ( rank == 0 )
            {
                const std::string path = outdir + "/geom_stats.csv";
                std::ofstream     out( path, std::ios::trunc );
                out << "h_w_min,h_w_max,h_w_mean,dr_min,dr_max,dr_mean,h_w_mean_over_dr_mean,n_wedges,n_dr\n";
                const ScalarType ratio = ( dr_mean > 0 ) ? ( hw_mean / dr_mean ) : ScalarType( 0 );
                out << hw_min << "," << hw_max << "," << hw_mean << "," << dr_min << "," << dr_max << "," << dr_mean
                    << "," << ratio << "," << hw_n << "," << dr_n << "\n";
                util::logroot << "[EV diag] h_w_mean=" << hw_mean << "  dr_mean=" << dr_mean << "  ratio=" << ratio
                              << "  (>=1.3 confirms hypothesis 2)" << std::endl;
            }
            geom_stats_written_ = true;
        }
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

        // Internal heating enters nondimensionally as H / c_p (upstream 2913b8c),
        // not as a raw rate.
        const ScalarType gamma =
            prm_.physics_parameters.internal_heating ?
                static_cast< ScalarType >( prm_.physics_parameters.h_number / prm_.physics_parameters.cp_profile ) :
                ScalarType( 0 );

        // OPT: shear-heating Phi depends only on (velocity, viscosity), both constant
        // across the energy substeps (Stokes + update_viscosity run once per outer
        // step). Build the velocity-only base F_base = gamma + (Di/Ra)*Phi_shear ONCE
        // here instead of every substep; each substep then only adds the T-dependent
        // adiabatic term. Numerically identical to per-substep assembly.
        if ( prm_.physics_parameters.compressible )
        {
            const ScalarType Di_h = static_cast< ScalarType >( prm_.physics_parameters.dissipation_number );
            const ScalarType Ra_h = static_cast< ScalarType >( prm_.physics_parameters.rayleigh_number );
            linalg::assign( heating_base_, gamma );
            // Internal heating carries 1/cp(r); Di, Ra, Pe already hold the reference values.
            if ( profiles_.valid && gamma != ScalarType( 0 ) )
            {
                Kokkos::parallel_for(
                    "ev_internal_heating_cp_scale",
                    grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                    ScaleByRadialProfile{ heating_base_.grid_data(), profiles_.cp, profiles_.cp, false, true } );
                Kokkos::fence();
            }
            // --shear-heating / --adiabatic-heating gate the two terms independently so a run
            // can isolate one of them; both default to on, so --compressible alone is unchanged.
            if ( prm_.physics_parameters.shear_heating )
            {
            shear_op_->assemble_phi_nodal( velocity_, heating_scratch_ );
            // Shear heating carries 1/(rho(r)*cp(r)).
            if ( profiles_.valid )
            {
                Kokkos::parallel_for(
                    "ev_shear_rho_cp_scale",
                    grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                    ScaleByRadialProfile{ heating_scratch_.grid_data(), profiles_.rho, profiles_.cp, true, true } );
                Kokkos::fence();
            }
            const ScalarType visc_scale_h = ( Ra_h != ScalarType( 0 ) ) ? Di_h / Ra_h : ScalarType( 0 );
            linalg::lincomb( heating_base_, { ScalarType( 1 ), visc_scale_h }, { heating_base_, heating_scratch_ } );
            }
        }

        for ( int i = 0; i < prm_.time_stepping_parameters.energy_substeps; ++i )
        {
            util::logroot << "Solving energy (EV, substep " << i << ") ..." << std::endl;

            // 0) Compressible (TALA) total nodal heat source
            //      F = γ_internal + (Di/Ra)·Φ_shear + S_adiabatic
            //    rebuilt each substep (T evolves; u is lagged). Used both as the
            //    RHS source (M·F) and inside the entropy-viscosity residual so
            //    ν_E vanishes where the full compressible balance holds.
            if ( prm_.physics_parameters.compressible )
            {
                const ScalarType Di = static_cast< ScalarType >( prm_.physics_parameters.dissipation_number );
                // F_base (= gamma + (Di/Ra)*Phi_shear, velocity-only) was assembled
                // once before the substep loop; only the adiabatic term is per-substep.

                // + S_adiabatic = −Di·(u·n)·T   (rising material cools; α is in Di).
                //   This term is ρ̄-free in BOTH formulations: canonical King divides
                //   the energy equation by ρ̄c̄ₚ, so ρ̄ cancels on the adiabatic term
                //   (unlike buoyancy and shear, which keep ρ̄ / 1/ρ̄). Full T is used.
                if ( prm_.physics_parameters.adiabatic_heating )
                {
                Kokkos::parallel_for(
                    "ev_adiabatic_source",
                    local_domain_md_range_policy_nodes( *domain_ ),
                    AdiabaticHeatingSource{ coords_shell_,
                                            coords_radii_,
                                            velocity_.grid_data(),
                                            T_.grid_data(),
                                            heating_scratch_.grid_data(),
                                            Di,
                                            ScalarType( -1 ),
                                            profiles_.alpha,
                                            profiles_.cp,
                                            profiles_.valid,
                                            /*divide_by_cp=*/true } );
                Kokkos::fence();
                linalg::lincomb(
                    heating_source_, { ScalarType( 1 ), ScalarType( 1 ) }, { heating_base_, heating_scratch_ } );
                }
                else
                {
                    linalg::assign( heating_source_, heating_base_ );
                }
            }

            // 1+2) per-wedge lap projection and ν_h.  Computed ONCE per outer step
            // (substep 0 / first Picard sweep) and frozen across the remaining
            // substeps: T evolves only mildly over the substeps, so reusing the
            // substep-0 stabilization field (ASPECT-style ν_h lagging) is a valid
            // speed/accuracy trade and removes the global Laplacian matvec +
            // entropy-stats + compute_nu_h from 4 of every 5 substeps.
            const bool need_nu_h = !nu_h_locked_for_step_;
            if ( need_nu_h )
            {
                // 1) Global Q1-nodal lap projection: lap_T = (K · T) / M_lumped
                //    with K the global Galerkin Laplacian (κ-weighted) and
                //    M_lumped the global lumped mass.  Continuity across
                //    element interfaces lets r_E collapse to ≈0 in smooth
                //    regions.
                linalg::apply( *A_kappa_, T_, lap_T_ );
                {
                    auto       lap_v = lap_T_.grid_data();
                    const auto m_v   = M_lumped_.grid_data();
                    const auto bm    = boundary_mask_;
                    Kokkos::parallel_for(
                        "ev_lap_T_lumped_mass_divide",
                        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                            { 0, 0, 0, 0 },
                            { lap_v.extent( 0 ), lap_v.extent( 1 ), lap_v.extent( 2 ), lap_v.extent( 3 ) } ),
                        KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                            // Zero lap at Dirichlet-boundary nodes: the Galerkin
                            // K·T at those nodes contains the wall-flux IBP term
                            // ∫_∂Ω(κ ∂T/∂n)φ_i which dominates the projection
                            // (~50× the next interior shell).  Setting to 0 here
                            // propagates a clean value into wedge-interior
                            // quadpoints via shape-function interpolation.
                            if ( util::has_flag( bm( s, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY ) )
                            {
                                lap_v( s, x, y, r ) = ScalarType( 0 );
                                return;
                            }
                            const ScalarType m = m_v( s, x, y, r );
                            lap_v( s, x, y, r ) =
                                ( m > ScalarType( 0 ) ) ? ( lap_v( s, x, y, r ) / m ) : ScalarType( 0 );
                        } );
                    Kokkos::fence();
                }

                // 2) Entropy stats (volume-weighted E_avg) and per-wedge ν_h.
                const auto stats = fe::wedge::operators::shell::compute_entropy_stats(
                    T_, ownership_mask_, *domain_, coords_shell_, coords_radii_, ev_params_ );
                fe::wedge::operators::shell::compute_nu_h(
                    nu_h_wedge_,
                    T_,
                    T_prev_,
                    velocity_,
                    lap_T_.grid_data(),
                    *domain_,
                    coords_shell_,
                    coords_radii_,
                    dt,
                    stats,
                    ev_params_,
                    gamma,
                    // Compressible: fold the full heat source into the residual
                    // Empty otherwise.
                    prm_.physics_parameters.compressible ? heating_source_.grid_data()
                                                         : grid::Grid4DDataScalar< ScalarType >{} );
            }
            if ( i == 0 )
            {
                nu_h_locked_for_step_ = true;
            }

            // 3+4) RHS:  q = M*T^n.  The entropy-viscosity term is either
            //   explicit (default): q -= dt*(A_evdiff*T^n)  [nu_h lagged on RHS], or
            //   implicit (--ev-implicit-nu-h): nu_h is folded into the LHS composite
            //   operator below, so nothing is added to the RHS here.
            linalg::apply( *M_, T_, q_ );
            if ( !prm_.energy_solver_parameters.implicit_nu_h )
            {
                linalg::apply( *A_evdiff_, T_, rhs_ev_ );
                linalg::lincomb( q_, { ScalarType( 1 ), -dt }, { q_, rhs_ev_ } );
            }

            // 4b) Heat-source RHS: q += dt · M · F.
            //     Compressible: F = internal + shear + adiabatic (heating_source_).
            //     Incompressible: F = γ (constant internal heating) as before.
            //     rhs_ev_ is finished with at this point and is reused as a
            //     scratch source-vector; tmp_ is also free until the Dirichlet
            //     enforcement below.
            if ( prm_.physics_parameters.compressible )
            {
                linalg::apply( *M_, heating_source_, tmp_ );
                linalg::lincomb( q_, { ScalarType( 1 ), dt }, { q_, tmp_ } );
            }
            else if ( gamma != ScalarType( 0 ) )
            {
                linalg::assign( rhs_ev_, gamma );
                linalg::apply( *M_, rhs_ev_, tmp_ );
                linalg::lincomb( q_, { ScalarType( 1 ), dt }, { q_, tmp_ } );
            }

            // 5) History rotation BEFORE the solve overwrites T.
            Kokkos::deep_copy( T_prev_.grid_data(), T_.grid_data() );

            // 6) Dirichlet BC vector + elimination from RHS.
            linalg::assign( g_, ScalarType( 0 ) );
            {
                auto       g_grid    = g_.grid_data();
                auto       mask      = boundary_mask_;
                const auto T_cmb_val = static_cast< ScalarType >( prm_.boundary_parameters.temperature_max );
                const auto T_top_val = static_cast< ScalarType >( prm_.boundary_parameters.temperature_min );
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
                *A_neumann_, *A_neumann_diag_, g_, tmp_, q_, boundary_mask_, grid::shell::ShellBoundaryFlag::BOUNDARY );

            // 7) Solve.  Explicit: (M + dt*A_galerkin)*T^{n+1} = q.
            //           Implicit nu_h: (M + dt*A_galerkin + dt*A_evdiff)*T^{n+1} = q.
            // nu_h is folded implicitly into A_ (set_nu_h_field in the ctor) when
            // --ev-implicit-nu-h; otherwise A_ is plain advection-diffusion and nu_h is
            // added on the RHS (explicit). Either way the solve operator is *A_.
            if ( use_float_basis_ )
                solve( *solver_float_, *A_, T_, q_ );
            else
                solve( *solver_double_, *A_, T_, q_ );

            if ( print_convergence )
            {
                util::logroot << "[EV energy FGMRES convergence]" << std::endl;
                table_->query_rows_equals( "tag", "fgmres_solver" ).print_pretty();
            }
            table_->clear();
        }
    }

    /// @brief Log the global dissipation balance for the compressible energy
    /// equation (dissipation theorem, Leng & Zhong 2008 / King et al. 2010):
    ///
    ///   Φ = (Di/Ra) · ∫ 2η ε̇_dev:ε̇_dev dV      (total viscous dissipation)
    ///   W = Di · ∫ ρ̄ · (u·n) · T dV              (total adiabatic work)
    ///
    /// For an energetically consistent formulation ⟨Φ⟩ = ⟨W⟩, so Φ/W → 1 (exact
    /// in ALA; a systematic few-percent offset in TALA). This is the primary,
    /// reference-free correctness gate for the shear + adiabatic heating terms:
    /// a ratio far from 1 flags a wrong sign, Di/Ra scaling, or viscosity
    /// normalisation. (α is folded into Di, matching the buoyancy Ra·δT·n.)
    void log_dissipation_balance()
    {
        const ScalarType Di = static_cast< ScalarType >( prm_.physics_parameters.dissipation_number );
        const ScalarType Ra = static_cast< ScalarType >( prm_.physics_parameters.rayleigh_number );

        // Φ: assemble the shear linear form ∫Φ_shear N_i (scale 1); the sum over
        // all test functions is ∫Φ_shear dV (partition of unity), obtained as
        // dot(1, ·). Scale by Di/Ra.
        shear_op_->set_scale( ScalarType( 1 ) );
        linalg::apply( *shear_op_, velocity_, heating_scratch_ );
        const ScalarType phi_int = linalg::dot( diag_ones_, heating_scratch_ );
        const ScalarType Phi     = ( Ra != ScalarType( 0 ) ) ? ( Di / Ra ) * phi_int : ScalarType( 0 );

        // W: nodal integrand Di·ρ̄·(u·n)·T, integrated via the lumped mass
        // (∫w dV = dot(w, M_lumped)). AdiabaticHeatingSource builds the ρ̄-FREE
        // Di·(u·n)·T — the form used in the SOLVED energy equation (divided by
        // ρ̄c̄ₚ, so ρ̄ cancels). The dissipation theorem compares the UN-divided
        // volumetric rates, so re-apply the reference density ρ̄ here (diagnostic
        // only) to put W in the same convention as Φ = (Di/Ra)·∫2η ε̇:ε̇ (which is
        // likewise the un-divided form). Without this weighting Φ/W is biased by
        // the mean ρ̄ (~0.86 in a Di≈0.45 shell) even when the physics is correct.
        Kokkos::parallel_for(
            "ev_dissip_W",
            local_domain_md_range_policy_nodes( *domain_ ),
            AdiabaticHeatingSource{ coords_shell_,
                                    coords_radii_,
                                    velocity_.grid_data(),
                                    T_.grid_data(),
                                    heating_source_.grid_data(),
                                    Di,
                                    ScalarType( 1 ),
                                    profiles_.alpha,
                                    profiles_.cp,
                                    profiles_.valid,
                                    /*divide_by_cp=*/false } );
        Kokkos::fence();
        {
            auto       w_v   = heating_source_.grid_data();
            const auto rho_v = rho_;
            Kokkos::parallel_for(
                "ev_dissip_W_rho_weight",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                    { 0, 0, 0, 0 },
                    { w_v.extent( 0 ), w_v.extent( 1 ), w_v.extent( 2 ), w_v.extent( 3 ) } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) { w_v( s, x, y, r ) *= rho_v( s, x, y, r ); } );
            Kokkos::fence();
        }
        const ScalarType W = linalg::dot( heating_source_, M_lumped_ );

        const ScalarType ratio = ( W != ScalarType( 0 ) ) ? Phi / W : ScalarType( 0 );
        util::logroot << "[TALA dissipation] Phi=" << Phi << "  W=" << W << "  Phi/W=" << ratio
                      << "  (expect ->1 for ALA; few-% off for TALA)" << std::endl;
    }

  private:
    std::shared_ptr< grid::shell::DistributedDomain >               domain_;
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell_;
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii_;
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_;
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask_;
    const linalg::VectorQ1Vec< ScalarType, 3 >&                     velocity_;
    linalg::VectorQ1Scalar< ScalarType >&                           T_;
    ScalarType                                                      h_;
    const Parameters&                                               prm_;
    std::shared_ptr< util::Table >                                  table_;

    // Compressible (TALA) heating. viscosity_ and rho_ alias the Stokes solver's
    // fine-level fields; they stay valid as those are updated in place. Empty
    // when running incompressible (no shear/adiabatic heating).
    grid::Grid4DDataScalar< ScalarType > viscosity_;
    grid::Grid4DDataScalar< ScalarType > rho_;
    // Total nodal heat-source F = internal + (Di/Ra)·Φ_shear + adiabatic, and a
    // scratch for the projected nodal Φ / adiabatic term. Allocated only when
    // compressible.
    linalg::VectorQ1Scalar< ScalarType >                                             heating_source_;
    linalg::VectorQ1Scalar< ScalarType >                                             heating_scratch_;
    linalg::VectorQ1Scalar< ScalarType >                                             heating_base_;
    linalg::VectorQ1Scalar< ScalarType >                                             diag_ones_;
    std::unique_ptr< fe::wedge::operators::shell::ShearHeatingKerngen< ScalarType > > shear_op_;

    RadialProfiles< ScalarType > profiles_;
    std::unique_ptr< AD_EV >                                                A_, A_neumann_, A_neumann_diag_;
    std::unique_ptr< TempMass >                                             M_;
    std::unique_ptr< EVDiffOp >                                             A_evdiff_, A_kappa_;
    bool                                                                    use_float_basis_ = false;
    std::unique_ptr< FGMRESDouble >                                         solver_double_;
    std::unique_ptr< FGMRESFloat >                                          solver_float_;

    linalg::VectorQ1Scalar< ScalarType >                                    g_, tmp_, q_, diag_;
    linalg::VectorQ1Scalar< ScalarType >                                    T_prev_;
    linalg::VectorQ1Scalar< ScalarType >                                    rhs_ev_;
    linalg::VectorQ1Scalar< ScalarType >                                    lap_T_;
    linalg::VectorQ1Scalar< ScalarType >                                    M_lumped_;
    linalg::VectorQ1Scalar< ScalarType >                                    T_backup_;
    linalg::VectorQ1Scalar< ScalarType >                                    T_prev_backup_;
    grid::Grid5DDataScalar< ScalarType >                                    nu_h_wedge_;
    fe::wedge::operators::shell::EntropyViscosityParameters< ScalarType >   ev_params_{};
    std::vector< linalg::VectorQ1Scalar< ScalarType > >                     tmp_gmres_;
    std::vector< BasisVecT >                                                basis_gmres_;

    // Q1-nodal diagnostic field (only allocated when ev_dump_nu_h is on).
    std::unique_ptr< linalg::VectorQ1Scalar< ScalarType > >                                    nu_h_nodal_diag_;
    std::unique_ptr< linalg::VectorQ1Scalar< ScalarType > >                                    nu_h_count_diag_;
    std::unique_ptr< communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType > > diag_send_, diag_recv_;

    // Hypothesis-2/3 diagnostic fields (only allocated when ev_dump_nu_h is on).
    grid::Grid5DDataScalar< ScalarType >                    h_w_wedge_;
    std::unique_ptr< linalg::VectorQ1Scalar< ScalarType > > h_w_nodal_diag_;
    std::unique_ptr< linalg::VectorQ1Scalar< ScalarType > > h_w_count_diag_;
    std::unique_ptr< linalg::VectorQ1Scalar< ScalarType > > lap_diag_;
    bool                                                    geom_stats_written_ = false;

    // Locked-by-Picard flag: false at the start of each timestep (set by
    // snapshot_for_picard), set to true once substep-0 has computed ν_h so
    // subsequent Picard iterations of the same step skip the recompute.
    bool nu_h_locked_for_step_ = false;
};



/// Fill the Dirichlet vector g with T_cmb on CMB nodes and T_surface on surface nodes (zero elsewhere).
/// A free function because CUDA does not permit an extended `__host__ __device__` lambda inside a private or
/// protected member function.
template < typename ScalarType >
void fill_dirichlet_temperature(
    const grid::shell::DistributedDomain&                           domain,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const ScalarType                                                T_cmb,
    const ScalarType                                                T_surface,
    linalg::VectorQ1Scalar< ScalarType >&                           g )
{
    linalg::assign( g, ScalarType( 0 ) );
    auto g_grid = g.grid_data();
    auto mask   = boundary_mask;
    Kokkos::parallel_for(
        "fill_dirichlet_temperature",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
            const auto flag = mask( sd, x, y, r );
            if ( flag == grid::shell::ShellBoundaryFlag::CMB )
                g_grid( sd, x, y, r ) = T_cmb;
            else if ( flag == grid::shell::ShellBoundaryFlag::SURFACE )
                g_grid( sd, x, y, r ) = T_surface;
        } );
    Kokkos::fence();
}

/// Semi-Lagrangian (MMOC) energy solve with split implicit diffusion.
///
/// Ported from branch mmoc-transport-v1 (commits d82bc2ee..61ca361b) onto the compressible branch.
/// One step is  A(dt) -> D(dt), where A is the modified method of characteristics
/// (\ref terra::fe::wedge::operators::shell::MMOCTransport, RK4 foot-point tracing, quintic reconstruction)
/// and D is the implicit Galerkin diffusion solve (the SUPG operator at zero velocity, i.e. M + dt*K_diff).
/// Lie splitting on purpose: the diffusion solve is backward Euler, so Strang would double the Krylov cost
/// without raising the order. Diffusion runs last so the Dirichlet values are the ones imposed at the end of
/// the step.
///
/// Compressible (TALA) heating is added to the diffusion right-hand side exactly as in the EV solver:
/// q += dt * M * F with F = gamma_internal + (Di/Ra) * Phi_shear + S_adiabatic, built from the velocity and
/// temperature after the transport step. Incompressible runs keep the constant internal heating gamma only.
///
/// The characteristic tracing has no stability limit of its own; the timestep is bounded by the requirement
/// that a departure point stays inside the ghost layer, i.e. a Courant number below
/// \ref MMOCTransport::max_courant (0.9 * ghost width).
template < typename ScalarType >
class MMOCSolver : public EnergySolver< ScalarType >
{
    using Transport   = fe::wedge::operators::shell::MMOCTransport< ScalarType >;
    using AD          = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;
    using TempMass    = fe::wedge::operators::shell::Mass< ScalarType >;
    using DiagSolverT = linalg::solvers::DiagonalSolver< AD >;
    using FGMRESType  = linalg::solvers::FGMRES< AD, DiagSolverT >;

  public:
    /// Radial profiles: kappa goes into the three AD operators (they multiply the scalar
    /// diffusivity by it); the rest are read by the heating terms.
    void set_radial_profiles( const RadialProfiles< ScalarType >& p ) override
    {
        profiles_ = p;
        if ( !p.valid )
            return;
        for ( auto* op : { A_.get(), A_neumann_.get(), A_neumann_diag_.get() } )
            if ( op != nullptr )
                op->set_kappa_profile( p.kappa );
    }
    MMOCSolver(
        const std::shared_ptr< grid::shell::DistributedDomain >&        domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
        const linalg::VectorQ1Vec< ScalarType, 3 >&                     velocity,
        linalg::VectorQ1Scalar< ScalarType >&                           T,
        ScalarType                                                      h,
        const Parameters&                                               prm,
        std::shared_ptr< util::Table >                                  table,
        const grid::Grid4DDataScalar< ScalarType >&                     viscosity = {},
        const grid::Grid4DDataScalar< ScalarType >&                     rho       = {} )
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
    , viscosity_( viscosity )
    , rho_( rho )
    , transport_( *domain, ownership_mask, fe::wedge::operators::shell::TimeSteppingScheme::RK4 )
    , u_prev_( "mmoc_u_prev", *domain, ownership_mask )
    , u_zero_( "mmoc_u_zero", *domain, ownership_mask )
    , g_( "mmoc_g", *domain, ownership_mask )
    , tmp_( "mmoc_tmp", *domain, ownership_mask )
    , q_( "mmoc_q", *domain, ownership_mask )
    , diag_( "mmoc_diag", *domain, ownership_mask )
    {
        util::logroot << "Setting up MMOC energy solver ..." << std::endl;

        if ( prm_.time_stepping_parameters.picard_iterations > 1 )
            T_backup_ = linalg::VectorQ1Scalar< ScalarType >( "mmoc_T_backup", *domain_, ownership_mask_ );

        linalg::assign( u_zero_, ScalarType( 0 ) );
        // Start-up: with no previous velocity available, treat the flow as steady over the first step.
        copy_velocity( velocity_, u_prev_ );

        const auto kappa = prm_.physics_parameters.thermal_diffusivity_nondim;
        A_ = std::make_unique< AD >(
            *domain_, coords_shell_, coords_radii_, boundary_mask_, u_zero_, kappa, ScalarType( 0 ),
            /*treat_boundary=*/true );
        A_neumann_ = std::make_unique< AD >(
            *domain_, coords_shell_, coords_radii_, boundary_mask_, u_zero_, kappa, ScalarType( 0 ),
            /*treat_boundary=*/false );
        A_neumann_diag_ = std::make_unique< AD >(
            *domain_, coords_shell_, coords_radii_, boundary_mask_, u_zero_, kappa, ScalarType( 0 ),
            /*treat_boundary=*/false, /*diagonal=*/true );
        M_ = std::make_unique< TempMass >( *domain_, coords_shell_, coords_radii_, false );

        constexpr int num_gmres_tmps = 14;
        tmp_gmres_.reserve( num_gmres_tmps );
        for ( int i = 0; i < num_gmres_tmps; ++i )
            tmp_gmres_.emplace_back( "tmp_mmoc_gmres", *domain_, ownership_mask_ );
        solver_ = std::make_unique< FGMRESType >(
            tmp_gmres_,
            linalg::solvers::FGMRESOptions{
                .restart                     = prm_.energy_solver_parameters.krylov_restart,
                .relative_residual_tolerance = prm_.energy_solver_parameters.krylov_relative_tolerance,
                .absolute_residual_tolerance = prm_.energy_solver_parameters.krylov_absolute_tolerance,
                .max_iterations              = prm_.energy_solver_parameters.krylov_max_iterations },
            table_,
            DiagSolverT( diag_ ) );

        // Compressible (TALA) heating: same fields and shear operator as the EV solver.
        if ( prm_.physics_parameters.compressible )
        {
            heating_source_  = linalg::VectorQ1Scalar< ScalarType >( "mmoc_heating_source", *domain_, ownership_mask_ );
            heating_scratch_ = linalg::VectorQ1Scalar< ScalarType >( "mmoc_heating_scratch", *domain_, ownership_mask_ );
            shear_op_        = std::make_unique< fe::wedge::operators::shell::ShearHeatingKerngen< ScalarType > >(
                *domain_, coords_shell_, coords_radii_, viscosity_ );
            const ScalarType Di = static_cast< ScalarType >( prm_.physics_parameters.dissipation_number );
            const ScalarType Ra = static_cast< ScalarType >( prm_.physics_parameters.rayleigh_number );
            shear_op_->set_scale( Ra != ScalarType( 0 ) ? Di / Ra : ScalarType( 0 ) );
            if ( prm_.devel_parameters.extended_diagnostics )
                log_hbm( "MMOC: + compressible heating source fields (2 Q1)" );
        }

        util::logroot << "MMOC energy solver ready (max Courant " << Transport::max_courant()
                      << ", RK4 tracing, quintic reconstruction, point-location tolerance "
                      << transport_.locate_tolerance() << ")." << std::endl;
    }

    ScalarType compute_dt( const int timestep ) override
    {
        const auto max_vel = kernels::common::max_vector_magnitude( velocity_.grid_data() );
        // Diffusion is implicit and the characteristic tracing is unconditionally stable; the bound is that
        // the departure point must stay inside the ghost layer (Courant < max_courant).
        const auto dt_courant = Transport::max_courant() * h_ / max_vel;
        const auto dt_cfl     = std::min(
            static_cast< ScalarType >( prm_.time_stepping_parameters.dt_scaling ) * h_ / max_vel, dt_courant );
        // The dt_min floor must never raise dt above the Courant-stable step.
        const auto dt_min_eff = std::min( static_cast< ScalarType >( prm_.time_stepping_parameters.dt_min ), dt_cfl );
        const auto dt         = std::clamp(
            ramp_dt( dt_cfl, timestep, prm_.time_stepping_parameters.initial_dt_ramp_steps ),
            dt_min_eff,
            static_cast< ScalarType >( prm_.time_stepping_parameters.dt_max ) );

        util::logroot << "Computing dt (MMOC, ghost-layer Courant bound) ..." << std::endl;
        log_timestep_info( prm_, timestep, max_vel, h_, dt_cfl, dt );
        return dt;
    }

    void snapshot_for_picard() override
    {
        // Called once per timestep, before the Picard loop and before this timestep's Stokes solve, so the
        // velocity still holds u^n here -- the field the characteristic tracing interpolates in time against.
        copy_velocity( velocity_, u_prev_ );
        if ( prm_.time_stepping_parameters.picard_iterations > 1 )
            Kokkos::deep_copy( T_backup_.grid_data(), T_.grid_data() );
    }

    void restore_for_picard() override { Kokkos::deep_copy( T_.grid_data(), T_backup_.grid_data() ); }

    void dump_diagnostics( int /*timestep*/, const std::string& /*outdir*/ ) override
    {
        if ( prm_.physics_parameters.compressible )
            log_dissipation_balance();
    }

    void step( ScalarType dt, bool print_convergence ) override
    {
        util::Timer timer_energy( "energy" );
        const int substeps = Transport::substeps_for_accuracy(
            static_cast< ScalarType >( prm_.time_stepping_parameters.dt_scaling ) );
        transport_.step( T_, velocity_, u_prev_, dt, substeps );
        diffuse( dt, print_convergence );

        if ( transport_.last_escapes() > 0 )
        {
            // Two causes, distinguishable by whether the count grows with dt: a Courant number above
            // max_courant(), or the fixed set of degenerate corner regions (pentagonal points of the
            // icosahedral grid and subdomain corners), whose count is independent of dt.
            util::logroot << "    NOTE: " << transport_.last_escapes()
                          << " departure points could not be located and were interpolated at the nearest point of the last located cell. If this "
                             "count grows with dt, lower dt_scaling below "
                          << Transport::max_courant()
                          << "; if it is constant, it is the fixed set of degenerate corner regions." << std::endl;
        }
    }

  private:
    /// Component-wise copy: the vector grid data is stored as separate views per component (SoA).
    static void copy_velocity(
        const linalg::VectorQ1Vec< ScalarType, 3 >& src,
        linalg::VectorQ1Vec< ScalarType, 3 >&       dst )
    {
        for ( int d = 0; d < 3; ++d )
            Kokkos::deep_copy( dst.grid_data().comp_[d], src.grid_data().comp_[d] );
    }

    /// Build the nodal heat source F for the diffusion RHS (compressible only): F = gamma + (Di/Ra)*Phi_shear
    /// + S_adiabatic, evaluated with the current (lagged) velocity and the transported temperature.
    void assemble_heating_source( const ScalarType gamma )
    {
        const ScalarType Di = static_cast< ScalarType >( prm_.physics_parameters.dissipation_number );
        const ScalarType Ra = static_cast< ScalarType >( prm_.physics_parameters.rayleigh_number );
        linalg::assign( heating_source_, gamma );
        // Internal heating carries 1/cp(r).
        if ( profiles_.valid && gamma != ScalarType( 0 ) )
        {
            Kokkos::parallel_for(
                "mmoc_internal_heating_cp_scale",
                grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                ScaleByRadialProfile{ heating_source_.grid_data(), profiles_.cp, profiles_.cp, false, true } );
            Kokkos::fence();
        }
        if ( prm_.physics_parameters.shear_heating )
        {
        shear_op_->assemble_phi_nodal( velocity_, heating_scratch_ );
        // Shear heating carries 1/(rho(r)*cp(r)).
        if ( profiles_.valid )
        {
            Kokkos::parallel_for(
                "mmoc_shear_rho_cp_scale",
                grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                ScaleByRadialProfile{ heating_scratch_.grid_data(), profiles_.rho, profiles_.cp, true, true } );
            Kokkos::fence();
        }
        const ScalarType visc_scale = ( Ra != ScalarType( 0 ) ) ? Di / Ra : ScalarType( 0 );
        linalg::lincomb( heating_source_, { ScalarType( 1 ), visc_scale }, { heating_source_, heating_scratch_ } );
        }
        if ( prm_.physics_parameters.adiabatic_heating )
        {
        Kokkos::parallel_for(
            "mmoc_adiabatic_source",
            grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
            AdiabaticHeatingSource{ coords_shell_,
                                    coords_radii_,
                                    velocity_.grid_data(),
                                    T_.grid_data(),
                                    heating_scratch_.grid_data(),
                                    Di,
                                    ScalarType( -1 ),
                                    profiles_.alpha,
                                    profiles_.cp,
                                    profiles_.valid,
                                    /*divide_by_cp=*/true } );
        Kokkos::fence();
        linalg::lincomb( heating_source_, { ScalarType( 1 ), ScalarType( 1 ) }, { heating_source_, heating_scratch_ } );
        }
    }

    void diffuse( const ScalarType dt, const bool print_convergence )
    {
        A_->dt()              = dt;
        A_neumann_->dt()      = dt;
        A_neumann_diag_->dt() = dt;
        {
            linalg::VectorQ1Scalar< ScalarType > ones( "ones", *domain_, ownership_mask_ );
            linalg::assign( ones, ScalarType( 1 ) );
            linalg::apply( *A_neumann_diag_, ones, diag_ );
            linalg::invert_entries( diag_ );
        }

        // RHS: q = M * T (after transport), plus dt * M * F for the heat sources.
        linalg::apply( *M_, T_, q_ );
        // Internal heating enters nondimensionally as H / c_p (upstream 2913b8c),
        // not as a raw rate.
        const ScalarType gamma =
            prm_.physics_parameters.internal_heating ?
                static_cast< ScalarType >( prm_.physics_parameters.h_number / prm_.physics_parameters.cp_profile ) :
                ScalarType( 0 );
        if ( prm_.physics_parameters.compressible )
        {
            assemble_heating_source( gamma );
            linalg::apply( *M_, heating_source_, tmp_ );
            linalg::lincomb( q_, { ScalarType( 1 ), dt }, { q_, tmp_ } );
        }
        else if ( gamma != ScalarType( 0 ) )
        {
            linalg::assign( g_, gamma );
            linalg::apply( *M_, g_, tmp_ );
            linalg::lincomb( q_, { ScalarType( 1 ), dt }, { q_, tmp_ } );
        }

        fill_dirichlet_temperature(
            *domain_,
            boundary_mask_,
            static_cast< ScalarType >( prm_.boundary_parameters.temperature_max ),
            static_cast< ScalarType >( prm_.boundary_parameters.temperature_min ),
            g_ );
        fe::strong_algebraic_dirichlet_enforcement_poisson_like(
            *A_neumann_, *A_neumann_diag_, g_, tmp_, q_, boundary_mask_, grid::shell::ShellBoundaryFlag::BOUNDARY );

        solve( *solver_, *A_, T_, q_ );

        if ( print_convergence )
        {
            util::logroot << "[MMOC diffusion FGMRES convergence]" << std::endl;
            table_->query_rows_equals( "tag", "fgmres_solver" ).print_pretty();
        }
        table_->clear();
    }

    /// Dissipation balance Phi/W (see EVSolver::log_dissipation_balance for the derivation).
    void log_dissipation_balance()
    {
        const ScalarType Di = static_cast< ScalarType >( prm_.physics_parameters.dissipation_number );
        const ScalarType Ra = static_cast< ScalarType >( prm_.physics_parameters.rayleigh_number );
        linalg::VectorQ1Scalar< ScalarType > ones( "mmoc_diss_ones", *domain_, ownership_mask_ );
        linalg::assign( ones, ScalarType( 1 ) );
        shear_op_->set_scale( ScalarType( 1 ) );
        linalg::apply( *shear_op_, velocity_, heating_scratch_ );
        shear_op_->set_scale( Ra != ScalarType( 0 ) ? Di / Ra : ScalarType( 0 ) );
        const ScalarType Phi = ( Ra != ScalarType( 0 ) ) ? ( Di / Ra ) * linalg::dot( ones, heating_scratch_ ) : ScalarType( 0 );
        Kokkos::parallel_for(
            "mmoc_dissip_W",
            grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
            AdiabaticHeatingSource{ coords_shell_, coords_radii_, velocity_.grid_data(), T_.grid_data(),
                                    heating_source_.grid_data(), Di, ScalarType( 1 ),
                                    profiles_.alpha, profiles_.cp, profiles_.valid,
                                    /*divide_by_cp=*/false } );
        Kokkos::fence();
        {
            auto       w_v   = heating_source_.grid_data();
            const auto rho_v = rho_;
            Kokkos::parallel_for(
                "mmoc_dissip_W_rho_weight",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                    { 0, 0, 0, 0 }, { w_v.extent( 0 ), w_v.extent( 1 ), w_v.extent( 2 ), w_v.extent( 3 ) } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) { w_v( s, x, y, r ) *= rho_v( s, x, y, r ); } );
            Kokkos::fence();
        }
        linalg::apply( *M_, ones, tmp_ ); // lumped mass
        const ScalarType W     = linalg::dot( heating_source_, tmp_ );
        const ScalarType ratio = ( W != ScalarType( 0 ) ) ? Phi / W : ScalarType( 0 );
        util::logroot << "[TALA dissipation] Phi=" << Phi << "  W=" << W << "  Phi/W=" << ratio
                      << "  (expect ->1 for ALA; few-% off for TALA)" << std::endl;
    }

    std::shared_ptr< grid::shell::DistributedDomain >               domain_;
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell_;
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii_;
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_;
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask_;
    const linalg::VectorQ1Vec< ScalarType, 3 >&                     velocity_;
    linalg::VectorQ1Scalar< ScalarType >&                           T_;
    ScalarType                                                      h_;
    const Parameters&                                               prm_;
    std::shared_ptr< util::Table >                                  table_;

    // Compressible (TALA) heating inputs (alias the Stokes solver's fine-level fields; empty if incompressible).
    grid::Grid4DDataScalar< ScalarType > viscosity_;
    grid::Grid4DDataScalar< ScalarType > rho_;
    linalg::VectorQ1Scalar< ScalarType > heating_source_;
    linalg::VectorQ1Scalar< ScalarType > heating_scratch_;
    std::unique_ptr< fe::wedge::operators::shell::ShearHeatingKerngen< ScalarType > > shear_op_;

    Transport                            transport_;
    linalg::VectorQ1Vec< ScalarType, 3 > u_prev_;
    linalg::VectorQ1Vec< ScalarType, 3 > u_zero_;

    RadialProfiles< ScalarType > profiles_;
    std::unique_ptr< AD >                               A_, A_neumann_, A_neumann_diag_;
    std::unique_ptr< TempMass >                         M_;
    std::unique_ptr< FGMRESType >                       solver_;
    linalg::VectorQ1Scalar< ScalarType >                g_, tmp_, q_, diag_;
    linalg::VectorQ1Scalar< ScalarType >                T_backup_;
    std::vector< linalg::VectorQ1Scalar< ScalarType > > tmp_gmres_;
};


template < typename ScalarType >
class FCTSolver : public EnergySolver< ScalarType >
{
  public:
    FCTSolver(
        const std::shared_ptr< grid::shell::DistributedDomain >&        domain,
        const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
        const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
        const linalg::VectorQ1Vec< ScalarType, 3 >&                     velocity,
        linalg::VectorQ1Scalar< ScalarType >&                           T,
        linalg::VectorFVScalar< ScalarType >&                           T_fct,
        const linalg::VectorFVVec< ScalarType, 3 >&                     fv_cell_centers,
        const fv::hex::DirichletBCs< ScalarType >&                      fct_bcs,
        const Parameters&                                               prm,
        std::shared_ptr< util::Table >                                  table )
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
    , fv_fct_bufs_( *domain_ )
    {
        // FCT Picard backup: only touched when iterating; leave empty otherwise.
        if ( prm_.time_stepping_parameters.picard_iterations > 1 )
            T_fct_backup_ = linalg::VectorFVScalar< ScalarType >( "T_fct_backup", *domain_ );

        linalg::assign( T_source_, ScalarType( 0 ) );

        // l2_project_fv_to_fe needs at least 5 Q1 scalar temporaries.
        constexpr int num_l2_proj_tmps = 5;
        l2_proj_tmps_.reserve( num_l2_proj_tmps );
        for ( int i = 0; i < num_l2_proj_tmps; ++i )
        {
            l2_proj_tmps_.emplace_back( "fct_l2_proj_tmp_" + std::to_string( i ), *domain_, ownership_mask );
        }
    }

    void snapshot_for_picard() override
    {
        if ( prm_.time_stepping_parameters.picard_iterations > 1 )
            Kokkos::deep_copy( T_fct_backup_.grid_data(), T_fct_.grid_data() );
    }

    void restore_for_picard() override { Kokkos::deep_copy( T_fct_.grid_data(), T_fct_backup_.grid_data() ); }

    ScalarType compute_dt( const int timestep ) override
    {
        const auto dt_stable = fv::hex::operators::compute_dt_stable(
            *domain_,
            velocity_,
            fv_cell_centers_.grid_data(),
            coords_shell_,
            coords_radii_,
            prm_.physics_parameters.thermal_diffusivity_nondim );
        const auto dt =
            std::min( prm_.time_stepping_parameters.dt_scaling * dt_stable, prm_.time_stepping_parameters.dt_max );

        util::logroot << "Computing dt (FCT stable) ..." << std::endl;
        util::logroot << "    dt_stable:                     " << dt_stable * prm_.physics_parameters.calc_time_Ma
                      << " Ma" << std::endl;
        util::logroot << "=>  dt (= dt_stable * dt_scaling): " << dt * prm_.physics_parameters.calc_time_Ma << " Ma"
                      << std::endl;
        return dt;
    }

    void step( ScalarType dt, bool /*print_convergence*/ ) override
    {
        util::Timer timer_energy( "energy" );
        util::logroot << "Setting up energy solve ..." << std::endl;

        {
            util::Timer timer_fct_substeps( "fct_substeps" );

            for ( int i = 0; i < prm_.energy_solver_parameters.energy_substeps; ++i )
            {
                util::logroot << "Solving energy (FCT, substep " << i << ") ..." << std::endl;

                {
                    util::Timer timer_fct_source_step( "fct_explicit_step_updating_source_term" );
                    if ( prm_.physics_parameters.internal_heating )
                    {
                        linalg::assign(
                            T_source_, prm_.physics_parameters.h_number / prm_.physics_parameters.cp_profile );
                    }
                    timer_fct_source_step.stop();

                    util::Timer timer_fct_step( "fct_explicit_step" );
                    fv::hex::operators::fct_explicit_step(
                        *domain_,
                        T_fct_,
                        velocity_,
                        fv_cell_centers_.grid_data(),
                        coords_shell_,
                        coords_radii_,
                        dt,
                        fv_fct_bufs_,
                        prm_.physics_parameters.thermal_diffusivity_nondim,
                        T_source_.grid_data(),
                        /*subtract_divergence=*/true,
                        boundary_mask_,
                        fct_bcs_ );
                    timer_fct_step.stop();
                }

                fv::hex::apply_dirichlet_bcs( T_fct_, boundary_mask_, fct_bcs_, *domain_ );
            }

            timer_fct_substeps.stop();
        }

        // Project T_fct -> Q1 T once after all substeps.
        {
            util::Timer timer_fct_projection( "fct_l2_projection" );
            fv::hex::l2_project_fv_to_fe_lumped( T_, T_fct_, *domain_, coords_shell_, coords_radii_, l2_proj_tmps_ );

            // Enforce Dirichlet BCs on the Q1 temperature.
            auto       T_grid    = T_.grid_data();
            auto       mask      = boundary_mask_;
            const auto T_cmb_val = static_cast< ScalarType >( prm_.boundary_parameters.temperature_max );
            const auto T_top_val = static_cast< ScalarType >( prm_.boundary_parameters.temperature_min );
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
    std::shared_ptr< grid::shell::DistributedDomain >               domain_;
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell_;
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii_;
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_;
    const linalg::VectorQ1Vec< ScalarType, 3 >&                     velocity_;
    linalg::VectorQ1Scalar< ScalarType >&                           T_;
    linalg::VectorFVScalar< ScalarType >&                           T_fct_;
    const linalg::VectorFVVec< ScalarType, 3 >&                     fv_cell_centers_;
    const fv::hex::DirichletBCs< ScalarType >&                      fct_bcs_;
    const Parameters&                                               prm_;
    std::shared_ptr< util::Table >                                  table_;

    // Owned scratch.
    linalg::VectorFVScalar< ScalarType >                T_source_;
    linalg::VectorFVScalar< ScalarType >                T_fct_backup_;
    fv::hex::operators::FVFCTBuffers< ScalarType >      fv_fct_bufs_;
    std::vector< linalg::VectorQ1Scalar< ScalarType > > l2_proj_tmps_;
};

} // namespace terra::mantlecirculation
