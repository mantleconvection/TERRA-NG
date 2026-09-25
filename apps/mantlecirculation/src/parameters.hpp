#pragma once

#include <fstream>
#include <set>
#include <string>
#include <variant>

#include "util/cli11_config.hpp" // Custom formatter
#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/result.hpp"
namespace terra::mantlecirculation {

using ScalarType = double;

struct MeshParameters
{
    int refinement_level_mesh_min   = 1;
    int refinement_level_mesh_max   = 4;
    int refinement_level_subdomains = 0;

    // Nondimensional radii
    double radius_min = 0.5;
    double radius_max = 1.0;
    // Dimensional radii in meter
    double radius_surface_m   = 6371000.0;
    double radius_cmb_m       = 3480000.0;
    double mantle_thickness_m = 2891000.0;

    // Anisotropic mesh refinement
    // Radial diamond level is set to (refinement_level_mesh_* + radial_extra_levels)
    // at every MG level, so the full hierarchy stays consistent
    // (both axes halve per step).
    // lat_sdr/rad_sdr >= 0 overrides refinement_level_subdomains per axis.
    int radial_extra_levels = -1;
    int lat_sdr             = -1;
    int rad_sdr             = -1;

    /// Selector for the radial-shell distribution.  All non-uniform variants
    /// use a tanh map (see grid::shell::make_tanh_*_cluster) parameterised by
    /// `radial_cluster_k`; `radial_cluster_k <= 0` collapses each variant to
    /// the uniform distribution.
    enum class RadialDistribution
    {
        UNIFORM,      ///< equispaced shells (default).
        TANH_BOTH,    ///< both-side clustering at CMB and surface.
        TANH_CMB,     ///< one-side clustering at the inner boundary (CMB).
        TANH_SURFACE, ///< one-side clustering at the outer boundary (surface).
    };

    RadialDistribution radial_distribution = RadialDistribution::UNIFORM;
    double             radial_cluster_k    = 1.0;
};

struct PlateParameters
{
    bool apply_plate_velocities = false;

    /// Evaluate the plate velocities in a Kokkos kernel instead of querying the oracle per point on the host.
    bool plates_on_device           = false;
    bool interpolate_plates_in_time = true;
    int  initial_plate_age          = 400;
    int  final_plate_age            = 0;

    std::string plates_topologies_path = "../../../TERRA-NG/data/plates/Chen2025-tomopac/topologies_0-410Ma.geojson";
    std::string plates_reconstructions_path = "../../../TERRA-NG/data/plates/Chen2025-tomopac/TomoPAC2.rot";

    double plate_velocity_scaling = 1.0;
};
struct BoundaryConditionsParameters
{
    enum class VelocityBC
    {
        NO_SLIP,
        FREE_SLIP,
    };

    VelocityBC velocity_bc_cmb     = VelocityBC::NO_SLIP;
    VelocityBC velocity_bc_surface = VelocityBC::NO_SLIP;

    // Nondimensional temperatures
    double temperature_min = 0.0;
    double temperature_max = 1.0;
    // Dimensional temperatures in Kelvin
    double temperature_cmb_K     = 3800.0;
    double temperature_surface_K = 300.0;
    double delta_T_K             = temperature_cmb_K - temperature_surface_K;

    PlateParameters plate_parameters{};
};

/// Choice of viscosity law for the temperature-dependent viscosity field.
///   CONSTANT          : eta(T) = const (taken from `reference_viscosity`, optionally
///                       multiplied by a radial profile if `radial_profile_enabled`).
///   FRANK_KAMENETSKII : eta(T) = rmu^(0.5 - T)  (Zhong et al. 2008).  T in [0,1].
///                       Total cold/hot viscosity contrast = rmu (rmu = 1 → constant
///                       viscosity).  See `ViscosityParameters::rmu`.
enum class ViscosityLaw
{
    CONSTANT,
    FK_BENCHMARK,
    FK_TYPE1,
    FK_TYPE2,
    FK_TYPE3,
    ARRHENIUS
};

struct ViscosityParameters
{
    /// Viscosity law selector — see ViscosityLaw class above.
    ViscosityLaw law = ViscosityLaw::CONSTANT;

    /// Parameters for temperature- and depth-dependence of viscosity.
    /// Frank-Kamenetskii (FK) type laws -> eta = eta_ref * exp( -E_a ( T - x ) + V_a * depth ), with x either 0 (type 1), 0.5 (type 2) or T_mean(d) (type 3).
    /// For benchmarking, another FK-like law is available (Zhong et al. 2008): eta = rmu^( 0.5 - T ), with parameter activation_energy being used in place of rmu.
    /// Ignored when `law == CONSTANT`.
    /// Total viscosity contrast is O( e^activation_energy ) or O( activation_energy ) for FK_BENCHMARK.
    /// Frank-Kamenetskii cold/hot contrast. Kept for config compatibility;
    /// nondimensionalise() converts it to activation_energy = ln(rmu).
    double rmu = 1.0;
    double activation_energy = 4.605;
    double activation_volume = 1.5;

    /// Path to the CSV file containing the radial viscosity profile.
    /// If empty, radially constant viscosity is assumed.
    std::string viscosity_profile_csv_path = "";

    /// CSV column name for the viscosity values
    std::string viscosity_profile_radii_key = "Radius (m)";
    std::string viscosity_profile_value_key = "Viscosity (Pa s)";

    double reference_viscosity = 1e23;
    double max_viscosity       = 1e25;
    double min_viscosity       = 1e19;
};

/// Initial temperature distribution.
///   POWER_LAW  : T_init from a radial power-law profile (legacy default).
///   CONDUCTIVE : T_init = analytic conduction profile (r_min*r_max/r - r_min) / D.
///                Use this in combination with a spherical harmonic perturbation for the standard mantle
///                convection benchmarks (Zhong et al. 2008 A/C cases).
///   FROM_FILE  : Custom T_init read from csv file with path set in 'Tref_profile_csv_path'.
enum class InitialTemperatureProfile
{
    POWER_LAW,
    CONDUCTIVE,
    ADIABATIC,
    FROM_FILE
};

/// Initial temperature perturbation.
/// Choice between random noise and a spherical harmonic perturbation pattern, defined with parameters 'sph_degree_l' and 'sph_order_m'.
/// Amplitudes for both cases are set with the parameter 'perturbation_amplitude', in fraction of global temperature difference.
enum class InitialPerturbation
{
    NOISE,
    SPHERICAL_HARMONICS
};

struct InitialTemperatureParameters
{
    /// Selectors for the initial temperature distribution — see InitialTemperatureProfile and InitialPerturbation.
    InitialTemperatureProfile profile      = InitialTemperatureProfile::FROM_FILE;
    InitialPerturbation       perturbation = InitialPerturbation::SPHERICAL_HARMONICS;

    // Reference temperature from file
    std::string Tref_profile_csv_path  = "TemperatureProfile_3800K.csv";
    std::string Tref_profile_value_key = "Temperature (K)";

    double perturbation_amplitude = 5e-2;

    /// Surface value of the adiabat T_ad(r) = T_ad,s * exp( Di * (r_max - r) ), in units of
    /// the nondimensional temperature. Only read when profile == ADIABATIC.
    double adiabat_surface_temperature = 0.0;

    /// Spherical-harmonic perturbation degree l and order m (l >= 0, |m| <= l). Set l to 0 to
    /// disable the SH perturbation entirely (then perturbation_amplitude is ignored).
    int sph_degree_l = 0;
    int sph_order_m  = 0;

    /// Optional second spherical harmonic for combined modes (e.g. cubic symmetry
    /// T_perturb = Y_4^0 + (5/7) * Y_4^4).  Set sph_degree_l_2 > 0 to enable.
    /// The total perturbation is perturbation_amplitude * (Y_l1^m1 + sph_factor_2 * Y_l2^m2).
    int    sph_degree_l_2 = 0;
    int    sph_order_m_2  = 0;
    double sph_factor_2   = 0.0;

    /// Broadband initial perturbation: when sph_degree_max >= max(1, sph_degree_min),
    /// the perturbation is a random-coefficient sum of Y_l^m over l in
    /// [sph_degree_min, sph_degree_max], all orders m in [-l, l], amplitude-normalised
    /// and scaled by perturbation_amplitude. Breaks the symmetry that a single
    /// low-degree mode (e.g. Y_4^0) locks the flow into. Takes precedence over the
    /// single/double-mode path above. Set sph_degree_max = 0 to keep that behaviour.
    int          sph_degree_min  = 0;
    int          sph_degree_max  = 0;
    /// Fixed RNG seed for the broadband coefficients (reproducible ICs).
    unsigned int sph_random_seed = 42;
};

// Work in progress
enum class CompressibleForm
{
    TALA,
    PDA,
    PDA_ENTROPY
};

struct PhysicsParameters
{
    double gravity = 9.81;

    // Non-dimensional numbers
    double rayleigh_number    = 1e5;
    double peclet_number      = 1.0;
    double dissipation_number = 1.0;
    double h_number           = 1.0;

    double thermal_diffusivity_dim    = 1.0;
    double thermal_diffusivity_nondim = 1.0;
    double characteristic_velocity    = 1e-10; // diffusive velocity

    double reference_density      = 4500;
    double surface_density_dim    = 3300;
    double surface_density_nondim = 1.0;

    double thermal_expansivity    = 2.5e-5;
    double thermal_conductivity   = 3.0;
    double specific_heat_capacity = 1230;
    double grueneisen_parameter   = 1.1;

    /// Independent switches for the two compressible heating terms. Both default to true, so
    /// with --compressible the behaviour is unchanged; setting either to false isolates the
    /// contribution of the other (BA / EBA / TALA variants, per-term attribution).
    bool shear_heating     = true;
    bool adiabatic_heating = true;

    bool   internal_heating      = false;
    double internal_heating_rate = 3e-12;

    bool             compressible      = false;
    /// Master switch: enable all three TALA sub-ingredients at once.
    CompressibleForm compressible_form = CompressibleForm::TALA;

    double calc_cm_per_year = 3e-4; // from non-dim velocity to cm/a
    double calc_time_Ma     = 1e6;  // from non-dim time to Ma

    // Parameter radial profiles -- to be done
    std::string density_profile_csv_path = "";
    std::string alpha_profile_csv_path   = "";
    std::string cp_profile_csv_path      = "";

    std::string radial_profiles_radii_key = "Radius (m)";
    std::string density_profile_value_key = "rho (kg/m^3)";
    std::string alpha_profile_value_key   = "alpha (1/K)";
    std::string cp_profile_value_key      = "Cp (J/kg K)";

    double alpha_profile = 1.0;
    double cp_profile    = 1.0;

    ViscosityParameters          viscosity_parameters{};
    InitialTemperatureParameters initial_temperature{};
};

/// Storage/working precision of the velocity-block multigrid V-cycle preconditioner.
/// The outer Stokes FGMRES, block preconditioner and operators stay double; only the
/// MG hierarchy (operators, smoothers, transfers, coarse solve, level vectors) runs in
/// this precision, with convert at the preconditioner boundary. Lower precision trims
/// the MG memory; the outer double Krylov solve absorbs the preconditioner inexactness.
enum class MGPrecision
{
    DOUBLE,
    FLOAT,
    HALF,
};

struct StokesSolverParameters
{
    /// When non-empty, replace the block-triangular MG/Schur preconditioner of the
    /// outer Stokes FGMRES with terra::ml::NeuralSolver running the named registered
    /// model (e.g. "cband"; the checkpoint comes from $TERRA_NEURAL_CHECKPOINT).
    /// Needs a build with -DTERRA_ENABLE_PYTHON=ON.
    std::string neural_precon = "";

    /// When non-empty, run terra::ml::NeuralSolver with the named registered model
    /// ONCE on the Stokes rhs before the (unchanged) FGMRES solve, overwriting the
    /// initial guess. The rhs-trained operators are in-distribution for exactly
    /// this deployment -- predicting the solution from the physical rhs -- unlike
    /// the preconditioner slot, which feeds them Krylov residuals.
    std::string neural_guess = "";

    /// Initial-guess policy for warm-start benchmarks. guess_extrap = N uses a
    /// polynomial time-extrapolation of the stored solution history (1 linear,
    /// 2 quadratic, 3 cubic); 0 keeps the default persistence warm start.
    /// guess_zero cold-starts every solve (the baseline).
    int  guess_extrap = 0;
    bool guess_zero   = false;
    /// guess_proj = k: residual-optimal projection of the initial guess onto the
    /// span of the last k solutions for the current operator (k matvecs + k x k
    /// solve). Takes precedence over guess_extrap.
    int guess_proj = 0;
    /// Stop FGMRES at ||f - K u|| <= krylov_relative_tolerance * ||f|| instead of
    /// relative to the initial residual. The default criterion measures the
    /// reduction from the warm-started residual, so a good initial guess buys
    /// nothing; this one fixes the absolute accuracy of every solve.
    bool tolerance_relative_to_rhs = false;
    /// Write (r = f - K u_guess, e = u - u_guess, eta) of every solve as raw
    /// float64 into this directory (single-rank; warm-start training pairs).
    std::string dump_pairs_dir;

    /// Replace the outer FGMRES by damped iterative refinement
    /// x <- x + ir_damping * M (f - K x), with M the configured preconditioner
    /// (block MG/Schur, or the neural model when --stokes-neural-precon is set).
    /// The per-step contraction directly measures M's quality as an approximate
    /// inverse -- no Krylov line search or orthogonalisation in the way.
    bool   iterative_refinement = false;
    double ir_damping           = 1.0;
    /// Auto-damped refinement: after z = M r, one matvec w = K z and the step
    /// x += alpha * z with alpha = <r,w>/<w,w> -- the residual minimiser along
    /// z. Overshooting directions are scaled down instead of amplified; costs
    /// one extra operator application per step, still no Krylov basis.
    bool ir_linesearch = false;
    /// Hybrid refinement: alternate the neural preconditioner and the block
    /// MG/Schur preconditioner step by step. Measures whether the learned
    /// operator adds per-iteration value on top of MG's own contraction.
    bool ir_hybrid = false;
    /// Heavy-ball acceleration of the refinement:
    /// x <- x + ir_damping * M r + ir_momentum * (x - x_prev).
    /// For a stationary contraction with rate rho, the optimal momentum
    /// accelerates the asymptotic rate to ~(1-sqrt(1-rho))/(1+sqrt(1-rho)).
    double ir_momentum = 0.0;
    /// Self-tuning acceleration: estimate the contraction rate rho online from
    /// the residual ratios (EMA after a short plain warmup), set the momentum
    /// to the heavy-ball optimum ((1-sqrt(1-rho))/(1+sqrt(1-rho)))^2 for that
    /// rho, and reset the momentum memory whenever the residual rises.
    /// Level- and model-independent, unlike a fixed --stokes-ir-momentum.
    bool ir_auto_momentum = false;

    int    krylov_restart            = 10;
    int    krylov_max_iterations     = 10;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;

    /// Relaxation scale on the Schur (pressure) block of the block-triangular
    /// preconditioner: effective Ŝ^{-1}_prec = schur_relaxation · Ŝ^{-1}. Analogue of
    /// HyTeG's Uzawa `relaxParamSchur`. Default 1.0 = unscaled (legacy). The optimal
    /// value drifts with viscosity contrast → a robustness knob at high contrast.
    double schur_relaxation = 1.0;

    /// Strength of the rigid-rotation null-space penalty of the viscous block for free-slip/free-slip runs
    /// (EpsilonDivDivKerngen::set_penalty_epsilon). 1.0 reproduces the Aug/Sep 2026 production binaries; the
    /// mmoc-transport-v1 branch ran with 1e-7. Large values pollute the Chebyshev eigenvalue estimate.
    double penalty_epsilon = 1.0;

    /// Refresh the fine-level A-block MG smoother (D^-1 + Chebyshev eigenvalue
    /// bounds) on every viscosity update. Without this the smoother stays tuned to
    /// the INITIAL viscosity and the MG stalls once a strongly-varying viscosity
    /// evolves (mirrors HyTeG's refresh-on-viscosity-update). Default on; cheap.
    bool refresh_viscous_pc = true;

    /// Also coarsen the evolving fine viscosity down the MG hierarchy on each refresh
    /// (weighted-average restriction, non-GCA) and re-tune ALL levels' smoothers, so
    /// the coarse operators track the current viscosity. Targets the weak coarse-grid
    /// correction that caps convergence at very high viscosity contrast. Default off.
    bool refresh_coarse_viscosity = false;

    /// Store the outer FGMRES Krylov basis in single precision while the operator,
    /// preconditioner and orthogonalization stay double. Roughly halves the FGMRES
    /// workspace memory; convergence is unaffected (the operator never sees float).
    bool   float_krylov_basis        = false;

    /// Precision of the velocity-block multigrid V-cycle preconditioner (see MGPrecision).
    MGPrecision mg_precision = MGPrecision::DOUBLE;

    int viscous_pc_num_vcycles                 = 1;
    int viscous_pc_chebyshev_order             = 2;
    int viscous_pc_num_smoothing_steps_prepost = 2;
    int viscous_pc_num_power_iterations        = 10;

    /// Galerkin coarse-grid approximation mode for the multigrid preconditioner of the
    /// viscous Stokes block.
    ///   0 : disabled — coarse operators are re-discretised on every level (cheap setup,
    ///       but coarse smoothers see a different operator than the fine one).  Default.
    ///   1 : full GCA — store and assemble the full coarse-grid Galerkin matrices for
    ///       every coarse level.  Most robust setup; usually faster overall at high
    ///       viscosity contrast / variable viscosity.
    ///   2 : adaptive GCA — only store/assemble the coarse-grid matrices for elements
    ///       flagged by `GCAElementsCollector`, leaving the rest re-discretised.
    ///       Uses less memory than mode 1 but slightly less robust.
    int gca = 0;

    /// Per-descent agglomeration factors for the viscous MG preconditioner.
    /// Length must equal num_mg_levels - 1 (one factor per coarse descent step).
    /// Empty = classical MG, all levels on MPI_COMM_WORLD.
    /// Factor f > 1 at descent i means the comm shrinks by f ranks going from
    /// MG level max-i-1 to level max-i-2. Factor 1 = identity (no shrink).
    std::vector< int > viscous_pc_agglom_factors = {};

    // Low-memory mode: FP16 Krylov basis for the Stokes and energy solves, Stokes and
    // energy restart lowered -> 5, and a single pre/post velocity-MG smoothing step
    // (vs 2). These minimise the FGMRES workspace, the dominant memory term at high
    // dofs/GCD.
    bool low_mem = false;

    /// Store the energy FGMRES Krylov basis in single precision (operator stays
    /// double). The energy advection-diffusion solve is well-conditioned and runs
    /// fine in reduced precision; this trims the energy FGMRES workspace.

    /// Entropy-viscosity stabilization parameters (only used when
    /// `energy_solver == ENTROPY_VISCOSITY`).  Defaults match ASPECT.
    double ev_alpha_max = 0.078; ///< First-order upwind cap on ν_h (= 0.026·d in 3D).
    double ev_alpha_E   = 1.0;   ///< Residual-branch scale.

    /// Assemble the entropy-viscosity artificial diffusion implicitly (LHS,
    /// ASPECT-style) instead of the explicit RHS lag. Removes the explicit-diffusion

    /// If true, log global min/max/mean of the per-wedge ν_h field once per
    /// output_frequency to <outdir>/nu_h_stats.csv (timestep, min, max, mean).
    bool ev_dump_nu_h = false;
};

/// Time-discretization scheme for the energy (temperature) equation.
///   SUPG : implicit SUPG-stabilised Galerkin advection-diffusion on the Q1 mesh,
///          solved by FGMRES.  Unconditionally stable (dt only bounded by the
///          *advection* CFL for accuracy), so allows much larger dt at moderate Pe.
///          Linear-solver convergence degrades at high Pe (Ra >> 1e6).
///   ENTROPY_VISCOSITY : entropy-viscosity-stabilised Galerkin on the Q1 mesh.
enum class EnergySolverType
{
    SUPG,
    ENTROPY_VISCOSITY,
    MMOC,
    FCT,
};
struct EnergySolverParameters
{
    /// Treat the entropy viscosity implicitly in the diffusion solve.
    bool implicit_nu_h = false;
    /// Hold the Krylov basis in single precision (low-memory mode).
    bool float_krylov_basis = false;
    int    krylov_restart            = 5;
    int    krylov_max_iterations     = 100;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;
    int    energy_substeps           = 1;

    /// Store the energy FGMRES Krylov basis in single precision (operator stays
    /// double). The energy advection-diffusion solve is well-conditioned and runs
    /// fine in reduced precision; this trims the energy FGMRES workspace.

    /// Entropy-viscosity stabilization parameters (only used when
    /// `energy_solver == ENTROPY_VISCOSITY`).  Defaults match ASPECT.
    double ev_alpha_max = 0.078; ///< First-order upwind cap on ν_h (= 0.026·d in 3D).
    double ev_alpha_E   = 1.0;   ///< Residual-branch scale.

    /// If true, log global min/max/mean of the per-wedge ν_h field once per
    /// output_frequency to <outdir>/nu_h_stats.csv (timestep, min, max, mean).
    bool ev_dump_nu_h = false;

    EnergySolverType energy_solver = EnergySolverType::ENTROPY_VISCOSITY;
};

struct TimeSteppingParameters
{
    double dt_scaling = 0.5;
    double t_end_Ma   = 100.0;
    double t_end      = 1.0;
    double dt_max_Ma  = 5.0;
    double dt_min_Ma  = 0.1;
    double dt_max     = 1.0;
    double dt_min     = 1.0;

    int max_timesteps         = 10;
    int timestep_initial      = 0;
    int initial_dt_ramp_steps = 20;

    int energy_substeps   = 1;
    int picard_iterations = 1;

    EnergySolverType energy_solver = EnergySolverType::ENTROPY_VISCOSITY;
};

struct IOParameters
{
    std::string outdir          = "output";
    bool        overwrite       = false;
    bool        output_pressure = true;

    std::string xdmf_dir                = "xdmf";
    std::string radial_profiles_out_dir = "radial_profiles";
    std::string timer_trees_dir         = "timer_trees";

    bool        load_checkpoint = false;
    std::string checkpoint_dir;
    int         checkpoint_step = -1;

    int output_frequency = 1;

    bool no_xdmf            = false;
    bool no_radial_profiles = false;
};

// This struct holds options that might be useful for debugging, benchmarking, etc., but are not intended to appear at the surface for 'standard' use.
struct DeveloperOptions
{
    bool nondimensional_input = false;
    bool output_dimensional   = true;

    // Some logging and parameter options
    bool extended_parameters = false; // If false, hide some of the parameters that are not relevant for 'standard' use.
    bool extended_diagnostics         = true; // Extended logging of solver diagnostics, memory footprint, etc.
    bool print_parameter_descriptions = true;

    /// Abort threshold for the negative-temperature guard, in units of the nondimensional
    /// temperature. MMOC is semi-Lagrangian with a quintic reconstruction and therefore not
    /// monotone, so a small undershoot below T_min is expected and harmless; a strict
    /// "any negative value" test aborts healthy runs on it. The run stops once the minimum
    /// temperature falls below -negative_temperature_tolerance; the actual minimum is logged
    /// whenever it sets a new (negative) record, so a genuine blow-up stays visible.
    /// Set to 0 to restore the strict behaviour.
    double negative_temperature_tolerance = 1e-3;
};

struct Parameters
{
    MeshParameters               mesh_parameters;
    BoundaryConditionsParameters boundary_parameters;
    StokesSolverParameters       stokes_solver_parameters;
    EnergySolverParameters       energy_solver_parameters;
    PhysicsParameters            physics_parameters;
    TimeSteppingParameters       time_stepping_parameters;
    IOParameters                 io_parameters;
    DeveloperOptions             devel_parameters;

    std::string output_config_file;
};

struct CLIHelp
{};

inline void nondimensionalise( Parameters& prm )
{
    auto& phys     = prm.physics_parameters;
    auto& mesh     = prm.mesh_parameters;
    auto& boundary = prm.boundary_parameters;
    auto& devel    = prm.devel_parameters;

    // --rho-weighted-tala is a master switch: enable all three TALA sub-ingredients.
    // (Individual --tala-* flags let us isolate one ingredient without the master.)
    auto& time     = prm.time_stepping_parameters;

    // Nondimensionalise, if not specified otherwise
    if ( !devel.nondimensional_input )
    {
        // --- Domain ---

        // Nondimensionalise radii with mantle thickness -- rescaled to unit sphere for output
        mesh.mantle_thickness_m = mesh.radius_surface_m - mesh.radius_cmb_m;
        mesh.radius_max         = mesh.radius_surface_m / mesh.mantle_thickness_m;
        mesh.radius_min         = mesh.radius_cmb_m / mesh.mantle_thickness_m;

        // --- Boundary conditions ---

        boundary.temperature_min = boundary.temperature_surface_K / boundary.delta_T_K;
        boundary.temperature_max = boundary.temperature_cmb_K / boundary.delta_T_K;

        // Frank-Kamenetskii contrast -> upstream's activation-energy field. The two FK
    // forms read that field differently:
    //   fk-benchmark: eta = E^(0.5 - T)          -> E must be rmu itself
    //   fk-type1/2/3: eta = eta_ref * exp(-E*T)  -> E must be ln(rmu)
    // rmu == 1 (isoviscous benchmark) must convert too: for fk-benchmark E is the base of
    // rmu^(0.5-T), so leaving E at its default (ln 100 = 4.605) silently gives a 4.6x contrast.
    if ( phys.viscosity_parameters.rmu > 0.0 )
    {
        const auto law = phys.viscosity_parameters.law;
        if ( law == ViscosityLaw::FK_BENCHMARK )
            phys.viscosity_parameters.activation_energy = phys.viscosity_parameters.rmu;
        else if ( law == ViscosityLaw::FK_TYPE1 || law == ViscosityLaw::FK_TYPE2 ||
                  law == ViscosityLaw::FK_TYPE3 )
            phys.viscosity_parameters.activation_energy = std::log( phys.viscosity_parameters.rmu );
    }

    // Surface density
        phys.surface_density_nondim = phys.surface_density_dim / phys.reference_density;

        // Viscosity limits. The reference viscosity is the scale that sets Ra, so it is
        // deliberately NOT clamped into [min, max]: upstream clamped it silently, which
        // rescaled Ra (A3 benchmark: 2.46e25 Pa s -> 1e25, Ra 7e3 -> 1.7e4) without any
        // trace beyond the printed Ra. Warn instead; the field clamp below still applies.
        if ( phys.viscosity_parameters.reference_viscosity < phys.viscosity_parameters.min_viscosity ||
             phys.viscosity_parameters.reference_viscosity > phys.viscosity_parameters.max_viscosity )
        {
            util::logroot << "[WARN] reference viscosity " << phys.viscosity_parameters.reference_viscosity
                          << " Pa s lies outside the viscosity clamp range [" << phys.viscosity_parameters.min_viscosity
                          << ", " << phys.viscosity_parameters.max_viscosity
                          << "] Pa s; the nondimensional field clamp will cut the law at "
                          << phys.viscosity_parameters.max_viscosity / phys.viscosity_parameters.reference_viscosity
                          << " x reference. Raise --viscosity-max / lower --viscosity-min if that is not intended."
                          << std::endl;
        }
        phys.viscosity_parameters.max_viscosity /= phys.viscosity_parameters.reference_viscosity;
        phys.viscosity_parameters.min_viscosity /= phys.viscosity_parameters.reference_viscosity;

        // Compute characteristic velocity and thermal diffusivity
        phys.characteristic_velocity =
            phys.thermal_conductivity /
            ( phys.reference_density * phys.specific_heat_capacity * mesh.mantle_thickness_m );

        phys.thermal_diffusivity_dim =
            phys.thermal_conductivity / ( phys.reference_density * phys.specific_heat_capacity );

        // Precompute conversion factors from non-dim to dimensional quantities
        phys.calc_cm_per_year = phys.characteristic_velocity * 60 * 60 * 24 * 365 * 100; // Velocity in cm/a

        phys.calc_time_Ma = mesh.mantle_thickness_m / ( phys.calc_cm_per_year * 1e4 ); // Time in Ma

        // Acount for plate velocity scaling
        if ( boundary.plate_parameters.apply_plate_velocities )
        {
            phys.calc_time_Ma /= boundary.plate_parameters.plate_velocity_scaling;
        }

        // Nondimensionalise time
        time.t_end  = time.t_end_Ma / phys.calc_time_Ma;
        time.dt_max = time.dt_max_Ma / phys.calc_time_Ma;
        time.dt_min = time.dt_min_Ma / phys.calc_time_Ma;

        // Compute nondimensional numbers
        // Rayleigh number = ( rho * alpha * g * L^3 * dT ) / ( eta * kappa )
        phys.rayleigh_number = ( phys.reference_density * phys.gravity * phys.thermal_expansivity *
                                 std::pow( mesh.mantle_thickness_m, 3 ) * boundary.delta_T_K ) /
                               ( phys.viscosity_parameters.reference_viscosity * phys.thermal_diffusivity_dim );

        // Peclet number = ( U * L ) / kappa -> should be 1
        phys.peclet_number = ( phys.characteristic_velocity * mesh.mantle_thickness_m ) / phys.thermal_diffusivity_dim;

        // Dissipation number = ( alpha * g * L ) / Cp
        phys.dissipation_number =
            ( phys.thermal_expansivity * phys.gravity * mesh.mantle_thickness_m ) / phys.specific_heat_capacity;

        // H-number = ( H * L ) / ( Cp * U * dT )
        phys.h_number = ( phys.internal_heating_rate * mesh.mantle_thickness_m ) /
                        ( phys.specific_heat_capacity * phys.characteristic_velocity * boundary.delta_T_K );
    }

    // else, take values directly from parameter file
    else
    {
        mesh.radius_max          = mesh.radius_surface_m;
        mesh.radius_min          = mesh.radius_cmb_m;
        boundary.temperature_min = boundary.temperature_surface_K;
        boundary.temperature_max = boundary.temperature_cmb_K;
        phys.h_number            = phys.internal_heating_rate;
        time.t_end               = time.t_end_Ma;
        time.dt_max              = time.dt_max_Ma;
        time.dt_min              = time.dt_min_Ma;

        std::clamp(
            phys.viscosity_parameters.reference_viscosity,
            phys.viscosity_parameters.min_viscosity,
            phys.viscosity_parameters.max_viscosity );

        phys.thermal_expansivity  = 1.0;
        phys.thermal_conductivity = 1.0;
        phys.reference_density    = 1.0;

        // Pe, Di, diffusivity should already be set to 1.
        // Ra is set through parameter file in this case.
    }
}

inline util::Result< std::variant< CLIHelp, Parameters > > parse_parameters( int argc, char** argv )
{
    Parameters parameters{};

    using util::add_flag_with_default;
    using util::add_option_with_default;

    // Minimal pre-parse to learn about --extended-parameters and --low-mem
    CLI::App pre{ "pre-parse" };
    add_flag_with_default( pre, "--extended-parameters", parameters.devel_parameters.extended_parameters )->group( "" );
    add_flag_with_default( pre, "--low-mem", parameters.stokes_solver_parameters.low_mem )->group( "" );
    pre.set_config( "--config" );
    pre.set_help_flag( "" );  //disable help-flag, so real app handles --help
    pre.allow_extras( true ); // ignore unrecognized command-line args

    try
    {
        pre.parse( argc, argv );
    }
    catch ( const CLI::ParseError& e )
    {
        return { "CLI parse error" };
    }

    // Set new defaults for low-memory mode
    if ( parameters.stokes_solver_parameters.low_mem )
    {
        parameters.stokes_solver_parameters.krylov_restart                         = 5;
        parameters.energy_solver_parameters.krylov_restart                         = 5;
        parameters.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost = 1;
        //float_krylov_basis parameter is set in regular parse, does not appear in parameter file.
    }

    // Regular parse
    CLI::App app{ "Mantle circulation simulation" };

    app.config_formatter( std::make_shared< ConfigGroupedNoDescriptions >() );

    // Allow config files
    app.set_config( "--config" );

    // --low-mem preset. Detect it BEFORE registering the options below so its five
    // constituent settings become the captured CLI *defaults*: FP16 Krylov basis for the
    // Stokes and energy solves, Stokes restart lowered 10 -> 5 (energy is already 5), and a
    // single pre/post velocity-MG smoothing step (vs 2). These minimise the FGMRES
    // workspace, the dominant memory term at high dofs/GCD. Making them defaults (rather
    // than patching after parse) means an explicit --stokes-*/--energy-* flag still
    // overrides through normal parsing, and --write-config-and-exit emits the expanded
    // preset so a generated config round-trips correctly.
    bool low_mem = false;
    for ( int i = 1; i < argc; ++i )
    {
        const std::string a{ argv[i] };
        if ( a == "--low-mem" || a.rfind( "--low-mem=", 0 ) == 0 )
        {
            low_mem = true;
            break;
        }
    }
    if ( low_mem )
    {
        parameters.stokes_solver_parameters.float_krylov_basis                     = true;
        parameters.energy_solver_parameters.float_krylov_basis                     = true;
        parameters.stokes_solver_parameters.krylov_restart                         = 5;
        parameters.energy_solver_parameters.krylov_restart                         = 5;
        parameters.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost = 1;
    }

    ///////////////
    /// General ///
    ///////////////

    add_option_with_default(
        app,
        "--write-config-and-exit",
        parameters.output_config_file,
        "Writes a config file with the passed (or default arguments) to the desired"
        "location to be then modified and passed."
        "E.g., '--write-config-and-exit my-config.toml'.\n" );

    ///////////////////////
    /// Domain and mesh ///
    ///////////////////////

    add_option_with_default( app, "--refinement-level-mesh-min", parameters.mesh_parameters.refinement_level_mesh_min )
        ->group( "Domain" );
    add_option_with_default( app, "--refinement-level-mesh-max", parameters.mesh_parameters.refinement_level_mesh_max )
        ->group( "Domain" );

    add_option_with_default(
        app, "--refinement-level-subdomains", parameters.mesh_parameters.refinement_level_subdomains )
        ->group( "Domain" );

    add_option_with_default( app, "--radius-cmb", parameters.mesh_parameters.radius_cmb_m )->group( "Domain" );
    add_option_with_default( app, "--radius-surface", parameters.mesh_parameters.radius_surface_m )->group( "Domain" );

    if ( parameters.devel_parameters.extended_parameters )
    {
        add_option_with_default( app, "--radial-extra-levels", parameters.mesh_parameters.radial_extra_levels )
            ->group( "Anisotropic mesh refinement" )
            ->description( "Per-MG-level offset added to the radial diamond refinement level relative to the "
                           "lateral one. Radial level at each MG level L becomes L + radial_extra_levels, so the "
                           "hierarchy coarsens uniformly in both axes. Default -1." );
        add_option_with_default( app, "--lat-sdr", parameters.mesh_parameters.lat_sdr )
            ->group( "Anisotropic mesh refinement" )
            ->description(
                "Override the lateral subdomain refinement level (otherwise --refinement-level-subdomains is used)." );
        add_option_with_default( app, "--rad-sdr", parameters.mesh_parameters.rad_sdr )
            ->group( "Anisotropic mesh refinement" )
            ->description(
                "Override the radial subdomain refinement level (otherwise --refinement-level-subdomains is used)." );

        std::map< std::string, MeshParameters::RadialDistribution > radial_distribution_map{
            { "uniform", MeshParameters::RadialDistribution::UNIFORM },
            { "tanh-both", MeshParameters::RadialDistribution::TANH_BOTH },
            { "tanh-cmb", MeshParameters::RadialDistribution::TANH_CMB },
            { "tanh-surface", MeshParameters::RadialDistribution::TANH_SURFACE },
        };
        add_option_with_default( app, "--radial-distribution", parameters.mesh_parameters.radial_distribution )
            ->transform( CLI::CheckedTransformer( radial_distribution_map, CLI::ignore_case ) )
            ->default_val( "uniform" )
            ->group( "Anisotropic mesh refinement" )
            ->description( "Radial shell distribution: 'uniform' (equispaced, default), 'tanh-both' "
                           "(cluster at both CMB and surface), 'tanh-cmb' (cluster at CMB only), "
                           "'tanh-surface' (cluster at surface only).  Cluster strength is set by "
                           "--radial-cluster-k." );
        add_option_with_default( app, "--radial-cluster-k", parameters.mesh_parameters.radial_cluster_k )
            ->group( "Anisotropic mesh refinement" )
            ->description( "Cluster-strength k for the tanh-based radial distributions.  k <= 0 "
                           "collapses to uniform; k ~ 1 mild clustering, k ~ 2 strong clustering." );
    }

    ///////////////////////////
    /// Boundary conditions ///
    ///////////////////////////

    std::map< std::string, BoundaryConditionsParameters::VelocityBC > velocity_bc_cmb_map{
        { "noslip", BoundaryConditionsParameters::VelocityBC::NO_SLIP },
        { "freeslip", BoundaryConditionsParameters::VelocityBC::FREE_SLIP },
    };

    std::map< std::string, BoundaryConditionsParameters::VelocityBC > velocity_bc_surface_map{
        { "noslip", BoundaryConditionsParameters::VelocityBC::NO_SLIP },
        { "freeslip", BoundaryConditionsParameters::VelocityBC::FREE_SLIP },
    };

    add_option_with_default( app, "--velocity-bc-cmb", parameters.boundary_parameters.velocity_bc_cmb )
        ->transform( CLI::CheckedTransformer( velocity_bc_cmb_map, CLI::ignore_case ) )
        ->default_val( "noslip" )
        ->group( "Boundary Conditions" );

    add_option_with_default( app, "--velocity-bc-surface", parameters.boundary_parameters.velocity_bc_surface )
        ->transform( CLI::CheckedTransformer( velocity_bc_surface_map, CLI::ignore_case ) )
        ->default_val( "noslip" )
        ->group( "Boundary Conditions" );

    add_option_with_default( app, "--temperature-cmb", parameters.boundary_parameters.temperature_cmb_K )
        ->group( "Boundary Conditions" );

    add_option_with_default( app, "--temperature-surface", parameters.boundary_parameters.temperature_surface_K )
        ->group( "Boundary Conditions" );

    // Plate parameters
    add_flag_with_default( app, "--plates-on-device", parameters.boundary_parameters.plate_parameters.plates_on_device )
        ->group( "Plate Parameters" )
        ->description( "Evaluate plate velocities on the device rather than on the host." );

    add_option_with_default(
        app, "--apply-plate-velocities", parameters.boundary_parameters.plate_parameters.apply_plate_velocities )
        ->group( "Plate Parameters" )
        ->description(
            "Assimilate plate velocities as surface boundary conditions. Enforces a no-slip condition at the surface." );

    add_option_with_default(
        app, "--initial-plate-age-Ma", parameters.boundary_parameters.plate_parameters.initial_plate_age )
        ->group( "Plate Parameters" );

    add_option_with_default(
        app, "--final-plate-age-Ma", parameters.boundary_parameters.plate_parameters.final_plate_age )
        ->group( "Plate Parameters" );

    add_flag_with_default( app,
                           "--interpolate-plates-in-time",
                           parameters.boundary_parameters.plate_parameters.interpolate_plates_in_time )
        ->group( "Plate Parameters" )
        ->description( "Interpolate between plate stages defined in the plate data (usually every 1 Ma)." );

    add_option_with_default(
        app, "--plates-topologies-path", parameters.boundary_parameters.plate_parameters.plates_topologies_path )
        ->group( "Plate Parameters" )
        ->description( "Paths to plate data." );

    add_option_with_default( app,
                             "--plates-reconstructions-path",
                             parameters.boundary_parameters.plate_parameters.plates_reconstructions_path )
        ->group( "Plate Parameters" );

    add_option_with_default(
        app, "--plate-velocity-scaling", parameters.boundary_parameters.plate_parameters.plate_velocity_scaling )
        ->group( "Plate Parameters" );

    // Non-dimensional temperature BC values. Only honoured when --set-nondimensional-numbers
    // is set; otherwise nondimensionalise() overwrites them from temperature-cmb/-surface / dT.
    add_option_with_default( app, "--temperature-bc-value-cmb", parameters.boundary_parameters.temperature_max )
        ->group( "Boundary Conditions" );
    add_option_with_default( app, "--temperature-bc-value-surface", parameters.boundary_parameters.temperature_min )
        ->group( "Boundary Conditions" );

    //////////////////////////////
    /// Geophysical parameters ///
    //////////////////////////////
    add_flag_with_default( app, "--compressible", parameters.physics_parameters.compressible )
        ->group( "Physical Parameters" );
    add_flag_with_default( app, "--shear-heating", parameters.physics_parameters.shear_heating )
        ->group( "Physical Parameters" )
        ->description( "Viscous dissipation source in the energy equation; only acts with --compressible." );
    add_flag_with_default( app, "--adiabatic-heating", parameters.physics_parameters.adiabatic_heating )
        ->group( "Physical Parameters" )
        ->description( "Adiabatic (de)compression source in the energy equation; only acts with --compressible." );
    add_option_with_default( app, "--grueneisen-parameter", parameters.physics_parameters.grueneisen_parameter )
        ->group( "Physical Parameters" )
        ->description( "Grueneisen parameter of the Adams-Williamson profile "
                       "rho(r) = rho_s * exp( Di*(r_max-r)/gamma ); larger flattens the density contrast." );
    add_option_with_default( app, "--surface-density", parameters.physics_parameters.surface_density_dim )
        ->group( "Physical Parameters" )
        ->description( "Surface density of the Adams-Williamson profile, kg/m^3." );

    std::map< std::string, CompressibleForm > compressible_form_map{
        { "tala", CompressibleForm::TALA },
        { "pda", CompressibleForm::PDA },
        { "pda-entropy", CompressibleForm::PDA_ENTROPY },
    };
    add_option_with_default( app, "--compressibility-formulation", parameters.physics_parameters.compressible_form )
        ->transform( CLI::CheckedTransformer( compressible_form_map, CLI::ignore_case ) )
        ->default_val( "tala" )
        ->group( "Physical Parameters" )
        ->description(
            "Formulation of compressibility, if active: Choose between 'tala', 'pda' (not yet supported) and 'pda-entropy' (not yet supported). See Gassmöller et al. (2020)." );

    add_flag_with_default( app, "--internal-heating-enabled", parameters.physics_parameters.internal_heating )
        ->group( "Physical Parameters" );
    add_option_with_default( app, "--internal-heating-rate", parameters.physics_parameters.internal_heating_rate )
        ->group( "Physical Parameters" );
    add_option_with_default( app, "--reference-density", parameters.physics_parameters.reference_density )
        ->group( "Physical Parameters" );
    add_option_with_default( app, "--thermal-expansivity", parameters.physics_parameters.thermal_expansivity )
        ->group( "Physical Parameters" );
    add_option_with_default( app, "--thermal-conductivity", parameters.physics_parameters.thermal_conductivity )
        ->group( "Physical Parameters" );
    add_option_with_default( app, "--specific-heat-capacity", parameters.physics_parameters.specific_heat_capacity )
        ->group( "Physical Parameters" );

    // Radial input profiles
    add_option_with_default( app, "--density-profile-path", parameters.physics_parameters.density_profile_csv_path )
        ->group( "Radial input profiles" )
        ->description(
            "File paths for custom radial input profiles. Leave empty for radially constant parameters / predefined solutions." );
    add_option_with_default( app, "--alpha-profile-path", parameters.physics_parameters.alpha_profile_csv_path )
        ->group( "Radial input profiles" );
    add_option_with_default( app, "--cp-profile-path", parameters.physics_parameters.cp_profile_csv_path )
        ->group( "Radial input profiles" );

    // Viscosity parameters
    add_option_with_default(
        app, "--reference-viscosity", parameters.physics_parameters.viscosity_parameters.reference_viscosity )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-profile-csv-path",
        parameters.physics_parameters.viscosity_parameters.viscosity_profile_csv_path )
        ->group( "Viscosity" );

    std::map< std::string, ViscosityLaw > viscosity_law_map{
        { "constant", ViscosityLaw::CONSTANT },
        { "fk-benchmark", ViscosityLaw::FK_BENCHMARK },
        { "fk-type1", ViscosityLaw::FK_TYPE1 },
        { "fk-type2", ViscosityLaw::FK_TYPE2 },
        { "fk-type3", ViscosityLaw::FK_TYPE3 },
        { "fk1", ViscosityLaw::FK_TYPE1 },
        { "fk2", ViscosityLaw::FK_TYPE2 },
        { "fk3", ViscosityLaw::FK_TYPE3 },
        { "arrhenius", ViscosityLaw::ARRHENIUS },
    };

    add_option_with_default( app, "--viscosity-law", parameters.physics_parameters.viscosity_parameters.law )
        ->transform( CLI::CheckedTransformer( viscosity_law_map, CLI::ignore_case ) )
        ->default_val( "constant" )
        ->group( "Viscosity" )
        ->description(
            "Viscosity law to use. 'constant' uses a constant or radial profile. Use 'fk-type{i}'/'fk{i}' with i between 1 and 3 for different variants of FRANK KAMENETSKII type laws that tie to a radial reference profile. 'fk-benchmark' computes eta = rmu^(0.5 - T) (Zhong et al. 2008), with parameter activation_energy in place of rmu." );

    add_option_with_default(
        app, "--activation-energy", parameters.physics_parameters.viscosity_parameters.activation_energy )
        ->group( "Viscosity" )
        ->description(
            "E_a controls magnitude of temp-dependence. Full viscosity contrast is O(e^E_a) for FK types 1 to 3 and O(E_a) for fk-benchmark." );
    add_option_with_default(
        app, "--activation_volume", parameters.physics_parameters.viscosity_parameters.activation_volume )
        ->group( "Viscosity" )
        ->description(
            "V_a > 0 adds a further depth-dependence  as exp( E_a * T + V_a * d ). Counteracts the natural decrease of viscosity with depth from pure (absolute) temp-dependence." );
    add_option_with_default( app, "--max-viscosity", parameters.physics_parameters.viscosity_parameters.max_viscosity )
        ->group( "Viscosity" )
        ->description( "Maximum and minimum limits to viscosity" );
    add_option_with_default( app, "--min-viscosity", parameters.physics_parameters.viscosity_parameters.min_viscosity )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-radii-key",
        parameters.physics_parameters.viscosity_parameters.viscosity_profile_radii_key )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-value-key",
        parameters.physics_parameters.viscosity_parameters.viscosity_profile_value_key )
        ->group( "Viscosity" );

    


    add_option_with_default( app, "--viscosity-rmu", parameters.physics_parameters.viscosity_parameters.rmu )
        ->group( "Viscosity" )
        ->description( "Base of the Frank-Kamenetskii viscosity law: eta = rmu^(0.5 - T) "
                       "(Zhong et al. 2008). Cold/hot contrast = rmu; rmu = 1 gives constant viscosity." );

    // Arrhenius (HyTeG type-3) law parameters: eta = clamp(exp(E_a*(1/(T+0.25)-1.45)+D*depth), min, max).
    add_option_with_default(
        app, "--viscosity-activation-energy", parameters.physics_parameters.viscosity_parameters.activation_energy )
        ->group( "Viscosity" )
        ->description( "Arrhenius activation energy E_a (--viscosity-law arrhenius)." );
    add_option_with_default(
        app, "--viscosity-depth-factor", parameters.physics_parameters.viscosity_parameters.activation_volume )
        ->group( "Viscosity" )
        ->description( "Arrhenius linear depth factor D (HyTeG depthViscosityFactor)." );
    add_option_with_default(
        app, "--viscosity-min", parameters.physics_parameters.viscosity_parameters.min_viscosity )
        ->group( "Viscosity" )
        ->description( "Lower clamp on the (nondim) Arrhenius viscosity (HyTeG minVisc)." );
    add_option_with_default(
        app, "--viscosity-max", parameters.physics_parameters.viscosity_parameters.max_viscosity )
        ->group( "Viscosity" )
        ->description( "Upper clamp on the (nondim) Arrhenius viscosity (HyTeG maxVisc)." );

    ///////////////////////////////
    /// Initial temperature      ///
    ///////////////////////////////

    std::map< std::string, InitialTemperatureProfile > init_temp_profile_map{
        { "power-law", InitialTemperatureProfile::POWER_LAW },
        { "conductive", InitialTemperatureProfile::CONDUCTIVE },
        { "adiabatic", InitialTemperatureProfile::ADIABATIC },
        { "from-file", InitialTemperatureProfile::FROM_FILE } };

    add_option_with_default(
        app, "--initial-temperature-profile", parameters.physics_parameters.initial_temperature.profile )
        ->transform( CLI::CheckedTransformer( init_temp_profile_map, CLI::ignore_case ) )
        ->default_val( "from-file" )
        ->group( "Initial Temperature" )
        ->description( "'power-law': T = ((r_max-r)/(r_max-r_min))^5 (default). "
                       "'from-file': read custom temperature profile from file."
                       "'conductive': T_ref = (r_min*r_max/r - r_min)/(r_max - r_min)." );

    add_option_with_default(
        app, "--temperature-profile-path", parameters.physics_parameters.initial_temperature.Tref_profile_csv_path )
        ->group( "Initial Temperature" )
        ->description( "Path to file for '--initial-temperature-profile == from-file'." );

    std::map< std::string, InitialPerturbation > init_temp_perturbation_map{
        { "noise", InitialPerturbation::NOISE },
        { "spherical-harmonics", InitialPerturbation::SPHERICAL_HARMONICS },
        { "sph", InitialPerturbation::SPHERICAL_HARMONICS } };

    add_option_with_default(
        app, "--initial-temperature-perturbation", parameters.physics_parameters.initial_temperature.perturbation )
        ->transform( CLI::CheckedTransformer( init_temp_perturbation_map, CLI::ignore_case ) )
        ->default_val( "spherical-harmonics" )
        ->group( "Initial Temperature" )
        ->description( "'noise' or 'spherical-harmonics' / 'sph' perturbation." );

    add_option_with_default(
        app, "--perturbation-amplitude", parameters.physics_parameters.initial_temperature.perturbation_amplitude )
        ->group( "Initial Temperature" )
        ->description( "Fractional amplitude of added noise/spherical harmonic perturbation." );
    add_option_with_default(
        app,
        "--adiabat-surface-temperature",
        parameters.physics_parameters.initial_temperature.adiabat_surface_temperature )
        ->group( "Physical Parameters" )
        ->description(
            "Surface temperature of the adiabat used by --initial-temperature-profile adiabatic, as both the "
            "initial condition and the radial reference for the buoyancy deviation." );

    add_option_with_default(
        app, "--initial-temperature-sph-degree", parameters.physics_parameters.initial_temperature.sph_degree_l )
        ->group( "Initial Temperature" )
        ->description( "Spherical harmonic degree l for initial temperature perturbation (0 = none)." );

    add_option_with_default(
        app, "--initial-temperature-sph-order", parameters.physics_parameters.initial_temperature.sph_order_m )
        ->group( "Initial Temperature" )
        ->description( "Spherical harmonic order m for initial temperature perturbation." );

    add_option_with_default(
        app, "--initial-temperature-sph-degree-2", parameters.physics_parameters.initial_temperature.sph_degree_l_2 )
        ->group( "Initial Temperature" )
        ->description( "Optional second spherical harmonic degree l2 (0 = none). For combined modes." );

    add_option_with_default(
        app, "--initial-temperature-sph-order-2", parameters.physics_parameters.initial_temperature.sph_order_m_2 )
        ->group( "Initial Temperature" )
        ->description( "Optional second spherical harmonic order m2." );

    add_option_with_default(
        app, "--initial-temperature-sph-factor-2", parameters.physics_parameters.initial_temperature.sph_factor_2 )
        ->group( "Initial Temperature" )
        ->description(
            "Weight factor for second spherical harmonic: T += eps * envelope * (Y_l1^m1 + factor_2 * Y_l2^m2)." );

    add_option_with_default(
        app, "--initial-temperature-sph-degree-min", parameters.physics_parameters.initial_temperature.sph_degree_min )
        ->group( "Initial Temperature" )
        ->description( "Broadband IC: lowest spherical-harmonic degree l (enable with --...-sph-degree-max >= 1)." );
    add_option_with_default(
        app, "--initial-temperature-sph-degree-max", parameters.physics_parameters.initial_temperature.sph_degree_max )
        ->group( "Initial Temperature" )
        ->description( "Broadband IC: highest degree l. Sum of random Y_l^m over [min,max] (HyTeG-style, e.g. 10..25)." );
    add_option_with_default(
        app, "--initial-temperature-sph-seed", parameters.physics_parameters.initial_temperature.sph_random_seed )
        ->group( "Initial Temperature" )
        ->description( "Fixed RNG seed for the broadband IC coefficients (reproducible)." );

    ///////////////////////////
    /// Time discretization ///
    ///////////////////////////

    add_option_with_default( app, "--dt-scaling", parameters.time_stepping_parameters.dt_scaling )
        ->description(
            "A robust (stable) dt is computed the the actual face-normal velocity fluxes and cell volumes via a "
            "parallel reduce over all cells. However, a smaller value might still be desired due to accuracy "
            "considerations. You can scale the computed dt using this value (e.g. set to 0.5 to half the estimated dt, "
            "set to 1.0 to just use the estimated dt)." )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--t-end", parameters.time_stepping_parameters.t_end_Ma )
        ->group( "Time Discretization" )
        ->description( "Final time in Ma." );
    add_option_with_default( app, "--dt-max", parameters.time_stepping_parameters.dt_max_Ma )
        ->group( "Time Discretization" )
        ->description( "Maximum/minimum time step size in Ma" );
    add_option_with_default( app, "--dt-min", parameters.time_stepping_parameters.dt_min_Ma )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--max-timesteps", parameters.time_stepping_parameters.max_timesteps )
        ->group( "Time Discretization" )
        ->description(
            "Simulation aborts when this time step index is reached. "
            "If a checkpoint is loaded, the simulation will start at the next step after the loaded checkpoint. "
            "This means the number of time steps executed might be smaller than what is passed in here." );
    add_option_with_default( app, "--initial-ramp-steps", parameters.time_stepping_parameters.initial_dt_ramp_steps )
        ->group( "Time Discretization" )
        ->description( "Amount of ramp-up timesteps at the beginning of the run." );
    add_option_with_default( app, "--energy-substeps", parameters.time_stepping_parameters.energy_substeps )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--picard-iterations", parameters.time_stepping_parameters.picard_iterations )
        ->group( "Time Discretization" )
        ->description( "Number of Picard (fixed-point) iterations per timestep. "
                       "Each iteration re-solves Stokes and energy from the same starting temperature. "
                       "Default: 1 (no iteration, current behavior)." );

    std::map< std::string, EnergySolverType > energy_solver_map{
        { "supg", EnergySolverType::SUPG },
        { "entropy_viscosity", EnergySolverType::ENTROPY_VISCOSITY },
        { "ev", EnergySolverType::ENTROPY_VISCOSITY },
        { "mmoc", EnergySolverType::MMOC },
        { "fct", EnergySolverType::FCT },
    };

    /////////////////////
    /// Stokes solver ///
    /////////////////////

    // --low-mem preset flag (its effect is applied up front in the pre-scan near the top
    // of this function, before the option defaults below are captured).
    add_flag_with_default( app, "--low-mem", low_mem )
        ->group( "Stokes Solver" )
        ->description(
            "Low-memory solver preset. Equivalent to --stokes-float-krylov-basis "
            "--energy-float-krylov-basis --stokes-krylov-restart 5 --energy-krylov-restart 5 "
            "--stokes-viscous-pc-num-smoothing-steps-prepost 1. Individual flags passed "
            "explicitly override the corresponding preset value." );

    add_option_with_default( app, "--stokes-krylov-restart", parameters.stokes_solver_parameters.krylov_restart )
        ->group( "Stokes Solver" );
    add_option_with_default( app, "--stokes-neural-precon", parameters.stokes_solver_parameters.neural_precon )
        ->group( "Stokes Solver" );
    add_option_with_default( app, "--stokes-neural-guess", parameters.stokes_solver_parameters.neural_guess )
        ->group( "Stokes Solver" );
    add_option_with_default( app, "--stokes-guess-extrap", parameters.stokes_solver_parameters.guess_extrap )
        ->group( "Stokes Solver" );
    add_flag_with_default( app, "--stokes-guess-zero", parameters.stokes_solver_parameters.guess_zero )
        ->group( "Stokes Solver" );
    add_option_with_default( app, "--stokes-guess-proj", parameters.stokes_solver_parameters.guess_proj )
        ->group( "Stokes Solver" );
    add_option_with_default( app, "--stokes-dump-pairs", parameters.stokes_solver_parameters.dump_pairs_dir )
        ->group( "Stokes Solver" );
    add_flag_with_default( app,
                           "--stokes-krylov-tolerance-relative-to-rhs",
                           parameters.stokes_solver_parameters.tolerance_relative_to_rhs )
        ->group( "Stokes Solver" );
    add_flag_with_default(
        app, "--stokes-iterative-refinement", parameters.stokes_solver_parameters.iterative_refinement )
        ->group( "Stokes Solver" );
    add_option_with_default( app, "--stokes-ir-damping", parameters.stokes_solver_parameters.ir_damping )
        ->group( "Stokes Solver" );
    add_flag_with_default( app, "--stokes-ir-linesearch", parameters.stokes_solver_parameters.ir_linesearch )
        ->group( "Stokes Solver" );
    add_flag_with_default( app, "--stokes-ir-hybrid", parameters.stokes_solver_parameters.ir_hybrid )
        ->group( "Stokes Solver" );
    add_option_with_default( app, "--stokes-ir-momentum", parameters.stokes_solver_parameters.ir_momentum )
        ->group( "Stokes Solver" );
    add_flag_with_default(
        app, "--stokes-ir-auto-momentum", parameters.stokes_solver_parameters.ir_auto_momentum )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-max-iterations", parameters.stokes_solver_parameters.krylov_max_iterations )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-relative-tolerance", parameters.stokes_solver_parameters.krylov_relative_tolerance )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-absolute-tolerance", parameters.stokes_solver_parameters.krylov_absolute_tolerance )
        ->group( "Stokes Solver" );
    static const std::map< std::string, MGPrecision > mg_precision_map{
        { "double", MGPrecision::DOUBLE },
        { "float", MGPrecision::FLOAT },
        { "single", MGPrecision::FLOAT },
        { "half", MGPrecision::HALF },
        { "fp16", MGPrecision::HALF },
    };
    add_option_with_default( app, "--stokes-mg-precision", parameters.stokes_solver_parameters.mg_precision )
        ->transform( CLI::CheckedTransformer( mg_precision_map, CLI::ignore_case ) )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-schur-relaxation", parameters.stokes_solver_parameters.schur_relaxation )
        ->group( "Stokes Solver" )
        ->description( "Relaxation scale on the Schur preconditioner (HyTeG Uzawa relaxParamSchur analogue). "
                       "1.0 = unscaled; tune for strongly-varying viscosity." );
    add_option_with_default( app, "--stokes-penalty-epsilon", parameters.stokes_solver_parameters.penalty_epsilon )
        ->group( "Stokes Solver" )
        ->description( "Rigid-rotation null-space penalty strength of the viscous block (fs/fs only)." );
    add_flag_with_default(
        app, "--stokes-refresh-viscous-pc", parameters.stokes_solver_parameters.refresh_viscous_pc )
        ->group( "Stokes Solver" )
        ->description( "Refresh the A-block MG smoother (D^-1 + Chebyshev bounds) on every viscosity update "
                       "(needed for strongly-varying/evolving viscosity; on by default)." );
    add_flag_with_default(
        app, "--stokes-refresh-coarse-viscosity", parameters.stokes_solver_parameters.refresh_coarse_viscosity )
        ->group( "Stokes Solver" )
        ->description( "Also restrict the evolving viscosity to coarse MG levels and re-tune all smoothers "
                       "(non-GCA coarse-op refresh; for very high viscosity contrast)." );
    add_flag_with_default(
        app, "--stokes-float-krylov-basis", parameters.stokes_solver_parameters.float_krylov_basis )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-viscous-pc-num-vcycles", parameters.stokes_solver_parameters.viscous_pc_num_vcycles )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-viscous-pc-cheby-order", parameters.stokes_solver_parameters.viscous_pc_chebyshev_order )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app,
        "--stokes-viscous-pc-num-smoothing-steps-prepost",
        parameters.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app,
        "--stokes-viscous-pc-num-power-iterations",
        parameters.stokes_solver_parameters.viscous_pc_num_power_iterations )
        ->group( "Stokes Solver" );

    if ( parameters.devel_parameters.extended_parameters )
    {
        add_option_with_default( app, "--stokes-gca", parameters.stokes_solver_parameters.gca )
            ->group( "Stokes Solver" )
            ->description( "Galerkin coarse-grid approximation mode for the viscous-block multigrid "
                           "preconditioner. 0 = disabled (default; coarse operators rediscretised), "
                           "1 = full GCA (more robust at variable viscosity), "
                           "2 = adaptive GCA (memory-saving, slightly less robust)." );
        app.add_option(
               "--stokes-viscous-pc-agglom-factors",
               parameters.stokes_solver_parameters.viscous_pc_agglom_factors,
               "Per-descent agglomeration factors for the viscous MG preconditioner. "
               "Space-separated list of length num_mg_levels-1. Example: \"2 2 1 1\". "
               "Empty (default) = classical MG with all levels on MPI_COMM_WORLD." )
            ->group( "Stokes Solver" )
            ->expected( 0, -1 )
            ->default_val( parameters.stokes_solver_parameters.viscous_pc_agglom_factors );
    }

    /////////////////////
    /// Energy solver ///
    /////////////////////

    

    add_option_with_default( app, "--energy-solver", parameters.energy_solver_parameters.energy_solver )
        ->transform( CLI::CheckedTransformer( energy_solver_map, CLI::ignore_case ) )
        ->default_val( "ev" )
        ->group( "Energy Solver" )
        ->description( "'fct': Explicit FCT advection-diffusion (default). "
                       "'supg': Implicit SUPG advection-diffusion with FGMRES solver." );

    add_option_with_default( app, "--energy-krylov-restart", parameters.energy_solver_parameters.krylov_restart )
        ->group( "Energy Solver" );
    add_flag_with_default(
        app, "--energy-float-krylov-basis", parameters.energy_solver_parameters.float_krylov_basis )
        ->group( "Energy Solver" );
    add_option_with_default(
        app, "--energy-krylov-max-iterations", parameters.energy_solver_parameters.krylov_max_iterations )
        ->group( "Energy Solver" );
    add_option_with_default(
        app, "--energy-krylov-relative-tolerance", parameters.energy_solver_parameters.krylov_relative_tolerance )
        ->group( "Energy Solver" );
    add_option_with_default(
        app, "--energy-krylov-absolute-tolerance", parameters.energy_solver_parameters.krylov_absolute_tolerance )
        ->group( "Energy Solver" );

    add_option_with_default( app, "--ev-alpha-max", parameters.energy_solver_parameters.ev_alpha_max )
        ->group( "Energy Solver" );
    add_option_with_default( app, "--ev-alpha-E", parameters.energy_solver_parameters.ev_alpha_E )
        ->group( "Energy Solver" );
    add_option_with_default( app, "--ev-dump-nu-h", parameters.energy_solver_parameters.ev_dump_nu_h )
        ->group( "Energy Solver" );
    add_option_with_default( app, "--ev-implicit-nu-h", parameters.energy_solver_parameters.implicit_nu_h )
        ->group( "Energy Solver" )
        ->description( "Assemble the entropy-viscosity artificial diffusion implicitly (LHS, "
                       "ASPECT-style) instead of the explicit RHS lag." );

    //////////////////////
    /// Input / output ///
    //////////////////////

    add_option_with_default( app, "--outdir", parameters.io_parameters.outdir )->group( "I/O" );
    add_flag_with_default( app, "--outdir-overwrite", parameters.io_parameters.overwrite )->group( "I/O" );
    add_option_with_default( app, "--output-pressure", parameters.io_parameters.output_pressure )->group( "I/O" );

    // Checkpoint loading
    add_option_with_default( app, "--load-checkpoint", parameters.io_parameters.load_checkpoint )
        ->group( "I/O" )
        ->description( "Starting from checkpoint" );
    add_option_with_default( app, "--checkpoint-dir", parameters.io_parameters.checkpoint_dir )->group( "I/O" );
    add_option_with_default( app, "--checkpoint-step", parameters.io_parameters.checkpoint_step )->group( "I/O" );

    add_option_with_default( app, "--output-frequency", parameters.io_parameters.output_frequency )
        ->group( "I/O" )
        ->description( "Write XDMF and radial profile output every N timesteps. Default: 1 (every timestep)." );

    add_flag_with_default( app, "--no-xdmf", parameters.io_parameters.no_xdmf )
        ->group( "I/O" )
        ->description( "Disable XDMF output." );

    add_flag_with_default( app, "--no-radial-profiles", parameters.io_parameters.no_radial_profiles )
        ->group( "I/O" )
        ->description( "Disable radial profile output." );

    //////////////////////////
    /// Developer settings ///
    //////////////////////////

    add_flag_with_default( app, "--extended-parameters", parameters.devel_parameters.extended_parameters )
        ->group( "Developer settings" )
        ->description( "Show hidden parameters. Not needed for 'standard' use." );
    add_flag_with_default(
        app,
        "--print-descriptions,!--print-descriptions-off",
        parameters.devel_parameters.print_parameter_descriptions )
        ->group( "Developer settings" )
        ->description( "Print parameter descriptions in config file. Use --print-descriptions-off to disable" );
    if ( parameters.devel_parameters.extended_parameters )
    {
        add_flag_with_default( app, "--nondim-input", parameters.devel_parameters.nondimensional_input )
            ->group( "Developer settings" )
            ->description(
                "Skip internal nondimensionalisation of provided parameters and take values unchanged from parameter file. Expects fully nondimensionalised input then. Intended for benchmarking purposes." );
        add_option_with_default( app, "--Rayleigh-number", parameters.physics_parameters.rayleigh_number )
            ->group( "Developer settings" )
            ->description(
                "Set Rayleigh number directly (would normally be computed from input params). Only works in combination with --nondim-input." );
        add_flag_with_default( app, "--output-dimensional", parameters.devel_parameters.output_dimensional )
            ->group( "Developer settings" )
            ->description( "Redimensionalise all output before writing." );
        add_flag_with_default( app, "--extended-diagnostics", parameters.devel_parameters.extended_diagnostics )
            ->group( "Developer settings" )
            ->description( "Log detailed diagnostics on solver setup, memory footprint, etc." );
        add_option_with_default(
            app, "--negative-temperature-tolerance", parameters.devel_parameters.negative_temperature_tolerance )
            ->group( "Developer settings" )
            ->description(
                "Abort once the minimum temperature falls below minus this value. MMOC is not monotone, so a "
                "small undershoot is expected; 0 restores the strict any-negative-value test." );
    }

    try
    {
        app.parse( argc, argv );
    }
    catch ( const CLI::ParseError& e )
    {
        app.exit( e );
        if ( e.get_exit_code() == static_cast< int >( CLI::ExitCodes::Success ) )
        {
            return { CLIHelp{} };
        }
        return { "CLI parse error" };
    }

    // Non-breaking guard against dead/misspelled config-file keys. CLI11 silently ignores
    // config entries that match no registered option (allow_config_extras is on), which has
    // masked real settings before (e.g. rayleigh-number / radius-min were unbound and had no
    // effect). Re-scan the config file and warn -- loudly but without aborting -- for every
    // key that binds to no option.
    if ( app.get_config_ptr() != nullptr && app.get_config_ptr()->count() > 0 )
    {
        std::set< std::string > known;
        for ( const auto* opt : app.get_options() )
        {
            for ( const auto& ln : opt->get_lnames() )
            {
                known.insert( ln );
            }
        }
        std::ifstream cfg( app.get_config_ptr()->as< std::string >() );
        std::string   line;
        while ( std::getline( cfg, line ) )
        {
            const auto first = line.find_first_not_of( " \t" );
            if ( first == std::string::npos || line[first] == '#' || line[first] == '[' )
            {
                continue; // blank, comment, or section header
            }
            const auto eq = line.find( '=', first );
            if ( eq == std::string::npos )
            {
                continue;
            }
            std::string key  = line.substr( first, eq - first );
            const auto  last = key.find_last_not_of( " \t" );
            if ( last != std::string::npos )
            {
                key.erase( last + 1 );
            }
            if ( !key.empty() && known.find( key ) == known.end() )
            {
                util::logroot << "[WARN] config key ignored (no matching option): " << key << std::endl;
            }
        }
    }

    // Cross-flag validation for anisotropic refinement.  The radial diamond
    // level at MG level L is (L + radial_extra_levels); we need that level to
    // be non-negative (otherwise (1 << rad_level) is UB) and to be at least the
    // radial subdomain refinement level so each subdomain holds >= 1 cell.
    {
        const auto& mp            = parameters.mesh_parameters;
        const int   mesh_min      = mp.refinement_level_mesh_min;
        const int   extra         = mp.radial_extra_levels;
        const int   lat_sdr_eff   = ( mp.lat_sdr >= 0 ) ? mp.lat_sdr : mp.refinement_level_subdomains;
        const int   rad_sdr_eff   = ( mp.rad_sdr >= 0 ) ? mp.rad_sdr : mp.refinement_level_subdomains;
        const int   rad_level_min = mesh_min + extra;

        if ( rad_level_min < 0 )
        {
            return {
                "Invalid refinement: refinement_level_mesh_min (" + std::to_string( mesh_min ) +
                ") + radial_extra_levels (" + std::to_string( extra ) + ") = " + std::to_string( rad_level_min ) +
                " is negative.  Radial mesh refinement level must be >= 0 at the coarsest "
                "MG level." };
        }
        if ( mesh_min < lat_sdr_eff )
        {
            return {
                "Invalid refinement: refinement_level_mesh_min (" + std::to_string( mesh_min ) +
                ") is less than the effective lateral subdomain refinement level (" + std::to_string( lat_sdr_eff ) +
                ").  Each lateral subdomain needs at least one cell at the coarsest MG level." };
        }
        if ( rad_level_min < rad_sdr_eff )
        {
            return {
                "Invalid refinement: refinement_level_mesh_min + radial_extra_levels (" +
                std::to_string( rad_level_min ) + ") is less than the effective radial subdomain refinement level (" +
                std::to_string( rad_sdr_eff ) +
                ").  Each radial subdomain needs at least one cell at the coarsest MG level. "
                "Consider lowering --rad-sdr or raising --radial-extra-levels." };
        }
    }

    // Nondimensionalise all relevant input parameters
    nondimensionalise( parameters );

    // Determine timestep_initial from checkpointing parameters
    if ( parameters.io_parameters.load_checkpoint )
    {
        parameters.time_stepping_parameters.timestep_initial = parameters.io_parameters.checkpoint_step;
    }

    // Plate velocities are imposed as a Dirichlet surface condition, so free slip cannot hold there.
    if ( parameters.boundary_parameters.plate_parameters.apply_plate_velocities &&
         parameters.boundary_parameters.velocity_bc_surface != BoundaryConditionsParameters::VelocityBC::NO_SLIP )
    {
        util::logroot << "\n## Plate velocities require NO-SLIP boundary at the surface. Setting accordingly..."
                      << std::endl;
        parameters.boundary_parameters.velocity_bc_surface = BoundaryConditionsParameters::VelocityBC::NO_SLIP;
    }

    util::logroot << "=========================================\n";
    util::logroot << "     Starting mantle circulation app     \n";
    util::logroot << "     Run with -h or --help for help      \n";
    util::logroot << "=========================================\n";

    util::print_general_info( argc, argv, util::logroot );
    util::print_cli_summary( app, util::logroot );

    // Matching nondimensional input with nondimensional output
    if ( parameters.devel_parameters.nondimensional_input )
    {
        parameters.devel_parameters.output_dimensional = false;

        util::logroot << "\n#############################################\n";
        util::logroot << "Nondimensional input chosen. Make sure all input parameters are set accordingly:\n";
        util::logroot << "--> T_surface, T_cmb, viscosity, Ra, internal_heating_rate, t_end, dt_max, dt_min.\n";
        util::logroot << "Output set to nondimensional.\n";
        util::logroot << "#############################################" << std::endl;
    }

    // Setting parameters for low-memory mode
    if ( parameters.stokes_solver_parameters.low_mem )
    {
        parameters.stokes_solver_parameters.float_krylov_basis = true;
        parameters.energy_solver_parameters.float_krylov_basis = true;
        parameters.stokes_solver_parameters.krylov_restart =
            std::min< int >( parameters.stokes_solver_parameters.krylov_restart, 5 );
        parameters.energy_solver_parameters.krylov_restart =
            std::min< int >( parameters.energy_solver_parameters.krylov_restart, 5 );
        parameters.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost =
            std::min< int >( parameters.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost, 1 );

        util::logroot
            << "\nWARNING: Low-memory mode set. Limiting --stokes-krylov-restart and --energy-krylov-restart to a maximum of 5 and --stokes-viscous-pc-num-smoothing-steps-prepost to 1."
            << std::endl;
        util::logroot << "Krylov basis functions are single precision." << std::endl;
    }
    util::logroot << std::endl;

    if ( !parameters.output_config_file.empty() )
    {
        util::logroot << "Writing config file to " << parameters.output_config_file << " and exiting." << std::endl;
        std::ofstream config_file( parameters.output_config_file );
        config_file << app.config_to_str( true, parameters.devel_parameters.print_parameter_descriptions );
    }

    return { parameters };
}

}; // namespace terra::mantlecirculation
