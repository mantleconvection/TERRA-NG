#pragma once

#include <string>
#include <variant>

#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/result.hpp"

namespace terra::mantlecirculation {

struct MeshParameters
{
    int refinement_level_mesh_min   = 1;
    int refinement_level_mesh_max   = 4;
    int refinement_level_subdomains = 0;

    double radius_min = 0.5;
    double radius_max = 1.0;
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
};

struct ViscosityParameters
{
    bool        radial_profile_enabled       = false;
    std::string radial_profile_csv_filename  = "radial_viscosity_profile.csv";
    std::string radial_profile_radii_key     = "radii";
    std::string radial_profile_viscosity_key = "viscosity";
    double      reference_viscosity          = 1.0;
};

struct PhysicsParameters
{
    double diffusivity     = 1.0;
    double rayleigh_number = 1e5;

    ViscosityParameters viscosity_parameters{};
};

struct StokesSolverParameters
{
    int    krylov_restart            = 10;
    int    krylov_max_iterations     = 10;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;

    int viscous_pc_num_vcycles                 = 1;
    int viscous_pc_chebyshev_order             = 2;
    int viscous_pc_num_smoothing_steps_prepost = 2;
    int viscous_pc_num_power_iterations        = 10;
};

struct EnergySolverParameters
{
    int    krylov_restart            = 5;
    int    krylov_max_iterations     = 100;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;
};

struct TimeSteppingParameters
{
    double pseudo_cfl = 0.5;
    double t_end      = 1.0;

    int max_timesteps = 10;

    int energy_substeps = 1;
};

struct IOParameters
{
    std::string outdir    = "output";
    bool        overwrite = false;

    std::string xdmf_dir                = "xdmf";
    std::string radial_profiles_out_dir = "radial_profiles";
    std::string timer_trees_dir         = "timer_trees";

    std::string checkpoint_dir;
    int         checkpoint_step = -1;

    int output_frequency = 1;

    bool no_xdmf = false;
    bool no_radial_profiles = false;
};

struct Parameters
{
    MeshParameters               mesh_parameters;
    BoundaryConditionsParameters boundary_conditions_parameters;
    StokesSolverParameters       stokes_solver_parameters;
    EnergySolverParameters       energy_solver_parameters;
    PhysicsParameters            physics_parameters;
    TimeSteppingParameters       time_stepping_parameters;
    IOParameters                 io_parameters;

    std::string output_config_file;
};

struct CLIHelp
{};

inline util::Result< std::variant< CLIHelp, Parameters > > parse_parameters( int argc, char** argv )
{
    CLI::App app{ "Mantle circulation simulation." };

    Parameters parameters{};

    using util::add_flag_with_default;
    using util::add_option_with_default;

    // Allow config files
    app.set_config( "--config" );

    ///////////////
    /// General ///
    ///////////////

    add_option_with_default(
        app,
        "--write-config-and-exit",
        parameters.output_config_file,
        "Writes a config file with the passed (or default arguments) to the desired location to be then modified and passed. E.g., '--write-config-and-exit my-config.toml'.\n"
        "IMPORTANT: THIS OPTION MUST BE REMOVED IN THE GENERATED CONFIG OR ELSE YOU WILL OVERWRITE IT AGAIN" )
        ->group( "General" );

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

    add_option_with_default( app, "--radius-min", parameters.mesh_parameters.radius_min )->group( "Domain" );
    add_option_with_default( app, "--radius-max", parameters.mesh_parameters.radius_max )->group( "Domain" );

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

    add_option_with_default( app, "--velocity-bc-cmb", parameters.boundary_conditions_parameters.velocity_bc_cmb )
        ->transform( CLI::CheckedTransformer( velocity_bc_cmb_map, CLI::ignore_case ) )
        ->default_val( "noslip" )
        ->group( "Boundary Conditions" );

    add_option_with_default(
        app, "--velocity-bc-surface", parameters.boundary_conditions_parameters.velocity_bc_surface )
        ->transform( CLI::CheckedTransformer( velocity_bc_surface_map, CLI::ignore_case ) )
        ->default_val( "noslip" )
        ->group( "Boundary Conditions" );

    //////////////////////////////
    /// Geophysical parameters ///
    //////////////////////////////

    add_option_with_default( app, "--diffusivity", parameters.physics_parameters.diffusivity );
    add_option_with_default( app, "--rayleigh-number", parameters.physics_parameters.rayleigh_number );

    const auto radial_profile_enabled =
        add_flag_with_default(
            app,
            "--viscosity-radial-profile",
            parameters.physics_parameters.viscosity_parameters.radial_profile_enabled )
            ->group( "Viscosity" )
            ->description(
                "Add this flag if you want to supply a radial viscosity profile. "
                "Then use further flags/arguments (starting with --viscosity-radial-profile-<...>) to specify the file path etc. "
                "If you omit this flag, the viscosity is set to const (eta = 1)." );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-csv-filename",
        parameters.physics_parameters.viscosity_parameters.radial_profile_csv_filename )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-radii-key",
        parameters.physics_parameters.viscosity_parameters.radial_profile_radii_key )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-value-key",
        parameters.physics_parameters.viscosity_parameters.radial_profile_viscosity_key )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app, "--viscosity-reference-value", parameters.physics_parameters.viscosity_parameters.reference_viscosity )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );

    ///////////////////////////
    /// Time discretization ///
    ///////////////////////////

    add_option_with_default( app, "--pseudo-cfl", parameters.time_stepping_parameters.pseudo_cfl )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--t-end", parameters.time_stepping_parameters.t_end )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--max-timesteps", parameters.time_stepping_parameters.max_timesteps )
        ->group( "Time Discretization" )
        ->description(
            "Simulation aborts when this time step index is reached. "
            "If a checkpoint is loaded, the simulation will start at the next step after the loaded checkpoint. "
            "This means the number of time steps executed might be smaller than what is passed in here." );
    add_option_with_default( app, "--energy-substeps", parameters.time_stepping_parameters.energy_substeps )
        ->group( "Time Discretization" );

    /////////////////////
    /// Stokes solver ///
    /////////////////////

    add_option_with_default( app, "--stokes-krylov-restart", parameters.stokes_solver_parameters.krylov_restart )
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

    /////////////////////
    /// Energy solver ///
    /////////////////////

    add_option_with_default( app, "--energy-krylov-restart", parameters.energy_solver_parameters.krylov_restart )
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

    //////////////////////
    /// Input / output ///
    //////////////////////

    add_option_with_default( app, "--outdir", parameters.io_parameters.outdir )->group( "I/O" );
    add_flag_with_default( app, "--outdir-overwrite", parameters.io_parameters.overwrite )->group( "I/O" );

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

    util::print_general_info( argc, argv, util::logroot );
    util::print_cli_summary( app, util::logroot );
    util::logroot << std::endl;

    if ( !parameters.output_config_file.empty() )
    {
        util::logroot << "Writing config file to " << parameters.output_config_file << " and exiting." << std::endl;
        std::ofstream config_file( parameters.output_config_file );
        config_file << app.config_to_str( true, true );
    }

    return { parameters };
}

}; // namespace terra::mantlecirculation