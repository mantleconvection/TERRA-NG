#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "communication/shell/communication.hpp"
#include "communication/shell/fv_communication.hpp"
#include "communication/shell/redistribute.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/strong_algebraic_freeslip_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg_kerngen.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "fv/hex/conversion.hpp"
#include "fv/hex/helpers.hpp"
#include "fv/hex/operators/fct_advection_diffusion.hpp"
#include "fv/hex/conversion.hpp"
#include "geophysics/viscosity/viscosity_interpolation.hpp"
#include "shell/spherical_harmonics.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "io/xdmf.hpp"
#include "kernels/common/grid_operations.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/diagonally_scaled_operator.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/diagonal_solver.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "mpi/mpi.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/power_iteration.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "src/diagnostics.hpp"
#include "src/build_radii.hpp"
#include "src/energy_solver.hpp"
#include "src/interpolators.hpp"
#include "src/io.hpp"
#include "src/parameters.hpp"
#include "src/stokes_solver.hpp"
#include "src/temperature_init.hpp"
#include "util/bit_masking.hpp"
#include "util/filesystem.hpp"
#include "util/logging.hpp"
#include "util/result.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

using ScalarType = double;

namespace terra::mantlecirculation {

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using linalg::solvers::TwoGridGCA;
using util::logroot;
using util::Ok;
using util::Result;

using grid::shell::BoundaryConditions;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;

Result<> run( const Parameters& prm )
{
    auto table = std::make_shared< util::Table >();

    if ( const auto create_directories_result = create_directories( prm.io_parameters );
         create_directories_result.is_err() )
    {
        return create_directories_result.error();
    }

    // Set up domains and masks (node ownership and boundary) for all levels.
    //
    // What do the various level indices mean?
    //
    // The refinement levels from the parameter file determine the global number of micro-elements, regardless
    // of the number of subdomains. Then subdomain refinement is applied. In order to refine the domain into
    // subdomains, the global refinement level must be greater or equal to the subdomain refinement level
    // (since we cannot split micro elements).
    //
    // Since we store various things in std::vectors, the indexing therein always starts with 0.
    // That may not be equal to the coarsest refinement level. So the index in the std::vectors must be set to
    //
    //   idx = refinement_level - min_refinement_level
    //
    // Better not mix that up.

    std::vector< std::shared_ptr< DistributedDomain > >               domains;
    std::vector< Grid3DDataVec< ScalarType, 3 > >                     coords_shell;
    std::vector< Grid2DDataScalar< ScalarType > >                     coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        ownership_mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    const int lat_sdr = ( prm.mesh_parameters.lat_sdr >= 0 ) ? prm.mesh_parameters.lat_sdr
                                                             : prm.mesh_parameters.refinement_level_subdomains;
    const int rad_sdr = ( prm.mesh_parameters.rad_sdr >= 0 ) ? prm.mesh_parameters.rad_sdr
                                                             : prm.mesh_parameters.refinement_level_subdomains;

    // MG-level communicator + subdomain-to-rank ladder for the (optional) MG
    // preconditioner agglomeration.  Every DistributedDomain built below is
    // created on its level's sub-comm, so all downstream Stokes objects (eta,
    // A_c, smoothers, inverse_diagonals, coarse_grid_solver, tmp_mg_*)
    // automatically live on the correct communicator.  StokesContext consumes
    // the same `agglom` for its upper-comm meshes and Redistribute plans.
    MGAgglomeration agglom( prm );

    for ( int level = prm.mesh_parameters.refinement_level_mesh_min;
          level <= prm.mesh_parameters.refinement_level_mesh_max;
          level++ )
    {
        const int idx       = level - prm.mesh_parameters.refinement_level_mesh_min;
        const int lat_level = level;
        const int rad_level = level + prm.mesh_parameters.radial_extra_levels;

        domains.push_back(
            std::make_shared< DistributedDomain >(
                DistributedDomain::create_uniform_on_comm(
                    agglom.comm( idx ),
                    lat_level,
                    build_shell_radii< double >( prm.mesh_parameters, ( 1 << rad_level ) + 1 ),
                    lat_sdr,
                    rad_sdr,
                    agglom.subdomain_fn( idx ) ) ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( (*domains[idx]) ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( (*domains[idx]) ) );
        ownership_mask_data.push_back( grid::setup_node_ownership_mask_data( (*domains[idx]) ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( (*domains[idx]) ) );
    }

    const auto subdomain_distr = grid::shell::subdomain_distribution( (*domains.back()) );
    logroot << "Subdomain distribution (subdomains per MPI process): \n";
    logroot << " - total: " << subdomain_distr.total << "\n";
    logroot << " - min:   " << subdomain_distr.min << "\n";
    logroot << " - avg:   " << subdomain_distr.avg << "\n";
    logroot << " - max:   " << subdomain_distr.max << "\n\n";

    const int  num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    Grid2DDataScalar< int > subdomain_shell_idx = grid::shell::subdomain_shell_idx( (*domains[velocity_level]) );

    // Set up the prognostic Q1 temperature.
    VectorQ1Scalar< ScalarType > T(
        "T", (*domains[velocity_level]), ownership_mask_data[velocity_level] );

    // Finite-volume functions/vectors.

    // FV cell-centred temperature field (the FCT prognostic variable).
    linalg::VectorFVScalar< ScalarType > T_fct( "T_fct", (*domains[velocity_level]) );
    // Pre-computed cell centres (with ghost layers filled once and reused every step).
    linalg::VectorFVVec< ScalarType, 3 > fv_cell_centers( "fv_cell_centers", (*domains[velocity_level]) );
    fv::hex::initialize_cell_centers(
        fv_cell_centers, (*domains[velocity_level]), coords_shell[velocity_level], coords_radii[velocity_level] );

    // Counting DoFs.
    int world_size = mpi::num_processes();

    const auto num_dofs_fe_scalar =
        kernels::common::count_masked< long >( ownership_mask_data[num_levels - 1], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_velocity = 3 * num_dofs_fe_scalar;
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( ownership_mask_data[num_levels - 2], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_temperature = domains[velocity_level]->domain_info().num_global_micro_hex_cells();

    logroot << "Degrees of freedom in (T,u,p) = (" << num_dofs_temperature << ", " << num_dofs_velocity << ", "
            << num_dofs_pressure << ")" << std::endl;
    logroot << "Avg DoFs/process in (T,u,p)   = (" << num_dofs_temperature / world_size << ", "
            << num_dofs_velocity / world_size << ", " << num_dofs_pressure / world_size << ")" << std::endl;

    // Setting up Stokes velocity boundary conditions.
    //
    // Currently, we can choose either no-slip or free-slip.
    //
    // Plates will also be a Dirichlet BCs (to be implemented).

    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };

    if ( prm.boundary_conditions_parameters.velocity_bc_cmb == BoundaryConditionsParameters::VelocityBC::FREE_SLIP )
    {
        grid::shell::set_boundary_condition_flag( bcs, CMB, FREESLIP );
    }

    if ( prm.boundary_conditions_parameters.velocity_bc_surface == BoundaryConditionsParameters::VelocityBC::FREE_SLIP )
    {
        grid::shell::set_boundary_condition_flag( bcs, SURFACE, FREESLIP );
    }

    // ---- Stokes solver context: viscosity hierarchy, GCA, MG, Schur, FGMRES.
    StokesContext< ScalarType > stokes(
        domains,
        coords_shell,
        coords_radii,
        ownership_mask_data,
        boundary_mask_data,
        bcs,
        agglom,
        prm,
        table );

    auto& u = stokes.solution();


    /////////////////////
    /// ENERGY SOLVER ///
    /////////////////////

    logroot << "Setting up energy equation solver ..." << std::endl;

    // FCT Dirichlet BCs (also used by FCTSolver below for the FV step).
    const fv::hex::DirichletBCs< ScalarType > fct_bcs{
        .T_cmb         = static_cast< ScalarType >( prm.boundary_conditions_parameters.temperature_cmb ),
        .T_surface     = static_cast< ScalarType >( prm.boundary_conditions_parameters.temperature_surface ),
        .apply_cmb     = true,
        .apply_surface = true };

    initialize_temperature_fields(
        T, T_fct, fct_bcs,
        (*domains[velocity_level]), coords_shell[velocity_level], coords_radii[velocity_level],
        fv_cell_centers,
        ownership_mask_data[velocity_level], boundary_mask_data[velocity_level],
        prm );

    // If temperature-dependent viscosity is enabled, compute the initial viscosity from the initial T.
    if ( prm.physics_parameters.viscosity_parameters.law != ViscosityLaw::CONSTANT )
    {
        logroot << "Computing initial temperature-dependent viscosity ..." << std::endl;
        stokes.update_viscosity( T );
    }

    table->add_row( {
        { "tag", "setup" },
        { "dofs_velocity", num_dofs_velocity },
        { "dofs_temperature", num_dofs_temperature },
        { "dofs_pressure", num_dofs_pressure },
        { "level_velocity", prm.mesh_parameters.refinement_level_mesh_max },
        { "level_pressure", prm.mesh_parameters.refinement_level_mesh_max - 1 },
    } );

    table->print_pretty();
    table->clear();

    // Setting up XDMF output (serves for both checkpointing and visualization).

    io::XDMFOutput xdmf_output(
        prm.io_parameters.outdir + "/" + prm.io_parameters.xdmf_dir,
        (*domains[velocity_level]),
        coords_shell[velocity_level],
        coords_radii[velocity_level] );

    xdmf_output.add( T.grid_data() );
    xdmf_output.add( u.block_1().grid_data() );
    xdmf_output.add( stokes.eta_fine().grid_data() );

    // Reference conductive temperature profile (also used for the Nusselt number).
    VectorQ1Scalar< ScalarType > T_ref( "T_ref", (*domains[velocity_level]), ownership_mask_data[velocity_level] );
    compute_reference_conductive_profile(
        T_ref,
        (*domains[velocity_level]),
        coords_shell[velocity_level],
        coords_radii[velocity_level] );

    xdmf_output.add( T_ref.grid_data() );

    int timestep_initial = 0;

    const bool loading_checkpoint = !prm.io_parameters.checkpoint_dir.empty() && prm.io_parameters.checkpoint_step >= 0;

    if ( loading_checkpoint )
    {
        timestep_initial = load_temperature_checkpoint(
            u.block_1(), T, T_fct,
            (*domains[velocity_level]), coords_shell[velocity_level], coords_radii[velocity_level],
            prm );
        // Continue XDMF output sequence from the checkpoint file step.
        xdmf_output.set_write_counter( prm.io_parameters.checkpoint_step );
    }

    ScalarType simulated_time = ScalarType( 0 );

    // We need some global h. Let's, for simplicity (does not need to be too accurate) just choose the smallest h in
    // radial direction.
    const auto h = grid::shell::min_radial_h( domains[velocity_level]->domain_info().radii() );

    // --- Energy solver (polymorphic dispatch via EnergySolver) ---
    // Construct before the initial XDMF write so that EV's optional
    // nu_h_nodal_view() can be registered with the XDMF output.

    std::unique_ptr< EnergySolver< ScalarType > > energy;
    switch ( prm.time_stepping_parameters.energy_solver )
    {
        case EnergySolverType::SUPG:
            energy = std::make_unique< SUPGSolver< ScalarType > >(
                domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
                boundary_mask_data[velocity_level], ownership_mask_data[velocity_level],
                u.block_1(), T, h, prm, table );
            break;
        case EnergySolverType::ENTROPY_VISCOSITY:
            energy = std::make_unique< EVSolver< ScalarType > >(
                domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
                boundary_mask_data[velocity_level], ownership_mask_data[velocity_level],
                u.block_1(), T, h, prm, table );
            break;
        case EnergySolverType::FCT:
            energy = std::make_unique< FCTSolver< ScalarType > >(
                domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
                boundary_mask_data[velocity_level], ownership_mask_data[velocity_level],
                u.block_1(), T, T_fct, fv_cell_centers, fct_bcs,
                prm, table );
            break;
    }

    // EV-specific: register the Q1-projected per-wedge ν_h diagnostic field
    // with XDMF if the energy solver exposes one.  Must happen before any
    // xdmf_output.write() call.
    if ( auto* nu_h_view = energy->nu_h_nodal_view() )
    {
        xdmf_output.add( nu_h_view->grid_data() );
    }

    if ( !prm.io_parameters.no_xdmf )
    {
        logroot << "Writing initial XDMF ..." << std::endl;
        xdmf_output.write();
    }

    if ( !prm.io_parameters.no_radial_profiles )
    {
        logroot << "Writing initial radial profiles ..." << std::endl;
        compute_and_write_radial_profiles(
            T, subdomain_shell_idx, (*domains[velocity_level]), prm.io_parameters, timestep_initial );
        compute_and_write_radial_profiles(
            stokes.eta_fine(), subdomain_shell_idx, (*domains[velocity_level]), prm.io_parameters, timestep_initial );
        compute_and_write_velocity_radial_profiles(
            u.block_1(),
            coords_shell[velocity_level],
            subdomain_shell_idx,
            (*domains[velocity_level]),
            ownership_mask_data[velocity_level],
            prm.io_parameters,
            timestep_initial );

        // EV-specific diagnostic profiles: per-wedge h_w (geometry-only,
        // available from construction).  lap_T_ is not meaningful before any
        // time step has run, so skip the lap profile here.
        if ( auto* hw_view = energy->h_w_diag_view() )
        {
            compute_and_write_radial_profiles(
                *hw_view, subdomain_shell_idx, (*domains[velocity_level]), prm.io_parameters, timestep_initial );
        }
    }


    // Time stepping

    logroot << "Starting time stepping!" << std::endl;

    // Compute Nusselt at timestep 0 (before any FCT steps) for diagnostics.
    {
        const auto Nu_top_0 = compute_nusselt(
            (*domains[velocity_level]), T, T_ref, coords_shell[velocity_level], coords_radii[velocity_level], boundary_mask_data[velocity_level], ownership_mask_data[velocity_level], true );
        const auto Nu_top_fv_0 = compute_nusselt_fv(
            (*domains[velocity_level]), T_fct,
            boundary_mask_data[velocity_level],
            prm.boundary_conditions_parameters.temperature_surface,
            prm.boundary_conditions_parameters.temperature_cmb,
            prm.mesh_parameters.radius_min, prm.mesh_parameters.radius_max, true );
        logroot << "Nu_top (Q1) = " << Nu_top_0 << ", Nu_top (FV) = " << Nu_top_fv_0
                << "  [timestep 0, before time stepping]" << std::endl;
    }

    for ( int timestep = timestep_initial + 1; timestep < prm.time_stepping_parameters.max_timesteps; timestep++ )
    {
        logroot << "\n### Timestep " << timestep << " ###" << std::endl;
        util::Timer timer_timestep( "timestep" );


        const int num_picard = prm.time_stepping_parameters.picard_iterations;

        // Snapshot any solver-internal state that needs restoring across Picard iterations.
        energy->snapshot_for_picard();

        // Compute dt once from current velocity (before Picard loop).
        const ScalarType dt = energy->compute_dt();

        for ( int picard = 0; picard < num_picard; picard++ )
        {
            if ( num_picard > 1 )
                logroot << "--- Picard iteration " << picard << " / " << num_picard << " ---" << std::endl;

            // Restore solver state to start-of-timestep so each Picard iteration redoes
            // the energy update from the same starting point.
            if ( picard > 0 )
            {
                energy->restore_for_picard();
            }

            // --- Stokes solve ---
            stokes.solve( T, /*log_convergence=*/( picard == num_picard - 1 ) );

            // --- Energy solve (polymorphic dispatch) ---
            energy->step( dt, /*print_convergence=*/( picard == num_picard - 1 ) );

            // Update viscosity from the new temperature field.
            stokes.update_viscosity( T );

        } // end Picard loop

        // Output stuff, logging etc.

        table->add_row( {} );

        const bool write_output = ( timestep % prm.io_parameters.output_frequency == 0 );

        if ( write_output && !prm.io_parameters.no_xdmf )
        {
            logroot << "Writing XDMF output ..." << std::endl;
            xdmf_output.write();
        }

        // Energy-solver-specific diagnostics dump first — refreshes EV
        // diagnostic views (lap_diag_) so the radial-profile pass below sees
        // up-to-date data.
        if ( write_output )
        {
            energy->dump_diagnostics( timestep, prm.io_parameters.outdir );
        }

        if ( write_output && !prm.io_parameters.no_radial_profiles )
        {
            logroot << "Writing radial profiles ..." << std::endl;
            compute_and_write_radial_profiles(
                T, subdomain_shell_idx, (*domains[velocity_level]), prm.io_parameters, timestep );
            compute_and_write_radial_profiles(
                stokes.eta_fine(), subdomain_shell_idx, (*domains[velocity_level]), prm.io_parameters, timestep );
            compute_and_write_velocity_radial_profiles(
                u.block_1(),
                coords_shell[velocity_level],
                subdomain_shell_idx,
                (*domains[velocity_level]),
                ownership_mask_data[velocity_level],
                prm.io_parameters,
                timestep );

            // EV-specific diagnostic profiles (refreshed by dump_diagnostics).
            if ( auto* lap_view = energy->lap_diag_view() )
            {
                compute_and_write_radial_profiles(
                    *lap_view, subdomain_shell_idx, (*domains[velocity_level]), prm.io_parameters, timestep );
            }
            if ( auto* hw_view = energy->h_w_diag_view() )
            {
                compute_and_write_radial_profiles(
                    *hw_view, subdomain_shell_idx, (*domains[velocity_level]), prm.io_parameters, timestep );
            }
        }

        // Compute Nusselt number at the surface every step; log to stdout
        // every 10 steps; append to <outdir>/nu.csv every step (rank 0).
        {
            const auto Nu_top = compute_nusselt(
                (*domains[velocity_level]),
                T,
                T_ref,
                coords_shell[velocity_level],
                coords_radii[velocity_level],
                boundary_mask_data[velocity_level],
                ownership_mask_data[velocity_level],
                /*at_surface=*/true );
            const auto Nu_top_fv = compute_nusselt_fv(
                (*domains[velocity_level]),
                T_fct,
                boundary_mask_data[velocity_level],
                prm.boundary_conditions_parameters.temperature_surface,
                prm.boundary_conditions_parameters.temperature_cmb,
                prm.mesh_parameters.radius_min,
                prm.mesh_parameters.radius_max,
                /*at_surface=*/true );
            if ( timestep % 10 == 0 )
            {
                logroot << "Nu_top (Q1) = " << Nu_top << ", Nu_top (FV) = " << Nu_top_fv << std::endl;
            }
            // Per-step CSV. simulated_time is updated below; the value here is
            // the time at the *end* of this step (current T just solved).
            if ( mpi::rank() == 0 )
            {
                const std::string path = prm.io_parameters.outdir + "/nu.csv";
                std::ofstream out( path, std::ios::app );
                if ( out.tellp() == 0 )
                {
                    out << "timestep,sim_time,Nu_top_Q1,Nu_top_FV\n";
                }
                const double t_end_of_step = simulated_time + prm.time_stepping_parameters.energy_substeps * dt;
                out << timestep << "," << t_end_of_step << "," << Nu_top << "," << Nu_top_fv << "\n";
            }
        }

        simulated_time += prm.time_stepping_parameters.energy_substeps * dt;

        logroot << "Simulated time: " << simulated_time << " (stopping at " << prm.time_stepping_parameters.t_end
                << ", we're at " << simulated_time / prm.time_stepping_parameters.t_end * 100.0 << "%)" << std::endl;
        timer_timestep.stop();

        write_timer_tree( prm.io_parameters, timestep );

        if ( simulated_time >= prm.time_stepping_parameters.t_end )
        {
            break;
        }

        if ( has_nan_or_inf( T ) )
        {
            logroot << "\nDETECTED NAN OR INF.\n\n"
                       "For some reason the temperature vector contains NaN or inf values.\n"
                       "Those might come from anywhere (not necessarily the energy solve).\n"
                       "To avoid burning compute time, the simulation will exit now.\n\n"
                       "You may be able to recover the simulation from an earlier checkpoint.\n\n"
                       "Good luck and bye."
                    << std::endl;
            break;
        }
    }

    return { Ok{} };
}
} // namespace terra::mantlecirculation

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const auto parameters = mantlecirculation::parse_parameters( argc, argv );

    if ( parameters.is_err() )
    {
        logroot << parameters.error() << std::endl;
        return EXIT_FAILURE;
    }

    if ( std::holds_alternative< mantlecirculation::CLIHelp >( parameters.unwrap() ) )
    {
        return EXIT_SUCCESS;
    }

    const auto actual_parameters = std::get< mantlecirculation::Parameters >( parameters.unwrap() );

    if ( !actual_parameters.output_config_file.empty() )
    {
        return EXIT_SUCCESS;
    }

    if ( auto run_result = run( actual_parameters ); run_result.is_err() )
    {
        logroot << run_result.error() << std::endl;
        return EXIT_FAILURE;
    }
}
