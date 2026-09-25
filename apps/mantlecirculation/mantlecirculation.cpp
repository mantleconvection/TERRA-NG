#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "communication/shell/communication.hpp"
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
#include "geophysics/viscosity/viscosity_interpolation.hpp"
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
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/power_iteration.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "mpi/mpi.hpp"
#include "shell/spherical_harmonics.hpp"
#include "src/build_radii.hpp"
#include "src/diagnostics.hpp"
#include "src/hbm_probe.hpp"
#include "src/energy_solver.hpp"
#include "src/hbm_probe.hpp"
#include "src/interpolators.hpp"
#include "src/io.hpp"
#include "src/parameters.hpp"
#include "src/plates.hpp"
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
using terra::kernels::common::scale;
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

    const int lat_sdr = ( prm.mesh_parameters.lat_sdr >= 0 ) ? prm.mesh_parameters.lat_sdr :
                                                               prm.mesh_parameters.refinement_level_subdomains;
    const int rad_sdr = ( prm.mesh_parameters.rad_sdr >= 0 ) ? prm.mesh_parameters.rad_sdr :
                                                               prm.mesh_parameters.refinement_level_subdomains;

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

        domains.push_back( std::make_shared< DistributedDomain >( DistributedDomain::create_uniform_on_comm(
            agglom.comm( idx ),
            lat_level,
            build_shell_radii< double >( prm.mesh_parameters, ( 1 << rad_level ) + 1 ),
            lat_sdr,
            rad_sdr,
            agglom.subdomain_fn( idx ) ) ) );
        coords_shell.push_back(
            grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( ( *domains[idx] ) ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( ( *domains[idx] ) ) );
        ownership_mask_data.push_back( grid::setup_node_ownership_mask_data( ( *domains[idx] ) ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( ( *domains[idx] ) ) );
    }

    const auto subdomain_distr = grid::shell::subdomain_distribution( ( *domains.back() ) );
    logroot << "Subdomain distribution (subdomains per MPI process): \n";
    logroot << " - total: " << subdomain_distr.total << "\n";
    logroot << " - min:   " << subdomain_distr.min << "\n";
    logroot << " - avg:   " << subdomain_distr.avg << "\n";
    logroot << " - max:   " << subdomain_distr.max << "\n\n";

    const int  num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    Grid2DDataScalar< int > subdomain_shell_idx = grid::shell::subdomain_shell_idx( ( *domains[velocity_level] ) );

    // Set up the prognostic Q1 temperature.
    VectorQ1Scalar< ScalarType > T( "T", ( *domains[velocity_level] ), ownership_mask_data[velocity_level] );
    VectorQ1Scalar< ScalarType > Tdev( "Tdev", ( *domains[velocity_level] ), ownership_mask_data[velocity_level] );

    // True when compressibility is on and any PDA form is used.
    const bool pda_form =
        ( prm.physics_parameters.compressible &&
          ( prm.physics_parameters.compressible_form == CompressibleForm::PDA ||
            prm.physics_parameters.compressible_form == CompressibleForm::PDA_ENTROPY ) );

    // Optional 3-D density field for PDA
    // Density is needed in Stokes and energy -- so we set it up here
    std::optional< VectorQ1Scalar< ScalarType > > density;
    // The 3-D density is also what the compressible (TALA) heating terms read,
    // so it is needed whenever compressibility is on, not only for the PDA form.
    // The EV and MMOC solvers bind the field at construction (their heating terms
    // ignore it when running incompressible), so it must exist for them as well.
    const bool energy_binds_density = prm.energy_solver_parameters.energy_solver == EnergySolverType::ENTROPY_VISCOSITY ||
                                      prm.energy_solver_parameters.energy_solver == EnergySolverType::MMOC;
    if ( pda_form || prm.physics_parameters.compressible || energy_binds_density )
    {
        density.emplace( "density", ( *domains[velocity_level] ), ownership_mask_data[velocity_level] );
    }

    // Radial parameter profiles
    Grid2DDataScalar< ScalarType > T_ref(
        "T_ref", coords_radii[velocity_level].extent( 0 ), coords_radii[velocity_level].extent( 1 ) );
    Grid2DDataScalar< ScalarType > rho_profile(
        "rho_profile", coords_radii[velocity_level].extent( 0 ), coords_radii[velocity_level].extent( 1 ) );
    Grid2DDataScalar< ScalarType > alpha_profile(
        "alpha_profile", coords_radii[velocity_level].extent( 0 ), coords_radii[velocity_level].extent( 1 ) );
    Grid2DDataScalar< ScalarType > cp_profile(
        "cp_profile", coords_radii[velocity_level].extent( 0 ), coords_radii[velocity_level].extent( 1 ) );
    Grid2DDataScalar< ScalarType > kappa_profile(
        "kappa_profile", coords_radii[velocity_level].extent( 0 ), coords_radii[velocity_level].extent( 1 ) );

    // Finite-volume functions/vectors.

    // FV cell-centred temperature field (the FCT prognostic variable).
    linalg::VectorFVScalar< ScalarType > T_fct( "T_fct", ( *domains[velocity_level] ) );
    // Pre-computed cell centres (with ghost layers filled once and reused every step).
    linalg::VectorFVVec< ScalarType, 3 > fv_cell_centers( "fv_cell_centers", ( *domains[velocity_level] ) );
    fv::hex::initialize_cell_centers(
        fv_cell_centers, ( *domains[velocity_level] ), coords_shell[velocity_level], coords_radii[velocity_level] );

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

    // Logging nondimensional numbers
    logroot << "\n----------Simulation parameters-----------" << std::endl;
    logroot << "Rayleigh number: " << prm.physics_parameters.rayleigh_number << std::endl;
    logroot << "Peclet number: " << prm.physics_parameters.peclet_number << std::endl;
    logroot << "Reference viscosity: " << prm.physics_parameters.viscosity_parameters.reference_viscosity << std::endl;
    logroot << "Viscosity clamp (x reference): [" << prm.physics_parameters.viscosity_parameters.min_viscosity << ", "
            << prm.physics_parameters.viscosity_parameters.max_viscosity << "]" << std::endl;
    logroot << "Thermal diffusivity: " << prm.physics_parameters.thermal_diffusivity_dim << std::endl;
    if ( !prm.devel_parameters.nondimensional_input )
        logroot << "Characteristic velocity: " << prm.physics_parameters.characteristic_velocity << std::endl;
    logroot << "------------------------------------------\n" << std::endl;

    // Fill radial profile arrays
    radial_profile_init(
        rho_profile,
        alpha_profile,
        cp_profile,
        kappa_profile,
        *domains[velocity_level],
        coords_radii[velocity_level],
        prm );

    // Initialise density Q1 field from radial profile -- before Stokes solver setup.
    // Also needed for the compressible (TALA) heating terms, which read a 3-D density.
    if ( density )
    {
        Kokkos::parallel_for(
            "RadialProfileToQ1",
            grid::shell::local_domain_md_range_policy_nodes( *domains[velocity_level] ),
            RadialProfileToQ1{ density->grid_data(), rho_profile } );
        Kokkos::fence();
    }

    // Setting up Stokes velocity boundary conditions.
    //
    // Currently, we can choose either no-slip or free-slip.
    //
    // Plate velocities are also a Dirichlet BC, imposed at the surface.

    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };

    if ( prm.boundary_parameters.velocity_bc_cmb == BoundaryConditionsParameters::VelocityBC::FREE_SLIP )
    {
        grid::shell::set_boundary_condition_flag( bcs, CMB, FREESLIP );
    }

    if ( prm.boundary_parameters.velocity_bc_surface == BoundaryConditionsParameters::VelocityBC::FREE_SLIP )
    {
        grid::shell::set_boundary_condition_flag( bcs, SURFACE, FREESLIP );
    }

    // ---- Stokes solver context: viscosity hierarchy, GCA, MG, Schur, FGMRES.
    if ( prm.devel_parameters.extended_diagnostics )
        log_hbm( "before StokesContext (domains + grids only)" );

    StokesContext< ScalarType > stokes(
        domains, coords_shell, coords_radii, ownership_mask_data, boundary_mask_data, bcs, agglom, prm, table );

    auto& u = stokes.solution();

    /////////////////////
    /// ENERGY SOLVER ///
    /////////////////////

    logroot << "Setting up energy equation solver ..." << std::endl;

    // FCT Dirichlet BCs (also used by FCTSolver below for the FV step).
    const fv::hex::DirichletBCs< ScalarType > fct_bcs{
        .T_cmb         = static_cast< ScalarType >( prm.boundary_parameters.temperature_max ),
        .T_surface     = static_cast< ScalarType >( prm.boundary_parameters.temperature_min ),
        .apply_cmb     = true,
        .apply_surface = true };

    initialize_temperature_fields(
        T,
        T_fct,
        T_ref,
        fct_bcs,
        ( *domains[velocity_level] ),
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        fv_cell_centers,
        ownership_mask_data[velocity_level],
        boundary_mask_data[velocity_level],
        prm );

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

    // Reference conductive temperature profile (also used for the Nusselt number).
    VectorQ1Scalar< ScalarType > T_cond( "T_cond", ( *domains[velocity_level] ), ownership_mask_data[velocity_level] );
    compute_reference_conductive_profile(
        T_cond, ( *domains[velocity_level] ), coords_shell[velocity_level], coords_radii[velocity_level], prm );

    // Setting up XDMF output (serves for both checkpointing and visualization).

    // Initialize two XDMFOutput insances - one at the finest level (velocity_level) and a second optional one if pressure (at pressure_level) is written out.
    // Both are created as std::optional objects so that access syntax remains uniform.
    std::optional< io::XDMFOutput< ScalarType > > xdmf_output;
    std::optional< io::XDMFOutput< ScalarType > > xdmf_output_pressure;

    const auto coords_scale_factor =
        prm.mesh_parameters.mantle_thickness_m /
        prm.mesh_parameters.radius_surface_m; // Used to rescale output coords to unit sphere

    xdmf_output.emplace(
        prm.io_parameters.outdir + "/" + prm.io_parameters.xdmf_dir,
        ( *domains[velocity_level] ),
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        coords_scale_factor );

    // Reference conductive temperature profile (also used for the Nusselt number).

    xdmf_output->add( T.grid_data() );                 // Temperature
    xdmf_output->add( Tdev.grid_data() );              // Temperature deviation
    xdmf_output->add( u.block_1().grid_data() );       // Velocity
    xdmf_output->add( stokes.eta_fine().grid_data() ); // Viscosity
    if ( density )
        xdmf_output->add( density->grid_data() ); // Density

    if ( prm.io_parameters.output_pressure )
    {
        xdmf_output_pressure.emplace(
            prm.io_parameters.outdir + "/" + prm.io_parameters.xdmf_dir + "_p",
            ( *domains[pressure_level] ),
            coords_shell[pressure_level],
            coords_radii[pressure_level],
            coords_scale_factor );

        xdmf_output_pressure->add( u.block_2().grid_data() ); // Pressure
    }

    // Helper function to gather all xdmf output fields for write_xdmf().
    // Combine with corresponding scaling factor for redimensionalisation and a flag specifying if the nondimensional field should be restored or not.
    auto collect_xdmf_fields = [&]() -> XdmfFields {
        XdmfFields fields;

        fields.scalar_fields = {
            { T.grid_data(), prm.boundary_parameters.delta_T_K, true },
            { Tdev.grid_data(), prm.boundary_parameters.delta_T_K, false },
            { stokes.eta_fine().grid_data(), prm.physics_parameters.viscosity_parameters.reference_viscosity, true },
        };

        fields.vector_fields = {
            { u.block_1().grid_data(), prm.physics_parameters.calc_cm_per_year, true },
        };

        if ( pda_form ) // Add density output
            fields.scalar_fields.push_back( { density->grid_data(), prm.physics_parameters.reference_density, true } );

        if ( prm.io_parameters.output_pressure )
            fields.pressure_field.emplace(
                u.block_2().grid_data(),
                prm.physics_parameters.viscosity_parameters.reference_viscosity *
                    prm.physics_parameters.characteristic_velocity / prm.mesh_parameters.mantle_thickness_m,
                true );

        return fields;
    };

    // Loading checkpoint
    if ( prm.io_parameters.load_checkpoint )
    {
        load_temperature_checkpoint(
            u.block_1(),
            T,
            T_fct,
            ( *domains[velocity_level] ),
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            prm );
    }

    // Update Tdev
    subtract_radial_profile( Tdev, T, T_ref, *domains[velocity_level] );

    // Compute temperature-dependent viscosity
    if ( prm.physics_parameters.viscosity_parameters.law != ViscosityLaw::CONSTANT )
    {
        logroot << "Computing initial temperature-dependent viscosity ..." << std::endl;
        if ( prm.physics_parameters.viscosity_parameters.law == ViscosityLaw::FK_TYPE3 )
        {
            stokes.update_viscosity( Tdev );
        }
        else
        {
            stokes.update_viscosity( T );
        }
    }

    // Update Tdev
    subtract_radial_profile( Tdev, T, T_ref, *domains[velocity_level] );

    // Setting XDMF file padding width according to max_timesteps.
    xdmf_output->set_pad_width(
        std::to_string( prm.time_stepping_parameters.timestep_initial + prm.time_stepping_parameters.max_timesteps - 1 )
            .size() );
    xdmf_output->set_is_dimensional( prm.devel_parameters.output_dimensional );

    if ( prm.io_parameters.output_pressure )
    {
        xdmf_output_pressure->set_pad_width(
            std::to_string(
                prm.time_stepping_parameters.timestep_initial + prm.time_stepping_parameters.max_timesteps - 1 )
                .size() );
        xdmf_output_pressure->set_is_dimensional( prm.devel_parameters.output_dimensional );
    }

    // ----- Plate velocity boundary condition -----
    // Model age for plate data; only meaningful in dimensional mode.
    ScalarType plate_age_Ma = static_cast< ScalarType >( prm.boundary_parameters.plate_parameters.initial_plate_age );
    int        last_plate_update_time = prm.boundary_parameters.plate_parameters.initial_plate_age;

    const ScalarType plate_velocity_nondim_scale = ScalarType( 1 ) / prm.physics_parameters.characteristic_velocity *
                                                   ( prm.boundary_parameters.plate_parameters.plate_velocity_scaling );

    std::shared_ptr< plates::PlateVelocityProvider > oracle;
    std::optional< VectorQ1IsoQ2Q1< ScalarType > >   plate_velocities;

    if ( prm.boundary_parameters.plate_parameters.apply_plate_velocities )
    {
        plate_velocities.emplace( "plate_velocities",
                                  *domains[velocity_level],
                                  *domains[pressure_level],
                                  ownership_mask_data[velocity_level],
                                  ownership_mask_data[pressure_level] );

        oracle = initialise_plates( prm.boundary_parameters.plate_parameters.plates_topologies_path,
                                    prm.boundary_parameters.plate_parameters.plates_reconstructions_path );

        extract_plate_velocities( plate_age_Ma,
                                  plate_velocities->block_1().grid_data(),
                                  *oracle,
                                  coords_shell[velocity_level],
                                  coords_radii[velocity_level],
                                  prm.boundary_parameters.plate_parameters.interpolate_plates_in_time,
                                  plate_velocity_nondim_scale,
                                  domains[velocity_level].get(),
                                  prm.boundary_parameters.plate_parameters.plates_on_device );
    }

    // ----- Initial Stokes solve -----
    logroot << "\n--------- Initial Stokes solve -----------------\n" << std::endl;

    // The PDA form passes the 3-D density unwrapped; TALA passes the radial profile.
    if ( pda_form )
        stokes.solve( Tdev,
                      plate_velocities,
                      density->grid_data(),
                      alpha_profile,
                      prm.physics_parameters.compressible,
                      /*log_convergence=*/true );
    else
        stokes.solve( Tdev,
                      plate_velocities,
                      rho_profile,
                      alpha_profile,
                      prm.physics_parameters.compressible,
                      /*log_convergence=*/true );

    ScalarType simulated_time    = ScalarType( 0 );
    ScalarType simulated_time_Ma = ScalarType( 0 );

    // Most negative temperature seen so far; only a new record is logged (see the guard below).
    ScalarType T_min_record = ScalarType( 0 );

    // We need some global h. Let's, for simplicity (does not need to be too accurate) just choose the smallest h in
    // radial direction.
    const auto h = grid::shell::min_radial_h( domains[velocity_level]->domain_info().radii() );

    // --- Energy solver (polymorphic dispatch via EnergySolver) ---
    // Construct before the initial XDMF write so that EV's optional
    // nu_h_nodal_view() can be registered with the XDMF output.

    std::unique_ptr< EnergySolver< ScalarType > > energy;
    switch ( prm.energy_solver_parameters.energy_solver )
    {
    case EnergySolverType::SUPG:
        energy = std::make_unique< SUPGSolver< ScalarType > >(
            domains[velocity_level],
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            boundary_mask_data[velocity_level],
            ownership_mask_data[velocity_level],
            u.block_1(),
            T,
            h,
            prm,
            table );
        break;
    case EnergySolverType::ENTROPY_VISCOSITY:
        energy = std::make_unique< EVSolver< ScalarType > >(
            domains[velocity_level],
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            boundary_mask_data[velocity_level],
            ownership_mask_data[velocity_level],
            u.block_1(),
            T,
            h,
            prm,
            table,
            // Compressible (TALA) heating inputs: fine-level viscosity and
            // reference density (aliased, updated in place). Ignored when
            // running incompressible.
            stokes.eta_fine().grid_data(),
            density->grid_data() );
        break;
    case EnergySolverType::MMOC:
        energy = std::make_unique< MMOCSolver< ScalarType > >(
            domains[velocity_level],
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            boundary_mask_data[velocity_level],
            ownership_mask_data[velocity_level],
            u.block_1(),
            T,
            h,
            prm,
            table,
            stokes.eta_fine().grid_data(),
            density->grid_data() );
        break;
    case EnergySolverType::FCT:
        energy = std::make_unique< FCTSolver< ScalarType > >(
            domains[velocity_level],
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            boundary_mask_data[velocity_level],
            ownership_mask_data[velocity_level],
            u.block_1(),
            T,
            T_fct,
            fv_cell_centers,
            fct_bcs,
            prm,
            table );
        break;
    }

    // Hand the radial material profiles to whichever energy solver was built. All four are
    // normalised by their reference values, so they are identically 1 for an incompressible
    // run and this call then changes nothing.
    energy->set_radial_profiles(
        RadialProfiles< ScalarType >{ rho_profile, alpha_profile, cp_profile, kappa_profile, true } );

    // fv_cell_centers is consumed only by the FCT advection solver after
    // initialization; for SUPG/EV it is dead weight (a 3-component FV field,
    // ~0.5 GB/GCD at production scale). Release it for the non-FCT solvers.
    if ( prm.energy_solver_parameters.energy_solver != EnergySolverType::FCT )
    {
        fv_cell_centers = linalg::VectorFVVec< ScalarType, 3 >();
    }

    // EV-specific: register the Q1-projected per-wedge ν_h diagnostic field
    // with XDMF if the energy solver exposes one.  Must happen before any
    // xdmf_output.write() call.
    if ( auto* nu_h_view = energy->nu_h_nodal_view() )
    {
        xdmf_output->add( nu_h_view->grid_data() );
    }

    if ( !prm.io_parameters.no_xdmf )
    {
        logroot << "Writing initial XDMF ..." << std::endl;

        // Write to xdmf
        auto fields = collect_xdmf_fields();
        write_xdmf(
            xdmf_output,
            xdmf_output_pressure,
            prm.time_stepping_parameters.timestep_initial,
            prm.devel_parameters.output_dimensional,
            fields.scalar_fields,
            fields.vector_fields,
            fields.pressure_field );
    }

    // ---Radial profiles---

    // Scaling factors for redimensionalisation
    ScalarType T_scale   = ScalarType( 1 );
    ScalarType eta_scale = ScalarType( 1 );
    ScalarType u_scale   = ScalarType( 1 );

    // We either want to write out dimensional depth or nondimensional radius
    std::vector< double > radial_coords = domains[velocity_level]->domain_info().radii();
    std::string           radial_label  = "radius";

    if ( prm.devel_parameters.output_dimensional )
    {
        T_scale   = prm.boundary_parameters.delta_T_K;
        eta_scale = prm.physics_parameters.viscosity_parameters.reference_viscosity;
        u_scale   = prm.physics_parameters.calc_cm_per_year;

        // Convert radii to dimensional depth
        std::transform( radial_coords.begin(), radial_coords.end(), radial_coords.begin(), [&]( double r ) {
            return std::max( 0.0, prm.mesh_parameters.radius_surface_m - r * prm.mesh_parameters.mantle_thickness_m );
        } );
        radial_label = "depth";
    }

    if ( !prm.io_parameters.no_radial_profiles )
    {
        logroot << "Writing initial radial profiles ..." << std::endl;
        compute_and_write_radial_profiles(
            T,
            subdomain_shell_idx,
            radial_coords,
            prm,
            prm.time_stepping_parameters.timestep_initial,
            T_scale,
            radial_label );
        compute_and_write_radial_profiles(
            stokes.eta_fine(),
            subdomain_shell_idx,
            radial_coords,
            prm,
            prm.time_stepping_parameters.timestep_initial,
            eta_scale,
            radial_label );
        compute_and_write_velocity_radial_profiles(
            u.block_1(),
            coords_shell[velocity_level],
            subdomain_shell_idx,
            ( *domains[velocity_level] ),
            radial_coords,
            ownership_mask_data[velocity_level],
            prm,
            prm.time_stepping_parameters.timestep_initial,
            u_scale,
            radial_label );

        // EV-specific diagnostic profiles: per-wedge h_w (geometry-only,
        // available from construction).  lap_T_ is not meaningful before any
        // time step has run, so skip the lap profile here.
        if ( auto* hw_view = energy->h_w_diag_view() )
        {
            compute_and_write_radial_profiles(
                *hw_view,
                subdomain_shell_idx,
                domains[velocity_level]->domain_info().radii(),
                prm,
                prm.time_stepping_parameters.timestep_initial );
        }
    }

    // Time stepping

    logroot << "Starting time stepping!" << std::endl;

    // Compute Nusselt at timestep 0 (before any FCT steps) for diagnostics.
    if ( prm.devel_parameters.extended_diagnostics )
    {
        const auto Nu_top_0 = compute_nusselt(
            ( *domains[velocity_level] ),
            T,
            T_cond,
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            boundary_mask_data[velocity_level],
            ownership_mask_data[velocity_level],
            true );
        const auto V_rms_0 = compute_v_rms(
            ( *domains[velocity_level] ), u.block_1(), coords_shell[velocity_level], coords_radii[velocity_level] );
        logroot << "Nu_top (Q1) = " << Nu_top_0 << ", V_rms = " << V_rms_0
                << "  [timestep 0, before time stepping]" << std::endl;
    }

    log_hbm( "after all solver setup (Stokes + Energy), before time stepping" );

    for ( int timestep = prm.time_stepping_parameters.timestep_initial + 1;
          timestep < prm.time_stepping_parameters.max_timesteps;
          timestep++ )
    {
        logroot << "\n### Timestep " << timestep << " ###" << std::endl;
        util::Timer timer_timestep( "timestep" );

        const int num_picard = prm.time_stepping_parameters.picard_iterations;

        // Per-step pre-hook (e.g. EV marks ν_h stale here, every step). Always called;
        // the backup copies inside are individually gated on num_picard > 1, so no
        // unallocated backup buffers are touched when there is a single Picard sweep.
        energy->snapshot_for_picard();

        // Compute dt once from current velocity (before Picard loop).
        ScalarType dt = energy->compute_dt( timestep );

        // Plate data is extracted once per timestep, before the Picard loop, and then
        // written into the velocity field on every Picard iteration.
        bool end_simulation_after_solve = false;
        if ( prm.boundary_parameters.plate_parameters.apply_plate_velocities )
        {
            // plate_age must never run past the requested final age.
            const ScalarType dt_to_final_age =
                ( plate_age_Ma - prm.boundary_parameters.plate_parameters.final_plate_age ) /
                prm.physics_parameters.calc_time_Ma;

            // Explicit last step, to deal with floating-point rounding around zero.
            if ( dt >= dt_to_final_age )
            {
                dt           = dt_to_final_age;
                plate_age_Ma = prm.boundary_parameters.plate_parameters.final_plate_age;

                logroot << "Last timestep changed to " << dt * prm.physics_parameters.calc_time_Ma
                        << " Ma to hit final plate age." << std::endl;

                end_simulation_after_solve = true;
            }
            else
            {
                plate_age_Ma -= dt * prm.physics_parameters.calc_time_Ma;
            }

            // Update every timestep when interpolating in time, otherwise every 1 Ma.
            bool plate_update = true;
            if ( !prm.boundary_parameters.plate_parameters.interpolate_plates_in_time )
            {
                if ( std::ceil( plate_age_Ma ) < last_plate_update_time )
                {
                    plate_update           = true;
                    last_plate_update_time = static_cast< int >( std::ceil( plate_age_Ma ) );
                }
                else
                    plate_update = false;
            }

            if ( plate_update )
            {
                extract_plate_velocities( plate_age_Ma,
                                          plate_velocities->block_1().grid_data(),
                                          *oracle,
                                          coords_shell[velocity_level],
                                          coords_radii[velocity_level],
                                          prm.boundary_parameters.plate_parameters.interpolate_plates_in_time,
                                          plate_velocity_nondim_scale,
                                          domains[velocity_level].get(),
                                          prm.boundary_parameters.plate_parameters.plates_on_device );
            }
        }

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

            // --- Energy solve (polymorphic dispatch) ---
            energy->step( dt, /*print_convergence=*/( picard == num_picard - 1 ) );

            // Update Tdev
            subtract_radial_profile( Tdev, T, T_ref, *domains[velocity_level] );

            // Update viscosity from the new temperature field.
            if ( prm.physics_parameters.viscosity_parameters.law == ViscosityLaw::FK_TYPE3 )
            {
                stokes.update_viscosity( Tdev );
            }
            else
            {
                stokes.update_viscosity( T );
            }

            // --- Stokes solve ---
            // Using full density for PDA, radial density profile else.
            // 3-D density for pda_form is passed unwrapped, see comment at
            // initial stokes solve.
            if ( pda_form )
                stokes.solve(
                    Tdev,
                    plate_velocities,
                    density->grid_data(),
                    alpha_profile,
                    prm.physics_parameters.compressible,
                    /*log_convergence=*/( picard == num_picard - 1 ) );
            else
                stokes.solve(
                    Tdev,
                    plate_velocities,
                    rho_profile,
                    alpha_profile,
                    prm.physics_parameters.compressible,
                    /*log_convergence=*/( picard == num_picard - 1 ) );

            // --- Stokes solve ---
            stokes.set_solve_time( simulated_time + prm.time_stepping_parameters.energy_substeps * dt );

            if ( timestep == prm.time_stepping_parameters.timestep_initial + 1 && picard == 0 )
                log_hbm( "after first Stokes solve (peak)" );

        } // end Picard loop

        // Output stuff, logging etc.

        table->add_row( {} );

        const bool write_output = ( timestep % prm.io_parameters.output_frequency == 0 );

        if ( write_output && !prm.io_parameters.no_xdmf )
        {
            // Write to xdmf
            auto fields = collect_xdmf_fields();

            write_xdmf(
                xdmf_output,
                xdmf_output_pressure,
                timestep,
                prm.devel_parameters.output_dimensional,
                fields.scalar_fields,
                fields.vector_fields,
                fields.pressure_field );
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
                T, subdomain_shell_idx, radial_coords, prm, timestep, T_scale, radial_label );
            compute_and_write_radial_profiles(
                stokes.eta_fine(), subdomain_shell_idx, radial_coords, prm, timestep, eta_scale, radial_label );
            compute_and_write_velocity_radial_profiles(
                u.block_1(),
                coords_shell[velocity_level],
                subdomain_shell_idx,
                ( *domains[velocity_level] ),
                radial_coords,
                ownership_mask_data[velocity_level],
                prm,
                timestep,
                u_scale,
                radial_label );

            // EV-specific diagnostic profiles (refreshed by dump_diagnostics).
            if ( auto* lap_view = energy->lap_diag_view() )
            {
                compute_and_write_radial_profiles(
                    *lap_view, subdomain_shell_idx, domains[velocity_level]->domain_info().radii(), prm, timestep );
            }
            if ( auto* hw_view = energy->h_w_diag_view() )
            {
                compute_and_write_radial_profiles(
                    *hw_view, subdomain_shell_idx, domains[velocity_level]->domain_info().radii(), prm, timestep );
            }
        }

        // Nusselt number: computed and appended to <outdir>/nu.csv at the
        // same cadence as XDMF output (output_frequency).
        if ( write_output && prm.devel_parameters.extended_diagnostics )
        {
            const auto Nu_top = compute_nusselt(
                ( *domains[velocity_level] ),
                T,
                T_cond,
                coords_shell[velocity_level],
                coords_radii[velocity_level],
                boundary_mask_data[velocity_level],
                ownership_mask_data[velocity_level],
                /*at_surface=*/true );
            const auto V_rms = compute_v_rms(
                ( *domains[velocity_level] ), u.block_1(), coords_shell[velocity_level], coords_radii[velocity_level] );
            if ( timestep % 10 == 0 )
            {
                logroot << "Nu_top (Q1) = " << Nu_top << ", V_rms = " << V_rms << std::endl;
            }
            // Per-step CSV. simulated_time is updated below; the value here is
            // the time at the *end* of this step (current T just solved).
            if ( mpi::rank() == 0 )
            {
                const std::string path = prm.io_parameters.outdir + "/nu.csv";
                std::ofstream     out( path, std::ios::app );
                if ( out.tellp() == 0 )
                {
                    out << "timestep,sim_time,Nu_top_Q1,V_rms\n";
                }
                const double t_end_of_step = simulated_time + prm.time_stepping_parameters.energy_substeps * dt;
                out << timestep << "," << t_end_of_step << "," << Nu_top << "," << V_rms << "\n";
            }
        }

        simulated_time += prm.time_stepping_parameters.energy_substeps * dt;
        simulated_time_Ma = simulated_time * prm.physics_parameters.calc_time_Ma;

        logroot << "Simulated time: " << simulated_time_Ma << " Ma\n";
        logroot << "  Stopping at " << prm.time_stepping_parameters.t_end_Ma << " Ma, "
                << std::round( simulated_time_Ma / prm.time_stepping_parameters.t_end_Ma * 100.0 * 10.0 ) / 10.0
                << "% done.\n";
        logroot << std::endl;

        timer_timestep.stop();

        if ( write_output )
        {
            write_timer_tree( prm.io_parameters, timestep );
        }

        if ( prm.boundary_parameters.plate_parameters.apply_plate_velocities )
        {
            // With plate assimilation the run ends at the requested final plate age,
            // not at t_end.
            if ( end_simulation_after_solve )
            {
                logroot << "Final plate age " << prm.boundary_parameters.plate_parameters.final_plate_age
                        << " Ma reached. Exiting simulation." << std::endl;
                logroot << "###################################################" << std::endl;
                break;
            }
        }
        else if ( simulated_time >= prm.time_stepping_parameters.t_end )
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

        // MMOC is semi-Lagrangian with a quintic reconstruction, i.e. not monotone, so a small
        // undershoot below T_min is expected and harmless -- a strict "any negative value" test
        // aborts healthy runs on it. Abort past a tolerance instead, and report the minimum
        // whenever it sets a new negative record so a genuine blow-up stays visible.
        const ScalarType T_min_global = kernels::common::min_entry( T.grid_data() );
        if ( T_min_global < T_min_record )
        {
            T_min_record = T_min_global;
            logroot << "[WARN] minimum temperature " << T_min_global << " is negative (abort below "
                    << -prm.devel_parameters.negative_temperature_tolerance << ")." << std::endl;
        }

        if ( T_min_global < -static_cast< ScalarType >( prm.devel_parameters.negative_temperature_tolerance ) )
        {
            logroot << "\nDETECTED NEGATIVE TEMPERATURE VALUES.\n"
                       "Minimum temperature: "
                    << T_min_global << "\nAborting simulation...\n"
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
