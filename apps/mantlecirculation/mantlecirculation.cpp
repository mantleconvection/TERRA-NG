#include <cmath>
#include <fstream>
#include <vector>

#include "communication/shell/communication.hpp"
#include "communication/shell/fv_communication.hpp"
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
#include "src/interpolators.hpp"
#include "src/io.hpp"
#include "src/parameters.hpp"
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

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< ScalarType, 3 > >                     coords_shell;
    std::vector< Grid2DDataScalar< ScalarType > >                     coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        ownership_mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    const int lat_sdr = ( prm.mesh_parameters.lat_sdr >= 0 ) ? prm.mesh_parameters.lat_sdr
                                                             : prm.mesh_parameters.refinement_level_subdomains;
    const int rad_sdr = ( prm.mesh_parameters.rad_sdr >= 0 ) ? prm.mesh_parameters.rad_sdr
                                                             : prm.mesh_parameters.refinement_level_subdomains;

    for ( int level = prm.mesh_parameters.refinement_level_mesh_min;
          level <= prm.mesh_parameters.refinement_level_mesh_max;
          level++ )
    {
        const int idx       = level - prm.mesh_parameters.refinement_level_mesh_min;
        const int lat_level = level;
        const int rad_level = level + prm.mesh_parameters.radial_extra_levels;

        domains.push_back(
            DistributedDomain::create_uniform(
                lat_level,
                rad_level,
                prm.mesh_parameters.radius_min,
                prm.mesh_parameters.radius_max,
                lat_sdr,
                rad_sdr ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        ownership_mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const auto subdomain_distr = grid::shell::subdomain_distribution( domains.back() );
    logroot << "Subdomain distribution (subdomains per MPI process): \n";
    logroot << " - total: " << subdomain_distr.total << "\n";
    logroot << " - min:   " << subdomain_distr.min << "\n";
    logroot << " - avg:   " << subdomain_distr.avg << "\n";
    logroot << " - max:   " << subdomain_distr.max << "\n\n";

    const int  num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    Grid2DDataScalar< int > subdomain_shell_idx = grid::shell::subdomain_shell_idx( domains[velocity_level] );

    // Set up Stokes vectors for the finest grid.

    const std::string label_stokes = "u";

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    std::vector< std::string >                             stok_vec_names = { label_stokes, "f", "tmp" };

    for ( const auto& name : stok_vec_names )
    {
        stok_vecs[name] = VectorQ1IsoQ2Q1< ScalarType >(
            name,
            domains[velocity_level],
            domains[pressure_level],
            ownership_mask_data[velocity_level],
            ownership_mask_data[pressure_level] );
    }

    auto& u = stok_vecs["u"];
    auto& f = stok_vecs["f"];

    // Set up viscosity.
    //
    // For simplicity, we do not optimize for the isoviscous case, but always use the full Stokes operator.
    // That means in the isoviscous case we choose a constant radial viscosity profile.
    //
    // If a temperature-dependent viscosity law is selected, the initial viscosity is set from the
    // radial/constant profile and then overwritten after the initial temperature field is set up.

    std::vector< Grid2DDataScalar< ScalarType > > radial_viscosity_profile;

    if ( !prm.physics_parameters.viscosity_parameters.radial_profile_enabled )
    {
        logroot << "Using constant viscosity profile." << std::endl;
        for ( int level = 0; level < num_levels; level++ )
        {
            radial_viscosity_profile.push_back(
                shell::interpolate_constant_radial_profile( coords_radii[level], 1.0 ) );
        }
    }
    else
    {
        logroot << "Using radially varying viscosity profile." << std::endl;
        for ( int level = 0; level < num_levels; level++ )
        {
            radial_viscosity_profile.push_back(
                shell::interpolate_radial_profile_into_subdomains_from_csv(
                    prm.physics_parameters.viscosity_parameters.radial_profile_csv_filename,
                    prm.physics_parameters.viscosity_parameters.radial_profile_radii_key,
                    prm.physics_parameters.viscosity_parameters.radial_profile_viscosity_key,
                    coords_radii[level] ) );
        }
    }

    // We project the viscosity into an FE space. Thus, we need some coefficient vectors.
    std::vector< VectorQ1Scalar< ScalarType > > eta;
    eta.reserve( num_levels );
    for ( int level = 0; level < num_levels; level++ )
    {
        if ( level == num_levels - 1 )
        {
            eta.emplace_back( "eta", domains[level], ownership_mask_data[level] );
        }
        else
        {
            eta.emplace_back( "eta_level_" + std::to_string( level ), domains[level], ownership_mask_data[level] );
        }
    }

    for ( int level = 0; level < num_levels; level++ )
    {
        // Note that although we perform GCA we need some approximation of the viscosity for the
        // coarse grids for the weighting of the mass matrix.
        geophysics::viscosity::RadialProfileViscosityInterpolator viscosity_interpolator(
            radial_viscosity_profile[level], prm.physics_parameters.viscosity_parameters.reference_viscosity );
        viscosity_interpolator.interpolate( eta[level].grid_data() );
    }

    // Setting up the (adaptive) Galerkin coarse grid approximation (AGCA / GCA)
    // Determine AGCA elements.
    VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", domains[0], ownership_mask_data[0] );
    int                          gca = 0;
    if ( gca == 2 )
    {
        linalg::assign( GCAElements, 0 );
        logroot << "Adaptive GCA: determining GCA elements on level " << velocity_level << std::endl;
        terra::linalg::solvers::GCAElementsCollector< ScalarType >(
            domains[velocity_level], eta[velocity_level].grid_data(), velocity_level, GCAElements.grid_data() );
    }
    else if ( gca == 1 )
    {
        logroot << "GCA on all elements " << std::endl;
        assign( GCAElements, 1 );
    }

    // Set up tmp vecs for FGMRES (Stokes). We need quite a few :(

    std::vector< VectorQ1IsoQ2Q1< ScalarType > > stokes_tmp_fgmres;

    const auto num_stokes_fgmres_tmps = 2 * prm.stokes_solver_parameters.krylov_restart + 4;

    stokes_tmp_fgmres.reserve( num_stokes_fgmres_tmps );
    for ( int i = 0; i < num_stokes_fgmres_tmps; i++ )
    {
        stokes_tmp_fgmres.emplace_back(
            "stokes_tmp_fgmres",
            domains[velocity_level],
            domains[pressure_level],
            ownership_mask_data[velocity_level],
            ownership_mask_data[pressure_level] );
    }

    // Set up tmp vecs for Stokes multigrid preconditioner.

    std::vector< VectorQ1Vec< ScalarType > > tmp_mg;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_2;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_r;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_e;

    for ( int level = 0; level < num_levels; level++ )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], ownership_mask_data[level] );
        tmp_mg_2.emplace_back( "tmp_mg_2_" + std::to_string( level ), domains[level], ownership_mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], ownership_mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], ownership_mask_data[level] );
        }
    }

    // Set up vectors for energy equation.

    const std::string label_temperature = "T";

    std::map< std::string, VectorQ1Scalar< ScalarType > > temp_vecs;
    std::vector< std::string >                            temp_vec_names = { label_temperature, "q" };
    constexpr int                                         num_temp_tmps  = 8;

    for ( int i = 0; i < num_temp_tmps; i++ )
    {
        temp_vec_names.push_back( "tmp_" + std::to_string( i ) );
    }

    for ( const auto& name : temp_vec_names )
    {
        temp_vecs[name] =
            VectorQ1Scalar< ScalarType >( name, domains[velocity_level], ownership_mask_data[velocity_level] );
    }

    auto& T = temp_vecs["T"];
    auto& q = temp_vecs["q"];

    // Finite-volume functions/vectors.

    // FV cell-centred temperature field (the FCT prognostic variable).
    linalg::VectorFVScalar< ScalarType > T_fct( "T_fct", domains[velocity_level] );
    // Pre-computed cell centres (with ghost layers filled once and reused every step).
    linalg::VectorFVVec< ScalarType, 3 > fv_cell_centers( "fv_cell_centers", domains[velocity_level] );
    fv::hex::initialize_cell_centers(
        fv_cell_centers, domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level] );
    // Pre-allocated FCT scratch buffers (reused every step).
    fv::hex::operators::FVFCTBuffers< ScalarType > fv_fct_bufs( domains[velocity_level] );
    // Temporaries for the FV→Q1 L2 projection (reused every step; share storage with temp_vecs).
    // l2_project_fv_to_fe requires at least 5 Q1 temporaries.
    std::vector< VectorQ1Scalar< ScalarType > > l2_proj_tmps = {
        temp_vecs["tmp_0"], temp_vecs["tmp_1"], temp_vecs["tmp_2"], temp_vecs["tmp_3"], temp_vecs["tmp_4"] };

    linalg::VectorFVScalar< ScalarType > T_source( "T_source", domains[velocity_level] );
    linalg::assign( T_source, 0.0 );

    // Counting DoFs.
    int world_size = mpi::num_processes();

    const auto num_dofs_fe_scalar =
        kernels::common::count_masked< long >( ownership_mask_data[num_levels - 1], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_velocity = 3 * num_dofs_fe_scalar;
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( ownership_mask_data[num_levels - 2], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_temperature = domains[velocity_level].domain_info().num_global_micro_hex_cells();

    logroot << "Degrees of freedom in (T,u,p) = (" << num_dofs_temperature << ", " << num_dofs_velocity << ", "
            << num_dofs_pressure << ")" << std::endl;
    logroot << "Avg DoFs/process in (T,u,p)   = (" << num_dofs_temperature / world_size << ", "
            << num_dofs_velocity / world_size << ", " << num_dofs_pressure / world_size << ")" << std::endl;

    // Set up operators.

    using Stokes      = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    using Gradient = Stokes::Block12Type;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

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

    // For strong BC elimination, we also need the Neumann operators.
    // So we have this set of BCs as well (will not be used in the solver later, just for the RHS set up).

    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };

    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        eta[velocity_level].grid_data(),
        bcs,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        eta[velocity_level].grid_data(),
        bcs_neumann,
        false );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // Multigrid operators

    logroot << "Setting up Stokes solver and preconditioners ..." << std::endl;

    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;

    // Coarse grid operators.
    // For GCA we need to store the local element matrices on the coarser grids.

    for ( int level = 0; level < num_levels - 1; level++ )
    {
        A_c.emplace_back(
            domains[level],
            coords_shell[level],
            coords_radii[level],
            boundary_mask_data[level],
            eta[level].grid_data(),
            bcs,
            false );
        if ( gca == 2 )
        {
            A_c.back().set_stored_matrix_mode(
                linalg::OperatorStoredMatrixMode::Selective, level, GCAElements.grid_data() );
        }
        else if ( gca == 1 )
        {
            A_c.back().set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Full, level, GCAElements.grid_data() );
        }
        P.emplace_back( linalg::OperatorApplyMode::Add );
        R.emplace_back( domains[level] );
    }

    // GCA assembly
    if ( gca > 0 )
    {
        for ( int level = num_levels - 2; level >= 0; level-- )
        {
            logroot << "Assembling GCA on level " << prm.mesh_parameters.refinement_level_mesh_min + level << std::endl;

            TwoGridGCA< ScalarType, Viscous >(
                ( level == num_levels - 2 ) ? K_neumann.block_11() : A_c[level + 1],
                A_c[level],
                level,
                GCAElements.grid_data() );
        }
    }

    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;

    for ( int level = 0; level < num_levels; level++ )
    {
        inverse_diagonals.emplace_back(
            "inverse_diagonal_" + std::to_string( level ), domains[level], ownership_mask_data[level] );

        VectorQ1Vec< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( level ), domains[level], ownership_mask_data[level] );

        linalg::assign( tmp, 1.0 );
        if ( level == num_levels - 1 )
        {
            K.block_11().set_diagonal( true );
            linalg::apply( K.block_11(), tmp, inverse_diagonals.back() );
            K.block_11().set_diagonal( false );
        }
        else
        {
            A_c[level].set_diagonal( true );
            linalg::apply( A_c[level], tmp, inverse_diagonals.back() );
            A_c[level].set_diagonal( false );
        }
        linalg::invert_entries( inverse_diagonals.back() );
    }

    // Set up solvers.

    // Multigrid preconditioner.

    logroot << "Setting up multigrid smoother ..." << std::endl;

    using Smoother = linalg::solvers::Chebyshev< Viscous >;

    std::vector< Smoother > smoothers;
    smoothers.reserve( num_levels );

    for ( int level = 0; level < num_levels; level++ )
    {
        std::vector< VectorQ1Vec< ScalarType > > smoother_tmps;
        smoother_tmps.push_back( tmp_mg[level] );
        smoother_tmps.push_back( tmp_mg_2[level] );

        smoothers.emplace_back(
            prm.stokes_solver_parameters.viscous_pc_chebyshev_order,
            inverse_diagonals[level],
            smoother_tmps,
            prm.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost,
            prm.stokes_solver_parameters.viscous_pc_num_power_iterations );
    }

    // --- Diagnostic: estimate and log the Chebyshev spectrum (max eigenvalue of D^{-1} A_viscous) per level. ---
    // This mirrors Chebyshev's internal estimate_max_eigenvalues; Chebyshev still does its own
    // estimation lazily on first solve, so this just reports what it will compute.
    for ( int level = 0; level < num_levels; level++ )
    {
        VectorQ1Vec< ScalarType > tmp_pi_it(
            "cheby_est_tmpIt", domains[level], ownership_mask_data[level] );
        VectorQ1Vec< ScalarType > tmp_pi_aux(
            "cheby_est_tmpAux", domains[level], ownership_mask_data[level] );
        const auto log_level = prm.mesh_parameters.refinement_level_mesh_min + level;
        auto&      A_lvl     = ( level == num_levels - 1 ) ? K.block_11() : A_c[level];
        linalg::DiagonallyScaledOperator< Viscous > inv_diag_A( A_lvl, inverse_diagonals[level] );
        const double lmax_est = linalg::solvers::power_iteration(
            inv_diag_A,
            tmp_pi_it,
            tmp_pi_aux,
            prm.stokes_solver_parameters.viscous_pc_num_power_iterations );
        logroot << "[Cheby estimate] level " << log_level
                << ": lambda_max(D^-1 A_viscous) ~ " << lmax_est
                << "  => lambda_max_cheby = " << 1.5 * lmax_est
                << ", lambda_min_cheby = " << 0.1 * lmax_est << std::endl;
    }

    logroot << "Setting up multigrid coarse grid solver ..." << std::endl;

    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;

    std::vector< VectorQ1Vec< ScalarType > > coarse_grid_tmps;
    coarse_grid_tmps.reserve( 4 );
    for ( int i = 0; i < 4; i++ )
    {
        coarse_grid_tmps.emplace_back( "tmp_coarse_grid", domains[0], ownership_mask_data[0] );
    }

    CoarseGridSolver coarse_grid_solver(
        linalg::solvers::IterativeSolverParameters{ 50, 1e-6, 1e-16 }, table, coarse_grid_tmps );
    coarse_grid_solver.set_tag( "coarse_grid_pcg" );

    logroot << "Setting up multigrid preconditioner ..." << std::endl;

    using PrecVisc = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;
    PrecVisc prec_11(
        P,
        R,
        A_c,
        tmp_mg_r,
        tmp_mg_e,
        tmp_mg,
        smoothers,
        smoothers,
        coarse_grid_solver,
        prm.stokes_solver_parameters.viscous_pc_num_vcycles,
        1e-6 );

    // Schur complement: lumped inverse diagonal of pressure mass

    logroot << "Setting up Schur complement preconditioner ..." << std::endl;

    VectorQ1Scalar< ScalarType > k_pm( "k_pm", domains[pressure_level], ownership_mask_data[pressure_level] );
    assign( k_pm, eta[pressure_level] );
    linalg::invert_entries( k_pm );

    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;
    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level], k_pm.grid_data(), false );
    pmass.set_lumped_diagonal( true );
    VectorQ1Scalar< ScalarType > lumped_diagonal_pmass(
        "lumped_diagonal_pmass", domains[pressure_level], ownership_mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( pressure_level ),
            domains[pressure_level],
            ownership_mask_data[pressure_level] );
        linalg::assign( tmp, 1.0 );
        linalg::apply( pmass, tmp, lumped_diagonal_pmass );
    }

    using PrecSchur = linalg::solvers::DiagonalSolver< PressureMass >;
    PrecSchur inv_lumped_pmass( lumped_diagonal_pmass );

    // Set up outer block-preconditioner

    logroot << "Setting up outer block-preconditioner ..." << std::endl;

    using PrecStokes = linalg::solvers::
        BlockTriangularPreconditioner2x2< Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;

    VectorQ1IsoQ2Q1< ScalarType > triangular_prec_tmp(
        "triangular_prec_tmp",
        domains[velocity_level],
        domains[pressure_level],
        ownership_mask_data[velocity_level],
        ownership_mask_data[pressure_level] );

    PrecStokes prec_stokes( K.block_11(), pmass, K.block_12(), triangular_prec_tmp, prec_11, inv_lumped_pmass );

    logroot << "Setting up FGMRES ..." << std::endl;

    linalg::solvers::FGMRES< Stokes, PrecStokes > stokes_fgmres(
        stokes_tmp_fgmres,
        { .restart                     = prm.stokes_solver_parameters.krylov_restart,
          .relative_residual_tolerance = prm.stokes_solver_parameters.krylov_relative_tolerance,
          .absolute_residual_tolerance = prm.stokes_solver_parameters.krylov_absolute_tolerance,
          .max_iterations              = prm.stokes_solver_parameters.krylov_max_iterations },
        table,
        prec_stokes );
    stokes_fgmres.set_tag( "stokes_fgmres" );

    /////////////////////
    /// ENERGY SOLVER ///
    /////////////////////

    logroot << "Setting up energy equation solver ..." << std::endl;

    // Set up the initial temperature.

    // --- Initialise temperature ---

    const fv::hex::DirichletBCs< ScalarType > fct_bcs{
        .T_cmb         = static_cast< ScalarType >( prm.boundary_conditions_parameters.temperature_cmb ),
        .T_surface     = static_cast< ScalarType >( prm.boundary_conditions_parameters.temperature_surface ),
        .apply_cmb     = true,
        .apply_surface = true };

    const auto& init_temp = prm.physics_parameters.initial_temperature;

    if ( init_temp.profile == InitialTemperatureProfile::CONDUCTIVE )
    {
        const bool has_sph_2 =
            ( init_temp.sph_degree_l_2 > 0 && init_temp.sph_factor_2 != 0.0 && init_temp.sph_epsilon != 0.0 );

        logroot << "Initial temperature: conductive profile";
        if ( init_temp.sph_degree_l > 0 && init_temp.sph_epsilon != 0.0 )
        {
            logroot << " + eps * (Y_" << init_temp.sph_degree_l << "^" << init_temp.sph_order_m;
            if ( has_sph_2 )
            {
                logroot << " + " << init_temp.sph_factor_2 << " * Y_" << init_temp.sph_degree_l_2 << "^"
                        << init_temp.sph_order_m_2;
            }
            logroot << ") (eps=" << init_temp.sph_epsilon << ")";
        }
        logroot << std::endl;

        // Compute spherical harmonic coefficients on unit sphere Q1 nodes.
        Grid3DDataScalar< ScalarType > sph_coeffs;
        const bool has_sph = ( init_temp.sph_degree_l > 0 && init_temp.sph_epsilon != 0.0 );
        if ( has_sph )
        {
            sph_coeffs = shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >(
                init_temp.sph_degree_l, init_temp.sph_order_m, coords_shell[velocity_level] );

            if ( has_sph_2 )
            {
                // Add factor_2 * Y_l2^m2 to the first harmonic in-place.
                auto sph_coeffs_2 = shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >(
                    init_temp.sph_degree_l_2, init_temp.sph_order_m_2, coords_shell[velocity_level] );
                const ScalarType factor_2 = static_cast< ScalarType >( init_temp.sph_factor_2 );
                Kokkos::parallel_for(
                    "combine spherical harmonics",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
                        { 0, 0, 0 },
                        { static_cast< int >( sph_coeffs.extent( 0 ) ),
                          static_cast< int >( sph_coeffs.extent( 1 ) ),
                          static_cast< int >( sph_coeffs.extent( 2 ) ) } ),
                    KOKKOS_LAMBDA( int sd, int x, int y ) {
                        sph_coeffs( sd, x, y ) += factor_2 * sph_coeffs_2( sd, x, y );
                    } );
                Kokkos::fence();
            }
        }

        // Fill Q1 T with conductive profile + spherical harmonic perturbation.
        Kokkos::parallel_for(
            "initial temp (conductive + sph. harm.)",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            ConductiveProfileInterpolator{
                domains[velocity_level].domain_info().radii().front(),
                domains[velocity_level].domain_info().radii().back(),
                init_temp.sph_epsilon,
                coords_shell[velocity_level],
                coords_radii[velocity_level],
                T.grid_data(),
                sph_coeffs,
                has_sph } );
        Kokkos::fence();
        // NOTE: do NOT call send_recv here. The kernel writes the same value to every
        // subdomain copy of each shared node already; send_recv (SUM reduction) would
        // accumulate them and produce N*value at shared nodes (visible in the XDMF dump).
        // The downstream FE->FV->FE projection re-establishes consistency.

        // Project Q1 T → FV T_fct.
        fv::hex::l2_project_fe_to_fv(
            T_fct, T, domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level] );
    }
    else
    {
        logroot << "Initial temperature: power-law + noise" << std::endl;

        Kokkos::parallel_for(
            "initial temp interpolation (FCT)",
            grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domains[velocity_level] ),
            FVInitialConditionInterpolator{
                domains[velocity_level].domain_info().radii().front(),
                domains[velocity_level].domain_info().radii().back(),
                fv_cell_centers.grid_data(),
                T_fct.grid_data() } );
        Kokkos::fence();

        Kokkos::parallel_for(
            "adding noise to temp (FCT)",
            grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domains[velocity_level] ),
            FVNoiseAdder{ T_fct.grid_data(), Kokkos::Random_XorShift64_Pool<>( 12345 ) } );
        Kokkos::fence();
    }

    // Enforce Dirichlet BCs on the FV field and exchange ghost layers.
    fv::hex::apply_dirichlet_bcs( T_fct, boundary_mask_data[velocity_level], fct_bcs, domains[velocity_level] );
    communication::shell::update_fv_ghost_layers( domains[velocity_level], T_fct.grid_data() );

    // Project T_fct to Q1 T via L2 projection for use as Stokes RHS and output.
    fv::hex::l2_project_fv_to_fe(
        T, T_fct, domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], l2_proj_tmps );

    // If temperature-dependent viscosity is enabled, compute the initial viscosity from the initial T.
    if ( prm.physics_parameters.viscosity_parameters.law != ViscosityLaw::CONSTANT )
    {
        logroot << "Computing initial temperature-dependent viscosity ..." << std::endl;
        Kokkos::parallel_for(
            "viscosity_from_temperature_init",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            ViscosityFromTemperature{ prm.physics_parameters.viscosity_parameters.law,
                                     prm.physics_parameters.viscosity_parameters.rmu,
                                     eta[velocity_level].grid_data(),
                                     T.grid_data() } );
        Kokkos::fence();
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
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level] );

    xdmf_output.add( T.grid_data() );
    xdmf_output.add( u.block_1().grid_data() );
    xdmf_output.add( eta[velocity_level].grid_data() );

    // Reference conductive temperature profile (also used for the Nusselt number).
    // T_ref = r_min * r_max / r - r_min : spherical steady-state conduction solution.
    VectorQ1Scalar< ScalarType > T_ref( "T_ref", domains[velocity_level], ownership_mask_data[velocity_level] );
    Kokkos::parallel_for(
        "conductive profile T_ref",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        ConductiveProfileInterpolator{
            domains[velocity_level].domain_info().radii().front(),
            domains[velocity_level].domain_info().radii().back(),
            ScalarType( 0 ),
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            T_ref.grid_data(),
            {},
            false } );
    Kokkos::fence();
    // NOTE: do NOT call send_recv here. The kernel writes the correct value to every
    // subdomain copy of each shared node already; calling send_recv would accumulate
    // them and produce N*value at shared nodes (where N is the sharing count).

    xdmf_output.add( T_ref.grid_data() );

    int timestep_initial = 0;

    const bool loading_checkpoint = !prm.io_parameters.checkpoint_dir.empty() && prm.io_parameters.checkpoint_step >= 0;

    if ( loading_checkpoint )
    {
        const int checkpoint_file_step = prm.io_parameters.checkpoint_step;

        // If a checkpoint-timestep is provided, use it as the actual simulation timestep.
        // Otherwise fall back to the checkpoint output step number.
        timestep_initial = ( prm.io_parameters.checkpoint_timestep >= 0 )
                             ? prm.io_parameters.checkpoint_timestep
                             : checkpoint_file_step;

        logroot << "Loading checkpoint from " << prm.io_parameters.checkpoint_dir
                << " (file step " << checkpoint_file_step << ", simulation timestep " << timestep_initial << ")"
                << std::endl;

        auto success_vel = io::read_xdmf_checkpoint_grid(
            prm.io_parameters.checkpoint_dir,
            label_stokes + "_u",
            checkpoint_file_step,
            domains[velocity_level],
            u.block_1().grid_data() );

        if ( success_vel.is_err() )
        {
            Kokkos::abort( success_vel.error().c_str() );
        }

        auto success_temp = io::read_xdmf_checkpoint_grid(
            prm.io_parameters.checkpoint_dir,
            label_temperature,
            checkpoint_file_step,
            domains[velocity_level],
            T.grid_data() );

        if ( success_temp.is_err() )
        {
            Kokkos::abort( success_temp.error().c_str() );
        }

        // Setting XDMF write counter to continue from the checkpoint file step.
        xdmf_output.set_write_counter( checkpoint_file_step );

        // T_fct is not stored in checkpoints (only Q1 T is).  Recover the FV cell-average
        // field from the restored Q1 T via an L2 projection.  Ghost layers are populated
        // inside l2_project_fe_to_fv, so the result is immediately usable by FCT kernels.
        fv::hex::l2_project_fe_to_fv(
            T_fct, T, domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level] );
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
            T, subdomain_shell_idx, domains[velocity_level], prm.io_parameters, timestep_initial );
        compute_and_write_radial_profiles(
            eta[velocity_level], subdomain_shell_idx, domains[velocity_level], prm.io_parameters, timestep_initial );
        compute_and_write_velocity_radial_profiles(
            u.block_1(),
            coords_shell[velocity_level],
            subdomain_shell_idx,
            domains[velocity_level],
            ownership_mask_data[velocity_level],
            prm.io_parameters,
            timestep_initial );
    }

    ScalarType simulated_time = prm.time_stepping_parameters.t_start;

    // We need some global h. Let's, for simplicity (does not need to be too accurate) just choose the smallest h in
    // radial direction.
    const auto h = grid::shell::min_radial_h( domains[velocity_level].domain_info().radii() );

    // --- SUPG energy solver setup (only constructed if needed) ---

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;
    using TempMass = fe::wedge::operators::shell::Mass< ScalarType >;

    const bool use_supg = ( prm.time_stepping_parameters.energy_solver == EnergySolverType::SUPG );

    // SUPG operator: A = M + dt * (K_diff + K_adv + K_supg), with Dirichlet rows.
    using DiagSolver = linalg::solvers::DiagonalSolver< AD >;

    std::unique_ptr< AD > supg_A, supg_A_neumann, supg_A_neumann_diag;
    std::unique_ptr< TempMass > supg_M;
    std::unique_ptr< linalg::solvers::FGMRES< AD, DiagSolver > > supg_solver;
    VectorQ1Scalar< ScalarType > supg_g( "supg_g", domains[velocity_level], ownership_mask_data[velocity_level] );
    VectorQ1Scalar< ScalarType > supg_tmp( "supg_tmp", domains[velocity_level], ownership_mask_data[velocity_level] );
    VectorQ1Scalar< ScalarType > supg_diag( "supg_diag", domains[velocity_level], ownership_mask_data[velocity_level] );

    if ( use_supg )
    {
        logroot << "Setting up SUPG energy solver ..." << std::endl;

        supg_A = std::make_unique< AD >(
            domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
            boundary_mask_data[velocity_level], u.block_1(),
            prm.physics_parameters.diffusivity, 0.0, /*treat_boundary=*/true );

        supg_A_neumann = std::make_unique< AD >(
            domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
            boundary_mask_data[velocity_level], u.block_1(),
            prm.physics_parameters.diffusivity, 0.0, /*treat_boundary=*/false );

        supg_A_neumann_diag = std::make_unique< AD >(
            domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
            boundary_mask_data[velocity_level], u.block_1(),
            prm.physics_parameters.diffusivity, 0.0, /*treat_boundary=*/false, /*diagonal=*/true );

        supg_M = std::make_unique< TempMass >(
            domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

        // Compute diagonal of the SUPG operator for Jacobi preconditioning.
        // We need to set a representative dt first; it will be updated each timestep.
        supg_A_neumann_diag->dt() = ScalarType( 1e-4 );
        linalg::assign( supg_diag, 0.0 );
        {
            VectorQ1Scalar< ScalarType > ones( "ones", domains[velocity_level], ownership_mask_data[velocity_level] );
            linalg::assign( ones, 1.0 );
            linalg::apply( *supg_A_neumann_diag, ones, supg_diag );
        }

        constexpr int num_supg_gmres_tmps = 14;
        std::vector< VectorQ1Scalar< ScalarType > > tmp_energy_gmres( num_supg_gmres_tmps );
        for ( int i = 0; i < num_supg_gmres_tmps; i++ )
        {
            tmp_energy_gmres[i] = VectorQ1Scalar< ScalarType >(
                "tmp_energy_gmres", domains[velocity_level], ownership_mask_data[velocity_level] );
        }

        supg_solver = std::make_unique< linalg::solvers::FGMRES< AD, DiagSolver > >(
            tmp_energy_gmres,
            linalg::solvers::FGMRESOptions{
                .restart                     = prm.energy_solver_parameters.krylov_restart,
                .relative_residual_tolerance = prm.energy_solver_parameters.krylov_relative_tolerance,
                .absolute_residual_tolerance = prm.energy_solver_parameters.krylov_absolute_tolerance,
                .max_iterations              = prm.energy_solver_parameters.krylov_max_iterations },
            table,
            DiagSolver( supg_diag ) );

        logroot << "SUPG energy solver ready." << std::endl;
    }

    // Time stepping

    logroot << "Starting time stepping!" << std::endl;

    // Compute Nusselt at timestep 0 (before any FCT steps) for diagnostics.
    {
        const auto Nu_top_0 = compute_nusselt(
            domains[velocity_level], T, T_ref, coords_shell[velocity_level], coords_radii[velocity_level], boundary_mask_data[velocity_level], ownership_mask_data[velocity_level], true );
        const auto Nu_top_fv_0 = compute_nusselt_fv(
            domains[velocity_level], T_fct, boundary_mask_data[velocity_level],
            prm.boundary_conditions_parameters.temperature_surface,
            prm.boundary_conditions_parameters.temperature_cmb,
            prm.mesh_parameters.radius_min, prm.mesh_parameters.radius_max, true );
        logroot << "Nu_top (Q1) = " << Nu_top_0 << ", Nu_top (FV) = " << Nu_top_fv_0
                << "  [timestep 0, before time stepping]" << std::endl;
    }

    // Backup of FV temperature for Picard iteration (re-do energy from same starting point).
    linalg::VectorFVScalar< ScalarType > T_fct_backup( "T_fct_backup", domains[velocity_level] );

    for ( int timestep = timestep_initial + 1; timestep < prm.time_stepping_parameters.max_timesteps; timestep++ )
    {
        logroot << "\n### Timestep " << timestep << " ###" << std::endl;
        util::Timer timer_timestep( "timestep" );


        const int num_picard = prm.time_stepping_parameters.picard_iterations;

        // Save T_fct at the start of the timestep so we can restore it for each Picard iteration.
        Kokkos::deep_copy( T_fct_backup.grid_data(), T_fct.grid_data() );

        // Compute dt once from current velocity (before Picard loop).
        ScalarType dt;
        if ( use_supg )
        {
            // SUPG: implicit diffusion, so dt is only constrained by advection CFL.
            const auto max_vel = kernels::common::max_vector_magnitude( u.block_1().grid_data() );
            const auto dt_advection = ( max_vel > 1e-12 ) ? ( h / max_vel ) : ScalarType( 1e-3 );
            dt = prm.time_stepping_parameters.pseudo_cfl * dt_advection;
            logroot << "Computing dt (SUPG advection CFL) ..." << std::endl;
            logroot << "    max_vel:                       " << max_vel << std::endl;
            logroot << "    h:                             " << h << std::endl;
            logroot << "=>  dt (= pseudo_cfl * h/v_max):   " << dt << std::endl;
        }
        else
        {
            const auto dt_stable = fv::hex::operators::compute_dt_stable(
                domains[velocity_level],
                u.block_1(),
                fv_cell_centers.grid_data(),
                coords_shell[velocity_level],
                coords_radii[velocity_level],
                prm.physics_parameters.diffusivity );
            dt = prm.time_stepping_parameters.pseudo_cfl * dt_stable;
            logroot << "Computing dt (FCT stable) ..." << std::endl;
            logroot << "    dt_stable:                     " << dt_stable << std::endl;
            logroot << "=>  dt (= dt_stable * pseudo_cfl): " << dt << std::endl;
        }

        for ( int picard = 0; picard < num_picard; picard++ )
        {
            if ( num_picard > 1 )
                logroot << "--- Picard iteration " << picard << " / " << num_picard << " ---" << std::endl;

            // Restore T_fct to start-of-timestep state (iterations > 0 redo energy from the same starting point).
            if ( picard > 0 )
            {
                Kokkos::deep_copy( T_fct.grid_data(), T_fct_backup.grid_data() );
            }

            // --- Stokes solve ---
            const auto solve_stokes_at_temperature = [&]( const VectorQ1Scalar< ScalarType >& T_for_buoyancy,
                                                          const bool                          print_convergence ) {
                util::Timer timer_stokes( "stokes" );

                logroot << "Setting up Stokes rhs ..." << std::endl;

                Kokkos::parallel_for(
                    "Stokes rhs interpolation",
                    local_domain_md_range_policy_nodes( domains[velocity_level] ),
                    RHSVelocityInterpolator(
                        coords_shell[velocity_level],
                        coords_radii[velocity_level],
                        stok_vecs["tmp"].block_1().grid_data(),
                        T_for_buoyancy.grid_data(),
                        prm.physics_parameters.rayleigh_number ) );

                linalg::apply( M, stok_vecs["tmp"].block_1(), stok_vecs["f"].block_1() );

                fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
                    stok_vecs["f"],
                    boundary_mask_data[velocity_level],
                    grid::shell::get_shell_boundary_flag( bcs, DIRICHLET ) );

                fe::strong_algebraic_freeslip_enforcement_in_place(
                    stok_vecs["f"],
                    coords_shell[velocity_level],
                    boundary_mask_data[velocity_level],
                    grid::shell::get_shell_boundary_flag( bcs, FREESLIP ) );

                logroot << "Solving Stokes ..." << std::endl;

                solve( stokes_fgmres, K, u, f );

                if ( print_convergence )
                {
                    table->query_rows_equals( "tag", "stokes_fgmres" ).print_pretty();
                    table->query_rows_equals( "tag", "coarse_grid_pcg" ).print_pretty();
                }
                table->clear();

                // "Normalize" pressure.
                const ScalarType avg_pressure_approximation =
                    kernels::common::masked_sum(
                        u.block_2().grid_data(), u.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
                    static_cast< ScalarType >( num_dofs_pressure );
                linalg::lincomb( u.block_2(), { 1.0 }, { u.block_2() }, -avg_pressure_approximation );

        // --- FCT explicit time-stepping ---
        // Compute the exact stable dt from the actual face-normal velocity fluxes and cell
        // volumes via a parallel reduce over all cells.  This is more accurate than the
        // h_min / u_max estimate, which ignores smaller lateral cells near pentagon vertices
        // of the icosahedral grid and diffusion stiffness on non-orthogonal faces.
        const auto dt_stable = fv::hex::operators::compute_dt_stable(
            domains[velocity_level],
            u.block_1(),
            fv_cell_centers.grid_data(),
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            prm.physics_parameters.diffusivity );
        const auto dt = prm.time_stepping_parameters.dt_scaling * dt_stable;

        logroot << "Computing dt (FCT stable) ..." << std::endl;
        logroot << "    dt_stable:                     " << dt_stable << std::endl;
        logroot << "=>  dt (= dt_stable * dt_scaling): " << dt << std::endl;

            // --- Energy solve ---

            util::Timer timer_energy( "energy" );

            logroot << "Setting up energy solve ..." << std::endl;

            if ( use_supg )
            {
                // --- SUPG implicit energy solve ---

                supg_A->dt()              = dt;
                supg_A_neumann->dt()      = dt;
                supg_A_neumann_diag->dt() = dt;

                // Update inverse diagonal for preconditioner.
                // supg_diag was inverted by DiagonalSolver constructor, so we recompute fresh
                // and invert manually. DiagonalSolver stores a reference to supg_diag.
                {
                    VectorQ1Scalar< ScalarType > ones( "ones", domains[velocity_level], ownership_mask_data[velocity_level] );
                    linalg::assign( ones, 1.0 );
                    linalg::apply( *supg_A_neumann_diag, ones, supg_diag );
                    linalg::invert_entries( supg_diag );
                }

                for ( int i = 0; i < prm.time_stepping_parameters.energy_substeps; i++ )
                {
                    logroot << "Solving energy (SUPG, substep " << i << ") ..." << std::endl;

                    // RHS: q = M * T^n
                    linalg::apply( *supg_M, T, q );

                    // Set up Dirichlet BC vector: supg_g = T_bc at boundary, 0 elsewhere.
                    linalg::assign( supg_g, 0.0 );
                    {
                        auto g_grid = supg_g.grid_data();
                        auto mask = boundary_mask_data[velocity_level];
                        const ScalarType T_cmb_val     = static_cast< ScalarType >( prm.boundary_conditions_parameters.temperature_cmb );
                        const ScalarType T_surface_val = static_cast< ScalarType >( prm.boundary_conditions_parameters.temperature_surface );
                        Kokkos::parallel_for(
                            "supg_dirichlet_g",
                            local_domain_md_range_policy_nodes( domains[velocity_level] ),
                            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                                const auto flag = mask( sd, x, y, r );
                                if ( flag == grid::shell::ShellBoundaryFlag::CMB )
                                    g_grid( sd, x, y, r ) = T_cmb_val;
                                else if ( flag == grid::shell::ShellBoundaryFlag::SURFACE )
                                    g_grid( sd, x, y, r ) = T_surface_val;
                            } );
                        Kokkos::fence();
                    }

                    // Eliminate Dirichlet BCs from RHS.
                    fe::strong_algebraic_dirichlet_enforcement_poisson_like(
                        *supg_A_neumann,
                        *supg_A_neumann_diag,
                        supg_g,
                        supg_tmp,
                        q,
                        boundary_mask_data[velocity_level],
                        grid::shell::ShellBoundaryFlag::BOUNDARY );

                    // Solve (M + dt*A) T^{n+1} = q.
                    solve( *supg_solver, *supg_A, T, q );

                    if ( picard == num_picard - 1 )
                    {
                        table->query_rows_equals( "tag", "fgmres_solver" ).print_pretty();
                    }
                    table->clear();
                }
            }
            else
            {
                // --- FCT explicit energy solve ---

                {
                    util::Timer timer_fct_substeps( "fct_substeps" );

                    for ( int i = 0; i < prm.time_stepping_parameters.energy_substeps; i++ )
                    {
                        logroot << "Solving energy (FCT, substep " << i << ") ..." << std::endl;

                        {
                            util::Timer timer_fct_source_step( "fct_explicit_step_updating_source_term" );
                            if ( prm.physics_parameters.constant_internal_heating )
                            {
                                linalg::assign( T_source, prm.physics_parameters.constant_internal_heating_value );
                            }
                            timer_fct_source_step.stop();

                            util::Timer timer_fct_step( "fct_explicit_step" );
                            fv::hex::operators::fct_explicit_step(
                                domains[velocity_level],
                                T_fct,
                                u.block_1(),
                                fv_cell_centers.grid_data(),
                                coords_shell[velocity_level],
                                coords_radii[velocity_level],
                                dt,
                                fv_fct_bufs,
                                prm.physics_parameters.diffusivity,
                                T_source.grid_data(),
                                /*subtract_divergence=*/true,
                                boundary_mask_data[velocity_level],
                                fct_bcs );
                            timer_fct_step.stop();
                        }

                        // Enforce Dirichlet BCs on T^{n+1} after the full FCT step.
                        fv::hex::apply_dirichlet_bcs(
                            T_fct, boundary_mask_data[velocity_level], fct_bcs, domains[velocity_level] );
                    }

                    timer_fct_substeps.stop();
                }

                // Project T_fct → Q1 T once after all substeps.
                {
                    util::Timer timer_fct_projection( "fct_l2_projection" );
                    fv::hex::l2_project_fv_to_fe(
                        T,
                        T_fct,
                        domains[velocity_level],
                        coords_shell[velocity_level],
                        coords_radii[velocity_level],
                        l2_proj_tmps );

                    // Enforce Dirichlet BCs on the Q1 temperature field after the L2 projection.
                    {
                        auto T_grid = T.grid_data();
                        auto mask = boundary_mask_data[velocity_level];
                        const ScalarType T_cmb_val     = static_cast< ScalarType >( prm.boundary_conditions_parameters.temperature_cmb );
                        const ScalarType T_surface_val = static_cast< ScalarType >( prm.boundary_conditions_parameters.temperature_surface );
                        Kokkos::parallel_for(
                            "enforce_T_dirichlet_bcs",
                            local_domain_md_range_policy_nodes( domains[velocity_level] ),
                            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                                const auto flag = mask( sd, x, y, r );
                                if ( flag == grid::shell::ShellBoundaryFlag::CMB )
                                    T_grid( sd, x, y, r ) = T_cmb_val;
                                else if ( flag == grid::shell::ShellBoundaryFlag::SURFACE )
                                    T_grid( sd, x, y, r ) = T_surface_val;
                            } );
                        Kokkos::fence();
                }

                timer_fct_projection.stop();
                }
            } // end FCT else

            timer_energy.stop();

            // Update viscosity from the new temperature field.
            if ( prm.physics_parameters.viscosity_parameters.law != ViscosityLaw::CONSTANT )
            {
                util::Timer timer_visc_update( "viscosity_update" );
                Kokkos::parallel_for(
                    "viscosity_from_temperature",
                    local_domain_md_range_policy_nodes( domains[velocity_level] ),
                    ViscosityFromTemperature{ prm.physics_parameters.viscosity_parameters.law,
                                             prm.physics_parameters.viscosity_parameters.rmu,
                                             eta[velocity_level].grid_data(),
                                             T.grid_data() } );
                Kokkos::fence();
                timer_visc_update.stop();
            }

        } // end Picard loop

        // Output stuff, logging etc.

        table->add_row( {} );

        const bool write_output = ( timestep % prm.io_parameters.output_frequency == 0 );

        if ( write_output && !prm.io_parameters.no_xdmf )
        {
            logroot << "Writing XDMF output ..." << std::endl;
            xdmf_output.write();
        }

        if ( write_output && !prm.io_parameters.no_radial_profiles )
        {
            logroot << "Writing radial profiles ..." << std::endl;
            compute_and_write_radial_profiles(
                T, subdomain_shell_idx, domains[velocity_level], prm.io_parameters, timestep );
            compute_and_write_radial_profiles(
                eta[velocity_level], subdomain_shell_idx, domains[velocity_level], prm.io_parameters, timestep );
            compute_and_write_velocity_radial_profiles(
                u.block_1(),
                coords_shell[velocity_level],
                subdomain_shell_idx,
                domains[velocity_level],
                ownership_mask_data[velocity_level],
                prm.io_parameters,
                timestep );
        }

        // Compute Nusselt number at the surface.
        if ( timestep % 10 == 0 )
        {
            const auto Nu_top = compute_nusselt(
                domains[velocity_level],
                T,
                T_ref,
                coords_shell[velocity_level],
                coords_radii[velocity_level],
                boundary_mask_data[velocity_level],
                ownership_mask_data[velocity_level],
                /*at_surface=*/true );
            const auto Nu_top_fv = compute_nusselt_fv(
                domains[velocity_level],
                T_fct,
                boundary_mask_data[velocity_level],
                prm.boundary_conditions_parameters.temperature_surface,
                prm.boundary_conditions_parameters.temperature_cmb,
                prm.mesh_parameters.radius_min,
                prm.mesh_parameters.radius_max,
                /*at_surface=*/true );
            logroot << "Nu_top (Q1) = " << Nu_top << ", Nu_top (FV) = " << Nu_top_fv << std::endl;
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
