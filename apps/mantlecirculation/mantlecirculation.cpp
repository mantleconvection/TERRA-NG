#include <fstream>
#include <vector>

#include "communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "geophysics/viscosity/viscosity_interpolation.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "io/xdmf.hpp"
#include "kernels/common/grid_operations.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/diagonally_scaled_operator.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/diagonal_solver.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/power_iteration.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
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

struct InitialConditionInterpolator
{
    ScalarType                                         r_min_;
    ScalarType                                         r_max_;
    Grid3DDataVec< ScalarType, 3 >                     grid_;
    Grid2DDataScalar< ScalarType >                     radii_;
    Grid4DDataScalar< ScalarType >                     data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_data_;
    bool                                               only_boundary_;

    InitialConditionInterpolator(
        const ScalarType                                          r_min,
        const ScalarType                                          r_max,
        const Grid3DDataVec< ScalarType, 3 >&                     grid,
        const Grid2DDataScalar< ScalarType >&                     radii,
        const Grid4DDataScalar< ScalarType >&                     data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask_data,
        bool                                                      only_boundary )
    : r_min_( r_min )
    , r_max_( r_max )
    , grid_( grid )
    , radii_( radii )
    , data_( data )
    , mask_data_( mask_data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto mask_value  = mask_data_( local_subdomain_id, x, y, r );
        const auto is_boundary = util::has_flag( mask_value, grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || is_boundary )
        {
            const dense::Vec< ScalarType, 3 > coords =
                grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
            const auto frac                      = ( r_max_ - coords.norm() ) / ( r_max_ - r_min_ );
            data_( local_subdomain_id, x, y, r ) = Kokkos::pow( frac, 5 );
        }
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_u_;
    Grid4DDataScalar< ScalarType > data_T_;
    ScalarType                     rayleigh_number_;

    RHSVelocityInterpolator(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataVec< ScalarType, 3 >& data_u,
        const Grid4DDataScalar< ScalarType >& data_T,
        ScalarType                            rayleigh_number )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , data_T_( data_T )
    , rayleigh_number_( rayleigh_number )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const auto n = coords.normalized();

        for ( int d = 0; d < 3; d++ )
        {
            data_u_( local_subdomain_id, x, y, r, d ) =
                rayleigh_number_ * n( d ) * data_T_( local_subdomain_id, x, y, r );
        }
    }
};

struct NoiseAdder
{
    Grid3DDataVec< ScalarType, 3 >              grid_;
    Grid2DDataScalar< ScalarType >              radii_;
    Grid4DDataScalar< ScalarType >              data_T_;
    Grid4DDataScalar< grid::NodeOwnershipFlag > mask_;
    Kokkos::Random_XorShift64_Pool<>            rand_pool_;

    NoiseAdder(
        const Grid3DDataVec< ScalarType, 3 >&              grid,
        const Grid2DDataScalar< ScalarType >&              radii,
        const Grid4DDataScalar< ScalarType >&              data_T,
        const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask )
    : grid_( grid )
    , radii_( radii )
    , data_T_( data_T )
    , mask_( mask )
    , rand_pool_( 12345 )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        auto generator = rand_pool_.get_state();

        const ScalarType eps          = 1e-1;
        const auto       perturbation = eps * ( 2.0 * generator.drand() - 1.0 );

        const auto process_ownes_point =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::NodeOwnershipFlag::OWNED );

        if ( process_ownes_point )
        {
            data_T_( local_subdomain_id, x, y, r ) =
                Kokkos::clamp( data_T_( local_subdomain_id, x, y, r ) + perturbation, 0.0, 1.0 );
        }
        else
        {
            data_T_( local_subdomain_id, x, y, r ) = 0.0;
        }

        rand_pool_.free_state( generator );
    }
};

Result<> run( const Parameters& prm )
{
    auto table = std::make_shared< util::Table >();

    if ( const auto create_directories_result = create_directories( prm.io_parameters );
         create_directories_result.is_err() )
    {
        return create_directories_result.error();
    }

    // Set up domains for all levels.
    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< ScalarType, 3 > >                     coords_shell;
    std::vector< Grid2DDataScalar< ScalarType > >                     coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        ownership_mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    for ( int level = prm.mesh_parameters.refinement_level_mesh_min;
          level <= prm.mesh_parameters.refinement_level_mesh_max;
          level++ )
    {
        const int idx = level - prm.mesh_parameters.refinement_level_mesh_min;

        domains.push_back(
            DistributedDomain::create_uniform(
                level,
                level,
                prm.mesh_parameters.radius_min,
                prm.mesh_parameters.radius_max,
                prm.mesh_parameters.refinement_level_subdomains,
                prm.mesh_parameters.refinement_level_subdomains ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        ownership_mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const auto subdomain_distr = grid::shell::subdomain_distribution( domains.back() );
    logroot << "Subdomain distribution: \n";
    logroot << " - total: " << subdomain_distr.total << "\n";
    logroot << " - min:   " << subdomain_distr.min << "\n";
    logroot << " - avg:   " << subdomain_distr.avg << "\n";
    logroot << " - max:   " << subdomain_distr.max << "\n\n";

    const int  num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    Grid2DDataScalar< int > subdomain_shell_idx = grid::shell::subdomain_shell_idx( domains[velocity_level] );

    // Set up Stokes vectors for the finest grid.

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    std::vector< std::string >                             stok_vec_names = { "u", "f", "tmp" };

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
        // Temp dep. visc. not yet implemented.
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

    // determine AGCA elements
    VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", domains[0], ownership_mask_data[0] );
    int                          gca = 1;
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

    // Set up tmp vecs for FGMRES

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

    // Set up tmp vecs for multigrid preconditioner.

    std::vector< VectorQ1Vec< ScalarType > > tmp_mg;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_r;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_e;

    for ( int level = 0; level < num_levels; level++ )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], ownership_mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], ownership_mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], ownership_mask_data[level] );
        }
    }

    // Set up temperature and viscosity vectors.

    std::map< std::string, VectorQ1Scalar< ScalarType > > temp_vecs;
    std::vector< std::string >                            temp_vec_names = { "T", "q" };
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

    // Counting DoFs.

    const auto num_dofs_temperature =
        kernels::common::count_masked< long >( ownership_mask_data[num_levels - 1], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_velocity = 3 * num_dofs_temperature;
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( ownership_mask_data[num_levels - 2], grid::NodeOwnershipFlag::OWNED );

    // Set up operators.

    using Stokes      = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    using Gradient = Stokes::Block12Type;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        eta[velocity_level].grid_data(),
        true,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        eta[velocity_level].grid_data(),
        false,
        false );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // Multigrid operators

    logroot << "Setting up Stokes solver and preconditioners ..." << std::endl;

    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;

    for ( int level = 0; level < num_levels - 1; level++ )
    {
        A_c.emplace_back(
            domains[level],
            coords_shell[level],
            coords_radii[level],
            boundary_mask_data[level],
            eta[level].grid_data(),
            true,
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

    // GCA
    if ( gca > 0 )
    {
        for ( int level = num_levels - 2; level >= 0; level-- )
        {
            logroot << "Assembling GCA on level " << level << std::endl;

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

    using Smoother = linalg::solvers::Jacobi< Viscous >;

    std::vector< Smoother > smoothers;
    smoothers.reserve( num_levels );

    // Estimate relaxation rates on every level.
    logroot << "Estimating ralaxation rates for Jacobi smoother for the viscous block on each level." << std::endl;
    for ( int level = 0; level < num_levels; level++ )
    {
        VectorQ1Vec< ScalarType > tmp_pi_0(
            "tmp_pi_0" + std::to_string( level ), domains[level], ownership_mask_data[level] );
        VectorQ1Vec< ScalarType > tmp_pi_1(
            "tmp_pi_1" + std::to_string( level ), domains[level], ownership_mask_data[level] );
        double max_ev = 0.0;
        if ( level == num_levels - 1 )
        {
            linalg::DiagonallyScaledOperator inv_diag_A( K.block_11(), inverse_diagonals[level] );
            max_ev = linalg::solvers::power_iteration(
                inv_diag_A, tmp_pi_0, tmp_pi_1, prm.stokes_solver_parameters.viscous_pc_num_power_iterations );
        }
        else
        {
            linalg::DiagonallyScaledOperator inv_diag_A( A_c[level], inverse_diagonals[level] );
            max_ev = linalg::solvers::power_iteration(
                inv_diag_A, tmp_pi_0, tmp_pi_1, prm.stokes_solver_parameters.viscous_pc_num_power_iterations );
        }
        const auto omega_opt = 2.0 / ( 1.1 * max_ev );

        logroot << " + level " << level << ": " << omega_opt << std::endl;

        smoothers.emplace_back(
            inverse_diagonals[level],
            prm.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost,
            tmp_mg[level],
            omega_opt );
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

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;

    constexpr auto mass_scaling = 1.0;

    AD A(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        u.block_1(),
        prm.physics_parameters.diffusivity,
        0.0,
        true,
        false,
        mass_scaling );

    AD A_neumann(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        u.block_1(),
        prm.physics_parameters.diffusivity,
        0.0,
        false,
        false,
        mass_scaling );

    AD A_neumann_diag(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        u.block_1(),
        prm.physics_parameters.diffusivity,
        0.0,
        false,
        true,
        mass_scaling );

    using TempMass = fe::wedge::operators::shell::Mass< ScalarType >;

    TempMass M_T( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // Set up the initial temperature.

    Kokkos::parallel_for(
        "initial temp interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        InitialConditionInterpolator(
            domains[velocity_level].domain_info().radii().front(),
            domains[velocity_level].domain_info().radii().back(),
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            T.grid_data(),
            boundary_mask_data[velocity_level],
            false ) );

    Kokkos::fence();

    Kokkos::parallel_for(
        "adding noise to temp",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        NoiseAdder(
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            T.grid_data(),
            ownership_mask_data[velocity_level] ) );

    communication::shell::send_recv(
        domains[velocity_level], T.grid_data(), communication::CommunicationReduction::SUM );

    const auto num_energy_fgmres_tmps = 2 * prm.energy_solver_parameters.krylov_restart + 4;

    std::vector< VectorQ1Scalar< ScalarType > > energy_tmp_fgmres;
    energy_tmp_fgmres.reserve( num_energy_fgmres_tmps );
    for ( int i = 0; i < num_energy_fgmres_tmps; i++ )
    {
        energy_tmp_fgmres.emplace_back(
            "energy_tmp_fgmres", domains[velocity_level], ownership_mask_data[velocity_level] );
    }

    linalg::solvers::FGMRES< AD > energy_solver(
        energy_tmp_fgmres,
        { .restart                     = prm.energy_solver_parameters.krylov_restart,
          .relative_residual_tolerance = prm.energy_solver_parameters.krylov_relative_tolerance,
          .absolute_residual_tolerance = prm.energy_solver_parameters.krylov_absolute_tolerance,
          .max_iterations              = prm.energy_solver_parameters.krylov_max_iterations },
        table );
    energy_solver.set_tag( "energy_fgmres" );

    table->add_row( {
        { "tag", "setup" },
        { "dofs_velocity", num_dofs_velocity },
        { "dofs_temperature", num_dofs_temperature },
        { "dofs_pressure", num_dofs_pressure },
        { "level_velocity", velocity_level },
        { "level_pressure", pressure_level },
    } );

    table->print_pretty();
    table->clear();

    io::XDMFOutput xdmf_output(
        prm.io_parameters.outdir + "/" + prm.io_parameters.xdmf_dir,
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level] );

    xdmf_output.add( T.grid_data() );
    xdmf_output.add( u.block_1().grid_data() );
    xdmf_output.add( eta[velocity_level].grid_data() );

    logroot << "Writing initial XDMF ..." << std::endl;

    xdmf_output.write();

    logroot << "Writing initial radial profiles ..." << std::endl;

    compute_and_write_radial_profiles( T, subdomain_shell_idx, domains[velocity_level], prm.io_parameters, 0 );
    compute_and_write_radial_profiles(
        eta[velocity_level], subdomain_shell_idx, domains[velocity_level], prm.io_parameters, 0 );

    ScalarType simulated_time = 0.0;

    // We need some global h. Let's, for simplicity (does not need to be too accurate) just choose the smallest h in
    // radial direction.
    const auto h = grid::shell::min_radial_h( domains[velocity_level].domain_info().radii() );

    // Time stepping

    logroot << "Starting time stepping!" << std::endl;

    for ( int timestep = 1; timestep < prm.time_stepping_parameters.max_timesteps; timestep++ )
    {
        logroot << "\n### Timestep " << timestep << " ###" << std::endl;

        // Set up rhs data for Stokes.

        util::Timer timer_stokes( "stokes" );

        logroot << "Setting up Stokes rhs ..." << std::endl;

        Kokkos::parallel_for(
            "Stokes rhs interpolation",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            RHSVelocityInterpolator(
                coords_shell[velocity_level],
                coords_radii[velocity_level],
                stok_vecs["tmp"].block_1().grid_data(),
                T.grid_data(),
                prm.physics_parameters.rayleigh_number ) );

        linalg::apply( M, stok_vecs["tmp"].block_1(), stok_vecs["f"].block_1() );

        fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
            stok_vecs["f"], boundary_mask_data[velocity_level], grid::shell::ShellBoundaryFlag::BOUNDARY );

        logroot << "Solving Stokes ..." << std::endl;

        // Solve Stokes.
        solve( stokes_fgmres, K, u, f );

        if ( true )
        {
            table->query_rows_equals( "tag", "stokes_fgmres" ).print_pretty();
        }
        else
        {
            const auto num_stokes_iterations =
                table->query_rows_equals( "tag", "stokes_fgmres" ).column_as_vector< int >( "iteration" ).size();
            table->query_rows_equals( "tag", "stokes_fgmres" )
                .query_rows_where(
                    "iteration",
                    [num_stokes_iterations]( const util::Table::Value& v ) {
                        return std::get< int >( v ) == 0 || std::get< int >( v ) == num_stokes_iterations - 1;
                    } )
                .print_pretty();
        }

        table->clear();

        // "Normalize" pressure.
        const ScalarType avg_pressure_approximation =
            kernels::common::masked_sum(
                u.block_2().grid_data(), u.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
            static_cast< ScalarType >( num_dofs_pressure );
        linalg::lincomb( u.block_2(), { 1.0 }, { u.block_2() }, -avg_pressure_approximation );

        timer_stokes.stop();

        util::Timer timer_energy( "energy" );

        logroot << "Setting up energy solve ..." << std::endl;

        // Max velocity magnitude.
        const auto max_vel = kernels::common::max_vector_magnitude( u.block_1().grid_data() );

        // Choose "suitable" small dt for accuracy - we have and implicit time-stepping scheme so we do not really need
        // a CFL in the classical sense. Still useful for time-step size restriction.
        const auto dt_advection = h / max_vel;
        // const auto dt_diffusion = ( h * h ) / prm.diffusivity;
        // const auto dt           = prm.pseudo_cfl * std::min( dt_advection, dt_diffusion );
        const auto dt = prm.time_stepping_parameters.pseudo_cfl * dt_advection;

        logroot << "Computing dt ..." << std::endl;
        logroot << "    max_vel: " << max_vel << std::endl;
        logroot << "    h:       " << h << std::endl;
        logroot << "=>  dt:      " << dt << std::endl;

        A.dt()              = dt;
        A_neumann.dt()      = dt;
        A_neumann_diag.dt() = dt;

        for ( int i = 0; i < prm.time_stepping_parameters.energy_substeps; i++ )
        {
            // Prepping for implicit Euler step.
            linalg::apply( M_T, T, q );

            // Set up the temperature boundary.
            assign( temp_vecs["tmp_0"], 0.0 );
            Kokkos::parallel_for(
                "boundary temp interpolation",
                local_domain_md_range_policy_nodes( domains[velocity_level] ),
                InitialConditionInterpolator(
                    domains[velocity_level].domain_info().radii().front(),
                    domains[velocity_level].domain_info().radii().back(),
                    coords_shell[velocity_level],
                    coords_radii[velocity_level],
                    temp_vecs["tmp_0"].grid_data(),
                    boundary_mask_data[velocity_level],
                    true ) );

            Kokkos::fence();

            fe::strong_algebraic_dirichlet_enforcement_poisson_like(
                A_neumann,
                A_neumann_diag,
                temp_vecs["tmp_0"],
                temp_vecs["tmp_1"],
                q,
                boundary_mask_data[velocity_level],
                grid::shell::ShellBoundaryFlag::BOUNDARY );

            logroot << "Solving energy ..." << std::endl;

            // Solve energy.
            solve( energy_solver, A, T, q );

            if ( true )
            {
                table->query_rows_equals( "tag", "energy_fgmres" ).print_pretty();
            }
            else
            {
                const auto num_energy_iterations =
                    table->query_rows_equals( "tag", "energy_fgmres" ).column_as_vector< int >( "iteration" ).size();
                table->query_rows_equals( "tag", "energy_fgmres" )
                    .query_rows_where(
                        "iteration",
                        [num_energy_iterations]( const util::Table::Value& v ) {
                            return std::get< int >( v ) == 0 || std::get< int >( v ) == num_energy_iterations - 1;
                        } )
                    .print_pretty();
            }

            table->clear();
        }

        timer_energy.stop();

        // Output stuff, logging etc.

        table->add_row( {} );

        logroot << "Writing XDMF output and radial profiles ..." << std::endl;

        xdmf_output.write();

        compute_and_write_radial_profiles(
            T, subdomain_shell_idx, domains[velocity_level], prm.io_parameters, timestep );
        compute_and_write_radial_profiles(
            eta[velocity_level], subdomain_shell_idx, domains[velocity_level], prm.io_parameters, timestep );

        simulated_time += prm.time_stepping_parameters.energy_substeps * dt;

        logroot << "Simulated time: " << simulated_time << " (stopping at " << prm.time_stepping_parameters.t_end
                << ", we're at " << simulated_time / prm.time_stepping_parameters.t_end * 100.0 << "%)" << std::endl;

        write_timer_tree( prm.io_parameters, timestep );

        if ( simulated_time >= prm.time_stepping_parameters.t_end )
        {
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
