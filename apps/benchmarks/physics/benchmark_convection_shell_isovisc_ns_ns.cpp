

#include <fstream>
#include <vector>

#include "communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/identity.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "io/xdmf.hpp"
#include "kernels/common/grid_operations.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pbicgstab.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "shell/radial_profiles.hpp"
#include "util/cli11_helper.hpp"
#include "util/cli11_wrapper.hpp"
#include "util/filesystem.hpp"
#include "util/info.hpp"
#include "util/init.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

using namespace terra;

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

using ScalarType = double;

struct Parameters
{
    int min_level;
    int max_level;

    ScalarType r_min;
    ScalarType r_max;

    ScalarType diffusivity;
    ScalarType rayleigh;

    ScalarType pseudo_cfl;
    ScalarType t_end;

    int max_timesteps;

    int substeps_per_stokes_solve;

    int        stokes_bicgstab_l;
    int        stokes_bicgstab_max_iterations;
    ScalarType stokes_bicgstab_relative_tolerance;
    ScalarType stokes_bicgstab_absolute_tolerance;

    int num_vcycles;
    int num_smoothing_steps_prepost;

    std::string outdir;
};

struct InitialConditionInterpolator
{
    Grid3DDataVec< ScalarType, 3 >                     grid_;
    Grid2DDataScalar< ScalarType >                     radii_;
    Grid4DDataScalar< ScalarType >                     data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_data_;
    bool                                               only_boundary_;

    InitialConditionInterpolator(
        const Grid3DDataVec< ScalarType, 3 >&                     grid,
        const Grid2DDataScalar< ScalarType >&                     radii,
        const Grid4DDataScalar< ScalarType >&                     data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_data,
        bool                                                      only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , boundary_mask_data_( boundary_mask_data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto mask_value  = boundary_mask_data_( local_subdomain_id, x, y, r );
        const auto is_boundary = util::has_flag( mask_value, grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || is_boundary )
        {
            const dense::Vec< ScalarType, 3 > coords =
                grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
            data_( local_subdomain_id, x, y, r ) = Kokkos::pow( 2.0 * ( 1.0 - coords.norm() ), 5 );
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
    Grid4DDataScalar< grid::NodeOwnershipFlag > ownership_mask_data_;
    Kokkos::Random_XorShift64_Pool<>            rand_pool_;

    NoiseAdder(
        const Grid3DDataVec< ScalarType, 3 >&              grid,
        const Grid2DDataScalar< ScalarType >&              radii,
        const Grid4DDataScalar< ScalarType >&              data_T,
        const Grid4DDataScalar< grid::NodeOwnershipFlag >& ownership_mask_data )
    : grid_( grid )
    , radii_( radii )
    , data_T_( data_T )
    , ownership_mask_data_( ownership_mask_data )
    , rand_pool_( 12345 )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        auto generator = rand_pool_.get_state();

        const ScalarType eps          = 1e-1;
        const auto       perturbation = eps * ( 2.0 * generator.drand() - 1.0 );

        const auto process_ownes_point =
            util::has_flag( ownership_mask_data_( local_subdomain_id, x, y, r ), grid::NodeOwnershipFlag::OWNED );

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

void run( const Parameters& prm, const std::shared_ptr< util::Table >& table )
{
    // Check outdir and create if it does not exist.
    util::prepare_empty_directory_or_abort( prm.outdir );

    const auto xdmf_dir            = prm.outdir + "/xdmf";
    const auto radial_profiles_dir = prm.outdir + "/radial_profiles";
    const auto timer_trees_dir     = prm.outdir + "/timer_trees";

    util::prepare_empty_directory_or_abort( xdmf_dir );
    util::prepare_empty_directory_or_abort( radial_profiles_dir );
    util::prepare_empty_directory_or_abort( timer_trees_dir );

    // Set up domains for all levels.
    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< ScalarType, 3 > >                     coords_shell;
    std::vector< Grid2DDataScalar< ScalarType > >                     coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        ownership_mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    for ( int level = prm.min_level; level <= prm.max_level; level++ )
    {
        const int idx = level - prm.min_level;

        domains.push_back(
            DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, prm.r_min, prm.r_max ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        ownership_mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const auto num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    // Set up Stokes vectors for the finest grid.

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    std::vector< std::string >                             stok_vec_names = { "u", "f" };

    const auto num_stokes_bicgstab_tmps = 2 * ( prm.stokes_bicgstab_l + 1 ) + 2;

    const auto num_stok_tmps = num_stokes_bicgstab_tmps;

    for ( int i = 0; i < num_stok_tmps; i++ )
    {
        stok_vec_names.push_back( "tmp_" + std::to_string( i ) );
    }

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

    // Set up tmp vecs for multigrid.

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

    // Set up temperature vectors.

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

    using Stokes      = fe::wedge::operators::shell::Stokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        true,
        false );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // Multigrid operators

    std::vector< Viscous >      A_diag;
    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;

    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;

    for ( int level = 0; level < num_levels; level++ )
    {
        A_diag.emplace_back(
            domains[level], coords_shell[level], coords_radii[level], boundary_mask_data[level], true, true );

        inverse_diagonals.emplace_back(
            "inverse_diagonal_" + std::to_string( level ), domains[level], ownership_mask_data[level] );

        VectorQ1Vec< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( level ), domains[level], ownership_mask_data[level] );

        linalg::assign( tmp, 1.0 );
        linalg::apply( A_diag[level], tmp, inverse_diagonals.back() );
        linalg::invert_entries( inverse_diagonals.back() );

        if ( level < num_levels - 1 )
        {
            A_c.emplace_back(
                domains[level], coords_shell[level], coords_radii[level], boundary_mask_data[level], true, false );
            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // Set up solvers.

    // Multigrid preconditioner.

    using Smoother = linalg::solvers::Jacobi< Viscous >;

    std::vector< Smoother > smoothers;
    for ( int level = 0; level < num_levels; level++ )
    {
        constexpr auto omega = 0.666;
        smoothers.emplace_back( inverse_diagonals[level], prm.num_smoothing_steps_prepost, tmp_mg[level], omega );
    }

    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;

    std::vector< VectorQ1Vec< ScalarType > > coarse_grid_tmps;
    for ( int i = 0; i < 4; i++ )
    {
        coarse_grid_tmps.emplace_back( "tmp_coarse_grid", domains[0], ownership_mask_data[0] );
    }

    CoarseGridSolver coarse_grid_solver(
        linalg::solvers::IterativeSolverParameters{ 10, 1e-6, 1e-16 }, table, coarse_grid_tmps );

    using PrecVisc = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;
    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_grid_solver, prm.num_vcycles, 1e-6 );

    using PrecSchur = linalg::solvers::IdentitySolver< Stokes::Block22Type >;
    PrecSchur prec_22;

    using PrecStokes =
        linalg::solvers::BlockDiagonalPreconditioner2x2< Stokes, Viscous, Stokes::Block22Type, PrecVisc, PrecSchur >;
    PrecStokes prec_stokes( K.block_11(), K.block_22(), prec_11, prec_22 );

    linalg::solvers::IterativeSolverParameters solver_params{
        prm.stokes_bicgstab_max_iterations,
        prm.stokes_bicgstab_relative_tolerance,
        prm.stokes_bicgstab_absolute_tolerance };

    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_bicgstab( num_stokes_bicgstab_tmps );
    for ( int i = 0; i < num_stokes_bicgstab_tmps; i++ )
    {
        tmp_bicgstab[i] = stok_vecs["tmp_" + std::to_string( i )];
    }

    linalg::solvers::PBiCGStab< Stokes, PrecStokes > pbicgstab(
        prm.stokes_bicgstab_l, solver_params, table, tmp_bicgstab, prec_stokes );

    /////////////////////
    /// ENERGY SOLVER ///
    /////////////////////

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;

    constexpr auto mass_scaling = 1.0;

    AD A(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        u.block_1(),
        prm.diffusivity,
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
        prm.diffusivity,
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
        prm.diffusivity,
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

    const auto                                  num_temp_tmps_energy = 14;
    std::vector< VectorQ1Scalar< ScalarType > > tmp_gmres( num_temp_tmps_energy );
    for ( int i = 0; i < num_temp_tmps_energy; i++ )
    {
        tmp_gmres[i] = VectorQ1Scalar< ScalarType >(
            "tmp_energy_gmres", domains[velocity_level], ownership_mask_data[velocity_level] );
    }

    linalg::solvers::FGMRES< AD > energy_solver(
        tmp_gmres,
        { .restart                     = 5,
          .relative_residual_tolerance = 1e-6,
          .absolute_residual_tolerance = 1e-12,
          .max_iterations              = 100 },
        table );

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
        xdmf_dir, domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level] );

    xdmf_output.add( T.grid_data() );
    xdmf_output.add( u.block_1().grid_data() );

    // Time stepping

    xdmf_output.write();

    auto shell_idx         = terra::grid::shell::subdomain_shell_idx( domains[velocity_level] );
    auto num_global_shells = static_cast< int >( domains[velocity_level].domain_info().radii().size() );
    auto profiles          = shell::radial_profiles_to_table< ScalarType >(
        shell::radial_profiles( T, shell_idx, num_global_shells ), domains[velocity_level].domain_info().radii() );
    std::ofstream out( radial_profiles_dir + "/radial_profiles_" + std::to_string( 0 ) + ".csv" );
    profiles.print_csv( out );

    ScalarType simulated_time = 0.0;

    // We need some global h. Let's, for simplicity (does not need to be too accurate) just choose the smallest h in
    // radial direction.
    const auto h = grid::shell::min_radial_h( domains[velocity_level].domain_info().radii() );

    for ( int timestep = 1; timestep < prm.max_timesteps; timestep++ )
    {
        if ( mpi::rank() == 0 )
        {
            std::cout << "Timestep " << timestep << std::endl;
        }

        // Set up rhs data for Stokes.

        util::Timer timer_stokes( "stokes" );

        if ( mpi::rank() == 0 )
        {
            std::cout << "Setting up Stokes rhs ..." << std::endl;
        }

        Kokkos::parallel_for(
            "Stokes rhs interpolation",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            RHSVelocityInterpolator(
                coords_shell[velocity_level],
                coords_radii[velocity_level],
                stok_vecs["tmp_1"].block_1().grid_data(),
                T.grid_data(),
                prm.rayleigh ) );

        linalg::apply( M, stok_vecs["tmp_1"].block_1(), stok_vecs["f"].block_1() );

        fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
            stok_vecs["f"], boundary_mask_data[velocity_level], grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( mpi::rank() == 0 )
        {
            std::cout << "Solving Stokes ..." << std::endl;
        }

        // Solve Stokes.
        solve( pbicgstab, K, u, f );

        table->query_rows_equals( "tag", "pbicgstab_solver" ).print_pretty();
        table->clear();

        // "Normalize" pressure.
        const ScalarType avg_pressure_approximation =
            kernels::common::masked_sum(
                u.block_2().grid_data(), u.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
            static_cast< ScalarType >( num_dofs_pressure );
        linalg::lincomb( u.block_2(), { 1.0 }, { u.block_2() }, -avg_pressure_approximation );

        timer_stokes.stop();

        util::Timer timer_energy( "energy" );

        if ( mpi::rank() == 0 )
        {
            std::cout << "Setting up energy solve ..." << std::endl;
        }

        // Max velocity magnitude.
        const auto max_vel = kernels::common::max_vector_magnitude( u.block_1().grid_data() );

        // Choose "suitable" small dt for accuracy - we have and implicit time-stepping scheme so we do not really need
        // a CFL in the classical sense. Still useful for time-step size restriction.
        const auto dt_advection = h / max_vel;
        // const auto dt_diffusion = ( h * h ) / prm.diffusivity;
        // const auto dt           = prm.pseudo_cfl * std::min( dt_advection, dt_diffusion );
        const auto dt = prm.pseudo_cfl * dt_advection;

        if ( mpi::rank() == 0 )
        {
            std::cout << "Computing dt ..." << std::endl;
            std::cout << "    max_vel: " << max_vel << std::endl;
            std::cout << "    h:       " << h << std::endl;
            std::cout << "=>  dt:      " << dt << std::endl;
        }

        A.dt()              = dt;
        A_neumann.dt()      = dt;
        A_neumann_diag.dt() = dt;

        for ( int i = 0; i < prm.substeps_per_stokes_solve; i++ )
        {
            // Prepping for implicit Euler step.
            linalg::apply( M_T, T, q );

            // Set up the temperature boundary.
            assign( temp_vecs["tmp_0"], 0.0 );
            Kokkos::parallel_for(
                "boundary temp interpolation",
                local_domain_md_range_policy_nodes( domains[velocity_level] ),
                InitialConditionInterpolator(
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

            if ( mpi::rank() == 0 )
            {
                std::cout << "Solving energy ..." << std::endl;
            }

            // Solve energy.
            solve( energy_solver, A, T, q );

            table->query_rows_equals( "tag", "fgmres_solver" ).print_pretty();
            table->clear();
        }

        timer_energy.stop();

        // Output stuff, logging etc.

        table->add_row( {} );

        {
            if ( mpi::rank() == 0 )
            {
                std::cout << "Writing XDMF output and radial profiles ..." << std::endl;
            }

            xdmf_output.write();
            profiles = shell::radial_profiles_to_table(
                shell::radial_profiles( T, shell_idx, num_global_shells ),
                domains[velocity_level].domain_info().radii() );
            std::ofstream out( radial_profiles_dir + "/radial_profiles_" + std::to_string( timestep ) + ".csv" );
            profiles.print_csv( out );
        }

        simulated_time += prm.substeps_per_stokes_solve * dt;
        if ( mpi::rank() == 0 )
        {
            std::cout << "Simulated time: " << simulated_time << " (stopping at " << prm.t_end << ", we're at "
                      << simulated_time / prm.t_end * 100.0 << "%)" << std::endl;
        }

        util::TimerTree::instance().aggregate_mpi();
        if ( mpi::rank() == 0 )
        {
            const auto timer_tree_file = timer_trees_dir + "/timer_tree_" + std::to_string( timestep ) + ".json";
            std::cout << "Writing timer tree to " << timer_tree_file << std::endl;
            std::ofstream out( timer_tree_file );
            out << util::TimerTree::instance().json_aggregate();
            out.close();
        }

        if ( simulated_time >= prm.t_end )
        {
            break;
        }
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    // Fill with default parameters.
    Parameters parameters{
        .min_level                          = 0,
        .max_level                          = 3,
        .r_min                              = 0.5,
        .r_max                              = 1.0,
        .diffusivity                        = 1.0,
        .rayleigh                           = 1e5,
        .pseudo_cfl                         = 0.5,
        .t_end                              = 1.0,
        .max_timesteps                      = 1000,
        .substeps_per_stokes_solve          = 1,
        .stokes_bicgstab_l                  = 2,
        .stokes_bicgstab_max_iterations     = 10,
        .stokes_bicgstab_relative_tolerance = 1e-6,
        .stokes_bicgstab_absolute_tolerance = 1e-12,
        .num_vcycles                        = 2,
        .num_smoothing_steps_prepost        = 2,
        .outdir                             = "" };

    CLI::App app{ "Isoviscous convection benchmark." };

    // We can just reference the struct entries.
    // The values therein are defaults.
    util::add_option_with_default( app, "--min-level", parameters.min_level );
    util::add_option_with_default( app, "--max-level", parameters.max_level );
    util::add_option_with_default( app, "--r-min", parameters.r_min );
    util::add_option_with_default( app, "--r-max", parameters.r_max );
    util::add_option_with_default( app, "--diffusivity", parameters.diffusivity );
    util::add_option_with_default( app, "--rayleigh", parameters.rayleigh );
    util::add_option_with_default( app, "--pseudo-cfl", parameters.pseudo_cfl );
    util::add_option_with_default( app, "--t-end", parameters.t_end );
    util::add_option_with_default( app, "--max-timesteps", parameters.max_timesteps );
    util::add_option_with_default( app, "--substeps-per-stokes-solve", parameters.substeps_per_stokes_solve );
    util::add_option_with_default( app, "--stokes-bicgstab-l", parameters.stokes_bicgstab_l );
    util::add_option_with_default( app, "--stokes-bicgstab-max-iterations", parameters.stokes_bicgstab_max_iterations );
    util::add_option_with_default(
        app, "--stokes-bicgstab-relative-tolerance", parameters.stokes_bicgstab_relative_tolerance );
    util::add_option_with_default(
        app, "--stokes-bicgstab-absolute-tolerance", parameters.stokes_bicgstab_absolute_tolerance );
    util::add_option_with_default( app, "--num-vcycles", parameters.num_vcycles );
    util::add_option_with_default( app, "--num-smoothing-steps-prepost", parameters.num_smoothing_steps_prepost );
    util::add_option_with_default( app, "--outdir", parameters.outdir )->required();

    CLI11_PARSE( app, argc, argv );

    util::print_general_info( argc, argv, std::cout );

    terra::util::print_cli_summary( app, std::cout );

    const auto table = std::make_shared< util::Table >();

    run( parameters, table );

    return 0;
}
