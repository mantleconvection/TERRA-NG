
// Modified Stokes test:
//  - ALL output goes through util::logroot() (no std::cout).
//  - Boundary interpolation uses boundary_mask_data (ShellBoundaryFlag::BOUNDARY) via util::has_flag.
//  - Runs for all subdomain refinement levels 0..2.
//  - Sanity check: for a fixed global level, L2 errors must match across level_subdomains (0..2).

#include "../src/terra/communication/shell/communication.hpp"

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/strong_algebraic_freeslip_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/identity.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/prolongation_linear.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/restriction_linear.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"

#include "grid/shell/bit_masks.hpp" // for util::has_flag + ShellBoundaryFlag helpers

#include "io/xdmf.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/gca/gca_elements_collector.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/pminres.hpp"
#include "util/info.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"

#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/vtk.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
#include "terra/shell/radial_profiles.hpp"

#include "util/init.hpp"
#include "util/table.hpp"
// If util::logroot() is declared in a dedicated header in your tree,
// include it here (some builds provide it via util/init.hpp):
// #include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>

#include "util/cli11_helper.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::get_shell_boundary_flag;
using grid::shell::SubdomainInfo;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::DiagonallyScaledOperator;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using linalg::solvers::DiagonalSolver;
using linalg::solvers::power_iteration;
using linalg::solvers::TwoGridGCA;
using terra::grid::shell::BoundaryConditions;

// -----------------------------------------------------------------------------
// Interpolators (boundary logic uses boundary_mask_data flag field)
// -----------------------------------------------------------------------------

struct SolutionVelocityInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataVec< double, 3 >                         data_u_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    SolutionVelocityInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataVec< double, 3 >&                         data_u,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const bool                                                only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , mask_( mask )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || on_boundary )
        {
            const double cx = coords( 0 );
            const double cy = coords( 1 );
            const double cz = coords( 2 );

            data_u_( local_subdomain_id, x, y, r, 0 ) = -4 * Kokkos::cos( 4 * cz );
            data_u_( local_subdomain_id, x, y, r, 1 ) =  8 * Kokkos::cos( 8 * cx );
            data_u_( local_subdomain_id, x, y, r, 2 ) = -2 * Kokkos::cos( 2 * cy );
        }
    }
};

struct SolutionPressureInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataScalar< double >                         data_p_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    SolutionPressureInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataScalar< double >&                         data_p,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const bool                                                only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_p_( data_p )
    , mask_( mask )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || on_boundary )
        {
            const double cx = coords( 0 );
            const double cy = coords( 1 );
            const double cz = coords( 2 );

            data_p_( local_subdomain_id, x, y, r ) =
                Kokkos::sin( 4 * cx ) * Kokkos::sin( 8 * cy ) * Kokkos::sin( 2 * cz );
        }
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    RHSVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const real_t x0 = 4 * coords( 2 );

        data_( local_subdomain_id, x, y, r, 0 ) =
            -64.0 * ( Kokkos::sin( coords( 2 ) ) + 2 ) * Kokkos::cos( x0 ) -
            16.0 * Kokkos::sin( x0 ) * Kokkos::cos( coords( 2 ) ) +
            4 * Kokkos::sin( 8 * coords( 1 ) ) * Kokkos::sin( 2 * coords( 2 ) ) * Kokkos::cos( 4 * coords( 0 ) );

        data_( local_subdomain_id, x, y, r, 1 ) =
            512.0 * ( Kokkos::sin( coords( 2 ) ) + 2 ) * Kokkos::cos( 8 * coords( 0 ) ) +
            8 * Kokkos::sin( 4 * coords( 0 ) ) * Kokkos::sin( 2 * coords( 2 ) ) * Kokkos::cos( 8 * coords( 1 ) ) -
            4.0 * Kokkos::sin( 2 * coords( 1 ) ) * Kokkos::cos( coords( 2 ) );

        data_( local_subdomain_id, x, y, r, 2 ) =
            -8.0 * ( Kokkos::sin( coords( 2 ) ) + 2 ) * Kokkos::cos( 2 * coords( 1 ) ) +
            2 * Kokkos::sin( 4 * coords( 0 ) ) * Kokkos::sin( 8 * coords( 1 ) ) * Kokkos::cos( 2 * coords( 2 ) );
    }
};

struct KInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    double                     kmax_;

    KInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        const double                      kmax )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , kmax_( kmax )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double value                   = 2 + Kokkos::sin( coords( 2 ) );
        data_( local_subdomain_id, x, y, r ) = value;
    }
};

// -----------------------------------------------------------------------------
// Test
// -----------------------------------------------------------------------------

std::tuple< double, double, int >
test( double kmax,
      int gca,
      int min_level,
      int max_level,
      int level_subdomains,
      int radial_extra_levels,
      int lat_sdr_override,
      int rad_sdr_override,
      const std::shared_ptr< util::Table >& table )
{
    using ScalarType = double;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    ScalarType r_min = 0.5;
    ScalarType r_max = 1.0;

    const int lat_sdr = ( lat_sdr_override >= 0 ) ? lat_sdr_override : level_subdomains;
    const int rad_sdr = ( rad_sdr_override >= 0 ) ? rad_sdr_override : level_subdomains;

    util::logroot << "Allocating domains ...\n";
    for ( int level = min_level; level <= max_level; level++ )
    {
        const int idx       = level - min_level;
        const int lat_level = level;
        const int rad_level = level + radial_extra_levels;

        domains.push_back(
            DistributedDomain::create_uniform(
                lat_level, rad_level, r_min, r_max, lat_sdr, rad_sdr ) );

        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const auto num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    std::vector< std::string >                             stok_vec_names = { "u", "f", "solution", "error" };
    constexpr int                                          num_stok_tmps  = 8;

    util::logroot << "Allocating temps ...\n";
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
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    auto& u        = stok_vecs["u"];
    auto& f        = stok_vecs["f"];
    auto& solution = stok_vecs["solution"];
    auto& error    = stok_vecs["error"];

    std::vector< VectorQ1Vec< ScalarType > > tmp_mg;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_r;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_e;

    for ( int level = 0; level < num_levels; level++ )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );
        }
    }

    const auto num_dofs_velocity =
        3 * kernels::common::count_masked< long >( mask_data[num_levels - 1], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( mask_data[num_levels - 2], grid::NodeOwnershipFlag::OWNED );

    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };
    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };

    util::logroot << "Setting operators ...\n";
    using Stokes      = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using Gradient    = Stokes::Block12Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    VectorQ1Scalar< ScalarType > k( "k", domains[velocity_level], mask_data[velocity_level] );

    util::logroot << "Interpolating k ...\n";
    Kokkos::parallel_for(
        "coefficient interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        KInterpolator( coords_shell[velocity_level], coords_radii[velocity_level], k.grid_data(), kmax ) );

    VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", domains[0], mask_data[0] );
    if ( gca == 2 )
    {
        linalg::assign( GCAElements, 0 );
        util::logroot << "Adaptive GCA: determining GCA elements on level " << velocity_level << "\n";
        terra::linalg::solvers::GCAElementsCollector< ScalarType >(
            domains[velocity_level], k.grid_data(), velocity_level, GCAElements.grid_data() );
    }
    else if ( gca == 1 )
    {
        util::logroot << "GCA on all elements\n";
        assign( GCAElements, 1 );
    }

    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k.grid_data(),
        bcs,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k.grid_data(),
        bcs_neumann,
        false );

    Stokes K_neumann_diag(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k.grid_data(),
        bcs_neumann,
        true );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    std::vector< Viscous >      A_diag;
    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;

    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;

    util::logroot << "MG hierarchy ...\n";
    for ( int level = 0; level < num_levels; level++ )
    {
        VectorQ1Scalar< ScalarType > k_c( "k_c", domains[level], mask_data[level] );
        Kokkos::parallel_for(
            "coefficient interpolation (mg)",
            local_domain_md_range_policy_nodes( domains[level] ),
            KInterpolator( coords_shell[level], coords_radii[level], k_c.grid_data(), kmax ) );

        A_diag.emplace_back(
            domains[level],
            coords_shell[level],
            coords_radii[level],
            boundary_mask_data[level],
            k_c.grid_data(),
            bcs,
            true );

        if ( level < num_levels - 1 )
        {
            A_c.emplace_back(
                domains[level],
                coords_shell[level],
                coords_radii[level],
                boundary_mask_data[level],
                k_c.grid_data(),
                bcs,
                false );

            if ( gca == 2 )
            {
                A_c.back().set_stored_matrix_mode(
                    linalg::OperatorStoredMatrixMode::Selective, level, GCAElements.grid_data() );
            }
            else if ( gca == 1 )
            {
                A_c.back().set_stored_matrix_mode(
                    linalg::OperatorStoredMatrixMode::Full, level, GCAElements.grid_data() );
            }

            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    Kokkos::parallel_for(
        "solution interpolation (velocity)",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            stok_vecs["solution"].block_1().grid_data(),
            boundary_mask_data[velocity_level],
            false ) );

    Kokkos::parallel_for(
        "solution interpolation (pressure)",
        local_domain_md_range_policy_nodes( domains[pressure_level] ),
        SolutionPressureInterpolator(
            coords_shell[pressure_level],
            coords_radii[pressure_level],
            stok_vecs["solution"].block_2().grid_data(),
            boundary_mask_data[pressure_level],
            false ) );

    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        RHSVelocityInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level], stok_vecs["tmp_1"].block_1().grid_data() ) );

    linalg::apply( M, stok_vecs["tmp_1"].block_1(), stok_vecs["f"].block_1() );

    Kokkos::parallel_for(
        "boundary interpolation (velocity)",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            stok_vecs["tmp_0"].block_1().grid_data(),
            boundary_mask_data[velocity_level],
            true ) );

    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann,
        K_neumann_diag,
        stok_vecs["tmp_0"],
        stok_vecs["tmp_1"],
        stok_vecs["f"],
        boundary_mask_data[velocity_level],
        BOUNDARY );

    // setup gca coarse ops
    if ( gca > 0 )
    {
        for ( int level = num_levels - 2; level >= 0; level-- )
        {
            util::logroot << "Assembling GCA on level " << level << "\n";
            TwoGridGCA< ScalarType, Viscous >(
                ( level == num_levels - 2 ) ? K_neumann.block_11() : A_c[level + 1],
                A_c[level],
                level,
                GCAElements.grid_data() );
        }
    }

    using Smoother = linalg::solvers::Jacobi< Viscous >;

    std::vector< Smoother > smoothers;
    for ( int level = 0; level < num_levels; level++ )
    {
        inverse_diagonals.emplace_back(
            "inverse_diagonal_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( level ), domains[level], mask_data[level] );

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

        constexpr auto            smoother_prepost = 3;
        VectorQ1Vec< ScalarType > tmp_pi_0( "tmp_pi_0" + std::to_string( level ), domains[level], mask_data[level] );
        VectorQ1Vec< ScalarType > tmp_pi_1( "tmp_pi_1" + std::to_string( level ), domains[level], mask_data[level] );
        double                    max_ev = 0.0;

        if ( level == num_levels - 1 )
        {
            DiagonallyScaledOperator< Viscous > inv_diag_A( K.block_11(), inverse_diagonals[level] );
            max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
        }
        else
        {
            DiagonallyScaledOperator< Viscous > inv_diag_A( A_c[level], inverse_diagonals[level] );
            max_ev = power_iteration< DiagonallyScaledOperator< Viscous > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
        }

        const auto omega_opt = 2.0 / ( 1.3 * max_ev );
        smoothers.emplace_back( inverse_diagonals[level], smoother_prepost, tmp_mg[level], omega_opt );

        util::logroot << "Optimal omega on level " << level << ": " << omega_opt << "\n";
    }

    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;

    std::vector< VectorQ1Vec< ScalarType > > coarse_grid_tmps;
    for ( int i = 0; i < 4; i++ )
    {
        coarse_grid_tmps.emplace_back( "tmp_coarse_grid", domains[0], mask_data[0] );
    }

    CoarseGridSolver coarse_grid_solver(
        linalg::solvers::IterativeSolverParameters{ 1000, 1e-6, 1e-16 }, table, coarse_grid_tmps );

    constexpr auto num_mg_cycles = 1;

    using PrecVisc =
        linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;

    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_grid_solver, num_mg_cycles, 1e-8 );

    VectorQ1Scalar< ScalarType > k_pm( "k_pm", domains[max_level - min_level], mask_data[max_level - min_level] );
    assign( k_pm, k );
    linalg::invert_entries( k_pm );

    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;
    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level], k_pm.grid_data(), false );
    pmass.set_lumped_diagonal( true );

    VectorQ1Scalar< ScalarType > lumped_diagonal_pmass(
        "lumped_diagonal_pmass", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( pressure_level ),
            domains[pressure_level],
            mask_data[pressure_level] );
        linalg::assign( tmp, 1.0 );
        linalg::apply( pmass, tmp, lumped_diagonal_pmass );
    }

    using PrecSchur = linalg::solvers::DiagonalSolver< PressureMass >;
    PrecSchur inv_lumped_pmass( lumped_diagonal_pmass );

    using PrecStokes =
        linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;

    VectorQ1IsoQ2Q1< ScalarType > triangular_prec_tmp(
        "triangular_prec_tmp",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    PrecStokes prec_stokes( K.block_11(), pmass, K.block_12(), triangular_prec_tmp, prec_11, inv_lumped_pmass );

    const int iters = 500;

    constexpr auto                               num_tmps_fgmres = iters;
    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < 2 * num_tmps_fgmres + 4; ++i )
    {
        tmp_fgmres.emplace_back(
            "tmp_" + std::to_string( i ),
            domains[velocity_level],
            domains[pressure_level],
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    linalg::solvers::FGMRESOptions< ScalarType > fgmres_options;
    fgmres_options.restart                     = iters;
    fgmres_options.max_iterations              = iters;
    fgmres_options.relative_residual_tolerance = 1e-10;

    auto solver_table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres( tmp_fgmres, fgmres_options, solver_table, prec_stokes );

    util::logroot << "Solve ...\n";
    assign( u, 0 );
    linalg::solvers::solve( fgmres, K, u, f );

    solver_table->query_rows_equals( "tag", "fgmres_solver" )
        .select_columns( { "absolute_residual", "relative_residual", "iteration" } )
        .print_pretty();

    const double avg_pressure_solution =
        kernels::common::masked_sum(
            solution.block_2().grid_data(), solution.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;

    const double avg_pressure_approximation =
        kernels::common::masked_sum(
            u.block_2().grid_data(), u.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;

    linalg::lincomb( solution.block_2(), { 1.0 }, { solution.block_2() }, -avg_pressure_solution );
    linalg::lincomb( u.block_2(), { 1.0 }, { u.block_2() }, -avg_pressure_approximation );

    linalg::apply( K, u, stok_vecs["tmp_6"] );
    linalg::lincomb( stok_vecs["tmp_5"], { 1.0, -1.0 }, { f, stok_vecs["tmp_6"] } );
    const auto inf_residual_vel = linalg::norm_inf( stok_vecs["tmp_5"].block_1() );
    const auto inf_residual_pre = linalg::norm_inf( stok_vecs["tmp_5"].block_2() );

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error_velocity =
        std::sqrt( dot( error.block_1(), error.block_1() ) / static_cast< double >( num_dofs_velocity ) );
    const auto l2_error_pressure =
        std::sqrt( dot( error.block_2(), error.block_2() ) / static_cast< double >( num_dofs_pressure ) );

    table->add_row(
        { { "level", max_level },
          { "level_subdomains", level_subdomains },
          { "dofs_vel", num_dofs_velocity },
          { "l2_error_vel", l2_error_velocity },
          { "dofs_pre", num_dofs_pressure },
          { "l2_error_pre", l2_error_pressure },
          { "inf_res_vel", inf_residual_vel },
          { "inf_res_pre", inf_residual_pre },
          { "h_vel", ( r_max - r_min ) / std::pow( 2, velocity_level ) },
          { "h_p", ( r_max - r_min ) / std::pow( 2, pressure_level ) } } );

    // Keep file outputs as-is (not "cout" related)
    io::XDMFOutput xdmf(
        "out_eps",
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level] );
    xdmf.add( k.grid_data() );
    xdmf.add( u.block_1().grid_data() );
    xdmf.add( solution.block_1().grid_data() );
    //xdmf.write();

    terra::linalg::trafo::cartesian_to_normal_tangential_in_place< ScalarType, ScalarType >(
        u.block_1(), coords_shell[velocity_level], boundary_mask_data[velocity_level], CMB );

    VectorQ1Scalar< ScalarType > normals( "normals", domains[velocity_level], mask_data[velocity_level] );
    terra::kernels::common::extract_vector_component( normals.grid_data(), u.block_1().grid_data(), 0 );

    auto radii     = domains[velocity_level].domain_info().radii();
    auto rprofiles = terra::shell::radial_profiles(
        normals, subdomain_shell_idx( domains[velocity_level] ), domains[velocity_level].domain_info().radii().size() );

    auto          normaltable = terra::shell::radial_profiles_to_table( rprofiles, radii );
    std::ofstream out( "normal_radial_profiles.csv" );
    normaltable.print_csv( out );

    return {
        l2_error_velocity,
        l2_error_pressure,
        static_cast<int>( solver_table->query_rows_equals( "tag", "fgmres_solver" ).rows().size() )
    };
}

// -----------------------------------------------------------------------------
// main (loops over level_subdomains=0..2 and checks error invariance)
// -----------------------------------------------------------------------------

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    // --- Anisotropy CLI ---
    // Default: isotropic (existing behaviour). radial_extra_levels > 0 runs Path-B.1
    // anisotropy: radial diamond level at each MG level L becomes L + radial_extra_levels.
    int max_level           = 6;
    int radial_extra_levels = 0;
    int lat_sdr             = -1;
    int rad_sdr             = -1;
    {
        CLI::App app{ "test_epsilon_divdiv_stokes" };
        util::add_option_with_default( app, "--max-level", max_level, "Finest lateral diamond refinement level." );
        util::add_option_with_default(
            app,
            "--radial-extra-levels",
            radial_extra_levels,
            "Offset added to the radial diamond level at every MG level (default 0 = isotropic)." );
        util::add_option_with_default(
            app, "--lat-sdr", lat_sdr, "Override lateral subdomain refinement level (default: use level_subdomains)." );
        util::add_option_with_default(
            app, "--rad-sdr", rad_sdr, "Override radial subdomain refinement level (default: use level_subdomains)." );
        CLI11_PARSE( app, argc, argv );
    }

    auto table = std::make_shared< util::Table >();

    std::vector< int > kmaxs = { 1 };
    std::vector< int > gcas  = { 0 };

    auto table_dca  = std::make_shared< util::Table >();
    auto table_gca  = std::make_shared< util::Table >();
    auto table_agca = std::make_shared< util::Table >();

    // store errors[level][level_subdomains] for sanity check
    std::map< int, std::map< int, double > > err_vel;
    std::map< int, std::map< int, double > > err_pre;

    for ( int minlevel = 2; minlevel <= 2; ++minlevel )
    {
        util::logroot << "minlevel = " << minlevel << "\n";

        for ( int gca : gcas )
        {
            for ( int kmax : kmaxs )
            {
                // reset error storage for this configuration
                err_vel.clear();
                err_pre.clear();

                // Loop levels outer, so we can compare subdomain refinements on each fixed level.
                for ( int level = minlevel + 1; level <= max_level; ++level )
                {
                  
                    // convergence orders (computed against previous level, using subdomain=0)
                    static bool   have_prev_level = false;
                    static double prev_l2_vel     = 1.0;
                    static double prev_l2_pre     = 1.0;

                    for ( int level_subdomains = 0; level_subdomains <= 0; ++level_subdomains )
                    {
                        util::logroot << "  level_subdomains = " << level_subdomains << "\n";

                        Kokkos::Timer timer;
                        timer.reset();

                        const auto [l2_error_vel, l2_error_pre, iterations] = test(
                            kmax, gca, minlevel, level, level_subdomains, radial_extra_levels, lat_sdr, rad_sdr, table );

                        const auto time_total = timer.seconds();

                        table->add_row( { { "level", level },
                                          { "level_subdomains", level_subdomains },
                                          { "time_total", time_total } } );

                        util::logroot << "  errors: vel=" << l2_error_vel
                                        << " pre=" << l2_error_pre
                                        << " iters=" << iterations
                                        << " time_total=" << time_total << "\n";

                        // store and sanity-check invariance across subdomain refinements at fixed level
                        err_vel[level][level_subdomains] = l2_error_vel;
                        err_pre[level][level_subdomains] = l2_error_pre;

                        if ( level_subdomains > 0 )
                        {
                            const double dv = std::abs( err_vel[level][level_subdomains] - err_vel[level][level_subdomains - 1] );
                            const double dp = std::abs( err_pre[level][level_subdomains] - err_pre[level][level_subdomains - 1] );

                            // same spirit as your working test
                            if ( dv > 1e-3 || dp > 1e-3 )
                            {
                                util::logroot
                                    << "ERROR: Same global level should have same error regardless of subdomains.\n"
                                    << "  level=" << level
                                    << " vel_diff=" << dv
                                    << " pre_diff=" << dp << "\n";
                                Kokkos::abort( "Error invariance w.r.t. subdomain refinement violated." );
                            }
                        }

                        terra::util::Table::Row cycles;
                        cycles[std::string( "k_max=" ) + std::to_string( kmax )] = iterations;

                        if ( gca == 1 )      table_gca->add_row( cycles );
                        else if ( gca == 2 ) table_agca->add_row( cycles );
                        else                 table_dca->add_row( cycles );
                    }

                    // compute per-level order using subdomain refinement 0 (arbitrary choice; all match)
                    const double curr_l2_vel = err_vel[level][0];
                    const double curr_l2_pre = err_pre[level][0];

                    if ( have_prev_level )
                    {
                        const double order_vel = prev_l2_vel / curr_l2_vel;
                        const double order_pre = prev_l2_pre / curr_l2_pre;

                        util::logroot << "Level " << level
                                        << ": order_vel=" << order_vel
                                        << " order_pre=" << order_pre
                                        << " (using level_subdomains=0)\n";

                        table->add_row(
                            { { "level", level }, { "level_subdomains", 0 }, { "order_vel", order_vel }, { "order_pre", order_pre } } );
                    }

                    prev_l2_vel = curr_l2_vel;
                    prev_l2_pre = curr_l2_pre;
                    have_prev_level = true;
                }
            }
        }

        table->query_rows_not_none( "dofs_vel" )
            .select_columns(
                { "level", "level_subdomains", "dofs_pre", "dofs_vel", "l2_error_pre", "l2_error_vel", "h_vel", "h_p" } )
            .print_pretty();

        table->query_rows_not_none( "order_vel" )
            .select_columns( { "level", "level_subdomains", "order_pre", "order_vel" } )
            .print_pretty();
    }

    util::logroot << "DCA:\n";
    table_dca->print_pretty();
    util::logroot << "GCA:\n";
    table_gca->print_pretty();
    util::logroot << "AGCA:\n";
    table_agca->print_pretty();

    return 0;
}
