// "Naked" viscous-block (A-block) multigrid convergence study under increasing
// viscosity contrast, with and without Galerkin coarse approximation (GCA).
//
// Unlike test_epsilon_divdiv_stokes_viscosity_contrast.cpp -- which wraps the
// viscous-block MG as the (1,1) preconditioner inside the full Stokes saddle-
// point FGMRES -- this test isolates the A-block: it builds *only* the viscous
// block A = K.block_11() (free-slip), an MG hierarchy on it (Cheby-Jacobi
// smoothers + coarse FGMRES), and runs the multigrid V-cycles *as the solver*
// for  A u = f  (f = strong-enforced radial buoyancy force).  The recorded
// per-cycle relative residual is the diagnostic.
//
// The point is to measure how A-block MG convergence degrades as the
// Frank-Kamenetskii viscosity contrast rmu grows, and whether GCA-assembled
// coarse operators (P^T A P, built top-down via TwoGridGCA from the Neumann
// fine block) recover convergence where re-discretized coarse operators (DCA)
// stall/diverge.
//
// Geometry, viscosity law and buoyancy mirror the A7 benchmark's initial state
// (Ra=7e3, free-slip both shells, Y_3^2 perturbation), same as the contrast
// test -- only the contrast rmu is swept here instead of fixed at 1e5.

#include "../src/terra/communication/shell/communication.hpp"

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/strong_algebraic_freeslip_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"

#include "grid/shell/bit_masks.hpp"

#include "terra/geophysics/viscosity/viscosity_interpolation.hpp"
#include "terra/shell/radial_profiles.hpp"

#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/fgmres_lowmem.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/gca/gca_elements_collector.hpp"

#include "gca_fixed.hpp"  // copy of gca.hpp with a searchable crossover-DoF remap (gcafix::TwoGridGCA)
#include "linalg/solvers/multigrid.hpp"
#include "terra/linalg/solvers/chebyshev.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"

// Full-Stokes outer solve: w-BFBT Schur preconditioner (from the mc app).
#include "../apps/mantlecirculation/src/polymorphic_schur_preconditioner.hpp"
#include "../apps/mantlecirculation/src/wbfbt_pressure_poisson_explicit_kw.hpp"
#include "../apps/mantlecirculation/src/wbfbt_schur_preconditioner.hpp"
#include "../apps/mantlecirculation/src/wbfbt_weighted_lumped_velocity_mass.hpp"

#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"

#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/table.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::local_domain_md_range_policy_nodes;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using linalg::solvers::TwoGridGCA;
using terra::grid::shell::BoundaryConditions;

namespace mc = terra::mantlecirculation;

// Krylov-basis storage precision for the full-Stokes outer FGMRES. DOUBLE = plain
// FGMRES (basis in solution precision); FLOAT/BF16 = FGMRESLowMem with the basis
// stored in single / bfloat16 (mirrors the mc app's float/bf16 Krylov-basis path).
enum class KrylovPrec { DOUBLE, FLOAT, BF16 };

inline const char* krylov_prec_name( KrylovPrec p )
{
    return p == KrylovPrec::BF16 ? "bf16" : ( p == KrylovPrec::FLOAT ? "single" : "double" );
}

// A7 benchmark geometry + buoyancy parameters.  rmu (viscosity contrast) is now
// a runtime parameter -- it is what this test sweeps.
constexpr double kRm             = 1.22;
constexpr double kRp             = 2.22;
constexpr double kRayleigh       = 7.0e3;
constexpr double kPerturbDefault = 0.01;

constexpr auto BOUNDARY = static_cast< grid::shell::ShellBoundaryFlag >(
    static_cast< uint8_t >( CMB ) | static_cast< uint8_t >( SURFACE ) );

// Real spherical harmonic Y_3^2(theta, phi) = sqrt(105/(32pi)) sin^2(theta) cos(theta) cos(2 phi).
KOKKOS_INLINE_FUNCTION
double Y32( const double theta, const double phi )
{
    constexpr double c  = 1.0219854764332823;
    const double     st = Kokkos::sin( theta );
    const double     ct = Kokkos::cos( theta );
    return c * st * st * ct * Kokkos::cos( 2.0 * phi );
}

// A7 initial T(r, theta, phi) clamped to [0, 1].
KOKKOS_INLINE_FUNCTION
double a7_initial_temperature( const double cx, const double cy, const double cz, double perturb_amp )
{
    const double rad = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
    double       T_lateral = 0.0;
    if ( perturb_amp != 0.0 && rad > 0.0 )
    {
        const double theta = Kokkos::acos( cz / rad );
        const double phi   = Kokkos::atan2( cy, cx );
        T_lateral          = perturb_amp * Y32( theta, phi );
    }
    return Kokkos::clamp( ( kRp - rad ) / ( kRp - kRm ) + T_lateral, 0.0, 1.0 );
}

// eta(T) = rmu^(0.5 - T), same as apps/mantlecirculation/src/interpolators.hpp.
struct FrankKamenetskiiViscosityInterpolator
{
    Grid3DDataVec< double, 3 >       grid_;
    Grid2DDataScalar< double >       radii_;
    grid::Grid4DDataScalar< double > data_;
    double                           rmu_;
    double                           perturb_amp_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const double T = a7_initial_temperature( c( 0 ), c( 1 ), c( 2 ), perturb_amp_ );
        data_( sd, x, y, r ) = Kokkos::pow( rmu_, 0.5 - T );
    }
};

// Radial low-viscosity BAND: eta = eta_band (low) inside |rad - r_center| < half_width,
// eta = 1 elsewhere — a weak channel of fixed physical width.  The band is wide
// enough to be resolved on the fine levels but sub-cell on the coarse levels, so DCA
// mis-resolves it while GCA's P^T A P inherits the fine-resolved band.
struct RadialBandViscosityInterpolator
{
    Grid3DDataVec< double, 3 >       grid_;
    Grid2DDataScalar< double >       radii_;
    grid::Grid4DDataScalar< double > data_;
    double                           eta_band_;
    double                           r_center_;
    double                           half_width_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c   = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const double                  rad = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
        data_( sd, x, y, r )              = ( Kokkos::fabs( rad - r_center_ ) < half_width_ ) ? eta_band_ : 1.0;
    }
};

// Radial buoyancy body force f(x) = -Ra * T(x) * r_hat.
struct RadialBuoyancyRHSInterpolator
{
    Grid3DDataVec< double, 3 >       grid_;
    Grid2DDataScalar< double >       radii_;
    grid::Grid4DDataVec< double, 3 > data_;
    double                           rayleigh_;
    double                           perturb_amp_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const double rad = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
        if ( rad <= 0.0 )
        {
            for ( int d = 0; d < 3; ++d )
                data_( sd, x, y, r, d ) = 0.0;
            return;
        }
        const double T        = a7_initial_temperature( c( 0 ), c( 1 ), c( 2 ), perturb_amp_ );
        const double f_radial = -rayleigh_ * T;
        for ( int d = 0; d < 3; ++d )
            data_( sd, x, y, r, d ) = f_radial * c( d ) / rad;
    }
};

// Inverse-diagonal-as-preconditioner for the coarse FGMRES (same pattern as the
// mc app's coarse solver in stokes_solver.hpp).
template < linalg::OperatorLike OperatorT >
struct InverseDiagonalPreconditioner
{
    using OperatorType       = OperatorT;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType      = linalg::DstOf< OperatorType >;

    SolutionVectorType inv_diag_;

    explicit InverseDiagonalPreconditioner( const SolutionVectorType& d )
    : inv_diag_( d )
    {}

    void solve_impl( OperatorType& /*A*/, SolutionVectorType& x, const RHSVectorType& b )
    {
        linalg::assign( x, b );
        linalg::scale_in_place( x, inv_diag_ );
    }
};

enum class BCKind { DIRICHLET, FREESLIP };

struct RunResult
{
    int    cycles;              // number of MG V-cycles performed
    double rel_residual_final;  // final relative residual
    double asymptotic_rate;     // residual_convergence_rate of the last cycle
    bool   converged;
};

// Build the viscous A-block MG and run it as the solver for  A u = f.
// gca: 0 = re-discretized coarse operators (DCA),
//      1 = full GCA (P^T A P on every coarse element),
//      2 = adaptive GCA (only on elements flagged by the gradient collector).
RunResult run_ablock_mg( int    min_level,
                         int    max_level,
                         int    cheby_order,
                         double perturb_amp,
                         double rmu,
                         int    gca,
                         int    max_cycles,
                         BCKind bc,
                         linalg::solvers::InterpolationMode gca_interp,
                         int    visc_profile,   // 0 = Frank-Kamenetskii (smooth), 1 = sharp radial layer
                         double coarse_tol,     // coarse FGMRES relative tolerance
                         bool   full_stokes,    // false = solve A-block with MG; true = full Stokes outer FGMRES
                         KrylovPrec krylov_prec ) // Krylov-basis precision for the full-Stokes outer FGMRES
{
    using ScalarType = double;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    for ( int level = min_level; level <= max_level; ++level )
    {
        const int idx = level - min_level;
        domains.push_back( DistributedDomain::create_uniform( level, level, kRm, kRp, 0, 0 ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const size_t num_levels     = domains.size();
    const size_t velocity_level = num_levels - 1;
    const size_t pressure_level = num_levels - 2;

    const auto bc_flag = ( bc == BCKind::DIRICHLET ) ? DIRICHLET : FREESLIP;
    BoundaryConditions bcs = {
        { CMB, bc_flag },
        { SURFACE, bc_flag },
    };
    // GCA descent always coarsens from the Neumann block (gca.hpp does not
    // transfer Dirichlet/free-slip BCs; they are imposed on the fine operator
    // and -- for Dirichlet -- algebraically on the RHS).
    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };

    using Stokes       = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous      = Stokes::Block11Type;
    using Gradient     = Stokes::Block12Type;
    using Divergence   = Stokes::Block21Type;
    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;
    using ViscousMass  = fe::wedge::operators::shell::VectorMass< ScalarType >;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    // Low-viscosity band: centre r=2.0, width 0.1 (half-width 0.05), eta = 1/rmu inside.
    const double r_center   = 2.0;
    const double half_width = 0.05;
    const double eta_band   = 1.0 / rmu;  // low-viscosity band, contrast = rmu

    // Per-level viscosity fields.
    std::vector< VectorQ1Scalar< ScalarType > > eta;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        eta.emplace_back( "eta_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( visc_profile == 2 || visc_profile == 3 )
        {
            // Published radial viscosity profile (min-normalized), interpolated by radius
            // onto this level's nodes.  T-independent.  2 = Stotz 2017 (~12000), 3 = Lin 2022 (~1057).
            const std::string csv =
                ( visc_profile == 2 )
                    ? "/home/hpc/iwia/iwia054h/terraneo/data/radialprofiles/ViscosityProfile_Stotz_et_al_2017.csv"
                    : "/home/hpc/iwia/iwia054h/terraneo/data/radialprofiles/ViscosityProfile_Lin_et_al_2022.csv";
            auto profile = shell::interpolate_radial_profile_into_subdomains_from_csv< ScalarType >(
                csv, "radius_normalized_1p22_2p22", "viscosity_scaled_by_min", coords_radii[level] );
            geophysics::viscosity::RadialProfileViscosityInterpolator< ScalarType >( profile, 1.0 )
                .interpolate( eta[level].grid_data() );
        }
        else if ( visc_profile == 1 )
        {
            Kokkos::parallel_for(
                "layer_viscosity_level_" + std::to_string( level ),
                local_domain_md_range_policy_nodes( domains[level] ),
                RadialBandViscosityInterpolator{
                    coords_shell[level], coords_radii[level], eta[level].grid_data(), eta_band, r_center, half_width } );
        }
        else
        {
            Kokkos::parallel_for(
                "fk_viscosity_level_" + std::to_string( level ),
                local_domain_md_range_policy_nodes( domains[level] ),
                FrankKamenetskiiViscosityInterpolator{
                    coords_shell[level], coords_radii[level], eta[level].grid_data(), rmu, perturb_amp } );
        }
        Kokkos::fence();
    }

    // Fine free-slip viscous block (the operator the MG solves), plus the
    // Neumann copy used as the GCA descent's finest source (matches the mc app:
    // GCA coarsens from K_neumann_->block_11()).
    Stokes K_op(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], eta[velocity_level].grid_data(),
        bcs, false );

    Stokes K_op_neumann(
        domains[velocity_level], domains[pressure_level],
        coords_shell[velocity_level], coords_radii[velocity_level],
        boundary_mask_data[velocity_level], eta[velocity_level].grid_data(),
        bcs_neumann, false );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // GCAElements map on the coarsest grid: which coarsest-grid elements get GCA.
    VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", domains[0], mask_data[0] );
    if ( gca == 2 )
    {
        linalg::assign( GCAElements, 0 );
        terra::linalg::solvers::GCAElementsCollector< ScalarType >(
            domains[velocity_level], eta[velocity_level].grid_data(),
            max_level - min_level, GCAElements.grid_data() );
    }
    else if ( gca == 1 )
    {
        linalg::assign( GCAElements, 1 );
    }

    // Coarse viscous operators + transfer operators.
    std::vector< Viscous >                   A_c;
    std::vector< Prolongation >              P;
    std::vector< Restriction >               R;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg, tmp_mg_r, tmp_mg_e;

    for ( size_t level = 0; level < num_levels; ++level )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );

            A_c.emplace_back(
                domains[level], coords_shell[level], coords_radii[level],
                boundary_mask_data[level], eta[level].grid_data(), bcs, false );

            if ( gca == 1 )
            {
                A_c.back().set_stored_matrix_mode(
                    linalg::OperatorStoredMatrixMode::Full, static_cast< int >( level ), GCAElements.grid_data() );
            }
            else if ( gca == 2 )
            {
                A_c.back().set_stored_matrix_mode(
                    linalg::OperatorStoredMatrixMode::Selective, static_cast< int >( level ), GCAElements.grid_data() );
            }

            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // GCA assembly, top-down: coarsest source is the Neumann fine block, then
    // each coarse op is coarsened from the next-finer coarse op.
    if ( gca > 0 )
    {
        for ( int level = static_cast< int >( num_levels ) - 2; level >= 0; --level )
        {
            TwoGridGCA< ScalarType, Viscous >(
                ( level == static_cast< int >( num_levels ) - 2 ) ? K_op_neumann.block_11() : A_c[level + 1],
                A_c[level],
                level,
                GCAElements.grid_data(),
                /*treat_boundary=*/true,
                gca_interp );
        }
    }

    // Inverse diagonals per level -- computed AFTER GCA assembly so the coarse
    // diagonals reflect the stored (P^T A P) matrices.
    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        inverse_diagonals.emplace_back(
            "inv_diag_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > ones(
            "inv_diag_ones_" + std::to_string( level ), domains[level], mask_data[level] );
        linalg::assign( ones, 1.0 );

        if ( level == velocity_level )
        {
            K_op.block_11().set_diagonal( true );
            linalg::apply( K_op.block_11(), ones, inverse_diagonals.back() );
            K_op.block_11().set_diagonal( false );
        }
        else
        {
            A_c[level].set_diagonal( true );
            linalg::apply( A_c[level], ones, inverse_diagonals.back() );
            A_c[level].set_diagonal( false );
        }
        linalg::invert_entries( inverse_diagonals.back() );
    }

    // Chebyshev smoothers per level.
    using Smoother = linalg::solvers::Chebyshev< Viscous >;
    std::vector< Smoother > smoothers;
    for ( size_t level = 0; level < num_levels; ++level )
    {
        std::vector< VectorQ1Vec< ScalarType > > cheby_tmps;
        cheby_tmps.emplace_back( "cheby_tmp_0_" + std::to_string( level ), domains[level], mask_data[level] );
        cheby_tmps.emplace_back( "cheby_tmp_1_" + std::to_string( level ), domains[level], mask_data[level] );

        constexpr int chebyshev_prepost = 3;
        smoothers.emplace_back( cheby_order, inverse_diagonals[level], cheby_tmps, chebyshev_prepost );
    }

    // Coarse FGMRES (matches the mc app's coarse solver).
    auto coarse_table = std::make_shared< util::Table >();
    using CoarseGridPrec   = InverseDiagonalPreconditioner< Viscous >;
    using CoarseGridSolver = linalg::solvers::FGMRES< Viscous, CoarseGridPrec >;
    constexpr int coarse_restart = 30;
    const int     coarse_max     = ( coarse_tol < 1e-7 ) ? 300 : 50;  // allow more iters for a tight coarse tol
    std::vector< VectorQ1Vec< ScalarType > > coarse_tmps;
    for ( int i = 0; i < 2 * coarse_restart + 4; ++i )
        coarse_tmps.emplace_back( "coarse_tmp", domains[0], mask_data[0] );
    linalg::solvers::FGMRESOptions< ScalarType > coarse_opts;
    coarse_opts.restart                     = coarse_restart;
    coarse_opts.max_iterations              = coarse_max;
    coarse_opts.relative_residual_tolerance = coarse_tol;
    coarse_opts.absolute_residual_tolerance = 1e-16;
    CoarseGridSolver coarse_solver( coarse_tmps, coarse_opts, coarse_table, CoarseGridPrec( inverse_diagonals[0] ) );
    coarse_solver.set_tag( "coarse_grid_fgmres" );

    // Multigrid: A-block solver (max_cycles V-cycles) OR a single V-cycle when used
    // as the (1,1) preconditioner inside the full-Stokes outer FGMRES.
    using MG = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;
    auto mg_table = std::make_shared< util::Table >();
    MG  mg( P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_solver,
            full_stokes ? 1 : max_cycles, full_stokes ? 1e-16 : 1e-10 );
    mg.collect_statistics( mg_table );

    // RHS: M * (-Ra * T * r_hat), strong-enforced for free-slip.  Built on a
    // full IsoQ2Q1 vector to reuse the free-slip enforcement, then the velocity
    // block is the A-block RHS.
    VectorQ1IsoQ2Q1< ScalarType > f_full(
        "f_full", domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
    VectorQ1IsoQ2Q1< ScalarType > tmp_full(
        "tmp_full", domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );

    Kokkos::parallel_for(
        "radial_buoyancy_rhs",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        RadialBuoyancyRHSInterpolator{
            coords_shell[velocity_level], coords_radii[velocity_level],
            tmp_full.block_1().grid_data(), kRayleigh, perturb_amp } );
    Kokkos::fence();

    linalg::apply( M, tmp_full.block_1(), f_full.block_1() );
    if ( bc == BCKind::FREESLIP )
    {
        fe::strong_algebraic_freeslip_enforcement_in_place(
            f_full, coords_shell[velocity_level], boundary_mask_data[velocity_level], BOUNDARY );
    }
    else
    {
        // Homogeneous no-slip: zero the velocity RHS on both shells.
        fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
            f_full, boundary_mask_data[velocity_level], BOUNDARY );
    }

    if ( !full_stokes )
    {
        // ---- A-block: solve  A u = f  with multigrid as the solver. ----
        VectorQ1Vec< ScalarType > u_vel( "u_vel", domains[velocity_level], mask_data[velocity_level] );
        linalg::assign( u_vel, 0.0 );
        linalg::solvers::solve( mg, K_op.block_11(), u_vel, f_full.block_1() );

        mg_table->query_rows_equals( "tag", "multigrid" )
            .select_columns( { "absolute_residual", "relative_residual", "residual_convergence_rate" } )
            .print_pretty();

        const auto rows = mg_table->query_rows_equals( "tag", "multigrid" ).rows();
        const int    cycles = rows.empty() ? 0 : static_cast< int >( rows.size() ) - 1;
        const double rel_residual_final =
            rows.empty() ? 1.0 : std::get< double >( rows.back().at( "relative_residual" ) );
        double asymptotic_rate = std::numeric_limits< double >::quiet_NaN();
        if ( rows.size() >= 2 && rows.back().count( "residual_convergence_rate" ) )
            asymptotic_rate = std::get< double >( rows.back().at( "residual_convergence_rate" ) );
        return { cycles, rel_residual_final, asymptotic_rate, rel_residual_final <= 1e-6 };
    }

    // ---- Full Stokes: outer FGMRES on the saddle point, with the MG (DCA/GCA) as
    //      the (1,1) preconditioner and a w-BFBT Schur preconditioner. ----
    // Pressure mass on the pressure level, scaled by 1/eta (the (2,2) PrecStokes block).
    VectorQ1Scalar< ScalarType > k_pm( "k_pm", domains[pressure_level], mask_data[pressure_level] );
    linalg::assign( k_pm, eta[pressure_level] );
    linalg::invert_entries( k_pm );
    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level], k_pm.grid_data(), false );
    pmass.set_lumped_diagonal( true );

    auto solver_table = std::make_shared< util::Table >();

    // w-BFBT Schur: explicit K_w (Neumann B/B^T) + lumped C_w(sqrt eta).
    VectorQ1Scalar< ScalarType > sqrt_eta_velocity(
        "sqrt_eta_velocity", domains[velocity_level], mask_data[velocity_level] );
    linalg::assign( sqrt_eta_velocity, eta[velocity_level] );
    {
        auto v = sqrt_eta_velocity.grid_data();
        Kokkos::parallel_for( "sqrt_eta", local_domain_md_range_policy_nodes( domains[velocity_level] ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                v( sd, x, y, r ) = Kokkos::sqrt( v( sd, x, y, r ) );
            } );
        Kokkos::fence();
    }
    using WBFBTCw = mc::WBFBTWeightedLumpedVelocityMass< ScalarType, 3 >;
    WBFBTCw c_w( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level],
                 mask_data[velocity_level] );
    c_w.refresh( sqrt_eta_velocity );

    using WBFBTKw = mc::ExplicitKwPressurePoissonSolver< ScalarType, Gradient, Divergence >;
    auto kw_solver = std::make_shared< WBFBTKw >(
        K_op_neumann.block_12(), K_op_neumann.block_21(), c_w.inv_diag_velocity(),
        domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level],
        /*max_iterations=*/200, /*relative_tol=*/static_cast< ScalarType >( 1e-6 ), solver_table );

    using PrecSchur  = mc::PolymorphicSchurPreconditioner< PressureMass >;
    using WBFBTSchur = mc::WBFBTSchurPreconditioner< Viscous, Gradient, Divergence, PressureMass >;
    PrecSchur prec_schur = PrecSchur::make( WBFBTSchur(
        K_op.block_11(), K_op_neumann.block_12(), K_op_neumann.block_21(), kw_solver, c_w.inv_diag_velocity(),
        domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] ) );

    using PrecStokes = linalg::solvers::BlockTriangularPreconditioner2x2<
        Stokes, Viscous, PressureMass, Gradient, MG, PrecSchur >;
    VectorQ1IsoQ2Q1< ScalarType > tri_tmp(
        "tri_tmp", domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
    PrecStokes prec_stokes( K_op.block_11(), pmass, K_op.block_12(), tri_tmp, mg, prec_schur );

    constexpr int outer_restart = 50;
    linalg::solvers::FGMRESOptions< ScalarType > outer_opts;
    outer_opts.restart                     = outer_restart;
    outer_opts.max_iterations              = max_cycles;  // reuse --max-cycles as outer iteration cap
    outer_opts.relative_residual_tolerance = 1e-6;
    outer_opts.absolute_residual_tolerance = 1e-16;
    auto outer_table = std::make_shared< util::Table >();

    VectorQ1IsoQ2Q1< ScalarType > u_stokes(
        "u_stokes", domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
    linalg::assign( u_stokes, 0.0 );

    // Outer FGMRES with the Krylov basis stored in double / single / bf16.
    //   double      -> plain FGMRES (basis == solution precision),
    //   float/bf16  -> FGMRESLowMem (3 full-precision scratch + 2*restart+1 reduced-precision basis),
    // mirroring the mc app's float/bf16 Krylov-basis workspace in stokes_solver.hpp.
    if ( krylov_prec == KrylovPrec::DOUBLE )
    {
        std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
        for ( int i = 0; i < 2 * outer_restart + 4; ++i )
            tmp_fgmres.emplace_back( "tmp_fgmres_" + std::to_string( i ),
                domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
        linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres( tmp_fgmres, outer_opts, outer_table, prec_stokes );
        fgmres.set_tag( "stokes_fgmres" );
        linalg::solvers::solve( fgmres, K_op, u_stokes, f_full );
    }
    else
    {
        std::vector< VectorQ1IsoQ2Q1< ScalarType > > work;  // FGMRESLowMem aliases r/w => 3 scratch
        for ( int i = 0; i < 3; ++i )
            work.emplace_back( "work_fgmres_" + std::to_string( i ),
                domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
        if ( krylov_prec == KrylovPrec::FLOAT )
        {
            using BasisF = VectorQ1IsoQ2Q1< float >;
            std::vector< BasisF > basis;
            for ( int i = 0; i < 2 * outer_restart + 1; ++i )
                basis.emplace_back( "basis_fgmres_" + std::to_string( i ),
                    domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
            linalg::solvers::FGMRESLowMem< Stokes, BasisF, PrecStokes > fgmres(
                work, basis, outer_opts, outer_table, prec_stokes );
            fgmres.set_tag( "stokes_fgmres" );
            linalg::solvers::solve( fgmres, K_op, u_stokes, f_full );
        }
        else  // BF16
        {
            using BasisH = VectorQ1IsoQ2Q1< Kokkos::Experimental::bhalf_t >;
            std::vector< BasisH > basis;
            for ( int i = 0; i < 2 * outer_restart + 1; ++i )
                basis.emplace_back( "basis_fgmres_" + std::to_string( i ),
                    domains[velocity_level], domains[pressure_level], mask_data[velocity_level], mask_data[pressure_level] );
            linalg::solvers::FGMRESLowMem< Stokes, BasisH, PrecStokes > fgmres(
                work, basis, outer_opts, outer_table, prec_stokes );
            fgmres.set_tag( "stokes_fgmres" );
            linalg::solvers::solve( fgmres, K_op, u_stokes, f_full );
        }
    }

    outer_table->query_rows_equals( "tag", "stokes_fgmres" )
        .select_columns( { "absolute_residual", "relative_residual", "iteration" } )
        .print_pretty();

    const auto rows    = outer_table->query_rows_equals( "tag", "stokes_fgmres" ).rows();
    const int  iters   = rows.empty() ? 0 : static_cast< int >( rows.size() ) - 1;
    const double rel_residual_final =
        rows.empty() ? 1.0 : std::get< double >( rows.back().at( "relative_residual" ) );
    return { iters, rel_residual_final, std::numeric_limits< double >::quiet_NaN(), rel_residual_final <= 1e-6 };
}

// Direct fidelity check: does GCA's stored-matrix coarse operator reproduce the
// TRUE Galerkin action R(A(P(x)))?  Builds a single coarse/fine level pair, the
// Neumann fine block A (the GCA descent's source), coarsens it into A_c via
// TwoGridGCA, then compares  A_c·x  against  R(A(P(x)))  using the MG's own
// constant prolongation P and restriction R.  Also checks constant-mode
// reproduction (A_c·1 vs R(A(P(1)))).  Neumann BCs throughout so no apply-time
// boundary masking muddies the interior comparison.
void verify_gca_rap( int max_level, double rmu, linalg::solvers::InterpolationMode interp )
{
    using ScalarType = double;
    using Stokes     = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous    = Stokes::Block11Type;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    const int cl = max_level - 1;  // coarse level
    const int fl = max_level;      // fine level

    auto make = [&]( int lvl ) {
        auto dom = DistributedDomain::create_uniform( lvl, lvl, kRm, kRp, 0, 0 );
        return dom;
    };
    DistributedDomain dom_c = make( cl ), dom_f = make( fl );
    auto coords_c = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom_c );
    auto coords_f = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom_f );
    auto radii_c  = grid::shell::subdomain_shell_radii< ScalarType >( dom_c );
    auto radii_f  = grid::shell::subdomain_shell_radii< ScalarType >( dom_f );
    auto mask_c   = grid::setup_node_ownership_mask_data( dom_c );
    auto mask_f   = grid::setup_node_ownership_mask_data( dom_f );
    auto bmask_c  = grid::shell::setup_boundary_mask_data( dom_c );
    auto bmask_f  = grid::shell::setup_boundary_mask_data( dom_f );

    BoundaryConditions bcs_neumann = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };

    // eta on both levels (FK law).
    VectorQ1Scalar< ScalarType > eta_c( "eta_c", dom_c, mask_c ), eta_f( "eta_f", dom_f, mask_f );
    Kokkos::parallel_for( "eta_c", local_domain_md_range_policy_nodes( dom_c ),
        FrankKamenetskiiViscosityInterpolator{ coords_c, radii_c, eta_c.grid_data(), rmu, kPerturbDefault } );
    Kokkos::parallel_for( "eta_f", local_domain_md_range_policy_nodes( dom_f ),
        FrankKamenetskiiViscosityInterpolator{ coords_f, radii_f, eta_f.grid_data(), rmu, kPerturbDefault } );
    Kokkos::fence();

    // Fine Neumann viscous block (GCA source) and coarse op (Neumann, stored).
    Viscous A_fine( dom_f, coords_f, radii_f, bmask_f, eta_f.grid_data(), bcs_neumann, false );
    Viscous A_c( dom_c, coords_c, radii_c, bmask_c, eta_c.grid_data(), bcs_neumann, false );

    VectorQ1Scalar< ScalarType > GCAElements( "GCAElements", dom_c, mask_c );
    linalg::assign( GCAElements, 1 );
    A_c.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Full, 0, GCAElements.grid_data() );

    TwoGridGCA< ScalarType, Viscous >( A_fine, A_c, 0, GCAElements.grid_data(), /*treat_boundary=*/true, interp );

    // A_self: the SAME fine operator, but in stored mode populated with its own
    // assemble_local_matrix (the matrices GCA reads), applied element-wise via the
    // slow path.  Comparing A_self vs the live A_fine isolates Q1 (assemble vs
    // apply); comparing A_c vs R*A_self*P isolates Q2 (the GCA triple product).
    Viscous A_self( dom_f, coords_f, radii_f, bmask_f, eta_f.grid_data(), bcs_neumann, false );
    VectorQ1Scalar< ScalarType > GCAElements_f( "GCAElements_f", dom_f, mask_f );
    linalg::assign( GCAElements_f, 1 );
    A_self.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Full, 0, GCAElements_f.grid_data() );
    {
        const auto& di = dom_f.domain_info();
        Kokkos::parallel_for(
            "populate_self",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { static_cast< long long >( dom_f.subdomains().size() ),
                  di.subdomain_num_nodes_per_side_laterally() - 1,
                  di.subdomain_num_nodes_per_side_laterally() - 1,
                  di.subdomain_num_nodes_radially() - 1 } ),
            KOKKOS_LAMBDA( const int sd, const int xx, const int yy, const int rr ) {
                A_self.set_local_matrix( sd, xx, yy, rr, 0, A_self.assemble_local_matrix( sd, xx, yy, rr, 0 ) );
                A_self.set_local_matrix( sd, xx, yy, rr, 1, A_self.assemble_local_matrix( sd, xx, yy, rr, 1 ) );
            } );
        Kokkos::fence();
    }

    Prolongation P( linalg::OperatorApplyMode::Replace );
    Restriction  R( dom_c );

    VectorQ1Vec< ScalarType > xc( "xc", dom_c, mask_c ), y_gca( "y_gca", dom_c, mask_c ),
        y_rap( "y_rap", dom_c, mask_c ), y_rap_self( "y_rap_self", dom_c, mask_c ), diff( "diff", dom_c, mask_c );
    VectorQ1Vec< ScalarType > xf( "xf", dom_f, mask_f ), axf( "axf", dom_f, mask_f ),
        axf_self( "axf_self", dom_f, mask_f ), fdiff( "fdiff", dom_f, mask_f );

    const char* interp_name = ( interp == linalg::solvers::InterpolationMode::Linear ) ? "linear" : "constant";
    util::logroot << "\n=== GCA fidelity check  (coarse lvl " << cl << " <- fine lvl " << fl
                  << ", rmu=" << rmu << ", interp=" << interp_name << ") ===\n";

    // Test vectors: a smooth field, and the constant field (constant-reproduction).
    for ( int tc = 0; tc < 2; ++tc )
    {
        if ( tc == 0 )
        {
            auto d = xc.grid_data();
            Kokkos::parallel_for( "fill_smooth", local_domain_md_range_policy_nodes( dom_c ),
                KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                    for ( int c = 0; c < 3; ++c )
                        d( sd, x, y, r, c ) =
                            Kokkos::sin( 0.7 * ( x + 1 ) + 0.3 * ( y + 1 ) + 0.5 * ( r + 1 ) + c );
                } );
            Kokkos::fence();
        }
        else
        {
            linalg::assign( xc, 1.0 );
        }

        linalg::apply( A_c, xc, y_gca );          // GCA stored-matrix action
        linalg::apply( P, xc, xf );               // prolongate coarse -> fine
        linalg::apply( A_fine, xf, axf );         // fine operator (live apply)
        linalg::apply( R, axf, y_rap );           // restrict fine -> coarse
        linalg::apply( A_self, xf, axf_self );    // fine operator (assemble_local_matrix, slow)
        linalg::apply( R, axf_self, y_rap_self ); // restrict fine -> coarse

        linalg::lincomb( diff, { 1.0, -1.0 }, { y_gca, y_rap } );
        const double n_gca  = linalg::norm_2( y_gca );
        const double n_rap  = linalg::norm_2( y_rap );
        const double n_diff = linalg::norm_2( diff );
        const double rel    = ( n_rap > 0 ) ? n_diff / n_rap : n_diff;

        util::logroot << "  [" << ( tc == 0 ? "smooth x" : "constant x" ) << "]  "
                      << "||A_c x|| = " << n_gca
                      << "   ||R A P x|| = " << n_rap
                      << "   ||A_c x - R A P x|| = " << n_diff
                      << "   rel = " << rel << "\n";

        if ( tc == 0 )
        {
            // Q1: live apply vs assemble_local_matrix (on the fine grid).
            linalg::lincomb( fdiff, { 1.0, -1.0 }, { axf_self, axf } );
            const double q1 = linalg::norm_2( axf ) > 0 ? linalg::norm_2( fdiff ) / linalg::norm_2( axf ) : 0.0;
            // Q2: GCA triple product vs true Galerkin of the SAME element matrices.
            linalg::lincomb( diff, { 1.0, -1.0 }, { y_gca, y_rap_self } );
            const double q2 = linalg::norm_2( y_rap_self ) > 0
                                  ? linalg::norm_2( diff ) / linalg::norm_2( y_rap_self ) : 0.0;
            util::logroot << "    Q1 (assemble_local_matrix vs live apply): rel = " << q1 << "\n"
                          << "    Q2 (GCA triple-product vs R*A_self*P):    rel = " << q2 << "\n";
        }
    }
    util::logroot << "  (rel ~ 0  => GCA faithfully assembles R A P; large => assembly bug)\n";

    // -------- Search the crossover-DoF permutation that makes gcafix == R*A_self*P --------
    // Reference: true Galerkin of the assemble_local_matrix operator, on smooth x.
    {
        auto d = xc.grid_data();
        Kokkos::parallel_for( "fill_smooth2", local_domain_md_range_policy_nodes( dom_c ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                for ( int c = 0; c < 3; ++c )
                    d( sd, x, y, r, c ) = Kokkos::sin( 0.7 * ( x + 1 ) + 0.3 * ( y + 1 ) + 0.5 * ( r + 1 ) + c );
            } );
        Kokkos::fence();
    }
    linalg::apply( P, xc, xf );
    linalg::apply( A_self, xf, axf_self );
    linalg::apply( R, axf_self, y_rap_self );
    const double ref_norm = linalg::norm_2( y_rap_self );

    Viscous A_c_fix( dom_c, coords_c, radii_c, bmask_c, eta_c.grid_data(), bcs_neumann, false );
    A_c_fix.set_stored_matrix_mode( linalg::OperatorStoredMatrixMode::Full, 0, GCAElements.grid_data() );

    const auto interp_fix = ( interp == linalg::solvers::InterpolationMode::Linear )
                                ? linalg::solvers::gcafix::InterpolationMode::Linear
                                : linalg::solvers::gcafix::InterpolationMode::Constant;

    int    perm[6] = { 0, 1, 2, 3, 4, 5 };
    int    best_perm[6];
    double best_rel = 1e300, identity_rel = -1.0;
    do
    {
        linalg::solvers::gcafix::TwoGridGCA< ScalarType, Viscous >(
            A_fine, A_c_fix, 0, GCAElements.grid_data(), /*treat_boundary=*/true, interp_fix, perm );
        linalg::apply( A_c_fix, xc, y_gca );
        linalg::lincomb( diff, { 1.0, -1.0 }, { y_gca, y_rap_self } );
        const double rel = ref_norm > 0 ? linalg::norm_2( diff ) / ref_norm : 0.0;
        const bool   is_identity =
            ( perm[0] == 0 && perm[1] == 1 && perm[2] == 2 && perm[3] == 3 && perm[4] == 4 && perm[5] == 5 );
        if ( is_identity )
            identity_rel = rel;
        if ( rel < best_rel )
        {
            best_rel = rel;
            for ( int i = 0; i < 6; ++i )
                best_perm[i] = perm[i];
        }
    } while ( std::next_permutation( perm, perm + 6 ) );

    util::logroot << "  [crossover-perm search]  identity rel = " << identity_rel
                  << "   best rel = " << best_rel << "   best perm = ["
                  << best_perm[0] << " " << best_perm[1] << " " << best_perm[2] << " "
                  << best_perm[3] << " " << best_perm[4] << " " << best_perm[5] << "]\n";

    // -------- Linear-reproduction test (first-order consistency of the coarse op) --------
    // x_lin = position field (a global linear/constant-strain field). For eta=1 Neumann
    // operators, A * x_lin = 0 at INTERIOR nodes (constant stress, zero divergence).
    // P_mg reproduces linears, so R*A_self*P*x_lin is ~0 in the interior.  If A_c (GCA)
    // does NOT annihilate x_lin in the interior, GCA's interpolation is not first-order
    // consistent -> that is the root cause of the 26% mismatch.
    {
        auto       d   = xc.grid_data();
        const auto g   = coords_c;
        const auto rad = radii_c;
        Kokkos::parallel_for( "fill_linear", local_domain_md_range_policy_nodes( dom_c ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, g, rad );
                for ( int comp = 0; comp < 3; ++comp )
                    d( sd, x, y, r, comp ) = c( comp );
            } );
        Kokkos::fence();
    }
    linalg::apply( A_c, xc, y_gca );
    linalg::apply( P, xc, xf );
    linalg::apply( A_self, xf, axf_self );
    linalg::apply( R, axf_self, y_rap_self );

    // Zero the boundary (CMB/SURFACE) coarse nodes so we measure the INTERIOR only.
    auto zero_boundary = [&]( VectorQ1Vec< ScalarType >& v ) {
        auto       vd = v.grid_data();
        const auto bm = bmask_c;
        Kokkos::parallel_for( "zero_bnd", local_domain_md_range_policy_nodes( dom_c ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                if ( util::has_flag( bm( sd, x, y, r ), CMB ) || util::has_flag( bm( sd, x, y, r ), SURFACE ) )
                    for ( int comp = 0; comp < 3; ++comp )
                        vd( sd, x, y, r, comp ) = 0.0;
            } );
        Kokkos::fence();
    };
    const double full_gca = linalg::norm_2( y_gca );
    const double full_ref = linalg::norm_2( y_rap_self );
    zero_boundary( y_gca );
    zero_boundary( y_rap_self );
    util::logroot << "  [linear x_lin]  interior ||A_c x_lin|| = " << linalg::norm_2( y_gca )
                  << "   interior ||R A P x_lin|| = " << linalg::norm_2( y_rap_self )
                  << "   (full: " << full_gca << " vs " << full_ref << ")\n"
                  << "    (interior ref ~0 & GCA large => GCA P does NOT reproduce linears => root cause)\n";
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int    min_level   = 2;
    int    max_level   = 5;
    int    cheby_order = 4;
    int    max_cycles  = 50;
    double perturb_amp = kPerturbDefault;
    int    single_gca  = -1;     // -1 => sweep schemes; otherwise run that gca only
    double single_rmu  = -1.0;   // <0 => sweep the rmu ladder; otherwise single rmu
    BCKind bc          = BCKind::DIRICHLET;  // default: full (no-slip) Dirichlet
    using linalg::solvers::InterpolationMode;
    InterpolationMode single_interp = InterpolationMode::Constant;  // for --gca single runs
    int    visc_profile = 0;  // 0 = Frank-Kamenetskii (smooth), 1 = sharp radial layer
    double coarse_tol   = 1e-6;  // coarse FGMRES relative tolerance
    bool   full_stokes  = false; // --solve stokes: outer FGMRES on full saddle point
    KrylovPrec krylov_prec = KrylovPrec::DOUBLE; // --krylov-precision double|single|bf16 (full-Stokes only)
    for ( int i = 1; i + 1 < argc; ++i )
    {
        const std::string a = argv[i];
        if ( a == "--min-level" )           min_level   = std::atoi( argv[i + 1] );
        else if ( a == "--max-level" )      max_level   = std::atoi( argv[i + 1] );
        else if ( a == "--cheby-order" )    cheby_order = std::atoi( argv[i + 1] );
        else if ( a == "--max-cycles" )     max_cycles  = std::atoi( argv[i + 1] );
        else if ( a == "--perturb-amp" )    perturb_amp = std::atof( argv[i + 1] );
        else if ( a == "--gca" )            single_gca  = std::atoi( argv[i + 1] );
        else if ( a == "--rmu" )            single_rmu  = std::atof( argv[i + 1] );
        else if ( a == "--bc" )
            bc = ( std::string( argv[i + 1] ) == "freeslip" ) ? BCKind::FREESLIP : BCKind::DIRICHLET;
        else if ( a == "--gca-interp" )
            single_interp = ( std::string( argv[i + 1] ) == "linear" ) ? InterpolationMode::Linear
                                                                       : InterpolationMode::Constant;
        else if ( a == "--visc-profile" )
            visc_profile = ( std::string( argv[i + 1] ) == "layer" )   ? 1
                           : ( std::string( argv[i + 1] ) == "stotz" )  ? 2
                           : ( std::string( argv[i + 1] ) == "lin" )    ? 3
                                                                        : 0;
        else if ( a == "--coarse-tol" )    coarse_tol  = std::atof( argv[i + 1] );
        else if ( a == "--solve" )         full_stokes = ( std::string( argv[i + 1] ) == "stokes" );
        else if ( a == "--krylov-precision" )
        {
            const std::string p = argv[i + 1];
            krylov_prec = ( p == "bf16" || p == "half" )                 ? KrylovPrec::BF16
                          : ( p == "single" || p == "float" )           ? KrylovPrec::FLOAT
                                                                        : KrylovPrec::DOUBLE;
        }
    }
    bool verify_gca = false;
    for ( int i = 1; i < argc; ++i )
        if ( std::string( argv[i] ) == "--verify-gca" ) verify_gca = true;

    if ( verify_gca )
    {
        const double r = ( single_rmu > 0.0 ) ? single_rmu : 1.0;
        verify_gca_rap( max_level, r, InterpolationMode::Constant );
        verify_gca_rap( max_level, r, InterpolationMode::Linear );
        return 0;
    }
    const char* bc_name = ( bc == BCKind::DIRICHLET ) ? "dirichlet/dirichlet (no-slip)" : "freeslip/freeslip";

    // Published radial profiles (Stotz/Lin) are fixed (rmu irrelevant) => single run.
    std::vector< double > rmus = ( visc_profile == 2 || visc_profile == 3 )
                                     ? std::vector< double >{ 1.0 }
                                     : ( single_rmu > 0.0 )
                                           ? std::vector< double >{ single_rmu }
                                           : std::vector< double >{ 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6 };

    // A scheme = (gca mode, GCA interpolation, label).  Default sweep compares
    // re-discretized coarse ops against full GCA with constant vs linear P.
    struct Scheme { int gca; InterpolationMode interp; const char* name; };
    std::vector< Scheme > schemes;
    if ( single_gca >= 0 )
        schemes = { { single_gca, single_interp,
                      ( single_gca == 0 )
                          ? "DCA(re-discretized)"
                          : ( single_interp == InterpolationMode::Linear ? "GCA(linear)" : "GCA(constant)" ) } };
    else
        schemes = {
            { 0, InterpolationMode::Constant, "DCA(re-discretized)" },
            { 1, InterpolationMode::Constant, "GCA(constant)" },
        };

    util::logroot << "=== " << ( full_stokes ? "FULL STOKES outer-FGMRES (w-BFBT Schur) " : "A-block (viscous) MG " )
                  << "convergence vs viscosity contrast ===\n"
                  << "    solve = " << ( full_stokes ? "stokes (cycles = outer FGMRES iters)" : "ablock (cycles = MG V-cycles)" )
                  << ( full_stokes ? std::string( ", krylov-basis = " ) + krylov_prec_name( krylov_prec ) : std::string() )
                  << ", BC = " << bc_name
                  << ", viscosity = "
                  << ( visc_profile == 3   ? "Lin et al. 2022 radial profile (contrast ~1057)"
                       : visc_profile == 2 ? "Stotz et al. 2017 radial profile (contrast ~12000)"
                       : visc_profile == 1 ? "low-viscosity band (r=2.0, width 0.1)"
                                           : "Frank-Kamenetskii (smooth)" )
                  << ", levels " << min_level << ".." << max_level
                  << ", cheby_order=" << cheby_order
                  << ", max_cycles=" << max_cycles
                  << ", coarse_tol=" << coarse_tol
                  << ", perturb_amp=" << perturb_amp << "\n";

    auto summary = std::make_shared< util::Table >();

    for ( double rmu : rmus )
    {
        for ( const auto& s : schemes )
        {
            util::logroot << "\n--- rmu = " << rmu << ", " << s.name << " ---\n";

            const auto res =
                run_ablock_mg( min_level, max_level, cheby_order, perturb_amp, rmu, s.gca, max_cycles, bc, s.interp,
                               visc_profile, coarse_tol, full_stokes, krylov_prec );

            summary->add_row(
                { { "rmu", rmu },
                  { "gca", s.gca },
                  { "krylov", std::string( krylov_prec_name( krylov_prec ) ) },
                  { "scheme", std::string( s.name ) },
                  { "cycles", res.cycles },
                  { "final_rel_res", res.rel_residual_final },
                  { "asympt_rate", res.asymptotic_rate },
                  { "converged", res.converged ? 1 : 0 } } );
        }
    }

    util::logroot << "\n=== SUMMARY: A-block MG convergence ===\n";
    summary->select_columns( { "rmu", "krylov", "scheme", "cycles", "final_rel_res", "asympt_rate", "converged" } )
        .print_pretty();

    return 0;
}
