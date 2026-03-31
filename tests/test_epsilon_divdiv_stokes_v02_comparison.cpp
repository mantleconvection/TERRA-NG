
// Comparison test for v02 vs v02b epsilon_divdiv operators in Stokes.
// Runs convergence study for both and outputs CSV for plotting.

#include "../src/terra/communication/shell/communication.hpp"

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
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

// Historical operator variants
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v02_split_dimij.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v02b_single_quadpoint.hpp"

#include "grid/shell/bit_masks.hpp"

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
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"

#include "util/init.hpp"
#include "util/table.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>

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
using terra::grid::shell::BoundaryConditions;

// =============================================================================
// Templated Stokes operator that accepts any Block11 type
// =============================================================================

template < typename Block11T, typename ScalarT = double, int VecDim = 3 >
class EpsDivDivStokesGeneric
{
  public:
    using SrcVectorType = linalg::VectorQ1IsoQ2Q1< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorQ1IsoQ2Q1< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

    using Block11Type = Block11T;
    using Block12Type = fe::wedge::operators::shell::Gradient< ScalarType >;
    using Block21Type = fe::wedge::operators::shell::Divergence< ScalarType >;
    using Block22Type = fe::wedge::operators::shell::Zero< ScalarType >;

  private:
    Block11Type                                        A_;
    fe::wedge::operators::shell::Gradient< ScalarType >   B_T_;
    fe::wedge::operators::shell::Divergence< ScalarType > B_;
    fe::wedge::operators::shell::Zero< ScalarType >       O_;
    bool diagonal_;

  public:
    EpsDivDivStokesGeneric(
        const grid::shell::DistributedDomain&                           domain_fine,
        const grid::shell::DistributedDomain&                           domain_coarse,
        const grid::Grid3DDataVec< ScalarType, 3 >&                     grid,
        const grid::Grid2DDataScalar< ScalarType >&                     radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const grid::Grid4DDataScalar< ScalarType >&                     k,
        BoundaryConditions                                              bcs,
        bool                                                            diagonal )
    : A_( domain_fine, grid, radii, mask, k, bcs, diagonal )
    , B_T_( domain_fine, domain_coarse, grid, radii, mask, bcs )
    , B_( domain_fine, domain_coarse, grid, radii, mask, bcs )
    , diagonal_( diagonal )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( !diagonal_ )
        {
            A_.set_operator_apply_and_communication_modes(
                linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::SkipCommunication );
        }

        B_T_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Add, linalg::OperatorCommunicationMode::CommunicateAdditively );
        B_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );

        apply( A_, src.block_1(), dst.block_1() );

        if ( !diagonal_ )
        {
            apply( B_T_, src.block_2(), dst.block_1() );
            apply( B_, src.block_1(), dst.block_2() );
        }

        A_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );
        B_T_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );
        B_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );
    }

    const Block11Type& block_11() const { return A_; }
    const Block12Type& block_12() const { return B_T_; }
    const Block21Type& block_21() const { return B_; }
    const Block22Type& block_22() const { return O_; }

    Block11Type& block_11() { return A_; }
    Block12Type& block_12() { return B_T_; }
    Block21Type& block_21() { return B_; }
    Block22Type& block_22() { return O_; }
};

// =============================================================================
// Interpolators
// =============================================================================

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
    : grid_( grid ), radii_( radii ), data_u_( data_u ), mask_( mask ), only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || on_boundary )
        {
            data_u_( local_subdomain_id, x, y, r, 0 ) = -4 * Kokkos::cos( 4 * coords( 2 ) );
            data_u_( local_subdomain_id, x, y, r, 1 ) =  8 * Kokkos::cos( 8 * coords( 0 ) );
            data_u_( local_subdomain_id, x, y, r, 2 ) = -2 * Kokkos::cos( 2 * coords( 1 ) );
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
    : grid_( grid ), radii_( radii ), data_p_( data_p ), mask_( mask ), only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || on_boundary )
        {
            data_p_( local_subdomain_id, x, y, r ) =
                Kokkos::sin( 4 * coords( 0 ) ) * Kokkos::sin( 8 * coords( 1 ) ) * Kokkos::sin( 2 * coords( 2 ) );
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
    : grid_( grid ), radii_( radii ), data_( data )
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
    : grid_( grid ), radii_( radii ), data_( data ), kmax_( kmax )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        data_( local_subdomain_id, x, y, r ) = 2 + Kokkos::sin( coords( 2 ) );
    }
};

// =============================================================================
// Templated test function
// =============================================================================

template < typename StokesType, typename ViscousType >
std::tuple< double, double, int >
run_test( double kmax,
          int min_level,
          int max_level,
          const std::shared_ptr< util::Table >& table,
          const std::string& variant_name )
{
    using ScalarType = double;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    ScalarType r_min = 0.5;
    ScalarType r_max = 1.0;

    int level_subdomains = 0;

    util::logroot << "[" << variant_name << "] Allocating domains for level " << max_level << " ...\n";
    for ( int level = min_level; level <= max_level; level++ )
    {
        const int idx = level - min_level;
        domains.push_back(
            DistributedDomain::create_uniform(
                level, level, r_min, r_max, level_subdomains, level_subdomains ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const auto num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    std::vector< std::string > stok_vec_names = { "u", "f", "solution", "error" };
    constexpr int num_stok_tmps = 8;

    for ( int i = 0; i < num_stok_tmps; i++ )
        stok_vec_names.push_back( "tmp_" + std::to_string( i ) );

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

    for ( int level = 0; level < (int) num_levels; level++ )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < (int) num_levels - 1 )
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

    using Gradient    = typename StokesType::Block12Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    VectorQ1Scalar< ScalarType > k( "k", domains[velocity_level], mask_data[velocity_level] );

    Kokkos::parallel_for(
        "coefficient interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        KInterpolator( coords_shell[velocity_level], coords_radii[velocity_level], k.grid_data(), kmax ) );

    StokesType K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k.grid_data(),
        bcs,
        false );

    StokesType K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k.grid_data(),
        bcs_neumann,
        false );

    StokesType K_neumann_diag(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k.grid_data(),
        bcs_neumann,
        true );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // MG hierarchy
    std::vector< ViscousType >  A_diag;
    std::vector< ViscousType >  A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;
    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;

    for ( int level = 0; level < (int) num_levels; level++ )
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

        if ( level < (int) num_levels - 1 )
        {
            A_c.emplace_back(
                domains[level],
                coords_shell[level],
                coords_radii[level],
                boundary_mask_data[level],
                k_c.grid_data(),
                bcs,
                false );

            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // Interpolate solution and RHS
    Kokkos::parallel_for(
        "solution interpolation (velocity)",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level],
            stok_vecs["solution"].block_1().grid_data(),
            boundary_mask_data[velocity_level], false ) );

    Kokkos::parallel_for(
        "solution interpolation (pressure)",
        local_domain_md_range_policy_nodes( domains[pressure_level] ),
        SolutionPressureInterpolator(
            coords_shell[pressure_level], coords_radii[pressure_level],
            stok_vecs["solution"].block_2().grid_data(),
            boundary_mask_data[pressure_level], false ) );

    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        RHSVelocityInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level],
            stok_vecs["tmp_1"].block_1().grid_data() ) );

    linalg::apply( M, stok_vecs["tmp_1"].block_1(), stok_vecs["f"].block_1() );

    Kokkos::parallel_for(
        "boundary interpolation (velocity)",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        SolutionVelocityInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level],
            stok_vecs["tmp_0"].block_1().grid_data(),
            boundary_mask_data[velocity_level], true ) );

    fe::strong_algebraic_velocity_dirichlet_enforcement_stokes_like(
        K_neumann, K_neumann_diag,
        stok_vecs["tmp_0"], stok_vecs["tmp_1"],
        stok_vecs["f"],
        boundary_mask_data[velocity_level], BOUNDARY );

    // Smoothers
    using Smoother = linalg::solvers::Jacobi< ViscousType >;
    std::vector< Smoother > smoothers;

    for ( int level = 0; level < (int) num_levels; level++ )
    {
        inverse_diagonals.emplace_back(
            "inverse_diagonal_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( level ), domains[level], mask_data[level] );
        linalg::assign( tmp, 1.0 );

        if ( level == (int) num_levels - 1 )
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

        constexpr auto smoother_prepost = 3;
        VectorQ1Vec< ScalarType > tmp_pi_0( "tmp_pi_0" + std::to_string( level ), domains[level], mask_data[level] );
        VectorQ1Vec< ScalarType > tmp_pi_1( "tmp_pi_1" + std::to_string( level ), domains[level], mask_data[level] );
        double max_ev = 0.0;

        if ( level == (int) num_levels - 1 )
        {
            DiagonallyScaledOperator< ViscousType > inv_diag_A( K.block_11(), inverse_diagonals[level] );
            max_ev = power_iteration< DiagonallyScaledOperator< ViscousType > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
        }
        else
        {
            DiagonallyScaledOperator< ViscousType > inv_diag_A( A_c[level], inverse_diagonals[level] );
            max_ev = power_iteration< DiagonallyScaledOperator< ViscousType > >( inv_diag_A, tmp_pi_0, tmp_pi_1, 100 );
        }

        const auto omega_opt = 2.0 / ( 1.3 * max_ev );
        smoothers.emplace_back( inverse_diagonals[level], smoother_prepost, tmp_mg[level], omega_opt );
    }

    using CoarseGridSolver = linalg::solvers::PCG< ViscousType >;

    std::vector< VectorQ1Vec< ScalarType > > coarse_grid_tmps;
    for ( int i = 0; i < 4; i++ )
        coarse_grid_tmps.emplace_back( "tmp_coarse_grid", domains[0], mask_data[0] );

    auto solver_table_cg = std::make_shared< util::Table >();
    CoarseGridSolver coarse_grid_solver(
        linalg::solvers::IterativeSolverParameters{ 1000, 1e-6, 1e-16 }, solver_table_cg, coarse_grid_tmps );

    constexpr auto num_mg_cycles = 1;

    using PrecVisc =
        linalg::solvers::Multigrid< ViscousType, Prolongation, Restriction, Smoother, CoarseGridSolver >;

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
            "inverse_diagonal_tmp_pmass", domains[pressure_level], mask_data[pressure_level] );
        linalg::assign( tmp, 1.0 );
        linalg::apply( pmass, tmp, lumped_diagonal_pmass );
    }

    using PrecSchur = linalg::solvers::DiagonalSolver< PressureMass >;
    PrecSchur inv_lumped_pmass( lumped_diagonal_pmass );

    using PrecStokes =
        linalg::solvers::BlockTriangularPreconditioner2x2<
            StokesType, ViscousType, PressureMass, Gradient, PrecVisc, PrecSchur >;

    VectorQ1IsoQ2Q1< ScalarType > triangular_prec_tmp(
        "triangular_prec_tmp",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    PrecStokes prec_stokes( K.block_11(), pmass, K.block_12(), triangular_prec_tmp, prec_11, inv_lumped_pmass );

    const int iters = 500;
    constexpr auto num_tmps_fgmres = 500;
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
    linalg::solvers::FGMRES< StokesType, PrecStokes > fgmres( tmp_fgmres, fgmres_options, solver_table, prec_stokes );

    util::logroot << "[" << variant_name << "] Solve ...\n";
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

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error_velocity =
        std::sqrt( dot( error.block_1(), error.block_1() ) / static_cast< double >( num_dofs_velocity ) );
    const auto l2_error_pressure =
        std::sqrt( dot( error.block_2(), error.block_2() ) / static_cast< double >( num_dofs_pressure ) );

    table->add_row(
        { { "variant", variant_name },
          { "level", max_level },
          { "dofs_vel", num_dofs_velocity },
          { "l2_error_vel", l2_error_velocity },
          { "dofs_pre", num_dofs_pressure },
          { "l2_error_pre", l2_error_pressure },
          { "h_vel", ( r_max - r_min ) / std::pow( 2, (int) velocity_level ) },
          { "h_p", ( r_max - r_min ) / std::pow( 2, (int) pressure_level ) } } );

    util::logroot << "[" << variant_name << "] level=" << max_level
                  << " l2_error_vel=" << l2_error_velocity
                  << " l2_error_pre=" << l2_error_pressure << "\n";

    return { l2_error_velocity, l2_error_pressure,
             static_cast< int >( solver_table->query_rows_equals( "tag", "fgmres_solver" ).rows().size() ) };
}

// =============================================================================
// main
// =============================================================================

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    const int max_level = 6;
    const int min_level = 2;
    const double kmax   = 1.0;

    auto table = std::make_shared< util::Table >();

    // --- v02 (6-point quadrature) ---
    using V02 = fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV02SplitDimij< double >;
    using StokesV02 = EpsDivDivStokesGeneric< V02 >;

    for ( int level = min_level + 1; level <= max_level; ++level )
    {
        run_test< StokesV02, V02 >( kmax, min_level, level, table, "v02" );
    }

    // --- v02b (1-point quadrature) ---
    using V02b = fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV02bSingleQuadpoint< double >;
    using StokesV02b = EpsDivDivStokesGeneric< V02b >;

    for ( int level = min_level + 1; level <= max_level; ++level )
    {
        run_test< StokesV02b, V02b >( kmax, min_level, level, table, "v02b" );
    }

    // Print summary table
    util::logroot << "\n=== CONVERGENCE SUMMARY ===\n";
    table->select_columns( { "variant", "level", "dofs_vel", "l2_error_vel", "dofs_pre", "l2_error_pre", "h_vel", "h_p" } )
        .print_pretty();

    // Write CSV for plotting
    {
        std::ofstream csv( "v02_v02b_comparison.csv" );
        table->select_columns( { "variant", "level", "dofs_vel", "l2_error_vel", "dofs_pre", "l2_error_pre", "h_vel", "h_p" } )
            .print_csv( csv );
    }

    util::logroot << "CSV written to v02_v02b_comparison.csv\n";

    MPI_Finalize();
    return 0;
}
