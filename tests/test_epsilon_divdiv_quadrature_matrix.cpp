// Quadrature study matrix: three manufactured solutions of increasing
// steepness/complexity (S1 smooth, S2 radial tanh front d=0.05, S3 undulating
// tanh front d=0.025), each measured on:
//   (1) 1pt-vs-6pt quadrature consistency (energy form) over levels,
//   (2) multigrid residual reduction per V-cycle (both operators, level 4),
//   (3) solution error vs analytic per V-cycle,
//   (4) discretization error over levels 2..4 (both operators).
//
// Viscosity mu = 1 everywhere; RHS generated symbolically (SymPy) for each
// solution; inhomogeneous Dirichlet BCs from the analytic field.
//
// Output rows:
//   CONS,<sol>,<level>,<h>,<rel_energy_err>
//   ITER,<sol>,<op>,<level>,<cycle>,<true_rel_residual>,<l2_err_vs_analytic>
//   DISC,<sol>,<op>,<level>,<h>,<l2_err>,<dofs>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen_hg.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_simple.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_simple_2pt.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_simple_stab.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::BoundaryConditions;
using grid::shell::DistributedDomain;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

// ---- analytic solutions -----------------------------------------------------

KOKKOS_INLINE_FUNCTION
void sol_eval( const int sol, const double px, const double py, const double pz,
               double& u0, double& u1, double& u2 )
{
    if ( sol == 1 )
    {
        u0 = Kokkos::sin( 2 * px ) * Kokkos::sin( 2 * pz ) * Kokkos::sinh( py );
        u1 = 2 * Kokkos::sin( 2 * py ) * Kokkos::sin( 2 * pz ) * Kokkos::sinh( px );
        u2 = 4 * Kokkos::sin( 2 * px ) * Kokkos::sin( 2 * py ) * Kokkos::sinh( pz );
    }
    else if ( sol == 2 )
    {
        const double r = Kokkos::sqrt( px * px + py * py + pz * pz );
        const double T = Kokkos::tanh( ( r - 0.75 ) / 0.05 );
        u0 = T * Kokkos::sin( 2 * px ) * Kokkos::cos( py );
        u1 = 2 * T * Kokkos::sin( 2 * py ) * Kokkos::cos( pz );
        u2 = 4 * T * Kokkos::sin( 2 * pz ) * Kokkos::cos( px );
    }
    else
    {
        const double r   = Kokkos::sqrt( px * px + py * py + pz * pz );
        const double phi = r - 0.75 - 0.08 * Kokkos::sin( 2 * px ) * Kokkos::cos( 2 * py ) * Kokkos::sin( pz );
        const double U   = Kokkos::tanh( phi / 0.025 );
        u0 = U * ( 1 + 0.3 * Kokkos::cos( 3 * py ) );
        u1 = 2 * U * ( 1 + 0.3 * Kokkos::cos( 3 * pz ) );
        u2 = 4 * U * ( 1 + 0.3 * Kokkos::cos( 3 * px ) );
    }
}

struct SolInterp
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataVec< double, 3 >                         data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    int                                                sol_;
    bool                                               only_boundary_;
    bool                                               bubble_;

    SolInterp(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataVec< double, 3 >&                         data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        int sol, bool only_boundary, bool bubble )
    : grid_( grid ), radii_( radii ), data_( data ), mask_( mask )
    , sol_( sol ), only_boundary_( only_boundary ), bubble_( bubble )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const bool on_boundary = util::has_flag( mask_( sd, x, y, r ), BOUNDARY );
        if ( only_boundary_ && !on_boundary )
            return;
        double u0, u1, u2;
        sol_eval( sol_, c( 0 ), c( 1 ), c( 2 ), u0, u1, u2 );
        double b = 1.0;
        if ( bubble_ )
        {
            const double rr = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
            b               = ( rr - 0.5 ) * ( 1.0 - rr ) / 0.0625;
        }
        data_( sd, x, y, r, 0 ) = b * u0;
        data_( sd, x, y, r, 1 ) = b * u1;
        data_( sd, x, y, r, 2 ) = b * u2;
    }
};

// ---- generated RHS interpolators -------------------------------------------

#define RHS_STRUCT( NAME, INCFILE )                                                       \
    struct NAME                                                                           \
    {                                                                                     \
        Grid3DDataVec< double, 3 > grid_;                                                 \
        Grid2DDataScalar< double > radii_;                                                \
        Grid4DDataVec< double, 3 > data_;                                                 \
        NAME( const Grid3DDataVec< double, 3 >& grid,                                     \
              const Grid2DDataScalar< double >& radii,                                    \
              const Grid4DDataVec< double, 3 >& data )                                    \
        : grid_( grid ), radii_( radii ), data_( data )                                   \
        {}                                                                                \
        KOKKOS_INLINE_FUNCTION                                                            \
        void operator()( const int sd, const int x, const int y, const int r ) const      \
        {                                                                                 \
            const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ ); \
            const real_t px = c( 0 ), py = c( 1 ), pz = c( 2 );                           \
            (void) px; (void) py; (void) pz;                                              \
            INCFILE                                                                       \
        }                                                                                 \
    };

#define INC_S1
struct RHS1
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    RHS1( const Grid3DDataVec< double, 3 >& grid,
          const Grid2DDataScalar< double >& radii,
          const Grid4DDataVec< double, 3 >& data )
    : grid_( grid ), radii_( radii ), data_( data )
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const real_t px = c( 0 ), py = c( 1 ), pz = c( 2 );
#include "matrix_rhs_s1.inc"
    }
};

struct RHS2
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    RHS2( const Grid3DDataVec< double, 3 >& grid,
          const Grid2DDataScalar< double >& radii,
          const Grid4DDataVec< double, 3 >& data )
    : grid_( grid ), radii_( radii ), data_( data )
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const real_t px = c( 0 ), py = c( 1 ), pz = c( 2 );
#include "matrix_rhs_s2.inc"
    }
};

struct RHS3
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    RHS3( const Grid3DDataVec< double, 3 >& grid,
          const Grid2DDataScalar< double >& radii,
          const Grid4DDataVec< double, 3 >& data )
    : grid_( grid ), radii_( radii ), data_( data )
    {}
    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const real_t px = c( 0 ), py = c( 1 ), pz = c( 2 );
#include "matrix_rhs_s3.inc"
    }
};

struct ZeroOnBoundaryV
{
    Grid4DDataVec< double, 3 >                         data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;

    ZeroOnBoundaryV(
        const Grid4DDataVec< double, 3 >&                         data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask )
    : data_( data ), mask_( mask )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        if ( util::has_flag( mask_( sd, x, y, r ), BOUNDARY ) )
            for ( int d = 0; d < 3; ++d )
                data_( sd, x, y, r, d ) = 0.0;
    }
};

template < linalg::OperatorLike OperatorT >
struct InverseDiagonalPreconditioner
{
    using OperatorType       = OperatorT;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType      = linalg::DstOf< OperatorType >;

    SolutionVectorType inv_diag_;

    explicit InverseDiagonalPreconditioner( const SolutionVectorType& d ) : inv_diag_( d ) {}

    void solve_impl( OperatorType& /*A*/, SolutionVectorType& x, const RHSVectorType& b )
    {
        linalg::assign( x, b );
        linalg::scale_in_place( x, inv_diag_ );
    }
};

using Op1pt = fe::wedge::operators::shell::EpsilonDivDivKerngenHG< double >;
using Op2ptK = fe::wedge::operators::shell::EpsilonDivDivKerngenHG< double, 3, double, 2 >;
using Op6pt = fe::wedge::operators::shell::EpsilonDivDivSimple< double, 3 >;
using Op2pt = fe::wedge::operators::shell::EpsilonDivDivSimple2pt< double, 3 >;
using OpStab = fe::wedge::operators::shell::EpsilonDivDivSimpleStab< double, 3 >;

template < typename Op >
Op make_op( const DistributedDomain&                                  domain,
            const Grid3DDataVec< double, 3 >&                         coords,
            const Grid2DDataScalar< double >&                         radii,
            const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& bmask,
            const Grid4DDataScalar< double >&                         k,
            const bool                                                dirichlet,
            const bool                                                diagonal )
{
    if constexpr ( std::is_same_v< Op, Op1pt > || std::is_same_v< Op, Op2ptK > )
    {
        BoundaryConditions bcs_d = { { CMB, DIRICHLET }, { SURFACE, DIRICHLET } };
        BoundaryConditions bcs_n = { { CMB, NEUMANN }, { SURFACE, NEUMANN } };
        Op op( domain, coords, radii, bmask, k, dirichlet ? bcs_d : bcs_n, diagonal );
        if ( std::getenv( "FORCE_SLOW_PATH" ) )
            op.set_kernel_path( Op::KernelPath::Slow );
        std::fprintf( stderr, "MAKEOP path=%d dirichlet=%d diagonal=%d\n",
                      static_cast< int >( op.kernel_path() ), int( dirichlet ), int( diagonal ) );
        return op;
    }
    else
    {
        return Op( domain, coords, radii, bmask, k, dirichlet, diagonal );
    }
}

// dispatch RHS interpolation for a solution id
void interp_rhs( const int sol, const DistributedDomain& domain,
                 const Grid3DDataVec< double, 3 >& coords,
                 const Grid2DDataScalar< double >& radii,
                 const Grid4DDataVec< double, 3 >& data )
{
    if ( sol == 1 )
        Kokkos::parallel_for( "rhs1", local_domain_md_range_policy_nodes( domain ),
                              RHS1( coords, radii, data ) );
    else if ( sol == 2 )
        Kokkos::parallel_for( "rhs2", local_domain_md_range_policy_nodes( domain ),
                              RHS2( coords, radii, data ) );
    else
        Kokkos::parallel_for( "rhs3", local_domain_md_range_policy_nodes( domain ),
                              RHS3( coords, radii, data ) );
    Kokkos::fence();
}

// ---- (1) consistency: energy-form 1pt-vs-6pt error at one level ------------

template < typename OpTest >
void consistency_t( const int sol, const int level, const char* op_label )
{
    const auto domain = DistributedDomain::create_uniform( level, level, 0.5, 1.0, 0, 0 );
    auto       masks  = grid::setup_node_ownership_mask_data( domain );
    auto       bmask  = grid::shell::setup_boundary_mask_data( domain );
    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto radii  = grid::shell::subdomain_shell_radii< double >( domain );

    VectorQ1Vec< double >    u( "u", domain, masks ), y1( "y1", domain, masks ),
        y6( "y6", domain, masks ), diff( "diff", domain, masks );
    VectorQ1Scalar< double > k( "k", domain, masks );
    linalg::assign( k, 1.0 );

    Kokkos::parallel_for( "interp u", local_domain_md_range_policy_nodes( domain ),
                          SolInterp( coords, radii, u.grid_data(), bmask, sol, false, true ) );
    Kokkos::fence();
    Kokkos::parallel_for( "zero ub", local_domain_md_range_policy_nodes( domain ),
                          ZeroOnBoundaryV( u.grid_data(), bmask ) );
    Kokkos::fence();

    OpTest A1 = make_op< OpTest >( domain, coords, radii, bmask, k.grid_data(), true, false );
    Op6pt  A6 = make_op< Op6pt >( domain, coords, radii, bmask, k.grid_data(), true, false );

    linalg::apply( A1, u, y1 );
    linalg::apply( A6, u, y6 );
    Kokkos::fence();
    Kokkos::parallel_for( "zb1", local_domain_md_range_policy_nodes( domain ),
                          ZeroOnBoundaryV( y1.grid_data(), bmask ) );
    Kokkos::parallel_for( "zb6", local_domain_md_range_policy_nodes( domain ),
                          ZeroOnBoundaryV( y6.grid_data(), bmask ) );
    Kokkos::fence();

    linalg::lincomb( diff, { 1.0, -1.0 }, { y1, y6 } );
    const double rel_en = std::abs( dot( u, diff ) ) / std::abs( dot( u, y6 ) );
    const double h      = 1.0 / static_cast< double >( 1 << level );
    std::printf( "CONS,S%d,%s,%d,%.6e,%.10e\n", sol, op_label, level, h, rel_en );
    std::fflush( stdout );
}

// ---- (2)+(3)+(4): instrumented MG solve ------------------------------------

template < typename Op >
void solve_mg( const int sol, const char* op_label, const int min_level, const int max_level,
               const int max_cycles )
{
    using ScalarType   = double;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;
    using Smoother     = linalg::solvers::Chebyshev< Op >;
    using CoarsePrec   = InverseDiagonalPreconditioner< Op >;
    using CoarseSolver = linalg::solvers::FGMRES< Op, CoarsePrec >;
    using MG = linalg::solvers::Multigrid< Op, Prolongation, Restriction, Smoother, CoarseSolver >;
    using Mass = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords;
    std::vector< Grid2DDataScalar< double > >                         radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        masks;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > bmasks;

    const int lat_sdr = std::getenv( "SCALE_LAT_SDR" ) ? std::atoi( std::getenv( "SCALE_LAT_SDR" ) ) : 0;
    const int rad_sdr = std::getenv( "SCALE_RAD_SDR" ) ? std::atoi( std::getenv( "SCALE_RAD_SDR" ) ) : 0;
    for ( int level = min_level; level <= max_level; ++level )
    {
        domains.push_back( DistributedDomain::create_uniform( level, level, 0.5, 1.0, lat_sdr, rad_sdr ) );
        coords.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains.back() ) );
        radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains.back() ) );
        masks.push_back( grid::setup_node_ownership_mask_data( domains.back() ) );
        bmasks.push_back( grid::shell::setup_boundary_mask_data( domains.back() ) );
    }
    const size_t num_levels = domains.size();
    const size_t fine       = num_levels - 1;

    std::vector< VectorQ1Scalar< ScalarType > > k;
    for ( size_t l = 0; l < num_levels; ++l )
    {
        k.emplace_back( "k_" + std::to_string( l ), domains[l], masks[l] );
        linalg::assign( k[l], 1.0 );
    }
    Kokkos::fence();

    std::vector< Op > A_c;
    for ( size_t l = 0; l + 1 < num_levels; ++l )
        A_c.push_back( make_op< Op >( domains[l], coords[l], radii[l], bmasks[l], k[l].grid_data(), true, false ) );
    Op A_fine = make_op< Op >( domains[fine], coords[fine], radii[fine], bmasks[fine], k[fine].grid_data(), true, false );

    std::vector< VectorQ1Vec< ScalarType > > inv_diag;
    for ( size_t l = 0; l < num_levels; ++l )
    {
        Op diag_op = make_op< Op >( domains[l], coords[l], radii[l], bmasks[l], k[l].grid_data(), true, true );
        inv_diag.emplace_back( "inv_diag_" + std::to_string( l ), domains[l], masks[l] );
        VectorQ1Vec< ScalarType > ones( "ones_" + std::to_string( l ), domains[l], masks[l] );
        linalg::assign( ones, 1.0 );
        linalg::apply( diag_op, ones, inv_diag.back() );
        linalg::invert_entries( inv_diag.back() );
    }

    std::vector< Prolongation >              P;
    std::vector< Restriction >               R;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg, tmp_mg_r, tmp_mg_e;
    for ( size_t l = 0; l < num_levels; ++l )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( l ), domains[l], masks[l] );
        if ( l < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( l ), domains[l], masks[l] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( l ), domains[l], masks[l] );
            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[l] );
        }
    }

    std::vector< Smoother > smoothers;
    for ( size_t l = 0; l < num_levels; ++l )
    {
        std::vector< VectorQ1Vec< ScalarType > > tmps;
        tmps.emplace_back( "cheb0_" + std::to_string( l ), domains[l], masks[l] );
        tmps.emplace_back( "cheb1_" + std::to_string( l ), domains[l], masks[l] );
        smoothers.emplace_back( 4, inv_diag[l], tmps, 3 );
    }

    auto          coarse_table = std::make_shared< util::Table >();
    constexpr int restart      = 30;
    std::vector< VectorQ1Vec< ScalarType > > coarse_tmps;
    for ( int i = 0; i < 2 * restart + 4; ++i )
        coarse_tmps.emplace_back( "coarse_tmp", domains[0], masks[0] );
    linalg::solvers::FGMRESOptions< ScalarType > copts;
    copts.restart                     = restart;
    copts.max_iterations              = 100;
    copts.relative_residual_tolerance = 1e-8;
    copts.absolute_residual_tolerance = 1e-16;
    CoarseSolver coarse( coarse_tmps, copts, coarse_table, CoarsePrec( inv_diag[0] ) );
    coarse.set_tag( "coarse" );

    auto mg_table = std::make_shared< util::Table >();
    MG   mg( P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse, 1, 1e-30 );
    mg.collect_statistics( mg_table );

    VectorQ1Vec< ScalarType > u( "u", domains[fine], masks[fine] );
    VectorQ1Vec< ScalarType > b( "b", domains[fine], masks[fine] );
    VectorQ1Vec< ScalarType > g( "g", domains[fine], masks[fine] );
    VectorQ1Vec< ScalarType > tmp( "tmp", domains[fine], masks[fine] );
    VectorQ1Vec< ScalarType > sol_v( "sol", domains[fine], masks[fine] );
    // The error is computed in the resid buffer after the residual norm,
    // saving one fine-level vector (needed to fit MT512 on 4 GPUs).
    VectorQ1Vec< ScalarType > resid( "resid", domains[fine], masks[fine] );

    Kokkos::parallel_for( "interp sol", local_domain_md_range_policy_nodes( domains[fine] ),
                          SolInterp( coords[fine], radii[fine], sol_v.grid_data(), bmasks[fine], sol, false, false ) );
    Kokkos::parallel_for( "interp g", local_domain_md_range_policy_nodes( domains[fine] ),
                          SolInterp( coords[fine], radii[fine], g.grid_data(), bmasks[fine], sol, true, false ) );
    Kokkos::fence();
    interp_rhs( sol, domains[fine], coords[fine], radii[fine], tmp.grid_data() );

    Mass M( domains[fine], coords[fine], radii[fine], false );
    linalg::apply( M, tmp, b );

    Op A_neumann      = make_op< Op >( domains[fine], coords[fine], radii[fine], bmasks[fine], k[fine].grid_data(), false, false );
    Op A_neumann_diag = make_op< Op >( domains[fine], coords[fine], radii[fine], bmasks[fine], k[fine].grid_data(), false, true );
    fe::strong_algebraic_dirichlet_enforcement_vectorlaplace_like(
        A_neumann, A_neumann_diag, g, tmp, b, bmasks[fine], BOUNDARY );
    Kokkos::fence();

    const auto num_dofs = kernels::common::count_masked< long >( masks[fine], grid::NodeOwnershipFlag::OWNED );

    linalg::assign( u, 0.0 );
    const double b_norm = std::sqrt( dot( b, b ) );

    const auto report = [&]( int cycle ) {
        linalg::apply( A_fine, u, resid );
        linalg::lincomb( resid, { 1.0, -1.0 }, { b, resid } );
        const double rel = std::sqrt( dot( resid, resid ) ) / b_norm;
        linalg::lincomb( resid, { 1.0, -1.0 }, { u, sol_v } );
        const double l2 = std::sqrt( dot( resid, resid ) / num_dofs );
        std::printf( "ITER,S%d,%s,%d,%d,%.10e,%.10e\n", sol, op_label, max_level, cycle, rel, l2 );
    };

    report( 0 );
    for ( int cycle = 1; cycle <= max_cycles; ++cycle )
    {
        linalg::solvers::solve( mg, A_fine, u, b );
        Kokkos::fence();
        report( cycle );
    }

    linalg::lincomb( resid, { 1.0, -1.0 }, { u, sol_v } );
    const double l2 = std::sqrt( dot( resid, resid ) / num_dofs );
    const double h  = 1.0 / static_cast< double >( 1 << max_level );
    std::printf( "DISC,S%d,%s,%d,%.6e,%.10e,%ld\n", sol, op_label, max_level, h, l2, num_dofs );

    // radial line cross-cut of the computed and analytic solutions
    auto uh0 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, u.grid_data().comp_[0] );
    auto uh1 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, u.grid_data().comp_[1] );
    auto uh2 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, u.grid_data().comp_[2] );
    auto sh0 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, sol_v.grid_data().comp_[0] );
    auto sh1 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, sol_v.grid_data().comp_[1] );
    auto sh2 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, sol_v.grid_data().comp_[2] );
    auto rh  = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, radii[fine] );
    const int nx = static_cast< int >( uh0.extent( 1 ) );
    const int ny = static_cast< int >( uh0.extent( 2 ) );
    const int nr = static_cast< int >( uh0.extent( 3 ) );
    const int xi = nx / 2, yi = ny / 2;
    for ( int r = 0; r < nr; ++r )
        std::printf( "PROF,S%d,%s,%d,%d,%.8e,%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                     sol, op_label, max_level, r, rh( 0, r ),
                     uh0( 0, xi, yi, r ), uh1( 0, xi, yi, r ), uh2( 0, xi, yi, r ),
                     sh0( 0, xi, yi, r ), sh1( 0, xi, yi, r ), sh2( 0, xi, yi, r ) );
    std::fflush( stdout );

    if ( std::getenv( "DUMP_FIELD" ) )
    {
        // full nodal field with coordinates (all subdomains, all nodes)
        auto ch = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords[fine] );
        char fname[128];
        std::snprintf( fname, sizeof( fname ), "field_S%d_%s_L%d.csv", sol, op_label, max_level );
        FILE* fp = std::fopen( fname, "w" );
        std::fprintf( fp, "x,y,z,u0,u1,u2,s0,s1,s2\n" );
        const int nsd = static_cast< int >( uh0.extent( 0 ) );
        for ( int sd = 0; sd < nsd; ++sd )
            for ( int xx = 0; xx < nx; ++xx )
                for ( int yy = 0; yy < ny; ++yy )
                    for ( int rr = 0; rr < nr; ++rr )
                    {
                        const double px = ch( sd, xx, yy, 0 ) * rh( sd, rr );
                        const double py = ch( sd, xx, yy, 1 ) * rh( sd, rr );
                        const double pz = ch( sd, xx, yy, 2 ) * rh( sd, rr );
                        std::fprintf( fp, "%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e\n",
                                      px, py, pz,
                                      uh0( sd, xx, yy, rr ), uh1( sd, xx, yy, rr ), uh2( sd, xx, yy, rr ),
                                      sh0( sd, xx, yy, rr ), sh1( sd, xx, yy, rr ), sh2( sd, xx, yy, rr ) );
                    }
        std::fclose( fp );
    }
}


// ---- production configuration: FGMRES outer, 1 V-cycle as preconditioner ---

template < typename Op >
void solve_fgmres( const int sol, const char* op_label, const int min_level, const int max_level )
{
    using ScalarType   = double;
    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;
    using Smoother     = linalg::solvers::Chebyshev< Op >;
    using CoarsePrec   = InverseDiagonalPreconditioner< Op >;
    using CoarseSolver = linalg::solvers::FGMRES< Op, CoarsePrec >;
    using MG = linalg::solvers::Multigrid< Op, Prolongation, Restriction, Smoother, CoarseSolver >;
    using OuterSolver = linalg::solvers::FGMRES< Op, MG >;
    using Mass = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords;
    std::vector< Grid2DDataScalar< double > >                         radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        masks;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > bmasks;

    const int lat_sdr = std::getenv( "SCALE_LAT_SDR" ) ? std::atoi( std::getenv( "SCALE_LAT_SDR" ) ) : 0;
    const int rad_sdr = std::getenv( "SCALE_RAD_SDR" ) ? std::atoi( std::getenv( "SCALE_RAD_SDR" ) ) : 0;
    for ( int level = min_level; level <= max_level; ++level )
    {
        domains.push_back( DistributedDomain::create_uniform( level, level, 0.5, 1.0, lat_sdr, rad_sdr ) );
        coords.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains.back() ) );
        radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains.back() ) );
        masks.push_back( grid::setup_node_ownership_mask_data( domains.back() ) );
        bmasks.push_back( grid::shell::setup_boundary_mask_data( domains.back() ) );
    }
    const size_t num_levels = domains.size();
    const size_t fine       = num_levels - 1;

    std::vector< VectorQ1Scalar< ScalarType > > k;
    for ( size_t l = 0; l < num_levels; ++l )
    {
        k.emplace_back( "k_" + std::to_string( l ), domains[l], masks[l] );
        linalg::assign( k[l], 1.0 );
    }
    Kokkos::fence();

    std::vector< Op > A_c;
    for ( size_t l = 0; l + 1 < num_levels; ++l )
        A_c.push_back( make_op< Op >( domains[l], coords[l], radii[l], bmasks[l], k[l].grid_data(), true, false ) );
    Op A_fine = make_op< Op >( domains[fine], coords[fine], radii[fine], bmasks[fine], k[fine].grid_data(), true, false );

    std::vector< VectorQ1Vec< ScalarType > > inv_diag;
    for ( size_t l = 0; l < num_levels; ++l )
    {
        Op diag_op = make_op< Op >( domains[l], coords[l], radii[l], bmasks[l], k[l].grid_data(), true, true );
        inv_diag.emplace_back( "inv_diag_" + std::to_string( l ), domains[l], masks[l] );
        VectorQ1Vec< ScalarType > ones( "ones_" + std::to_string( l ), domains[l], masks[l] );
        linalg::assign( ones, 1.0 );
        linalg::apply( diag_op, ones, inv_diag.back() );
        linalg::invert_entries( inv_diag.back() );
    }

    std::vector< Prolongation >              P;
    std::vector< Restriction >               R;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg, tmp_mg_r, tmp_mg_e;
    for ( size_t l = 0; l < num_levels; ++l )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( l ), domains[l], masks[l] );
        if ( l < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( l ), domains[l], masks[l] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( l ), domains[l], masks[l] );
            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[l] );
        }
    }

    std::vector< Smoother > smoothers;
    for ( size_t l = 0; l < num_levels; ++l )
    {
        std::vector< VectorQ1Vec< ScalarType > > tmps;
        tmps.emplace_back( "cheb0_" + std::to_string( l ), domains[l], masks[l] );
        tmps.emplace_back( "cheb1_" + std::to_string( l ), domains[l], masks[l] );
        smoothers.emplace_back( 4, inv_diag[l], tmps, 3 );
    }

    auto          coarse_table = std::make_shared< util::Table >();
    constexpr int restart      = 30;
    std::vector< VectorQ1Vec< ScalarType > > coarse_tmps;
    for ( int i = 0; i < 2 * restart + 4; ++i )
        coarse_tmps.emplace_back( "coarse_tmp", domains[0], masks[0] );
    linalg::solvers::FGMRESOptions< ScalarType > copts;
    copts.restart                     = restart;
    copts.max_iterations              = 100;
    copts.relative_residual_tolerance = 1e-8;
    copts.absolute_residual_tolerance = 1e-16;
    CoarseSolver coarse( coarse_tmps, copts, coarse_table, CoarsePrec( inv_diag[0] ) );
    coarse.set_tag( "coarse" );

    auto mg_table = std::make_shared< util::Table >();
    MG   mg( P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse, 1, 1e-30 );

    VectorQ1Vec< ScalarType > u( "u", domains[fine], masks[fine] );
    VectorQ1Vec< ScalarType > b( "b", domains[fine], masks[fine] );
    VectorQ1Vec< ScalarType > g( "g", domains[fine], masks[fine] );
    VectorQ1Vec< ScalarType > tmp( "tmp", domains[fine], masks[fine] );
    VectorQ1Vec< ScalarType > sol_v( "sol", domains[fine], masks[fine] );
    VectorQ1Vec< ScalarType > err( "err", domains[fine], masks[fine] );

    Kokkos::parallel_for( "interp sol", local_domain_md_range_policy_nodes( domains[fine] ),
                          SolInterp( coords[fine], radii[fine], sol_v.grid_data(), bmasks[fine], sol, false, false ) );
    Kokkos::parallel_for( "interp g", local_domain_md_range_policy_nodes( domains[fine] ),
                          SolInterp( coords[fine], radii[fine], g.grid_data(), bmasks[fine], sol, true, false ) );
    Kokkos::fence();
    interp_rhs( sol, domains[fine], coords[fine], radii[fine], tmp.grid_data() );

    Mass M( domains[fine], coords[fine], radii[fine], false );
    linalg::apply( M, tmp, b );

    Op A_neumann      = make_op< Op >( domains[fine], coords[fine], radii[fine], bmasks[fine], k[fine].grid_data(), false, false );
    Op A_neumann_diag = make_op< Op >( domains[fine], coords[fine], radii[fine], bmasks[fine], k[fine].grid_data(), false, true );
    fe::strong_algebraic_dirichlet_enforcement_vectorlaplace_like(
        A_neumann, A_neumann_diag, g, tmp, b, bmasks[fine], BOUNDARY );
    Kokkos::fence();

    // outer FGMRES with the single V-cycle as right preconditioner
    auto outer_table = std::make_shared< util::Table >();
    std::vector< VectorQ1Vec< ScalarType > > outer_tmps;
    for ( int i = 0; i < 2 * restart + 4; ++i )
        outer_tmps.emplace_back( "outer_tmp", domains[fine], masks[fine] );
    linalg::solvers::FGMRESOptions< ScalarType > oopts;
    oopts.restart                     = restart;
    oopts.max_iterations              = 40;
    oopts.relative_residual_tolerance = 1e-10;
    oopts.absolute_residual_tolerance = 1e-16;
    OuterSolver outer( outer_tmps, oopts, outer_table, mg );
    outer.set_tag( "outer" );

    linalg::assign( u, 0.0 );
    linalg::solvers::solve( outer, A_fine, u, b );
    Kokkos::fence();

    {
        const auto q   = outer_table->query_rows_equals( "tag", "outer" );
        int        idx = 0;
        for ( const auto& row : q.rows() )
        {
            if ( !row.count( "relative_residual" ) )
                continue;
            std::printf( "FITR,S%d,%s,%d,%d,%.10e\n", sol, op_label, max_level, idx++,
                         std::get< double >( row.at( "relative_residual" ) ) );
        }
    }

    const auto num_dofs = kernels::common::count_masked< long >( masks[fine], grid::NodeOwnershipFlag::OWNED );
    linalg::lincomb( err, { 1.0, -1.0 }, { u, sol_v } );
    const double l2 = std::sqrt( dot( err, err ) / num_dofs );
    std::printf( "FERR,S%d,%s,%d,%.10e,%ld\n", sol, op_label, max_level, l2, num_dofs );
    std::fflush( stdout );
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        const bool prof_only  = std::getenv( "PROF_ONLY" ) != nullptr;
        const bool run_2pt    = std::getenv( "RUN_2PT" ) != nullptr;
        const bool run_fgmres = std::getenv( "RUN_FGMRES" ) != nullptr;
        if ( const char* diff_env = std::getenv( "RUN_OPDIFF" ) )
        {
            // Diagnostic: elementwise-apply diff between the kerngen operator
            // (stab from STAB_C_KERNGEN) and the reference Simple-stab operator
            // (stab from STAB_C), on the interpolated S1 field. Prints per-shell
            // L2 of the difference to localize discrepancies.
            const int  level  = std::atoi( diff_env );
            const auto domain = DistributedDomain::create_uniform( level, level, 0.5, 1.0, 0, 0 );
            const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
            const auto radii  = grid::shell::subdomain_shell_radii< double >( domain );
            const auto masks  = grid::setup_node_ownership_mask_data( domain );
            const auto bmask  = grid::shell::setup_boundary_mask_data( domain );

            VectorQ1Scalar< double > k( "k", domain, masks );
            linalg::assign( k, 1.0 );

            VectorQ1Vec< double > u( "u", domain, masks ), y1( "y1", domain, masks ), y2( "y2", domain, masks );
            const bool bubble = std::getenv( "OPDIFF_NOBUBBLE" ) == nullptr;
            Kokkos::parallel_for( "interp sol", local_domain_md_range_policy_nodes( domain ),
                                  SolInterp( coords, radii, u.grid_data(), bmask, 1, false, bubble ) );
            Kokkos::fence();
            if ( std::getenv( "OPDIFF_2PT" ) )
            {
                // 2-qp kerngen: fast DN path vs reference. Reference is
                // Simple2pt by default, or kerngen-2pt forced to Slow with
                // OPDIFF_REF_SLOW (internal fast-vs-slow consistency).
                Op2ptK A_f = make_op< Op2ptK >( domain, coords, radii, bmask, k.grid_data(), true, false );
                linalg::apply( A_f, u, y1 );
                if ( std::getenv( "OPDIFF_REF_SLOW" ) )
                {
                    Op2ptK A_r = make_op< Op2ptK >( domain, coords, radii, bmask, k.grid_data(), true, false );
                    A_r.set_kernel_path( Op2ptK::KernelPath::Slow );
                    linalg::apply( A_r, u, y2 );
                }
                else
                {
                    Op2pt A_r( domain, coords, radii, bmask, k.grid_data(), true, false );
                    linalg::apply( A_r, u, y2 );
                }
            }
            else
            {
                Op1pt A_k = make_op< Op1pt >( domain, coords, radii, bmask, k.grid_data(), true, false );
                Op1pt A_s = make_op< Op1pt >( domain, coords, radii, bmask, k.grid_data(), true, false );
                A_s.set_kernel_path( Op1pt::KernelPath::Slow );
                linalg::apply( A_k, u, y1 );
                linalg::apply( A_s, u, y2 );
            }
            Kokkos::fence();

            auto h10 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, y1.grid_data().comp_[0] );
            auto h11 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, y1.grid_data().comp_[1] );
            auto h12 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, y1.grid_data().comp_[2] );
            auto h20 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, y2.grid_data().comp_[0] );
            auto h21 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, y2.grid_data().comp_[1] );
            auto h22 = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, y2.grid_data().comp_[2] );

            auto hm = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, masks );

            const int nr = static_cast< int >( h10.extent( 3 ) );
            for ( int r = 0; r < nr; ++r )
            {
                double d2 = 0.0, n2 = 0.0;
                for ( size_t sd = 0; sd < h10.extent( 0 ); ++sd )
                    for ( size_t xx = 0; xx < h10.extent( 1 ); ++xx )
                        for ( size_t yy = 0; yy < h10.extent( 2 ); ++yy )
                        {
                            if ( hm( sd, xx, yy, r ) != grid::NodeOwnershipFlag::OWNED )
                                continue;
                            const double d0 = h10( sd, xx, yy, r ) - h20( sd, xx, yy, r );
                            const double d1 = h11( sd, xx, yy, r ) - h21( sd, xx, yy, r );
                            const double dd2 = h12( sd, xx, yy, r ) - h22( sd, xx, yy, r );
                            d2 += d0 * d0 + d1 * d1 + dd2 * dd2;
                            n2 += h20( sd, xx, yy, r ) * h20( sd, xx, yy, r ) +
                                  h21( sd, xx, yy, r ) * h21( sd, xx, yy, r ) +
                                  h22( sd, xx, yy, r ) * h22( sd, xx, yy, r );
                        }
                std::printf( "OPDIFF,%d,%d,%.6e,%.6e\n", level, r, std::sqrt( d2 ), std::sqrt( n2 ) );
            }

            // Top-10 worst nodes to localize the discrepancy pattern.
            struct Worst { double d; int sd, x, y, r, c; };
            std::vector< Worst > worst;
            for ( size_t sd = 0; sd < h10.extent( 0 ); ++sd )
                for ( size_t xx = 0; xx < h10.extent( 1 ); ++xx )
                    for ( size_t yy = 0; yy < h10.extent( 2 ); ++yy )
                        for ( int r = 0; r < nr; ++r )
                        {
                            if ( hm( sd, xx, yy, r ) != grid::NodeOwnershipFlag::OWNED )
                                continue;
                            const double dv[3] = { h10( sd, xx, yy, r ) - h20( sd, xx, yy, r ),
                                                   h11( sd, xx, yy, r ) - h21( sd, xx, yy, r ),
                                                   h12( sd, xx, yy, r ) - h22( sd, xx, yy, r ) };
                            for ( int c = 0; c < 3; ++c )
                                worst.push_back( { std::abs( dv[c] ), int( sd ), int( xx ), int( yy ), r, c } );
                        }
            std::partial_sort( worst.begin(), worst.begin() + std::min< size_t >( 10, worst.size() ),
                               worst.end(), []( const Worst& a, const Worst& b ) { return a.d > b.d; } );
            for ( int i = 0; i < 10 && i < int( worst.size() ); ++i )
                std::printf( "WORST,sd=%d,x=%d,y=%d,r=%d,c=%d,|d|=%.4e\n", worst[i].sd, worst[i].x,
                             worst[i].y, worst[i].r, worst[i].c, worst[i].d );
        }
        else if ( const char* scale_env = std::getenv( "RUN_SCALE" ) )
        {
            // Scaling study: stand-alone MG on the kerngen operator for the
            // smooth solution S1 on one fine level (isotropic lat=rad=level).
            // RUN_QP=2 selects the 2-radial-qp kerngen instantiation; otherwise
            // the label follows the STAB_C_KERNGEN env: 1pt (unset) vs 1ptK.
            const int  lmax   = std::atoi( scale_env );
            const bool two_qp = std::getenv( "RUN_QP" ) && std::atoi( std::getenv( "RUN_QP" ) ) == 2;
            if ( two_qp )
                solve_mg< Op2ptK >( 1, "2ptK", 1, lmax, 12 );
            else
                solve_mg< Op1pt >( 1, std::getenv( "STAB_C_KERNGEN" ) ? "1ptK" : "1pt", 1, lmax, 12 );
        }
        else
        for ( int sol = 1; sol <= 3; ++sol )
        {
            if ( std::getenv( "RUN_STAB" ) )
            {
                solve_mg< OpStab >( sol, "1ptS", 1, 4, 12 );
                solve_fgmres< OpStab >( sol, "1ptS", 1, 4 );
                continue;
            }
            if ( std::getenv( "RUN_KSTAB" ) )
            {
                // kerngen with fused stabilization (STAB_C_KERNGEN must be set)
                for ( int level = 1; level <= 5; ++level )
                    consistency_t< Op1pt >( sol, level, "1ptK" );
                for ( int max_level = 2; max_level <= 4; ++max_level )
                {
                    const int cycles = ( max_level == 4 ) ? 12 : 8;
                    solve_mg< Op1pt >( sol, "1ptK", 1, max_level, cycles );
                }
                solve_fgmres< Op1pt >( sol, "1ptK", 1, 4 );
                continue;
            }
            if ( run_fgmres )
            {
                solve_fgmres< Op1pt >( sol, "1pt", 1, 4 );
                solve_fgmres< Op2pt >( sol, "2pt", 1, 4 );
                solve_fgmres< Op6pt >( sol, "6pt", 1, 4 );
                continue;
            }
            if ( run_2pt )
            {
                for ( int level = 1; level <= 5; ++level )
                    consistency_t< Op2pt >( sol, level, "2pt" );
                for ( int max_level = 2; max_level <= 4; ++max_level )
                {
                    const int cycles = ( max_level == 4 ) ? 12 : 8;
                    solve_mg< Op2pt >( sol, "2pt", 1, max_level, cycles );
                }
                continue;
            }
            if ( prof_only )
            {
                solve_mg< Op1pt >( sol, "1pt", 1, 4, 5 );
                solve_mg< Op6pt >( sol, "6pt", 1, 4, 5 );
                continue;
            }
            for ( int level = 1; level <= 5; ++level )
                consistency_t< Op1pt >( sol, level, "1pt" );

            for ( int max_level = 2; max_level <= 4; ++max_level )
            {
                const int cycles = ( max_level == 4 ) ? 12 : 8;
                solve_mg< Op1pt >( sol, "1pt", 1, max_level, cycles );
                solve_mg< Op6pt >( sol, "6pt", 1, max_level, cycles );
            }
        }
    }
    return 0;
}
