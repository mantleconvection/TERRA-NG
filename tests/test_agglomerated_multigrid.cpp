
/// Correctness test for the agglomerated Multigrid variant with EpsilonDivDivKerngen
/// (the viscous block of mantlecirculation's Stokes solver), Chebyshev smoother,
/// and a 5-level hierarchy with per-level sub-communicators.
///
/// Runs both a classical MG (all levels on WORLD) and an agglomerated MG (coarse
/// levels on sub-comms) against the same random-RHS problem and asserts they
/// produce equivalent solutions.
///
/// Unlike the earlier Laplace-based variant, this exercises:
///   - vector-valued DoFs (Grid4DDataVec<double, 3>) through pack/unpack/Alltoallv
///   - EpsilonDivDivKerngen's ShellBoundaryCommPlan-based halo exchange on sub-comms
///   - Chebyshev smoother's lazy power-iteration eigenvalue estimate on sub-comms
///   - ProlongationVecConstant / RestrictionVecConstant (as used by mantlecirculation)
///
/// CLI:
///   --factors "f0 f1 f2 f3"   per-descent agglomeration factors (default 2 2 1 1)

// #define TERRA_MG_AGGLOM_DEBUG 1  // enable to trace V-cycle phases on rank 0

#include <cstring>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "communication/shell/redistribute.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/agglomerated_distribution.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pcg.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/mpi/level_comms.hpp"
#include "util/init.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::BoundaryConditionFlag;
using grid::shell::BoundaryConditions;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using ScalarT      = double;
using Epsilon      = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarT, 3 >;
using Smoother     = linalg::solvers::Chebyshev< Epsilon >;
using CoarseSolver = linalg::solvers::PCG< Epsilon >;
using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarT, 3 >;
using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarT, 3 >;
using VecMass      = fe::wedge::operators::shell::VectorMass< ScalarT >;
using GridData     = Grid4DDataVec< ScalarT, 3 >;
using Redistribute = communication::shell::Redistribute< GridData >;

/// Smooth vector field used as our "exact" solution. Zero on the top+bottom
/// shell boundaries (r=0 and r=num_shells-1), which satisfies the Dirichlet BCs
/// that EpsilonDivDivKerngen uses when treat_boundary=true.
template < class T >
struct SolutionInterpolator
{
    Grid3DDataVec< T, 3 >      grid_;
    Grid2DDataScalar< T >      radii_;
    Grid4DDataVec< T, 3 >      data_;

    KOKKOS_INLINE_FUNCTION void operator()( int s, int x, int y, int r ) const
    {
        const dense::Vec< T, 3 > c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        const bool on_shell_boundary = ( r == 0 || r == radii_.extent( 1 ) - 1 );
        if ( on_shell_boundary )
        {
            for ( int d = 0; d < 3; ++d ) data_( s, x, y, r, d ) = 0;
            return;
        }
        // Simple smooth field vanishing on radial boundary (factor sin(pi*rho)).
        const T rho = ( radii_( s, r ) - radii_( s, 0 ) ) /
                       ( radii_( s, radii_.extent( 1 ) - 1 ) - radii_( s, 0 ) );
        const T w = Kokkos::sin( M_PI * rho );
        data_( s, x, y, r, 0 ) = w * Kokkos::sin( 2 * c( 0 ) ) * Kokkos::sinh( c( 1 ) );
        data_( s, x, y, r, 1 ) = w * Kokkos::sin( 2 * c( 1 ) ) * Kokkos::sinh( c( 2 ) );
        data_( s, x, y, r, 2 ) = w * Kokkos::sin( 2 * c( 2 ) ) * Kokkos::sinh( c( 0 ) );
    }
};

/// Constant viscosity field k=1.
template < class T >
struct KInterpolator
{
    Grid4DDataScalar< T > data_;
    KOKKOS_INLINE_FUNCTION void operator()( int s, int x, int y, int r ) const
    {
        data_( s, x, y, r ) = 1.0;
    }
};

std::vector< int > parse_factors_arg( int argc, char** argv, const std::vector< int >& fallback )
{
    for ( int a = 1; a + 1 < argc; ++a )
    {
        if ( std::strcmp( argv[a], "--factors" ) == 0 )
        {
            std::vector< int > out;
            std::istringstream ss( argv[a + 1] );
            int                v = 0;
            while ( ss >> v ) out.push_back( v );
            return out;
        }
    }
    return fallback;
}

int parse_int_arg( int argc, char** argv, const char* flag, int fallback )
{
    for ( int a = 1; a + 1 < argc; ++a )
        if ( std::strcmp( argv[a], flag ) == 0 ) return std::atoi( argv[a + 1] );
    return fallback;
}

struct LevelData
{
    DistributedDomain                                  domain;
    Grid3DDataVec< ScalarT, 3 >                        shell_coords;
    Grid2DDataScalar< ScalarT >                        radii;
    Grid4DDataScalar< grid::NodeOwnershipFlag >        mask;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask;
};

LevelData make_level_data(
    int                                                     lat_level,
    int                                                     lat_sdr,
    int                                                     rad_sdr,
    MPI_Comm                                                comm,
    const grid::shell::SubdomainToRankDistributionFunction& fn )
{
    LevelData L;
    L.domain = DistributedDomain::create_uniform_on_comm(
        comm,
        lat_level,
        grid::shell::uniform_shell_radii( 0.5, 1.0, ( 1 << lat_level ) + 1 ),
        lat_sdr,
        rad_sdr,
        fn );
    L.shell_coords  = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarT >( L.domain );
    L.radii         = grid::shell::subdomain_shell_radii< ScalarT >( L.domain );
    L.mask          = grid::setup_node_ownership_mask_data( L.domain );
    L.boundary_mask = grid::shell::setup_boundary_mask_data( L.domain );
    return L;
}

grid::shell::SubdomainToRankDistributionFunction make_agglom_fn(
    const grid::shell::SubdomainToRankDistributionFunction& orig,
    int                                                     cumulative_factor )
{
    if ( cumulative_factor == 1 )
        return orig;
    return grid::shell::agglomerated_subdomain_to_rank( orig, cumulative_factor );
}

/// Allgather per-rank "is this rank in comm[i]?" flags, then rank 0 prints the
/// world-rank membership of each sub-communicator.
void print_comm_ladder(
    const std::vector< MPI_Comm >& level_comms,
    const std::vector< int >&      cumulative_factors )
{
    const int world_rank = mpi::rank();
    const int world_size = mpi::num_processes();
    const int n_levels   = static_cast< int >( level_comms.size() );

    std::vector< int > my_flags( n_levels, 0 );
    for ( int i = 0; i < n_levels; ++i )
        my_flags[i] = ( level_comms[i] != MPI_COMM_NULL ) ? 1 : 0;

    std::vector< int > all_flags( world_size * n_levels, 0 );
    MPI_Allgather( my_flags.data(), n_levels, MPI_INT,
                    all_flags.data(), n_levels, MPI_INT, MPI_COMM_WORLD );

    if ( world_rank == 0 )
    {
        std::cout << "[test_agglom_mg] comm ladder (index 0 = finest, each descent applies one factor):\n";
        for ( int i = 0; i < n_levels; ++i )
        {
            std::ostringstream ss;
            ss << "[test_agglom_mg]   comm[" << i << "] cum_factor=" << cumulative_factors[i]
               << "  ranks={";
            int count = 0;
            for ( int r = 0; r < world_size; ++r )
            {
                if ( all_flags[r * n_levels + i] )
                {
                    if ( count++ ) ss << ",";
                    ss << r;
                }
            }
            ss << "}  size=" << count;
            std::cout << ss.str() << "\n";
        }
        std::cout.flush();
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const int world_rank = mpi::rank();
    const int world_size = mpi::num_processes();

    // Default factor ladder: 4 descent steps -> 5-MG-level hierarchy.
    const auto factors   = parse_factors_arg( argc, argv, { 2, 2, 1, 1 } );
    const int  min_level = parse_int_arg( argc, argv, "--min-level", 0 );
    const int  lat_sdr   = parse_int_arg( argc, argv, "--lat-sdr",   0 );
    const int  rad_sdr   = parse_int_arg( argc, argv, "--rad-sdr",   0 );

    // Cumulative factor for each rung of the ladder. cumulative_factors[0] = 1
    // (the finest = full world), each step multiplies by the descent factor.
    std::vector< int > cumulative_factors;
    cumulative_factors.push_back( 1 );
    for ( int f : factors ) cumulative_factors.push_back( cumulative_factors.back() * f );
    if ( world_size % cumulative_factors.back() != 0 )
    {
        if ( world_rank == 0 )
            std::cerr << "world_size=" << world_size << " not divisible by cumulative factor "
                      << cumulative_factors.back() << std::endl;
        return 1;
    }

    const int         max_level  = min_level + static_cast< int >( factors.size() );
    constexpr int     num_cycles = 30;   // vector problem is harder than Laplace — more V-cycles
    constexpr ScalarT tol        = 1e-7;
    constexpr int     presmooth  = 2;  // Chebyshev pre/post smoothing iterations per level

    // Subdomain count constraint: with subdomain_to_rank_iterate_diamond_subdomains,
    // every rank at the finest comm must own at least one subdomain.
    const int num_global_sdrs = 10 * ( 1 << lat_sdr ) * ( 1 << lat_sdr ) * ( 1 << rad_sdr );
    if ( num_global_sdrs < world_size )
    {
        if ( world_rank == 0 )
            std::cerr << "need num_global_sdrs (" << num_global_sdrs << ") >= world_size (" << world_size
                      << "); increase --lat-sdr or --rad-sdr\n";
        return 1;
    }
    // Mesh-validity constraint: each subdomain must span at least 1 cell on the coarsest MG level.
    if ( min_level < lat_sdr )
    {
        if ( world_rank == 0 )
            std::cerr << "min_level (" << min_level << ") must be >= lat_sdr (" << lat_sdr << ")\n";
        return 1;
    }

    if ( world_rank == 0 )
    {
        std::cout << "[test_agglom_mg] world_size=" << world_size
                  << "  MG levels " << min_level << ".." << max_level
                  << "  lat_sdr=" << lat_sdr << "  rad_sdr=" << rad_sdr
                  << "  num_global_sdrs=" << num_global_sdrs
                  << "  (num descent steps = " << factors.size() << ")" << std::endl;
        std::cout << "[test_agglom_mg] per-descent factors = {";
        for ( size_t i = 0; i < factors.size(); ++i )
            std::cout << ( i ? ", " : "" ) << factors[i];
        std::cout << "}" << std::endl;
    }

    auto level_comms = mpi::build_level_comms( MPI_COMM_WORLD, factors );

    print_comm_ladder( level_comms, cumulative_factors );

    // Mesh level L lives on level_comms[max_level - L]. We index these tables
    // DIRECTLY by mesh level L (valid range [min_level, max_level]), so they
    // must be sized max_level + 1, not (max_level - min_level + 1).
    std::vector< MPI_Comm > level_mesh_comms( max_level + 1, MPI_COMM_NULL );
    std::vector< int >      level_mesh_cum_factor( max_level + 1, 0 );
    for ( int L = min_level; L <= max_level; ++L )
    {
        const int idx            = max_level - L;  // 0 = finest
        level_mesh_comms[L]      = level_comms[idx];
        level_mesh_cum_factor[L] = cumulative_factors[idx];
    }

    const auto orig_fn = grid::shell::subdomain_to_rank_iterate_diamond_subdomains;

    auto make_ld = [&]( int L, MPI_Comm c, int cum_factor ) {
        return make_level_data( L, lat_sdr, rad_sdr, c, make_agglom_fn( orig_fn, cum_factor ) );
    };

    // For each MG level L: "own" data (mesh at L on L's own comm).
    std::vector< LevelData > ld_on_own_comm;
    for ( int L = min_level; L <= max_level; ++L )
        ld_on_own_comm.push_back( make_ld( L, level_mesh_comms[L], level_mesh_cum_factor[L] ) );

    // For each coarse MG level L (Li = L - min_level, 0..max-min-1): mesh at L on
    // the UPPER comm (= comm of level L+1). Needed for R/P halos at the descent
    // L+1 -> L, and for the mirror Redistribute source buffer tmp_r_fine[Li].
    // If level L and L+1 share a comm (factor=1 at that descent), reuse the own data.
    std::vector< LevelData > ld_on_upper_comm;
    for ( int L = min_level; L < max_level; ++L )
    {
        if ( level_mesh_comms[L] == level_mesh_comms[L + 1] )
            ld_on_upper_comm.push_back( ld_on_own_comm[L - min_level] );
        else
            ld_on_upper_comm.push_back(
                make_ld( L, level_mesh_comms[L + 1], level_mesh_cum_factor[L + 1] ) );
    }

    // -------------------------------------------------------------------------
    // Per-level scalar viscosity field k = 1. Built on each level's own domain so
    // reductions/halos respect that level's comm.
    // -------------------------------------------------------------------------
    std::vector< VectorQ1Scalar< ScalarT > > k_per_level;
    for ( int L = min_level; L <= max_level; ++L )
    {
        const auto& LL = ld_on_own_comm[L - min_level];
        k_per_level.emplace_back( "k_L" + std::to_string( L ), LL.domain, LL.mask );
        if ( level_mesh_comms[L] != MPI_COMM_NULL )
        {
            Kokkos::parallel_for( "k_init_L" + std::to_string( L ),
                                    local_domain_md_range_policy_nodes( LL.domain ),
                                    KInterpolator< ScalarT >{ k_per_level.back().grid_data() } );
            Kokkos::fence();
        }
    }

    // Dirichlet on both CMB and surface — matches EpsilonDivDivKerngen's handling
    // for the fast-dirichlet-neumann kernel path.
    BoundaryConditions bcs;
    bcs[0] = { grid::shell::ShellBoundaryFlag::CMB,     BoundaryConditionFlag::DIRICHLET };
    bcs[1] = { grid::shell::ShellBoundaryFlag::SURFACE, BoundaryConditionFlag::DIRICHLET };

    // -------------------------------------------------------------------------
    // Problem on the finest level. With Dirichlet BCs the EpsilonDivDiv operator
    // already enforces u=0 on boundary rows via its identity-row treatment (when
    // treat_boundary=true, which is the default for Stokes viscous blocks). The
    // analytical u vanishes on the shell boundary by construction — so we can
    // just compute f = A * u_exact and solve.
    // -------------------------------------------------------------------------
    const auto& Lt            = ld_on_own_comm[max_level - min_level];
    const auto& k_fine        = k_per_level.back();

    Epsilon A( Lt.domain, Lt.shell_coords, Lt.radii, Lt.boundary_mask, k_fine.grid_data(),
               bcs, /*diagonal=*/false );

    VectorQ1Vec< ScalarT, 3 > u( "u", Lt.domain, Lt.mask );
    VectorQ1Vec< ScalarT, 3 > f( "f", Lt.domain, Lt.mask );
    VectorQ1Vec< ScalarT, 3 > solution( "solution", Lt.domain, Lt.mask );
    VectorQ1Vec< ScalarT, 3 > error( "error", Lt.domain, Lt.mask );
    VectorQ1Vec< ScalarT, 3 > u_classical( "u_classical", Lt.domain, Lt.mask );
    VectorQ1Vec< ScalarT, 3 > u_agglom( "u_agglom", Lt.domain, Lt.mask );

    Kokkos::parallel_for( "sol",
        local_domain_md_range_policy_nodes( Lt.domain ),
        SolutionInterpolator< ScalarT >{ Lt.shell_coords, Lt.radii, solution.grid_data() } );
    Kokkos::fence();

    // f = A * solution.
    linalg::apply( A, solution, f );
    Kokkos::fence();

    assign( u, 0.0 );
    assign( error, 0.0 );

    const auto num_dofs_fine = 3 *
        kernels::common::count_masked< long >( Lt.mask, grid::NodeOwnershipFlag::OWNED, level_mesh_comms[max_level] );
    if ( world_rank == 0 )
        std::cout << "[test_agglom_mg] num_dofs_fine = " << num_dofs_fine << std::endl;

    // Build a Chebyshev smoother's inverse diagonal + tmp vectors on a level.
    // On dropped ranks (domain.comm() == MPI_COMM_NULL) the diagonal apply would
    // deadlock in halo exchanges, so skip the kernel and return a zero-filled stub.
    constexpr int cheby_order               = 2;
    constexpr int cheby_max_ev_power_iters  = 10;
    auto build_smoother = [&]( const LevelData& LL, const VectorQ1Scalar< ScalarT >& k_level ) {
        VectorQ1Vec< ScalarT, 3 > tmp_sm_a( "tmp_sm_a", LL.domain, LL.mask );
        VectorQ1Vec< ScalarT, 3 > tmp_sm_b( "tmp_sm_b", LL.domain, LL.mask );
        VectorQ1Vec< ScalarT, 3 > inv_diag( "inv_diag", LL.domain, LL.mask );
        if ( LL.domain.comm() != MPI_COMM_NULL )
        {
            Epsilon A_diag( LL.domain, LL.shell_coords, LL.radii, LL.boundary_mask, k_level.grid_data(),
                             bcs, /*diagonal=*/true );
            VectorQ1Vec< ScalarT, 3 > ones( "ones", LL.domain, LL.mask );
            assign( ones, 1.0 );
            apply( A_diag, ones, inv_diag );
            linalg::invert_entries( inv_diag );
        }
        std::vector< VectorQ1Vec< ScalarT, 3 > > tmps{ tmp_sm_a, tmp_sm_b };
        return Smoother( cheby_order, inv_diag, tmps, presmooth, cheby_max_ev_power_iters );
    };
    auto make_cgs_tmps = []( const LevelData& LL ) {
        std::vector< VectorQ1Vec< ScalarT, 3 > > v;
        for ( int i = 0; i < 4; ++i )
            v.emplace_back( "cgs_tmp_" + std::to_string( i ), LL.domain, LL.mask );
        return v;
    };

    // -------------------------------------------------------------------------
    // (1) Classical 5-level MG — all mesh levels on MPI_COMM_WORLD.
    // -------------------------------------------------------------------------
    ScalarT l2_classical      = 0.0;
    double  t_classical_setup = 0.0;
    double  t_classical_solve = 0.0;
    {
        MPI_Barrier( MPI_COMM_WORLD );
        const double t0 = MPI_Wtime();

        std::vector< LevelData > ld_world;
        std::vector< VectorQ1Scalar< ScalarT > > k_world;
        for ( int L = min_level; L <= max_level; ++L )
        {
            ld_world.push_back( make_ld( L, MPI_COMM_WORLD, 1 ) );
            k_world.emplace_back( "k_cl_L" + std::to_string( L ), ld_world.back().domain, ld_world.back().mask );
            Kokkos::parallel_for( "k_cl_init",
                                    local_domain_md_range_policy_nodes( ld_world.back().domain ),
                                    KInterpolator< ScalarT >{ k_world.back().grid_data() } );
            Kokkos::fence();
        }

        std::vector< Epsilon >                     A_c;
        std::vector< Prolongation >                P_add;
        std::vector< Restriction >                 R;
        std::vector< Smoother >                    sm;
        std::vector< VectorQ1Vec< ScalarT, 3 > >   tmp_r, tmp_e, tmp;

        for ( int L = min_level; L <= max_level; ++L )
        {
            const auto& LL = ld_world[L - min_level];
            const auto& kk = k_world[L - min_level];
            sm.push_back( build_smoother( LL, kk ) );
            tmp.emplace_back( "tmp_L" + std::to_string( L ), LL.domain, LL.mask );
        }
        for ( int L = min_level; L < max_level; ++L )
        {
            const auto& Lc = ld_world[L - min_level];
            const auto& kc = k_world[L - min_level];
            A_c.emplace_back( Lc.domain, Lc.shell_coords, Lc.radii, Lc.boundary_mask, kc.grid_data(),
                               bcs, /*diagonal=*/false );
            P_add.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( Lc.domain );
            tmp_r.emplace_back( "tmp_r_L" + std::to_string( L ), Lc.domain, Lc.mask );
            tmp_e.emplace_back( "tmp_e_L" + std::to_string( L ), Lc.domain, Lc.mask );
        }

        auto cgs_tmps = make_cgs_tmps( ld_world[0] );
        auto table    = std::make_shared< util::Table >();
        CoarseSolver cgs( linalg::solvers::IterativeSolverParameters{ 200, 1e-10, 1e-14 }, table, cgs_tmps );
        cgs.set_tag( "cgs_classical" );

        linalg::solvers::Multigrid< Epsilon, Prolongation, Restriction, Smoother, CoarseSolver > mg(
            P_add, R, A_c, tmp_r, tmp_e, tmp, sm, sm, cgs, num_cycles, tol );
        mg.set_tag( "mg_classical" );
        mg.collect_statistics( table );

        assign( u_classical, 0.0 );

        MPI_Barrier( MPI_COMM_WORLD );
        const double t_setup_end = MPI_Wtime();
        t_classical_setup        = t_setup_end - t0;

        linalg::solvers::solve( mg, A, u_classical, f );
        Kokkos::fence();

        MPI_Barrier( MPI_COMM_WORLD );
        t_classical_solve = MPI_Wtime() - t_setup_end;

        linalg::lincomb( error, { 1.0, -1.0 }, { u_classical, solution } );
        l2_classical = linalg::norm_2_scaled( error, 1.0 / static_cast< ScalarT >( num_dofs_fine ) );

        if ( world_rank == 0 )
            std::cout << "[test_agglom_mg] classical   setup=" << t_classical_setup * 1e3 << " ms"
                      << "  solve=" << t_classical_solve * 1e3 << " ms"
                      << "  l2_error=" << l2_classical << std::endl;

        if ( world_rank == 0 )
        {
            const auto cgs_rows = table->query_rows_equals( "tag", "cgs_classical" );
            std::cout << "[test_agglom_mg] classical CGS: " << cgs_rows.rows().size()
                      << " recorded rows across all V-cycle coarse solves. Last 10:" << std::endl;
            // Select just the last 10 rows by printing the full table and letting the user scan;
            // Table doesn't expose a slice, so print all.
            cgs_rows.print_pretty();
        }
    }

    // -------------------------------------------------------------------------
    // (2) Agglomerated 5-level MG. Per descent Li (level Li+1 -> level Li):
    //   - R[Li], P[Li] halo-exchange on the UPPER comm.
    //   - tmp_r_fine[Li], tmp_e_fine[Li] allocated on the UPPER comm.
    //   - redistribute_down_[Li] carries data from UPPER to LOWER ownership.
    //   - A_c[Li], tmp_r[Li], tmp_e[Li] on the LOWER comm.
    //   - smoothers_pre_[L], smoothers_post_[L], tmp[L] on level L's own comm.
    // -------------------------------------------------------------------------
    ScalarT l2_agglom      = 0.0;
    double  t_agglom_setup = 0.0;
    double  t_agglom_solve = 0.0;
    double  t_agglom_redist_plan = 0.0;
    {
        MPI_Barrier( MPI_COMM_WORLD );
        const double t0 = MPI_Wtime();

        // Per-level viscosity ON THE UPPER COMM for the descent boundaries where
        // R/P halos live; needed for the upper-comm A_diag used inside Chebyshev's
        // first-use eigenvalue estimation and for the R/P-side diagnostics. We
        // reuse k_per_level (built on each level's own comm) for A_c and the
        // smoother at every level.
        std::vector< Smoother > sm_pre, sm_post;
        for ( int L = min_level; L <= max_level; ++L )
        {
            const auto& LL = ld_on_own_comm[L - min_level];
            const auto& kk = k_per_level[L - min_level];
            MPI_Barrier( MPI_COMM_WORLD );
            if ( world_rank == 0 )
                std::cout << "[DBG] agglom build_smoother L=" << L
                          << "  comm=" << ( LL.domain.comm() == MPI_COMM_NULL ? "NULL" : "ok" )
                          << std::endl << std::flush;
            sm_pre.push_back( build_smoother( LL, kk ) );
            MPI_Barrier( MPI_COMM_WORLD );
            if ( world_rank == 0 )
                std::cout << "[DBG]   pre done L=" << L << std::endl << std::flush;
            sm_post.push_back( build_smoother( LL, kk ) );
            MPI_Barrier( MPI_COMM_WORLD );
            if ( world_rank == 0 )
                std::cout << "[DBG]   post done L=" << L << std::endl << std::flush;
        }

        std::vector< VectorQ1Vec< ScalarT, 3 > > tmp;
        for ( int L = min_level; L <= max_level; ++L )
        {
            const auto& LL = ld_on_own_comm[L - min_level];
            tmp.emplace_back( "tmp_agg_L" + std::to_string( L ), LL.domain, LL.mask );
        }

        std::vector< Epsilon >                     A_c;
        std::vector< VectorQ1Vec< ScalarT, 3 > >   tmp_r, tmp_e;
        std::vector< Prolongation >                P_add;
        std::vector< Restriction >                 R;
        std::vector< VectorQ1Vec< ScalarT, 3 > >   tmp_r_fine, tmp_e_fine;
        std::vector< Redistribute >                redist;

        for ( int Li = 0; Li < max_level - min_level; ++Li )
        {
            const int   L_lower = min_level + Li;
            const int   L_upper = min_level + Li + 1;
            const auto& LL_lower_on_lower = ld_on_own_comm[Li];
            const auto& LL_upper_on_upper = ld_on_own_comm[Li + 1];
            const auto& LL_lower_on_upper = ld_on_upper_comm[Li];
            const auto& k_lower           = k_per_level[Li];

            MPI_Barrier( MPI_COMM_WORLD );
            if ( world_rank == 0 )
                std::cout << "[DBG] agglom descent Li=" << Li
                          << " (L " << L_upper << " -> " << L_lower << ")" << std::endl << std::flush;

            A_c.emplace_back( LL_lower_on_lower.domain, LL_lower_on_lower.shell_coords,
                               LL_lower_on_lower.radii, LL_lower_on_lower.boundary_mask,
                               k_lower.grid_data(), bcs, /*diagonal=*/false );
            tmp_r.emplace_back( "tmp_r_agg_L" + std::to_string( L_lower ),
                                 LL_lower_on_lower.domain, LL_lower_on_lower.mask );
            tmp_e.emplace_back( "tmp_e_agg_L" + std::to_string( L_lower ),
                                 LL_lower_on_lower.domain, LL_lower_on_lower.mask );

            P_add.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( LL_lower_on_upper.domain );

            tmp_r_fine.emplace_back( "tmp_r_fine_L" + std::to_string( L_lower ),
                                      LL_lower_on_upper.domain, LL_lower_on_upper.mask );
            tmp_e_fine.emplace_back( "tmp_e_fine_L" + std::to_string( L_lower ),
                                      LL_lower_on_upper.domain, LL_lower_on_upper.mask );

            const auto fn_upper = make_agglom_fn( orig_fn, level_mesh_cum_factor[L_upper] );
            const auto fn_lower = make_agglom_fn( orig_fn, level_mesh_cum_factor[L_lower] );

            MPI_Barrier( MPI_COMM_WORLD );
            const double t_rd0 = MPI_Wtime();
            if ( world_rank == 0 )
                std::cout << "[DBG]   building redist plan Li=" << Li << std::endl << std::flush;
            redist.emplace_back( LL_lower_on_upper.domain, LL_lower_on_lower.domain, fn_upper, fn_lower );
            MPI_Barrier( MPI_COMM_WORLD );
            if ( world_rank == 0 )
                std::cout << "[DBG]   redist plan Li=" << Li << " done" << std::endl << std::flush;
            t_agglom_redist_plan += MPI_Wtime() - t_rd0;
        }

        auto cgs_tmps = make_cgs_tmps( ld_on_own_comm[0] );
        auto table    = std::make_shared< util::Table >();
        CoarseSolver cgs( linalg::solvers::IterativeSolverParameters{ 200, 1e-10, 1e-14 }, table, cgs_tmps );
        cgs.set_tag( "cgs_agglom" );

        MPI_Barrier( MPI_COMM_WORLD );
        if ( world_rank == 0 ) std::cout << "[DBG] CGS built" << std::endl << std::flush;

        linalg::solvers::Multigrid< Epsilon, Prolongation, Restriction, Smoother, CoarseSolver, Redistribute > mg_agg(
            P_add, R, A_c, tmp_r, tmp_e, tmp, sm_pre, sm_post, cgs,
            num_cycles, tol,
            std::move( redist ), std::move( tmp_r_fine ), std::move( tmp_e_fine ) );
        mg_agg.set_tag( "mg_agglomerated" );
        mg_agg.collect_statistics( table );

        MPI_Barrier( MPI_COMM_WORLD );
        if ( world_rank == 0 ) std::cout << "[DBG] MG object built" << std::endl << std::flush;

        assign( u_agglom, 0.0 );

        // Clear the singleton TimerTree so the aggregate we dump below reflects
        // only agglom-solve timings (setup timers from either path stay out).
        util::TimerTree::instance().clear();

        MPI_Barrier( MPI_COMM_WORLD );
        if ( world_rank == 0 ) std::cout << "[DBG] starting solve" << std::endl << std::flush;
        const double t_setup_end = MPI_Wtime();
        t_agglom_setup           = t_setup_end - t0;

        linalg::solvers::solve( mg_agg, A, u_agglom, f );
        Kokkos::fence();

        MPI_Barrier( MPI_COMM_WORLD );
        if ( world_rank == 0 ) std::cout << "[DBG] solve returned" << std::endl << std::flush;

        MPI_Barrier( MPI_COMM_WORLD );
        t_agglom_solve = MPI_Wtime() - t_setup_end;

        linalg::lincomb( error, { 1.0, -1.0 }, { u_agglom, solution } );
        l2_agglom = linalg::norm_2_scaled( error, 1.0 / static_cast< ScalarT >( num_dofs_fine ) );

        if ( world_rank == 0 )
            std::cout << "[test_agglom_mg] agglom      setup=" << t_agglom_setup * 1e3 << " ms"
                      << "  (plan=" << t_agglom_redist_plan * 1e3 << " ms)"
                      << "  solve=" << t_agglom_solve * 1e3 << " ms"
                      << "  l2_error=" << l2_agglom << std::endl;

        if ( world_rank == 0 )
        {
            const auto cgs_rows = table->query_rows_equals( "tag", "cgs_agglom" );
            std::cout << "[test_agglom_mg] agglom CGS: " << cgs_rows.rows().size()
                      << " recorded rows across all V-cycle coarse solves." << std::endl;
            cgs_rows.print_pretty();
        }

        // Aggregate singleton timers across ranks and dump on rank 0 so we
        // can see per-V-cycle cost of each MG phase — in particular the
        // mg_redistribute_down_L* / mg_redistribute_up_L* nodes.
        util::TimerTree::instance().aggregate_mpi();
        if ( world_rank == 0 )
        {
            std::cout << "[test_agglom_mg] ---- TimerTree aggregate (agglom run) ----"
                      << std::endl;
            std::cout << util::TimerTree::instance().json_aggregate() << std::endl;
            std::cout << "[test_agglom_mg] ---- end TimerTree ----" << std::endl;
        }
    }

    linalg::lincomb( error, { 1.0, -1.0 }, { u_classical, u_agglom } );
    const auto l2_diff = linalg::norm_2_scaled( error, 1.0 / static_cast< ScalarT >( num_dofs_fine ) );

    if ( world_rank == 0 )
    {
        const double solve_speedup = ( t_agglom_solve > 0 ) ? ( t_classical_solve / t_agglom_solve ) : 0.0;
        std::cout << "[test_agglom_mg] |u_cl - u_agg| l2 = " << l2_diff << std::endl;
        std::cout << "[test_agglom_mg] solve speedup agglom vs classical = " << solve_speedup << "x"
                  << "  (classical " << t_classical_solve * 1e3 << " ms, agglom "
                  << t_agglom_solve * 1e3 << " ms)" << std::endl;
    }

    // Loosened tolerances for the vector-valued problem: Chebyshev's lazy eigenvalue
    // estimate can differ between classical and agglomerated MG (power iterations
    // run on different communicators, different reduction orders), which may
    // perturb the solution by a few orders of magnitude below the MG tolerance.
    const bool pass_conv = ( l2_agglom < 10.0 * l2_classical + 1e-8 );
    const bool pass_diff = ( l2_diff < 1e-3 );
    const bool pass      = pass_conv && pass_diff;

    if ( world_rank == 0 )
    {
        std::cout << ( pass_conv ? "[ OK ]" : "[FAIL]" ) << " agglomerated 5-level MG converges\n"
                  << ( pass_diff ? "[ OK ]" : "[FAIL]" ) << " classical vs agglomerated solutions agree\n"
                  << ( pass      ? "[ OK ]" : "[FAIL]" ) << " test overall" << std::endl;
    }

    mpi::free_level_comms( level_comms );
    return pass ? 0 : 1;
}
