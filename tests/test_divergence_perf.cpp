/// @file  test_divergence_perf.cpp
/// @brief Wall-clock comparison of `Divergence` (old) vs `DivergenceKerngen`
///        (new, team-based, fast path) at a realistic problem size.
///
/// For each BC set, we run (warmup + N timed) applies of
///    - old `Divergence`
///    - new `DivergenceKerngen` pinned to its slow path
///    - new `DivergenceKerngen` on the default (fast) path
/// and print per-apply times plus speedups.

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/operators/shell/divergence.hpp"
#include "fe/wedge/operators/shell/divergence_kerngen.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::BoundaryConditionFlag;
using grid::shell::BoundaryConditions;
using grid::shell::DistributedDomain;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::OperatorApplyMode;
using linalg::OperatorCommunicationMode;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

struct VectorFieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    VectorFieldInterpolator(
        const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r, const Grid4DDataVec< double, 3 >& d )
    : grid_( g ), radii_( r ), data_( d )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r, 0 ) = Kokkos::sin( 1.7 * c( 0 ) ) * Kokkos::cos( 0.9 * c( 1 ) );
        data_( s, x, y, r, 1 ) = Kokkos::cos( 2.3 * c( 1 ) ) * Kokkos::sin( 0.7 * c( 2 ) );
        data_( s, x, y, r, 2 ) = Kokkos::sin( 1.1 * c( 2 ) ) * Kokkos::cos( 1.3 * c( 0 ) );
    }
};

static std::string bc_name( BoundaryConditionFlag f )
{
    if ( f == DIRICHLET ) return "DIR";
    if ( f == FREESLIP )  return "FS";
    if ( f == NEUMANN )   return "NEU";
    return "?";
}

// Time `repeats` back-to-back applies. Returns total wall seconds (Kokkos::fence before & after).
template < typename Op, typename Src, typename Dst >
static double timed_applies( Op& op, const Src& src, Dst& dst, int repeats )
{
    Kokkos::fence();
    Kokkos::Timer t;
    for ( int i = 0; i < repeats; ++i )
    {
        linalg::apply( op, src, dst );
    }
    Kokkos::fence();
    return t.seconds();
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    using ScalarType = double;
    using OpOld      = fe::wedge::operators::shell::Divergence< ScalarType >;
    using OpNew      = fe::wedge::operators::shell::DivergenceKerngen< ScalarType >;

    // --- Knobs (override via CLI) ---
    int level_fine = 5;       // coarse = 4 => ~17x17x17 per subdomain
    int lat_sdr    = 0;
    int rad_sdr    = 0;
    int warmup     = 5;
    int repeats    = 50;

    for ( int i = 1; i < argc; ++i )
    {
        std::string a( argv[i] );
        if ( a == "--level"   && i + 1 < argc ) level_fine = std::atoi( argv[++i] );
        else if ( a == "--lat-sdr" && i + 1 < argc ) lat_sdr = std::atoi( argv[++i] );
        else if ( a == "--rad-sdr" && i + 1 < argc ) rad_sdr = std::atoi( argv[++i] );
        else if ( a == "--warmup"  && i + 1 < argc ) warmup  = std::atoi( argv[++i] );
        else if ( a == "--repeats" && i + 1 < argc ) repeats = std::atoi( argv[++i] );
    }

    const int level_coarse = level_fine - 1;
    const double r_min = 0.5, r_max = 1.0;

    auto domain_fine   = DistributedDomain::create_uniform( level_fine,   level_fine,   r_min, r_max, lat_sdr, rad_sdr );
    auto domain_coarse = DistributedDomain::create_uniform( level_coarse, level_coarse, r_min, r_max, lat_sdr, rad_sdr );

    const auto coords_fine = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_fine );
    const auto radii_fine  = grid::shell::subdomain_shell_radii< ScalarType >( domain_fine );

    auto mask_fine          = grid::setup_node_ownership_mask_data( domain_fine );
    auto mask_coarse        = grid::setup_node_ownership_mask_data( domain_coarse );
    auto boundary_mask_fine = grid::shell::setup_boundary_mask_data( domain_fine );

    VectorQ1Vec< ScalarType, 3 > src     ( "src",      domain_fine,   mask_fine   );
    VectorQ1Scalar< ScalarType > dst_old ( "dst_old",  domain_coarse, mask_coarse );
    VectorQ1Scalar< ScalarType > dst_slow( "dst_slow", domain_coarse, mask_coarse );
    VectorQ1Scalar< ScalarType > dst_fast( "dst_fast", domain_coarse, mask_coarse );

    Kokkos::parallel_for(
        "interp_src",
        local_domain_md_range_policy_nodes( domain_fine ),
        VectorFieldInterpolator( coords_fine, radii_fine, src.grid_data() ) );
    Kokkos::fence();

    std::cout << "Divergence perf  —  level_fine=" << level_fine
              << ", lat_sdr=" << lat_sdr << ", rad_sdr=" << rad_sdr
              << ", warmup=" << warmup << ", repeats=" << repeats << std::endl;
    std::cout << "  fine   subdomains per rank: " << domain_fine.subdomains().size()
              << "  (" << ( 1 << level_fine ) << "^3 hex cells each)" << std::endl;

    std::vector< std::pair< BoundaryConditionFlag, BoundaryConditionFlag > > bc_sets = {
        { DIRICHLET, DIRICHLET },
        { FREESLIP,  DIRICHLET },
        { FREESLIP,  FREESLIP  }
    };

    for ( auto bc : bc_sets )
    {
        BoundaryConditions bcs = { { CMB, bc.first }, { SURFACE, bc.second } };

        OpOld op_old(
            domain_fine, domain_coarse, coords_fine, radii_fine, boundary_mask_fine, bcs,
            OperatorApplyMode::Replace, OperatorCommunicationMode::CommunicateAdditively );
        OpNew op_slow(
            domain_fine, domain_coarse, coords_fine, radii_fine, boundary_mask_fine, bcs,
            OperatorApplyMode::Replace, OperatorCommunicationMode::CommunicateAdditively );
        op_slow.force_slow_path();
        OpNew op_fast(
            domain_fine, domain_coarse, coords_fine, radii_fine, boundary_mask_fine, bcs,
            OperatorApplyMode::Replace, OperatorCommunicationMode::CommunicateAdditively );

        // Warmup each path independently.
        (void) timed_applies( op_old,  src, dst_old,  warmup );
        (void) timed_applies( op_slow, src, dst_slow, warmup );
        (void) timed_applies( op_fast, src, dst_fast, warmup );

        const double t_old  = timed_applies( op_old,  src, dst_old,  repeats );
        const double t_slow = timed_applies( op_slow, src, dst_slow, repeats );
        const double t_fast = timed_applies( op_fast, src, dst_fast, repeats );

        const double us_old  = 1e6 * t_old  / repeats;
        const double us_slow = 1e6 * t_slow / repeats;
        const double us_fast = 1e6 * t_fast / repeats;

        std::cout << "\n[ BCs: CMB=" << bc_name( bc.first ) << ", SURFACE=" << bc_name( bc.second )
                  << "   (new op path: " << op_fast.path_name() << ") ]" << std::endl;
        std::cout << std::fixed << std::setprecision( 2 );
        std::cout << "  old Divergence           : " << us_old  << " us/apply"
                  << "   (" << t_old  << " s total)" << std::endl;
        std::cout << "  new Divergence (slow)    : " << us_slow << " us/apply"
                  << "   (" << t_slow << " s total)"
                  << "   speedup vs old = " << ( us_old / us_slow ) << "x" << std::endl;
        std::cout << "  new Divergence (fast)    : " << us_fast << " us/apply"
                  << "   (" << t_fast << " s total)"
                  << "   speedup vs old = " << ( us_old / us_fast ) << "x" << std::endl;
    }

    return 0;
}
