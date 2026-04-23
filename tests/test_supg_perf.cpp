/// @file  test_supg_perf.cpp
/// @brief Wall-clock comparison of `UnsteadyAdvectionDiffusionSUPG` (legacy)
///        vs `UnsteadyAdvectionDiffusionSUPGKerngen` (fused team-based).

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg_kerngen.hpp"
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
using grid::shell::DistributedDomain;
using linalg::OperatorApplyMode;
using linalg::OperatorCommunicationMode;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

struct ScalarTempInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    ScalarTempInterpolator( const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r,
                            const Grid4DDataScalar< double >& d ) : grid_( g ), radii_( r ), data_( d ) {}
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r ) = 1.0 + 0.5 * Kokkos::sin( 1.3 * c( 0 ) );
    }
};

struct VelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    VelocityInterpolator( const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r,
                          const Grid4DDataVec< double, 3 >& d ) : grid_( g ), radii_( r ), data_( d ) {}
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r, 0 ) = 0.4 * Kokkos::sin( 1.7 * c( 2 ) );
        data_( s, x, y, r, 1 ) = 0.3 * Kokkos::cos( 1.1 * c( 0 ) );
        data_( s, x, y, r, 2 ) = 0.5 * Kokkos::sin( 0.9 * c( 1 ) );
    }
};

template < typename Op, typename Src, typename Dst >
static double timed_applies( Op& op, const Src& src, Dst& dst, int repeats )
{
    Kokkos::fence();
    Kokkos::Timer t;
    for ( int i = 0; i < repeats; ++i )
        linalg::apply( op, src, dst );
    Kokkos::fence();
    return t.seconds();
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    using ScalarType = double;
    using OpOld      = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;
    using OpNew      = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;

    int    level   = 5;
    int    lat_sdr = 0;
    int    rad_sdr = 0;
    int    warmup  = 5;
    int    repeats = 50;
    double dt          = 0.01;
    double kappa       = 1.0;
    double mass_scale  = 1.0;
    bool   treat_bnd   = true;
    bool   diag        = false;
    bool   lump        = false;

    for ( int i = 1; i < argc; ++i )
    {
        std::string a( argv[i] );
        if      ( a == "--level"   && i + 1 < argc ) level   = std::atoi( argv[++i] );
        else if ( a == "--lat-sdr" && i + 1 < argc ) lat_sdr = std::atoi( argv[++i] );
        else if ( a == "--rad-sdr" && i + 1 < argc ) rad_sdr = std::atoi( argv[++i] );
        else if ( a == "--warmup"  && i + 1 < argc ) warmup  = std::atoi( argv[++i] );
        else if ( a == "--repeats" && i + 1 < argc ) repeats = std::atoi( argv[++i] );
    }

    const double r_min = 0.5, r_max = 1.0;
    auto domain = DistributedDomain::create_uniform( level, level, r_min, r_max, lat_sdr, rad_sdr );

    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto radii  = grid::shell::subdomain_shell_radii< ScalarType >( domain );
    auto mask          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask = grid::shell::setup_boundary_mask_data( domain );

    VectorQ1Scalar< ScalarType > src    ( "src",     domain, mask );
    VectorQ1Vec< ScalarType, 3 > vel    ( "vel",     domain, mask );
    VectorQ1Scalar< ScalarType > dst_old( "dst_old", domain, mask );
    VectorQ1Scalar< ScalarType > dst_slow( "dst_slow", domain, mask );
    VectorQ1Scalar< ScalarType > dst_fast( "dst_fast", domain, mask );

    Kokkos::parallel_for( "interp_T",
        local_domain_md_range_policy_nodes( domain ),
        ScalarTempInterpolator( coords, radii, src.grid_data() ) );
    Kokkos::parallel_for( "interp_vel",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator( coords, radii, vel.grid_data() ) );
    Kokkos::fence();

    std::cout << "SUPG perf  —  level=" << level << ", lat_sdr=" << lat_sdr << ", rad_sdr=" << rad_sdr
              << ", warmup=" << warmup << ", repeats=" << repeats << std::endl;
    std::cout << "  subdomains/rank=" << domain.subdomains().size() << "  (" << (1 << level) << "^3 hex/subdomain)" << std::endl;

    OpOld op_old( domain, coords, radii, boundary_mask, vel, kappa, dt,
                  treat_bnd, diag, mass_scale, lump,
                  OperatorApplyMode::Replace, OperatorCommunicationMode::CommunicateAdditively );
    OpNew op_slow( domain, coords, radii, boundary_mask, vel, kappa, dt,
                   treat_bnd, diag, mass_scale, lump,
                   OperatorApplyMode::Replace, OperatorCommunicationMode::CommunicateAdditively );
    op_slow.force_slow_path();
    OpNew op_fast( domain, coords, radii, boundary_mask, vel, kappa, dt,
                   treat_bnd, diag, mass_scale, lump,
                   OperatorApplyMode::Replace, OperatorCommunicationMode::CommunicateAdditively );

    (void) timed_applies( op_old,  src, dst_old,  warmup );
    (void) timed_applies( op_slow, src, dst_slow, warmup );
    (void) timed_applies( op_fast, src, dst_fast, warmup );

    const double t_old  = timed_applies( op_old,  src, dst_old,  repeats );
    const double t_slow = timed_applies( op_slow, src, dst_slow, repeats );
    const double t_fast = timed_applies( op_fast, src, dst_fast, repeats );

    const double us_old  = 1e6 * t_old  / repeats;
    const double us_slow = 1e6 * t_slow / repeats;
    const double us_fast = 1e6 * t_fast / repeats;

    std::cout << std::fixed << std::setprecision( 2 );
    std::cout << "  old SUPG         : " << us_old  << " us/apply"
              << "   (" << t_old  << " s total)" << std::endl;
    std::cout << "  new SUPG (slow)  : " << us_slow << " us/apply"
              << "   (" << t_slow << " s total)"
              << "   speedup vs old = " << ( us_old / us_slow ) << "x" << std::endl;
    std::cout << "  new SUPG (fast)  : " << us_fast << " us/apply"
              << "   (" << t_fast << " s total)"
              << "   speedup vs old = " << ( us_old / us_fast ) << "x" << std::endl;

    return 0;
}
