// Test: point location on random shell points, comparing the walking and O(1) kernels on host and GPU.
//
// The structured checks in test_mmoc_point_location map a fixed set of reference coordinates forward from a
// known cell, so every sample sits at the same relative position inside its wedge and is known in advance to
// belong to the subdomain being queried. Both crutches are removed here:
//
//   1. Random sampling: points are drawn uniformly by volume from the shell, so they meet the cell structure
//      at arbitrary offsets and the cone tests at arbitrary angles rather than at a handful of repeated
//      configurations.
//   2. Subdomain search: the owning subdomain is not assumed. Each point is offered to every subdomain and
//      the answer is the *set* that claims it -- which is how a departure point actually arrives in the
//      transport scheme, where nothing says in advance which diamond it fell into. Every point must be
//      claimed by at least one subdomain, since the diamonds tile the sphere.
//   3. Walking versus O(1): locate_point() and locate_point_direct() must agree on the claim set and on the
//      wedge within it.
//   4. Host versus GPU: each of the two location routines runs in its own Kokkos parallel_for over
//      (point, subdomain) -- one thread per departure point, neighbouring threads walking different numbers
//      of steps and taking different branches, which is the divergence a host-only test cannot exercise.
//      The discrete results must match the host exactly; the reference coordinates only to a tolerance,
//      since the device contracts multiply-adds into FMAs and the host does not.
//
// Points near a radial boundary can be legitimately clamped: the wedge's radial coordinate is the cone radius
// out to the chord plane of the spherical triangle, which sags inside the sphere, so a point close to the
// surface can have a cone radius beyond r_max. Clamping is expected there and asserted to happen nowhere else.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <mpi.h>

#include "fe/wedge/sl/point_location.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;
using ScalarType = double;

using fe::wedge::sl::IndexBounds;
using fe::wedge::sl::WedgeCell;

namespace
{

ScalarType linear_field( const dense::Vec< ScalarType, 3 >& x )
{
    return 1.0 + 2.0 * x( 0 ) + 3.0 * x( 1 ) - 0.5 * x( 2 );
}

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        if ( g_failures < 20 )
            std::cout << "  FAIL: " << what << std::endl;
    }
}

} // namespace

/// Runs the checks described at the top of the file for one refinement level and sample size.
///
/// The sample is generated once on the host and copied to the device, so the same points run through all four
/// combinations of { walking, O(1) } x { host, device } and any difference in the results is a difference in
/// the location code rather than in the sampling. Each routine gets its own parallel_for over
/// (point, subdomain), which is how the transport scheme calls this code -- one thread per departure point,
/// every thread walking a different number of steps. Keeping the two in separate kernels stops the walk's
/// divergence from setting the occupancy for both.
///
/// Self-consistency is what is asserted: all four combinations must agree on the claim set and on the wedge
/// within it, every point must be claimed by at least one subdomain (the diamonds tile the sphere, so a point
/// nobody claims is a hole in the location code), and the cell that comes back must map forward onto the point
/// it was found from. A point on a shared diamond edge is legitimately claimed by more than one subdomain, so
/// a claim count above one is counted and reported rather than treated as an error.
void test_random_shell_points( const int level, const int num_points )
{
    const ScalarType r_min = 0.5;
    const ScalarType r_max = 1.0;

    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond(
        level, level, r_min, r_max );

    const auto coords_shell_d = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii_d = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    auto coords_shell = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_shell_d );
    auto coords_radii = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_radii_d );

    const int num_subdomains = static_cast< int >( domain.subdomains().size() );
    const int num_nodes_lat  = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const int num_nodes_rad  = domain.domain_info().subdomain_num_nodes_radially();

    const IndexBounds bounds{ num_nodes_lat, num_nodes_lat, num_nodes_rad };

    Kokkos::View< uint8_t*** > all_valid( "all_valid", num_subdomains, num_nodes_lat, num_nodes_lat );
    Kokkos::deep_copy( all_valid, static_cast< uint8_t >( 1 ) );
    auto all_valid_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, all_valid );

    const auto box = fe::wedge::sl::corner_box_from_bounds( bounds );

    // The same fixed budgets the rest of the direct-location tests use.
    const int        direct_max_refinements = 2;
    const int        direct_max_walk_steps  = 4;
    const ScalarType eps                    = 1e-12;
    const int        max_steps              = 8 * num_nodes_lat; // reference walk, may cross the whole diamond

    const ScalarType tol = 1e-11;

    // The wedge's radial coordinate is the *cone* radius rho, measured along the ray out to the plane of the
    // spherical triangle rather than to the sphere itself, and that plane sags inside the sphere. For a
    // triangle whose vertices span an angle Theta we have rho <= |X| / cos( Theta ), so a point closer to a
    // shell boundary than a factor ( 1 - cos Theta ) can have rho outside [ r_min, r_max ] and be clamped even
    // though |X| is comfortably inside the shell. That is the discretisation, not a location failure -- but it
    // may only ever happen in that thin band, so the band is what gets asserted. Theta is over-estimated as two
    // cells' worth of the 90 degrees a diamond spans.
    const ScalarType theta_cell = 2.0 * ( M_PI / 2.0 ) / static_cast< ScalarType >( num_nodes_lat - 1 );
    const ScalarType sag_band   = 1.0 - std::cos( theta_cell );

    // How far apart the host and device reference coordinates may drift.
    //
    // They do not match bit for bit, and are not expected to: nvcc contracts multiply-adds into FMAs inside
    // the 3x3 solve of wedge_lateral_cone_coords where the host compiler does not, so the two evaluate
    // different but equally valid operation sequences. The gap is then amplified by the conditioning of that
    // solve -- the three unit-sphere vertices of a wedge grow more nearly coplanar as the mesh refines, so the
    // system stiffens with the level. Measured maxima are 1.0e-13 at level 3 and 5.8e-12 at level 5, growing
    // roughly with the square of the cells per diamond side, which is what a 1 / theta^2 conditioning argument
    // predicts.
    //
    // The bound below follows that square with a wide margin. It is a guard against something structurally
    // breaking on the device, not a sharp estimate: it is calibrated on two levels only, and a drift of even
    // 1e-11 in a barycentric weight is far below anything that could move an interpolated value. The discrete
    // outputs, which are what the transport scheme actually branches on, are held to exact equality instead.
    const ScalarType cells_per_side = static_cast< ScalarType >( num_nodes_lat - 1 );
    const ScalarType device_ref_tol = 4e-14 * cells_per_side * cells_per_side;

    // ---- the sample ----------------------------------------------------------------------------------------
    // Generated once on the host and copied to the device, so host and device see bit-identical inputs and any
    // difference in the results is a difference in the location code, not in the sampling.
    Kokkos::View< ScalarType** > points_d( "random_points", num_points, 3 );
    auto                         points = Kokkos::create_mirror_view( points_d );

    // Fixed seed: a failing sample has to be reproducible from the point index printed with the failure.
    std::mt19937_64                              rng( 0x9E3779B97F4A7C15ull + static_cast< uint64_t >( level ) );
    std::uniform_real_distribution< ScalarType > uniform( 0.0, 1.0 );

    for ( int p = 0; p < num_points; ++p )
    {
        // Uniform on the sphere: z uniform in [-1, 1] together with a uniform azimuth is area-preserving.
        const ScalarType z   = 2.0 * uniform( rng ) - 1.0;
        const ScalarType phi = 2.0 * M_PI * uniform( rng );
        const ScalarType s   = std::sqrt( std::max( 0.0, 1.0 - z * z ) );

        // Uniform by volume rather than in r, so the sampling is not biased towards the CMB.
        const ScalarType r3_lo = r_min * r_min * r_min;
        const ScalarType r3_hi = r_max * r_max * r_max;
        const ScalarType r     = std::cbrt( r3_lo + uniform( rng ) * ( r3_hi - r3_lo ) );

        points( p, 0 ) = r * s * std::cos( phi );
        points( p, 1 ) = r * s * std::sin( phi );
        points( p, 2 ) = r * z;
    }
    Kokkos::deep_copy( points_d, points );

    // ---- result storage ------------------------------------------------------------------------------------
    // One row per ( point, subdomain, method ), method 0 = walking, 1 = O(1). Storing the raw results instead
    // of reducing counters inside the kernel keeps the kernel trivial and lets every assertion -- including
    // host-versus-device equality -- run on the host where a failure can name the point that caused it.
    enum
    {
        METHOD_WALK   = 0,
        METHOD_DIRECT = 1,
        NUM_METHODS   = 2
    };
    // Slots: found, cell.x, cell.y, cell.r, cell.w, clamped_radially, escaped_laterally.
    enum
    {
        SLOT_FOUND   = 0,
        SLOT_X       = 1,
        SLOT_Y       = 2,
        SLOT_R       = 3,
        SLOT_W       = 4,
        SLOT_CLAMPED = 5,
        SLOT_ESCAPED = 6,
        NUM_SLOTS    = 7
    };

    Kokkos::View< int**** > flags_d( "loc_flags", num_points, num_subdomains, NUM_METHODS, NUM_SLOTS );
    Kokkos::View< ScalarType**** > ref_d( "loc_ref", num_points, num_subdomains, NUM_METHODS, 3 );

    auto flags_host = Kokkos::create_mirror_view( flags_d ); // filled by the host pass
    auto ref_host   = Kokkos::create_mirror_view( ref_d );

    // ---- host pass -----------------------------------------------------------------------------------------
    for ( int p = 0; p < num_points; ++p )
    {
        dense::Vec< ScalarType, 3 > X;
        for ( int d = 0; d < 3; ++d )
            X( d ) = points( p, d );

        for ( int sd = 0; sd < num_subdomains; ++sd )
        {
            const auto walked = fe::wedge::sl::locate_point(
                X, sd, WedgeCell{ 0, 0, 0, 0 }, coords_shell, coords_radii, bounds, max_steps, eps,
                /*clamp_radially=*/true, r_min, r_max, all_valid_h );

            const auto direct = fe::wedge::sl::locate_point_direct(
                X, sd, coords_shell, coords_radii, box, bounds, direct_max_refinements, direct_max_walk_steps,
                eps, /*clamp_radially=*/true, r_min, r_max, all_valid_h );

            const fe::wedge::sl::LocateResult< ScalarType > res[NUM_METHODS] = { walked, direct };
            for ( int m = 0; m < NUM_METHODS; ++m )
            {
                flags_host( p, sd, m, SLOT_FOUND )   = res[m].found ? 1 : 0;
                flags_host( p, sd, m, SLOT_X )       = res[m].cell.x;
                flags_host( p, sd, m, SLOT_Y )       = res[m].cell.y;
                flags_host( p, sd, m, SLOT_R )       = res[m].cell.r;
                flags_host( p, sd, m, SLOT_W )       = res[m].cell.w;
                flags_host( p, sd, m, SLOT_CLAMPED ) = res[m].clamped_radially ? 1 : 0;
                flags_host( p, sd, m, SLOT_ESCAPED ) = res[m].escaped_laterally ? 1 : 0;
                ref_host( p, sd, m, 0 )              = res[m].xi;
                ref_host( p, sd, m, 1 )              = res[m].eta;
                ref_host( p, sd, m, 2 )              = res[m].zeta;
            }
        }
    }

    // ---- device passes -------------------------------------------------------------------------------------
    // One thread per ( point, subdomain ). Neighbouring threads walk different numbers of steps and take
    // different branches, so this is where a kernel that only works when every lane agrees would fall over.
    //
    // The two routines get a kernel each rather than sharing one. They have very different shapes -- the walk
    // is an unbounded loop whose trip count varies per thread, the O(1) path is a couple of 3x3 solves and a
    // four-step walk -- so putting them in one kernel would let the walk's register pressure and divergence
    // set the occupancy for both and make the timings below meaningless.
    auto cs  = coords_shell_d;
    auto cr_ = coords_radii_d;
    auto pts = points_d;
    auto fl  = flags_d;
    auto rf  = ref_d;

    const auto policy = Kokkos::MDRangePolicy< Kokkos::Rank< 2 > >( { 0, 0 }, { num_points, num_subdomains } );

    // Writes one method's LocateResult into the shared result views.
    const auto store = KOKKOS_LAMBDA( const int p, const int sd, const int m,
                                      const fe::wedge::sl::LocateResult< ScalarType >& res )
    {
        fl( p, sd, m, SLOT_FOUND )   = res.found ? 1 : 0;
        fl( p, sd, m, SLOT_X )       = res.cell.x;
        fl( p, sd, m, SLOT_Y )       = res.cell.y;
        fl( p, sd, m, SLOT_R )       = res.cell.r;
        fl( p, sd, m, SLOT_W )       = res.cell.w;
        fl( p, sd, m, SLOT_CLAMPED ) = res.clamped_radially ? 1 : 0;
        fl( p, sd, m, SLOT_ESCAPED ) = res.escaped_laterally ? 1 : 0;
        rf( p, sd, m, 0 )            = res.xi;
        rf( p, sd, m, 1 )            = res.eta;
        rf( p, sd, m, 2 )            = res.zeta;
    };

    // A single timed sample each, after a warm-up pass to take the first-launch cost out of it. Indicative
    // only -- the point of the split is that the two are measurable apart, not that this is a benchmark.
    ScalarType walk_seconds   = 0.0;
    ScalarType direct_seconds = 0.0;

    for ( int rep = 0; rep < 2; ++rep )
    {
        Kokkos::Timer timer;

        Kokkos::parallel_for(
            "mmoc_random_points_walk", policy, KOKKOS_LAMBDA( const int p, const int sd ) {
                dense::Vec< ScalarType, 3 > X;
                for ( int d = 0; d < 3; ++d )
                    X( d ) = pts( p, d );

                store( p, sd, METHOD_WALK,
                       fe::wedge::sl::locate_point( X, sd, WedgeCell{ 0, 0, 0, 0 }, cs, cr_, bounds, max_steps,
                                                    eps, /*clamp_radially=*/true, r_min, r_max, all_valid ) );
            } );
        Kokkos::fence();
        walk_seconds = timer.seconds();

        timer.reset();

        Kokkos::parallel_for(
            "mmoc_random_points_direct", policy, KOKKOS_LAMBDA( const int p, const int sd ) {
                dense::Vec< ScalarType, 3 > X;
                for ( int d = 0; d < 3; ++d )
                    X( d ) = pts( p, d );

                store( p, sd, METHOD_DIRECT,
                       fe::wedge::sl::locate_point_direct( X, sd, cs, cr_, box, bounds, direct_max_refinements,
                                                           direct_max_walk_steps, eps, /*clamp_radially=*/true,
                                                           r_min, r_max, all_valid ) );
            } );
        Kokkos::fence();
        direct_seconds = timer.seconds();
    }

    auto flags_dev = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, flags_d );
    auto ref_dev   = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, ref_d );

    // ---- analysis ------------------------------------------------------------------------------------------
    long long claim_disagreements = 0; // walk and O(1) disagree on whether a subdomain owns the point
    long long cell_disagreements  = 0; // both claim the point, but land in different wedges
    long long unclaimed_points    = 0; // no subdomain claimed it -- a hole in the tiling
    long long multi_claim_points  = 0; // on a shared diamond edge, where several diamonds legitimately claim it
    long long budget_exhausted    = 0; // O(1) path ran out of steps instead of reporting a clean escape
    long long radially_clamped    = 0; // cone radius fell outside the shell near a radial boundary (see above)
    long long device_mismatches   = 0; // device and host disagree on the discrete result -- always a bug
    long long device_ref_outliers = 0; // reference coordinates differ by more than FMA round-off explains

    ScalarType max_direct_error = 0.0;
    ScalarType max_interp_error = 0.0;
    ScalarType max_device_delta = 0.0; // largest host-vs-device difference in the reference coordinates

    for ( int p = 0; p < num_points; ++p )
    {
        dense::Vec< ScalarType, 3 > X;
        for ( int d = 0; d < 3; ++d )
            X( d ) = points( p, d );

        const std::string at = "point " + std::to_string( p ) + " (level " + std::to_string( level ) + ")";

        int num_claims = 0;

        for ( int sd = 0; sd < num_subdomains; ++sd )
        {
            const std::string where = at + " sd=" + std::to_string( sd );

            // Host versus device, for both methods.
            //
            // The *discrete* result -- which subdomain claims the point, which wedge, and the clamp / escape
            // flags -- must be identical. Those are the outputs the transport scheme branches on, so a host
            // and a device rank that disagreed there would advect differently.
            //
            // The reference coordinates are only required to agree to a tolerance. They cannot be expected to
            // match bit for bit: nvcc contracts multiply-adds into FMAs inside the 3x3 solve of
            // wedge_lateral_cone_coords, which the host compiler does not, so the two evaluate a different
            // (equally valid) sequence of operations. See the reported max delta for how large that gets.
            for ( int m = 0; m < NUM_METHODS; ++m )
            {
                const char* method = ( m == METHOD_WALK ) ? "walking" : "O(1)";

                bool same = true;
                for ( int slot = 0; slot < NUM_SLOTS; ++slot )
                    same = same && ( flags_host( p, sd, m, slot ) == flags_dev( p, sd, m, slot ) );

                if ( !same )
                {
                    ++device_mismatches;
                    check( false, std::string( "device disagrees with host (" ) + method + "): " + where );
                    continue;
                }

                if ( flags_host( p, sd, m, SLOT_FOUND ) == 0 )
                    continue;

                for ( int k = 0; k < 3; ++k )
                {
                    const ScalarType delta = std::abs( ref_host( p, sd, m, k ) - ref_dev( p, sd, m, k ) );
                    max_device_delta       = std::max( max_device_delta, delta );
                    if ( delta > device_ref_tol )
                    {
                        ++device_ref_outliers;
                        check( false,
                               std::string( "device reference coordinates differ from host beyond FMA "
                                            "round-off (" ) +
                                   method + "): " + where );
                    }
                }
            }

            const bool walk_found   = flags_host( p, sd, METHOD_WALK, SLOT_FOUND ) != 0;
            const bool direct_found = flags_host( p, sd, METHOD_DIRECT, SLOT_FOUND ) != 0;

            if ( walk_found != direct_found )
            {
                ++claim_disagreements;
                check( false, "walking and O(1) disagree on ownership: " + where );
                continue;
            }

            if ( !walk_found )
            {
                // Neither claims the point. The reference walk gets far enough to report a clean lateral
                // escape; the O(1) path may instead just exhaust its deliberately tiny budget, which is the
                // same verdict reached for less work. Counted so the difference stays visible.
                if ( flags_host( p, sd, METHOD_WALK, SLOT_ESCAPED ) != 0 &&
                     flags_host( p, sd, METHOD_DIRECT, SLOT_ESCAPED ) == 0 )
                    ++budget_exhausted;
                continue;
            }

            ++num_claims;

            bool same_cell = true;
            for ( int slot = SLOT_X; slot <= SLOT_W; ++slot )
                same_cell = same_cell && ( flags_host( p, sd, METHOD_WALK, slot ) ==
                                           flags_host( p, sd, METHOD_DIRECT, slot ) );

            if ( !same_cell )
            {
                ++cell_disagreements;
                check( false, "walking and O(1) disagree on the wedge: " + where );
                continue;
            }

            WedgeCell cell;
            cell.x = flags_host( p, sd, METHOD_DIRECT, SLOT_X );
            cell.y = flags_host( p, sd, METHOD_DIRECT, SLOT_Y );
            cell.r = flags_host( p, sd, METHOD_DIRECT, SLOT_R );
            cell.w = flags_host( p, sd, METHOD_DIRECT, SLOT_W );

            const ScalarType xi   = ref_host( p, sd, METHOD_DIRECT, 0 );
            const ScalarType eta  = ref_host( p, sd, METHOD_DIRECT, 1 );
            const ScalarType zeta = ref_host( p, sd, METHOD_DIRECT, 2 );

            // The reference coordinates must be a point of the wedge that was returned.
            check( xi >= -tol && eta >= -tol && xi + eta <= 1.0 + tol && std::abs( zeta ) <= 1.0 + tol,
                   "reference coordinates outside the reference wedge: " + where );

            // Self-consistency: mapping the located cell and reference coordinates forward has to return the
            // point they were found from.
            const auto X_back =
                fe::wedge::sl::wedge_forward_map( sd, cell, coords_shell, coords_radii, xi, eta, zeta );
            const ScalarType err = ( X_back - X ).norm();

            if ( flags_host( p, sd, METHOD_DIRECT, SLOT_CLAMPED ) != 0 )
            {
                // Clamping moved the point onto the shell, so the round trip cannot reproduce it. What must
                // hold is that clamping only fired in the chord-sag band next to a radial boundary.
                ++radially_clamped;
                const ScalarType r = X.norm();
                check( r > r_max * ( 1.0 - sag_band ) || r < r_min * ( 1.0 + sag_band ),
                       "radial clamp fired away from a shell boundary: " + where );
                continue;
            }

            max_direct_error = std::max( max_direct_error, err );
            check( err < tol, "random point round-trip error: " + where );

            // A wedge one cell off can still round-trip if the reference coordinates absorb the shift, so
            // check the quantity the transport scheme actually consumes: the Q1 interpolant of a linear field
            // is exact only if the located wedge really contains the point.
            int nx[3], ny[3];
            fe::wedge::sl::wedge_lateral_node_indices( cell, nx, ny );

            ScalarType interpolated = 0.0;
            for ( int j = 0; j < 6; ++j )
            {
                const int                   lateral = j % 3;
                const int                   radial  = j / 3;
                dense::Vec< ScalarType, 3 > node;
                for ( int d = 0; d < 3; ++d )
                    node( d ) = coords_shell( sd, nx[lateral], ny[lateral], d );
                node = node * coords_radii( sd, cell.r + radial );

                interpolated += linear_field( node ) * fe::wedge::shape_lat( j, xi, eta ) *
                                fe::wedge::shape_rad( j, zeta );
            }

            const ScalarType interp_err = std::abs( interpolated - linear_field( X ) );
            max_interp_error            = std::max( max_interp_error, interp_err );
            check( interp_err < tol, "Q1 linear reproduction at a random point: " + where );
        }

        if ( num_claims == 0 )
        {
            ++unclaimed_points;
            check( false, "no subdomain claimed " + at );
        }
        else if ( num_claims > 1 )
        {
            ++multi_claim_points;
        }
    }

    std::cout << std::scientific << std::setprecision( 4 );
    std::cout << "level=" << level << "  random points=" << num_points << " x " << num_subdomains
              << " subdomains x { walk, O(1) } x { host, device }" << std::endl;
    std::cout << "  claim disagreements : " << claim_disagreements << "   cell disagreements: "
              << cell_disagreements << std::endl;
    std::cout << "  device mismatches   : " << device_mismatches << " (discrete)   " << device_ref_outliers
              << " (ref coords beyond " << device_ref_tol << ")   max host-device delta: " << max_device_delta
              << std::endl;
    std::cout << "  unclaimed points    : " << unclaimed_points << "   on a shared edge: " << multi_claim_points
              << "   O(1) budget exhausted instead of escaping: " << budget_exhausted << std::endl;
    std::cout << "  max |X_back - X|    : " << max_direct_error << "   max linear interp err: "
              << max_interp_error << std::endl;
    std::cout << "  radially clamped    : " << radially_clamped << " of " << num_points
              << "   (cone-radius sag band = " << sag_band << " of the shell radius)" << std::endl;
    std::cout << "  device kernels      : walk " << walk_seconds << " s   O(1) " << direct_seconds
              << " s   (one sample each over " << num_points * num_subdomains << " threads)" << std::endl;

    check( claim_disagreements == 0, "walking and O(1) location disagree on subdomain ownership" );
    check( cell_disagreements == 0, "walking and O(1) location disagree on the located wedge" );
    check( unclaimed_points == 0, "some random shell points were claimed by no subdomain" );
    check( device_mismatches == 0, "device location disagrees with the host location" );
    check( device_ref_outliers == 0, "device reference coordinates drift further from the host than FMA "
                                     "contraction accounts for" );
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    test_random_shell_points( 3, 2000 );
    test_random_shell_points( 5, 2000 );

    if ( g_failures == 0 )
    {
        std::cout << "\ntest_mmoc_point_location_random: PASSED" << std::endl;
    }
    else
    {
        std::cout << "\ntest_mmoc_point_location_random: FAILED (" << g_failures << " checks)" << std::endl;
    }
    return g_failures == 0 ? 0 : 1;
}
