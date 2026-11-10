// Test: the cubic (PCHIP) reconstruction of fe/wedge/sl/point_location.hpp, in isolation.
//
// The cubic evaluators only ever see a field view, the shell radii and the lateral validity mask -- never a
// domain -- so the whole test runs on a hand-built 5x5x5 index block with invented, deliberately non-uniform
// radii. No grid, no point location, no transport step: a failure here is a failure of the interpolation.
//
// What is checked:
//
//   1. pchip_interpolate_1d, the one-dimensional kernel: it reproduces the nodal values, is exact for a
//      quadratic on non-uniform nodes (the three-point parabolic slope estimates are the exact ones there),
//      stays exact for linear data once the limiter is on, and keeps both properties on the shortened three-
//      and two-node stencils used where a full window does not fit.
//   2. The limiter itself, on step data: unlimited, the one-sided end slope drags the first interval to
//      -0.125; limited, the interval does not leave the data at all.
//   3. evaluate_cubic_scalar / evaluate_cubic_vec on the tensor product: exact to round-off for a field that
//      is quadratic in each lateral index direction and in the radius, over every cell of the block, both
//      triangles of each hex cell, and reference coordinates including the cell corners. Q1 runs alongside on
//      the same data and is asserted to be wrong by ~1e-1 -- without that guard the exactness check would
//      pass just as happily if the cubic path silently fell back to Q1 everywhere.
//   4. The two Q1 fallbacks: a cell outside the stencil bounds, and a stencil window covering an invalid
//      lateral node, must each reproduce evaluate_q1_scalar exactly.
//   5. The local bound, on fields with a jump: the limited evaluation must stay inside the range of the
//      containing cell's own eight nodes, where the unlimited one undershoots to -0.19. Note that for a foot
//      point inside its cell the limiter alone already delivers that bound -- each sweep is bounded by its
//      own interval's endpoints and the three compose -- so the clip after the sweeps is only load-bearing
//      for a foot point that has drifted out of its cell, which is checked separately.

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "fe/wedge/sl/point_location.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;
using ScalarType = double;

using fe::wedge::sl::IndexBounds;
using fe::wedge::sl::StencilBounds;
using fe::wedge::sl::StencilRange;
using fe::wedge::sl::WedgeCell;

namespace
{

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        std::cout << "  FAIL: " << what << std::endl;
    }
}

std::string fmt( const ScalarType x )
{
    std::ostringstream os;
    os << std::scientific << std::setprecision( 4 ) << x;
    return os.str();
}

void check_close( const ScalarType got, const ScalarType expected, const ScalarType tol, const std::string& what )
{
    check( std::abs( got - expected ) <= tol, what + " (got " + fmt( got ) + ", expected " + fmt( expected ) + ")" );
}

// ---- 1. the one-dimensional kernel -----------------------------------------------------------------------

void test_pchip_1d()
{
    // Non-uniform on purpose: this is what the radial pass sees, where the layer thicknesses vary.
    const ScalarType xs[4] = { 0.0, 0.7, 1.0, 2.2 };

    const auto quadratic = []( const ScalarType x ) { return 0.5 - 1.25 * x + 0.75 * x * x; };
    const auto linear    = []( const ScalarType x ) { return -0.3 + 1.7 * x; };

    ScalarType fq[4], fl[4];
    for ( int k = 0; k < 4; ++k )
    {
        fq[k] = quadratic( xs[k] );
        fl[k] = linear( xs[k] );
    }

    // Nodal reproduction holds whatever the slopes are -- the Hermite basis is interpolatory -- so it must
    // hold with the limiter on as well.
    for ( const bool monotone : { false, true } )
        for ( int k = 0; k < 4; ++k )
            check_close( fe::wedge::sl::pchip_interpolate_1d( xs, fq, 4, xs[k], monotone ),
                         fq[k],
                         1e-15,
                         "pchip does not reproduce node " + std::to_string( k ) );

    for ( const ScalarType xq : { 0.05, 0.3, 0.7, 0.9, 1.4, 2.0 } )
    {
        // Unlimited, the parabolic slope estimates are exact for a quadratic, so the Hermite cubic collapses
        // onto the quadratic itself. (With the limiter it would not: this quadratic has its vertex inside the
        // range, and flattening a smooth extremum is exactly what PCHIP is documented to do.)
        check_close( fe::wedge::sl::pchip_interpolate_1d( xs, fq, 4, xq, false ),
                     quadratic( xq ),
                     1e-14,
                     "pchip is not exact for a quadratic at x = " + fmt( xq ) );

        // Linear data survives the limiter untouched: all secants are equal, so no slope is capped.
        check_close( fe::wedge::sl::pchip_interpolate_1d( xs, fl, 4, xq, true ),
                     linear( xq ),
                     1e-14,
                     "monotone pchip is not exact for a linear field at x = " + fmt( xq ) );
    }

    // The shortened stencils, used where a four-node window does not fit against a boundary.
    check_close( fe::wedge::sl::pchip_interpolate_1d( xs, fq, 3, 0.5, false ),
                 quadratic( 0.5 ),
                 1e-14,
                 "three-node pchip is not exact for a quadratic" );
    check_close( fe::wedge::sl::pchip_interpolate_1d( xs, fl, 2, 0.3, true ),
                 linear( 0.3 ),
                 1e-14,
                 "two-node pchip is not exact for a linear field" );

    // The limiter on a step.
    const ScalarType step_x[4] = { 0.0, 1.0, 2.0, 3.0 };
    const ScalarType step_f[4] = { 0.0, 0.0, 1.0, 1.0 };

    check_close( fe::wedge::sl::pchip_interpolate_1d( step_x, step_f, 4, 0.5, false ),
                 -0.125,
                 1e-14,
                 "the unlimited pchip no longer undershoots step data, so the limiter check below is vacuous" );
    check_close( fe::wedge::sl::pchip_interpolate_1d( step_x, step_f, 4, 0.5, true ),
                 0.0,
                 1e-15,
                 "monotone pchip undershoots step data" );
    check_close( fe::wedge::sl::pchip_interpolate_1d( step_x, step_f, 4, 2.5, true ),
                 1.0,
                 1e-15,
                 "monotone pchip overshoots step data" );
}

// ---- the hand-built index block --------------------------------------------------------------------------

constexpr int        num_nodes            = 5;
constexpr ScalarType shell_radii[num_nodes] = { 0.50, 0.62, 0.71, 0.86, 1.00 };

using FieldView    = Kokkos::View< ScalarType****, Kokkos::HostSpace >;
using VecFieldView = Kokkos::View< ScalarType*****, Kokkos::HostSpace >;
using RadiiView    = Kokkos::View< ScalarType**, Kokkos::HostSpace >;
using ValidView    = Kokkos::View< uint8_t***, Kokkos::HostSpace >;

// Reference coordinates inside the wedge (xi, eta >= 0, xi + eta <= 1), including a corner and the centroid.
constexpr int        num_samples             = 5;
constexpr ScalarType sample_xi[num_samples]  = { 0.1, 0.5, 0.0, 1.0 / 3.0, 0.6 };
constexpr ScalarType sample_eta[num_samples] = { 0.2, 0.25, 0.0, 1.0 / 3.0, 0.3 };
constexpr int        num_zetas               = 5;
constexpr ScalarType sample_zeta[num_zetas]  = { -1.0, -0.4, 0.0, 0.7, 1.0 };

/// Quadratic in each lateral index coordinate and in the radius -- the largest space a tensor product of
/// parabolic-slope Hermite cubics reproduces exactly, so the reconstruction owes us this one to round-off.
ScalarType tri_quadratic( const ScalarType u, const ScalarType v, const ScalarType r )
{
    return 1.0 + 0.7 * u - 0.4 * v + 1.3 * r + 0.25 * u * u - 0.5 * u * v + 0.9 * v * r + 0.3 * u * u * r -
           0.6 * v * v;
}

/// Largest amount by which the limited reconstruction leaves the range of the containing cell's own eight
/// nodes, over every cell of the block, both triangles and every sample point.
ScalarType worst_cell_range_escape( const FieldView&     field,
                                    const RadiiView&     radii,
                                    const StencilBounds& stencil,
                                    const ValidView&     valid )
{
    ScalarType worst = 0.0;

    for ( int x = 0; x + 1 < num_nodes; ++x )
        for ( int y = 0; y + 1 < num_nodes; ++y )
            for ( int r = 0; r + 1 < num_nodes; ++r )
                for ( int w = 0; w < 2; ++w )
                    for ( int s = 0; s < num_samples; ++s )
                        for ( int t = 0; t < num_zetas; ++t )
                        {
                            const WedgeCell cell{ x, y, r, w };

                            const ScalarType value = fe::wedge::sl::evaluate_cubic_scalar(
                                field, 0, cell, sample_xi[s], sample_eta[s], sample_zeta[t], radii, stencil,
                                valid, true );

                            ScalarType lo = 0.0, hi = 0.0;
                            fe::wedge::sl::cell_value_range( field, 0, cell, lo, hi );

                            worst = std::max( worst, std::max( lo - value, value - hi ) );
                        }

    return worst;
}

// ---- 2. the tensor-product evaluation --------------------------------------------------------------------

void test_tensor_product()
{
    RadiiView    radii( "radii", 1, num_nodes );
    ValidView    valid( "valid", 1, num_nodes, num_nodes );
    FieldView    field( "field", 1, num_nodes, num_nodes, num_nodes );
    VecFieldView vfield( "vfield", 1, num_nodes, num_nodes, num_nodes, 2 );

    for ( int k = 0; k < num_nodes; ++k )
        radii( 0, k ) = shell_radii[k];
    Kokkos::deep_copy( valid, static_cast< uint8_t >( 1 ) );

    for ( int i = 0; i < num_nodes; ++i )
        for ( int j = 0; j < num_nodes; ++j )
            for ( int k = 0; k < num_nodes; ++k )
            {
                field( 0, i, j, k ) = tri_quadratic(
                    static_cast< ScalarType >( i ), static_cast< ScalarType >( j ), shell_radii[k] );
                vfield( 0, i, j, k, 0 ) = field( 0, i, j, k );
                vfield( 0, i, j, k, 1 ) = 2.0 - 0.5 * field( 0, i, j, k );
            }

    const IndexBounds   bounds{ num_nodes, num_nodes, num_nodes };
    const StencilBounds stencil = fe::wedge::sl::full_stencil_bounds( bounds );

    ScalarType max_cubic_error = 0.0;
    ScalarType max_vec_error   = 0.0;
    ScalarType max_q1_error    = 0.0;

    for ( int x = 0; x + 1 < num_nodes; ++x )
        for ( int y = 0; y + 1 < num_nodes; ++y )
            for ( int r = 0; r + 1 < num_nodes; ++r )
                for ( int w = 0; w < 2; ++w )
                    for ( int s = 0; s < num_samples; ++s )
                        for ( int t = 0; t < num_zetas; ++t )
                        {
                            const WedgeCell  cell{ x, y, r, w };
                            const ScalarType xi   = sample_xi[s];
                            const ScalarType eta  = sample_eta[s];
                            const ScalarType zeta = sample_zeta[t];

                            // Where the point sits in the space the reconstruction actually works in: the
                            // lateral index coordinates and the cone radius.
                            ScalarType u = 0.0, v = 0.0;
                            fe::wedge::sl::wedge_lateral_index_coords( cell, xi, eta, u, v );
                            const ScalarType rho =
                                fe::wedge::sl::wedge_radius_from_zeta( 0, cell, radii, zeta );
                            const ScalarType exact = tri_quadratic( u, v, rho );

                            const ScalarType cubic = fe::wedge::sl::evaluate_cubic_scalar(
                                field, 0, cell, xi, eta, zeta, radii, stencil, valid, false );
                            const ScalarType q1 =
                                fe::wedge::sl::evaluate_q1_scalar( field, 0, cell, xi, eta, zeta );

                            const auto vec = fe::wedge::sl::evaluate_cubic_vec< ScalarType, 2 >(
                                vfield, 0, cell, xi, eta, zeta, radii, stencil, valid, false );

                            max_cubic_error = std::max( max_cubic_error, std::abs( cubic - exact ) );
                            max_q1_error    = std::max( max_q1_error, std::abs( q1 - exact ) );
                            max_vec_error   = std::max(
                                max_vec_error,
                                std::max( std::abs( vec( 0 ) - exact ),
                                          std::abs( vec( 1 ) - ( 2.0 - 0.5 * exact ) ) ) );
                        }

    std::cout << "  tri-quadratic field: cubic err=" << fmt( max_cubic_error )
              << "  cubic_vec err=" << fmt( max_vec_error ) << "  Q1 err=" << fmt( max_q1_error ) << std::endl;

    check( max_cubic_error < 1e-12, "evaluate_cubic_scalar is not exact for a tri-quadratic field" );
    check( max_vec_error < 1e-12, "evaluate_cubic_vec disagrees with the scalar reconstruction" );

    // Without this the checks above would pass on a silent fallback: Q1 cannot reproduce this field, so the
    // exactness only means something as long as Q1 is measurably wrong on the same data.
    check( max_q1_error > 1e-3, "Q1 reproduces the test field, so the exactness checks above prove nothing" );
}

// ---- 3. the fallbacks and the local bound ----------------------------------------------------------------

void test_fallbacks_and_limiter()
{
    RadiiView radii( "radii", 1, num_nodes );
    ValidView valid( "valid", 1, num_nodes, num_nodes );
    FieldView field( "field", 1, num_nodes, num_nodes, num_nodes );

    for ( int k = 0; k < num_nodes; ++k )
        radii( 0, k ) = shell_radii[k];
    Kokkos::deep_copy( valid, static_cast< uint8_t >( 1 ) );

    for ( int i = 0; i < num_nodes; ++i )
        for ( int j = 0; j < num_nodes; ++j )
            for ( int k = 0; k < num_nodes; ++k )
                field( 0, i, j, k ) = tri_quadratic(
                    static_cast< ScalarType >( i ), static_cast< ScalarType >( j ), shell_radii[k] );

    const IndexBounds   bounds{ num_nodes, num_nodes, num_nodes };
    const StencilBounds stencil = fe::wedge::sl::full_stencil_bounds( bounds );

    const ScalarType xi = 0.3, eta = 0.25, zeta = -0.2;

    // (a) A cell outside the stencil bounds. Trimming the lateral range to 1..4 puts cell.x = 0 outside it,
    //     and the stencil polynomial would have to be extrapolated to reach the point -- worse than a linear
    //     evaluation inside the cell -- so the point must be handed back to Q1.
    {
        const WedgeCell     cell{ 0, 2, 2, 0 };
        const StencilBounds trimmed{ StencilRange{ 1, num_nodes - 1 },
                                     StencilRange{ 0, num_nodes - 1 },
                                     StencilRange{ 0, num_nodes - 1 } };

        check_close( fe::wedge::sl::evaluate_cubic_scalar(
                         field, 0, cell, xi, eta, zeta, radii, trimmed, valid, false ),
                     fe::wedge::sl::evaluate_q1_scalar( field, 0, cell, xi, eta, zeta ),
                     1e-14,
                     "a cell outside the stencil bounds does not fall back to Q1" );
    }

    // (b) An invalid lateral node inside the window. The window of the interior cell (2, 2) is centred, so it
    //     spans nodes 1..4 laterally; marking (1, 1) invalid -- a degenerate diagonal ghost corner -- leaves
    //     no usable structured neighbourhood and must send the evaluation to Q1 as well.
    {
        const WedgeCell cell{ 2, 2, 2, 0 };

        const ScalarType with_valid_stencil =
            fe::wedge::sl::evaluate_cubic_scalar( field, 0, cell, xi, eta, zeta, radii, stencil, valid, false );

        valid( 0, 1, 1 ) = 0;
        const ScalarType fallback =
            fe::wedge::sl::evaluate_cubic_scalar( field, 0, cell, xi, eta, zeta, radii, stencil, valid, false );
        valid( 0, 1, 1 ) = 1;

        check_close( fallback,
                     fe::wedge::sl::evaluate_q1_scalar( field, 0, cell, xi, eta, zeta ),
                     1e-14,
                     "an invalid lateral node in the stencil does not fall back to Q1" );
        check( std::abs( fallback - with_valid_stencil ) > 1e-6,
               "the validity mask made no difference, so the fallback check above is vacuous" );
    }

    // (c) The local bound. With the limiter on, the reconstruction must stay inside the range of the
    //     containing cell's own eight nodes -- the property Q1 has for free, and the one a semi-Lagrangian
    //     step needs: a value outside that range is a new extremum the transport then has to carry, and the
    //     caller's range clip turns a systematic undershoot into added mass. Nothing clips the result to that
    //     range; it falls out of the per-interval bound of the Fritsch-Carlson limiter, composed over the
    //     three sweeps. That composition is the part worth testing, so it is checked on a jump across the
    //     radial pass, on a diagonal jump that drives all three passes across a discontinuity at once, and on
    //     random fields -- two hand-picked discontinuities say little about a tensor product.
    for ( int i = 0; i < num_nodes; ++i )
        for ( int j = 0; j < num_nodes; ++j )
            for ( int k = 0; k < num_nodes; ++k )
                field( 0, i, j, k ) = ( k >= 2 ) ? 1.0 : 0.0;

    const ScalarType radial_escape = worst_cell_range_escape( field, radii, stencil, valid );

    // The same field unlimited: without the limiter the radial pass undershoots to about -0.19, which is what
    // makes the bound above a real check rather than a restatement of data that was already in range.
    ScalarType min_unlimited = 0.0;
    for ( int r = 0; r + 1 < num_nodes; ++r )
        for ( int t = 0; t < num_zetas; ++t )
        {
            const WedgeCell cell{ 2, 2, r, 0 };
            min_unlimited = std::min( min_unlimited,
                                      fe::wedge::sl::evaluate_cubic_scalar( field, 0, cell, sample_xi[0],
                                                                            sample_eta[0], sample_zeta[t],
                                                                            radii, stencil, valid, false ) );
        }

    for ( int i = 0; i < num_nodes; ++i )
        for ( int j = 0; j < num_nodes; ++j )
            for ( int k = 0; k < num_nodes; ++k )
                field( 0, i, j, k ) = ( i + j + k >= 6 ) ? 1.0 : 0.0;

    const ScalarType diagonal_escape = worst_cell_range_escape( field, radii, stencil, valid );

    // Random 0/1 patterns are the hardest case a limiter sees; uniform noise alternates with them.
    uint64_t   rng  = 0x9e3779b97f4a7c15ull;
    const auto next = [&rng]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return static_cast< ScalarType >( ( rng >> 11 ) & 0xfffffu ) / static_cast< ScalarType >( 0x100000u );
    };

    ScalarType random_escape = 0.0;
    for ( int trial = 0; trial < 200; ++trial )
    {
        const bool binary = ( trial % 2 == 0 );
        for ( int i = 0; i < num_nodes; ++i )
            for ( int j = 0; j < num_nodes; ++j )
                for ( int k = 0; k < num_nodes; ++k )
                {
                    const ScalarType x  = next();
                    field( 0, i, j, k ) = binary ? ( ( x < 0.5 ) ? 0.0 : 1.0 ) : x;
                }

        random_escape = std::max( random_escape, worst_cell_range_escape( field, radii, stencil, valid ) );
    }

    std::cout << "  cell-range escape: radial jump=" << fmt( radial_escape )
              << "  diagonal jump=" << fmt( diagonal_escape ) << "  random fields=" << fmt( random_escape )
              << "  unlimited min=" << fmt( min_unlimited ) << std::endl;

    check( radial_escape <= 1e-15, "the monotone cubic leaves its cell's range across a radial jump" );
    check( diagonal_escape <= 1e-15, "the monotone cubic leaves its cell's range across a diagonal jump" );
    check( random_escape <= 1e-15, "the monotone cubic leaves its cell's range on a random field" );
    check( min_unlimited < -1e-3,
           "the unlimited cubic no longer undershoots the jump, so the bound checks above are vacuous" );

    // (d) A foot point that has drifted out of its own cell -- which is the case the clip after the three
    //     limited sweeps actually exists for. locate_point reports xi, eta and zeta straight from the
    //     closed-form inverse without clamping them onto the reference wedge, so a departure point can arrive
    //     here sitting in a neighbouring cell of the stencil. Each sweep is then bounded by *that* interval's
    //     nodes rather than by the located cell's, and the limiter alone no longer keeps the value inside the
    //     range of the data the cell actually holds. Back on the radial jump field, cell (2, 2, 0) is
    //     uniformly zero while zeta = 1.8 reaches a cell further out, into the jump.
    for ( int i = 0; i < num_nodes; ++i )
        for ( int j = 0; j < num_nodes; ++j )
            for ( int k = 0; k < num_nodes; ++k )
                field( 0, i, j, k ) = ( k >= 2 ) ? 1.0 : 0.0;

    const WedgeCell  drifted{ 2, 2, 0, 0 };
    const ScalarType drifted_zeta = 1.8;

    const ScalarType drifted_limited = fe::wedge::sl::evaluate_cubic_scalar(
        field, 0, drifted, 0.1, 0.2, drifted_zeta, radii, stencil, valid, true );
    const ScalarType drifted_unlimited = fe::wedge::sl::evaluate_cubic_scalar(
        field, 0, drifted, 0.1, 0.2, drifted_zeta, radii, stencil, valid, false );

    check_close( drifted_limited,
                 0.0,
                 1e-15,
                 "a drifted foot point is not clipped into the range of its own cell's nodes" );
    check( drifted_unlimited > 1e-3,
           "the drifted foot point no longer leaves its cell's range, so the clip check above is vacuous" );
}

} // namespace

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    test_pchip_1d();
    test_tensor_product();
    test_fallbacks_and_limiter();

    if ( g_failures == 0 )
    {
        std::cout << "\ntest_mmoc_cubic_interpolation: PASSED" << std::endl;
    }
    else
    {
        std::cout << "\ntest_mmoc_cubic_interpolation: FAILED (" << g_failures << " checks)" << std::endl;
    }
    return g_failures == 0 ? 0 : 1;
}
