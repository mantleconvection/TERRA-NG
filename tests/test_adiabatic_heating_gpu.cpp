// Manufactured-solution unit test for the compressible (TALA) ADIABATIC heat source and the W integral
// used in the dissipation balance -- the two pieces with no coverage today. Mirrors test_shear_heating.
//
// It exercises the REAL app functor terra::mantlecirculation::AdiabaticHeatingSource (S = pref*Di*(u.n)*T)
// and the same W = integral of Di*(u.n)*T dV reduction (lumped mass) that log_dissipation_balance() uses.
//
// Manufactured fields on the shell (rMin..rMax), full sphere:
//     u(x) = (x,y,z)          => u.n = r          (n = x/|x|, radial unit)
//     T(x) = r = |x|
//   so the integrand is  Di*(u.n)*T = Di*r^2, with the CLOSED-FORM integral
//     W = Di * integral_{rMin}^{rMax} r^2 * 4 pi r^2 dr = Di * 4 pi (rMax^5 - rMin^5) / 5.
//
// Checks:
//   (1) functor scaling, to machine eps: node value == Di*r^2                      (pins Di, u.n, T)
//   (2) sign: with prefactor -1 the source is <= 0 everywhere (rising material cools), and == -w
//   (3) W integral vs the closed form, O(h^2) tolerance                            (pins the reduction)
//
// Needs a GPU (fields live in device memory) -- run on an MI250X/H100 node.

// interpolators.hpp is not self-contained -- it uses NodeOwnershipFlag / ShellBoundaryFlag, which the app
// pulls in before it. Provide them first.
#include "terra/grid/bit_masks.hpp"
#include "terra/grid/shell/bit_masks.hpp"
#include "../apps/mantlecirculation/src/interpolators.hpp" // terra::mantlecirculation::AdiabaticHeatingSource
#include "linalg/vector_q1.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <cmath>
#include <cstdio>

using namespace terra;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using ST = double;

static int g_fail = 0, g_checks = 0;
#define CHECK( cond )                                                       \
    do                                                                      \
    {                                                                       \
        ++g_checks;                                                         \
        if ( !( cond ) )                                                    \
        {                                                                   \
            std::printf( "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond ); \
            ++g_fail;                                                       \
        }                                                                   \
    } while ( 0 )

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        const unsigned level = 5u;
        const ST       rMin = 0.5, rMax = 1.0, Di = 0.5;

        const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, rMin, rMax );
        auto       mask   = grid::setup_node_ownership_mask_data( domain );
        const auto cs     = grid::shell::subdomain_unit_sphere_single_shell_coords< ST >( domain );
        const auto cr     = grid::shell::subdomain_shell_radii< ST >( domain );

        VectorQ1Vec< ST, 3 >  u( "u", domain, mask );
        VectorQ1Scalar< ST >  T( "T", domain, mask ), w( "w", domain, mask ), s( "s", domain, mask );
        VectorQ1Scalar< ST >  ref( "ref", domain, mask ), ones( "ones", domain, mask ), Mlump( "Mlump", domain, mask );
        VectorQ1Scalar< ST >  diff( "diff", domain, mask );

        // fill u = coords, T = |coords| = r, ref = Di*r^2 (the expected integrand w)
        auto u0 = u.grid_data().comp_[0], u1 = u.grid_data().comp_[1], u2 = u.grid_data().comp_[2];
        auto Tg = T.grid_data(), rg = ref.grid_data();
        Kokkos::parallel_for(
            "fill", grid::shell::local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( const int id, const int x, const int y, const int r ) {
                const dense::Vec< ST, 3 > c = grid::shell::coords( id, x, y, r, cs, cr );
                u0( id, x, y, r ) = c( 0 ); u1( id, x, y, r ) = c( 1 ); u2( id, x, y, r ) = c( 2 );
                const ST rr       = Kokkos::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 ) );
                Tg( id, x, y, r ) = rr;
                rg( id, x, y, r ) = Di * rr * rr;
            } );
        Kokkos::fence();
        linalg::assign( ones, ST( 1 ) );

        // run the REAL adiabatic functor with prefactor +1 -> w = Di*(u.n)*T
        Kokkos::parallel_for(
            "adiab_plus", grid::shell::local_domain_md_range_policy_nodes( domain ),
            terra::mantlecirculation::AdiabaticHeatingSource{ cs, cr, u.grid_data(), T.grid_data(),
                                                              w.grid_data(), Di, ST( +1 ) } );
        // and with prefactor -1 -> the actual source S = -Di*(u.n)*T
        Kokkos::parallel_for(
            "adiab_minus", grid::shell::local_domain_md_range_policy_nodes( domain ),
            terra::mantlecirculation::AdiabaticHeatingSource{ cs, cr, u.grid_data(), T.grid_data(),
                                                              s.grid_data(), Di, ST( -1 ) } );
        Kokkos::fence();

        // ---- (1) functor scaling to machine eps + (2) sign, on host mirrors ----
        auto wh = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, w.grid_data() );
        auto rh = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, ref.grid_data() );
        auto sh = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, s.grid_data() );
        const int nsub = (int) wh.extent( 0 ), nx = (int) wh.extent( 1 ), ny = (int) wh.extent( 2 ),
                  nr = (int) wh.extent( 3 );
        ST max_scale_err = 0, max_sign = -1e300, max_sum = 0;
        for ( int sdi = 0; sdi < nsub; ++sdi )
            for ( int x = 0; x < nx; ++x )
                for ( int y = 0; y < ny; ++y )
                    for ( int r = 0; r < nr; ++r )
                    {
                        max_scale_err = std::max( max_scale_err, std::fabs( wh( sdi, x, y, r ) - rh( sdi, x, y, r ) ) );
                        max_sign      = std::max( max_sign, sh( sdi, x, y, r ) );                        // must be <= 0
                        max_sum       = std::max( max_sum, std::fabs( sh( sdi, x, y, r ) + wh( sdi, x, y, r ) ) );
                    }
        std::printf( "  (1) functor: max|w - Di*r^2| = %.3e\n", max_scale_err );
        std::printf( "  (2) sign:    max(S_adiab)    = %.3e   max|S + w| = %.3e\n", max_sign, max_sum );
        CHECK( max_scale_err < 1e-10 ); // scaling exact
        CHECK( max_sign <= 1e-14 );     // rising (u_r>0), T>0  =>  S_adiab <= 0
        CHECK( max_sum < 1e-10 );       // prefactor -1 is exactly minus the +1 field

        // ---- (3) W = integral Di*(u.n)*T dV  via lumped mass (M_lumped = M*1), vs closed form ----
        fe::wedge::operators::shell::Mass< ST > mass( domain, cs, cr );
        linalg::apply( mass, ones, Mlump );
        const ST W          = linalg::dot( w, Mlump );
        const ST W_analytic = Di * 4.0 * M_PI * ( std::pow( rMax, 5 ) - std::pow( rMin, 5 ) ) / 5.0;
        const ST relerr     = std::fabs( W - W_analytic ) / W_analytic;
        std::printf( "  (3) W = %.6f   analytic = %.6f   rel.err = %.3e\n", W, W_analytic, relerr );
        CHECK( relerr < 0.02 ); // O(h^2) quadrature/geometry error at level 5
    }

    const bool ok = ( g_fail == 0 );
    std::printf( "test_adiabatic_heating_gpu: %s (%d checks)\n", ok ? "ALL PASS" : "FAILURE(S)", g_checks );
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
