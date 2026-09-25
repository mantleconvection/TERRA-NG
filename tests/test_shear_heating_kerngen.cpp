// Validates the kernel-generated shear-heating operator (ShearHeatingKerngen)
// two ways:
//   (1) against the closed-form volume integral of the viscous dissipation for
//       a linear velocity field with viscosity mu = r^2 (same manufactured case
//       as test_shear_heating.cpp), and
//   (2) against the reference ShearHeatingSimple operator, node for node.
//
// Linear velocity  ux = x+y+z, uy = 2x+y+z, uz = x+3y+z  gives a constant
// velocity gradient, so eps_dev:eps_dev = 14.5 everywhere, and
//   int Phi dV = int 2 mu (eps_dev:eps_dev) dV = 2*14.5 * int r^2 dV
//              = 2*14.5 * (4/5) pi (rMax^5 - rMin^5).
// The operator assembles int Phi N_i, so s.f_dst = int Phi dV; dividing by 2
// recovers 14.5*(4/5)pi(rMax^5-rMin^5).

#include "fe/wedge/operators/shell/shear_heating_kerngen.hpp"
#include "fe/wedge/operators/shell/shear_heating_simple.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

// Fills a VectorQ1Vec velocity with the linear field, and (redundantly) the
// three scalar components + viscosity used by the reference simple operator.
struct FieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    grid::Grid4DDataVec< double, 3 > u_;
    Grid4DDataScalar< double > ux_, uy_, uz_, mu_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        const double ux = c( 0 ) + c( 1 ) + c( 2 );
        const double uy = 2.0 * c( 0 ) + c( 1 ) + c( 2 );
        const double uz = c( 0 ) + 3.0 * c( 1 ) + c( 2 );
        u_( s, x, y, r, 0 ) = ux;
        u_( s, x, y, r, 1 ) = uy;
        u_( s, x, y, r, 2 ) = uz;
        ux_( s, x, y, r )   = ux;
        uy_( s, x, y, r )   = uy;
        uz_( s, x, y, r )   = uz;
        mu_( s, x, y, r )   = c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) + c( 2 ) * c( 2 );
    }
};

using ScalarType = double;

// Runs the manufactured case at one refinement level. Returns the absolute
// error of the assembled shear-heating volume integral vs the analytical value,
// and writes the relative difference to the reference simple operator via
// *rel_diff_out.
static double run_level( unsigned level, double rMin, double rMax, double* rel_diff_out )
{
    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, rMin, rMax );

    auto mask_data    = grid::setup_node_ownership_mask_data( domain );
    const auto coords = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto radii  = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Vec< ScalarType, 3 > u( "u", domain, mask_data );
    VectorQ1Scalar< ScalarType > mu( "mu", domain, mask_data );
    VectorQ1Scalar< ScalarType > ux( "ux", domain, mask_data );
    VectorQ1Scalar< ScalarType > uy( "uy", domain, mask_data );
    VectorQ1Scalar< ScalarType > uz( "uz", domain, mask_data );
    VectorQ1Scalar< ScalarType > s_h( "s_h", domain, mask_data );
    VectorQ1Scalar< ScalarType > f_kerngen( "f_kerngen", domain, mask_data );
    VectorQ1Scalar< ScalarType > f_simple( "f_simple", domain, mask_data );

    linalg::assign( s_h, ScalarType( 1 ) );

    Kokkos::parallel_for(
        "field_interp",
        local_domain_md_range_policy_nodes( domain ),
        FieldInterpolator{
            coords, radii, u.grid_data(), ux.grid_data(), uy.grid_data(), uz.grid_data(), mu.grid_data() } );
    Kokkos::fence();

    // Kerngen operator (scale = 1).
    fe::wedge::operators::shell::ShearHeatingKerngen< ScalarType > op_kerngen(
        domain, coords, radii, mu.grid_data() );
    linalg::apply( op_kerngen, u, f_kerngen );

    // Reference simple operator (its scalar src is unused inside; pass s_h).
    fe::wedge::operators::shell::ShearHeatingSimple< ScalarType > op_simple(
        domain, coords, radii, mu.grid_data(), ux.grid_data(), uy.grid_data(), uz.grid_data() );
    linalg::apply( op_simple, s_h, f_simple );

    const double analytical =
        14.5 * ( 4.0 / 5.0 ) * M_PI * ( std::pow( rMax, 5 ) - std::pow( rMin, 5 ) );
    const double integral       = linalg::dot( s_h, f_kerngen ) / 2.0;
    const double integral_error = std::abs( integral - analytical );

    VectorQ1Scalar< ScalarType > diff( "diff", domain, mask_data );
    linalg::assign( diff, ScalarType( 0 ) );
    linalg::lincomb( diff, { ScalarType( 1 ), ScalarType( -1 ) }, { f_kerngen, f_simple } );
    const double diff_norm = std::sqrt( linalg::dot( diff, diff ) );
    const double ref_norm  = std::sqrt( linalg::dot( f_simple, f_simple ) );
    *rel_diff_out          = ref_norm > 0 ? diff_norm / ref_norm : diff_norm;

    return integral_error;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const double rMin = 0.5;
    const double rMax = 1.0;

    // Convergence sweep: error should shrink ~monotonically with refinement.
    const unsigned levels[] = { 2u, 3u, 4u, 5u, 6u };
    const int      n        = sizeof( levels ) / sizeof( levels[0] );
    double         err[16]  = {};
    double         reld[16] = {};

    std::printf( "%-6s %-16s %-10s %-14s\n", "level", "abs_error", "order", "rel_vs_simple" );
    for ( int i = 0; i < n; ++i )
    {
        reld[i] = 0.0;
        err[i]  = run_level( levels[i], rMin, rMax, &reld[i] );
        if ( i > 0 && err[i] > 0.0 )
        {
            const double order = std::log( err[i - 1] / err[i] ) / std::log( 2.0 );
            std::printf( "%-6u %-16.3e %-10.2f %-14.3e\n", levels[i], err[i], order, reld[i] );
        }
        else
        {
            std::printf( "%-6u %-16.3e %-10s %-14.3e\n", levels[i], err[i], "-", reld[i] );
        }
    }

    // Correctness gates at the finest level.
    if ( err[n - 1] > 0.025 )
    {
        Kokkos::abort( "ShearHeatingKerngen: analytical integral error too high!" );
    }
    if ( reld[n - 1] > 1e-10 )
    {
        Kokkos::abort( "ShearHeatingKerngen: disagrees with ShearHeatingSimple!" );
    }
    // Convergence gate: finest error must be below coarsest by a clear margin.
    if ( !( err[n - 1] < err[0] ) )
    {
        Kokkos::abort( "ShearHeatingKerngen: error did not decrease under refinement!" );
    }

    return 0;
}
