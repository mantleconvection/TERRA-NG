// Run-to-run reproducibility of the operators of the mantle circulation app with the atomic scatter
// (default) and with the fixed-point scatter (TERRA_DETERMINISTIC_SCATTER=1, see
// kernels/common/deterministic.hpp): EpsDivDiv (matvec and diagonal), gradient, divergence, the SUPG
// energy operator, the mass operators and the constant grid transfers.
//
// Every operator is applied to the same input several times; the max norm of the difference to the first
// apply is reported. With the fixed-point scatter it must be exactly zero (Replace and Add mode, with and
// without the additive seam communication); the atomic scatter is reported for reference. The fixed-point
// result is also compared against the atomic one (round-off), and both are timed.
//
// Usage: test_deterministic_scatter [min_level] [max_level] [repeats]

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/divergence_kerngen.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/gradient_kerngen.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg_kerngen.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "kernels/common/deterministic.hpp"
#include "linalg/vector_w1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::BoundaryConditions;
using grid::shell::DistributedDomain;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;

struct VectorFieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r, 0 ) = Kokkos::sin( 2.0 * c( 0 ) ) * Kokkos::cosh( c( 1 ) );
        data_( s, x, y, r, 1 ) = Kokkos::sin( 3.0 * c( 1 ) ) * Kokkos::cosh( c( 2 ) );
        data_( s, x, y, r, 2 ) = Kokkos::cos( 4.0 * c( 2 ) ) * Kokkos::cosh( c( 0 ) );
    }
};

struct ScalarFieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r ) = 1.0 + 0.3 * c( 0 ) * c( 1 ) + Kokkos::cos( 2.5 * c( 2 ) ) * Kokkos::sin( 1.5 * c( 0 ) );
    }
};

struct CoeffInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r ) = Kokkos::exp( 2.0 * Kokkos::sin( 3.0 * c( 0 ) ) * Kokkos::cos( 2.0 * c( 1 ) + c( 2 ) ) );
    }
};

int failures = 0;

/// One row: `repeats` Replace applies and `repeats` Add applies, max difference to
/// the first of each; the Replace result in double for the cross-comparison; time.
template < typename Op, typename Src, typename Dst >
struct Row
{
    double rep_diff = 0, add_diff = 0, ref_inf = 1, us = 0;
};

template < typename Op, typename Src, typename Dst >
Row< Op, Src, Dst > run_row( Op& op, const Src& src, Dst& y0, Dst& y, Dst& ya0, Dst& ya, Dst& diff, const int repeats,
                             const linalg::OperatorCommunicationMode comm )
{
    using SS = typename Dst::ScalarType;
    Row< Op, Src, Dst > row;
    // The mass operators and the grid transfers take their modes in the constructor
    // (Replace, additive communication): no Add arm for them.
    constexpr bool has_modes = requires { op.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Replace, comm ); };
    if constexpr ( has_modes )
        op.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Replace, comm );
    linalg::apply( op, src, y0 );
    Kokkos::fence();
    row.ref_inf = std::max( static_cast< double >( linalg::norm_inf( y0 ) ), 1e-300 );
    for ( int i = 1; i < repeats; ++i )
    {
        linalg::apply( op, src, y );
        Kokkos::fence();
        linalg::assign( diff, y );
        linalg::lincomb( diff, { SS( 1 ), SS( -1 ) }, { diff, y0 } );
        row.rep_diff = std::max( row.rep_diff, static_cast< double >( linalg::norm_inf( diff ) ) );
    }
    // Add mode onto the same prefilled vector every time.
    if constexpr ( has_modes )
    {
        op.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Add, comm );
        linalg::assign( ya0, y0 );
        linalg::apply( op, src, ya0 );
        Kokkos::fence();
        for ( int i = 1; i < repeats; ++i )
        {
            linalg::assign( ya, y0 );
            linalg::apply( op, src, ya );
            Kokkos::fence();
            linalg::assign( diff, ya );
            linalg::lincomb( diff, { SS( 1 ), SS( -1 ) }, { diff, ya0 } );
            row.add_diff = std::max( row.add_diff, static_cast< double >( linalg::norm_inf( diff ) ) );
        }
    }
    // Time the Replace apply.
    if constexpr ( has_modes )
        op.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Replace, comm );
    linalg::apply( op, src, y );
    Kokkos::fence();
    Kokkos::Timer timer;
    const int     n = 10;
    for ( int i = 0; i < n; ++i )
        linalg::apply( op, src, y );
    Kokkos::fence();
    row.us = timer.seconds() / n * 1e6;
    return row;
}

/// Both scatters of one operator: report and check.
template < typename Op, typename Src, typename Dst >
void check( const char* name, Op& op, const Src& src, Dst& y0, Dst& y, Dst& ya0, Dst& ya, Dst& diff, Dst& ydet,
            const int repeats, const double tol )
{
    using SS   = typename Dst::ScalarType;
    using Comm = linalg::OperatorCommunicationMode;
    // The deterministic scatter: the environment switch for the coloured per-cell
    // kernels, the operator's own switch where it has one (the warp-owner kernels).
    auto set_det = [&]( const bool on ) { det::set_requested( on ); };
    constexpr bool has_modes =
        requires { op.set_operator_apply_and_communication_modes( linalg::OperatorApplyMode::Replace, Comm::SkipCommunication ); };
    const std::pair< Comm, const char* > arms[] = { { Comm::CommunicateAdditively, "comm" }, { Comm::SkipCommunication, "nocomm" } };
    for ( int arm = 0; arm < ( has_modes ? 2 : 1 ); ++arm )
    {
        const auto& [comm, cname] = arms[arm];
        set_det( false );
        const auto a = run_row( op, src, y0, y, ya0, ya, diff, repeats, comm );
        linalg::assign( ydet, y0 ); // the atomic result
        set_det( true );
        const auto d = run_row( op, src, y0, y, ya0, ya, diff, repeats, comm );
        linalg::lincomb( ydet, { SS( 1 ), SS( -1 ) }, { ydet, y0 } );
        const double cross = static_cast< double >( linalg::norm_inf( ydet ) ) / d.ref_inf;

        const bool det_ok   = ( d.rep_diff == 0.0 ) && ( d.add_diff == 0.0 );
        const bool cross_ok = std::isfinite( cross ) && cross <= tol;
        const bool ok       = det_ok && cross_ok;
        std::cout << "    " << std::setw( 10 ) << name << " " << std::setw( 6 ) << cname << std::scientific << std::setprecision( 2 ) << "   atomic: rep " << a.rep_diff / a.ref_inf << " add "
                  << a.add_diff / a.ref_inf << std::fixed << std::setprecision( 1 ) << " " << std::setw( 8 ) << a.us << " us"
                  << std::scientific << std::setprecision( 2 ) << "   fixed-point: rep " << d.rep_diff / d.ref_inf << " add "
                  << d.add_diff / d.ref_inf << std::fixed << std::setprecision( 1 ) << " " << std::setw( 8 ) << d.us << " us"
                  << " (" << std::setprecision( 3 ) << d.us / a.us << "x)" << std::scientific << std::setprecision( 2 )
                  << "   fixed-point vs atomic " << cross << ( ok ? "" : "   <-- FAIL" ) << std::endl;
        if ( !ok )
            ++failures;
    }
}

template < typename S >
void run_level( const int level, const int repeats, BoundaryConditions bcs, const char* bcname )
{
    using Eps  = fe::wedge::operators::shell::EpsilonDivDivKerngen< S >;
    using Grad = fe::wedge::operators::shell::GradientKerngen< S >;
    using Div  = fe::wedge::operators::shell::DivergenceKerngen< S >;
    using Supg = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< S >;

    const auto dom_f = DistributedDomain::create_uniform( level, level, 1.22, 2.22, 0, 0 );
    const auto dom_c = DistributedDomain::create_uniform( level - 1, level - 1, 1.22, 2.22, 0, 0 );

    auto mask_f  = grid::setup_node_ownership_mask_data( dom_f );
    auto mask_c  = grid::setup_node_ownership_mask_data( dom_c );
    auto bmask_f = grid::shell::setup_boundary_mask_data( dom_f );

    const auto coords_d = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom_f );
    const auto radii_d  = grid::shell::subdomain_shell_radii< double >( dom_f );
    const auto coords_c = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( dom_c );
    const auto radii_c  = grid::shell::subdomain_shell_radii< double >( dom_c );
    const auto coords   = grid::shell::subdomain_unit_sphere_single_shell_coords< S >( dom_f );
    const auto radii    = grid::shell::subdomain_shell_radii< S >( dom_f );

    VectorW1Vec< double >    ud( "ud", dom_f, mask_f );
    VectorW1Scalar< double > pd( "pd", dom_c, mask_c );
    VectorW1Scalar< double > kd( "kd", dom_f, mask_f );
    VectorW1Scalar< double > Td( "Td", dom_f, mask_f );
    Kokkos::parallel_for( "u", grid::shell::local_domain_md_range_policy_nodes( dom_f ),
                          VectorFieldInterpolator{ coords_d, radii_d, ud.grid_data() } );
    Kokkos::parallel_for( "p", grid::shell::local_domain_md_range_policy_nodes( dom_c ),
                          ScalarFieldInterpolator{ coords_c, radii_c, pd.grid_data() } );
    Kokkos::parallel_for( "k", grid::shell::local_domain_md_range_policy_nodes( dom_f ),
                          CoeffInterpolator{ coords_d, radii_d, kd.grid_data() } );
    Kokkos::parallel_for( "T", grid::shell::local_domain_md_range_policy_nodes( dom_f ),
                          ScalarFieldInterpolator{ coords_d, radii_d, Td.grid_data() } );
    Kokkos::fence();

    VectorW1Vec< S >    u( "u", dom_f, mask_f );
    VectorW1Scalar< S > p( "p", dom_c, mask_c ), k( "k", dom_f, mask_f ), T( "T", dom_f, mask_f );
    linalg::convert( ud, u );
    linalg::convert( pd, p );
    linalg::convert( kd, k );
    linalg::convert( Td, T );

    const double tol = std::is_same_v< S, double > ? 1e-12 : 1e-5;
    std::cout << "level " << level << "  " << bcname << "  storage " << ( std::is_same_v< S, double > ? "double" : "float" )
              << "  (relative max-norm differences; rep = repeated Replace applies, add = repeated Add applies)" << std::endl;

    {
        Eps op( dom_f, coords, radii, bmask_f, k.grid_data(), bcs, false );
        VectorW1Vec< S > y0( "y0", dom_f, mask_f ), y( "y", dom_f, mask_f ), ya0( "ya0", dom_f, mask_f ),
            ya( "ya", dom_f, mask_f ), diff( "diff", dom_f, mask_f ), ydet( "ydet", dom_f, mask_f );
        check( "epsdivdiv", op, u, y0, y, ya0, ya, diff, ydet, repeats, tol );
    }
    {
        Eps op( dom_f, coords, radii, bmask_f, k.grid_data(), bcs, true );
        VectorW1Vec< S > y0( "y0", dom_f, mask_f ), y( "y", dom_f, mask_f ), ya0( "ya0", dom_f, mask_f ),
            ya( "ya", dom_f, mask_f ), diff( "diff", dom_f, mask_f ), ydet( "ydet", dom_f, mask_f );
        check( "eps diag", op, u, y0, y, ya0, ya, diff, ydet, repeats, tol );
    }
    {
        Grad op( dom_f, dom_c, coords, radii, bmask_f, bcs );
        VectorW1Vec< S > y0( "y0", dom_f, mask_f ), y( "y", dom_f, mask_f ), ya0( "ya0", dom_f, mask_f ),
            ya( "ya", dom_f, mask_f ), diff( "diff", dom_f, mask_f ), ydet( "ydet", dom_f, mask_f );
        check( "gradient", op, p, y0, y, ya0, ya, diff, ydet, repeats, tol );
    }
    {
        Div op( dom_f, dom_c, coords, radii, bmask_f, bcs );
        VectorW1Scalar< S > y0( "y0", dom_c, mask_c ), y( "y", dom_c, mask_c ), ya0( "ya0", dom_c, mask_c ),
            ya( "ya", dom_c, mask_c ), diff( "diff", dom_c, mask_c ), ydet( "ydet", dom_c, mask_c );
        check( "divergence", op, u, y0, y, ya0, ya, diff, ydet, repeats, tol );
    }
    {
        Supg op( dom_f, coords, radii, bmask_f, u, 1e-3, S( 1e-2 ), /*treat_boundary=*/true );
        VectorW1Scalar< S > y0( "y0", dom_f, mask_f ), y( "y", dom_f, mask_f ), ya0( "ya0", dom_f, mask_f ),
            ya( "ya", dom_f, mask_f ), diff( "diff", dom_f, mask_f ), ydet( "ydet", dom_f, mask_f );
        check( "supg", op, T, y0, y, ya0, ya, diff, ydet, repeats, tol );
    }
    // The per-cell kernels of the app: mass operators and the constant grid transfers.
    {
        fe::wedge::operators::shell::Mass< S > op( dom_f, coords, radii );
        VectorW1Scalar< S > y0( "y0", dom_f, mask_f ), y( "y", dom_f, mask_f ), ya0( "ya0", dom_f, mask_f ),
            ya( "ya", dom_f, mask_f ), diff( "diff", dom_f, mask_f ), ydet( "ydet", dom_f, mask_f );
        check( "mass", op, T, y0, y, ya0, ya, diff, ydet, repeats, tol );
    }
    {
        fe::wedge::operators::shell::KMass< S > op( dom_f, coords, radii, k.grid_data(), false );
        VectorW1Scalar< S > y0( "y0", dom_f, mask_f ), y( "y", dom_f, mask_f ), ya0( "ya0", dom_f, mask_f ),
            ya( "ya", dom_f, mask_f ), diff( "diff", dom_f, mask_f ), ydet( "ydet", dom_f, mask_f );
        check( "kmass", op, T, y0, y, ya0, ya, diff, ydet, repeats, tol );
    }
    {
        fe::wedge::operators::shell::VectorMass< S > op( dom_f, coords, radii, false );
        VectorW1Vec< S > y0( "y0", dom_f, mask_f ), y( "y", dom_f, mask_f ), ya0( "ya0", dom_f, mask_f ),
            ya( "ya", dom_f, mask_f ), diff( "diff", dom_f, mask_f ), ydet( "ydet", dom_f, mask_f );
        check( "vecmass", op, u, y0, y, ya0, ya, diff, ydet, repeats, tol );
    }
    VectorW1Vec< S > uc( "uc", dom_c, mask_c );
    {
        fe::wedge::operators::shell::RestrictionVecConstant< S > op( dom_c );
        VectorW1Vec< S > y0( "y0", dom_c, mask_c ), y( "y", dom_c, mask_c ), ya0( "ya0", dom_c, mask_c ),
            ya( "ya", dom_c, mask_c ), diff( "diff", dom_c, mask_c ), ydet( "ydet", dom_c, mask_c );
        check( "restrict", op, u, y0, y, ya0, ya, diff, ydet, repeats, tol );
        linalg::assign( uc, y0 );
    }
    {
        fe::wedge::operators::shell::ProlongationVecConstant< S > op;
        VectorW1Vec< S > y0( "y0", dom_f, mask_f ), y( "y", dom_f, mask_f ), ya0( "ya0", dom_f, mask_f ),
            ya( "ya", dom_f, mask_f ), diff( "diff", dom_f, mask_f ), ydet( "ydet", dom_f, mask_f );
        check( "prolong", op, uc, y0, y, ya0, ya, diff, ydet, repeats, tol );
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    {
        const int min_level = ( argc > 1 ) ? std::atoi( argv[1] ) : 4;
        const int max_level = ( argc > 2 ) ? std::atoi( argv[2] ) : 6;
        const int repeats   = ( argc > 3 ) ? std::atoi( argv[3] ) : 6;

        struct BC
        {
            const char*                          name;
            grid::shell::BoundaryConditionFlag cmb, surface;
        };
        const BC bc_sets[] = { { "D/D", DIRICHLET, DIRICHLET }, { "N/D", NEUMANN, DIRICHLET } };

        for ( int level = min_level; level <= max_level; ++level )
            for ( const auto& bc : bc_sets )
            {
                BoundaryConditions bcs = { { CMB, bc.cmb }, { SURFACE, bc.surface } };
                run_level< double >( level, repeats, bcs, bc.name );
            }
    }
    std::cout << ( failures == 0 ? "PASSED" : "FAILED" ) << " (" << failures << " row(s) not reproducible or off)" << std::endl;
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
