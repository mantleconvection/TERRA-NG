#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

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
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

struct VectorFieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    VectorFieldInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        data_( local_subdomain_id, x, y, r, 0 ) =
            Kokkos::sin( 2.0 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );
        data_( local_subdomain_id, x, y, r, 1 ) =
            Kokkos::sin( 3.0 * coords( 1 ) ) * Kokkos::sinh( coords( 1 ) );
        data_( local_subdomain_id, x, y, r, 2 ) =
            Kokkos::sin( 4.0 * coords( 2 ) ) * Kokkos::sinh( coords( 1 ) );
    }
};

struct ScalarCoeffInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    ScalarCoeffInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const auto rr     = radii_( local_subdomain_id, r );

        data_( local_subdomain_id, x, y, r ) =
            1.0 + 0.1 * ( rr - 0.75 ) +
            0.05 * Kokkos::cos( coords( 0 ) ) * Kokkos::cosh( 0.25 * coords( 1 ) );
    }
};

template < typename ScalarT >
void compare_hex_vs_default( int level, bool diagonal, int repeats = 5 )
{
    using Op = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarT >;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarT >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarT >( domain );

    VectorQ1Vec< ScalarT > src( "src", domain, mask_data );

    VectorQ1Vec< ScalarT > dst_default( "dst_default", domain, mask_data );
    VectorQ1Vec< ScalarT > dst_hex( "dst_hex", domain, mask_data );
    VectorQ1Vec< ScalarT > err_hex_vs_default( "err_hex_vs_default", domain, mask_data );

    VectorQ1Scalar< ScalarT > k_coeff( "k", domain, mask_data );

    Kokkos::parallel_for(
        "interpolate src",
        local_domain_md_range_policy_nodes( domain ),
        VectorFieldInterpolator( coords_shell, coords_radii, src.grid_data() ) );

    Kokkos::parallel_for(
        "interpolate k",
        local_domain_md_range_policy_nodes( domain ),
        ScalarCoeffInterpolator( coords_shell, coords_radii, k_coeff.grid_data() ) );

    Kokkos::fence();

    // Dirichlet/Dirichlet BCs select the FastDirichletNeumann path; the hex path
    // is an opt-in alternative under the same BC family but uses a different
    // quadrature (1-pt Gauss on the hex instead of 1x1 Felippa on 2 wedges).
    BoundaryConditions bcs_dirichlet_dirichlet = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET }
    };

    // Default DN operator (uses the thread-per-cell FastDirichletNeumann path).
    Op op_default(
        domain,
        coords_shell,
        coords_radii,
        boundary_mask_data,
        k_coeff.grid_data(),
        bcs_dirichlet_dirichlet,
        diagonal,
        linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode::Off );

    // Hex-path operator: same construction, then force the hex 1-pt Gauss path.
    Op op_hex(
        domain,
        coords_shell,
        coords_radii,
        boundary_mask_data,
        k_coeff.grid_data(),
        bcs_dirichlet_dirichlet,
        diagonal,
        linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode::Off );

    op_hex.set_kernel_path( Op::KernelPath::FastDirichletNeumannHex );

    // Warmup
    linalg::apply( op_default, src, dst_default );
    linalg::apply( op_hex,     src, dst_hex );
    Kokkos::fence();

    // Timings
    Kokkos::Timer timer_default;
    for ( int i = 0; i < repeats; ++i )
    {
        linalg::apply( op_default, src, dst_default );
    }
    Kokkos::fence();
    const double t_default = timer_default.seconds();

    Kokkos::Timer timer_hex;
    for ( int i = 0; i < repeats; ++i )
    {
        linalg::apply( op_hex, src, dst_hex );
    }
    Kokkos::fence();
    const double t_hex = timer_hex.seconds();

    // Element-wise difference between hex and default outputs.
    // NOTE: hex is a DIFFERENT operator (1-pt Gauss on hex vs Felippa-1x1 on
    // two wedges), so this is NOT expected to be bit-equal. Reported here to
    // quantify how far apart the two discretizations sit on a smooth input.
    linalg::lincomb( err_hex_vs_default, { 1.0, -1.0 }, { dst_hex, dst_default } );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );

    const auto l2_default      = std::sqrt( dot( dst_default, dst_default ) / num_dofs );
    const auto l2_hex          = std::sqrt( dot( dst_hex, dst_hex ) / num_dofs );
    const auto l2_err          = std::sqrt( dot( err_hex_vs_default, err_hex_vs_default ) / num_dofs );
    const auto inf_err         = linalg::norm_inf( err_hex_vs_default );
    const auto rel_l2          = ( l2_default > 0.0 ) ? ( l2_err / l2_default ) : 0.0;

    std::cout << "  repeats           = " << repeats << std::endl;
    std::cout << "  default           = " << t_default << " s  (" << ( t_default / repeats ) << " s/apply)" << std::endl;
    std::cout << "  hex               = " << t_hex     << " s  (" << ( t_hex     / repeats ) << " s/apply)" << std::endl;
    if ( t_hex > 0.0 )
    {
        std::cout << "  default/hex       = " << ( t_default / t_hex ) << "x" << std::endl;
    }
    std::cout << "  ||dst_default||_2 = " << l2_default << std::endl;
    std::cout << "  ||dst_hex||_2     = " << l2_hex << std::endl;
    std::cout << "  [hex vs default] L2 = " << l2_err << ", inf = " << inf_err
              << ",  rel L2 = " << rel_l2 << std::endl;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    constexpr int repeats = 1;

    for ( auto diagonal : { true, false } )
    {
        std::cout << "==================================================" << std::endl;
        std::cout << "EpsilonDivDivKerngen hex vs default path comparison" << std::endl;
        std::cout << "BCs: CMB=DIRICHLET, SURFACE=DIRICHLET" << std::endl;
        std::cout << "diagonal = " << diagonal << std::endl;

        for ( int level = 0; level < 6; ++level )
        {
            std::cout << "level = " << level << std::endl;
            compare_hex_vs_default< double >( level, diagonal, repeats );
        }
    }

    return 0;
}
