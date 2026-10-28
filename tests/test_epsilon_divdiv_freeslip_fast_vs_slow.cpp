#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "linalg/vector_w1.hpp"
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
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;

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
        const auto rr     = radii_( local_subdomain_id, r );
        (void)rr;

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
void compare_epsilon_divdiv_path_comparison( int level, bool diagonal, int repeats = 5 )
{
    using Op = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarT >;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarT >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarT >( domain );

    VectorW1Vec< ScalarT > src( "src", domain, mask_data );

    // Freeslip/Dirichlet outputs (fast + slow)
    VectorW1Vec< ScalarT > dst_fast_fs( "dst_fast_fs", domain, mask_data );
    VectorW1Vec< ScalarT > dst_slow_fs( "dst_slow_fs", domain, mask_data );

    // Dirichlet/Dirichlet outputs (fast + slow)
    VectorW1Vec< ScalarT > dst_dd_fast( "dst_dd_fast", domain, mask_data );
    VectorW1Vec< ScalarT > dst_dd_slow( "dst_dd_slow", domain, mask_data );

    // Error vectors
    VectorW1Vec< ScalarT > err_fast_vs_slow( "err_fast_vs_slow", domain, mask_data );
    VectorW1Vec< ScalarT > err_dd_fast_vs_slow( "err_dd_fast_vs_slow", domain, mask_data );
    VectorW1Vec< ScalarT > err_fast_vs_dd( "err_fast_vs_dd", domain, mask_data );
    VectorW1Vec< ScalarT > err_slow_vs_dd( "err_slow_vs_dd", domain, mask_data );

    VectorW1Scalar< ScalarT > k_coeff( "k", domain, mask_data );
    VectorW1Scalar< ScalarT > gca_elements( "gca_elements", domain, mask_data );

    Kokkos::parallel_for(
        "interpolate src",
        local_domain_md_range_policy_nodes( domain ),
        VectorFieldInterpolator( coords_shell, coords_radii, src.grid_data() ) );

    Kokkos::parallel_for(
        "interpolate k",
        local_domain_md_range_policy_nodes( domain ),
        ScalarCoeffInterpolator( coords_shell, coords_radii, k_coeff.grid_data() ) );

    assign( gca_elements, 0 );
    Kokkos::fence();

    BoundaryConditions bcs_freeslip_dirichlet = {
        { CMB, FREESLIP },
        { SURFACE, DIRICHLET }
    };

    BoundaryConditions bcs_dirichlet_dirichlet = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET }
    };

    // 1) Fast freeslip operator (expected fast path)
    Op op_fast_fs(
        domain,
        coords_shell,
        coords_radii,
        boundary_mask_data,
        k_coeff.grid_data(),
        bcs_freeslip_dirichlet,
        diagonal,
        linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode::Off );

    // 2) Slow freeslip operator (force slow path via stored-matrix mode)
    Op op_slow_fs(
        domain,
        coords_shell,
        coords_radii,
        boundary_mask_data,
        k_coeff.grid_data(),
        bcs_freeslip_dirichlet,
        diagonal,
        linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode::Off );

    op_slow_fs.set_stored_matrix_mode(
        linalg::OperatorStoredMatrixMode::Selective, /*level_range=*/0, gca_elements.grid_data() );

    // 3) Fast Dirichlet-Dirichlet operator
    Op op_dd_fast(
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

    // 4) Slow Dirichlet-Dirichlet operator (force slow path via stored-matrix mode)
    Op op_dd_slow(
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

    op_dd_slow.set_stored_matrix_mode(
        linalg::OperatorStoredMatrixMode::Selective, /*level_range=*/0, gca_elements.grid_data() );

    // Warmup
    linalg::apply( op_fast_fs, src, dst_fast_fs );
    linalg::apply( op_slow_fs, src, dst_slow_fs );
    linalg::apply( op_dd_fast, src, dst_dd_fast );
    linalg::apply( op_dd_slow, src, dst_dd_slow );
    Kokkos::fence();

    // Timings
    Kokkos::Timer timer_fast_fs;
    for ( int i = 0; i < repeats; ++i )
    {
        linalg::apply( op_fast_fs, src, dst_fast_fs );
    }
    Kokkos::fence();
    const double t_fast_fs = timer_fast_fs.seconds();

    Kokkos::Timer timer_slow_fs;
    for ( int i = 0; i < repeats; ++i )
    {
        linalg::apply( op_slow_fs, src, dst_slow_fs );
    }
    Kokkos::fence();
    const double t_slow_fs = timer_slow_fs.seconds();

    Kokkos::Timer timer_dd_fast;
    for ( int i = 0; i < repeats; ++i )
    {
        linalg::apply( op_dd_fast, src, dst_dd_fast );
    }
    Kokkos::fence();
    const double t_dd_fast = timer_dd_fast.seconds();

    Kokkos::Timer timer_dd_slow;
    for ( int i = 0; i < repeats; ++i )
    {
        linalg::apply( op_dd_slow, src, dst_dd_slow );
    }
    Kokkos::fence();
    const double t_dd_slow = timer_dd_slow.seconds();

    // Comparisons
    linalg::lincomb( err_fast_vs_slow, { 1.0, -1.0 }, { dst_fast_fs, dst_slow_fs } );
    linalg::lincomb( err_dd_fast_vs_slow, { 1.0, -1.0 }, { dst_dd_fast, dst_dd_slow } );

    // Cross-BC comparisons (generally nonzero by design)
    linalg::lincomb( err_fast_vs_dd, { 1.0, -1.0 }, { dst_fast_fs, dst_dd_fast } );
    linalg::lincomb( err_slow_vs_dd, { 1.0, -1.0 }, { dst_slow_fs, dst_dd_fast } );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );

    const auto l2_fast_vs_slow      = std::sqrt( dot( err_fast_vs_slow, err_fast_vs_slow ) / num_dofs );
    const auto inf_fast_vs_slow     = linalg::norm_inf( err_fast_vs_slow );

    const auto l2_dd_fast_vs_slow   = std::sqrt( dot( err_dd_fast_vs_slow, err_dd_fast_vs_slow ) / num_dofs );
    const auto inf_dd_fast_vs_slow  = linalg::norm_inf( err_dd_fast_vs_slow );

    const auto l2_fast_vs_dd        = std::sqrt( dot( err_fast_vs_dd, err_fast_vs_dd ) / num_dofs );
    const auto inf_fast_vs_dd       = linalg::norm_inf( err_fast_vs_dd );

    const auto l2_slow_vs_dd        = std::sqrt( dot( err_slow_vs_dd, err_slow_vs_dd ) / num_dofs );
    const auto inf_slow_vs_dd       = linalg::norm_inf( err_slow_vs_dd );

    std::cout << "  repeats           = " << repeats << std::endl;

    std::cout << "  fast(freeslip)    = " << t_fast_fs << " s  (" << ( t_fast_fs / repeats ) << " s/apply)" << std::endl;
    std::cout << "  slow(freeslip)    = " << t_slow_fs << " s  (" << ( t_slow_fs / repeats ) << " s/apply)" << std::endl;
    std::cout << "  fast(dirichlet)   = " << t_dd_fast << " s  (" << ( t_dd_fast / repeats ) << " s/apply)" << std::endl;
    std::cout << "  slow(dirichlet)   = " << t_dd_slow << " s  (" << ( t_dd_slow / repeats ) << " s/apply)" << std::endl;

    if ( t_fast_fs > 0.0 )
    {
        std::cout << "  slow/fast(fs)     = " << ( t_slow_fs / t_fast_fs ) << "x" << std::endl;
        std::cout << "  dd_fast/fast(fs)  = " << ( t_dd_fast / t_fast_fs ) << "x" << std::endl;
    }
    if ( t_dd_fast > 0.0 )
    {
        std::cout << "  slow/fast(dd)     = " << ( t_dd_slow / t_dd_fast ) << "x" << std::endl;
    }

    // Main correctness checks (same BCs, different execution paths)
    std::cout << "  [fast_fs vs slow_fs] L2 = " << l2_fast_vs_slow
              << ", inf = " << inf_fast_vs_slow << std::endl;
    std::cout << "  [fast_dd vs slow_dd] L2 = " << l2_dd_fast_vs_slow
              << ", inf = " << inf_dd_fast_vs_slow << std::endl;

    // Cross-BC comparisons (informational)
    std::cout << "  [fast_fs vs dd_fast] L2 = " << l2_fast_vs_dd
              << ", inf = " << inf_fast_vs_dd << std::endl;
    std::cout << "  [slow_fs vs dd_fast] L2 = " << l2_slow_vs_dd
              << ", inf = " << inf_slow_vs_dd << std::endl;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    constexpr int repeats = 1;

    for ( auto diagonal : { true, false } )
    {
        std::cout << "==================================================" << std::endl;
        std::cout << "EpsilonDivDivKerngen path comparison" << std::endl;
        std::cout << "BC set A: CMB=FREESLIP, SURFACE=DIRICHLET (fast vs slow)" << std::endl;
        std::cout << "BC set B: CMB=DIRICHLET, SURFACE=DIRICHLET (fast vs slow)" << std::endl;
        std::cout << "diagonal = " << diagonal << std::endl;

        for ( int level = 0; level < 6; ++level )
        {
            std::cout << "level = " << level << std::endl;
            compare_epsilon_divdiv_path_comparison< double >( level, diagonal, repeats );
        }
    }

    return 0;
}