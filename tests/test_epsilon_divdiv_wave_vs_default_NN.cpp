// NeumannNeumann variant of test_epsilon_divdiv_wave_vs_default.
//
// Isolates whether the wave-vs-default matvec mismatch is in interior cells
// (no boundary special-casing) or boundary cells (Dirichlet-specific branch).
// With NN BCs the kernel still goes through the FastDirichletNeumann path but
// treat_boundary_dirichlet is false everywhere, so the bdry-Dirichlet branch
// doesn't fire and matvec range is always [0, 6) — same handling for all cells.

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
void compare_wave_vs_default_NN( int level, bool diagonal, int repeats = 1 )
{
    using Op = fe::wedge::operators::shell::EpsilonDivDivKerngen< ScalarT >;

    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarT >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarT >( domain );

    VectorQ1Vec< ScalarT > src( "src", domain, mask_data );
    VectorQ1Vec< ScalarT > dst_default( "dst_default", domain, mask_data );
    VectorQ1Vec< ScalarT > dst_wave( "dst_wave", domain, mask_data );
    VectorQ1Vec< ScalarT > err_wave_vs_default( "err_wave_vs_default", domain, mask_data );
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

    // Neumann/Neumann BCs: same FastDirichletNeumann path, but treat_boundary_dirichlet
    // is false at every cell so the special bdry-Dirichlet branch never fires.
    BoundaryConditions bcs_neumann_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN }
    };

    Op op_default(
        domain, coords_shell, coords_radii,
        boundary_mask_data, k_coeff.grid_data(),
        bcs_neumann_neumann, diagonal,
        linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode::Off );

    Op op_wave(
        domain, coords_shell, coords_radii,
        boundary_mask_data, k_coeff.grid_data(),
        bcs_neumann_neumann, diagonal,
        linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode::Off );

    op_wave.set_kernel_path( Op::KernelPath::FastDirichletNeumannWave );

    linalg::apply( op_default, src, dst_default );
    linalg::apply( op_wave,    src, dst_wave );
    Kokkos::fence();

    linalg::lincomb( err_wave_vs_default, { 1.0, -1.0 }, { dst_wave, dst_default } );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    const auto l2_default = std::sqrt( dot( dst_default, dst_default ) / num_dofs );
    const auto l2_wave    = std::sqrt( dot( dst_wave, dst_wave ) / num_dofs );
    const auto l2_err     = std::sqrt( dot( err_wave_vs_default, err_wave_vs_default ) / num_dofs );
    const auto inf_err    = linalg::norm_inf( err_wave_vs_default );

    std::cout << "  ||dst_default||_2 = " << l2_default << std::endl;
    std::cout << "  ||dst_wave||_2    = " << l2_wave << std::endl;
    std::cout << "  [wave vs default] L2 = " << l2_err << ", inf = " << inf_err << std::endl;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    for ( auto diagonal : { true, false } )
    {
        std::cout << "==================================================" << std::endl;
        std::cout << "EpsilonDivDivKerngen wave vs default (NeumannNeumann)" << std::endl;
        std::cout << "diagonal = " << diagonal << std::endl;
        for ( int level = 0; level < 6; ++level )
        {
            std::cout << "level = " << level << std::endl;
            compare_wave_vs_default_NN< double >( level, diagonal, 1 );
        }
    }
    return 0;
}
