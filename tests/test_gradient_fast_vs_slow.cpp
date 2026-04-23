/// @file  test_gradient_fast_vs_slow.cpp
/// @brief Correctness test comparing the original `Gradient` operator against the
///        team-based fused-arithmetic `GradientKerngen` operator across all
///        combinations of BCs × apply mode × communication mode × kernel paths,
///        over a range of mesh levels and subdomain refinements.
///
/// Mirror of test_divergence_fast_vs_slow.cpp. Gradient is the transpose of
/// Divergence: src is scalar (coarse pressure), dst is vec3 (fine velocity).

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/operators/shell/gradient.hpp"
#include "fe/wedge/operators/shell/gradient_kerngen.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <sstream>
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
using grid::shell::ShellBoundaryFlag;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::OperatorApplyMode;
using linalg::OperatorCommunicationMode;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

// Scalar pressure input.
struct ScalarPressureInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    ScalarPressureInterpolator(
        const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r, const Grid4DDataScalar< double >& d )
    : grid_( g ), radii_( r ), data_( d )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r ) = Kokkos::sin( 1.3 * c( 0 ) ) * Kokkos::cos( 0.8 * c( 1 ) ) + 0.3 * Kokkos::sin( 1.7 * c( 2 ) );
    }
};

// Vector velocity seed (for Add mode comparison).
struct VectorSeedInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    VectorSeedInterpolator(
        const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r, const Grid4DDataVec< double, 3 >& d )
    : grid_( g ), radii_( r ), data_( d )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r, 0 ) = 0.11 * Kokkos::sin( 0.7 * c( 0 ) );
        data_( s, x, y, r, 1 ) = 0.13 * Kokkos::cos( 0.9 * c( 1 ) );
        data_( s, x, y, r, 2 ) = 0.17 * Kokkos::sin( 1.1 * c( 2 ) );
    }
};

static std::string bc_name( BoundaryConditionFlag f )
{
    if ( f == DIRICHLET ) return "DIR";
    if ( f == FREESLIP )  return "FS";
    if ( f == NEUMANN )   return "NEU";
    return "?";
}
static std::string apply_mode_name( OperatorApplyMode m )
{
    return m == OperatorApplyMode::Replace ? "Replace" : "Add";
}
static std::string comm_mode_name( OperatorCommunicationMode m )
{
    return m == OperatorCommunicationMode::CommunicateAdditively ? "CommAdd" : "SkipComm";
}

static double compare_once(
    const int                 level_fine,
    const int                 lat_sdr,
    const int                 rad_sdr,
    BoundaryConditions        bcs,
    OperatorApplyMode         apply_mode,
    OperatorCommunicationMode comm_mode,
    bool                      force_slow_new,
    double                    tol )
{
    using ScalarType = double;
    using OpOld      = fe::wedge::operators::shell::Gradient< ScalarType >;
    using OpNew      = fe::wedge::operators::shell::GradientKerngen< ScalarType >;

    const int    level_coarse = level_fine - 1;
    const double r_min = 0.5, r_max = 1.0;

    auto domain_fine   = DistributedDomain::create_uniform( level_fine,   level_fine,   r_min, r_max, lat_sdr, rad_sdr );
    auto domain_coarse = DistributedDomain::create_uniform( level_coarse, level_coarse, r_min, r_max, lat_sdr, rad_sdr );

    const auto coords_fine  = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_fine );
    const auto radii_fine   = grid::shell::subdomain_shell_radii< ScalarType >( domain_fine );
    const auto coords_coarse= grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_coarse );
    const auto radii_coarse = grid::shell::subdomain_shell_radii< ScalarType >( domain_coarse );

    auto mask_fine          = grid::setup_node_ownership_mask_data( domain_fine );
    auto mask_coarse        = grid::setup_node_ownership_mask_data( domain_coarse );
    auto boundary_mask_fine = grid::shell::setup_boundary_mask_data( domain_fine );

    VectorQ1Scalar< ScalarType >    src    ( "src",     domain_coarse, mask_coarse );
    VectorQ1Vec< ScalarType, 3 >    dst_old( "dst_old", domain_fine,   mask_fine   );
    VectorQ1Vec< ScalarType, 3 >    dst_new( "dst_new", domain_fine,   mask_fine   );
    VectorQ1Vec< ScalarType, 3 >    err    ( "err",     domain_fine,   mask_fine   );

    Kokkos::parallel_for(
        "interp_src_p",
        local_domain_md_range_policy_nodes( domain_coarse ),
        ScalarPressureInterpolator( coords_coarse, radii_coarse, src.grid_data() ) );

    Kokkos::parallel_for(
        "seed_old",
        local_domain_md_range_policy_nodes( domain_fine ),
        VectorSeedInterpolator( coords_fine, radii_fine, dst_old.grid_data() ) );
    Kokkos::parallel_for(
        "seed_new",
        local_domain_md_range_policy_nodes( domain_fine ),
        VectorSeedInterpolator( coords_fine, radii_fine, dst_new.grid_data() ) );
    Kokkos::fence();

    OpOld op_old( domain_fine, domain_coarse, coords_fine, radii_fine, boundary_mask_fine, bcs, apply_mode, comm_mode );
    OpNew op_new( domain_fine, domain_coarse, coords_fine, radii_fine, boundary_mask_fine, bcs, apply_mode, comm_mode );
    if ( force_slow_new )
        op_new.force_slow_path();

    linalg::apply( op_old, src, dst_old );
    linalg::apply( op_new, src, dst_new );
    Kokkos::fence();

    linalg::lincomb( err, { 1.0, -1.0 }, { dst_old, dst_new } );

    const auto num_dofs = std::max< long >(
        1, kernels::common::count_masked< long >( mask_fine, grid::NodeOwnershipFlag::OWNED ) );
    const double l2_err  = std::sqrt( dot( err, err ) / num_dofs );
    const double inf_err = linalg::norm_inf( err );

    const double worst = std::max( l2_err, inf_err );
    const bool   pass  = worst <= tol;

    std::ostringstream tag;
    tag << "lvl=" << level_fine
        << " lat_sdr=" << lat_sdr
        << " rad_sdr=" << rad_sdr
        << " cmb=" << bc_name( terra::grid::shell::get_boundary_condition_flag( bcs, CMB ) )
        << " sur=" << bc_name( terra::grid::shell::get_boundary_condition_flag( bcs, SURFACE ) )
        << " apply=" << apply_mode_name( apply_mode )
        << " comm="  << comm_mode_name( comm_mode )
        << " path=" << ( force_slow_new ? "slow" : op_new.path_name() );

    std::cout << "    " << ( pass ? "[ OK ]" : "[FAIL]" ) << "  " << tag.str()
              << "   L2=" << l2_err << "  Linf=" << inf_err << std::endl;

    return worst;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const std::vector< int > levels_fine = { 2, 3, 4 };
    const std::vector< std::pair< int, int > > sdr_combos = {
        { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    const std::vector< std::pair< BoundaryConditionFlag, BoundaryConditionFlag > > bc_sets = {
        { DIRICHLET, DIRICHLET },
        { FREESLIP,  DIRICHLET },
        { DIRICHLET, FREESLIP  },
        { FREESLIP,  FREESLIP  },
        { NEUMANN,   DIRICHLET },
        { DIRICHLET, NEUMANN   },
        { NEUMANN,   NEUMANN   },
    };
    const std::vector< OperatorApplyMode >         apply_modes = { OperatorApplyMode::Replace, OperatorApplyMode::Add };
    const std::vector< OperatorCommunicationMode > comm_modes  = {
        OperatorCommunicationMode::SkipCommunication, OperatorCommunicationMode::CommunicateAdditively
    };

    const double tol_slow = 1e-12;
    const double tol_fast = 1e-10;

    int    num_tests = 0;
    int    num_fail  = 0;
    double worst     = 0.0;

    for ( int level : levels_fine )
    {
        for ( auto [lat_sdr, rad_sdr] : sdr_combos )
        {
            for ( auto bc : bc_sets )
            {
                BoundaryConditions bcs = { { CMB, bc.first }, { SURFACE, bc.second } };

                for ( auto am : apply_modes )
                {
                    for ( auto cm : comm_modes )
                    {
                        {
                            double w = compare_once( level, lat_sdr, rad_sdr, bcs, am, cm,
                                                     /*force_slow_new=*/true, tol_slow );
                            worst = std::max( worst, w );
                            ++num_tests;
                            if ( w > tol_slow ) ++num_fail;
                        }
                        {
                            double w = compare_once( level, lat_sdr, rad_sdr, bcs, am, cm,
                                                     /*force_slow_new=*/false, tol_fast );
                            worst = std::max( worst, w );
                            ++num_tests;
                            if ( w > tol_fast ) ++num_fail;
                        }
                    }
                }
            }
        }
    }

    std::cout << "\n============================================================\n";
    std::cout << " GradientKerngen vs Gradient:  " << ( num_tests - num_fail ) << "/" << num_tests
              << " passing (worst=" << worst << ")\n";
    std::cout << "============================================================" << std::endl;

    return ( num_fail == 0 ) ? 0 : 1;
}
