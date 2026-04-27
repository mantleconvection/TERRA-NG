/// @file  test_divergence_fast_vs_slow.cpp
/// @brief Correctness test comparing the original `Divergence` operator against the
///        team-based `DivergenceKerngen` operator across all combinations of
///          - boundary condition pairs (Dirichlet / Freeslip / Neumann on CMB and SURFACE),
///          - `OperatorApplyMode` (Replace, Add),
///          - `OperatorCommunicationMode` (SkipCommunication, CommunicateAdditively),
///          - kernel paths of the new operator (Slow + whatever fast path host dispatch picks),
///          - mesh refinement levels,
///          - lateral subdomain refinements (to exercise multiple subdomains per rank).
///
/// For each configuration, the test applies both operators to the same velocity source
/// and compares the resulting pressure outputs via L2 and L-infinity norms. Because the
/// new operator's slow path reproduces the original element-matrix math, it is expected
/// to match bit-identically (modulo FMA reassociation). The fast paths may differ only
/// by floating-point rounding; we check against a loose tolerance scaled by the number
/// of DoFs.

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/operators/shell/divergence.hpp"
#include "fe/wedge/operators/shell/divergence_kerngen.hpp"
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

// ----- Velocity source functor --------------------------------------------------
struct VectorFieldInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    VectorFieldInterpolator(
        const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r, const Grid4DDataVec< double, 3 >& d )
    : grid_( g ), radii_( r ), data_( d )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r, 0 ) = Kokkos::sin( 1.7 * c( 0 ) ) * Kokkos::cos( 0.9 * c( 1 ) );
        data_( s, x, y, r, 1 ) = Kokkos::cos( 2.3 * c( 1 ) ) * Kokkos::sin( 0.7 * c( 2 ) );
        data_( s, x, y, r, 2 ) = Kokkos::sin( 1.1 * c( 2 ) ) * Kokkos::cos( 1.3 * c( 0 ) );
    }
};

// ----- Pre-seed dst to a non-zero value so Add mode actually carries initial state --
struct ScalarSeedInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;

    ScalarSeedInterpolator(
        const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r, const Grid4DDataScalar< double >& d )
    : grid_( g ), radii_( r ), data_( d )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r ) = 0.25 * ( 1.0 + Kokkos::sin( 0.6 * c( 0 ) + 1.1 * c( 1 ) ) );
    }
};

// ----- Helpers to print BC/mode names --------------------------------------------
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

// ----- One comparison: old Divergence vs new DivergenceKerngen --------------------
//
// Returns max of (L2, Linf) normalised error across the comparison.
// `force_slow_new` selects whether the new operator should be run on the slow path.
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
    using OpOld      = fe::wedge::operators::shell::Divergence< ScalarType >;
    using OpNew      = fe::wedge::operators::shell::DivergenceKerngen< ScalarType >;

    const int  level_coarse = level_fine - 1;
    const double r_min = 0.5, r_max = 1.0;

    // Build both the fine (velocity) and coarse (pressure) domains at matching (lat_sdr, rad_sdr).
    auto domain_fine   = DistributedDomain::create_uniform( level_fine,   level_fine,   r_min, r_max, lat_sdr, rad_sdr );
    auto domain_coarse = DistributedDomain::create_uniform( level_coarse, level_coarse, r_min, r_max, lat_sdr, rad_sdr );

    const auto coords_fine   = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_fine );
    const auto radii_fine    = grid::shell::subdomain_shell_radii< ScalarType >( domain_fine );

    auto mask_fine           = grid::setup_node_ownership_mask_data( domain_fine );
    auto mask_coarse         = grid::setup_node_ownership_mask_data( domain_coarse );
    auto boundary_mask_fine  = grid::shell::setup_boundary_mask_data( domain_fine );

    VectorQ1Vec< ScalarType, 3 > src    ( "src",     domain_fine,   mask_fine   );
    VectorQ1Scalar< ScalarType > dst_old( "dst_old", domain_coarse, mask_coarse );
    VectorQ1Scalar< ScalarType > dst_new( "dst_new", domain_coarse, mask_coarse );
    VectorQ1Scalar< ScalarType > err    ( "err",     domain_coarse, mask_coarse );

    Kokkos::parallel_for(
        "interp_src",
        local_domain_md_range_policy_nodes( domain_fine ),
        VectorFieldInterpolator( coords_fine, radii_fine, src.grid_data() ) );

    // Seed dst (non-zero) so Replace vs Add differs.
    {
        const auto coords_coarse = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain_coarse );
        const auto radii_coarse  = grid::shell::subdomain_shell_radii< ScalarType >( domain_coarse );
        Kokkos::parallel_for(
            "seed_old",
            local_domain_md_range_policy_nodes( domain_coarse ),
            ScalarSeedInterpolator( coords_coarse, radii_coarse, dst_old.grid_data() ) );
        Kokkos::parallel_for(
            "seed_new",
            local_domain_md_range_policy_nodes( domain_coarse ),
            ScalarSeedInterpolator( coords_coarse, radii_coarse, dst_new.grid_data() ) );
    }
    Kokkos::fence();

    OpOld op_old( domain_fine, domain_coarse, coords_fine, radii_fine, boundary_mask_fine, bcs, apply_mode, comm_mode );
    OpNew op_new( domain_fine, domain_coarse, coords_fine, radii_fine, boundary_mask_fine, bcs, apply_mode, comm_mode );
    if ( force_slow_new )
        op_new.force_slow_path();

    linalg::apply( op_old, src, dst_old );
    linalg::apply( op_new, src, dst_new );
    Kokkos::fence();

    // diff = dst_old - dst_new
    linalg::lincomb( err, { 1.0, -1.0 }, { dst_old, dst_new } );

    const auto num_dofs = std::max< long >(
        1, kernels::common::count_masked< long >( mask_coarse, grid::NodeOwnershipFlag::OWNED ) );
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

    // Configuration sweep ---------------------------------------------------------
    const std::vector< int > levels_fine = { 2, 3, 4 }; // coarse = level_fine - 1, so min level_fine = 2
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

    // Tolerances. Slow path should match the reference to FMA-reorder noise.
    // Fast path may differ by a few units-in-last-place × num_dofs.
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
                        // 1) Slow path of the new operator vs old (should be ~identical).
                        {
                            double w = compare_once( level, lat_sdr, rad_sdr, bcs, am, cm,
                                                     /*force_slow_new=*/true, tol_slow );
                            worst = std::max( worst, w );
                            ++num_tests;
                            if ( w > tol_slow ) ++num_fail;
                        }
                        // 2) Default (fast) path vs old.
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
    std::cout << " DivergenceKerngen vs Divergence:  " << ( num_tests - num_fail ) << "/" << num_tests
              << " passing (worst=" << worst << ")\n";
    std::cout << "============================================================" << std::endl;

    return ( num_fail == 0 ) ? 0 : 1;
}
