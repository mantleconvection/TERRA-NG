/// @file  test_supg_fast_vs_slow.cpp
/// @brief Correctness test comparing `UnsteadyAdvectionDiffusionSUPG` (legacy)
///        against `UnsteadyAdvectionDiffusionSUPGKerngen` (fused, team-based)
///        across all combinations of:
///          - boundary flavour (TreatBoundary on / off × CMB/SURFACE Dirichlet or nothing)
///          - Diagonal-only  (on / off)
///          - LumpedMass     (on / off)
///          - apply mode     (Replace / Add)
///          - communication  (CommunicateAdditively / SkipCommunication)
///          - kernel path    (forced slow / default fast)
///          - a small sweep of mesh levels and subdomain refinements.

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg_kerngen.hpp"
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
using grid::shell::DistributedDomain;
using linalg::OperatorApplyMode;
using linalg::OperatorCommunicationMode;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

struct ScalarTempInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    ScalarTempInterpolator( const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r,
                            const Grid4DDataScalar< double >& d ) : grid_( g ), radii_( r ), data_( d ) {}
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r ) = 1.0 + 0.5 * Kokkos::sin( 1.3 * c( 0 ) ) * Kokkos::cos( 0.9 * c( 1 ) )
                            + 0.3 * Kokkos::sin( 1.1 * c( 2 ) );
    }
};

struct VelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    VelocityInterpolator( const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r,
                          const Grid4DDataVec< double, 3 >& d ) : grid_( g ), radii_( r ), data_( d ) {}
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        // Choose a non-trivial, non-zero velocity field so SUPG stabilisation activates.
        data_( s, x, y, r, 0 ) = 0.4 * Kokkos::sin( 1.7 * c( 2 ) );
        data_( s, x, y, r, 1 ) = 0.3 * Kokkos::cos( 1.1 * c( 0 ) );
        data_( s, x, y, r, 2 ) = 0.5 * Kokkos::sin( 0.9 * c( 1 ) );
    }
};

struct ScalarSeedInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    ScalarSeedInterpolator( const Grid3DDataVec< double, 3 >& g, const Grid2DDataScalar< double >& r,
                            const Grid4DDataScalar< double >& d ) : grid_( g ), radii_( r ), data_( d ) {}
    KOKKOS_INLINE_FUNCTION
    void operator()( const int s, const int x, const int y, const int r ) const
    {
        const auto c = grid::shell::coords( s, x, y, r, grid_, radii_ );
        data_( s, x, y, r ) = 0.1 * Kokkos::cos( 0.7 * c( 0 ) + 0.5 * c( 1 ) );
    }
};

static double compare_once(
    const int                 level_fine,
    const int                 lat_sdr,
    const int                 rad_sdr,
    const bool                treat_boundary,
    const bool                diagonal,
    const bool                lumped_mass,
    const double              dt,
    const double              mass_scaling,
    const double              diffusivity,
    OperatorApplyMode         apply_mode,
    OperatorCommunicationMode comm_mode,
    bool                      force_slow_new,
    double                    tol )
{
    using ScalarType = double;
    using OpOld      = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;
    using OpNew      = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;

    const double r_min = 0.5, r_max = 1.0;
    auto domain = DistributedDomain::create_uniform( level_fine, level_fine, r_min, r_max, lat_sdr, rad_sdr );

    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto radii  = grid::shell::subdomain_shell_radii< ScalarType >( domain );
    auto mask          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask = grid::shell::setup_boundary_mask_data( domain );

    VectorQ1Scalar< ScalarType > src    ( "src",     domain, mask );
    VectorQ1Vec< ScalarType, 3 > vel    ( "vel",     domain, mask );
    VectorQ1Scalar< ScalarType > dst_old( "dst_old", domain, mask );
    VectorQ1Scalar< ScalarType > dst_new( "dst_new", domain, mask );
    VectorQ1Scalar< ScalarType > err    ( "err",     domain, mask );

    Kokkos::parallel_for( "interp_T",
        local_domain_md_range_policy_nodes( domain ),
        ScalarTempInterpolator( coords, radii, src.grid_data() ) );
    Kokkos::parallel_for( "interp_vel",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator( coords, radii, vel.grid_data() ) );
    Kokkos::parallel_for( "seed_old",
        local_domain_md_range_policy_nodes( domain ),
        ScalarSeedInterpolator( coords, radii, dst_old.grid_data() ) );
    Kokkos::parallel_for( "seed_new",
        local_domain_md_range_policy_nodes( domain ),
        ScalarSeedInterpolator( coords, radii, dst_new.grid_data() ) );
    Kokkos::fence();

    OpOld op_old( domain, coords, radii, boundary_mask, vel, diffusivity, dt,
                  treat_boundary, diagonal, mass_scaling, lumped_mass,
                  apply_mode, comm_mode );
    OpNew op_new( domain, coords, radii, boundary_mask, vel, diffusivity, dt,
                  treat_boundary, diagonal, mass_scaling, lumped_mass,
                  apply_mode, comm_mode );
    if ( force_slow_new )
        op_new.force_slow_path();

    linalg::apply( op_old, src, dst_old );
    linalg::apply( op_new, src, dst_new );
    Kokkos::fence();

    linalg::lincomb( err, { 1.0, -1.0 }, { dst_old, dst_new } );

    const auto num_dofs = std::max< long >(
        1, kernels::common::count_masked< long >( mask, grid::NodeOwnershipFlag::OWNED ) );
    const double l2_err  = std::sqrt( dot( err, err ) / num_dofs );
    const double inf_err = linalg::norm_inf( err );

    const double worst = std::max( l2_err, inf_err );
    const bool   pass  = worst <= tol;

    std::ostringstream tag;
    tag << "lvl=" << level_fine
        << " lat_sdr=" << lat_sdr
        << " rad_sdr=" << rad_sdr
        << " TB=" << treat_boundary
        << " diag=" << diagonal
        << " lump=" << lumped_mass
        << " dt=" << dt
        << " m=" << mass_scaling
        << " apply=" << ( apply_mode == OperatorApplyMode::Replace ? "Rep" : "Add" )
        << " comm=" << ( comm_mode == OperatorCommunicationMode::CommunicateAdditively ? "CA" : "Sk" )
        << " path=" << ( force_slow_new ? "slow" : op_new.path_name() );

    std::cout << "    " << ( pass ? "[ OK ]" : "[FAIL]" ) << "  " << tag.str()
              << "   L2=" << l2_err << "  Linf=" << inf_err << std::endl;

    return worst;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const std::vector< int > levels = { 2, 3, 4 };
    const std::vector< std::pair< int, int > > sdr_combos = {
        { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    const std::vector< bool > treat_boundary_opts = { false, true };
    const std::vector< bool > diagonal_opts       = { false, true };
    const std::vector< bool > lumped_mass_opts    = { false, true };
    const std::vector< double > dt_values         = { 0.01, 1.0 };
    const std::vector< double > mass_scalings     = { 1.0, 1.5 };  // BDF2 uses 1.5, BDF1 uses 1.0
    const double diffusivity = 1.0;
    const std::vector< OperatorApplyMode >         apply_modes = { OperatorApplyMode::Replace, OperatorApplyMode::Add };
    const std::vector< OperatorCommunicationMode > comm_modes  = {
        OperatorCommunicationMode::SkipCommunication, OperatorCommunicationMode::CommunicateAdditively };

    const double tol_slow = 1e-12;
    const double tol_fast = 1e-9;

    int    num_tests = 0;
    int    num_fail  = 0;
    double worst     = 0.0;

    for ( int level : levels )
    for ( auto [lat_sdr, rad_sdr] : sdr_combos )
    for ( bool tb    : treat_boundary_opts )
    for ( bool diag  : diagonal_opts )
    for ( bool lump  : lumped_mass_opts )
    for ( double dt  : dt_values )
    for ( double m   : mass_scalings )
    for ( auto am    : apply_modes )
    for ( auto cm    : comm_modes )
    {
        // 1) Slow path of new vs old — should match to ~machine epsilon.
        {
            double w = compare_once( level, lat_sdr, rad_sdr, tb, diag, lump, dt, m, diffusivity,
                                     am, cm, /*force_slow=*/true, tol_slow );
            worst = std::max( worst, w );
            ++num_tests;
            if ( w > tol_slow ) ++num_fail;
        }
        // 2) Fast path of new vs old — should match to fast-path FMA tolerance.
        {
            double w = compare_once( level, lat_sdr, rad_sdr, tb, diag, lump, dt, m, diffusivity,
                                     am, cm, /*force_slow=*/false, tol_fast );
            worst = std::max( worst, w );
            ++num_tests;
            if ( w > tol_fast ) ++num_fail;
        }
    }

    std::cout << "\n============================================================\n";
    std::cout << " SUPG Kerngen vs legacy:  " << ( num_tests - num_fail ) << "/" << num_tests
              << " passing (worst=" << worst << ")\n";
    std::cout << "============================================================" << std::endl;

    return ( num_fail == 0 ) ? 0 : 1;
}
