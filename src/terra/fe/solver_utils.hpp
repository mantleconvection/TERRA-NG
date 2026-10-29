#pragma once

#include <utility>

#include "grid/grid_types.hpp"
#include "grid/shell/bit_masks.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_w1.hpp"
#include "linalg/vector_w1_iso_w2_w1.hpp"
#include "util/bit_masking.hpp"
#include "util/logging.hpp"

namespace terra::fe {

/// dst = K · [0; 1], i.e. the Stokes operator applied to zero velocity and constant pressure.
/// The velocity block is Bᵀ·1 (with K's boundary treatment), the pressure block B·0.
template < typename ScalarType, typename StokesOperator >
void apply_to_constant_pressure(
    StokesOperator&                                           K,
    const grid::shell::DistributedDomain&                     domain_u,
    const grid::shell::DistributedDomain&                     domain_p,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&  mask_u,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&  mask_p,
    linalg::VectorW1IsoW2W1< ScalarType >&                    dst )
{
    linalg::VectorW1IsoW2W1< ScalarType > src( "constant_pressure_src", domain_u, domain_p, mask_u, mask_p );
    linalg::assign( src.block_1(), ScalarType( 0 ) );
    linalg::assign( src.block_2(), ScalarType( 1 ) );
    linalg::apply( K, src, dst );
}

namespace detail {

/// Unit normals n = g / |g| (oriented like r̂) at the nodes of radial layer `r` that carry the
/// `boundary` flag; r̂ (coords) everywhere else. Returns a new lateral field shaped like coords.
template < typename ScalarType >
grid::Grid3DDataVec< ScalarType, 3 > layer_normals(
    const grid::Grid4DDataVec< ScalarType, 3 >&                     g,
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const int                                                       r,
    const grid::shell::ShellBoundaryFlag                            boundary,
    const char*                                                     label )
{
    grid::Grid3DDataVec< ScalarType, 3 > n( label, coords.extent( 0 ), coords.extent( 1 ), coords.extent( 2 ) );
    Kokkos::deep_copy( n, coords );

    const auto mask = boundary_mask;
    const auto c    = coords;
    Kokkos::parallel_for(
        label,
        Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >( { 0, 0, 0 }, { n.extent( 0 ), n.extent( 1 ), n.extent( 2 ) } ),
        KOKKOS_LAMBDA( const int s, const int x, const int y ) {
            if ( !util::has_flag( mask( s, x, y, r ), boundary ) )
                return;

            const ScalarType gx   = g( s, x, y, r, 0 );
            const ScalarType gy   = g( s, x, y, r, 1 );
            const ScalarType gz   = g( s, x, y, r, 2 );
            const ScalarType norm = Kokkos::sqrt( gx * gx + gy * gy + gz * gz );
            if ( norm == ScalarType( 0 ) )
                return;

            // Orient like r̂ (sign is irrelevant for the projection, kept for readability).
            const ScalarType dot = gx * c( s, x, y, 0 ) + gy * c( s, x, y, 1 ) + gz * c( s, x, y, 2 );
            const ScalarType sc  = ( dot < 0 ? ScalarType( -1 ) : ScalarType( 1 ) ) / norm;
            n( s, x, y, 0 )      = sc * gx;
            n( s, x, y, 1 )      = sc * gy;
            n( s, x, y, 2 )      = sc * gz;
        } );
    Kokkos::fence();
    return n;
}

} // namespace detail

/// Free-slip normals consistent with the discrete gradient: n_i = g_i / |g_i| with g = Bᵀ_Neumann·1
/// (oriented like r̂). With these, P·Bᵀ·1 = 0, so constant pressure is an exact null mode of K;
/// with r̂ it is only approximately one (see pressure_nullspace_residual()). Returns {cmb, surface}
/// normal fields, laterally shaped like coords_shell; nodes off the respective boundary keep r̂.
/// A field is only allocated for a free-slip boundary; for any other boundary it is r̂ (coords_shell
/// itself, no copy).
template < typename ScalarType, typename StokesOperator >
std::pair< grid::Grid3DDataVec< ScalarType, 3 >, grid::Grid3DDataVec< ScalarType, 3 > > consistent_freeslip_normals(
    StokesOperator&                                                   K_neumann,
    const grid::shell::DistributedDomain&                             domain_u,
    const grid::shell::DistributedDomain&                             domain_p,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&          mask_u,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&          mask_p,
    const grid::Grid3DDataVec< ScalarType, 3 >&                       coords,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&   boundary_mask,
    const grid::shell::BoundaryConditions&                            bcs )
{
    using grid::shell::BoundaryConditionFlag;
    using grid::shell::ShellBoundaryFlag;

    const bool fs_cmb  = grid::shell::get_boundary_condition_flag( bcs, ShellBoundaryFlag::CMB ) ==
                        BoundaryConditionFlag::FREESLIP;
    const bool fs_surf = grid::shell::get_boundary_condition_flag( bcs, ShellBoundaryFlag::SURFACE ) ==
                         BoundaryConditionFlag::FREESLIP;
    if ( !fs_cmb && !fs_surf )
        return { coords, coords };

    // g = Bᵀ_Neumann · 1 (velocity block of K_neumann · [0; 1]).
    linalg::VectorW1IsoW2W1< ScalarType > dst( "normals_dst", domain_u, domain_p, mask_u, mask_p );
    apply_to_constant_pressure( K_neumann, domain_u, domain_p, mask_u, mask_p, dst );
    const auto g     = dst.block_1().grid_data();
    const int  r_top = static_cast< int >( g.extent( 3 ) ) - 1;

    const auto n_cmb = fs_cmb ? detail::layer_normals( g, coords, boundary_mask, 0, ShellBoundaryFlag::CMB,
                                                       "freeslip_normals_cmb" ) :
                                coords;
    const auto n_surf = fs_surf ? detail::layer_normals( g, coords, boundary_mask, r_top, ShellBoundaryFlag::SURFACE,
                                                         "freeslip_normals_surface" ) :
                                  coords;
    return { n_cmb, n_surf };
}

/// Diagnostic: apply the BC-enforced Stokes operator K to [u = 0; p = 1] and store the velocity
/// block of the result in `out`. This is Bᵀ·1 with boundary rows treated as in the solve, so it
/// vanishes iff constant pressure is an exact null mode. The pressure block (B·0) must be zero.
/// Norms are logged relative to the same product with K_neumann (unconstrained boundaries).
template < typename ScalarType, typename StokesOperator, typename StokesOperatorNeumann >
void pressure_nullspace_residual(
    StokesOperator&                                           K,
    StokesOperatorNeumann&                                    K_neumann,
    const grid::shell::DistributedDomain&                     domain_u,
    const grid::shell::DistributedDomain&                     domain_p,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&  mask_u,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&  mask_p,
    linalg::VectorW1Vec< ScalarType >&                        out )
{
    linalg::VectorW1IsoW2W1< ScalarType > dst( "nullspace_dst", domain_u, domain_p, mask_u, mask_p );

    apply_to_constant_pressure( K_neumann, domain_u, domain_p, mask_u, mask_p, dst );
    const ScalarType ref_norm = linalg::norm_2( dst.block_1() );
    const ScalarType ref_max  = linalg::max_abs_entry( dst.block_1() );

    apply_to_constant_pressure( K, domain_u, domain_p, mask_u, mask_p, dst );
    const ScalarType norm   = linalg::norm_2( dst.block_1() );
    const ScalarType maxabs = linalg::max_abs_entry( dst.block_1() );
    const ScalarType p_max  = linalg::max_abs_entry( dst.block_2() );

    linalg::assign( out, dst.block_1() );

    util::logroot << "[Pressure nullspace check] K·[0; 1]:\n"
                  << "    ||(K·[0;1])_u||_2      = " << norm << "   (Neumann ref: " << ref_norm
                  << ", ratio " << ( ref_norm > 0 ? norm / ref_norm : ScalarType( 0 ) ) << ")\n"
                  << "    max|(K·[0;1])_u|       = " << maxabs << "   (Neumann ref: " << ref_max
                  << ", ratio " << ( ref_max > 0 ? maxabs / ref_max : ScalarType( 0 ) ) << ")\n"
                  << "    max|(K·[0;1])_p|       = " << p_max << "   (expected 0)" << std::endl;
}

} // namespace terra::fe
