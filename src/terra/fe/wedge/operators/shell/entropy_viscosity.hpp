#pragma once

#include <limits>

#include "communication/shell/communication.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/bit_masks.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

/// \brief Entropy-viscosity stabilization for the advection-diffusion equation.
///
/// Implements the scheme from Kronbichler, Heister & Bangerth (GJI 191, 2012),
/// which itself adopts Guermond, Pasquetti & Popov (JCP 230, 2011).
///
/// The idea: instead of SUPG's streamline diffusion (which is *always on* and
/// smears sharp features), add an *isotropic* artificial diffusion ν_h(x) that
/// scales with an entropy residual r_E.  In smooth regions r_E ≈ 0 so no
/// diffusion is added; near shocks/layers r_E is large and the scheme kicks in
/// with a diffusion bounded above by first-order upwind.
///
/// Full formulas (ASPECT §3.2.6, Eq. 15):
///   E(T)     = ½ (T − T_m)²,                    T_m = ½(T_min + T_max)
///   r_E|_K   = ‖∂_t E + (T − T_m)·(u·∇T − κ∇²T − γ)‖_{∞, K}
///   E_avg    = (1/|Ω|) ∫_Ω E(T)
///   D        = ‖E − E_avg‖_{∞, Ω}
///   ν_h|_K   = min(α_max · h_K · ‖u‖_{∞,K},
///                  α_E   · h_K² · r_E|_K / D)
///   α_max    = 0.026 · d     (→ 0.078 in 3D)
///   α_E      = 1.0
///
/// Time-lagged: r_E uses (E^{n−1} − E^{n−2})/dt + values at midpoint
/// (T^{n−1}+T^{n−2})/2, so ν_h is computed from past states and the solve for
/// T^{n+1} stays linear.
///
/// The resulting ν_h is then used as the *coefficient* of a Galerkin
/// ∇·(ν_h ∇T*) term, which can be assembled by `DivKGrad` (already in
/// terraneo) with k_ = ν_h.  The operator's output is then added to the RHS
/// of the implicit advection-diffusion solve (IMPES-style lagged
/// stabilization).
///
/// This header is stabilization-only: it provides the kernels to *compute*
/// ν_h.  Assembling the actual RHS contribution uses existing infrastructure.

template < typename ScalarT >
struct EntropyViscosityParameters
{
    /// First-order upwind cap on the artificial diffusion.  ASPECT uses
    /// 0.026 · spatial_dim; in 3D that is 0.078.
    ScalarT alpha_max = ScalarT( 0.078 );

    /// Residual scaling constant.  ASPECT uses 1.0.  Tezduyar-style
    /// shock-capturing frameworks tune this in [0.1, 2.0].
    ScalarT alpha_E = ScalarT( 1.0 );

    /// Floor on the normalization D.  When the solution is nearly constant,
    /// ‖E − E_avg‖ → 0 and the residual-scaled branch blows up.  Guarding
    /// with a small floor keeps ν_h well defined and the min(·, ·) cap takes
    /// over.  ASPECT uses an effectively-unbounded D when near-constant; we
    /// guard explicitly to avoid NaNs in pure-advection tests where T is
    /// identically the initial cone shape away from the front.
    ScalarT D_floor = ScalarT( 1e-14 );
};

/// Global scalar stats derived from the current temperature field.
///
/// All members are MPI-reduced (valid on every rank).  Produced by
/// `compute_entropy_stats`.
template < typename ScalarT >
struct EntropyStats
{
    ScalarT T_min; ///< global min of T over owned nodes
    ScalarT T_max; ///< global max of T over owned nodes
    ScalarT T_m;   ///< (T_min + T_max) / 2 — the midpoint used in E(T)
    ScalarT E_avg; ///< volume-averaged entropy ⟨E⟩ = (1/|Ω|) ∫ E dV
    ScalarT D;     ///< global ∞-norm of (E − E_avg) over owned nodes; the
                   ///< normalization factor in the residual branch of ν_h
};

/// Signed max reduction over owned Q1 nodes.  kernels::common has
/// `min_entry` and `max_abs_entry` but not a signed `max_entry`; we provide
/// one locally so the EV code does not depend on a header edit.
template < typename ScalarT >
ScalarT max_entry_owned(
    const grid::Grid4DDataScalar< ScalarT >&                     x,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&     ownership_mask,
    MPI_Comm                                                     comm = MPI_COMM_WORLD )
{
    ScalarT max_val = std::numeric_limits< ScalarT >::lowest();

    Kokkos::parallel_reduce(
        "ev_max_entry_owned",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int id, int i, int j, int k, ScalarT& lmax ) {
            if ( util::has_flag( ownership_mask( id, i, j, k ), grid::NodeOwnershipFlag::OWNED ) )
            {
                lmax = Kokkos::max( lmax, x( id, i, j, k ) );
            }
        },
        Kokkos::Max< ScalarT >( max_val ) );

    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &max_val, 1, mpi::mpi_datatype< ScalarT >(), MPI_MAX, comm );
    return max_val;
}

/// Same for signed min (same caveat as above — kernels::common::min_entry
/// does not respect an ownership mask).
template < typename ScalarT >
ScalarT min_entry_owned(
    const grid::Grid4DDataScalar< ScalarT >&                     x,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&     ownership_mask,
    MPI_Comm                                                     comm = MPI_COMM_WORLD )
{
    ScalarT min_val = std::numeric_limits< ScalarT >::max();

    Kokkos::parallel_reduce(
        "ev_min_entry_owned",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int id, int i, int j, int k, ScalarT& lmin ) {
            if ( util::has_flag( ownership_mask( id, i, j, k ), grid::NodeOwnershipFlag::OWNED ) )
            {
                lmin = Kokkos::min( lmin, x( id, i, j, k ) );
            }
        },
        Kokkos::Min< ScalarT >( min_val ) );

    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &min_val, 1, mpi::mpi_datatype< ScalarT >(), MPI_MIN, comm );
    return min_val;
}

/// Compute the global entropy stats (T_min, T_max, T_m, E_avg, D) used by
/// the ν_h formula.
///
/// Pass the ownership mask so only *owned* nodes contribute — interior-ghost
/// / halo nodes carry partial data that would double-count in reductions.
///
/// Implementation notes:
///   - T_m uses the global [min, max] range (NOT the volume-average), matching
///     the ASPECT choice (KHB 2012, p. 7).  This keeps the entropy symmetric
///     around the midpoint of the physical range, independent of domain-mass
///     bias.
///   - E_avg is computed as a simple nodal average here, not a proper FEM
///     volume integral.  That's consistent with per-cell-constant ν_h — the
///     normalization just sets a scale; being accurate to the mass matrix
///     buys nothing and would add a solve.  If numerical sensitivity proves
///     problematic, we can swap to ‖M·E‖_1 / ‖M·1‖_1.
///   - D uses the signed-magnitude ∞-norm over owned nodes.  A tiny floor
///     (params.D_floor) prevents division by zero when T is nearly constant.
template < typename ScalarT >
EntropyStats< ScalarT > compute_entropy_stats(
    const linalg::VectorQ1Scalar< ScalarT >&                     T,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&     ownership_mask,
    const EntropyViscosityParameters< ScalarT >&                 params,
    MPI_Comm                                                     comm = MPI_COMM_WORLD )
{
    EntropyStats< ScalarT > stats;

    const auto T_data = T.grid_data();

    // --- T range ----------------------------------------------------------
    stats.T_min = min_entry_owned( T_data, ownership_mask, comm );
    stats.T_max = max_entry_owned( T_data, ownership_mask, comm );
    stats.T_m   = ScalarT( 0.5 ) * ( stats.T_min + stats.T_max );

    const ScalarT T_m_local = stats.T_m;

    // --- ⟨E⟩ and |Ω| as ownership-mask nodal averages ---------------------
    // We reduce (sum_E, n_owned) jointly by interpreting the ownership mask
    // as a weight of 1.  Two reductions so we keep both sums exact.
    ScalarT  sum_E = 0;
    long int n_owned_local = 0;

    Kokkos::parallel_reduce(
        "ev_entropy_sum",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { T_data.extent( 0 ), T_data.extent( 1 ), T_data.extent( 2 ), T_data.extent( 3 ) } ),
        KOKKOS_LAMBDA( int id, int i, int j, int k, ScalarT& acc ) {
            if ( util::has_flag( ownership_mask( id, i, j, k ), grid::NodeOwnershipFlag::OWNED ) )
            {
                const ScalarT d = T_data( id, i, j, k ) - T_m_local;
                acc += ScalarT( 0.5 ) * d * d;
            }
        },
        sum_E );

    Kokkos::parallel_reduce(
        "ev_owned_count",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { T_data.extent( 0 ), T_data.extent( 1 ), T_data.extent( 2 ), T_data.extent( 3 ) } ),
        KOKKOS_LAMBDA( int id, int i, int j, int k, long int& acc ) {
            if ( util::has_flag( ownership_mask( id, i, j, k ), grid::NodeOwnershipFlag::OWNED ) )
            {
                acc += 1;
            }
        },
        n_owned_local );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &sum_E,        1, mpi::mpi_datatype< ScalarT >(),  MPI_SUM, comm );
    MPI_Allreduce( MPI_IN_PLACE, &n_owned_local, 1, MPI_LONG,                        MPI_SUM, comm );

    stats.E_avg = ( n_owned_local > 0 ) ? ( sum_E / static_cast< ScalarT >( n_owned_local ) ) : ScalarT( 0 );

    // --- D = max|E − E_avg| ---------------------------------------------
    ScalarT D_local = 0;
    const ScalarT E_avg_local = stats.E_avg;

    Kokkos::parallel_reduce(
        "ev_D_max",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { T_data.extent( 0 ), T_data.extent( 1 ), T_data.extent( 2 ), T_data.extent( 3 ) } ),
        KOKKOS_LAMBDA( int id, int i, int j, int k, ScalarT& lmax ) {
            if ( util::has_flag( ownership_mask( id, i, j, k ), grid::NodeOwnershipFlag::OWNED ) )
            {
                const ScalarT d  = T_data( id, i, j, k ) - T_m_local;
                const ScalarT E  = ScalarT( 0.5 ) * d * d;
                const ScalarT dE = Kokkos::abs( E - E_avg_local );
                lmax             = Kokkos::max( lmax, dE );
            }
        },
        Kokkos::Max< ScalarT >( D_local ) );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &D_local, 1, mpi::mpi_datatype< ScalarT >(), MPI_MAX, comm );

    stats.D = Kokkos::max( D_local, params.D_floor );

    return stats;
}

/// Compute the per-cell entropy viscosity ν_h.
///
/// Implementation notes (first-cut, simplified):
///  - Evaluates r_E at the 8 corner nodes of each hex cell (not Felippa
///    quadrature points).  Taking the ∞-norm over 8 corners is conservative
///    and requires no quadrature machinery.
///  - The diffusion contribution `−κ∇²T` enters via `lap_T_pointwise`.  The
///    caller is responsible for producing this Q1 nodal field; the standard
///    recipe is `lap_T_pointwise = (DivKGrad(κ) · T) / M_lumped`, which is
///    the lumped-mass projection of the FE-assembled (weak) Laplacian back
///    to a pointwise estimate of `−∇·(κ∇T)`.  Pass a zero-filled vector to
///    drop the term (e.g. for pure advection tests, κ = 0).
///  - ∂_t E uses simple first-order backward difference: (E^n − E^{n-1})/dt.
///    ASPECT uses BDF-2 with lagged indices (E^{n-1} − E^{n-2}) to keep the
///    stabilization explicit; we can swap later.  For now T_nm1 holds the
///    "previous" temperature.
///  - ∇T is approximated as a face-averaged central difference in the local
///    (x̂, ŷ, r̂) cell basis.  For spherical-shell wedge cells the three local
///    axes are approximately orthogonal and this is accurate to O(h).
///  - h_K is the cell diagonal divided by √3 — the "equivalent edge length"
///    of a cube with the same diagonal.  Cheap and robust for non-uniform
///    wedge cells.
///
/// Inputs:
///  - `nu_h`: pre-allocated per-cell output.  Must have extents
///    (num_subdomains, N−1, N−1, N_r−1) matching `local_domain_md_range_policy_cells`.
///  - `T_n`, `T_nm1`: Q1 temperature at current and previous timestep.
///  - `u`: Q1 vector velocity (nodal).
///  - `lap_data`: Q1 nodal estimate of `−κ∇²T` (or zero to skip).
///  - `domain`, `grid_coords`, `radii`: shell geometry.
///  - `dt`, `stats`, `params`: step size and pre-computed EV parameters.
template < typename ScalarT >
void compute_nu_h(
    grid::Grid4DDataScalar< ScalarT >&                           nu_h,
    const linalg::VectorQ1Scalar< ScalarT >&                     T_n,
    const linalg::VectorQ1Scalar< ScalarT >&                     T_nm1,
    const linalg::VectorQ1Vec< ScalarT, 3 >&                     u,
    const grid::Grid4DDataScalar< ScalarT >&                     lap_data,
    const grid::shell::DistributedDomain&                        domain,
    const grid::Grid3DDataVec< ScalarT, 3 >&                     grid_coords,
    const grid::Grid2DDataScalar< ScalarT >&                     radii,
    ScalarT                                                      dt,
    const EntropyStats< ScalarT >&                               stats,
    const EntropyViscosityParameters< ScalarT >&                 params )
{
    const auto T_data    = T_n.grid_data();
    const auto T_prev    = T_nm1.grid_data();
    const auto u_data    = u.grid_data();

    const ScalarT T_m       = stats.T_m;
    const ScalarT inv_D     = ScalarT( 1 ) / stats.D;
    const ScalarT inv_dt    = ScalarT( 1 ) / dt;
    const ScalarT alpha_max = params.alpha_max;
    const ScalarT alpha_E   = params.alpha_E;

    // Offsets of the 8 hex-cell corners in (x_cell, y_cell, r_cell) index space.
    // Bit-ordering: bit 0 = x, bit 1 = y, bit 2 = r.  So corner c(b) has
    //   (offs_x, offs_y, offs_r) = (b & 1, (b >> 1) & 1, (b >> 2) & 1).

    Kokkos::parallel_for(
        "compute_nu_h",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int id, int xc, int yc, int rc ) {
            // Gather corners in physical space + field values at those corners.
            dense::Vec< ScalarT, 3 > x_corner[8];
            ScalarT                  T_c[8];
            ScalarT                  Tp_c[8];
            dense::Vec< ScalarT, 3 > u_c[8];
            ScalarT                  Lap_c[8]; // ≈ −κ∇²T (pointwise) at each corner

            for ( int b = 0; b < 8; ++b )
            {
                const int ox = ( b & 1 );
                const int oy = ( ( b >> 1 ) & 1 );
                const int or_ = ( ( b >> 2 ) & 1 );
                const int xx = xc + ox;
                const int yy = yc + oy;
                const int rr = rc + or_;

                x_corner[b] = grid::shell::coords( id, xx, yy, rr, grid_coords, radii );
                T_c[b]      = T_data( id, xx, yy, rr );
                Tp_c[b]     = T_prev( id, xx, yy, rr );
                u_c[b]( 0 ) = u_data( id, xx, yy, rr, 0 );
                u_c[b]( 1 ) = u_data( id, xx, yy, rr, 1 );
                u_c[b]( 2 ) = u_data( id, xx, yy, rr, 2 );
                Lap_c[b]    = lap_data( id, xx, yy, rr );
            }

            // Face-averaged T and corner-averaged positions in the three local
            // directions.  The "x0 face" is where the x-offset is 0, and so on.
            ScalarT                  T_x0 = 0, T_x1 = 0, T_y0 = 0, T_y1 = 0, T_r0 = 0, T_r1 = 0;
            dense::Vec< ScalarT, 3 > X_x0{}, X_x1{}, X_y0{}, X_y1{}, X_r0{}, X_r1{};
            for ( int b = 0; b < 8; ++b )
            {
                const ScalarT q = ScalarT( 0.25 );
                if ( ( b & 1 ) == 0 )       { T_x0 += q * T_c[b]; X_x0 = X_x0 + x_corner[b] * q; }
                else                         { T_x1 += q * T_c[b]; X_x1 = X_x1 + x_corner[b] * q; }
                if ( ( ( b >> 1 ) & 1 ) == 0 ) { T_y0 += q * T_c[b]; X_y0 = X_y0 + x_corner[b] * q; }
                else                            { T_y1 += q * T_c[b]; X_y1 = X_y1 + x_corner[b] * q; }
                if ( ( ( b >> 2 ) & 1 ) == 0 ) { T_r0 += q * T_c[b]; X_r0 = X_r0 + x_corner[b] * q; }
                else                            { T_r1 += q * T_c[b]; X_r1 = X_r1 + x_corner[b] * q; }
            }

            // Local-axis edge vectors + lengths.
            const dense::Vec< ScalarT, 3 > ex_vec = X_x1 - X_x0;
            const dense::Vec< ScalarT, 3 > ey_vec = X_y1 - X_y0;
            const dense::Vec< ScalarT, 3 > er_vec = X_r1 - X_r0;
            const ScalarT                  dx    = ex_vec.norm();
            const ScalarT                  dy    = ey_vec.norm();
            const ScalarT                  dr    = er_vec.norm();

            // Guard against degenerate cells.
            const ScalarT eps = ScalarT( 1e-30 );
            if ( dx < eps || dy < eps || dr < eps )
            {
                nu_h( id, xc, yc, rc ) = ScalarT( 0 );
                return;
            }

            const dense::Vec< ScalarT, 3 > ex_hat = ex_vec * ( ScalarT( 1 ) / dx );
            const dense::Vec< ScalarT, 3 > ey_hat = ey_vec * ( ScalarT( 1 ) / dy );
            const dense::Vec< ScalarT, 3 > er_hat = er_vec * ( ScalarT( 1 ) / dr );

            // Central differences in the local frame, then rotate to physical.
            // Assumes (ex_hat, ey_hat, er_hat) ≈ orthogonal, true for shell cells.
            const ScalarT                  dT_dx = ( T_x1 - T_x0 ) / dx;
            const ScalarT                  dT_dy = ( T_y1 - T_y0 ) / dy;
            const ScalarT                  dT_dr = ( T_r1 - T_r0 ) / dr;
            const dense::Vec< ScalarT, 3 > grad_T =
                ex_hat * dT_dx + ey_hat * dT_dy + er_hat * dT_dr;

            // Cell size h_K: diagonal / sqrt(3), the "equivalent cubic edge".
            const dense::Vec< ScalarT, 3 > diag = x_corner[7] - x_corner[0];
            const ScalarT                  h_K  = diag.norm() / Kokkos::sqrt( ScalarT( 3 ) );

            // r_E at each corner, take the max over the 8 corners.
            ScalarT r_E_max   = 0;
            ScalarT u_max_norm = 0;
            for ( int b = 0; b < 8; ++b )
            {
                const ScalarT dT     = T_c[b]  - T_m;
                const ScalarT dT_p   = Tp_c[b] - T_m;
                const ScalarT E      = ScalarT( 0.5 ) * dT   * dT;
                const ScalarT Ep     = ScalarT( 0.5 ) * dT_p * dT_p;
                const ScalarT dE_dt  = ( E - Ep ) * inv_dt;
                const ScalarT u_dot_gradT =
                    u_c[b]( 0 ) * grad_T( 0 ) + u_c[b]( 1 ) * grad_T( 1 ) + u_c[b]( 2 ) * grad_T( 2 );
                // Full KHB residual:  ∂_t E + (T − T_m)·(u·∇T − κ∇²T).
                // `Lap_c[b]` already encodes −κ∇²T (lumped-mass-projected), so add it here.
                const ScalarT r_E_corner = Kokkos::abs( dE_dt + dT * ( u_dot_gradT + Lap_c[b] ) );
                r_E_max                  = Kokkos::max( r_E_max, r_E_corner );
                u_max_norm               = Kokkos::max( u_max_norm, u_c[b].norm() );
            }

            // ν_h = min(α_max · h · ‖u‖_∞,  α_E · h² · r_E / D)
            const ScalarT nu_max = alpha_max * h_K * u_max_norm;
            const ScalarT nu_E   = alpha_E * h_K * h_K * r_E_max * inv_D;
            nu_h( id, xc, yc, rc ) = Kokkos::min( nu_max, nu_E );
        } );

    Kokkos::fence();
}

/// Backward-compatible overload: drops the `−κ∇²T` term from the residual.
/// Allocates a transient zero-filled `lap_data` view matching T's extents and
/// forwards to the primary `compute_nu_h`.  Use this when the diffusion-
/// residual contribution is not needed (e.g. pure advection, κ = 0).  Tests
/// that wire up a proper lumped-mass-projected Laplacian call the primary
/// overload with their own `lap_data`.
template < typename ScalarT >
void compute_nu_h(
    grid::Grid4DDataScalar< ScalarT >&                           nu_h,
    const linalg::VectorQ1Scalar< ScalarT >&                     T_n,
    const linalg::VectorQ1Scalar< ScalarT >&                     T_nm1,
    const linalg::VectorQ1Vec< ScalarT, 3 >&                     u,
    const grid::shell::DistributedDomain&                        domain,
    const grid::Grid3DDataVec< ScalarT, 3 >&                     grid_coords,
    const grid::Grid2DDataScalar< ScalarT >&                     radii,
    ScalarT                                                      dt,
    const EntropyStats< ScalarT >&                               stats,
    const EntropyViscosityParameters< ScalarT >&                 params )
{
    // Kokkos::View memory is zero-initialized by default for trivial scalars,
    // so this view is the zero field with no extra fill needed.
    const auto T_grid = T_n.grid_data();
    grid::Grid4DDataScalar< ScalarT > lap_zero(
        "ev_lap_zero",
        T_grid.extent( 0 ), T_grid.extent( 1 ), T_grid.extent( 2 ), T_grid.extent( 3 ) );

    compute_nu_h( nu_h, T_n, T_nm1, u, lap_zero, domain, grid_coords, radii,
                  dt, stats, params );
}

/// Project a per-cell ν_h field onto Q1 nodes by averaging the (up-to-)eight
/// cells touching each node.
///
/// Cells outside the local subdomain extent are silently skipped — halo
/// contributions from neighbouring subdomains are not included.  At subdomain
/// boundaries the nodal value therefore under-samples by roughly one cell
/// in each missing direction; ν_h is small anyway in smooth regions, and
/// this loss is only at the boundary strip.  The effect on the final solve
/// is expected to be a small perturbation of the boundary coupling, bounded
/// by `nu_h_max · h`.
///
/// If you need fully consistent values, follow this call with a
/// `pack_send_and_recv_local_subdomain_boundaries` / `unpack_and_reduce`
/// round-trip using the sum and a neighbour count — see DivKGrad's
/// apply_impl for the idiom.
template < typename ScalarT >
void project_nu_h_to_nodes(
    linalg::VectorQ1Scalar< ScalarT >&                     nu_h_nodal,
    const grid::Grid4DDataScalar< ScalarT >&               nu_h_cells,
    const grid::shell::DistributedDomain&                  domain )
{
    auto            nu_node = nu_h_nodal.grid_data();
    const long long nx_c    = nu_h_cells.extent( 1 );
    const long long ny_c    = nu_h_cells.extent( 2 );
    const long long nr_c    = nu_h_cells.extent( 3 );

    Kokkos::parallel_for(
        "project_nu_h_to_nodes",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( int id, int xn, int yn, int rn ) {
            ScalarT sum   = 0;
            int     count = 0;
            for ( int ox = -1; ox <= 0; ++ox )
            {
                for ( int oy = -1; oy <= 0; ++oy )
                {
                    for ( int or_ = -1; or_ <= 0; ++or_ )
                    {
                        const int xc = xn + ox;
                        const int yc = yn + oy;
                        const int rc = rn + or_;
                        if ( xc >= 0 && xc < nx_c && yc >= 0 && yc < ny_c && rc >= 0 && rc < nr_c )
                        {
                            sum += nu_h_cells( id, xc, yc, rc );
                            ++count;
                        }
                    }
                }
            }
            nu_node( id, xn, yn, rn ) = ( count > 0 ) ? sum / static_cast< ScalarT >( count ) : ScalarT( 0 );
        } );

    Kokkos::fence();
}

} // namespace terra::fe::wedge::operators::shell
