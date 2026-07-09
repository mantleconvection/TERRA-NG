#pragma once

#include <limits>

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
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
/// Full formulas (KHB 2012, GJI 191, page 7 — exact paper text):
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
};

/// Floor on the normalization D used in compute_entropy_stats.  When the
/// solution is nearly constant ‖E − E_avg‖ → 0 and the residual-scaled
/// branch blows up; this baked-in floor keeps ν_h well defined and lets the
/// min(·, ·) cap take over.  Guards against NaNs in pure-advection tests
/// where T equals the initial cone shape away from the front.
template < typename ScalarT >
KOKKOS_INLINE_FUNCTION constexpr ScalarT entropy_viscosity_D_floor()
{
    return ScalarT( 1e-14 );
}

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
        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
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
        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
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
/// Implementation notes:
///   - T_m uses the global [min, max] range (NOT the volume-average), matching
///     the ASPECT choice (KHB 2012, p. 7).  This keeps the entropy symmetric
///     around the midpoint of the physical range, independent of domain-mass
///     bias.
///   - E_avg is a true FE volume integral evaluated on the same Felippa 3×2
///     quadrature points as the per-wedge ν_h kernel:
///       E_avg = (Σ_wedges Σ_q w_q·|det J(q)|·E(q)) / (Σ_wedges Σ_q w_q·|det J(q)|)
///     where E(q) = ½·(T(q) − T_m)² and T(q) is interpolated from the 6
///     wedge nodes via Σ_j N_j(q)·T_j.
///   - D uses the signed-magnitude ∞-norm of (E − E_avg) over owned nodes.
///     A tiny floor (entropy_viscosity_D_floor) prevents division by zero
///     when T is nearly constant.
template < typename ScalarT >
EntropyStats< ScalarT > compute_entropy_stats(
    const linalg::VectorQ1Scalar< ScalarT >&                     T,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&     ownership_mask,
    const grid::shell::DistributedDomain&                        domain,
    const grid::Grid3DDataVec< ScalarT, 3 >&                     grid_coords,
    const grid::Grid2DDataScalar< ScalarT >&                     radii,
    const EntropyViscosityParameters< ScalarT >&                 params,
    MPI_Comm                                                     comm = MPI_COMM_WORLD )
{
    EntropyStats< ScalarT > stats;

    const auto T_data = T.grid_data();

    // T range — same as the nodal version.
    stats.T_min = min_entry_owned( T_data, ownership_mask, comm );
    stats.T_max = max_entry_owned( T_data, ownership_mask, comm );
    stats.T_m   = ScalarT( 0.5 ) * ( stats.T_min + stats.T_max );

    const ScalarT T_m_local = stats.T_m;
    const auto    grid_lat  = grid_coords;
    const auto    radii_v   = radii;

    constexpr int num_q = quadrature::quad_felippa_3x2_num_quad_points;

    // E_avg as ⟨E⟩ = ∫ E dV / |Ω|, summed locally then MPI-reduced.
    // Each owning rank only contributes integrals over cells in its
    // subdomain; cells are not shared between ranks (only nodes are),
    // so this is partition-of-unity at the integration level.
    ScalarT sum_E_dV = 0;
    ScalarT sum_dV   = 0;

    Kokkos::parallel_reduce(
        "ev_E_avg_volume_int",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int id, int xc, int yc, int rc, ScalarT& acc_E, ScalarT& acc_V ) {
            dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_lat, id, xc, yc );

            const ScalarT r_1 = radii_v( id, rc );
            const ScalarT r_2 = radii_v( id, rc + 1 );

            dense::Vec< ScalarT, 3 > qp[num_q];
            ScalarT                  qw[num_q];
            quadrature::quad_felippa_3x2_quad_points( qp );
            quadrature::quad_felippa_3x2_quad_weights( qw );

            dense::Vec< ScalarT, num_nodes_per_wedge > T_w[num_wedges_per_hex_cell];
            extract_local_wedge_scalar_coefficients( T_w, id, xc, yc, rc, T_data );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < num_q; ++q )
                {
                    const dense::Mat< ScalarT, 3, 3 > J =
                        jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                    const ScalarT abs_det = Kokkos::abs( J.det() );
                    if ( abs_det < ScalarT( 1e-30 ) )
                    {
                        continue;
                    }

                    ScalarT T_q = 0;
                    for ( int j = 0; j < num_nodes_per_wedge; ++j )
                    {
                        T_q += shape( j, qp[q] ) * T_w[wedge]( j );
                    }
                    const ScalarT d  = T_q - T_m_local;
                    const ScalarT Eq = ScalarT( 0.5 ) * d * d;

                    acc_E += qw[q] * abs_det * Eq;
                    acc_V += qw[q] * abs_det;
                }
            }
        },
        sum_E_dV,
        sum_dV );

    Kokkos::fence();

    MPI_Allreduce( MPI_IN_PLACE, &sum_E_dV, 1, mpi::mpi_datatype< ScalarT >(), MPI_SUM, comm );
    MPI_Allreduce( MPI_IN_PLACE, &sum_dV,   1, mpi::mpi_datatype< ScalarT >(), MPI_SUM, comm );

    stats.E_avg = ( sum_dV > ScalarT( 0 ) ) ? ( sum_E_dV / sum_dV ) : ScalarT( 0 );

    // D normalisation: volume-weighted L² norm of (E − E_avg), evaluated at the
    // same Felippa quadrature as E_avg.  Tracks the entropy field's overall
    // magnitude rather than a single BL-extremum node.
    //   D = ( ∫(E − E_avg)² dV / |Ω| )^½
    const ScalarT E_avg_local      = stats.E_avg;
    ScalarT       sum_dE2_dV_local = 0;

    Kokkos::parallel_reduce(
        "ev_D_l2_volume",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int id, int xc, int yc, int rc, ScalarT& acc ) {
            dense::Vec< ScalarT, 3 >
                wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_lat, id, xc, yc );

            const ScalarT r_1 = radii_v( id, rc );
            const ScalarT r_2 = radii_v( id, rc + 1 );

            dense::Vec< ScalarT, 3 > qp[num_q];
            ScalarT                  qw[num_q];
            quadrature::quad_felippa_3x2_quad_points( qp );
            quadrature::quad_felippa_3x2_quad_weights( qw );

            dense::Vec< ScalarT, num_nodes_per_wedge > T_w[num_wedges_per_hex_cell];
            extract_local_wedge_scalar_coefficients( T_w, id, xc, yc, rc, T_data );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < num_q; ++q )
                {
                    const dense::Mat< ScalarT, 3, 3 > J =
                        jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                    const ScalarT abs_det = Kokkos::abs( J.det() );
                    if ( abs_det < ScalarT( 1e-30 ) ) continue;
                    ScalarT T_q = 0;
                    for ( int j = 0; j < num_nodes_per_wedge; ++j )
                        T_q += shape( j, qp[q] ) * T_w[wedge]( j );
                    const ScalarT d  = T_q - T_m_local;
                    const ScalarT Eq = ScalarT( 0.5 ) * d * d;
                    const ScalarT dE = Eq - E_avg_local;
                    acc += qw[q] * abs_det * dE * dE;
                }
            }
        },
        sum_dE2_dV_local );
    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &sum_dE2_dV_local, 1,
                   mpi::mpi_datatype< ScalarT >(), MPI_SUM, comm );
    // sum_dV is already the global volume from the E_avg reduction above.
    const ScalarT D_local = ( sum_dV > ScalarT( 0 ) )
                                ? Kokkos::sqrt( sum_dE2_dV_local / sum_dV )
                                : ScalarT( 0 );

    stats.D = Kokkos::max( D_local, entropy_viscosity_D_floor< ScalarT >() );

    return stats;
}

/// Compute per-wedge entropy viscosity ν_h on the shell.
///
/// One scalar ν_h per wedge (2 wedges per hex), evaluated at the same
/// Felippa 3×2 quadrature points used by the existing FE operators (Mass,
/// DivKGrad, AD-SUPG, ...). At each quad point we compute:
///   - field interpolants T(q), Tp(q), u(q), Lap(q) via Σ_j N_j(q)·field_j
///     over the 6 wedge nodes,
///   - FE-consistent physical-space gradient grad_T(q) = J^{-T}(q)·∇_ξ T(q)
///     using the existing `jac()` and `inv_transposed()` helpers,
///   - the strong-form KHB-2012 residual
///       r_E = |∂_t E + (T−T_m)·(u·∇T − κ∇²T − γ)|.
/// The per-wedge r_E is the L²-mean over Felippa quadpoints
///   r_E_w = ( ∫ r_E² dV / V_wedge )^½
/// folded into the standard ν_h cap:
///   ν_h_w = min( α_max·h_w·‖u‖_∞,w,  α_E·h_w²·r_E_w / D )
/// with `h_w = (r_2 − r_1)`, the radial cell extent — the BL-normal length
/// scale, which is the relevant scale for stabilising the temperature-gradient
/// direction in shell convection.  V_wedge^{1/3} mixes lateral and radial
/// scales and over-inflates ν_E in BL-touching wedges where dr ≪ lateral edge.
///
/// The Lap input is a Q1-nodal field carrying the global lumped-mass
/// projection of −κ∇²T (i.e. (K·T)/M_lumped, with K the global Galerkin
/// Laplacian and M_lumped the global lumped mass).  Continuity across
/// element interfaces makes r_E collapse to ≈0 in smooth regions, so ν_h
/// honours its no-stabilization-needed limit.  The kernel extracts the 6
/// wedge-local nodal values per wedge and interpolates
///   Lap(q) = Σ_j N_j(q) · lap_w[wedge](j).
template < typename ScalarT >
void compute_nu_h(
    grid::Grid5DDataScalar< ScalarT >&                           nu_h_wedge,
    const linalg::VectorQ1Scalar< ScalarT >&                     T_n,
    const linalg::VectorQ1Scalar< ScalarT >&                     T_nm1,
    const linalg::VectorQ1Vec< ScalarT, 3 >&                     u,
    const grid::Grid4DDataScalar< ScalarT >&                     lap_data,
    const grid::shell::DistributedDomain&                        domain,
    const grid::Grid3DDataVec< ScalarT, 3 >&                     grid_coords,
    const grid::Grid2DDataScalar< ScalarT >&                     radii,
    ScalarT                                                      dt,
    const EntropyStats< ScalarT >&                               stats,
    const EntropyViscosityParameters< ScalarT >&                 params,
    ScalarT                                                      gamma = ScalarT( 0 ) )
{
    const auto T_data = T_n.grid_data();
    const auto T_prev = T_nm1.grid_data();
    const auto u_data = u.grid_data();
    const auto lap_v  = lap_data;

    const ScalarT T_m       = stats.T_m;
    const ScalarT inv_D     = ScalarT( 1 ) / stats.D;
    const ScalarT inv_dt    = ScalarT( 1 ) / dt;
    const ScalarT alpha_max = params.alpha_max;
    const ScalarT alpha_E   = params.alpha_E;
    const ScalarT gamma_    = gamma;
    const auto    grid_lat  = grid_coords;
    const auto    radii_v   = radii;

    constexpr int num_q = quadrature::quad_felippa_3x2_num_quad_points;

    Kokkos::parallel_for(
        "compute_nu_h",
        grid::shell::local_domain_md_range_policy_cells( domain ),
        KOKKOS_LAMBDA( int id, int xc, int yc, int rc ) {
            // Wedge surface coords on the unit sphere, both wedges of the hex.
            dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_lat, id, xc, yc );

            // Inner / outer radii of this hex's radial slab.
            const ScalarT r_1 = radii_v( id, rc );
            const ScalarT r_2 = radii_v( id, rc + 1 );

            // Felippa 3x2 quadrature.
            dense::Vec< ScalarT, 3 > qp[num_q];
            ScalarT                  qw[num_q];
            quadrature::quad_felippa_3x2_quad_points( qp );
            quadrature::quad_felippa_3x2_quad_weights( qw );

            // Per-wedge gather of T_n, T_{n-1}, u (3 components), and the
            // Q1-nodal Lap projection.
            dense::Vec< ScalarT, num_nodes_per_wedge > T_w[num_wedges_per_hex_cell];
            dense::Vec< ScalarT, num_nodes_per_wedge > Tp_w[num_wedges_per_hex_cell];
            dense::Vec< ScalarT, num_nodes_per_wedge > Lap_w[num_wedges_per_hex_cell];
            dense::Vec< ScalarT, num_nodes_per_wedge > u_w[num_wedges_per_hex_cell][3];

            extract_local_wedge_scalar_coefficients( T_w,   id, xc, yc, rc, T_data );
            extract_local_wedge_scalar_coefficients( Tp_w,  id, xc, yc, rc, T_prev );
            extract_local_wedge_scalar_coefficients( Lap_w, id, xc, yc, rc, lap_v );
            for ( int d = 0; d < 3; ++d )
            {
                dense::Vec< ScalarT, num_nodes_per_wedge > comp[num_wedges_per_hex_cell];
                extract_local_wedge_vector_coefficients( comp, id, xc, yc, rc, d, u_data );
                u_w[0][d] = comp[0];
                u_w[1][d] = comp[1];
            }

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                ScalarT V_wedge    = 0;
                ScalarT r_E_sq_int = 0; // ∫ r_E² dV (L²-mean projection)
                ScalarT u_max_norm = 0;

                for ( int q = 0; q < num_q; ++q )
                {
                    // Full 3D Jacobian J = ∂x/∂(ξ,η,ζ) at this quad point.
                    const dense::Mat< ScalarT, 3, 3 > J =
                        jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                    const ScalarT det     = J.det();
                    const ScalarT abs_det = Kokkos::abs( det );
                    if ( abs_det < ScalarT( 1e-30 ) )
                    {
                        continue; // degenerate — skip this quad point
                    }
                    const dense::Mat< ScalarT, 3, 3 > J_inv_t = J.inv_transposed( det );

                    // Interpolate fields and compute physical grad_T at q.
                    ScalarT                  T_q   = 0;
                    ScalarT                  Tp_q  = 0;
                    ScalarT                  Lap_q = 0;
                    dense::Vec< ScalarT, 3 > u_q{};
                    dense::Vec< ScalarT, 3 > grad_T_q{};

                    for ( int j = 0; j < num_nodes_per_wedge; ++j )
                    {
                        const ScalarT N_j        = shape( j, qp[q] );
                        const auto    grad_xi_j  = grad_shape( j, qp[q] );
                        const auto    grad_phys  = J_inv_t * grad_xi_j;

                        T_q   += N_j * T_w[wedge]( j );
                        Tp_q  += N_j * Tp_w[wedge]( j );
                        Lap_q += N_j * Lap_w[wedge]( j );
                        for ( int d = 0; d < 3; ++d )
                        {
                            u_q( d )      += N_j * u_w[wedge][d]( j );
                            grad_T_q( d ) += T_w[wedge]( j ) * grad_phys( d );
                        }
                    }

                    // Strong-form residual at the quad point.
                    const ScalarT dT     = T_q  - T_m;
                    const ScalarT dT_p   = Tp_q - T_m;
                    const ScalarT E      = ScalarT( 0.5 ) * dT   * dT;
                    const ScalarT Ep     = ScalarT( 0.5 ) * dT_p * dT_p;
                    const ScalarT dE_dt  = ( E - Ep ) * inv_dt;
                    const ScalarT u_dot_g =
                        u_q( 0 ) * grad_T_q( 0 ) + u_q( 1 ) * grad_T_q( 1 ) + u_q( 2 ) * grad_T_q( 2 );
                    // KHB 2012 residual (page 7, formula above eq. 16):
                    //   r_E = ∂_t E + (T − T_m)·(u·∇T − κ∇²T − γ).
                    // The (T − T_m) factor multiplies the entire RHS of the
                    // T-PDE residual.  Vanishes on smooth solutions of the
                    // temperature equation.
                    // Lap_q already encodes −κ∇²T (lumped-mass-projected).
                    const ScalarT r_E_q =
                        Kokkos::abs( dE_dt + dT * ( u_dot_g + Lap_q - gamma_ ) );

                    r_E_sq_int += qw[q] * abs_det * r_E_q * r_E_q;
                    u_max_norm = Kokkos::max( u_max_norm, u_q.norm() );
                    V_wedge   += qw[q] * abs_det;
                }

                // h_w = radial cell extent (r_2 − r_1).  BL-normal length
                // scale; V^{1/3} mixed lateral and radial scales and inflated
                // ν_E in BL-touching wedges where dr ≪ lateral edge.
                const ScalarT h_w = r_2 - r_1;

                // Per-wedge r_E as L²-mean over Felippa quadpoints:
                //   r_E_w = ( ∫ r_E² dV / V_w )^½.
                const ScalarT r_E_w = ( V_wedge > ScalarT( 0 ) )
                                          ? Kokkos::sqrt( r_E_sq_int / V_wedge )
                                          : ScalarT( 0 );

                const ScalarT nu_w = Kokkos::min( alpha_max * h_w * u_max_norm,
                                                  alpha_E * h_w * h_w * r_E_w * inv_D );

                nu_h_wedge( id, xc, yc, rc, wedge ) = nu_w;
            }
        } );

    Kokkos::fence();
}

} // namespace terra::fe::wedge::operators::shell
