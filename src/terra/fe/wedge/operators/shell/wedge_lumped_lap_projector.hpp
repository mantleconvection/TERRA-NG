#pragma once

#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

/// \brief Per-wedge L2(lumped-mass) projection of κ·∇²T.
///
/// For each wedge w of each hex cell we precompute, at construction time, a
/// 6×6 "projector" matrix
///   P_w[i][j] = κ · K_w[i][j] / M_w_lumped[i],
/// where K_w is the wedge-local Galerkin Laplacian
///   K_w[i][j] = ∫_w ∇φ_i · ∇φ_j dV
/// and M_w_lumped[i] = Σ_j ∫_w φ_i φ_j dV is the row-sum lumping of the
/// wedge-local consistent mass.  Both integrals use the same Felippa 3×2
/// quadrature as the surrounding FE operators.
///
/// At apply time, for each wedge we form
///   lap_w[i] = (P_w · T_w)[i]      (i = 0..5)
/// which is the per-wedge nodal projection of κ·∇²T, evaluated weakly on
/// this single wedge with that wedge's own lumped mass.  The output is
/// stored as a 6-vector per wedge in a Kokkos View shaped
/// (num_subdomains, Nc_x, Nc_y, Nc_r, num_wedges_per_hex_cell) of
/// dense::Vec<ScalarT, 6>.  The entropy-viscosity kernel then interpolates
///   Lap(q) = Σ_j N_j(q) · lap_w[j]
/// at each Felippa quad point of the wedge.
///
/// Per-wedge lumping (rather than the global lumped mass shared between
/// wedges) keeps the projection a strictly local operation — no halo
/// exchange, no global Q1 lap field — and lines up naturally with the
/// per-wedge ν_h granularity used downstream.
template < typename ScalarT >
class WedgeLumpedLapProjector
{
  public:
    using ScalarType     = ScalarT;
    using ProjMatStorage =
        Kokkos::View< dense::Mat< ScalarT, num_nodes_per_wedge, num_nodes_per_wedge > **** [num_wedges_per_hex_cell],
                      grid::Layout >;
    using LapStorage =
        Kokkos::View< dense::Vec< ScalarT, num_nodes_per_wedge > **** [num_wedges_per_hex_cell], grid::Layout >;

  private:
    grid::shell::DistributedDomain    domain_;
    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;
    ScalarT                           kappa_;
    ProjMatStorage                    P_w_;

  public:
    WedgeLumpedLapProjector(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        ScalarT                                  kappa )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , kappa_( kappa )
    {
        const auto num_sub = static_cast< long long >( domain_.subdomains().size() );
        const auto nx_c    = domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
        const auto nr_c    = domain_.domain_info().subdomain_num_nodes_radially() - 1;
        P_w_ = ProjMatStorage( "wedge_lap_projector_P_w", num_sub, nx_c, nx_c, nr_c );
        assemble();
    }

    /// Allocate a per-wedge nodal-lap output buffer matching this projector's
    /// domain.  Caller owns it.
    LapStorage make_lap_storage( const std::string& name ) const
    {
        const auto num_sub = static_cast< long long >( domain_.subdomains().size() );
        const auto nx_c    = domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
        const auto nr_c    = domain_.domain_info().subdomain_num_nodes_radially() - 1;
        return LapStorage( name, num_sub, nx_c, nx_c, nr_c );
    }

    /// Apply the precomputed P_w to T to fill `lap_w_out`.  Strictly local —
    /// no halo exchange.
    void apply( const linalg::VectorQ1Scalar< ScalarT >& T, LapStorage& lap_w_out ) const
    {
        const auto T_data = T.grid_data();
        const auto P_w    = P_w_;

        Kokkos::parallel_for(
            "wedge_lumped_lap_apply",
            grid::shell::local_domain_md_range_policy_cells( domain_ ),
            KOKKOS_LAMBDA( int s, int xc, int yc, int rc ) {
                dense::Vec< ScalarT, num_nodes_per_wedge > T_w[num_wedges_per_hex_cell];
                extract_local_wedge_scalar_coefficients( T_w, s, xc, yc, rc, T_data );

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
                {
                    const auto& P = P_w( s, xc, yc, rc, wedge );
                    auto&       L = lap_w_out( s, xc, yc, rc, wedge );
                    L = P * T_w[wedge];
                }
            } );
        Kokkos::fence();
    }

  private:
    void assemble()
    {
        const auto    grid_lat  = grid_;
        const auto    radii_v   = radii_;
        const ScalarT kappa     = kappa_;
        const auto    P_w       = P_w_;

        constexpr int num_q = quadrature::quad_felippa_3x2_num_quad_points;

        Kokkos::parallel_for(
            "wedge_lumped_lap_assemble",
            grid::shell::local_domain_md_range_policy_cells( domain_ ),
            KOKKOS_LAMBDA( int s, int xc, int yc, int rc ) {
                dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
                wedge_surface_physical_coords( wedge_phy_surf, grid_lat, s, xc, yc );

                const ScalarT r_1 = radii_v( s, rc );
                const ScalarT r_2 = radii_v( s, rc + 1 );

                dense::Vec< ScalarT, 3 > qp[num_q];
                ScalarT                  qw[num_q];
                quadrature::quad_felippa_3x2_quad_points( qp );
                quadrature::quad_felippa_3x2_quad_weights( qw );

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
                {
                    dense::Mat< ScalarT, num_nodes_per_wedge, num_nodes_per_wedge > K_w{};
                    dense::Vec< ScalarT, num_nodes_per_wedge >                      M_w_lumped{};

                    for ( int q = 0; q < num_q; ++q )
                    {
                        const dense::Mat< ScalarT, 3, 3 > J = jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                        const ScalarT det     = J.det();
                        const ScalarT abs_det = Kokkos::abs( det );
                        if ( abs_det < ScalarT( 1e-30 ) )
                        {
                            continue;
                        }
                        const dense::Mat< ScalarT, 3, 3 > J_inv_t = J.inv_transposed( det );

                        dense::Vec< ScalarT, 3 > grad_phy[num_nodes_per_wedge];
                        ScalarT                  N_q[num_nodes_per_wedge];
                        for ( int k = 0; k < num_nodes_per_wedge; ++k )
                        {
                            grad_phy[k] = J_inv_t * grad_shape( k, qp[q] );
                            N_q[k]      = shape( k, qp[q] );
                        }

                        const ScalarT w_d = qw[q] * abs_det;

                        for ( int i = 0; i < num_nodes_per_wedge; ++i )
                        {
                            // Lumped mass row-sum: Σ_j N_i N_j = N_i · 1 (partition of unity).
                            M_w_lumped( i ) += w_d * N_q[i];
                            for ( int j = 0; j < num_nodes_per_wedge; ++j )
                            {
                                K_w( i, j ) += w_d * grad_phy[i].dot( grad_phy[j] );
                            }
                        }
                    }

                    auto& P = P_w( s, xc, yc, rc, wedge );
                    for ( int i = 0; i < num_nodes_per_wedge; ++i )
                    {
                        const ScalarT inv_m =
                            ( M_w_lumped( i ) > ScalarT( 0 ) ) ? ( ScalarT( 1 ) / M_w_lumped( i ) ) : ScalarT( 0 );
                        for ( int j = 0; j < num_nodes_per_wedge; ++j )
                        {
                            P( i, j ) = kappa * K_w( i, j ) * inv_m;
                        }
                    }
                }
            } );
        Kokkos::fence();
    }
};

} // namespace terra::fe::wedge::operators::shell
