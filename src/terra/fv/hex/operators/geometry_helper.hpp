
#pragma once

/// @file geometry_helper.hpp
/// @brief Shared FV hex cell geometry: face-normal surface integrals, velocity fluxes, cell volume.
///
/// Used by both fct_advection.hpp (advection only) and advection_diffusion.hpp (advection + diffusion).
/// Cell centers must have been initialised via fv::hex::initialize_cell_centers so that ghost layers
/// hold valid neighbour values — required for the outward-normal sign check.

#include "fe/triangle/quadrature/quadrature.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/grid_types.hpp"

namespace terra::fv::hex::operators::detail {

/// @brief Stateless geometry helper for a single FV hex cell.
///
/// Computes:
///   - `beta[6]`   : velocity flux ∫ u·n dS through each of the 6 faces (positive = outward).
///   - `M_ii`      : cell volume.
///   - `S_f[6]`    : area-weighted outward normal vector for each face (∫ n dS).
///
/// Neighbour ordering (faces 0..5):
///   face 0: x-1   face 1: x+1
///   face 2: y-1   face 3: y+1
///   face 4: r-1   face 5: r+1
template < typename ScalarT >
struct GeometryHelper
{
    using Vec3 = dense::Vec< ScalarT, 3 >;

    static constexpr int num_neighbors                = 6;
    static constexpr int cell_offset_x[num_neighbors] = { -1, 1, 0, 0, 0, 0 };
    static constexpr int cell_offset_y[num_neighbors] = { 0, 0, -1, 1, 0, 0 };
    static constexpr int cell_offset_r[num_neighbors] = { 0, 0, 0, 0, -1, 1 };

    enum class WedgeContribution : int { NONE = 0, TRIANGLE = 1, QUAD = 2 };
    enum class CrossProductType : int  { XI_ETA = 0, XI_ZETA = 1, ETA_ZETA = 2 };

    KOKKOS_INLINE_FUNCTION
    static constexpr WedgeContribution contribution( const int hex_direction, const int wedge_id )
    {
        if ( wedge_id == 0 )
        {
            switch ( hex_direction )
            {
            case 0: return WedgeContribution::QUAD;
            case 2: return WedgeContribution::QUAD;
            case 4: return WedgeContribution::TRIANGLE;
            case 5: return WedgeContribution::TRIANGLE;
            default: return WedgeContribution::NONE;
            }
        }
        else
        {
            switch ( hex_direction )
            {
            case 1: return WedgeContribution::QUAD;
            case 3: return WedgeContribution::QUAD;
            case 4: return WedgeContribution::TRIANGLE;
            case 5: return WedgeContribution::TRIANGLE;
            default: return WedgeContribution::NONE;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    static constexpr void map_face(
        const int         hex_direction,
        const int         wedge_id,
        const double      s,
        const double      t,
        double&           xi,
        double&           eta,
        double&           zeta,
        CrossProductType& cross_product_type )
    {
        if ( wedge_id == 0 )
        {
            switch ( hex_direction )
            {
            case 0: xi = 0;   eta = s;  zeta = t;    cross_product_type = CrossProductType::ETA_ZETA; break;
            case 2: xi = s;   eta = 0;  zeta = t;    cross_product_type = CrossProductType::XI_ZETA;  break;
            case 4: xi = s;   eta = t;  zeta = -1.0; cross_product_type = CrossProductType::XI_ETA;   break;
            case 5: xi = s;   eta = t;  zeta =  1.0; cross_product_type = CrossProductType::XI_ETA;   break;
            default: xi = eta = zeta = 0; cross_product_type = CrossProductType::XI_ETA; break;
            }
        }
        else
        {
            switch ( hex_direction )
            {
            case 1: xi = 0;   eta = s;  zeta = t;    cross_product_type = CrossProductType::ETA_ZETA; break;
            case 3: xi = s;   eta = 0;  zeta = t;    cross_product_type = CrossProductType::XI_ZETA;  break;
            case 4: xi = s;   eta = t;  zeta = -1.0; cross_product_type = CrossProductType::XI_ETA;   break;
            case 5: xi = s;   eta = t;  zeta =  1.0; cross_product_type = CrossProductType::XI_ETA;   break;
            default: xi = eta = zeta = 0; cross_product_type = CrossProductType::XI_ETA; break;
            }
        }
    }

    /// @brief Compute beta[6], M_ii, and S_f[6] for the cell at (x_cell, y_cell, r_cell).
    ///
    /// @param beta    [out] Velocity flux ∫ u·n dS per face; positive = flow out of cell.
    /// @param M_ii    [out] Cell volume.
    /// @param S_f     [out] Area-weighted outward normal ∫ n dS per face.
    ///                      All three are needed by advection-diffusion; only beta and M_ii by pure advection.
    KOKKOS_INLINE_FUNCTION
    static void compute_geometry(
        const grid::Grid3DDataVec< ScalarT, 3 >&  grid,
        const grid::Grid2DDataScalar< ScalarT >&  radii,
        const grid::Grid4DDataVec< ScalarT, 3 >&  cell_centers,
        const grid::Grid4DDataVec< ScalarT, 3 >&  vel_grid,
        const int                                  local_subdomain_id,
        const int                                  x_cell,
        const int                                  y_cell,
        const int                                  r_cell,
        ScalarT ( &beta )[num_neighbors],
        ScalarT&                                   M_ii,
        Vec3 ( &S_f )[num_neighbors] )
    {
        // Surface coords for both wedges.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                               [fe::wedge::num_nodes_per_wedge_surface] = {};
        fe::wedge::wedge_surface_physical_coords( wedge_phy_surf, grid, local_subdomain_id, x_cell - 1, y_cell - 1 );

        const ScalarT r_1 = radii( local_subdomain_id, r_cell - 1 );
        const ScalarT r_2 = radii( local_subdomain_id, r_cell );

        // Velocity coefficients (3 components × 2 wedges × 6 nodes).
        dense::Vec< ScalarT, 6 > vel_coeffs[3][fe::wedge::num_wedges_per_hex_cell];
        for ( int d = 0; d < 3; d++ )
            fe::wedge::extract_local_wedge_vector_coefficients(
                vel_coeffs[d], local_subdomain_id, x_cell - 1, y_cell - 1, r_cell - 1, d, vel_grid );

        // Quadrature for face integrals.
        constexpr auto num_quad_points_tri = fe::triangle::quadrature::quad_triangle_3_num_quad_points;
        dense::Vec< ScalarT, 2 > quad_points_tri[num_quad_points_tri];
        ScalarT                  quad_weights_tri[num_quad_points_tri];
        fe::triangle::quadrature::quad_triangle_3_quad_points( quad_points_tri );
        fe::triangle::quadrature::quad_triangle_3_quad_weights( quad_weights_tri );

        constexpr auto num_quad_points_line                    = 2;
        ScalarT        quad_points_line[num_quad_points_line]  = { -1.0 / Kokkos::sqrt( 3 ), 1.0 / Kokkos::sqrt( 3 ) };
        ScalarT        quad_weights_line[num_quad_points_line] = { 1.0, 1.0 };

        // Cell center.
        const Vec3 cell_center{
            cell_centers( local_subdomain_id, x_cell, y_cell, r_cell, 0 ),
            cell_centers( local_subdomain_id, x_cell, y_cell, r_cell, 1 ),
            cell_centers( local_subdomain_id, x_cell, y_cell, r_cell, 2 ) };

        // Initialise outputs.
        for ( int n = 0; n < num_neighbors; ++n )
        {
            beta[n] = ScalarT( 0 );
            S_f[n]  = {};
        }
        M_ii = ScalarT( 0 );

        // Face integrals.
        for ( int neighbor = 0; neighbor < num_neighbors; ++neighbor )
        {
            for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
            {
                const auto contr = contribution( neighbor, wedge );
                if ( contr == WedgeContribution::NONE ) continue;

                if ( contr == WedgeContribution::TRIANGLE )
                {
                    for ( int q = 0; q < num_quad_points_tri; ++q )
                    {
                        const auto s = quad_points_tri[q]( 0 );
                        const auto t = quad_points_tri[q]( 1 );

                        ScalarT xi = 0, eta = 0, zeta = 0;
                        CrossProductType cpt;
                        map_face( neighbor, wedge, s, t, xi, eta, zeta, cpt );

                        Vec3 dx_dxi{}, dx_deta{}, dx_dzeta{}, u{};
                        for ( int i = 0; i < fe::wedge::num_nodes_per_wedge; ++i )
                        {
                            const auto shape_i = fe::wedge::shape( i, xi, eta, zeta );
                            const auto grad_i  = fe::wedge::grad_shape( i, xi, eta, zeta );
                            const Vec3 x_i     = wedge_phy_surf[wedge][i % 3] * ( i < 3 ? r_1 : r_2 );
                            const Vec3 u_i{ vel_coeffs[0][wedge]( i ), vel_coeffs[1][wedge]( i ), vel_coeffs[2][wedge]( i ) };
                            dx_dxi   = dx_dxi   + grad_i( 0 ) * x_i;
                            dx_deta  = dx_deta  + grad_i( 1 ) * x_i;
                            dx_dzeta = dx_dzeta + grad_i( 2 ) * x_i;
                            u        = u        + shape_i * u_i;
                        }

                        const Vec3 n = dx_dxi.cross( dx_deta ); // cpt must be XI_ETA here
                        beta[neighbor] += quad_weights_tri[q] * u.dot( n );
                        S_f[neighbor]   = S_f[neighbor] + quad_weights_tri[q] * n;
                    }
                }
                else // QUAD
                {
                    for ( int q_a = 0; q_a < num_quad_points_line; ++q_a )
                    {
                        for ( int q_b = 0; q_b < num_quad_points_line; ++q_b )
                        {
                            // map_face maps (s, t) as:
                            //   s → lateral reference coordinate (xi or eta) ∈ [0,1]
                            //   t → zeta (radial direction)                  ∈ [-1,1]
                            //
                            // The lateral coordinate lives on the standard triangle, so its
                            // valid range is [0,1], not [-1,1].  Remap the Gauss point from
                            // [-1,1] to [0,1] and include the Jacobian factor 1/2.
                            //
                            // t maps to zeta in all four QUAD cases (directions 0,1,2,3) —
                            // confirmed in map_face above.  The radial reference coordinate
                            // already spans [-1,1], so t needs no remapping.
                            const auto s = ScalarT( 0.5 ) * ( quad_points_line[q_a] + ScalarT( 1 ) );
                            const auto t = quad_points_line[q_b];
                            const auto w = ScalarT( 0.5 ) * quad_weights_line[q_a] * quad_weights_line[q_b];

                            ScalarT xi = 0, eta = 0, zeta = 0;
                            CrossProductType cpt;
                            map_face( neighbor, wedge, s, t, xi, eta, zeta, cpt );

                            Vec3 dx_dxi{}, dx_deta{}, dx_dzeta{}, u{};
                            for ( int i = 0; i < fe::wedge::num_nodes_per_wedge; ++i )
                            {
                                const auto shape_i = fe::wedge::shape( i, xi, eta, zeta );
                                const auto grad_i  = fe::wedge::grad_shape( i, xi, eta, zeta );
                                const Vec3 x_i     = wedge_phy_surf[wedge][i % 3] * ( i < 3 ? r_1 : r_2 );
                                const Vec3 u_i{ vel_coeffs[0][wedge]( i ), vel_coeffs[1][wedge]( i ), vel_coeffs[2][wedge]( i ) };
                                dx_dxi   = dx_dxi   + grad_i( 0 ) * x_i;
                                dx_deta  = dx_deta  + grad_i( 1 ) * x_i;
                                dx_dzeta = dx_dzeta + grad_i( 2 ) * x_i;
                                u        = u        + shape_i * u_i;
                            }

                            Vec3 n{};
                            if ( cpt == CrossProductType::XI_ZETA )
                                n = dx_dxi.cross( dx_dzeta );
                            else
                                n = dx_deta.cross( dx_dzeta );

                            beta[neighbor] += w * u.dot( n );
                            S_f[neighbor]   = S_f[neighbor] + w * n;
                        }
                    }
                }
            }

            // Ensure S_f[neighbor] points outward from cell i toward cell j.
            // Neighbour cell centre is available in ghost layers after initialize_cell_centers;
            // on curved domains this is the symmetric, correct reference direction.
            const int  nx = x_cell + cell_offset_x[neighbor];
            const int  ny = y_cell + cell_offset_y[neighbor];
            const int  nr = r_cell + cell_offset_r[neighbor];
            const Vec3 neighbor_center{
                cell_centers( local_subdomain_id, nx, ny, nr, 0 ),
                cell_centers( local_subdomain_id, nx, ny, nr, 1 ),
                cell_centers( local_subdomain_id, nx, ny, nr, 2 ) };

            if ( S_f[neighbor].dot( neighbor_center - cell_center ) < 0 )
            {
                S_f[neighbor]  = S_f[neighbor] * ScalarT( -1 );
                beta[neighbor] *= ScalarT( -1 );
            }
        }

        // Cell volume = mass matrix diagonal entry.
        constexpr auto num_quad_points_vol = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;
        dense::Vec< ScalarT, 3 > quad_points_vol[num_quad_points_vol];
        ScalarT                  quad_weights_vol[num_quad_points_vol];
        fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points_vol );
        fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights_vol );

        for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
            for ( int q = 0; q < num_quad_points_vol; ++q )
            {
                const auto J       = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points_vol[q] );
                const auto abs_det = Kokkos::abs( J.det() );
                M_ii += abs_det * quad_weights_vol[q];
            }
    }

}; // struct GeometryHelper

} // namespace terra::fv::hex::operators::detail
