#pragma once

#include <type_traits>

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/linear_form.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector_w1.hpp"

namespace terra::fe::wedge::linearforms::shell::detail {

template < typename T >
concept HasGridData = requires( const T& t )
{
    t.grid_data();
};
} // namespace terra::fe::wedge::linearforms::shell::detail

namespace terra::fe::wedge::linearforms::shell {

/// \brief Linear form for the TALA/PDA compressibility term in compressible Stokes.
///
/// Given a scalar FE function \f$\rho\f$ (density) and a vectorial FE function \f$\mathbf{u}\f$
/// (velocity), this linear form evaluates
/// \f[
///   f_i = \int_\Omega \frac{1}{\rho} \nabla\rho \cdot \mathbf{u} \, \phi_i \, \mathrm{d}x
/// \f]
/// into a scalar finite element coefficient vector, where \f$\phi_i\f$ are the scalar W1 test
/// functions on the spherical shell mesh.
///
/// Update: Templated class now supports density \f$\rho\f$ as either a scalar FE function
/// (VectorW1Scalar / Grid4DDataScalar) or as radial profile (Grid2DDataScalar).
///
/// This term arises in the pressure equation of the Truncated Anelastic Liquid Approximation
/// (TALA) and the Projected Density Approximation (PDA) for compressible Stokes flow in
/// geodynamics. See the [Stokes documentation](@ref stokes-compressible) for the full context.
///
/// \note The sign convention at the call site depends on the (2,1) block of the Stokes operator.
/// The `Divergence` block computes \f$-(q, \mathrm{div}\, u)\f$, so the mass conservation
/// equation \f$-(q, \mathrm{div}\, u) = f_p\f$ requires
/// \f$f_p = +\frac{1}{\rho}\nabla\rho\cdot\mathbf{u}\f$ term (positive sign).
///
/// \note \f$\rho\f$ must not vanish in the domain; no singular-value protection is applied.
///
/// **Concept.** This class satisfies \ref terra::linalg::LinearFormLike. Evaluation writes
/// the assembled coefficient vector into \p dst via `linalg::apply`:
///
/// \code{.cpp}
/// InvRhoGradRhoDotU< double > L( domain, grid, radii, rho, velocity );
/// linalg::VectorW1Scalar< double > g( domain );
/// linalg::apply( L, g );   // fills g_i = ∫ (1/ρ) ∇ρ·u φ_i dx
/// \endcode
///
/// The default `OperatorApplyMode::Replace` zeroes \p dst before accumulation. Pass
/// `OperatorApplyMode::Add` to add into an existing vector instead.
///
template < typename ScalarT, typename RhoFieldType, int VelocityVecDim = 3 >
class InvRhoGradRhoDotU
{
  public:
    using DstVectorType = linalg::VectorW1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

    // Evaluate whether rho is VectorW1Scalar (3-D) or radial profile
    static constexpr bool rho_is_radial_profile = std::is_same_v< RhoFieldType, grid::Grid2DDataScalar< ScalarT > >;

    using RhoGridType = std::
        conditional_t< rho_is_radial_profile, grid::Grid2DDataScalar< ScalarT >, grid::Grid4DDataScalar< ScalarT > >;

  private:
    grid::shell::DistributedDomain domain_;
    grid::shell::DistributedDomain domain_fine_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_fine_;
    grid::Grid2DDataScalar< ScalarT > radii_fine_;

    RhoFieldType rho_;

    linalg::VectorW1Vec< ScalarT, VelocityVecDim > velocity_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    // Kokkos views set in apply_impl() before the parallel launch.
    grid::Grid4DDataScalar< ScalarType >              dst_;
    RhoGridType                                       rho_grid_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

  public:
    InvRhoGradRhoDotU(
        const grid::shell::DistributedDomain&                 domain,
        const grid::shell::DistributedDomain&                 domain_fine,
        const grid::Grid3DDataVec< ScalarT, 3 >&              grid,
        const grid::Grid3DDataVec< ScalarT, 3 >&              grid_fine,
        const grid::Grid2DDataScalar< ScalarT >&              radii,
        const grid::Grid2DDataScalar< ScalarT >&              radii_fine,
        const RhoFieldType&                                   rho_fine,
        const linalg::VectorW1Vec< ScalarT, VelocityVecDim >& velocity_fine,
        const linalg::OperatorApplyMode                       operator_apply_mode = linalg::OperatorApplyMode::Replace,
        const linalg::OperatorCommunicationMode               operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , domain_fine_( domain_fine )
    , grid_( grid )
    , radii_( radii )
    , grid_fine_( grid_fine )
    , radii_fine_( radii_fine )
    , rho_( rho_fine )
    , velocity_( velocity_fine )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    void apply_impl( DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        dst_      = dst.grid_data();
        vel_grid_ = velocity_.grid_data();

        // unwrap rho if necessary
        if constexpr ( detail::HasGridData< RhoFieldType > )
            rho_grid_ = rho_.grid_data();
        else
            rho_grid_ = rho_;

        Kokkos::parallel_for(
            "inv_rho_grad_rho_dot_u", grid::shell::local_domain_md_range_policy_cells( domain_fine_ ), *this );
        Kokkos::fence();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    /// \brief Kokkos kernel: per-cell contribution to
    ///        \f$ f_i = \int_E \frac{1}{\rho} \nabla\rho \cdot \mathbf{u} \, \phi_i \, \mathrm{d}x \f$.
    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // -----------------------------------------------------------------------
        // Geometry
        // -----------------------------------------------------------------------
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_fine_, local_subdomain_id, x_cell, y_cell );

        const ScalarT r_1 = radii_fine_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_fine_( local_subdomain_id, r_cell + 1 );

        // -----------------------------------------------------------------------
        // Quadrature
        // -----------------------------------------------------------------------
        constexpr auto num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        // -----------------------------------------------------------------------
        // Extract local coefficients: rho (scalar, 3-D case only) and u (vector per dimension)
        // -----------------------------------------------------------------------
        dense::Vec< ScalarT, 6 > rho_coeffs[num_wedges_per_hex_cell] = {};

        // Coefficient extraction only makes sense for the 3-D FE field;
        // radial profile is read directly per-shell inside the quad-point loop below.
        if constexpr ( !rho_is_radial_profile )
        {
            extract_local_wedge_scalar_coefficients(
                rho_coeffs, local_subdomain_id, x_cell, y_cell, r_cell, rho_grid_ );
        }

        dense::Vec< ScalarT, 6 > vel_coeffs[VelocityVecDim][num_wedges_per_hex_cell];
        for ( int d = 0; d < VelocityVecDim; d++ )
        {
            extract_local_wedge_vector_coefficients(
                vel_coeffs[d], local_subdomain_id, x_cell, y_cell, r_cell, d, vel_grid_ );
        }

        // -----------------------------------------------------------------------
        // Per-wedge local contributions
        // -----------------------------------------------------------------------
        dense::Vec< ScalarT, num_nodes_per_wedge > contrib[num_wedges_per_hex_cell] = {};

        const int fine_radial_wedge_index = r_cell % 2;

        for ( int q = 0; q < num_quad_points; q++ )
        {
            const ScalarT w = quad_weights[q];

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();

                const int fine_lateral_wedge_index = fine_lateral_wedge_idx( x_cell, y_cell, wedge );

                // ----------------------------------------------------------------
                // Interpolate rho and compute physical gradient of rho at quad pt
                // ----------------------------------------------------------------
                ScalarT                  rho_q      = 0;
                dense::Vec< ScalarT, 3 > grad_rho_q = {};

                // 3-D FE interpolation path
                if constexpr ( !rho_is_radial_profile )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const ScalarT phi_j          = shape( j, quad_points[q] );
                        const auto    grad_phi_j_phy = J_inv_transposed * grad_shape( j, quad_points[q] );

                        rho_q += rho_coeffs[wedge]( j ) * phi_j;
                        grad_rho_q = grad_rho_q + rho_coeffs[wedge]( j ) * grad_phi_j_phy;
                    }
                }

                // radial-profile path -- rho = rho(r)
                // Physical radius and direction come from the same affine maps used
                // to build the Jacobian above (forward_map_rad / forward_map_lat).
                else
                {
                    const ScalarT xi   = quad_points[q]( 0 );
                    const ScalarT eta  = quad_points[q]( 1 );
                    const ScalarT zeta = quad_points[q]( 2 );

                    const auto lat_dir = forward_map_lat(
                        wedge_phy_surf[wedge][0], wedge_phy_surf[wedge][1], wedge_phy_surf[wedge][2], xi, eta );
                    const auto    r_hat  = lat_dir.normalized();
                    const ScalarT r_phys = forward_map_rad( r_1, r_2, zeta );

                    const ScalarT rho_1 = rho_grid_( local_subdomain_id, r_cell );
                    const ScalarT rho_2 = rho_grid_( local_subdomain_id, r_cell + 1 );

                    const ScalarT t = ( r_phys - r_1 ) / ( r_2 - r_1 );
                    rho_q           = rho_1 + t * ( rho_2 - rho_1 );

                    const ScalarT drho_dr = ( rho_2 - rho_1 ) / ( r_2 - r_1 );
                    grad_rho_q            = drho_dr * r_hat;
                }

                // ----------------------------------------------------------------
                // Interpolate velocity at quad pt
                // ----------------------------------------------------------------
                dense::Vec< ScalarT, VelocityVecDim > u_q = {};
                for ( int d = 0; d < VelocityVecDim; d++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        u_q( d ) += vel_coeffs[d][wedge]( j ) * shape( j, quad_points[q] );
                    }
                }

                // ----------------------------------------------------------------
                // Integrand: (1/rho) * grad(rho) . u
                // ----------------------------------------------------------------
                ScalarT dot = 0;
                for ( int d = 0; d < VelocityVecDim; d++ )
                {
                    dot += grad_rho_q( d ) * u_q( d );
                }
                const ScalarT integrand = dot / rho_q;

                // ----------------------------------------------------------------
                // Accumulate test-function contributions
                // ----------------------------------------------------------------
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    auto shape_coarse_i =
                        shape_coarse( i, fine_radial_wedge_index, fine_lateral_wedge_index, quad_points[q] );
                    contrib[wedge]( i ) += w * integrand * shape_coarse_i * det;
                    // contrib[wedge]( i ) += w * integrand * shape( i, quad_points[q] ) * det;
                }
            }
        }

        atomically_add_local_wedge_scalar_coefficients(
            dst_, local_subdomain_id, x_cell / 2, y_cell / 2, r_cell / 2, contrib );
    }
};

static_assert( linalg::LinearFormLike< InvRhoGradRhoDotU< double, linalg::VectorW1Scalar< double > > > );
static_assert( linalg::LinearFormLike< InvRhoGradRhoDotU< double, grid::Grid2DDataScalar< double > > > );

} // namespace terra::fe::wedge::linearforms::shell
