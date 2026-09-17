#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/linear_form.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::linearforms::shell {

/// \brief Linear form for shear heating term in energy equation.

template < typename ScalarT, typename CoefficientT, int VelocityVecDim = 3 >
class ShearHeatingTerm
{
  public:
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    linalg::VectorQ1Scalar< ScalarT >              eta_;
    linalg::VectorQ1Vec< ScalarT, VelocityVecDim > velocity_;

    CoefficientT coefficient_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    // Kokkos views set in apply_impl() before the parallel launch.
    grid::Grid4DDataScalar< ScalarType >              dst_;
    grid::Grid4DDataScalar< ScalarType >              eta_grid_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

  public:
    ShearHeatingTerm(
        const grid::shell::DistributedDomain&                 domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&              grid,
        const grid::Grid2DDataScalar< ScalarT >&              radii,
        const linalg::VectorQ1Scalar< ScalarT >&              eta,
        const linalg::VectorQ1Vec< ScalarT, VelocityVecDim >& velocity,
        const CoefficientT&                                   coefficient,
        const linalg::OperatorApplyMode                       operator_apply_mode = linalg::OperatorApplyMode::Replace,
        const linalg::OperatorCommunicationMode               operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , eta_( eta )
    , velocity_( velocity )
    , coefficient_( coefficient )
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
        eta_grid_ = eta_.grid_data();
        vel_grid_ = velocity_.grid_data();

        Kokkos::parallel_for(
            "shear_heating_term", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    /// \brief Kokkos kernel: per-cell contribution to
    ///        \f$ f_i = \int_E c \, 2\eta \, \varepsilon'(\mathbf{u}) : \varepsilon'(\mathbf{u}) \, \phi_i \, \mathrm{d}x \f$,
    ///        with \f$ \varepsilon' = \varepsilon(\mathbf{u}) - \tfrac{1}{3} (\nabla\cdot\mathbf{u}) I \f$.
    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        // Quadrature points.
        constexpr int num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        {
            // Gather surface points for each wedge.
            dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

            // Gather wedge radii.
            const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
            const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

            dense::Vec< ScalarT, 6 > eta[num_wedges_per_hex_cell];

            extract_local_wedge_scalar_coefficients( eta, local_subdomain_id, x_cell, y_cell, r_cell, eta_grid_ );

            dense::Vec< ScalarT, 6 > vel_coeffs[VelocityVecDim][num_wedges_per_hex_cell];
            for ( int d = 0; d < VelocityVecDim; d++ )
            {
                extract_local_wedge_vector_coefficients(
                    vel_coeffs[d], local_subdomain_id, x_cell, y_cell, r_cell, d, vel_grid_ );
            }

            // Compute the local element matrix.

            for ( int q = 0; q < num_quad_points; q++ )
            {
                const auto w  = quad_weights[q];
                const auto qp = quad_points[q];

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, qp );
                    const auto det              = Kokkos::abs( J.det() );
                    const auto J_inv_transposed = J.inv().transposed();

                    ///////////////////////////////////////////////////////////////////////////////////////
                    // We need to implement the shear heating term
                    // \phi = 2\mu (\eps - \frac{1}{3}\nabla\cdot{u}) : (\eps - \frac{1}{3}\nabla\cdot{u})
                    // where, \eps = \frac{1}{2}(\nabla u + (\nabla u)^T)
                    ///////////////////////////////////////////////////////////////////////////////////////

                    ScalarType eta_eval = 0.0;

                    ScalarType dux_dx_eval = 0.0;
                    ScalarType dux_dy_eval = 0.0;
                    ScalarType dux_dz_eval = 0.0;

                    ScalarType duy_dx_eval = 0.0;
                    ScalarType duy_dy_eval = 0.0;
                    ScalarType duy_dz_eval = 0.0;

                    ScalarType duz_dx_eval = 0.0;
                    ScalarType duz_dy_eval = 0.0;
                    ScalarType duz_dz_eval = 0.0;

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto shape_j = shape( j, qp );

                        eta_eval += shape_j * eta[wedge]( j );

                        const auto grad_j = grad_shape( j, qp );

                        dux_dx_eval += ( J_inv_transposed * grad_j )( 0 ) * vel_coeffs[0][wedge]( j );
                        dux_dy_eval += ( J_inv_transposed * grad_j )( 1 ) * vel_coeffs[0][wedge]( j );
                        dux_dz_eval += ( J_inv_transposed * grad_j )( 2 ) * vel_coeffs[0][wedge]( j );

                        duy_dx_eval += ( J_inv_transposed * grad_j )( 0 ) * vel_coeffs[1][wedge]( j );
                        duy_dy_eval += ( J_inv_transposed * grad_j )( 1 ) * vel_coeffs[1][wedge]( j );
                        duy_dz_eval += ( J_inv_transposed * grad_j )( 2 ) * vel_coeffs[1][wedge]( j );

                        duz_dx_eval += ( J_inv_transposed * grad_j )( 0 ) * vel_coeffs[2][wedge]( j );
                        duz_dy_eval += ( J_inv_transposed * grad_j )( 1 ) * vel_coeffs[2][wedge]( j );
                        duz_dz_eval += ( J_inv_transposed * grad_j )( 2 ) * vel_coeffs[2][wedge]( j );
                    }

                    const auto shear_heating_qp =
                        2 * Kokkos::pow( 0.5 * dux_dy_eval + 0.5 * duy_dx_eval, 2 ) +
                        2 * Kokkos::pow( 0.5 * dux_dz_eval + 0.5 * duz_dx_eval, 2 ) +
                        2 * Kokkos::pow( 0.5 * duy_dz_eval + 0.5 * duz_dy_eval, 2 ) +
                        Kokkos::pow(
                            -0.33333333333333331 * dux_dx_eval - 0.33333333333333331 * duy_dy_eval +
                                0.66666666666666674 * duz_dz_eval,
                            2 ) +
                        Kokkos::pow(
                            -0.33333333333333331 * dux_dx_eval + 0.66666666666666674 * duy_dy_eval -
                                0.33333333333333331 * duz_dz_eval,
                            2 ) +
                        Kokkos::pow(
                            0.66666666666666674 * dux_dx_eval - 0.33333333333333331 * duy_dy_eval -
                                0.33333333333333331 * duz_dz_eval,
                            2 );

                    const ScalarT coeff = coefficient_( local_subdomain_id, x_cell, y_cell, r_cell, wedge, qp );

                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        const auto u = shape( i, qp );

                        dst[wedge]( i ) += w * coeff * shear_heating_qp * ( 2.0 * eta_eval ) * u * det;
                    }
                }
            }
        }

        {
            atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
        }
    }
};

} // namespace terra::fe::wedge::linearforms::shell
