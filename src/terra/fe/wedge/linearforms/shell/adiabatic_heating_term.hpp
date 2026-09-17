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

/// \brief Linear form for the adiabatic heating term in the energy equation.

template < typename ScalarT, typename CoefficientT, int VelocityVecDim = 3 >
class AdiabaticHeatingTerm
{
  public:
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    linalg::VectorQ1Scalar< ScalarT >              T_;
    linalg::VectorQ1Vec< ScalarT, VelocityVecDim > velocity_;

    CoefficientT coefficient_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    // Kokkos views set in apply_impl() before the parallel launch.
    grid::Grid4DDataScalar< ScalarType >              dst_;
    grid::Grid4DDataScalar< ScalarType >              T_grid_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

  public:
    AdiabaticHeatingTerm(
        const grid::shell::DistributedDomain&                 domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&              grid,
        const grid::Grid2DDataScalar< ScalarT >&              radii,
        const linalg::VectorQ1Scalar< ScalarT >&              T,
        const linalg::VectorQ1Vec< ScalarT, VelocityVecDim >& velocity,
        const CoefficientT&                                   coefficient,
        const linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        const linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , T_( T )
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
        T_grid_   = T_.grid_data();
        vel_grid_ = velocity_.grid_data();

        Kokkos::parallel_for(
            "adiabatic_heating_term", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    /// \brief Kokkos kernel: per-cell contribution to
    ///        \f$ f_i = -\int_E c \, (\mathbf{u} \cdot \hat{\mathbf{r}}) \, T \, \phi_i \, \mathrm{d}x \f$.
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

            dense::Vec< ScalarT, 6 > T[num_wedges_per_hex_cell];

            extract_local_wedge_scalar_coefficients(
                T, local_subdomain_id, x_cell, y_cell, r_cell, T_grid_ );

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
                    // We need to implement the adiabatic heating term
                    // coeff * (u\cdot g) * T
                    ///////////////////////////////////////////////////////////////////////////////////////

                    ScalarType T_eval = 0.0;

                    dense::Vec<ScalarT, VelocityVecDim> vel_eval;
                    vel_eval.fill(0.0);

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto shape_j = shape( j, qp );

                        T_eval += shape_j * T[wedge]( j );
                        vel_eval(0) += shape_j * vel_coeffs[0][wedge]( j );
                        vel_eval(1) += shape_j * vel_coeffs[1][wedge]( j );
                        vel_eval(2) += shape_j * vel_coeffs[2][wedge]( j );
                    }

                    const auto lat_dir = forward_map_lat(
                        wedge_phy_surf[wedge][0], wedge_phy_surf[wedge][1], wedge_phy_surf[wedge][2], qp(0), qp(1) );

                    const auto r_hat  = lat_dir.normalized();

                    // we need vector pointing in the direction of gravity,
                    // and hopefully that will not invert for the Earth anytime soon in this Universe.

                    const ScalarT adiabatic_heating_qp = (-r_hat.dot(vel_eval) * T_eval);

                    const ScalarT coeff = coefficient_( local_subdomain_id, x_cell, y_cell, r_cell, wedge, qp );

                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        const auto u = shape( i, qp );

                        dst[wedge]( i ) += w * coeff * adiabatic_heating_qp * u * det;
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
