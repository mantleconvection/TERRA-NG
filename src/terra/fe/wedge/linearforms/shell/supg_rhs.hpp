#pragma once
#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/linear_form.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector_w1.hpp"

namespace terra::fe::wedge::linearforms::shell {

/// \brief Linear form to assemble part of the RHS vector for a SUPG-stabilized method-of-lines
/// discretization of the advection-diffusion equation.
///
/// \note See \ref terra::fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG for notes
/// on the discretization. This linear form evaluates what is therein called \f$F_{\mathrm{SUPG}}\f$.
///
/// Given finite element functions \f$f\f$ and \f$\mathbf{u}\f$, this linear form evaluates
/// \f[
///   (F_{\mathrm{SUPG}})_i = \sum_E \int_E \tau_E (\mathbf{u} \cdot \nabla \phi_i) f
/// \f]
/// into a finite element coefficient vector, where \f$\tau_E\f$ is the element-local SUPG stabilization
/// parameter (computed on-the-fly, like in \ref terra::fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG).
///
template < typename ScalarT, int VelocityVecDim = 3 >
class SUPGRHS
{
  public:
    using DstVectorType = linalg::VectorW1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    linalg::VectorW1Scalar< ScalarT >              f_;
    linalg::VectorW1Vec< ScalarT, VelocityVecDim > velocity_;

    ScalarT diffusivity_;

    grid::Grid4DDataScalar< ScalarType >              dst_;
    grid::Grid4DDataScalar< ScalarType >              f_grid_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

  public:
    SUPGRHS(
        const grid::shell::DistributedDomain&                 domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&              grid,
        const grid::Grid2DDataScalar< ScalarT >&              radii,
        const linalg::VectorW1Scalar< ScalarT >&              f,
        const linalg::VectorW1Vec< ScalarT, VelocityVecDim >& velocity,
        const ScalarT                                         diffusivity,

        const linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        const linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , f_( f )
    , velocity_( velocity )
    , diffusivity_( diffusivity )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
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
        f_grid_   = f_.grid_data();

        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Gather surface points for each wedge.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // Quadrature points.
        constexpr auto num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        // Interpolating velocity into quad points.

        dense::Vec< ScalarT, VelocityVecDim > vel_interp[num_wedges_per_hex_cell][num_quad_points];
        dense::Vec< ScalarT, 6 >              vel_coeffs[VelocityVecDim][num_wedges_per_hex_cell];

        for ( int d = 0; d < VelocityVecDim; d++ )
        {
            extract_local_wedge_vector_coefficients(
                vel_coeffs[d], local_subdomain_id, x_cell, y_cell, r_cell, d, vel_grid_ );
        }

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    for ( int d = 0; d < VelocityVecDim; d++ )
                    {
                        vel_interp[wedge][q]( d ) += vel_coeffs[d][wedge]( i ) * shape_i;
                    }
                }
            }
        }

        // Interpolating f into quad points.
        ScalarT                  f_interp[num_wedges_per_hex_cell][num_quad_points] = {};
        dense::Vec< ScalarT, 6 > f_coeffs[num_wedges_per_hex_cell]                  = {};

        extract_local_wedge_scalar_coefficients( f_coeffs, local_subdomain_id, x_cell, y_cell, r_cell, f_grid_ );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    f_interp[wedge][q] += f_coeffs[wedge]( i ) * shape_i;
                }
            }
        }

        // Let's compute the streamline diffusivity.

        ScalarT streamline_diffusivity[num_wedges_per_hex_cell] = {};

        // Far from accurate but for now assume h = r.
        const auto h = r_2 - r_1;

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            ScalarT tau_accum = 0.0;
            ScalarT waccum    = 0.0;

            for ( int q = 0; q < num_quad_points; ++q )
            {
                // get velocity at this quad point
                const auto&   uq         = vel_interp[wedge][q];
                const ScalarT vel_norm_q = uq.norm();

                const ScalarT tau_q = operators::shell::supg_tau< ScalarT >( vel_norm_q, diffusivity_, h, 1e-08 );

                // quadrature weight for this point (if you have weights)
                const ScalarT wq = quad_weights[q]; // if not available, use 1.0
                tau_accum += tau_q * wq;
                waccum += wq;
            }

            // final cell/wedge tau: volume-weighted average
            ScalarT tau_cell              = ( waccum > 0.0 ) ? ( tau_accum / waccum ) : 0.0;
            streamline_diffusivity[wedge] = tau_cell;
        }

        // Local contributions
        dense::Vec< ScalarT, num_nodes_per_wedge > contrib[num_wedges_per_hex_cell] = {};

        for ( int q = 0; q < num_quad_points; q++ )
        {
            const auto w = quad_weights[q];

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    const auto grad_i  = J_inv_transposed * grad_shape( i, quad_points[q] );

                    contrib[wedge]( i ) += w * streamline_diffusivity[wedge] * vel_interp[wedge][q].dot( grad_i ) *
                                           f_interp[wedge][q] * det;
                }
            }
        }

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, contrib );
    }
};

static_assert( linalg::LinearFormLike< SUPGRHS< double > > );

} // namespace terra::fe::wedge::linearforms::shell