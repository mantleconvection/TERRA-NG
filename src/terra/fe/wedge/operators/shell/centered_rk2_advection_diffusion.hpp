
#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell {

/// \brief Matrix-free Galerkin **rate** operator for the unsteady advection-diffusion equation.
///
/// Computes
/// \f[
///    \mathrm{dst} = -\bigl( K_\text{adv} + K_\text{diff} \bigr)\,\mathrm{src}
///                 = -\!\!\int_\Omega \phi_i \,(\mathbf{u}\!\cdot\!\nabla\phi_j)\,\mathrm{d}x \,T_j
///                   -\!\!\int_\Omega \kappa\,\nabla\phi_i\!\cdot\!\nabla\phi_j\,\mathrm{d}x \,T_j .
/// \f]
/// In the lumped-mass time-stepping picture this is the (assembled, additively reduced)
/// right-hand side of \f$ M_\text{lumped}\,\dot{T} = -K T + F \f$ from which the actual
/// nodal rate \f$\dot{T}_i\f$ is recovered by pointwise division by the lumped mass diagonal.
///
/// This is the central building block of the explicit centred Runge-Kutta-2 energy integrator
/// that mirrors TERRA's `advect`/`advance` pair (TERRA-group/src/code/energy.f:6-181 and
/// convct.F:207-244). Unlike `UnsteadyAdvectionDiffusionSUPG`, there is **no** SUPG
/// stabilisation, **no** mass term in the bilinear form, **no** time-step or Dirichlet
/// row treatment in the kernel, and **no** lumping flag — the operator is purely the
/// stiffness contribution \f$K\f$ with the leading minus sign baked in. Boundary
/// conditions are handled outside, by overwriting Dirichlet nodes after each RK substage,
/// matching TERRA's strategy of zeroing `dhdt` at top/bottom rows.
///
/// The advection term uses the **centred** Galerkin form
/// \f$\int \phi_i (\mathbf{u}\cdot\nabla\phi_j)\f$ — no upwinding, no limiter. This is
/// the Q1 analogue of TERRA's face-centred update
/// \f$\tfrac{1}{2}(T_i + T_j)\cdot\mathbf{u}\!\cdot\!\mathbf{n}\f$ at energy.f:73.
template < typename ScalarT, int VelocityVecDim = 3 >
class Q1CenteredAdvDiffRateOperator
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    linalg::VectorQ1Vec< ScalarT, VelocityVecDim > velocity_;

    ScalarT diffusivity_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType >              src_;
    grid::Grid4DDataScalar< ScalarType >              dst_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

  public:
    Q1CenteredAdvDiffRateOperator(
        const grid::shell::DistributedDomain&                 domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&              grid,
        const grid::Grid2DDataScalar< ScalarT >&              radii,
        const linalg::VectorQ1Vec< ScalarT, VelocityVecDim >& velocity,
        const ScalarT                                         diffusivity,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , velocity_( velocity )
    , diffusivity_( diffusivity )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    ScalarT&       diffusivity() { return diffusivity_; }
    const ScalarT& diffusivity() const { return diffusivity_; }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "centered_rk2_rate_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_      = src.grid_data();
        dst_      = dst.grid_data();
        vel_grid_ = velocity_.grid_data();

        util::Timer timer_kernel( "centered_rk2_rate_kernel" );
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "centered_rk2_rate_comm" );

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

        // Interpolate velocity to quadrature points using Q1 wedge basis.
        // (Same pattern as UnsteadyAdvectionDiffusionSUPG kernel.)
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

        // Assemble the local stiffness matrix K_e = K_adv_e + K_diff_e per wedge.
        // (No SUPG term, no mass term, no time step.)
        dense::Mat< ScalarT, 6, 6 > K[num_wedges_per_hex_cell] = {};

        for ( int q = 0; q < num_quad_points; q++ )
        {
            const auto w = quad_weights[q];

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();

                const auto vel = vel_interp[wedge][q];

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    const auto grad_i  = J_inv_transposed * grad_shape( i, quad_points[q] );

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto grad_j = J_inv_transposed * grad_shape( j, quad_points[q] );

                        // Centred Galerkin advection: int phi_i (u . grad phi_j)
                        const auto advection = ( vel.dot( grad_j ) ) * shape_i;
                        // Diffusion: int kappa grad phi_i . grad phi_j
                        const auto diffusion = diffusivity_ * ( grad_i ).dot( grad_j );

                        K[wedge]( i, j ) += w * ( advection + diffusion ) * det;
                    }
                }
            }
        }

        // Local source vector (T_in restricted to this hex cell).
        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

        // Local rate contribution (rate form: dst = -K * src).
        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];
        dst[0] = K[0] * src[0];
        dst[1] = K[1] * src[1];
        for ( int i = 0; i < 6; i++ )
        {
            dst[0]( i ) = -dst[0]( i );
            dst[1]( i ) = -dst[1]( i );
        }

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
    }
};

static_assert( linalg::OperatorLike< Q1CenteredAdvDiffRateOperator< double > > );

} // namespace terra::fe::wedge::operators::shell
