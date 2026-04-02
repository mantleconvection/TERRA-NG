
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

template < typename ScalarT >
KOKKOS_INLINE_FUNCTION ScalarT
    supg_tau( const ScalarT vel_norm, const ScalarT kappa, const ScalarT h, const ScalarT Pe_tol = 1e-8 )
{
    const ScalarT kappa_min     = ScalarT( 1e-8 );
    const ScalarT kappa_virtual = Kokkos::max( kappa, kappa_min );

    // Guarding against zero velocity.
    const ScalarT eps_vel = ScalarT( 1e-12 );

    if ( vel_norm <= eps_vel )
    {
        return 0.0;
    }

    // For small Peclet numbers (diffusion-dominated flow) set tau to 0 (no stabilization).
    const ScalarT Pe = vel_norm * h / ( 2.0 * kappa_virtual );

    if ( Pe <= Pe_tol )
    {
        return 0.0;
    }

    // Clamp tau to avoid huge values
    const ScalarT tau_max = ScalarT( 10.0 ) * h / Kokkos::max( vel_norm, eps_vel );

    // Finally, compute tau.
    // Note: coth(Pe) = 1/tanh(Pe)
    const ScalarT coth_term = ScalarT( 1.0 ) / Kokkos::tanh( Pe ) - ScalarT( 1.0 ) / Pe;
    const ScalarT tau       = ( h / ( 2.0 * vel_norm ) ) * coth_term;

    return ( tau > tau_max ) ? tau_max : tau;
    // return tau;
}

/// \brief Linear operator for a method-of-lines discretization of the unsteady advection-diffusion equation with SUPG
/// (streamline upwind Petrov-Galerkin) stabilization.
///
/// # Continuous problem
///
/// The unsteady advection-diffusion equation is given by
/// \f[
///   \frac{\partial}{\partial t}T + \mathbf{u} \cdot \nabla T - \nabla \cdot (\kappa \nabla T) = f
/// \f]
/// where \f$ T \f$ is the scalar temperature solution, \f$ \mathbf{u} \f$ a given velocity field,
/// \f$ f \f$ a given source term, and \f$ \kappa \f$ a given diffusivity function.
///
/// # Space discretization
///
/// \note We assume here that we have \f$ \kappa|_E = \mathrm{const} \f$ on each element \f$E\f$ which simplifies the
/// implementation of the SUPG stabilization for linear finite elements as certain terms drop. Currently, \f$\kappa =
/// \mathrm{const}\f$ globally, but once that is changed, we will need to average on every element it in the kernel or
/// pass it in as an elementwise constant (FE) function.
///
/// We first discretize in space, then in time (method of lines). After discretization in space, we get the system of
/// ODEs in time
/// \f[
///   M \frac{d}{dt}T + (C + K + G)T = F + F_\mathrm{SUPG}
/// \f]
/// where
///
/// Term         | Description  | Bilinear form
/// -------------|--------------|--------------
/// \f$M_{ij}\f$ | mass         | \f$ \int \phi_i \phi_j \f$
/// \f$C_{ij}\f$ | advection    | \f$ \int \phi_i (\mathbf{u} \cdot \nabla \phi_j) \f$
/// \f$K_{ij}\f$ | diffusion    | \f$ \int \nabla \phi_i \cdot (\kappa \nabla \phi_j) \f$
/// \f$G_{ij}\f$ | SUPG adv-adv | \f$ \sum_E \int_E \tau_E (\mathbf{u} \cdot \nabla \phi_i) (\mathbf{u} \cdot \nabla \phi_j) \f$
/// \f$F_i\f$    | forcing      | \f$ \int \phi_i f \f$
/// \f$(F_\mathrm{SUPG})_{i}\f$ | SUPG forcing | \f$ \sum_E \int_E \tau_E (\mathbf{u} \cdot \nabla \phi_i) f \f$.
///
/// \note After brief testing, it seems that in general the term \f$F_\mathrm{SUPG}\f$ does not always improve the
///       computed solutions. It even appears to slightly increase the error sometimes. So setting
///       \f$F_\mathrm{SUPG} = 0\f$ can really be just fine in certain settings.
///
/// # Time discretization
///
/// For the time discretization, we employ implicit BDF schemes. A general formula is
///
/// \f[
///   (\alpha_0 M + \Delta t A) T^{n+1} = - \sum_{j=1}^k \alpha_j M T^{n+1-j} + \Delta t R^{n+1}
/// \f]
///
/// where \f$A = (C + K + G)\f$ and \f$R = F + F_{\mathrm{SUPG}}\f$.
///
/// We recover the common BDF1 (backward or implicit Euler) and BDF2 schemes by choosing:
///
/// Scheme                          | \f$k\f$ | \f$\alpha\f$                         | full equation
/// --------------------------------|---------|--------------------------------------|--------------
/// BDF1 (backward/implicit Euler)  | 1       | \f$[1, -1]\f$                        | \f$(M + \Delta t A) T^{n+1} = M T^{n} + \Delta t R^{n+1}\f$
/// BDF2                            | 2       | \f$[\frac{3}{2}, -2, \frac{1}{2}]\f$ | \f$(\frac{3}{2} M + \Delta t A) T^{n+1} = 2 M T^{n} - \frac{1}{2} M T^{n-1} + \Delta t R^{n+1}\f$
///
/// The RHS term must be built with appropriate other classes/function. This class is only concerned with the
/// matrix-free evaluation of the LHS system matrix
/// \f[
///    \alpha_0 M + \Delta t A.
/// \f]
///
/// The parameters \f$\alpha_0\f$ and \f$\Delta t\f$ are set through the constructor via `mass_scaling` and `dt`
/// respectively.
///
/// # SUPG stabilization
///
/// Several choices for the stabilization parameter \f$ \tau_E \f$ are possible.
/// As it is commonly selected in the literature (see, e.g., John and Knobloch (2006)), we set on element \f$E\f$
/// \f[
///  \tau_E = \frac{h_E}{2 \|\mathbf{u}\|} \left(\coth(\mathrm{Pe_E}) - \frac{1}{\mathrm{Pe_E}}\right)
/// \f]
/// where
/// \f[
///  \mathrm{Pe_E} = \frac{\|\mathbf{u}\|h_E}{2 \kappa}
/// \f]
/// with some precautionary measures to avoid edge cases that would result in division by zero etc.
///
template < typename ScalarT, int VelocityVecDim = 3 >
class UnsteadyAdvectionDiffusionSUPG
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    linalg::VectorQ1Vec< ScalarT, VelocityVecDim > velocity_;

    ScalarT diffusivity_;
    ScalarT dt_;

    bool    treat_boundary_;
    bool    diagonal_;
    ScalarT mass_scaling_;
    bool    lumped_mass_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType >              src_;
    grid::Grid4DDataScalar< ScalarType >              dst_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

  public:
    UnsteadyAdvectionDiffusionSUPG(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const linalg::VectorQ1Vec< ScalarT, VelocityVecDim >&           velocity,
        const ScalarT                                                   diffusivity,
        const ScalarT                                                   dt,
        bool                                                            treat_boundary,
        bool                                                            diagonal     = false,
        ScalarT                                                         mass_scaling = 1.0,
        bool                                                            lumped_mass  = false,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , boundary_mask_( boundary_mask )
    , velocity_( velocity )
    , diffusivity_( diffusivity )
    , dt_( dt )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , mass_scaling_( mass_scaling )
    , lumped_mass_( lumped_mass )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    ScalarT&       dt() { return dt_; }
    const ScalarT& dt() const { return dt_; }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "ad_supg_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_      = src.grid_data();
        dst_      = dst.grid_data();
        vel_grid_ = velocity_.grid_data();

        util::Timer timer_kernel( "ad_supg_kernel" );
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "ad_supg_comm" );

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

        // Let's compute the streamline diffusivity.

        ScalarT streamline_diffusivity[num_wedges_per_hex_cell];

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

                const ScalarT tau_q = supg_tau< ScalarT >( vel_norm_q, diffusivity_, h, 1e-08 );

                // quadrature weight for this point (if you have weights)
                const ScalarT wq = quad_weights[q]; // if not available, use 1.0
                tau_accum += tau_q * wq;
                waccum += wq;
            }

            // final cell/wedge tau: volume-weighted average
            ScalarT tau_cell              = ( waccum > 0.0 ) ? ( tau_accum / waccum ) : 0.0;
            streamline_diffusivity[wedge] = tau_cell;
        }

        // Compute the local element matrix.
        dense::Mat< ScalarT, 6, 6 > A[num_wedges_per_hex_cell] = {};
        dense::Mat< ScalarT, 6, 6 > M[num_wedges_per_hex_cell] = {};

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
                        const auto shape_j = shape( j, quad_points[q] );
                        const auto grad_j  = J_inv_transposed * grad_shape( j, quad_points[q] );

                        const auto mass      = shape_i * shape_j;
                        const auto diffusion = diffusivity_ * ( grad_i ).dot( grad_j );
                        const auto advection = ( vel.dot( grad_j ) ) * shape_i;
                        const auto streamline_diffusion =
                            streamline_diffusivity[wedge] * ( vel.dot( grad_j ) ) * ( vel.dot( grad_i ) );

                        M[wedge]( i, j ) += w * mass_scaling_ * mass * det;
                        A[wedge]( i, j ) += w * dt_ * ( diffusion + advection + streamline_diffusion ) * det;
                    }
                }
            }
        }

        if ( lumped_mass_ )
        {
            dense::Vec< ScalarT, 6 > ones;
            ones.fill( 1.0 );
            M[0] = dense::Mat< ScalarT, 6, 6 >::diagonal_from_vec( M[0] * ones );
            M[1] = dense::Mat< ScalarT, 6, 6 >::diagonal_from_vec( M[1] * ones );
        }

        if ( treat_boundary_ )
        {
            const int at_cmb_boundary = util::has_flag(
                boundary_mask_( local_subdomain_id, x_cell, y_cell, r_cell ), grid::shell::ShellBoundaryFlag::CMB );
            const int at_surface_boundary = util::has_flag(
                boundary_mask_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ),
                grid::shell::ShellBoundaryFlag::SURFACE );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                dense::Mat< ScalarT, 6, 6 > boundary_mask;
                boundary_mask.fill( 1.0 );
                if ( at_cmb_boundary )
                {
                    // Inner boundary (CMB).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i < 3 || j < 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                if ( at_surface_boundary )
                {
                    // Outer boundary (surface).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i >= 3 || j >= 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                M[wedge].hadamard_product( boundary_mask );
                A[wedge].hadamard_product( boundary_mask );
            }
        }

        if ( diagonal_ )
        {
            M[0] = M[0].diagonal();
            M[1] = M[1].diagonal();
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        dst[0] = ( M[0] + A[0] ) * src[0];
        dst[1] = ( M[1] + A[1] ) * src[1];

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
    }
};

static_assert( linalg::OperatorLike< UnsteadyAdvectionDiffusionSUPG< double > > );

} // namespace terra::fe::wedge::operators::shell