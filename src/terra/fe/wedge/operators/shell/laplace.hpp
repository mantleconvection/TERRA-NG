
#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_w1.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class Laplace
{
  public:
    using SrcVectorType = linalg::VectorW1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorW1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    Laplace(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        bool                                                            treat_boundary,
        bool                                                            diagonal,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , mask_( mask )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "laplace_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

        util::Timer timer_kernel( "laplace_kernel" );
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "laplace_comm" );
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

        // Compute the local element matrix.

        ScalarType src_local_hex[8] = { 0 };
        ScalarType dst_local_hex[8] = { 0 };

        for ( int i = 0; i < 8; i++ )
        {
            constexpr int hex_offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
            constexpr int hex_offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
            constexpr int hex_offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

            src_local_hex[i] = src_(
                local_subdomain_id, x_cell + hex_offset_x[i], y_cell + hex_offset_y[i], r_cell + hex_offset_r[i] );
        }

        const bool at_bot_boundary =
            util::has_flag( mask_( local_subdomain_id, x_cell, y_cell, r_cell ), grid::shell::ShellBoundaryFlag::CMB );
        const bool at_top_boundary = util::has_flag(
            mask_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), grid::shell::ShellBoundaryFlag::SURFACE );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                const auto quad_point  = quad_points[q];
                const auto quad_weight = quad_weights[q];

                // 1. Compute Jacobian and inverse at this quadrature point.

                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = J.det();
                const auto abs_det          = Kokkos::abs( det );
                const auto J_inv_transposed = J.inv_transposed( det );

                // 2. Compute physical gradients for all nodes at this quadrature point.
                dense::Vec< ScalarType, 3 > grad_phy[num_nodes_per_wedge];
                for ( int k = 0; k < num_nodes_per_wedge; k++ )
                {
                    grad_phy[k] = J_inv_transposed * grad_shape( k, quad_point );
                }

                if ( diagonal_ )
                {
                    diagonal( src_local_hex, dst_local_hex, wedge, quad_weight, abs_det, grad_phy );
                }
                else if ( treat_boundary_ && at_bot_boundary )
                {
                    // Bottom boundary dirichlet
                    dirichlet_bot( src_local_hex, dst_local_hex, wedge, quad_weight, abs_det, grad_phy );
                }
                else if ( treat_boundary_ && at_top_boundary )
                {
                    // Top boundary dirichlet
                    dirichlet_top( src_local_hex, dst_local_hex, wedge, quad_weight, abs_det, grad_phy );
                }
                else
                {
                    neumann( src_local_hex, dst_local_hex, wedge, quad_weight, abs_det, grad_phy );
                }
            }
        }

        for ( int i = 0; i < 8; i++ )
        {
            constexpr int hex_offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
            constexpr int hex_offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
            constexpr int hex_offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

            Kokkos::atomic_add(
                &dst_(
                    local_subdomain_id, x_cell + hex_offset_x[i], y_cell + hex_offset_y[i], r_cell + hex_offset_r[i] ),
                dst_local_hex[i] );
        }
    }

    KOKKOS_INLINE_FUNCTION void neumann(
        ScalarType*                        src_local_hex,
        ScalarType*                        dst_local_hex,
        const int                          wedge,
        const ScalarType                   quad_weight,
        const ScalarType                   abs_det,
        const dense::Vec< ScalarType, 3 >* grad_phy ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        dense::Vec< ScalarType, 3 > grad_u;
        grad_u.fill( 0.0 );
        for ( int j = 0; j < num_nodes_per_wedge; j++ )
        {
            grad_u = grad_u +
                     src_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]] * grad_phy[j];
        }

        // 4. Add the test function contributions.
        for ( int i = 0; i < num_nodes_per_wedge; i++ )
        {
            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                quad_weight * grad_phy[i].dot( grad_u ) * abs_det;
        }
    }

    KOKKOS_INLINE_FUNCTION void dirichlet_bot(
        ScalarType*                        src_local_hex,
        ScalarType*                        dst_local_hex,
        const int                          wedge,
        const ScalarType                   quad_weight,
        const ScalarType                   abs_det,
        const dense::Vec< ScalarType, 3 >* grad_phy ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        dense::Vec< ScalarType, 3 > grad_u;
        grad_u.fill( 0.0 );
        for ( int j = 3; j < num_nodes_per_wedge; j++ )
        {
            grad_u = grad_u +
                     src_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]] * grad_phy[j];
        }

        // 4. Add the test function contributions.
        for ( int i = 3; i < num_nodes_per_wedge; i++ )
        {
            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                quad_weight * grad_phy[i].dot( grad_u ) * abs_det;
        }

        // Diagonal for top part
        for ( int i = 0; i < 3; i++ )
        {
            const auto grad_u_diag =
                src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] * grad_phy[i];

            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                quad_weight * grad_phy[i].dot( grad_u_diag ) * abs_det;
        }
    }

    KOKKOS_INLINE_FUNCTION void dirichlet_top(
        ScalarType*                        src_local_hex,
        ScalarType*                        dst_local_hex,
        const int                          wedge,
        const ScalarType                   quad_weight,
        const ScalarType                   abs_det,
        const dense::Vec< ScalarType, 3 >* grad_phy ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        dense::Vec< ScalarType, 3 > grad_u;
        grad_u.fill( 0.0 );
        for ( int j = 0; j < 3; j++ )
        {
            grad_u = grad_u +
                     src_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]] * grad_phy[j];
        }

        // 4. Add the test function contributions.
        for ( int i = 0; i < 3; i++ )
        {
            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                quad_weight * grad_phy[i].dot( grad_u ) * abs_det;
        }

        // Diagonal for top part
        for ( int i = 3; i < num_nodes_per_wedge; i++ )
        {
            const auto grad_u_diag =
                src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] * grad_phy[i];

            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                quad_weight * grad_phy[i].dot( grad_u_diag ) * abs_det;
        }
    }

    KOKKOS_INLINE_FUNCTION void diagonal(
        ScalarType*                        src_local_hex,
        ScalarType*                        dst_local_hex,
        const int                          wedge,
        const ScalarType                   quad_weight,
        const ScalarType                   abs_det,
        const dense::Vec< ScalarType, 3 >* grad_phy ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        // 4. Add the test function contributions.
        for ( int i = 0; i < num_nodes_per_wedge; i++ )
        {
            const auto grad_u =
                src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] * grad_phy[i];

            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                quad_weight * grad_phy[i].dot( grad_u ) * abs_det;
        }
    }
};

static_assert( linalg::OperatorLike< Laplace< float > > );
static_assert( linalg::OperatorLike< Laplace< double > > );

} // namespace terra::fe::wedge::operators::shell