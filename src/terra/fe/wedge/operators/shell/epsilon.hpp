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

template < typename ScalarT, int VecDim = 3 >
class Epsilon
{
  public:
    using SrcVectorType = linalg::VectorW1Vec< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorW1Vec< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >    grid_;
    grid::Grid2DDataScalar< ScalarT >    radii_;
    grid::Grid4DDataScalar< ScalarType > k_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

  public:
    Epsilon(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        const grid::Grid4DDataScalar< ScalarT >& k,
        bool                                     treat_boundary,
        bool                                     diagonal,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , k_( k )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "vector_laplace_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "VectorLaplace: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "VectorLaplace: src/dst mismatch" );
        }

        util::Timer timer_kernel( "vector_laplace_kernel" );
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "vector_laplace_comm" );

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

        dense::Vec< ScalarT, 6 > k_local_hex[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

        // FE dimensions: velocity coupling components of epsilon operator

        for ( int dimi = 0; dimi < 3; ++dimi )
        {
            for ( int dimj = 0; dimj < 3; ++dimj )
            {
                if ( diagonal_ and dimi != dimj )
                    continue;

                ScalarType src_local_hex[8] = { 0 };
                ScalarType dst_local_hex[8] = { 0 };

                constexpr int hex_offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
                constexpr int hex_offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
                constexpr int hex_offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
                for ( int i = 0; i < 8; i++ )
                {
                    src_local_hex[i] = src_(
                        local_subdomain_id,
                        x_cell + hex_offset_x[i],
                        y_cell + hex_offset_y[i],
                        r_cell + hex_offset_r[i],
                        dimi );
                }

                // spatial dimensions: quadrature points and wedge
                for ( int q = 0; q < num_quad_points; q++ )
                {
                    const auto quad_weight = quad_weights[q];

                    for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                    {
                        dense::Mat< ScalarT, VecDim, VecDim > J =
                            jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                        const auto                                  det              = J.det();
                        const auto                                  abs_det          = Kokkos::abs( det );
                        const dense::Mat< ScalarT, VecDim, VecDim > J_inv_transposed = J.inv_transposed( det );
                        ScalarType                                  k_eval           = 0.0;
                        for ( int k = 0; k < num_nodes_per_wedge; k++ )
                        {
                            k_eval += shape( k, quad_points[q] ) * k_local_hex[wedge]( k );
                        }

                        dense::Mat< ScalarT, VecDim, VecDim > sym_grad_i[num_nodes_per_wedge];
                        dense::Mat< ScalarT, VecDim, VecDim > sym_grad_j[num_nodes_per_wedge];
                        for ( int k = 0; k < num_nodes_per_wedge; k++ )
                        {
                            dense::Mat< ScalarT, VecDim, VecDim > grad_i =
                                J_inv_transposed * dense::Mat< ScalarT, VecDim, VecDim >::from_single_col_vec(
                                                       grad_shape( k, quad_points[q] ), dimi );
                            sym_grad_i[k] = ( grad_i + grad_i.transposed() ) * 0.5;

                            dense::Mat< ScalarT, VecDim, VecDim > grad_j =
                                J_inv_transposed * dense::Mat< ScalarT, VecDim, VecDim >::from_single_col_vec(
                                                       grad_shape( k, quad_points[q] ), dimj );
                            sym_grad_j[k] = ( grad_j + grad_j.transposed() ) * 0.5;
                        }

                        if ( diagonal_ )
                        {
                            diagonal(
                                src_local_hex,
                                dst_local_hex,
                                k_eval,
                                wedge,
                                quad_weight,
                                abs_det,
                                sym_grad_i,
                                sym_grad_j,
                                dimi,
                                dimj );
                        }
                        else if ( treat_boundary_ && r_cell == 0 )
                        {
                            // Bottom boundary dirichlet
                            dirichlet_cmb(
                                src_local_hex,
                                dst_local_hex,
                                k_eval,
                                wedge,
                                quad_weight,
                                abs_det,
                                sym_grad_i,
                                sym_grad_j,
                                dimi,
                                dimj );
                        }
                        else if ( treat_boundary_ && r_cell + 1 == radii_.extent( 1 ) - 1 )
                        {
                            // Top boundary dirichlet
                            dirichlet_surface(
                                src_local_hex,
                                dst_local_hex,
                                k_eval,
                                wedge,
                                quad_weight,
                                abs_det,
                                sym_grad_i,
                                sym_grad_j,
                                dimi,
                                dimj );
                        }
                        else
                        {
                            neumann(
                                src_local_hex,
                                dst_local_hex,
                                k_eval,
                                wedge,
                                quad_weight,
                                abs_det,
                                sym_grad_i,
                                sym_grad_j,
                                dimi,
                                dimj );
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
                            local_subdomain_id,
                            x_cell + hex_offset_x[i],
                            y_cell + hex_offset_y[i],
                            r_cell + hex_offset_r[i],
                            dimj ),
                        dst_local_hex[i] );
                }
            }
        }
    }

    KOKKOS_INLINE_FUNCTION void neumann(
        ScalarType                      src_local_hex[8],
        ScalarType                      dst_local_hex[8],
        const ScalarType                k_eval,
        const int                       wedge,
        const ScalarType                quad_weight,
        const ScalarType                abs_det,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_i,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_j,
        const int                       dimi,
        const int                       dimj ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        dense::Mat< ScalarType, 3, 3 > grad_u;

        grad_u.fill( 0.0 );
        for ( int j = 0; j < num_nodes_per_wedge; j++ )
        {
            grad_u = grad_u + sym_grad_i[j] *
                                  src_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]];
        }

        // 4. Add the test function contributions.
        for ( int i = 0; i < num_nodes_per_wedge; i++ )
        {
            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                2 * quad_weight * k_eval * ( sym_grad_j[i] ).double_contract( grad_u ) * abs_det;
        }
    }

    KOKKOS_INLINE_FUNCTION void dirichlet_cmb(
        ScalarType                      src_local_hex[8],
        ScalarType                      dst_local_hex[8],
        const ScalarType                k_eval,
        const int                       wedge,
        const ScalarType                quad_weight,
        const ScalarType                abs_det,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_i,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_j,
        const int                       dimi,
        const int                       dimj ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        dense::Mat< ScalarType, 3, 3 > grad_u;
        grad_u.fill( 0.0 );
        for ( int j = 3; j < num_nodes_per_wedge; j++ )
        {
            grad_u = grad_u + sym_grad_i[j] *
                                  src_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]];
        }

        // 4. Add the test function contributions.
        for ( int i = 3; i < num_nodes_per_wedge; i++ )
        {
            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                2 * quad_weight * k_eval * ( sym_grad_j[i] ).double_contract( grad_u ) * abs_det;
        }

        // Diagonal for top part
        if ( dimi == dimj )
        {
            for ( int i = 0; i < 3; i++ )
            {
                const auto grad_u_diag =
                    sym_grad_i[i] * src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];

                dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                    2 * quad_weight * k_eval * ( sym_grad_j[i] ).double_contract( grad_u_diag ) * abs_det;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION void dirichlet_surface(
        ScalarType                      src_local_hex[8],
        ScalarType                      dst_local_hex[8],
        const ScalarType                k_eval,
        const int                       wedge,
        const ScalarType                quad_weight,
        const ScalarType                abs_det,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_i,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_j,
        const int                       dimi,
        const int                       dimj ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        dense::Mat< ScalarType, 3, 3 > grad_u;

        grad_u.fill( 0.0 );
        for ( int j = 0; j < 3; j++ )
        {
            grad_u = grad_u + sym_grad_i[j] *
                                  src_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]];
        }

        // 4. Add the test function contributions.
        for ( int i = 0; i < 3; i++ )
        {
            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                2 * quad_weight * k_eval * ( sym_grad_j[i] ).double_contract( grad_u ) * abs_det;
        }

        // Diagonal for top part
        if ( dimi == dimj )
        {
            for ( int i = 3; i < num_nodes_per_wedge; i++ )
            {
                const auto grad_u_diag =
                    sym_grad_i[i] * src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];

                dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                    2 * quad_weight * k_eval * ( sym_grad_j[i] ).double_contract( grad_u_diag ) * abs_det;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION void diagonal(
        ScalarType                      src_local_hex[8],
        ScalarType                      dst_local_hex[8],
        const ScalarType                k_eval,
        const int                       wedge,
        const ScalarType                quad_weight,
        const ScalarType                abs_det,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_i,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_j,
        const int                       dimi,
        const int                       dimj ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        for ( int i = 0; i < num_nodes_per_wedge; i++ )
        {
            const auto grad_u_diag =
                sym_grad_i[i] * src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];

            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                2 * quad_weight * k_eval * ( sym_grad_j[i] ).double_contract( grad_u_diag ) * abs_det;
        }
    }
};

static_assert( linalg::OperatorLike< Epsilon< float > > );
static_assert( linalg::OperatorLike< Epsilon< double > > );

} // namespace terra::fe::wedge::operators::shell
