#pragma once

#include <cstdlib>

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

template < typename ScalarT, int VecDim = 3 >
class EpsilonDivDivSimpleStab
{
  public:
    using SrcVectorType           = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType           = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType              = ScalarT;

  private:
    bool storeLMatrices_ =
        false; // set to let apply_impl() know, that it should store the local matrices after assembling them
    bool                    single_quadpoint_ = false;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >    grid_;
    grid::Grid2DDataScalar< ScalarT >    radii_;
    grid::Grid4DDataScalar< ScalarType > k_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;


    bool treat_boundary_;
    ScalarT stab_c_ = 0.3;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

  public:
    EpsilonDivDivSimpleStab(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >  grid,
        const grid::Grid2DDataScalar< ScalarT >  radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const grid::Grid4DDataScalar< ScalarT >  k,
        bool                                     treat_boundary,
        bool                                     diagonal,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii ) , mask_( mask )
    , k_( k )
    , treat_boundary_( treat_boundary )
    , stab_c_( []() {
          const char* e = std::getenv( "STAB_C" );
          return e ? static_cast< ScalarT >( std::atof( e ) ) : ScalarT( 0.3 );
      }() )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }

    /// @brief Getter for domain member
    const grid::shell::DistributedDomain& get_domain() const { return domain_; }

    /// @brief Getter for radii member
    grid::Grid2DDataScalar< ScalarT > get_radii() const { return radii_; }

    /// @brief Getter for grid member
    grid::Grid3DDataVec< ScalarT, 3 > get_grid() const { return grid_; }

   

    /// @brief S/Getter for diagonal member
    void set_diagonal( bool v ) { diagonal_ = v; }

   

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
        // Compute the local element matrix.
        dense::Mat< ScalarT, 18, 18 > A[num_wedges_per_hex_cell] = {};

     
            // Gather surface points for each wedge.
            dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

            // Gather wedge radii.
            const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
            const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

            // Quadrature points.
            constexpr auto num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;

            dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
            ScalarT                  quad_weights[num_quad_points];

            quadrature::quad_felippa_1x1_quad_points( quad_points );
            quadrature::quad_felippa_1x1_quad_weights( quad_weights );

            dense::Vec< ScalarT, 6 > k[num_wedges_per_hex_cell];
            extract_local_wedge_scalar_coefficients( k, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

            // FE dimensions: velocity coupling components of epsilon operator
            for ( int dimi = 0; dimi < 3; ++dimi )
            {
                for ( int dimj = 0; dimj < 3; ++dimj )
                {
                    if ( diagonal_ and dimi != dimj )
                        continue;

                    // spatial dimensions: quadrature points and wedge
                    for ( int q = 0; q < num_quad_points; q++ )
                    {
                        const auto w = quad_weights[q];

                        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                        {
                            dense::Mat< ScalarT, 3, 3 > J   = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                            const auto                  det = J.det();
                            const auto                  abs_det          = Kokkos::abs( det );
                            const auto                  J_inv_transposed = J.inv_transposed( det );
                            ScalarType                  k_eval           = 0.0;
                            for ( int j = 0; j < num_nodes_per_wedge; j++ )
                            {
                                k_eval += shape( j, quad_points[q] ) * k[wedge]( j );
                            }
                            // FE dimensions: local DoFs/associated shape functions
                            for ( int i = 0; i < num_nodes_per_wedge; i++ )
                            {
                                // basis functions are vectors with VecDim components -> build tensorial gradients
                                dense::Mat< ScalarT, 3, 3 > grad_i =
                                    J_inv_transposed * dense::Mat< ScalarT, VecDim, VecDim >::from_single_col_vec(
                                                           grad_shape( i, quad_points[q] ), dimi );
                                dense::Mat< ScalarT, 3, 3 > sym_grad_i = ( grad_i + grad_i.transposed() );
                                ScalarType                  div_i      = grad_i( dimi, dimi );

                                for ( int j = 0; j < num_nodes_per_wedge; j++ )
                                {
                                    dense::Mat< ScalarT, 3, 3 > grad_j =
                                        J_inv_transposed * dense::Mat< ScalarT, VecDim, VecDim >::from_single_col_vec(
                                                               grad_shape( j, quad_points[q] ), dimj );

                                    dense::Mat< ScalarT, 3, 3 > sym_grad_j = ( grad_j + grad_j.transposed() );

                                    ScalarType div_j = grad_j( dimj, dimj );

                                    A[wedge]( i + num_nodes_per_wedge * dimi, j + num_nodes_per_wedge * dimj ) +=
                                        w * k_eval * abs_det *
                                        ( 0.5 * sym_grad_i.double_contract( sym_grad_j ) - 2.0 / 3.0 * div_i * div_j );
                                }
                            }
                        }
                    }
                }
            }

            // ---- hourglass stabilization (rank-6 gamma correction) --------
            // gamma_1 = nodal pattern of (xi-1/3)*zeta, gamma_2 of (eta-1/3)*zeta,
            // applied per displacement component. eps is scaled from the mean
            // diagonal of the assembled 1-point local matrix.
            {
                constexpr ScalarT g1[6] = { 1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0,
                                            -1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0 };
                constexpr ScalarT g2[6] = { 1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0,
                                            -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0 };
                const ScalarT     norm2_g = 4.0 / 3.0;  // |g|^2 = 6*(1/9)+2*... = 4/3

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    ScalarT diag_mean = 0.0;
                    for ( int i = 0; i < 18; i++ )
                        diag_mean += A[wedge]( i, i );
                    diag_mean /= 18.0;
                    const ScalarT eps = stab_c_ * diag_mean / norm2_g;

                    for ( int d = 0; d < 3; d++ )
                    {
                        for ( int i = 0; i < num_nodes_per_wedge; i++ )
                        {
                            for ( int j = 0; j < num_nodes_per_wedge; j++ )
                            {
                                A[wedge]( i + num_nodes_per_wedge * d, j + num_nodes_per_wedge * d ) +=
                                    eps * ( g1[i] * g1[j] + g2[i] * g2[j] );
                            }
                        }
                    }
                }
            }

            if ( treat_boundary_ )
            {
                 const bool at_bot_boundary =
            util::has_flag( mask_( local_subdomain_id, x_cell, y_cell, r_cell ), grid::shell::ShellBoundaryFlag::CMB );
        const bool at_top_boundary = util::has_flag(
            mask_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), grid::shell::ShellBoundaryFlag::SURFACE );
              for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                        {
                dense::Mat< ScalarT, 18, 18 > boundary_mask;
                            boundary_mask.fill( 1.0 );

                for ( int dimi = 0; dimi < 3; ++dimi )
                {
                    for ( int dimj = 0; dimj < 3; ++dimj )
                    {
                        
                         
                            if ( at_bot_boundary )
                            {
                                // Inner boundary (CMB).
                                for ( int i = 0; i < 6; i++ )
                                {
                                    for ( int j = 0; j < 6; j++ )
                                    {
                                        if ( ( dimi == dimj && i != j && ( i < 3 || j < 3 ) ) or
                                             ( dimi != dimj && ( i < 3 || j < 3 ) ) )
                                        {
                                            boundary_mask(
                                                i + num_nodes_per_wedge * dimi, j + num_nodes_per_wedge * dimj ) = 0.0;
                                        }
                                    }
                                }
                            }

                            if ( at_top_boundary )
                            {
                                // Outer boundary (surface).
                                for ( int i = 0; i < 6; i++ )
                                {
                                    for ( int j = 0; j < 6; j++ )
                                    {
                                        if ( ( dimi == dimj && i != j && ( i >= 3 || j >= 3 ) ) or
                                             ( dimi != dimj && ( i >= 3 || j >= 3 ) ) )
                                        {
                                            boundary_mask(
                                                i + num_nodes_per_wedge * dimi, j + num_nodes_per_wedge * dimj ) = 0.0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                

                            A[wedge].hadamard_product( boundary_mask );
                }
            }
       

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

       
            dense::Vec< ScalarT, 18 > src[num_wedges_per_hex_cell];
            for ( int dimj = 0; dimj < 3; dimj++ )
            {
                dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
                extract_local_wedge_vector_coefficients(
                    src_d, local_subdomain_id, x_cell, y_cell, r_cell, dimj, src_ );

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        src[wedge]( dimj * num_nodes_per_wedge + i ) = src_d[wedge]( i );
                    }
                }
            }
            //extract_local_wedge_vector_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, dimj, src_ );

            dense::Vec< ScalarT, 18 > dst[num_wedges_per_hex_cell];

            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            //atomically_add_local_wedge_vector_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst );
            for ( int dimi = 0; dimi < 3; dimi++ )
            {
                dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
                dst_d[0] = dst[0].template slice< 6 >( dimi * num_nodes_per_wedge );
                dst_d[1] = dst[1].template slice< 6 >( dimi * num_nodes_per_wedge );

                atomically_add_local_wedge_vector_coefficients(
                    dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst_d );
            }
        
    }
};

static_assert( linalg::OperatorLike< EpsilonDivDivSimpleStab< float > > );
static_assert( linalg::OperatorLike< EpsilonDivDivSimpleStab< double > > );

} // namespace terra::fe::wedge::operators::shell
