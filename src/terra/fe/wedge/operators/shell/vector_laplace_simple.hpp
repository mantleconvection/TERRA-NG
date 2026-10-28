
#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_w1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, int VecDim = 3 >
class VectorLaplaceSimple
{
  public:
    using SrcVectorType = linalg::VectorW1Vec< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorW1Vec< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

  public:
    VectorLaplaceSimple(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        bool                                     treat_boundary,
        bool                                     diagonal,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

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
        // First all the r-independent stuff.
        // Gather surface points for each wedge.

        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Compute lateral part of Jacobian.

        constexpr auto num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        dense::Mat< ScalarT, 3, 3 > jac_lat_inv_t[num_wedges_per_hex_cell][num_quad_points] = {};
        ScalarT                     det_jac_lat[num_wedges_per_hex_cell][num_quad_points]   = {};

        jacobian_lat_inverse_transposed_and_determinant( jac_lat_inv_t, det_jac_lat, wedge_phy_surf, quad_points );

        dense::Vec< ScalarT, 3 > g_rad[num_wedges_per_hex_cell][num_nodes_per_wedge][num_quad_points] = {};
        dense::Vec< ScalarT, 3 > g_lat[num_wedges_per_hex_cell][num_nodes_per_wedge][num_quad_points] = {};

        lateral_parts_of_grad_phi( g_rad, g_lat, jac_lat_inv_t, quad_points );

        // Only now we introduce radially dependent terms.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // For now, compute the local element matrix. We'll improve that later.
        dense::Mat< ScalarT, 6, 6 > A[num_wedges_per_hex_cell] = {};

        // TODO: this can be absorbed into g_lat.
        // TODO: ALSO we can sometimes avoid division if we pull the r^2 and grad_r out of the determinant and replace
        //       the prefactors for the g_lat and g_rad but this is very form-specific.
        const ScalarT grad_r     = grad_forward_map_rad( r_1, r_2 );
        const ScalarT grad_r_inv = 1.0 / grad_r;

        for ( int q = 0; q < num_quad_points; q++ )
        {
            // TODO: We could precompute that per quadrature point and store in a View globally to avoid the division.
            const ScalarT r     = forward_map_rad( r_1, r_2, quad_points[q]( 2 ) );
            const ScalarT r_inv = 1.0 / r;

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto grad_i = grad_shape_full( g_rad, g_lat, r_inv, grad_r_inv, wedge, i, q );
                        const auto grad_j = grad_shape_full( g_rad, g_lat, r_inv, grad_r_inv, wedge, j, q );

                        const auto det = det_full( det_jac_lat, r, grad_r, wedge, q );

                        A[wedge]( i, j ) += quad_weights[q] * ( grad_i.dot( grad_j ) * det );
                    }
                }
            }
        }

        if ( treat_boundary_ )
        {
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                dense::Mat< ScalarT, 6, 6 > boundary_mask;
                boundary_mask.fill( 1.0 );
                if ( r_cell == 0 )
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

                if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
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

                A[wedge].hadamard_product( boundary_mask );
            }
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        for ( int d = 0; d < VecDim; d++ )
        {
            dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
            extract_local_wedge_vector_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, d, src_ );

            dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            atomically_add_local_wedge_vector_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, d, dst );
        }
    }
};

static_assert( linalg::OperatorLike< VectorLaplaceSimple< float, 3 > > );
static_assert( linalg::OperatorLike< VectorLaplaceSimple< double, 3 > > );

} // namespace terra::fe::wedge::operators::shell