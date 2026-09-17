
#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, typename CoefficientT = grid::Grid4DDataScalar< ScalarT > >
class KMass
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    CoefficientT k_;

    bool diagonal_;
    bool lumped_diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    KMass(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        CoefficientT                             k,
        const bool                               diagonal,
        const bool                               lumped_diagonal     = false,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , k_( k )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    /// @brief S/Getter for diagonal member
    void set_diagonal( bool v ) { diagonal_ = v; }

    /// @brief S/Getter for lumped diagonal member
    void set_lumped_diagonal( bool v ) { lumped_diagonal_ = v; }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );

        Kokkos::fence();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            std::vector< std::unique_ptr< std::array< int, 11 > > > expected_recvs_metadata;
            std::vector< std::unique_ptr< MPI_Request > >           expected_recvs_requests;

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

        ScalarT det_jac_lat[num_wedges_per_hex_cell][num_quad_points] = {};

        jacobian_lat_determinant( det_jac_lat, wedge_phy_surf, quad_points );

        // Only now we introduce radially dependent terms.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // For now, compute the local element matrix. We'll improve that later.
        dense::Mat< ScalarT, 6, 6 > A[num_wedges_per_hex_cell] = {};

        const ScalarT grad_r = grad_forward_map_rad( r_1, r_2 );

        dense::Vec< ScalarT, 6 > k[num_wedges_per_hex_cell];

        if constexpr ( std::is_same_v< CoefficientT, grid::Grid4DDataScalar< ScalarType > > )
        {
            extract_local_wedge_scalar_coefficients( k, local_subdomain_id, x_cell, y_cell, r_cell, k_ );
        }

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                const ScalarT r = forward_map_rad( r_1, r_2, quad_points[q]( 2 ) );

                ScalarType k_eval = 0.0;

                if constexpr ( std::is_same_v< CoefficientT, grid::Grid4DDataScalar< ScalarType > > )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        k_eval += shape( j, quad_points[q] ) * k[wedge]( j );
                    }
                }
                else
                {
                    k_eval = k_( local_subdomain_id, x_cell, y_cell, r_cell, wedge, quad_points[q] );
                }

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const ScalarT shape_i = shape_lat( i, quad_points[q] ) * shape_rad( i, quad_points[q] );
                        const ScalarT shape_j = shape_lat( j, quad_points[q] ) * shape_rad( j, quad_points[q] );

                        A[wedge]( i, j ) +=
                            quad_weights[q] * k_eval * ( shape_i * shape_j * r * r * grad_r * det_jac_lat[wedge][q] );
                    }
                }
            }
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }
        else if ( lumped_diagonal_ )
        {
            dense::Vec< ScalarT, 6 > ones;
            ones.fill( 1.0 );
            A[0] = dense::Mat< ScalarT, 6, 6 >::diagonal_from_vec( A[0] * ones );
            A[1] = dense::Mat< ScalarT, 6, 6 >::diagonal_from_vec( A[1] * ones );
        }

        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
    }
};

static_assert( linalg::OperatorLike< KMass< float > > );
static_assert( linalg::OperatorLike< KMass< double > > );

} // namespace terra::fe::wedge::operators::shell
