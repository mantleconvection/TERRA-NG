
#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/triangle/quadrature/quadrature.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class BoundaryMass
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;

    grid::shell::ShellBoundaryFlag boundary_flag_;
    ScalarT                         zeta_boundary_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    BoundaryMass(
        const grid::shell::DistributedDomain&                            domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                         grid,
        const grid::Grid2DDataScalar< ScalarT >&                         radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const grid::shell::ShellBoundaryFlag                            boundary_flag,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , mask_( mask )
    , boundary_flag_( boundary_flag )
    , zeta_boundary_( boundary_flag == grid::shell::ShellBoundaryFlag::CMB ? -1.0 : 1.0 )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {
        if ( domain_.domain_info().num_subdomains_in_radial_direction() != 1 )
        {
            Kokkos::abort( "BoundaryMass only implemented for 1 radial subdomain." );
        }
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( boundary_flag_ == grid::shell::ShellBoundaryFlag::CMB )
        {
            Kokkos::parallel_for(
                "matvec",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                    { 0, 0, 0, 0 },
                    { static_cast< long long >( domain_.subdomains().size() ),
                      domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                      domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                      1 } ),
                *this );
        }
        else if ( boundary_flag_ == grid::shell::ShellBoundaryFlag::SURFACE )
        {
            Kokkos::parallel_for(
                "matvec",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                    { 0, 0, 0, domain_.domain_info().subdomain_num_nodes_radially() - 2 },
                    { static_cast< long long >( domain_.subdomains().size() ),
                      domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                      domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                      domain_.domain_info().subdomain_num_nodes_radially() - 1 } ),
                *this );
        }

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

        constexpr auto num_quad_points = triangle::quadrature::quad_triangle_3_num_quad_points;

        dense::Vec< ScalarT, 2 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        triangle::quadrature::quad_triangle_3_quad_points( quad_points );
        triangle::quadrature::quad_triangle_3_quad_weights( quad_weights );

        // Only now we introduce radially dependent terms.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // For now, compute the local element matrix. We'll improve that later.
        dense::Mat< ScalarT, 6, 6 > A[num_wedges_per_hex_cell] = {};

        // const ScalarT grad_r = grad_forward_map_rad( r_1, r_2 );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                const dense::Mat< ScalarT, 3, 3 > j_lat_3x3 = jac_lat(
                    wedge_phy_surf[wedge][0],
                    wedge_phy_surf[wedge][1],
                    wedge_phy_surf[wedge][2],
                    quad_points[q]( 0 ),
                    quad_points[q]( 1 ) );

                dense::Mat< ScalarT, 3, 2 > j_lat{};
                j_lat( 0, 0 ) = j_lat_3x3( 0, 0 );
                j_lat( 0, 1 ) = j_lat_3x3( 0, 1 );
                j_lat( 1, 0 ) = j_lat_3x3( 1, 0 );
                j_lat( 1, 1 ) = j_lat_3x3( 1, 1 );
                j_lat( 2, 0 ) = j_lat_3x3( 2, 0 );
                j_lat( 2, 1 ) = j_lat_3x3( 2, 1 );
                j_lat         = j_lat * forward_map_rad( r_1, r_2, zeta_boundary_ );

                const auto det = Kokkos::sqrt( Kokkos::abs( ( j_lat.transposed() * j_lat ).det() ) );

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const dense::Vec< ScalarT, 3 > qp_lat{
                            quad_points[q]( 0 ), quad_points[q]( 1 ), zeta_boundary_ };

                        const ScalarT shape_i = shape_lat( i, qp_lat ) * shape_rad( i, qp_lat );
                        const ScalarT shape_j = shape_lat( j, qp_lat ) * shape_rad( j, qp_lat );

                        A[wedge]( i, j ) += quad_weights[q] * ( shape_i * shape_j * det );
                    }
                }
            }
        }

        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
    }
};

static_assert( linalg::OperatorLike< BoundaryMass< float > > );
static_assert( linalg::OperatorLike< BoundaryMass< double > > );

} // namespace terra::fe::wedge::operators::shell