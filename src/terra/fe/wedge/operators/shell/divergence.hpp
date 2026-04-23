
#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"
#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"

namespace terra::fe::wedge::operators::shell {

using grid::shell::get_boundary_condition_flag;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using terra::grid::shell::BoundaryConditionFlag;
using terra::grid::shell::BoundaryConditions;
using terra::grid::shell::ShellBoundaryFlag;
using terra::linalg::trafo::trafo_mat_cartesian_to_normal_tangential;

template < typename ScalarT >
class Divergence
{
  public:
    using SrcVectorType = linalg::VectorQ1Vec< ScalarT, 3 >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_fine_;
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_fine_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_fine_;

    BoundaryConditions bcs_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT >                        recv_buffers_;
    terra::communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataScalar< ScalarT > >    comm_plan_;

    grid::Grid4DDataVec< ScalarType, 3 > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    Divergence(
        const grid::shell::DistributedDomain&                           domain_fine,
        const grid::shell::DistributedDomain&                           domain_coarse,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid_fine,
        const grid::Grid2DDataScalar< ScalarT >&                        radii_fine,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_fine,
        BoundaryConditions                                              bcs,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_fine_( domain_fine )
    , domain_coarse_( domain_coarse )
    , grid_fine_( grid_fine )
    , radii_( radii_fine )
    , boundary_mask_fine_( boundary_mask_fine )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , recv_buffers_( domain_coarse )
    , comm_plan_( domain_coarse )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "divergence_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        util::Timer timer_kernel( "divergence_kernel" );
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_fine_ ), *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "divergence_comm" );
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Gather surface points for each wedge.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_fine_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // Quadrature points.
        constexpr auto num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );

        const int fine_radial_wedge_index = r_cell % 2;

        // Compute the local element matrix.
        dense::Mat< ScalarT, 6, 18 > A[num_wedges_per_hex_cell] = {};

        for ( int q = 0; q < num_quad_points; q++ )
        {
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const int fine_lateral_wedge_index = fine_lateral_wedge_idx( x_cell, y_cell, wedge );

                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i =
                        shape_coarse( i, fine_radial_wedge_index, fine_lateral_wedge_index, quad_points[q] );

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto grad_j = grad_shape( j, quad_points[q] );

                        for ( int d = 0; d < 3; d++ )
                        {
                            A[wedge]( i, d * 6 + j ) +=
                                quad_weights[q] * ( -( J_inv_transposed * grad_j )(d) *shape_i * det );
                        }
                    }
                }
            }
        }

        bool at_cmb = util::has_flag( boundary_mask_fine_( local_subdomain_id, x_cell, y_cell, r_cell ), CMB );
        bool at_surface =
            util::has_flag( boundary_mask_fine_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), SURFACE );

        dense::Vec< ScalarT, 18 > src[num_wedges_per_hex_cell];
        for ( int d = 0; d < 3; d++ )
        {
            dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
            extract_local_wedge_vector_coefficients( src_d, local_subdomain_id, x_cell, y_cell, r_cell, d, src_ );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    src[wedge]( d * 6 + i ) = src_d[wedge]( i );
                }
            }
        }

        // Boundary treatment
        dense::Mat< ScalarT, 6, 18 > boundary_mask;
        boundary_mask.fill( 1.0 );

        dense::Mat< ScalarT, 18, 18 > R[num_wedges_per_hex_cell];

        if ( at_cmb || at_surface )
        {
            // Inner boundary (CMB).
            ShellBoundaryFlag     sbf = at_cmb ? CMB : SURFACE;
            BoundaryConditionFlag bcf = get_boundary_condition_flag( bcs_, sbf );

            if ( bcf == DIRICHLET )
            {
                for ( int dimj = 0; dimj < 3; ++dimj )
                {
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            if ( at_cmb && ( j < 3 ) || at_surface && ( j >= 3 ) )
                            {
                                boundary_mask( i, dimj * num_nodes_per_wedge + j ) = 0.0;
                            }
                        }
                    }
                }
            }
            else if ( bcf == FREESLIP )
            {
                
                dense::Mat< ScalarT, 6, 18 > A_tmp[num_wedges_per_hex_cell] = { 0 };

                // reorder source dofs for nodes instead of velocity dims in src vector and local matrix
                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    for ( int node_idxi = 0; node_idxi < num_nodes_per_wedge; node_idxi++ )
                    {
                        for ( int dimj = 0; dimj < 3; ++dimj )
                        {
                            for ( int node_idxj = 0; node_idxj < num_nodes_per_wedge; node_idxj++ )
                            {
                                A_tmp[wedge]( node_idxi, node_idxj * 3 + dimj ) =
                                    A[wedge]( node_idxi, node_idxj + dimj * num_nodes_per_wedge );
                            }
                        }
                    }
                    reorder_local_dofs( DoFOrdering::DIMENSIONWISE, DoFOrdering::NODEWISE, src[wedge] );
                }

                // assemble rotation matrices for boundary nodes
                // e.g. if we are at CMB, we need to rotate DoFs 0, 1, 2 of each wedge
                // at SURFACE, we need to rotate DoFs 3, 4, 5

                constexpr int layer_hex_offset_x[2][3] = { { 0, 1, 0 }, { 1, 0, 1 } };
                constexpr int layer_hex_offset_y[2][3] = { { 0, 0, 1 }, { 1, 1, 0 } };

                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    // make rotation matrix unity
                    for ( int i = 0; i < 18; ++i )
                    {
                        R[wedge]( i, i ) = 1.0;
                    }

                    for ( int boundary_node_idx = 0; boundary_node_idx < 3; boundary_node_idx++ )
                    {
                        // compute normal
                        dense::Vec< double, 3 > normal = grid::shell::coords(
                            local_subdomain_id,
                            x_cell + layer_hex_offset_x[wedge][boundary_node_idx],
                            y_cell + layer_hex_offset_y[wedge][boundary_node_idx],
                            r_cell + ( at_cmb ? 0 : 1 ),
                            grid_fine_,
                            radii_ );
                      

                        // compute rotation matrix for DoFs on current node
                        auto R_i = trafo_mat_cartesian_to_normal_tangential( normal );

                        // insert into wedge-local rotation matrix
                        int offset_in_R = at_cmb ? 0 : 9;
                        for ( int dimi = 0; dimi < 3; ++dimi )
                        {
                            for ( int dimj = 0; dimj < 3; ++dimj )
                            {
                                R[wedge](
                                    offset_in_R + boundary_node_idx * 3 + dimi,
                                    offset_in_R + boundary_node_idx * 3 + dimj ) = R_i( dimi, dimj );
                            }
                        }
                    }

                    // transform local matrix to rotated/ normal-tangential space: pre/post multiply with rotation matrices
                    // TODO transpose this way?
                    A[wedge] = A_tmp[wedge] * R[wedge].transposed();
                    // transform source dofs to nt-space
                    auto src_tmp = R[wedge] * src[wedge];
                    for ( int i = 0; i < 18; ++i )
                    {
                        src[wedge]( i ) = src_tmp( i );
                    }

                    // eliminate normal components: Dirichlet on the normal-tangential system
                    int node_start = at_surface ? 3 : 0;
                    int node_end   = at_surface ? 6 : 3;
                    for ( int node_idx = node_start; node_idx < node_end; node_idx++ )
                    {
                        int idx = node_idx * 3;
                        for ( int k = 0; k < 6; ++k )
                        {
                            boundary_mask( k, idx ) = 0.0;
                        }
                    }
                }
            }
            else if ( bcf == NEUMANN ) {}
        }

        // apply boundary mask
        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            A[wedge].hadamard_product( boundary_mask );
        }

        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        // no need to reorder or post trafo the pressure:
        // independent of dof ordering, div ops map to the same 6 coarse-grid pressure dofs in the same orderinng

        atomically_add_local_wedge_scalar_coefficients(
            dst_, local_subdomain_id, x_cell / 2, y_cell / 2, r_cell / 2, dst );
    }
};

static_assert( linalg::OperatorLike< Divergence< double > > );

} // namespace terra::fe::wedge::operators::shell