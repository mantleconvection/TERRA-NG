
#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/gca/local_matrix_storage.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class DivKGrad
{
  public:
    using SrcVectorType                 = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType                 = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType                    = ScalarT;
    static constexpr int LocalMatrixDim = 6;
    using LocalMatrixStorage            = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >;

  private:
    LocalMatrixStorage local_matrix_storage_;

    bool single_quadpoint_ = false;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< ScalarType >                     k_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;
    linalg::OperatorStoredMatrixMode  operator_stored_matrix_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    dense::Vec< ScalarT, 3 > quad_points_3x2_[quadrature::quad_felippa_3x2_num_quad_points];
    ScalarT                  quad_weights_3x2_[quadrature::quad_felippa_3x2_num_quad_points];
    dense::Vec< ScalarT, 3 > quad_points_1x1_[quadrature::quad_felippa_1x1_num_quad_points];
    ScalarT                  quad_weights_1x1_[quadrature::quad_felippa_1x1_num_quad_points];

  public:
    DivKGrad(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const grid::Grid4DDataScalar< ScalarType >&                     k,
        bool                                                            treat_boundary,
        bool                                                            diagonal,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode operator_stored_matrix_mode = linalg::OperatorStoredMatrixMode::Off )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , k_( k )
    , mask_( mask )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , operator_stored_matrix_mode_( operator_stored_matrix_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {
        quadrature::quad_felippa_1x1_quad_points( quad_points_1x1_ );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights_1x1_ );
        quadrature::quad_felippa_3x2_quad_points( quad_points_3x2_ );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights_3x2_ );
    }

    /// @brief Getter for mask member
    KOKKOS_INLINE_FUNCTION
    bool has_flag(
        const int                      local_subdomain_id,
        const int                      x_cell,
        const int                      y_cell,
        const int                      r_cell,
        grid::shell::ShellBoundaryFlag flag ) const
    {
        return util::has_flag( mask_( local_subdomain_id, x_cell, y_cell, r_cell ), flag );
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    /// @brief S/Getter for diagonal member
    void set_diagonal( bool v ) { diagonal_ = v; }

    /// @brief Getter for coefficient
    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }

    /// @brief Getter for domain member
    const grid::shell::DistributedDomain& get_domain() const { return domain_; }

    /// @brief Getter for radii member
    grid::Grid2DDataScalar< ScalarT > get_radii() const { return radii_; }

    /// @brief Getter for grid member
    grid::Grid3DDataVec< ScalarT, 3 > get_grid() const { return grid_; }

    /// @brief S/Getter for quadpoint member
    void set_single_quadpoint( bool v ) { single_quadpoint_ = v; }

    void set_stored_matrix_mode(
        linalg::OperatorStoredMatrixMode     operator_stored_matrix_mode,
        int                                  level_range,
        grid::Grid4DDataScalar< ScalarType > GCAElements )
    {
        operator_stored_matrix_mode_ = operator_stored_matrix_mode;

        // allocate storage if necessary
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            local_matrix_storage_ = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >(
                domain_, operator_stored_matrix_mode_, level_range, GCAElements );
        }
    }

    linalg::OperatorStoredMatrixMode get_stored_matrix_mode() { return operator_stored_matrix_mode_; }

    /// @brief Set the local matrix stored in the operator
    KOKKOS_INLINE_FUNCTION
    void set_local_matrix(
        const int                                                    local_subdomain_id,
        const int                                                    x_cell,
        const int                                                    y_cell,
        const int                                                    r_cell,
        const int                                                    wedge,
        const dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim >& mat ) const
    {
        // request from storage
        KOKKOS_ASSERT( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off );
        local_matrix_storage_.set_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge, mat );
    }

    /// @brief Retrives the local matrix
    /// if there is stored local matrices, the desired local matrix is loaded and returned
    /// if not, the local matrix is assembled on-the-fly
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > get_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        // request from storage
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            if ( !local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) )
            {
                Kokkos::abort( "No matrix found at that spatial index." );
            }
            return local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
        }
        else
        {
            return assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
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

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

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
        constexpr int hex_offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
        constexpr int hex_offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
        constexpr int hex_offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

        // use stored matrices (at least on some elements)
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A[num_wedges_per_hex_cell] = { 0 };

            if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Full )
            {
                A[0] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                A[1] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
            }
            else if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Selective )
            {
                if ( local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 ) &&
                     local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 ) )
                {
                    A[0] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                    A[1] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
                }
                else
                {
                    // Kokkos::abort("Matrix not found.");
                    A[0] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                    A[1] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
                }
            }

            if ( diagonal_ )
            {
                A[0] = A[0].diagonal();
                A[1] = A[1].diagonal();
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

            dense::Vec< ScalarT, LocalMatrixDim > src[num_wedges_per_hex_cell];
            extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

            dense::Vec< ScalarT, LocalMatrixDim > dst[num_wedges_per_hex_cell];

            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
        }
        else
        {
            // assemble on-the-fly

            // Gather surface points for each wedge.
            dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

            // Gather wedge radii.
            const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
            const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

            // Quadrature points.
            int num_quad_points = single_quadpoint_ ? quadrature::quad_felippa_1x1_num_quad_points :
                                                      quadrature::quad_felippa_3x2_num_quad_points;

            dense::Vec< ScalarT, 6 > k_local_hex[num_wedges_per_hex_cell];
            extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

            ScalarType src_local_hex[8] = { 0 };
            ScalarType dst_local_hex[8] = { 0 };

            for ( int i = 0; i < 8; i++ )
            {
                src_local_hex[i] = src_(
                    local_subdomain_id, x_cell + hex_offset_x[i], y_cell + hex_offset_y[i], r_cell + hex_offset_r[i] );
            }

            // Compute the local element matrix.

            for ( int q = 0; q < num_quad_points; q++ )
            {
                const auto w  = single_quadpoint_ ? quad_weights_1x1_[q] : quad_weights_3x2_[q];
                const auto qp = single_quadpoint_ ? quad_points_1x1_[q] : quad_points_3x2_[q];

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    dense::Vec< ScalarType, 3 > grad[num_nodes_per_wedge];
                    ScalarType                  jdet_keval_quadweight = 0;

                    assemble_trial_test_vecs(
                        wedge, qp, w, r_1, r_2, wedge_phy_surf, k_local_hex, grad, jdet_keval_quadweight );

                    // dot of coeff dofs and element-local shape functions to evaluate the coefficent on the current element
                    ScalarType k_eval = 0.0;

                    for ( int k = 0; k < num_nodes_per_wedge; k++ )
                    {
                        k_eval += shape( k, qp ) * k_local_hex[wedge]( k );
                    }

                    jdet_keval_quadweight *= k_eval;

                    fused_local_mv( src_local_hex, dst_local_hex, wedge, jdet_keval_quadweight, grad, r_cell );
                }
            }

            for ( int i = 0; i < 8; i++ )
            {
                Kokkos::atomic_add(
                    &dst_(
                        local_subdomain_id,
                        x_cell + hex_offset_x[i],
                        y_cell + hex_offset_y[i],
                        r_cell + hex_offset_r[i] ),
                    dst_local_hex[i] );
            }
        }
    }

    KOKKOS_INLINE_FUNCTION void assemble_trial_test_vecs(
        const int                          wedge,
        const dense::Vec< ScalarType, 3 >& quad_point,
        const ScalarType                   quad_weight,
        const ScalarT                      r_1,
        const ScalarT                      r_2,
        dense::Vec< ScalarT, 3 > ( *wedge_phy_surf )[3],
        const dense::Vec< ScalarT, 6 >* k_local_hex,
        dense::Vec< ScalarType, 3 >*    grad,
        ScalarType&                     jdet_quadweight ) const
    {
        dense::Mat< ScalarType, 3, 3 >       J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_point );
        const auto                           det              = J.det();
        const auto                           abs_det          = Kokkos::abs( det );
        const dense::Mat< ScalarType, 3, 3 > J_inv_transposed = J.inv_transposed( det );

        for ( int k = 0; k < num_nodes_per_wedge; k++ )
        {
            grad[k] = J_inv_transposed * grad_shape( k, quad_point );
        }
        jdet_quadweight = quad_weight * abs_det;
    }

    /// @brief assemble the local matrix and return it for a given element, wedge, and vectorial component
    /// (determined by dimi, dimj)
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > assemble_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        // Gather surface points for each wedge.
        // TODO gather this for only 1 wedge
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        dense::Vec< ScalarT, 6 > k_local_hex[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

        // Compute the local element matrix.
        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A = { 0 };
        int num_quad_points = single_quadpoint_ ? quadrature::quad_felippa_1x1_num_quad_points :
                                                  quadrature::quad_felippa_3x2_num_quad_points;

        // spatial dimensions: quadrature points and wedge
        for ( int q = 0; q < num_quad_points; q++ )
        {
            const auto w  = single_quadpoint_ ? quad_weights_1x1_[q] : quad_weights_3x2_[q];
            const auto qp = single_quadpoint_ ? quad_points_1x1_[q] : quad_points_3x2_[q];

            dense::Vec< ScalarType, 3 > grad[num_nodes_per_wedge];
            ScalarType                  jdet_keval_quadweight = 0;
            assemble_trial_test_vecs(
                wedge, qp, w, r_1, r_2, wedge_phy_surf, k_local_hex, grad, jdet_keval_quadweight );

            // dot of coeff dofs and element-local shape functions to evaluate the coefficent on the current element
            ScalarType k_eval = 0.0;

            for ( int k = 0; k < num_nodes_per_wedge; k++ )
            {
                k_eval += shape( k, qp ) * k_local_hex[wedge]( k );
            }

            jdet_keval_quadweight *= k_eval;

            // propagate on local matrix by outer product of test and trial vecs
            for ( int i = 0; i < num_nodes_per_wedge; i++ )
            {
                for ( int j = 0; j < num_nodes_per_wedge; j++ )
                {
                    A( i, j ) += jdet_keval_quadweight * grad[j].dot( grad[i] );
                    // for the div, we just extract the component from the gradient vector
                }
            }
        }

        if ( treat_boundary_ )
        {
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > boundary_mask;
            boundary_mask.fill( 1.0 );

            if ( r_cell == 0 )
            {
                // Inner boundary (CMB).
                for ( int i = 0; i < 6; i++ )
                {
                    for ( int j = 0; j < 6; j++ )
                    {
                        // on diagonal components of the vectorial diffusion operator, we exclude the diagonal entries from elimination
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
                        // on diagonal components of the vectorial diffusion operator, we exclude the diagonal entries from elimination
                        if ( i != j && ( i >= 3 || j >= 3 ) )
                        {
                            boundary_mask( i, j ) = 0.0;
                        }
                    }
                }
            }
            A.hadamard_product( boundary_mask );
        }

        return A;
    }

    // executes the fused local matvec on an element, given the assembled trial and test vectors
    KOKKOS_INLINE_FUNCTION void fused_local_mv(
        ScalarType                   src_local_hex[8],
        ScalarType                   dst_local_hex[8],
        const int                    wedge,
        const ScalarType             jdet_keval_quadweight,
        dense::Vec< ScalarType, 3 >* grad,
        int                          r_cell ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        dense::Vec< ScalarType, 3 > grad_u;
        grad_u.fill( 0.0 );

        const bool at_cmb        = r_cell == 0;
        const bool at_surface    = r_cell + 1 == radii_.extent( 1 ) - 1;
        int        cmb_shift     = 0;
        int        surface_shift = 0;

        // Compute ∇u at this quadrature point.
        if ( !diagonal_ )
        {
            if ( treat_boundary_ && at_cmb )
            {
                // at the core-mantle boundary, we exclude dofs that are lower-indexed than the dof on the boundary
                cmb_shift = 3;
            }
            else if ( treat_boundary_ && at_surface )
            {
                // at the surface boundary, we exclude dofs that are higher-indexed than the dof on the boundary
                surface_shift = 3;
            }

            // accumulate the element-local gradient/divergence of the trial function (loop over columns of local matrix/local dofs)
            // by dot of trial vec and src dofs
            for ( int i = 0 + cmb_shift; i < num_nodes_per_wedge - surface_shift; i++ )
            {
                grad_u = grad_u +
                         grad[i] * src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];
            }

            // Add the test function contributions.
            // for each row of the local matrix (test-functions):
            // multiply trial part (fully assembled for the current element from loop above) with test part corresponding to the current row/dof
            // += due to contributions from other elements
            for ( int j = 0 + cmb_shift; j < num_nodes_per_wedge - surface_shift; j++ )
            {
                dst_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]] +=
                    jdet_keval_quadweight * ( grad[j] ).dot( grad_u );
                // for the div, we just extract the component from the gradient vector
            }
        }

        // Dirichlet DoFs are only to be eliminated on diagonal blocks of epsilon
        if ( diagonal_ || ( ( treat_boundary_ && ( at_cmb || at_surface ) ) ) )
        {
            // for the diagonal elements at the boundary, we switch the shifts
            for ( int i = 0 + surface_shift; i < num_nodes_per_wedge - cmb_shift; i++ )
            {
                const auto grad_u_diag =
                    grad[i] * src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];

                dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                    jdet_keval_quadweight * ( grad[i] ).dot( grad_u_diag );
            }
        }
    }
};

static_assert( linalg::GCACapable< DivKGrad< float > > );
static_assert( linalg::GCACapable< DivKGrad< double > > );

} // namespace terra::fe::wedge::operators::shell
