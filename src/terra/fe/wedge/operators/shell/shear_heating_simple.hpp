
#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class ShearHeatingSimple
{
  public:
    using SrcVectorType                 = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType                 = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType                    = ScalarT;
    using Grid4DDataLocalMatrices       = terra::grid::Grid4DDataMatrices< ScalarType, 6, 6, 2 >;
    static constexpr int LocalMatrixDim = 6;

  private:
    bool storeLMatrices_ =
        false; // set to let apply_impl() know, that it should store the local matrices after assembling them
    bool applyStoredLMatrices_ =
        false; // set to make apply_impl() load and use the stored LMatrices for the operator application
    Grid4DDataLocalMatrices lmatrices_;
    bool                    single_quadpoint_ = false;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >    grid_;
    grid::Grid2DDataScalar< ScalarT >    radii_;
    
    grid::Grid4DDataScalar< ScalarType > coeff_times_mu_;
    grid::Grid4DDataScalar< ScalarType > ux_;
    grid::Grid4DDataScalar< ScalarType > uy_;
    grid::Grid4DDataScalar< ScalarType > uz_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    dense::Vec< ScalarT, 3 > quad_points_3x2_[quadrature::quad_felippa_3x2_num_quad_points];
    ScalarT                  quad_weights_3x2_[quadrature::quad_felippa_3x2_num_quad_points];
    dense::Vec< ScalarT, 3 > quad_points_1x1_[quadrature::quad_felippa_1x1_num_quad_points];
    ScalarT                  quad_weights_1x1_[quadrature::quad_felippa_1x1_num_quad_points];

  public:
    ShearHeatingSimple(
        const grid::shell::DistributedDomain&       domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&    grid,
        const grid::Grid2DDataScalar< ScalarT >&    radii,
        const grid::Grid4DDataScalar< ScalarType >& coeff_times_mu,
        const grid::Grid4DDataScalar< ScalarType >& ux,
        const grid::Grid4DDataScalar< ScalarType >& uy,
        const grid::Grid4DDataScalar< ScalarType >& uz,
        bool                                        treat_boundary,
        bool                                        diagonal,
        linalg::OperatorApplyMode                   operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode           operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , coeff_times_mu_(coeff_times_mu)
    , ux_( ux )
    , uy_( uy )
    , uz_( uz )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {
        quadrature::quad_felippa_1x1_quad_points( quad_points_1x1_ );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights_1x1_ );
        quadrature::quad_felippa_3x2_quad_points( quad_points_3x2_ );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights_3x2_ );
    }

    /// @brief Getter for domain member
    const grid::shell::DistributedDomain& get_domain() const { return domain_; }

    /// @brief Getter for radii member
    grid::Grid2DDataScalar< ScalarT > get_radii() const { return radii_; }

    /// @brief Getter for grid member
    grid::Grid3DDataVec< ScalarT, 3 > get_grid() const { return grid_; }

    /// @brief S/Getter for diagonal member
    void set_diagonal( bool v ) { diagonal_ = v; }

    /// @brief S/Getter for quadpoint member
    void set_single_quadpoint( bool v ) { single_quadpoint_ = v; }

    /// @brief Retrives the local matrix stored in the operator
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, 6, 6 > get_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge )
    {
        assert( lmatrices_.data() != nullptr );

        return lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
    }

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
        Kokkos::abort( "Not implemented." );
    }

    /// @brief Setter/Getter for app applyStoredLMatrices_: usage of stored local matrices during apply
    void setApplyStoredLMatrices( bool v ) { applyStoredLMatrices_ = v; }

    /// @brief
    /// allocates memory for the local matrices
    /// calls kernel with storeLMatrices_ = true to assemble and store the local matrices
    /// sets applyStoredLMatrices_, such that future applies use the stored local matrices
    void store_lmatrices()
    {
        storeLMatrices_ = true;
        if ( lmatrices_.data() == nullptr )
        {
            lmatrices_ = Grid4DDataLocalMatrices(
                "LaplaceSimple::lmatrices_",
                domain_.subdomains().size(),
                domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain_.domain_info().subdomain_num_nodes_radially() - 1 );
            Kokkos::parallel_for(
                "assemble_store_lmatrices", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
            Kokkos::fence();
        }
        storeLMatrices_       = false;
        applyStoredLMatrices_ = true;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( storeLMatrices_ or applyStoredLMatrices_ )
            assert( lmatrices_.data() != nullptr );

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
        dense::Mat< ScalarT, 6, 6 > A[num_wedges_per_hex_cell] = {};
        if ( !applyStoredLMatrices_ )
        {
            // Gather surface points for each wedge.
            dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

            // Gather wedge radii.
            const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
            const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

            // Quadrature points.
            int num_quad_points = single_quadpoint_ ? quadrature::quad_felippa_1x1_num_quad_points :
                                                      quadrature::quad_felippa_3x2_num_quad_points;

            dense::Vec< ScalarT, 6 > coeff_times_mu[num_wedges_per_hex_cell];

            dense::Vec< ScalarT, 6 > ux[num_wedges_per_hex_cell];
            dense::Vec< ScalarT, 6 > uy[num_wedges_per_hex_cell];
            dense::Vec< ScalarT, 6 > uz[num_wedges_per_hex_cell];

            extract_local_wedge_scalar_coefficients( coeff_times_mu, local_subdomain_id, x_cell, y_cell, r_cell, coeff_times_mu_ );

            extract_local_wedge_scalar_coefficients( ux, local_subdomain_id, x_cell, y_cell, r_cell, ux_ );
            extract_local_wedge_scalar_coefficients( uy, local_subdomain_id, x_cell, y_cell, r_cell, uy_ );
            extract_local_wedge_scalar_coefficients( uz, local_subdomain_id, x_cell, y_cell, r_cell, uz_ );

            // Compute the local element matrix.

            for ( int q = 0; q < num_quad_points; q++ )
            {
                const auto w  = single_quadpoint_ ? quad_weights_1x1_[q] : quad_weights_3x2_[q];
                const auto qp = single_quadpoint_ ? quad_points_1x1_[q] : quad_points_3x2_[q];

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, qp );
                    const auto det              = Kokkos::abs( J.det() );
                    const auto J_inv_transposed = J.inv().transposed();
                    
                    ///////////////////////////////////////////////////////////////////////////////////////
                    // We need to implement the shear heating term
                    // \phi = 2\mu (\eps - \frac{1}{3}\nabla\cdot{u}) : (\eps - \frac{1}{3}\nabla\cdot{u})
                    // where, \eps = \frac{1}{2}(\nabla u + (\nabla u)^T)
                    // here the parameter coeff_times_mu means that the shear heating term 
                    // can be scaled in such a way that it can be turned off/on on certain 
                    // parts of the domain
                    ///////////////////////////////////////////////////////////////////////////////////////

                    ScalarType coeff_times_mu_eval = 0.0;

                    ScalarType dux_dx_eval = 0.0;
                    ScalarType dux_dy_eval = 0.0;
                    ScalarType dux_dz_eval = 0.0;

                    ScalarType duy_dx_eval = 0.0;
                    ScalarType duy_dy_eval = 0.0;
                    ScalarType duy_dz_eval = 0.0;

                    ScalarType duz_dx_eval = 0.0;
                    ScalarType duz_dy_eval = 0.0;
                    ScalarType duz_dz_eval = 0.0;

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto shape_j = shape( j, qp );

                        coeff_times_mu_eval += shape_j * coeff_times_mu[wedge]( j );

                        const auto grad_j = grad_shape( j, qp );

                        dux_dx_eval += ( J_inv_transposed * grad_j )( 0 ) * ux[wedge]( j );
                        dux_dy_eval += ( J_inv_transposed * grad_j )( 1 ) * ux[wedge]( j );
                        dux_dz_eval += ( J_inv_transposed * grad_j )( 2 ) * ux[wedge]( j );

                        duy_dx_eval += ( J_inv_transposed * grad_j )( 0 ) * uy[wedge]( j );
                        duy_dy_eval += ( J_inv_transposed * grad_j )( 1 ) * uy[wedge]( j );
                        duy_dz_eval += ( J_inv_transposed * grad_j )( 2 ) * uy[wedge]( j );

                        duz_dx_eval += ( J_inv_transposed * grad_j )( 0 ) * uz[wedge]( j );
                        duz_dy_eval += ( J_inv_transposed * grad_j )( 1 ) * uz[wedge]( j );
                        duz_dz_eval += ( J_inv_transposed * grad_j )( 2 ) * uz[wedge]( j );
                    }

                    const auto shear_heating_qp =
                        2 * std::pow( 0.5 * dux_dy_eval + 0.5 * duy_dx_eval, 2 ) +
                        2 * std::pow( 0.5 * dux_dz_eval + 0.5 * duz_dx_eval, 2 ) +
                        2 * std::pow( 0.5 * duy_dz_eval + 0.5 * duz_dy_eval, 2 ) +
                        std::pow( -0.33333333333333331 * dux_dx_eval - 0.33333333333333331 * duy_dy_eval +
                                 0.66666666666666674 * duz_dz_eval,
                             2 ) +
                        std::pow( -0.33333333333333331 * dux_dx_eval + 0.66666666666666674 * duy_dy_eval -
                                 0.33333333333333331 * duz_dz_eval,
                             2 ) +
                        std::pow( 0.66666666666666674 * dux_dx_eval - 0.33333333333333331 * duy_dy_eval -
                                 0.33333333333333331 * duz_dz_eval,
                             2 );

                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        const auto u = shape( i, qp );

                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            const auto v = shape( j, qp );

                            A[wedge]( i, j ) += w * shear_heating_qp * (2.0 * coeff_times_mu_eval) * u * v * det;
                        }
                    }
                }
            }
        }
        else
        {
            // load LMatrix for both local wedges
            A[0] = lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
            A[1] = lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        if ( storeLMatrices_ )
        {
            // write local matrices to mem
            lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 0 ) = A[0];
            lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 1 ) = A[1];
        }
        else
        {
            // apply local matrices to local DoFs
            dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
            extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

            dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
        }
    }
};

static_assert( linalg::OperatorLike< ShearHeatingSimple< float > > );
static_assert( linalg::OperatorLike< ShearHeatingSimple< double > > );

} // namespace terra::fe::wedge::operators::shell