#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

/// \brief ∇·(ν ∇·) on the shell with a per-wedge piecewise-constant
///        coefficient ν_w.
///
/// One scalar per wedge (2 per hex), passed in as a Grid5DDataScalar of shape
/// (num_subdomains, Nc_x, Nc_y, Nc_r, num_wedges_per_hex_cell).  The local
/// 6×6 Galerkin Laplacian K_w is built per wedge from the same Felippa 3×2
/// quadrature + jac()/inv_transposed()/grad_shape() pattern used by Mass /
/// Laplace / DivKGrad, then multiplied by ν_w and scattered additively.
///
/// Used by the entropy-viscosity stabilization in EVSolver as the operator
/// that turns the per-wedge stabilization field ν_h_w into the RHS
/// contribution ∫ ν_h ∇T · ∇φ.  No boundary treatment / diagonal mode /
/// GCA storage — the call site is a single explicit apply per step.
template < typename ScalarT >
class WedgeConstantDivKGrad
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;
    grid::Grid5DDataScalar< ScalarT > nu_;
    // Optional spatially-constant coefficient. When use_scalar_nu_ is set the
    // kernel uses scalar_nu_ directly and nu_ stays empty (unallocated), so a
    // uniform coefficient costs no per-wedge field.
    ScalarT                           scalar_nu_     = ScalarT( 0 );
    bool                              use_scalar_nu_ = false;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    WedgeConstantDivKGrad(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        const grid::Grid5DDataScalar< ScalarT >& nu_wedge,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , nu_( nu_wedge )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    /// Constant-coefficient overload: ν is a single scalar everywhere, so no
    /// per-wedge Grid5D field is stored (nu_ stays empty). Use for uniform ν.
    WedgeConstantDivKGrad(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        ScalarT                                  scalar_nu,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , scalar_nu_( scalar_nu )
    , use_scalar_nu_( true )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    /// Read-only coefficient handle (so the caller can update ν in place).
    const grid::Grid5DDataScalar< ScalarT >& nu_wedge_grid_data() const { return nu_; }

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
            throw std::runtime_error( "WedgeConstantDivKGrad: src/dst extent mismatch" );
        }

        Kokkos::parallel_for(
            "wedge_constant_div_k_grad_matvec",
            grid::shell::local_domain_md_range_policy_cells( domain_ ),
            *this );

        Kokkos::fence();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries(
                domain_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Wedge surface coords on the unit sphere.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        constexpr int num_q = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > qp[num_q];
        ScalarT                  qw[num_q];
        quadrature::quad_felippa_3x2_quad_points( qp );
        quadrature::quad_felippa_3x2_quad_weights( qw );

        dense::Vec< ScalarT, num_nodes_per_wedge > src_w[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src_w, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

        dense::Vec< ScalarT, num_nodes_per_wedge > dst_w[num_wedges_per_hex_cell] = {};

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
        {
            const ScalarT nu_w =
                use_scalar_nu_ ? scalar_nu_ : nu_( local_subdomain_id, x_cell, y_cell, r_cell, wedge );

            for ( int q = 0; q < num_q; ++q )
            {
                const dense::Mat< ScalarT, 3, 3 > J = jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                const ScalarT det     = J.det();
                const ScalarT abs_det = Kokkos::abs( det );
                if ( abs_det < ScalarT( 1e-30 ) )
                {
                    continue;
                }
                const dense::Mat< ScalarT, 3, 3 > J_inv_t = J.inv_transposed( det );

                dense::Vec< ScalarT, 3 > grad_phy[num_nodes_per_wedge];
                for ( int k = 0; k < num_nodes_per_wedge; ++k )
                {
                    grad_phy[k] = J_inv_t * grad_shape( k, qp[q] );
                }

                // grad u(q) = Σ_j src_w(j) · grad_phy[j]
                dense::Vec< ScalarT, 3 > grad_u{};
                for ( int j = 0; j < num_nodes_per_wedge; ++j )
                {
                    grad_u = grad_u + src_w[wedge]( j ) * grad_phy[j];
                }

                // (K_w · src_w)(i) += w_q · |det J| · ν_w · grad_phy[i] · grad_u
                const ScalarT scale = qw[q] * abs_det * nu_w;
                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                {
                    dst_w[wedge]( i ) += scale * grad_phy[i].dot( grad_u );
                }
            }
        }

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst_w );
    }
};

static_assert( linalg::OperatorLike< WedgeConstantDivKGrad< float > > );
static_assert( linalg::OperatorLike< WedgeConstantDivKGrad< double > > );

} // namespace terra::fe::wedge::operators::shell
