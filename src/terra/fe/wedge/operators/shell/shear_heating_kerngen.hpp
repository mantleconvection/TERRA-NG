
#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell {

/// @brief Viscous / shear-heating linear form, kernel-generated style.
///
/// Assembles the Q1-scalar right-hand-side contribution
///
///     dst_i = scale * ∫_Ω  Φ  N_i  dx ,      Φ = 2 η · ε̇_dev : ε̇_dev
///
/// where ε̇ = ½(∇u + ∇uᵀ) is the strain rate of the input velocity field,
/// ε̇_dev = ε̇ − ⅓ tr(ε̇) I its deviatoric part, η a scalar (viscosity)
/// coefficient field, N_i the Q1 test function, and `scale` a runtime scalar
/// (the caller passes the nondimensional Di/Ra prefactor).
///
/// This is the production, kernel-generated counterpart to ShearHeatingSimple:
/// the velocity is read directly from a VectorQ1Vec (not pre-split ux/uy/uz),
/// the per-quadpoint Φ is factored into a device helper so the same math backs
/// both the linear-form assembly and a nodal-Φ projection (needed by the
/// entropy-viscosity residual), and the apply path follows the same
/// OperatorApplyMode / OperatorCommunicationMode contract as the sibling
/// gradient_kerngen / divergence_kerngen operators.
///
/// Only the legacy (slow) element path is implemented; it runs on every
/// backend. Fast GPU-specialised paths are a follow-up.
template < typename ScalarT >
class ShearHeatingKerngen
{
  public:
    using SrcVectorType = linalg::VectorQ1Vec< ScalarT, 3 >;  ///< velocity u
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;  ///< heating RHS
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    grid::Grid4DDataScalar< ScalarType > eta_;  ///< scalar viscosity coefficient

    /// Runtime scalar prefactor applied to the assembled linear form
    /// (the caller sets Di/Ra). Defaults to 1.
    ScalarT scale_ = ScalarT( 1 );

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT >             recv_buffers_;
    communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataScalar< ScalarT > > comm_plan_;

    grid::Grid4DDataVec< ScalarType, 3 > src_;  ///< velocity (device view)
    grid::Grid4DDataScalar< ScalarType > dst_;  ///< heating RHS (device view)

    /// When true, assemble ∫ Φ N_i (the plain lumped-projection numerator) with
    /// scale_ = 1 and no Di/Ra — used by assemble_phi_nodal() to build a nodal
    /// Φ field for the entropy-viscosity residual. When false (default) the
    /// operator is the scaled physical heating source scale_·∫Φ N_i.
    bool phi_numerator_mode_ = false;

  public:
    ShearHeatingKerngen(
        const grid::shell::DistributedDomain&       domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&    grid,
        const grid::Grid2DDataScalar< ScalarT >&    radii,
        const grid::Grid4DDataScalar< ScalarType >& eta,
        linalg::OperatorApplyMode                   operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode           operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , eta_( eta )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , recv_buffers_( domain )
    , comm_plan_( domain )
    {}

    const grid::shell::DistributedDomain& get_domain() const { return domain_; }
    grid::Grid2DDataScalar< ScalarT >     get_radii() const { return radii_; }
    grid::Grid3DDataVec< ScalarT, 3 >     get_grid() const { return grid_; }

    /// Runtime prefactor for the assembled linear form (caller sets Di/Ra).
    void    set_scale( ScalarT s ) { scale_ = s; }
    ScalarT get_scale() const { return scale_; }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    /// @brief Pointwise viscous-dissipation Φ = 2η · ε̇_dev:ε̇_dev at one
    /// quadrature point of one wedge, from the per-wedge nodal velocity
    /// components and the physical velocity gradient.
    ///
    /// @param eta_q  scalar viscosity evaluated at the quad point
    /// @param grad_u physical velocity gradient, grad_u[a][b] = ∂u_a/∂x_b
    KOKKOS_INLINE_FUNCTION static ScalarT
        phi_at_quadpoint( ScalarT eta_q, const ScalarT ( &grad_u )[3][3] )
    {
        // Strain rate ε̇ = ½(∇u + ∇uᵀ).
        ScalarT e[3][3];
        for ( int a = 0; a < 3; ++a )
            for ( int b = 0; b < 3; ++b )
                e[a][b] = ScalarT( 0.5 ) * ( grad_u[a][b] + grad_u[b][a] );

        // Deviatoric part ε̇_dev = ε̇ − ⅓ tr(ε̇) I.
        const ScalarT tr_over_3 = ( e[0][0] + e[1][1] + e[2][2] ) / ScalarT( 3 );
        for ( int a = 0; a < 3; ++a )
            e[a][a] -= tr_over_3;

        // ε̇_dev : ε̇_dev  (full 3x3 double contraction; off-diagonals counted twice).
        ScalarT dd = ScalarT( 0 );
        for ( int a = 0; a < 3; ++a )
            for ( int b = 0; b < 3; ++b )
                dd += e[a][b] * e[a][b];

        return ScalarT( 2 ) * eta_q * dd;
    }

    /// @brief Fill @p phi_out with the lumped-nodal viscous dissipation Φ.
    ///
    /// Assembles the linear form ∫ Φ N_i and divides by the lumped mass row
    /// ∫ N_i, giving a Q1-nodal Φ field suitable for interpolation inside the
    /// entropy-viscosity residual. Uses scale_ = 1 (the residual carries its
    /// own Di/Ra prefactor). @p phi_out is overwritten.
    void assemble_phi_nodal( const SrcVectorType& u, DstVectorType& phi_out )
    {
        // Numerator: ∫ Φ N_i  (unscaled), additive across subdomain interfaces.
        const bool                              saved_num  = phi_numerator_mode_;
        const ScalarT                           saved_sc   = scale_;
        const linalg::OperatorApplyMode         saved_am   = operator_apply_mode_;
        const linalg::OperatorCommunicationMode saved_cm   = operator_communication_mode_;

        phi_numerator_mode_          = true;
        scale_                       = ScalarT( 1 );
        operator_apply_mode_         = linalg::OperatorApplyMode::Replace;
        operator_communication_mode_ = linalg::OperatorCommunicationMode::CommunicateAdditively;
        apply_impl( u, phi_out );

        phi_numerator_mode_          = saved_num;
        scale_                       = saved_sc;
        operator_apply_mode_         = saved_am;
        operator_communication_mode_ = saved_cm;

        // Lumped mass row ∫ N_i (assembled the same way) then phi_out /= mass.
        DstVectorType lumped( "shear_phi_lumped_mass", domain_, phi_out.mask_data() );
        assemble_lumped_mass_( lumped );

        auto phi_v  = phi_out.grid_data();
        auto mass_v = lumped.grid_data();
        Kokkos::parallel_for(
            "shear_phi_nodal_divide",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 0, 0, 0 },
                { phi_v.extent( 0 ), phi_v.extent( 1 ), phi_v.extent( 2 ), phi_v.extent( 3 ) } ),
            KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                const ScalarT m = mass_v( s, x, y, r );
                phi_v( s, x, y, r ) = ( m > ScalarT( 0 ) ) ? ( phi_v( s, x, y, r ) / m ) : ScalarT( 0 );
            } );
        Kokkos::fence();
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "shear_heating_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "ShearHeatingKerngen: src/grid extent mismatch" );
        }

        Kokkos::parallel_for(
            "shear_heating_apply_kernel",
            grid::shell::local_domain_md_range_policy_cells( domain_ ),
            KOKKOS_CLASS_LAMBDA( const int s, const int x_cell, const int y_cell, const int r_cell ) {
                this->process_cell( s, x_cell, y_cell, r_cell );
            } );
        Kokkos::fence();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

  private:
    /// @brief Assemble the element linear form for one hex cell (both wedges).
    KOKKOS_INLINE_FUNCTION void
        process_cell( const int s, const int x_cell, const int y_cell, const int r_cell ) const
    {
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, s, x_cell, y_cell );

        const ScalarT r_1 = radii_( s, r_cell );
        const ScalarT r_2 = radii_( s, r_cell + 1 );

        constexpr int num_q = quadrature::quad_felippa_3x2_num_quad_points;
        dense::Vec< ScalarT, 3 > qp[num_q];
        ScalarT                  qw[num_q];
        quadrature::quad_felippa_3x2_quad_points( qp );
        quadrature::quad_felippa_3x2_quad_weights( qw );

        // Per-wedge nodal gathers: viscosity and the three velocity components.
        dense::Vec< ScalarT, num_nodes_per_wedge > eta_w[num_wedges_per_hex_cell];
        dense::Vec< ScalarT, num_nodes_per_wedge > u_w[num_wedges_per_hex_cell][3];

        extract_local_wedge_scalar_coefficients( eta_w, s, x_cell, y_cell, r_cell, eta_ );
        for ( int d = 0; d < 3; ++d )
        {
            dense::Vec< ScalarT, num_nodes_per_wedge > comp[num_wedges_per_hex_cell];
            extract_local_wedge_vector_coefficients( comp, s, x_cell, y_cell, r_cell, d, src_ );
            u_w[0][d] = comp[0];
            u_w[1][d] = comp[1];
        }

        dense::Vec< ScalarT, num_nodes_per_wedge > dst[num_wedges_per_hex_cell] = {};

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
        {
            for ( int q = 0; q < num_q; ++q )
            {
                const auto    J    = jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                const ScalarT det  = Kokkos::abs( J.det() );
                const auto    Jinv = J.inv().transposed();

                // Physical velocity gradient grad_u[a][b] = ∂u_a/∂x_b and the
                // interpolated viscosity at this quad point.
                ScalarT grad_u[3][3] = {};
                ScalarT eta_q        = ScalarT( 0 );
                for ( int j = 0; j < num_nodes_per_wedge; ++j )
                {
                    const ScalarT N_j       = shape( j, qp[q] );
                    const auto    grad_phys = Jinv * grad_shape( j, qp[q] );
                    eta_q += N_j * eta_w[wedge]( j );
                    for ( int a = 0; a < 3; ++a )
                        for ( int b = 0; b < 3; ++b )
                            grad_u[a][b] += grad_phys( b ) * u_w[wedge][a]( j );
                }

                const ScalarT phi = phi_at_quadpoint( eta_q, grad_u );

                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                    dst[wedge]( i ) += qw[q] * phi * shape( i, qp[q] ) * det;
            }

            if ( !phi_numerator_mode_ && scale_ != ScalarT( 1 ) )
                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                    dst[wedge]( i ) *= scale_;
        }

        atomically_add_local_wedge_scalar_coefficients( dst_, s, x_cell, y_cell, r_cell, dst );
    }

  public:
    /// @brief Assemble the lumped mass row ∫ N_i into @p out (Replace + additive).
    /// Public because it hosts an extended KOKKOS_LAMBDA, which nvcc forbids in
    /// private/protected member functions.
    void assemble_lumped_mass_( DstVectorType& out )
    {
        assign( out, 0 );
        auto          out_v = out.grid_data();
        const auto    grid  = grid_;
        const auto    radii = radii_;
        constexpr int num_q = quadrature::quad_felippa_3x2_num_quad_points;

        Kokkos::parallel_for(
            "shear_phi_lumped_mass_assemble",
            grid::shell::local_domain_md_range_policy_cells( domain_ ),
            KOKKOS_LAMBDA( const int s, const int x_cell, const int y_cell, const int r_cell ) {
                dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
                wedge_surface_physical_coords( wedge_phy_surf, grid, s, x_cell, y_cell );

                const ScalarT r_1 = radii( s, r_cell );
                const ScalarT r_2 = radii( s, r_cell + 1 );

                dense::Vec< ScalarT, 3 > qp[num_q];
                ScalarT                  qw[num_q];
                quadrature::quad_felippa_3x2_quad_points( qp );
                quadrature::quad_felippa_3x2_quad_weights( qw );

                dense::Vec< ScalarT, num_nodes_per_wedge > loc[num_wedges_per_hex_cell] = {};
                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
                    for ( int q = 0; q < num_q; ++q )
                    {
                        const auto    J   = jac( wedge_phy_surf[wedge], r_1, r_2, qp[q] );
                        const ScalarT det = Kokkos::abs( J.det() );
                        for ( int i = 0; i < num_nodes_per_wedge; ++i )
                            loc[wedge]( i ) += qw[q] * shape( i, qp[q] ) * det;
                    }
                atomically_add_local_wedge_scalar_coefficients( out_v, s, x_cell, y_cell, r_cell, loc );
            } );
        Kokkos::fence();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            terra::communication::shell::send_recv_with_plan( comm_plan_, out_v, recv_buffers_ );
        }
    }
};

static_assert( linalg::OperatorLike< ShearHeatingKerngen< float > > );
static_assert( linalg::OperatorLike< ShearHeatingKerngen< double > > );

} // namespace terra::fe::wedge::operators::shell
