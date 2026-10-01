#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/linear_form.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::linearforms::shell {

/// \brief Selects which heating source term of the energy equation a HeatingTerm assembles.
enum class HeatingTermKind
{
    /// \f$ f_i = -\int_E c \, (\mathbf{u} \cdot \hat{\mathbf{r}}) \, T \, \phi_i \, \mathrm{d}x \f$
    /// The scalar field passed to the operator is the temperature \f$ T \f$.
    Adiabatic,

    /// \f$ f_i = \int_E c \, 2\eta \, \varepsilon'(\mathbf{u}) : \varepsilon'(\mathbf{u}) \, \phi_i \, \mathrm{d}x \f$
    /// with \f$ \varepsilon' = \varepsilon(\mathbf{u}) - \tfrac{1}{3} (\nabla\cdot\mathbf{u}) I \f$.
    /// The scalar field passed to the operator is the viscosity \f$ \eta \f$.
    Shear
};

/// \brief Linear form for a heating source term in the energy equation (adiabatic or shear heating).
///
/// Both terms share all of the boilerplate (member data, apply/halo exchange, quadrature setup, geometry
/// gathering, local coefficient extraction, accumulation into the destination vector). They only differ in the
/// pointwise integrand evaluated at each quadrature point, which is selected at compile time through
/// \p Kind and branched on with `if constexpr` inside the wedge loop.
///
/// In both cases the integrand is multiplied by a coefficient \f$ c \f$ obtained from \p CoefficientT, which is
/// called as `coefficient( local_subdomain_id, x_cell, y_cell, r_cell, wedge, qp )`.
///
/// \tparam ScalarT         floating point type
/// \tparam CoefficientT    functor returning the coefficient \f$ c \f$ at a quadrature point
/// \tparam Kind            which heating term to assemble (see HeatingTermKind)
/// \tparam VelocityVecDim  number of velocity components (must be 3)
template < typename ScalarT, typename CoefficientT, HeatingTermKind Kind, int VelocityVecDim = 3 >
class HeatingTerm
{
    static_assert( VelocityVecDim == 3, "HeatingTerm requires a 3D velocity field." );

  public:
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    /// Scalar input field: temperature T (Adiabatic) or viscosity eta (Shear).
    linalg::VectorQ1Scalar< ScalarT >              field_;
    linalg::VectorQ1Vec< ScalarT, VelocityVecDim > velocity_;

    CoefficientT coefficient_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    // Kokkos views set in apply_impl() before the parallel launch.
    grid::Grid4DDataScalar< ScalarType >              dst_;
    grid::Grid4DDataScalar< ScalarType >              field_grid_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

    static constexpr const char* kernel_label()
    {
        if constexpr ( Kind == HeatingTermKind::Adiabatic )
        {
            return "adiabatic_heating_term";
        }
        else
        {
            return "shear_heating_term";
        }
    }

  public:
    /// \param field  temperature T for HeatingTermKind::Adiabatic, viscosity eta for HeatingTermKind::Shear
    HeatingTerm(
        const grid::shell::DistributedDomain&                 domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&              grid,
        const grid::Grid2DDataScalar< ScalarT >&              radii,
        const linalg::VectorQ1Scalar< ScalarT >&              field,
        const linalg::VectorQ1Vec< ScalarT, VelocityVecDim >& velocity,
        const CoefficientT&                                   coefficient,
        const linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        const linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , field_( field )
    , velocity_( velocity )
    , coefficient_( coefficient )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    void apply_impl( DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        dst_        = dst.grid_data();
        field_grid_ = field_.grid_data();
        vel_grid_   = velocity_.grid_data();

        Kokkos::parallel_for( kernel_label(), grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    /// \brief Kokkos kernel: per-cell contribution to the right-hand side (see HeatingTermKind for the formulas).
    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        // Quadrature points.
        constexpr int num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        {
            // Gather surface points for each wedge.
            dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

            // Gather wedge radii.
            const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
            const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

            // Local nodal values of the scalar field (T or eta) and of the velocity.
            dense::Vec< ScalarT, 6 > field[num_wedges_per_hex_cell];
            extract_local_wedge_scalar_coefficients( field, local_subdomain_id, x_cell, y_cell, r_cell, field_grid_ );

            dense::Vec< ScalarT, 6 > vel_coeffs[VelocityVecDim][num_wedges_per_hex_cell];
            for ( int d = 0; d < VelocityVecDim; d++ )
            {
                extract_local_wedge_vector_coefficients(
                    vel_coeffs[d], local_subdomain_id, x_cell, y_cell, r_cell, d, vel_grid_ );
            }

            // Assemble the local element vector.

            for ( int q = 0; q < num_quad_points; q++ )
            {
                const auto w  = quad_weights[q];
                const auto qp = quad_points[q];

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    const auto J   = jac( wedge_phy_surf[wedge], r_1, r_2, qp );
                    const auto det = Kokkos::abs( J.det() );

                    // Scalar field (T or eta) at the quadrature point.
                    ScalarT field_eval = 0.0;
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        field_eval += shape( j, qp ) * field[wedge]( j );
                    }

                    // Pointwise integrand (everything except the coefficient, the test function and the
                    // quadrature weight / Jacobian determinant). This is the only part that depends on Kind.
                    ScalarT integrand = 0.0;

                    if constexpr ( Kind == HeatingTermKind::Adiabatic )
                    {
                        ///////////////////////////////////////////////////////////////////////////////////////
                        // Adiabatic heating: -(u . r_hat) * T
                        // r_hat points radially outwards, i.e. against gravity.
                        ///////////////////////////////////////////////////////////////////////////////////////

                        dense::Vec< ScalarT, VelocityVecDim > vel_eval;
                        vel_eval.fill( 0.0 );

                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            const auto shape_j = shape( j, qp );

                            vel_eval( 0 ) += shape_j * vel_coeffs[0][wedge]( j );
                            vel_eval( 1 ) += shape_j * vel_coeffs[1][wedge]( j );
                            vel_eval( 2 ) += shape_j * vel_coeffs[2][wedge]( j );
                        }

                        const auto lat_dir = forward_map_lat(
                            wedge_phy_surf[wedge][0],
                            wedge_phy_surf[wedge][1],
                            wedge_phy_surf[wedge][2],
                            qp( 0 ),
                            qp( 1 ) );

                        const auto r_hat = lat_dir.normalized();

                        integrand = -r_hat.dot( vel_eval ) * field_eval;
                    }
                    else
                    {
                        ///////////////////////////////////////////////////////////////////////////////////////
                        // Shear heating: 2 * eta * eps' : eps'
                        // with eps = 1/2 (grad u + grad u^T) and eps' = eps - 1/3 (div u) I.
                        ///////////////////////////////////////////////////////////////////////////////////////

                        const auto J_inv_transposed = J.inv().transposed();

                        // du[a][b] = d u_a / d x_b
                        ScalarT du[3][3] = {};

                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            const auto grad_j = J_inv_transposed * grad_shape( j, qp );

                            for ( int a = 0; a < 3; a++ )
                            {
                                for ( int b = 0; b < 3; b++ )
                                {
                                    du[a][b] += grad_j( b ) * vel_coeffs[a][wedge]( j );
                                }
                            }
                        }

                        const ScalarT div_u = du[0][0] + du[1][1] + du[2][2];

                        // Frobenius norm squared of the deviatoric strain rate.
                        ScalarT eps_dev_sq = 0.0;
                        for ( int a = 0; a < 3; a++ )
                        {
                            for ( int b = 0; b < 3; b++ )
                            {
                                ScalarT eps_ab = 0.5 * ( du[a][b] + du[b][a] );
                                if ( a == b )
                                {
                                    eps_ab -= div_u / 3.0;
                                }
                                eps_dev_sq += eps_ab * eps_ab;
                            }
                        }

                        integrand = 2.0 * field_eval * eps_dev_sq;
                    }

                    const ScalarT coeff = coefficient_( local_subdomain_id, x_cell, y_cell, r_cell, wedge, qp );

                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        dst[wedge]( i ) += w * coeff * integrand * shape( i, qp ) * det;
                    }
                }
            }
        }

        {
            atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
        }
    }
};

/// \brief Adiabatic heating linear form. Takes the temperature as scalar field.
template < typename ScalarT, typename CoefficientT, int VelocityVecDim = 3 >
using AdiabaticHeatingTerm = HeatingTerm< ScalarT, CoefficientT, HeatingTermKind::Adiabatic, VelocityVecDim >;

/// \brief Shear (viscous dissipation) heating linear form. Takes the viscosity as scalar field.
template < typename ScalarT, typename CoefficientT, int VelocityVecDim = 3 >
using ShearHeatingTerm = HeatingTerm< ScalarT, CoefficientT, HeatingTermKind::Shear, VelocityVecDim >;

} // namespace terra::fe::wedge::linearforms::shell