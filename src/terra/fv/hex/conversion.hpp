
#pragma once
#include "communication/shell/fv_communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fv::hex {

template < typename T, typename ScalarType, typename GridScalarType >
concept FVProjectionFunctor = requires( const T& self, const dense::Vec< GridScalarType, 3 >& x ) {
    { self.operator()( x ) } -> std::same_as< ScalarType >;
};

/// @brief L2 projection of an analytical function into a finite volume function.
///
/// Use this function if you want to represent an analytical function with a finite volume function.
///
/// Given some function \f$u = u(x)\f$ (as a @ref FVProjectionFunctor), this function computes for every finite volume
/// cell \f$K\f$ the value
/// \f[
///     u_K = \frac{1}{|K|} \int_K u(x) \ dx
/// \f]
/// with
/// \f[
///     |K| = \int_K 1 \ dx
/// \f]
/// and writes \f$u_K\f$ into the respective finite volume cell.
///
/// @note The `operator()` method of the functor must be annotated with `KOKKOS_INLINE_FUNCTION`. Otherwise, this might
/// not work on GPUs.
///
/// @param dst [out] finite volume function that is being written to
/// @param src functor that implements the @ref FVProjectionFunctor concept
/// @param coords_shell coords of the unit shell
/// @param coords_radii radii of all shells
template < typename ScalarType, typename GridScalarType, FVProjectionFunctor< ScalarType, GridScalarType > Functor >
void l2_project_analytical_to_fv(
    linalg::VectorFVScalar< ScalarType >&           dst,
    const Functor&                                  src,
    const grid::Grid3DDataVec< GridScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< GridScalarType >& coords_radii )
{
    grid::Grid4DDataScalar< ScalarType > fv_grid = dst.grid_data();
    Kokkos::parallel_for(
        "l2_project_into_fv_function",
        Kokkos::MDRangePolicy(
            { 0, 1, 1, 1 },
            { fv_grid.extent( 0 ), fv_grid.extent( 1 ) - 1, fv_grid.extent( 2 ) - 1, fv_grid.extent( 3 ) - 1 } ),
        KOKKOS_LAMBDA(
            const int local_subdomain_id, const int hex_cell_x, const int hex_cell_y, const int hex_cell_r ) {
            constexpr auto              num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;
            dense::Vec< ScalarType, 3 > quad_points[num_quad_points];
            ScalarType                  quad_weights[num_quad_points];
            fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
            fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

            ScalarType volume   = 0.0;
            ScalarType integral = 0.0;

            dense::Vec< ScalarType, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                                      [fe::wedge::num_nodes_per_wedge_surface] = {};
            fe::wedge::wedge_surface_physical_coords(
                wedge_phy_surf, coords_shell, local_subdomain_id, hex_cell_x - 1, hex_cell_y - 1 );

            const auto r_1 = coords_radii( local_subdomain_id, hex_cell_r - 1 );
            const auto r_2 = coords_radii( local_subdomain_id, hex_cell_r );

            for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < num_quad_points; ++q )
                {
                    const auto J       = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                    const auto det     = J.det();
                    const auto abs_det = Kokkos::abs( det );
                    const auto Jw      = abs_det * quad_weights[q];
                    volume += Jw;

                    const auto x = fe::wedge::forward_map(
                        wedge_phy_surf[wedge][0],
                        wedge_phy_surf[wedge][1],
                        wedge_phy_surf[wedge][2],
                        r_1,
                        r_2,
                        quad_points[q]( 0 ),
                        quad_points[q]( 1 ),
                        quad_points[q]( 2 ) );

                    integral += src( x ) * Jw;
                }
            }

            fv_grid( local_subdomain_id, hex_cell_x, hex_cell_y, hex_cell_r ) = integral / volume;
        } );
}

/// @brief L2 projection from a finite volume function into a Q1 (wedge) finite element function.
///
/// Use this function if you want to convert from the finite volume to the finite element space.
///
/// Given some finite volume function \f$u_h^{FV}\f$ this function solves
/// \f[
///     \int_\Omega u_h^{FE} v \ d\Omega = \int_\Omega u_h^{FV} v \ d\Omega, \quad \forall v \in V_h
/// \f]
/// for \f$u_h^{FE}\f$. In matrix form:
/// \f[
///     M u^{FE} = b
/// \f]
/// where \f$M\f$ is the finite element mass matrix and
/// \f[
///     b_i = \int_\Omega u^{FV}(x) \, \phi_i(x) \ dx = \sum_K u^{FV}_K \int_\Omega \phi_i(x) \ dx.
/// \f]
///
/// Internally, this function
///     1. sets up \f$b\f$ locally
///     2. communicates \f$b\f$ to complete the assembly
///     3. solves the mass matrix system with CG
///
/// @param dst [out] finite element function that is the solution of the mass matrix system
/// @param src finite volume vector to convert
/// @param domain distributed domain (required for communication)
/// @param coords_shell coords of the unit shell
/// @param coords_radii radii of all shells
/// @param tmps at least 5 tmp vectors (required for \f$b\f$ and the CG solver)
template < typename ScalarType, typename GridScalarType >
void l2_project_fv_to_fe(
    linalg::VectorQ1Scalar< ScalarType >&                dst,
    const linalg::VectorFVScalar< ScalarType >&          src,
    const grid::shell::DistributedDomain&                domain,
    const grid::Grid3DDataVec< GridScalarType, 3 >&      coords_shell,
    const grid::Grid2DDataScalar< GridScalarType >&      coords_radii,
    std::vector< linalg::VectorQ1Scalar< ScalarType > >& tmps )
{
    if ( tmps.size() < 5 )
    {
        Kokkos::abort( "At least 5 tmp vectors required." );
    }

    auto b = tmps[4];

    linalg::assign( b, 0.0 );

    grid::Grid4DDataScalar< ScalarType > fv_grid = src.grid_data();
    grid::Grid4DDataScalar< ScalarType > b_grid  = b.grid_data();

    Kokkos::parallel_for(
        "l2_project_fv_to_fe_rhs_assembly",
        Kokkos::MDRangePolicy(
            { 0, 1, 1, 1 },
            { fv_grid.extent( 0 ), fv_grid.extent( 1 ) - 1, fv_grid.extent( 2 ) - 1, fv_grid.extent( 3 ) - 1 } ),
        KOKKOS_LAMBDA(
            const int local_subdomain_id, const int hex_cell_x, const int hex_cell_y, const int hex_cell_r ) {
            const auto hex_cell_x_no_gl = hex_cell_x - 1;
            const auto hex_cell_y_no_gl = hex_cell_y - 1;
            const auto hex_cell_r_no_gl = hex_cell_r - 1;

            constexpr auto              num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;
            dense::Vec< ScalarType, 3 > quad_points[num_quad_points];
            ScalarType                  quad_weights[num_quad_points];
            fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
            fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

            dense::Vec< ScalarType, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                                      [fe::wedge::num_nodes_per_wedge_surface] = {};
            fe::wedge::wedge_surface_physical_coords(
                wedge_phy_surf, coords_shell, local_subdomain_id, hex_cell_x - 1, hex_cell_y - 1 );

            const auto r_1 = coords_radii( local_subdomain_id, hex_cell_r - 1 );
            const auto r_2 = coords_radii( local_subdomain_id, hex_cell_r );

            const auto fv_value = fv_grid( local_subdomain_id, hex_cell_x, hex_cell_y, hex_cell_r );

            dense::Vec< ScalarType, 6 > b_local[fe::wedge::num_wedges_per_hex_cell] = {};

            for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
            {
                for ( int i = 0; i < fe::wedge::num_nodes_per_wedge; ++i )
                {
                    for ( int q = 0; q < num_quad_points; ++q )
                    {
                        const auto J         = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                        const auto det       = J.det();
                        const auto abs_det   = Kokkos::abs( det );
                        const auto shape_i_q = fe::wedge::shape( i, quad_points[q] );

                        b_local[wedge]( i ) += shape_i_q * abs_det * quad_weights[q];
                    }

                    b_local[wedge]( i ) *= fv_value;
                }
            }

            fe::wedge::atomically_add_local_wedge_scalar_coefficients(
                b_grid, local_subdomain_id, hex_cell_x_no_gl, hex_cell_y_no_gl, hex_cell_r_no_gl, b_local );
        } );

    communication::shell::send_recv( domain, b_grid );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;
    Mass M( domain, coords_shell, coords_radii );

    linalg::solvers::PCG< Mass > solver(
        linalg::solvers::IterativeSolverParameters( 1000, 1e-12, 1e-12 ), nullptr, tmps );

    linalg::solvers::solve( solver, M, dst, b );
}

/// @brief Bound-preserving FV-to-FE transfer via a lumped mass matrix.
///
/// Computes, for every Q1 node \f$i\f$,
/// \f[
///     u_i^{FE} = \frac{b_i}{D_i},
///     \quad
///     b_i   = \sum_K u_K^{FV} \int_K \phi_i \, dx,
///     \quad
///     D_i   = \sum_K          \int_K \phi_i \, dx
/// \f]
/// where \f$\phi_i\f$ are the Q1 wedge shape functions.  Because every shape
/// function is non-negative and the shape functions form a partition of unity,
/// \f$u_i^{FE}\f$ is a convex combination of the surrounding FV cell values.
/// Consequently this projection is **monotone**: if \f$u^{FV} \in [a, b]\f$
/// then \f$u^{FE} \in [a, b]\f$, with no Gibbs-type over- or undershoots.
///
/// Compare with @ref l2_project_fv_to_fe which uses a consistent mass matrix
/// and solves a global linear system.  The lumped variant is cheaper (no
/// solver, no MPI reduction for convergence) but slightly less L2-accurate in
/// smooth cases.  It is the preferred choice when bound-preservation matters
/// (e.g.\ compositional fields, density anomalies that feed directly into
/// buoyancy).
///
/// @param dst    [out] FE scalar field receiving the projected values.
/// @param src    FV scalar field to project.
/// @param domain Distributed domain (required for MPI ghost communication).
/// @param coords_shell Lateral node coordinates of the unit-sphere surface.
/// @param coords_radii Radial shell radii.
/// @param tmps   At least 2 temporary Q1 vectors (used for \f$b\f$ and \f$D\f$).
template < typename ScalarType, typename GridScalarType >
void l2_project_fv_to_fe_lumped(
    linalg::VectorQ1Scalar< ScalarType >&                dst,
    const linalg::VectorFVScalar< ScalarType >&          src,
    const grid::shell::DistributedDomain&                domain,
    const grid::Grid3DDataVec< GridScalarType, 3 >&      coords_shell,
    const grid::Grid2DDataScalar< GridScalarType >&      coords_radii,
    std::vector< linalg::VectorQ1Scalar< ScalarType > >& tmps )
{
    if ( tmps.size() < 2 )
    {
        Kokkos::abort( "At least 2 tmp vectors required." );
    }

    auto b = tmps[0]; // numerator:   sum_K u_K * integral(phi_i, K)
    auto D = tmps[1]; // denominator: sum_K       integral(phi_i, K)  (lumped diagonal)

    linalg::assign( b, ScalarType( 0 ) );
    linalg::assign( D, ScalarType( 0 ) );

    grid::Grid4DDataScalar< ScalarType > fv_grid = src.grid_data();
    grid::Grid4DDataScalar< ScalarType > b_grid  = b.grid_data();
    grid::Grid4DDataScalar< ScalarType > d_grid  = D.grid_data();

    Kokkos::parallel_for(
        "l2_project_fv_to_fe_lumped_assembly",
        Kokkos::MDRangePolicy(
            { 0, 1, 1, 1 },
            { fv_grid.extent( 0 ), fv_grid.extent( 1 ) - 1, fv_grid.extent( 2 ) - 1, fv_grid.extent( 3 ) - 1 } ),
        KOKKOS_LAMBDA(
            const int local_subdomain_id, const int hex_cell_x, const int hex_cell_y, const int hex_cell_r ) {
            const auto hex_cell_x_no_gl = hex_cell_x - 1;
            const auto hex_cell_y_no_gl = hex_cell_y - 1;
            const auto hex_cell_r_no_gl = hex_cell_r - 1;

            constexpr auto              num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;
            dense::Vec< ScalarType, 3 > quad_points[num_quad_points];
            ScalarType                  quad_weights[num_quad_points];
            fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
            fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

            dense::Vec< ScalarType, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                                      [fe::wedge::num_nodes_per_wedge_surface] = {};
            fe::wedge::wedge_surface_physical_coords(
                wedge_phy_surf, coords_shell, local_subdomain_id, hex_cell_x - 1, hex_cell_y - 1 );

            const auto r_1 = coords_radii( local_subdomain_id, hex_cell_r - 1 );
            const auto r_2 = coords_radii( local_subdomain_id, hex_cell_r );

            const auto fv_value = fv_grid( local_subdomain_id, hex_cell_x, hex_cell_y, hex_cell_r );

            // d_local accumulates integral(phi_i, K); b_local = fv_value * d_local
            dense::Vec< ScalarType, 6 > b_local[fe::wedge::num_wedges_per_hex_cell] = {};
            dense::Vec< ScalarType, 6 > d_local[fe::wedge::num_wedges_per_hex_cell] = {};

            for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
            {
                for ( int i = 0; i < fe::wedge::num_nodes_per_wedge; ++i )
                {
                    for ( int q = 0; q < num_quad_points; ++q )
                    {
                        const auto J        = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                        const auto abs_det  = Kokkos::abs( J.det() );
                        const auto phi_i_Jw = fe::wedge::shape( i, quad_points[q] ) * abs_det * quad_weights[q];

                        d_local[wedge]( i ) += phi_i_Jw;
                    }

                    b_local[wedge]( i ) = fv_value * d_local[wedge]( i );
                }
            }

            fe::wedge::atomically_add_local_wedge_scalar_coefficients(
                b_grid, local_subdomain_id, hex_cell_x_no_gl, hex_cell_y_no_gl, hex_cell_r_no_gl, b_local );
            fe::wedge::atomically_add_local_wedge_scalar_coefficients(
                d_grid, local_subdomain_id, hex_cell_x_no_gl, hex_cell_y_no_gl, hex_cell_r_no_gl, d_local );
        } );

    communication::shell::send_recv( domain, b_grid );
    communication::shell::send_recv( domain, d_grid );

    // dst = b / D  (element-wise)
    linalg::assign( dst, b );
    linalg::invert_entries( D );
    linalg::scale_in_place( dst, D );
}

/// @brief L2 projection from a Q1 finite element function into a finite volume function.
///
/// For each FV cell \f$K\f$ computes the exact volume-weighted cell average of the Q1 field:
/// \f[
///     u_K^{FV} = \frac{1}{|K|} \int_K u_h^{FE}(x) \, dx
///              = \frac{1}{|K|} \int_K \sum_i u_i \, \phi_i(\xi(x)) \, dx
/// \f]
/// where \f$\phi_i\f$ are the wedge Q1 shape functions and \f$u_i\f$ are the Q1 nodal values
/// surrounding the cell.  The integral is evaluated with the same Felippa quadrature rule
/// used throughout the codebase, so the result is consistent with `l2_project_fv_to_fe`.
///
/// Ghost layers of `dst` are populated via MPI exchange on exit so the result is immediately
/// usable as input to FCT kernels.
///
/// @param dst          [out] FV scalar field.
/// @param src          Q1 finite element scalar field (read-only).
/// @param domain       Distributed domain (used for MPI ghost-layer exchange).
/// @param coords_shell Lateral node coordinates of the unit-sphere surface.
/// @param coords_radii Radial shell radii.
template < typename ScalarType, typename GridScalarType >
void l2_project_fe_to_fv(
    linalg::VectorFVScalar< ScalarType >&                     dst,
    const linalg::VectorQ1Scalar< ScalarType >&               src,
    const grid::shell::DistributedDomain&                     domain,
    const grid::Grid3DDataVec< GridScalarType, 3 >&           coords_shell,
    const grid::Grid2DDataScalar< GridScalarType >&           coords_radii )
{
    grid::Grid4DDataScalar< ScalarType > fv_grid  = dst.grid_data();
    grid::Grid4DDataScalar< ScalarType > q1_grid  = src.grid_data();

    Kokkos::parallel_for(
        "l2_project_fe_to_fv",
        Kokkos::MDRangePolicy(
            { 0, 1, 1, 1 },
            { fv_grid.extent( 0 ), fv_grid.extent( 1 ) - 1, fv_grid.extent( 2 ) - 1, fv_grid.extent( 3 ) - 1 } ),
        KOKKOS_LAMBDA(
            const int local_subdomain_id, const int hex_cell_x, const int hex_cell_y, const int hex_cell_r ) {
            // Q1 cell coordinates (0-indexed, no ghost-layer offset).
            const auto x_no_gl = hex_cell_x - 1;
            const auto y_no_gl = hex_cell_y - 1;
            const auto r_no_gl = hex_cell_r - 1;

            constexpr auto              num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;
            dense::Vec< ScalarType, 3 > quad_points[num_quad_points];
            ScalarType                  quad_weights[num_quad_points];
            fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
            fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

            // Gather Q1 nodal values for the two wedges of this hex cell.
            dense::Vec< ScalarType, 6 > local_u[fe::wedge::num_wedges_per_hex_cell] = {};
            fe::wedge::extract_local_wedge_scalar_coefficients(
                local_u, local_subdomain_id, x_no_gl, y_no_gl, r_no_gl, q1_grid );

            dense::Vec< ScalarType, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                                      [fe::wedge::num_nodes_per_wedge_surface] = {};
            fe::wedge::wedge_surface_physical_coords(
                wedge_phy_surf, coords_shell, local_subdomain_id, x_no_gl, y_no_gl );

            const auto r_1 = coords_radii( local_subdomain_id, r_no_gl );
            const auto r_2 = coords_radii( local_subdomain_id, hex_cell_r );

            ScalarType volume   = ScalarType( 0 );
            ScalarType integral = ScalarType( 0 );

            for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
            {
                for ( int q = 0; q < num_quad_points; ++q )
                {
                    const auto J       = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                    const auto abs_det = Kokkos::abs( J.det() );
                    const auto Jw      = abs_det * quad_weights[q];
                    volume += Jw;

                    // Evaluate Q1 function at the quadrature point via shape functions.
                    ScalarType u_q = ScalarType( 0 );
                    for ( int i = 0; i < fe::wedge::num_nodes_per_wedge; ++i )
                        u_q += local_u[wedge]( i ) * fe::wedge::shape< ScalarType >( i, quad_points[q] );

                    integral += u_q * Jw;
                }
            }

            fv_grid( local_subdomain_id, hex_cell_x, hex_cell_y, hex_cell_r ) = integral / volume;
        } );

    Kokkos::fence();
    communication::shell::update_fv_ghost_layers( domain, fv_grid );
}

} // namespace terra::fv::hex