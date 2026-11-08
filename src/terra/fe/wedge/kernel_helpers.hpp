#pragma once

namespace terra::fe::wedge {

constexpr int num_wedges_per_hex_cell     = 2;
constexpr int num_nodes_per_wedge_surface = 3;
constexpr int num_nodes_per_wedge         = 6;

/*! @brief Extracts the (unit sphere) surface vertex coords of the two wedges of a hex cell.
   
    Useful for wedge-based kernels that update two wedges that make up a hex cell at once.
   
    \code
    2--3
    |\ |
    | \|   =>  [(p0, p1, p2), (p3, p2, p1)]
    0--1
    \endcode
   
    @param wedge_surf_phy_coords [out] first dim: wedge/triangle index, second dim: vertex index
    @param lateral_grid          [in]  the unit sphere vertex coordinates
    @param local_subdomain_id    [in]  shell subdomain id on this process
    @param x_cell                [in]  hex cell x-coordinate
    @param y_cell                [in]  hex cell y-coordinate
*/
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION void wedge_surface_physical_coords(
    dense::Vec< T, 3 > ( &wedge_surf_phy_coords )[num_wedges_per_hex_cell][num_nodes_per_wedge_surface],
    const grid::Grid3DDataVec< T, 3 >& lateral_grid,
    const int                          local_subdomain_id,
    const int                          x_cell,
    const int                          y_cell )
{
    // Extract vertex positions of quad
    // (0, 0), (1, 0), (0, 1), (1, 1).
    dense::Vec< T, 3 > quad_surface_coords[2][2];

    for ( int x = x_cell; x <= x_cell + 1; x++ )
    {
        for ( int y = y_cell; y <= y_cell + 1; y++ )
        {
            for ( int d = 0; d < 3; d++ )
            {
                quad_surface_coords[x - x_cell][y - y_cell]( d ) = lateral_grid( local_subdomain_id, x, y, d );
            }
        }
    }

    // Sort coords for the two wedge surfaces.
    wedge_surf_phy_coords[0][0] = quad_surface_coords[0][0];
    wedge_surf_phy_coords[0][1] = quad_surface_coords[1][0];
    wedge_surf_phy_coords[0][2] = quad_surface_coords[0][1];

    wedge_surf_phy_coords[1][0] = quad_surface_coords[1][1];
    wedge_surf_phy_coords[1][1] = quad_surface_coords[0][1];
    wedge_surf_phy_coords[1][2] = quad_surface_coords[1][0];
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION void wedge_0_surface_physical_coords(
    dense::Vec< T, 3 >*                wedge_surf_phy_coords,
    const grid::Grid3DDataVec< T, 3 >& lateral_grid,
    const int                          local_subdomain_id,
    const int                          x_cell,
    const int                          y_cell )
{
    // Extract vertex positions of quad
    // (0, 0), (1, 0), (0, 1), (1, 1).
    dense::Vec< T, 3 > quad_surface_coords[2][2];

    for ( int x = x_cell; x <= x_cell + 1; x++ )
    {
        for ( int y = y_cell; y <= y_cell + 1; y++ )
        {
            for ( int d = 0; d < 3; d++ )
            {
                quad_surface_coords[x - x_cell][y - y_cell]( d ) = lateral_grid( local_subdomain_id, x, y, d );
            }
        }
    }

    // Sort coords for the two wedge surfaces.
    wedge_surf_phy_coords[0] = quad_surface_coords[0][0];
    wedge_surf_phy_coords[1] = quad_surface_coords[1][0];
    wedge_surf_phy_coords[2] = quad_surface_coords[0][1];
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION void wedge_1_surface_physical_coords(
    dense::Vec< T, 3 >*                wedge_surf_phy_coords,
    const grid::Grid3DDataVec< T, 3 >& lateral_grid,
    const int                          local_subdomain_id,
    const int                          x_cell,
    const int                          y_cell )
{
    // Extract vertex positions of quad
    // (0, 0), (1, 0), (0, 1), (1, 1).
    dense::Vec< T, 3 > quad_surface_coords[2][2];

    for ( int x = x_cell; x <= x_cell + 1; x++ )
    {
        for ( int y = y_cell; y <= y_cell + 1; y++ )
        {
            for ( int d = 0; d < 3; d++ )
            {
                quad_surface_coords[x - x_cell][y - y_cell]( d ) = lateral_grid( local_subdomain_id, x, y, d );
            }
        }
    }

    // Sort coords for the two wedge surfaces.
    wedge_surf_phy_coords[0] = quad_surface_coords[1][1];
    wedge_surf_phy_coords[1] = quad_surface_coords[0][1];
    wedge_surf_phy_coords[2] = quad_surface_coords[1][0];
}

/// @brief Computes the transposed inverse of the Jacobian of the lateral forward map from the reference triangle
///        to the triangle on the unit sphere and the absolute determinant of that Jacobian at the passed quadrature
///        points.
///
/// @param jac_lat_inv_t           [out] transposed inverse of the Jacobian of the lateral map
/// @param det_jac_lat             [out] absolute of the determinant of the Jacobian of the lateral map
/// @param wedge_surf_phy_coords   [in]  coords of the triangular wedge surfaces on the unit sphere (compute via
///                                      wedge_surface_physical_coords())
/// @param quad_points             [in]  the quadrature points
template < std::floating_point T, int NumQuadPoints >
KOKKOS_INLINE_FUNCTION constexpr void jacobian_lat_inverse_transposed_and_determinant(
    dense::Mat< T, 3, 3 > ( &jac_lat_inv_t )[num_wedges_per_hex_cell][NumQuadPoints],
    T ( &det_jac_lat )[num_wedges_per_hex_cell][NumQuadPoints],
    const dense::Vec< T, 3 > wedge_surf_phy_coords[num_wedges_per_hex_cell][num_nodes_per_wedge_surface],
    const dense::Vec< T, 3 > quad_points[NumQuadPoints] )
{
    for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
    {
        for ( int q = 0; q < NumQuadPoints; q++ )
        {
            const auto jac_lat = wedge::jac_lat(
                wedge_surf_phy_coords[wedge][0],
                wedge_surf_phy_coords[wedge][1],
                wedge_surf_phy_coords[wedge][2],
                quad_points[q]( 0 ),
                quad_points[q]( 1 ) );

            det_jac_lat[wedge][q] = Kokkos::abs( jac_lat.det() );

            jac_lat_inv_t[wedge][q] = jac_lat.inv().transposed();
        }
    }
}

/// @brief Like jacobian_lat_inverse_transposed_and_determinant() but only computes the determinant (cheaper if the
///        transposed inverse of the Jacobian is not required).
///
/// @param det_jac_lat             [out] absolute of the determinant of the Jacobian of the lateral map
/// @param wedge_surf_phy_coords   [in]  coords of the triangular wedge surfaces on the unit sphere (compute via
///                                      wedge_surface_physical_coords())
/// @param quad_points             [in]  the quadrature points
template < std::floating_point T, int NumQuadPoints >
KOKKOS_INLINE_FUNCTION constexpr void jacobian_lat_determinant(
    T ( &det_jac_lat )[num_wedges_per_hex_cell][NumQuadPoints],
    const dense::Vec< T, 3 > wedge_surf_phy_coords[num_wedges_per_hex_cell][num_nodes_per_wedge_surface],
    const dense::Vec< T, 3 > quad_points[NumQuadPoints] )
{
    for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
    {
        for ( int q = 0; q < NumQuadPoints; q++ )
        {
            const auto jac_lat = wedge::jac_lat(
                wedge_surf_phy_coords[wedge][0],
                wedge_surf_phy_coords[wedge][1],
                wedge_surf_phy_coords[wedge][2],
                quad_points[q]( 0 ),
                quad_points[q]( 1 ) );

            det_jac_lat[wedge][q] = Kokkos::abs( jac_lat.det() );
        }
    }
}

/// @brief Computes the radially independent parts of the physical shape function gradients
///
/// \code
///   g_rad_j = jac_lat_inv_t * ( (∂/∂xi N_lat_j) N_rad_j, (∂/∂eta N_lat_j) N_rad_j, 0       )^T
///   g_lat_j = jac_lat_inv_t * (                       0,                        0, N_lat_j )^T
/// \endcode
///
/// where j is a node of the wedge. Computes those for all 6 nodes j = 0, ..., 5 of a wedge.
///
/// Those terms can technically be precomputed and are independent of r (identical for all wedges on a radial beam).
///
/// From thes we can later compute
///
/// \code
///   jac_inv_t * grad_N_j
/// \endcode
///
/// via
///
/// \code
///   jac_inv_t * grad_N_j = (1 / r(zeta)) g_rad_j + (∂ N_rad_j / ∂zeta) * (1 / (∂r / ∂zeta)) * g_lat_j
/// \endcode
///
/// @param g_rad         [out] g_rad - see above
/// @param g_lat         [out] g_lat - see above
/// @param jac_lat_inv_t [in]  transposed inverse of lateral Jacobian - see
///                            jacobian_lat_inverse_transposed_and_determinant()
/// @param quad_points   [in]  the quadrature points
template < std::floating_point T, int NumQuadPoints >
KOKKOS_INLINE_FUNCTION constexpr void lateral_parts_of_grad_phi(
    dense::Vec< T, 3 > ( &g_rad )[num_wedges_per_hex_cell][num_nodes_per_wedge][NumQuadPoints],
    dense::Vec< T, 3 > ( &g_lat )[num_wedges_per_hex_cell][num_nodes_per_wedge][NumQuadPoints],
    const dense::Mat< T, 3, 3 > jac_lat_inv_t[num_wedges_per_hex_cell][NumQuadPoints],
    const dense::Vec< T, 3 >    quad_points[NumQuadPoints] )
{
    for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
    {
        for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
        {
            for ( int q = 0; q < NumQuadPoints; q++ )
            {
                g_rad[wedge][node_idx][q] =
                    jac_lat_inv_t[wedge][q] *
                    dense::Vec< T, 3 >{
                        grad_shape_lat_xi< T >( node_idx ) * shape_rad( node_idx, quad_points[q]( 2 ) ),
                        grad_shape_lat_eta< T >( node_idx ) * shape_rad( node_idx, quad_points[q]( 2 ) ),
                        0.0 };

                g_lat[wedge][node_idx][q] =
                    jac_lat_inv_t[wedge][q] *
                    dense::Vec< T, 3 >{ 0.0, 0.0, shape_lat( node_idx, quad_points[q]( 0 ), quad_points[q]( 1 ) ) };
            }
        }
    }
}

/// @brief Computes and returns J^-T grad(N_j).
///
/// @param g_rad          [in] see lateral_parts_of_grad_phi()
/// @param g_lat          [in] see lateral_parts_of_grad_phi()
/// @param r_inv          [in] 1 / r (where r is the physical radius of the quadrature point - compute with
///                            forward_map_rad())
/// @param grad_r_inv     [in] 1 / grad_r (where grad_r is the radial component of the gradient of the forward map in
///                            radial direction - compute with grad_forward_map_rad())
/// @param wedge_idx      [in] wedge index of the hex cell (0 or 1)
/// @param node_idx       [in] node index in the wedge (0, 1, ..., 5)
/// @param quad_point_idx [in] index of the quadrature point
template < std::floating_point T, int NumQuadPoints >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< T, 3 > grad_shape_full(
    const dense::Vec< T, 3 > g_rad[num_wedges_per_hex_cell][num_nodes_per_wedge][NumQuadPoints],
    const dense::Vec< T, 3 > g_lat[num_wedges_per_hex_cell][num_nodes_per_wedge][NumQuadPoints],
    const T                  r_inv,
    const T                  grad_r_inv,
    const int                wedge_idx,
    const int                node_idx,
    const int                quad_point_idx )
{
    return r_inv * g_rad[wedge_idx][node_idx][quad_point_idx] +
           grad_shape_rad< T >( node_idx ) * grad_r_inv * g_lat[wedge_idx][node_idx][quad_point_idx];
}

/// @brief Computes |det(J)|.
///
/// @param det_jac_lat    [in] see jacobian_lat_determinant()
/// @param r              [in] the physical radius of the quadrature point - compute with
///                            forward_map_rad()
/// @param grad_r         [in] the radial component of the gradient of the forward map in
///                            radial direction - compute with grad_forward_map_rad())
/// @param wedge_idx      [in] wedge index of the hex cell (0 or 1)
/// @param quad_point_idx [in] index of the quadrature point
template < std::floating_point T, int NumQuadPoints >
KOKKOS_INLINE_FUNCTION constexpr T det_full(
    const T   det_jac_lat[num_wedges_per_hex_cell][NumQuadPoints],
    const T   r,
    const T   grad_r,
    const int wedge_idx,
    const int quad_point_idx )
{
    return r * r * grad_r * det_jac_lat[wedge_idx][quad_point_idx];
}

/// @brief Extracts the local vector coefficients for the two wedges of a hex cell from the global coefficient vector.
///
/// \code
/// r = r_cell + 1 (outer)
/// 6--7
/// |\ |
/// | \|
/// 4--5
///
/// r = r_cell (inner)
/// 2--3
/// |\ |
/// | \|
/// 0--1
///
/// v0 = (0, 1, 2, 4, 5, 6)
/// v1 = (3, 2, 1, 7, 6, 5)
/// \endcode
///
/// @param local_coefficients  [out] the local coefficient vector
/// @param local_subdomain_id  [in]  shell subdomain id on this process
/// @param x_cell              [in]  hex cell x-coordinate
/// @param y_cell              [in]  hex cell y-coordinate
/// @param r_cell              [in]  hex cell r-coordinate
/// @param global_coefficients [in]  the global coefficient vector
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION void extract_local_wedge_scalar_coefficients(
    dense::Vec< T, 6 > ( &local_coefficients )[2],
    const int                          local_subdomain_id,
    const int                          x_cell,
    const int                          y_cell,
    const int                          r_cell,
    const grid::Grid4DDataScalar< T >& global_coefficients )
{
    local_coefficients[0]( 0 ) = global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell );
    local_coefficients[0]( 1 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell );
    local_coefficients[0]( 2 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell );
    local_coefficients[0]( 3 ) = global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell + 1 );
    local_coefficients[0]( 4 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
    local_coefficients[0]( 5 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );

    local_coefficients[1]( 0 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell );
    local_coefficients[1]( 1 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell );
    local_coefficients[1]( 2 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell );
    local_coefficients[1]( 3 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 );
    local_coefficients[1]( 4 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
    local_coefficients[1]( 5 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
}

/// @brief Extracts the local vector coefficients for the two wedges of a hex cell from the global coefficient vector.
///
/// \code
/// r = r_cell + 1 (outer)
/// 6--7
/// |\ |
/// | \|
/// 4--5
///
/// r = r_cell (inner)
/// 2--3
/// |\ |
/// | \|
/// 0--1
///
/// v0 = (0, 1, 2, 4, 5, 6)
/// v1 = (3, 2, 1, 7, 6, 5)
/// \endcode
///
/// @param local_coefficients  [out] the local coefficient vector
/// @param local_subdomain_id  [in]  shell subdomain id on this process
/// @param x_cell              [in]  hex cell x-coordinate
/// @param y_cell              [in]  hex cell y-coordinate
/// @param r_cell              [in]  hex cell r-coordinate
/// @param d                   [in]  vector-element of the vector-valued global view
/// @param global_coefficients [in]  the global coefficient vector
template < std::floating_point T, int VecDim >
KOKKOS_INLINE_FUNCTION void extract_local_wedge_vector_coefficients(
    dense::Vec< T, 6 > ( &local_coefficients )[2],
    const int                               local_subdomain_id,
    const int                               x_cell,
    const int                               y_cell,
    const int                               r_cell,
    const int                               d,
    const grid::Grid4DDataVec< T, VecDim >& global_coefficients )
{
    local_coefficients[0]( 0 ) = global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell, d );
    local_coefficients[0]( 1 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell, d );
    local_coefficients[0]( 2 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell, d );
    local_coefficients[0]( 3 ) = global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell + 1, d );
    local_coefficients[0]( 4 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, d );
    local_coefficients[0]( 5 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, d );

    local_coefficients[1]( 0 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, d );
    local_coefficients[1]( 1 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell, d );
    local_coefficients[1]( 2 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell, d );
    local_coefficients[1]( 3 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, d );
    local_coefficients[1]( 4 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, d );
    local_coefficients[1]( 5 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, d );
}

/// @brief Performs an atomic add of the two local wedge coefficient vectors of a hex cell into the global coefficient
/// vector.
///
/// \code
/// r = r_cell + 1 (outer)
/// 6--7
/// |\ |
/// | \|
/// 4--5
///
/// r = r_cell (inner)
/// 2--3
/// |\ |
/// | \|
/// 0--1
///
/// v0 = (0, 1, 2, 4, 5, 6)
/// v1 = (3, 2, 1, 7, 6, 5)
/// \endcode
///
/// @param global_coefficients [inout] the global coefficient vector
/// @param local_subdomain_id  [in]    shell subdomain id on this process
/// @param x_cell              [in]    hex cell x-coordinate
/// @param y_cell              [in]    hex cell y-coordinate
/// @param r_cell              [in]    hex cell r-coordinate
/// @param local_coefficients  [in]    the local coefficient vector
template < std::floating_point T >
KOKKOS_INLINE_FUNCTION void atomically_add_local_wedge_scalar_coefficients(
    const grid::Grid4DDataScalar< T >& global_coefficients,
    const int                          local_subdomain_id,
    const int                          x_cell,
    const int                          y_cell,
    const int                          r_cell,
    const dense::Vec< T, 6 > ( &local_coefficients )[2] )
{
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell ), local_coefficients[0]( 0 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell ),
        local_coefficients[0]( 1 ) + local_coefficients[1]( 2 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell ),
        local_coefficients[0]( 2 ) + local_coefficients[1]( 1 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), local_coefficients[0]( 3 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ),
        local_coefficients[0]( 4 ) + local_coefficients[1]( 5 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ),
        local_coefficients[0]( 5 ) + local_coefficients[1]( 4 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell ), local_coefficients[1]( 0 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 ), local_coefficients[1]( 3 ) );
}

/// @brief Performs an atomic add of the two local wedge coefficient vectors of a hex cell into the global coefficient
/// vector.
///
/// \code
/// r = r_cell + 1 (outer)
/// 6--7
/// |\ |
/// | \|
/// 4--5
///
/// r = r_cell (inner)
/// 2--3
/// |\ |
/// | \|
/// 0--1
///
/// v0 = (0, 1, 2, 4, 5, 6)
/// v1 = (3, 2, 1, 7, 6, 5)
/// \endcode
///
/// @param global_coefficients [inout] the global coefficient vector
/// @param local_subdomain_id  [in]    shell subdomain id on this process
/// @param x_cell              [in]    hex cell x-coordinate
/// @param y_cell              [in]    hex cell y-coordinate
/// @param r_cell              [in]    hex cell r-coordinate
/// @param d                   [in]    vector-element of the vector-valued global view
/// @param local_coefficients  [in]    the local coefficient vector
template < std::floating_point T, int VecDim >
KOKKOS_INLINE_FUNCTION void atomically_add_local_wedge_vector_coefficients(
    const grid::Grid4DDataVec< T, VecDim >& global_coefficients,
    const int                               local_subdomain_id,
    const int                               x_cell,
    const int                               y_cell,
    const int                               r_cell,
    const int                               d,
    const dense::Vec< T, 6 >                local_coefficients[2] )
{
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell, d ), local_coefficients[0]( 0 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell, d ),
        local_coefficients[0]( 1 ) + local_coefficients[1]( 2 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell, d ),
        local_coefficients[0]( 2 ) + local_coefficients[1]( 1 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell + 1, d ), local_coefficients[0]( 3 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, d ),
        local_coefficients[0]( 4 ) + local_coefficients[1]( 5 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, d ),
        local_coefficients[0]( 5 ) + local_coefficients[1]( 4 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, d ), local_coefficients[1]( 0 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, d ), local_coefficients[1]( 3 ) );
}

/// @brief Returns the lateral wedge index with respect to a coarse grid wedge from the fine wedge indices.
///
/// Each coarse grid wedge is laterally divided into four fine wedges (radially into two).
/// The lateral four fine wedges are enumerated from 0 to 3.
///
/// This function returns that lateral index given a fine grid index of a wedge.
///
/// Here are two coarse wedges (view from the "top"), each with four fine grid wedges (enumerated from 0 to 3).
///
/// @code
///
/// Coarse wedge idx = 0
/*!
    +
    |\
    | \
    |  \
    | 2 \
    +----+
    |\ 3 |\
    | \  | \
    |  \ |  \
    | 0 \| 1 \
    +----+----+
   
    Coarse wedge idx = 1
   
    +----+----+
     \ 1 |\ 0 |
      \  | \  |
       \ |  \ |
        \| 3 \|
         +----+
          \ 2 |
           \  |
            \ |
             \|
              +
   
    @endcode
   
    This function now computes the indices from a grid of fine wedges that looks like this:
   
    @code
   
    x_cell_fine % 1:
   
    +----+----+
    |\ 0 |\ 1 |
    | \  | \  |
    |  \ |  \ |
    | 0 \| 1 \|
    +----+----+
    |\ 0 |\ 1 |
    | \  | \  |
    |  \ |  \ |
    | 0 \| 1 \|
    +----+----+
   
    y_cell_fine % 1:
   
    +----+----+
    |\ 1 |\ 1 |
    | \  | \  |
    |  \ |  \ |
    | 1 \| 1 \|
    +----+----+
    |\ 0 |\ 0 |
    | \  | \  |
    |  \ |  \ |
    | 0 \| 0 \|
    +----+----+
   
    wedge_idx_fine:
   
    +----+----+
    |\ 1 |\ 1 |
    | \  | \  |
    |  \ |  \ |
    | 0 \| 0 \|
    +----+----+
    |\ 1 |\ 1 |
    | \  | \  |
    |  \ |  \ |
    | 0 \| 0 \|
    +----+----+
   
    Resulting/returned values:
   
    +----+----+
    |\ 1 |\ 0 |
    | \  | \  |
    |  \ |  \ |
    | 2 \| 3 \|
    +----+----+
    |\ 3 |\ 2 |
    | \  | \  |
    |  \ |  \ |
    | 0 \| 1 \|
    +----+----+
   
    @endcode
*/
KOKKOS_INLINE_FUNCTION
constexpr int fine_lateral_wedge_idx( const int x_cell_fine, const int y_cell_fine, const int wedge_idx_fine )
{
    // wedge, y, x
    constexpr int indices[2][2][2] = { { { 0, 1 }, { 2, 3 } }, { { 3, 2 }, { 1, 0 } } };
    const int     x_mod            = x_cell_fine % 2;
    const int     y_mod            = y_cell_fine % 2;
    return indices[wedge_idx_fine][y_mod][x_mod];
}

enum class DoFOrdering
{
    NODEWISE,
    DIMENSIONWISE,
};


template < typename ScalarT > KOKKOS_INLINE_FUNCTION
constexpr void
    reorder_local_dofs( const DoFOrdering doo_from, const DoFOrdering doo_to, dense::Vec< ScalarT, 18 >& dofs )
{
    if ( doo_from == doo_to )
    {
        return;
    }

    dense::Vec< ScalarT, 18 > tmp_dofs;
    if ( (doo_from == DoFOrdering::NODEWISE) && (doo_to == DoFOrdering::DIMENSIONWISE) )
    {
        for ( int dim = 0; dim < 3; ++dim )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                tmp_dofs(node_idx + dim * num_nodes_per_wedge) = dofs(node_idx * 3 + dim);
            }
        }
    }
    else if ( (doo_from == DoFOrdering::DIMENSIONWISE) && (doo_to == DoFOrdering::NODEWISE) )
    {
        for ( int dim = 0; dim < 3; ++dim )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                tmp_dofs(node_idx * 3 + dim) = dofs(node_idx + dim * num_nodes_per_wedge);
            }
        }
    }

    for (int i = 0; i < 18; ++i) {
        dofs(i) = tmp_dofs(i);
    }
}

} // namespace terra::fe::wedge
