#pragma once

#include "linalg/vector_q1.hpp"
#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/bit_masking.hpp"

namespace terra::linalg::trafo {

/// \brief Constructs a robust orthonormal transformation matrix
///        from Cartesian to (normal–tangential–tangential) coordinates.
///
/// Given a surface normal \p n, this function returns the 3×3 matrix
/// \f$ R \f$ whose rows form a right-handed orthonormal basis
/// \f$ \{ \hat{n}, \mathbf{t}_1, \mathbf{t}_2 \} \f$ such that:
/// \f[
///   \mathbf{v}_{\text{local}} = R \, \mathbf{v}_{\text{cartesian}},
///   \qquad
///   \mathbf{v}_{\text{cartesian}} = R^{\mathsf{T}} \, \mathbf{v}_{\text{local}} .
/// \f]
///
/// The basis vectors are constructed as:
/// - \f$ \hat{n} = \frac{\mathbf{n}}{||\mathbf{n}||} \f$
/// - \f$ \mathbf{t}_1 = \mathrm{normalize}(\mathbf{r} \times \hat{n}) \f$,
///   where \f$ \mathbf{r} \f$ is the coordinate axis least aligned with \f$ \hat{n} \f$
/// - \f$ \mathbf{t}_2 = \hat{n} \times \mathbf{t}_1 \f$
///
/// This ensures numerical stability even when \f$ \hat{n} \f$ is nearly axis-aligned.
///
/// \tparam ScalarType Floating-point scalar type (e.g. float, double).
/// \param n_input Surface normal vector at the point of interest (not required to be unit length).
/// \return 3×3 transformation matrix \f$ R \f$ with rows \f$ [\,\hat{n}^\mathrm{T},\, \mathbf{t}_1^\mathrm{T},\, \mathbf{t}_2^\mathrm{T}\,] \f$.
template < std::floating_point ScalarType >
KOKKOS_INLINE_FUNCTION dense::Mat< ScalarType, 3, 3 >
                       trafo_mat_cartesian_to_normal_tangential( const dense::Vec< ScalarType, 3 >& n_input )
{
    using Vec3 = dense::Vec< ScalarType, 3 >;
    using Mat3 = dense::Mat< ScalarType, 3, 3 >;

    // 1. normalize normal
    const Vec3       n  = n_input.normalized();
    const ScalarType nx = Kokkos::fabs( n( 0 ) );
    const ScalarType ny = Kokkos::fabs( n( 1 ) );
    const ScalarType nz = Kokkos::fabs( n( 2 ) );

    // 2. choose reference axis least aligned with n
    Vec3 ref;
    if ( nx <= ny && nx <= nz )
    {
        ref = { ScalarType( 1 ), ScalarType( 0 ), ScalarType( 0 ) };
    }
    else if ( ny <= nx && ny <= nz )
    {
        ref = { ScalarType( 0 ), ScalarType( 1 ), ScalarType( 0 ) };
    }
    else
    {
        ref = { ScalarType( 0 ), ScalarType( 0 ), ScalarType( 1 ) };
    }

    // 3. construct tangents
    Vec3             t1  = ref.cross( n ).normalized();
    const ScalarType eps = std::is_same_v< ScalarType, double > ? 1e-15 : 1e-6;
    if ( t1.norm() < eps )
    {
        t1 = Vec3{ n( 1 ), -n( 0 ), ScalarType( 0 ) }.normalized();
    }

    const Vec3 t2 = n.cross( t1 ).normalized();

    // 4. assemble transformation matrix (rows = n, t1, t2)
    Mat3 R = Mat3::from_row_vecs( n, t1, t2 );

    return R;
}

/// \brief Constructs the inverse transformation matrix from (normal–tangential–tangential) to Cartesian coordinates.
///
/// This function returns the transpose of the orthonormal transformation matrix
/// \f$ R \f$ produced by \ref trafo_mat_cartesian_to_normal_tangential, since
/// for an orthonormal basis \f$ R^{-1} = R^{\mathsf{T}} \f$.
///
/// Hence, for any vector \f$ \mathbf{v} \f$:
/// \f[
///   \mathbf{v}_{\text{local}} = R \, \mathbf{v}_{\text{cartesian}}, \qquad
///   \mathbf{v}_{\text{cartesian}} = R^{\mathsf{T}} \, \mathbf{v}_{\text{local}} .
/// \f]
///
/// \tparam ScalarType Floating-point scalar type (e.g. float, double).
/// \param n_input Surface normal vector at the point of interest (not required to be unit length).
/// \return The 3×3 inverse transformation matrix \f$ R^{\mathsf{T}} \f$
///         that maps local (n, t₁, t₂) coordinates back to Cartesian space.
template < std::floating_point ScalarType >
KOKKOS_INLINE_FUNCTION dense::Mat< ScalarType, 3, 3 >
                       trafo_mat_normal_tangential_to_cartesian_trafo( const dense::Vec< ScalarType, 3 >& n_input )
{
    return trafo_mat_cartesian_to_normal_tangential< ScalarType >( n_input ).transposed();
}

template < std::floating_point ScalarType, std::floating_point ScalarTypeGrid, util::FlagLike FlagType >
void cartesian_to_normal_tangential_in_place(
    VectorQ1Vec< ScalarType, 3 >&                   vec_cartesian,
    const grid::Grid3DDataVec< ScalarTypeGrid, 3 >& coords_shell,
    const grid::Grid4DDataScalar< FlagType >        mask_data,
    const FlagType&                                 flag )
{
    auto data = vec_cartesian.grid_data();
    auto mask = vec_cartesian.mask_data();

    Kokkos::parallel_for(
        "cartesian_to_normal_tangential_in_place",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
            { 0, 0, 0, 0 }, { data.extent( 0 ), data.extent( 1 ), data.extent( 2 ), data.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            if ( !util::has_flag( mask_data( local_subdomain, i, j, k ), flag ) )
            {
                return;
            }

            dense::Vec< ScalarType, 3 > vec_local_cart{};
            for ( int d = 0; d < 3; ++d )
            {
                vec_local_cart( d ) = data( local_subdomain, i, j, k, d );
            }

            dense::Vec< ScalarType, 3 > normal;
            //int sign = flag == grid::shell::ShellBoundaryFlag::CMB ? -1 : 1;
            for ( int d = 0; d < 3; ++d )
            {
                normal( d ) = coords_shell( local_subdomain, i, j, d );
            }

            const auto R                           = trafo_mat_cartesian_to_normal_tangential( normal );
            const auto vec_local_normal_tangential = R * vec_local_cart;

            for ( int d = 0; d < 3; ++d )
            {
                data( local_subdomain, i, j, k, d ) = vec_local_normal_tangential( d );
            }
        } );
    Kokkos::fence();
}

template < std::floating_point ScalarType, std::floating_point ScalarTypeGrid, util::FlagLike FlagType >
void normal_tangential_to_cartesian_in_place(
    VectorQ1Vec< ScalarType, 3 >&                   vec_normal_tangential,
    const grid::Grid3DDataVec< ScalarTypeGrid, 3 >& coords_shell,
    const grid::Grid4DDataScalar< FlagType >        mask_data,
    const FlagType&                                 flag )
{
    auto data = vec_normal_tangential.grid_data();
    auto mask = vec_normal_tangential.mask_data();

    Kokkos::parallel_for(
        "cartesian_to_normal_tangential_in_place",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
            { 0, 0, 0, 0 }, { data.extent( 0 ), data.extent( 1 ), data.extent( 2 ), data.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            if ( !util::has_flag( mask_data( local_subdomain, i, j, k ), flag ) )
            {
                return;
            }

            dense::Vec< ScalarType, 3 > vec_local_normal_tangential{};
            for ( int d = 0; d < 3; ++d )
            {
                vec_local_normal_tangential( d ) = data( local_subdomain, i, j, k, d );
            }

            dense::Vec< ScalarType, 3 > normal;
            for ( int d = 0; d < 3; ++d )
            {
                normal( d ) = coords_shell( local_subdomain, i, j, d );
            }

            const auto Rt             = trafo_mat_normal_tangential_to_cartesian_trafo( normal );
            const auto vec_local_cart = Rt * vec_local_normal_tangential;

            for ( int d = 0; d < 3; ++d )
            {
                data( local_subdomain, i, j, k, d ) = vec_local_cart( d );
            }
        } );
    Kokkos::fence();
}

} // namespace terra::linalg::trafo