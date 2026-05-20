
#pragma once
#include "fe/triangle/quadrature/quadrature.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature/quadrature.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fv::hex::operators {

template < typename ScalarT >
class Mass
{
  public:
    using SrcVectorType = linalg::VectorFVScalar< ScalarT >;
    using DstVectorType = linalg::VectorFVScalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    Mass(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , boundary_mask_( boundary_mask )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        // TODO: communicate ghost layers

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 1, 1, 1 },
                { src.grid_data().extent( 0 ),
                  src.grid_data().extent( 1 ) - 1,
                  src.grid_data().extent( 2 ) - 1,
                  src.grid_data().extent( 3 ) - 1 } ),
            *this );
    }

  private:
    using Vec3 = dense::Vec< ScalarType, 3 >;

  public:
    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Gather surface points for each wedge.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                               [fe::wedge::num_nodes_per_wedge_surface] = {};
        fe::wedge::wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell - 1, y_cell - 1 );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell - 1 );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell );

        // Quadrature points.
        constexpr auto num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
        fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        // Cell volume
        ScalarType M_ii = 0;
        for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
        {
            for ( int q = 0; q < num_quad_points; ++q )
            {
                const auto J       = fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det     = J.det();
                const auto abs_det = Kokkos::abs( det );
                const auto Jw      = abs_det * quad_weights[q];
                M_ii += Jw;
            }
        }

        // Mass is diagonal.
        dst_( local_subdomain_id, x_cell, y_cell, r_cell ) = M_ii * src_( local_subdomain_id, x_cell, y_cell, r_cell );
    }
};

} // namespace terra::fv::hex::operators