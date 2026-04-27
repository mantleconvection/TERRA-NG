
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

using terra::fe::wedge::num_nodes_per_wedge;
using terra::fe::wedge::num_wedges_per_hex_cell;

namespace terra::linalg::solvers {

KOKKOS_INLINE_FUNCTION
int map_to_coarse_element( const int fine_cell, const int level_range )
{
    int coarse_cell = fine_cell;
    for ( int l = 0; l < level_range; ++l )
    {
        coarse_cell = Kokkos::floor( coarse_cell / 2 );
    }
    return coarse_cell;
}

template < typename ScalarT >
class GCAElementsCollector
{
  public:
    using ScalarType = ScalarT;

  private:
    grid::shell::DistributedDomain fine_domain_;

    // fine grid coefficient
    grid::Grid4DDataScalar< ScalarType > k_;

    // coarsest grid boolean field for elements on which a GCA hierarchy has to be built
    grid::Grid4DDataScalar< ScalarType > GCAElements_;

    const int level_range_;

  public:
    GCAElementsCollector(
        const grid::shell::DistributedDomain&       fine_domain,
        const grid::Grid4DDataScalar< ScalarType >& k,
        const int                                   level_range,
        grid::Grid4DDataScalar< ScalarType >        GCAElements )
    : fine_domain_( fine_domain )
    , k_( k )
    , GCAElements_( GCAElements )
    , level_range_( level_range )
    {
        Kokkos::parallel_for(
            "evaluate coefficient gradient", grid::shell::local_domain_md_range_policy_cells( fine_domain_ ), *this );
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        dense::Vec< ScalarT, 6 > k[num_wedges_per_hex_cell];
        terra::fe::wedge::extract_local_wedge_scalar_coefficients( k, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

        constexpr auto num_quad_points = fe::wedge::quadrature::quad_felippa_1x1_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];

        fe::wedge::quadrature::quad_felippa_1x1_quad_points( quad_points );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            const auto qp = quad_points[0];

            dense::Vec< ScalarType, 3 > k_grad_eval = { 0 };
            for ( int j = 0; j < num_nodes_per_wedge; j++ )
            {
                k_grad_eval = k_grad_eval + terra::fe::wedge::grad_shape( j, qp ) * k[wedge]( j );
            }
            auto k_grad_norm = k_grad_eval.norm();
            if ( k_grad_norm > 10 )
            {
                // Todo: map to parent coarsest element
                int x_cell_coarsest = map_to_coarse_element( x_cell, level_range_ );
                int y_cell_coarsest = map_to_coarse_element( y_cell, level_range_ );
                int r_cell_coarsest = map_to_coarse_element( r_cell, level_range_ );

                GCAElements_( local_subdomain_id, x_cell_coarsest, y_cell_coarsest, r_cell_coarsest ) = 1;
            }
        }
    }
};
} // namespace terra::linalg::solvers