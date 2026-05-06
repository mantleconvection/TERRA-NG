#pragma once

#include <vector>

#include "grid/shell/spherical_shell.hpp"
#include "parameters.hpp"

namespace terra::mantlecirculation {

/// @brief Build the radial-shell vector for one MG level according to the
///        MeshParameters' selected RadialDistribution.
///
/// `n_shells` is the number of radial nodes (= radial cells + 1) at this level.
/// Selects between equispaced layers and the three tanh-clustered variants
/// (both-side, CMB-only, surface-only).  The cluster strength `k` is taken
/// from `mesh.radial_cluster_k`; `k <= 0` collapses each variant to uniform.
template < typename ScalarT >
std::vector< ScalarT > build_shell_radii( const MeshParameters& mesh, int n_shells )
{
    const ScalarT r_min = static_cast< ScalarT >( mesh.radius_min );
    const ScalarT r_max = static_cast< ScalarT >( mesh.radius_max );
    const ScalarT k     = static_cast< ScalarT >( mesh.radial_cluster_k );

    using namespace terra::grid::shell;
    switch ( mesh.radial_distribution )
    {
    case MeshParameters::RadialDistribution::UNIFORM:
        return uniform_shell_radii< ScalarT >( r_min, r_max, n_shells );
    case MeshParameters::RadialDistribution::TANH_BOTH:
        return mapped_shell_radii< ScalarT >(
            r_min, r_max, n_shells, make_tanh_boundary_cluster< ScalarT >( k ) );
    case MeshParameters::RadialDistribution::TANH_CMB:
        return mapped_shell_radii< ScalarT >(
            r_min, r_max, n_shells, make_tanh_inner_cluster< ScalarT >( k ) );
    case MeshParameters::RadialDistribution::TANH_SURFACE:
        return mapped_shell_radii< ScalarT >(
            r_min, r_max, n_shells, make_tanh_outer_cluster< ScalarT >( k ) );
    }
    return uniform_shell_radii< ScalarT >( r_min, r_max, n_shells );
}

} // namespace terra::mantlecirculation
