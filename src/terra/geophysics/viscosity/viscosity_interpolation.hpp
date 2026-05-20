#pragma once

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"

namespace terra::geophysics::viscosity {

/// @brief Helper class for interpolation of purely radially dependent viscosity profiles.
///
/// Requires as input the already radially interpolated profile on a 2D grid (layout: (local_subdomain_id, r)).
/// Such a grid can be initialized using functions like
/// - \ref terra::shell::interpolate_radial_profile_into_subdomains
/// - \ref terra::shell::interpolate_radial_profile_into_subdomains_from_csv
/// - \ref terra::shell::interpolate_constant_radial_profile
///
/// If requested, the viscosity can be scaled by 1.0 / reference_viscosity and also clamped to some upper and lower
/// bounds.
template < typename ScalarType >
class RadialProfileViscosityInterpolator
{
  public:
    /// @brief Creates an interpolator class.
    ///
    /// The order is: first clamping, then scaling by the inverse of the reference.
    ///
    /// @param radial_viscosity_profile see class description
    /// @param reference_viscosity the resulting profile is scaled by 1.0 / reference_viscosity
    /// @param viscosity_lower_bound lower bound for clamping
    /// @param viscosity_upper_bound upper bound for clamping
    explicit RadialProfileViscosityInterpolator(
        const grid::Grid2DDataScalar< ScalarType >& radial_viscosity_profile,
        const ScalarType&                           reference_viscosity   = 1.0,
        const ScalarType&                           viscosity_lower_bound = std::numeric_limits< ScalarType >::lowest(),
        const ScalarType&                           viscosity_upper_bound = std::numeric_limits< ScalarType >::max() )
    : radial_viscosity_profile_( radial_viscosity_profile )
    , reference_viscosity_( reference_viscosity )
    , one_over_reference_viscosity_( 1.0 / reference_viscosity_ )
    , viscosity_lower_bound_( viscosity_lower_bound )
    , viscosity_upper_bound_( viscosity_upper_bound )
    {}

    /// @brief Runs a kernel to interpolate the viscosity profile into a full grid.
    void interpolate( const grid::Grid4DDataScalar< ScalarType >& dst_grid )
    {
        data_ = dst_grid;
        Kokkos::parallel_for(
            "viscosity_interpolation",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 0, 0, 0 },
                { dst_grid.extent( 0 ), dst_grid.extent( 1 ), dst_grid.extent( 2 ), dst_grid.extent( 3 ) } ),
            *this );
        Kokkos::fence();
    }

    /// @brief Call `interpolate()` to run this kernel.
    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        auto visc = radial_viscosity_profile_( local_subdomain_id, r );

        if ( visc < viscosity_lower_bound_ )
        {
            visc = viscosity_lower_bound_;
        }

        if ( visc > viscosity_upper_bound_ )
        {
            visc = viscosity_upper_bound_;
        }

        data_( local_subdomain_id, x, y, r ) = one_over_reference_viscosity_ * visc;
    }

  private:
    grid::Grid2DDataScalar< ScalarType > radial_viscosity_profile_;
    ScalarType                           reference_viscosity_;
    ScalarType                           one_over_reference_viscosity_;
    ScalarType                           viscosity_lower_bound_;
    ScalarType                           viscosity_upper_bound_;

    grid::Grid4DDataScalar< ScalarType > data_;
};

} // namespace terra::geophysics::viscosity