#pragma once

#include "parameters.hpp"

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "util/bit_masking.hpp"

namespace terra::mantlecirculation {

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;

struct InitialConditionInterpolator
{
    ScalarType                                         r_min_;
    ScalarType                                         r_max_;
    Grid3DDataVec< ScalarType, 3 >                     grid_;
    Grid2DDataScalar< ScalarType >                     radii_;
    Grid4DDataScalar< ScalarType >                     data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_data_;
    bool                                               only_boundary_;

    InitialConditionInterpolator(
        const ScalarType                                          r_min,
        const ScalarType                                          r_max,
        const Grid3DDataVec< ScalarType, 3 >&                     grid,
        const Grid2DDataScalar< ScalarType >&                     radii,
        const Grid4DDataScalar< ScalarType >&                     data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask_data,
        bool                                                      only_boundary )
    : r_min_( r_min )
    , r_max_( r_max )
    , grid_( grid )
    , radii_( radii )
    , data_( data )
    , mask_data_( mask_data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const auto mask_value  = mask_data_( local_subdomain_id, x, y, r );
        const auto is_boundary = util::has_flag( mask_value, grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || is_boundary )
        {
            const dense::Vec< ScalarType, 3 > coords =
                grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
            const auto frac                      = ( r_max_ - coords.norm() ) / ( r_max_ - r_min_ );
            data_( local_subdomain_id, x, y, r ) = Kokkos::pow( frac, 5 );
        }
    }
};

/// Initial condition for Q1 temperature (conductive profile + spherical harmonic perturbation):
/// T = T_ref(r) + eps * Y_l^m(theta, phi)
/// where T_ref is the steady-state spherical conduction solution:
///   T_ref(r) = r_min * r_max / r  -  r_min
struct ConductiveProfileInterpolator
{
    ScalarType                          r_min_, r_max_, eps_;
    Grid3DDataVec< ScalarType, 3 >      grid_;
    Grid2DDataScalar< ScalarType >      radii_;
    Grid4DDataScalar< ScalarType >      data_;
    Grid3DDataScalar< ScalarType >      sph_coeffs_;
    bool                                has_sph_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const ScalarType radius = coords.norm();

        // Guard against zero radius (non-owned ghost nodes may have zero coordinates).
        if ( radius < ScalarType( 1e-15 ) )
        {
            data_( sd, x, y, r ) = ScalarType( 0 );
            return;
        }

        const ScalarType T_ref = r_min_ * r_max_ / radius - r_min_;

        ScalarType T_val = T_ref;
        if ( has_sph_ )
        {
            T_val += eps_ * sph_coeffs_( sd, x, y );
        }

        data_( sd, x, y, r ) = T_val;
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_u_;
    Grid4DDataScalar< ScalarType > data_T_;
    ScalarType                     rayleigh_number_;

    RHSVelocityInterpolator(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataVec< ScalarType, 3 >& data_u,
        const Grid4DDataScalar< ScalarType >& data_T,
        ScalarType                            rayleigh_number )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , data_T_( data_T )
    , rayleigh_number_( rayleigh_number )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const auto n = coords.normalized();

        for ( int d = 0; d < 3; d++ )
        {
            data_u_( local_subdomain_id, x, y, r, d ) =
                rayleigh_number_ * n( d ) * data_T_( local_subdomain_id, x, y, r );
        }
    }
};

struct NoiseAdder
{
    Grid3DDataVec< ScalarType, 3 >              grid_;
    Grid2DDataScalar< ScalarType >              radii_;
    Grid4DDataScalar< ScalarType >              data_T_;
    Grid4DDataScalar< grid::NodeOwnershipFlag > mask_;
    Kokkos::Random_XorShift64_Pool<>            rand_pool_;

    NoiseAdder(
        const Grid3DDataVec< ScalarType, 3 >&              grid,
        const Grid2DDataScalar< ScalarType >&              radii,
        const Grid4DDataScalar< ScalarType >&              data_T,
        const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask )
    : grid_( grid )
    , radii_( radii )
    , data_T_( data_T )
    , mask_( mask )
    , rand_pool_( 12345 )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        auto generator = rand_pool_.get_state();

        const ScalarType eps          = 1e-1;
        const auto       perturbation = eps * ( 2.0 * generator.drand() - 1.0 );

        const auto process_ownes_point =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::NodeOwnershipFlag::OWNED );

        if ( process_ownes_point )
        {
            data_T_( local_subdomain_id, x, y, r ) =
                Kokkos::clamp( data_T_( local_subdomain_id, x, y, r ) + perturbation, 0.0, 1.0 );
        }
        else
        {
            data_T_( local_subdomain_id, x, y, r ) = 0.0;
        }

        rand_pool_.free_state( generator );
    }
};

/// Initial condition for FV cell-centred temperature: same radial profile as the Q1 version,
/// evaluated at the precomputed cell centres.
struct FVInitialConditionInterpolator
{
    ScalarType                     r_min_, r_max_;
    Grid4DDataVec< ScalarType, 3 > cell_centers_;
    Grid4DDataScalar< ScalarType > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const ScalarType cx     = cell_centers_( id, x, y, r, 0 );
        const ScalarType cy     = cell_centers_( id, x, y, r, 1 );
        const ScalarType cz     = cell_centers_( id, x, y, r, 2 );
        const ScalarType radius = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
        const ScalarType frac   = ( r_max_ - radius ) / ( r_max_ - r_min_ );
        data_( id, x, y, r )    = Kokkos::pow( frac, ScalarType( 5 ) );
    }
};

/// Noise adder for FV cells.  All non-ghost cells are owned by the local subdomain,
/// so no ownership mask is needed.
struct FVNoiseAdder
{
    Grid4DDataScalar< ScalarType >   data_T_;
    Kokkos::Random_XorShift64_Pool<> rand_pool_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        auto             gen          = rand_pool_.get_state();
        const ScalarType eps          = 1e-1;
        const ScalarType perturbation = eps * ( 2.0 * gen.drand() - 1.0 );
        data_T_( id, x, y, r )        = Kokkos::clamp( data_T_( id, x, y, r ) + perturbation, 0.0, 1.0 );
        rand_pool_.free_state( gen );
    }
};

/// Computes viscosity from temperature according to the selected viscosity law.
struct ViscosityFromTemperature
{
    ViscosityLaw                   law_;
    ScalarType                     rmu_;
    Grid4DDataScalar< ScalarType > eta_;
    Grid4DDataScalar< ScalarType > T_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const ScalarType T_val = T_( id, x, y, r );

        switch ( law_ )
        {
        case ViscosityLaw::FRANK_KAMENETSKII:
            // Zhong et al. (2008) form: mu = rmu^(0.5 - T).
            // Total viscosity contrast (cold/hot) = rmu.
            eta_( id, x, y, r ) = Kokkos::pow( rmu_, ScalarType( 0.5 ) - T_val );
            break;
        case ViscosityLaw::CONSTANT:
        default:
            // eta is already set, nothing to do.
            break;
        }
    }
};

} // namespace terra::mantlecirculation
