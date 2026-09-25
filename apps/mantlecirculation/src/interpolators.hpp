#pragma once

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "parameters.hpp"
#include "util/bit_masking.hpp"

namespace terra::mantlecirculation {

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;

// Interpolate from radial profile to Q1 field
struct RadialProfileToQ1
{
    Grid4DDataScalar< ScalarType > data_;
    Grid2DDataScalar< ScalarType > radial_profile_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        data_( id, x, y, r ) = radial_profile_( id, r );
    }
};

// Subtracts laterally constant profile data from Grid4DDataScalar.
// Computes src_(id, x, y, r) - profile_(id, r) = dst_(id, x, y, r).
struct SubtractRadialProfile
{
    Grid2DDataScalar< ScalarType > profile_;
    Grid4DDataScalar< ScalarType > src_;
    Grid4DDataScalar< ScalarType > dst_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        dst_( id, x, y, r ) = src_( id, x, y, r ) - profile_( id, r );
    }
};

/// Initial condition for Q1 temperature (conductive profile + spherical harmonic perturbation):
/// T = T_cond(r) + eps * Y_l^m(theta, phi)
/// where T_cond is the steady-state spherical conduction solution:
///   T_cond(r) = r_min * r_max / r  -  r_min
struct ConductiveProfileInterpolator
{

    ScalarType                     r_min_, r_max_, eps_;
    ScalarType                     T_min_;
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > data_;
    Grid3DDataScalar< ScalarType > sph_coeffs_;
    bool                           has_sph_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( sd, x, y, r, grid_, radii_ );
        const ScalarType                  radius = coords.norm();

        // Guard against zero radius (non-owned ghost nodes may have zero coordinates).
        if ( radius < ScalarType( 1e-15 ) )
        {
            data_( sd, x, y, r ) = ScalarType( 0 );
            return;
        }

        const ScalarType T_cond =
            adiabatic_ ? ( T_ad_s_ * Kokkos::exp( Di_ * ( r_max_ - radius ) ) ) :
                         ( ( r_min_ * r_max_ / radius - r_min_ ) / ( r_max_ - r_min_ ) + T_min_ );

        ScalarType T_val = T_cond;
        if ( has_sph_ )
        {
            T_val += eps_ * sph_coeffs_( sd, x, y );
        }

        data_( sd, x, y, r ) = T_val;
    }

    // Canonical TALA (match HyTeG): start on the adiabat T_bar(r) = T_ad,s * exp(Di * (r_max - r))
    // rather than the conductive profile, so the background is consistent with the
    // buoyancy reference (Tdev(t=0) = perturbation). Off by default, which reproduces
    // upstream's conductive background exactly.
    bool       adiabatic_ = false;
    ScalarType Di_        = ScalarType( 0 );
    ScalarType T_ad_s_    = ScalarType( 0 );
};

template < typename RhoFieldType >
struct BuoyancyForceAssembly
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_f_;
    Grid4DDataScalar< ScalarType > data_T_;
    RhoFieldType                   data_rho_;
    Grid2DDataScalar< ScalarType > alpha_;
    ScalarType                     rayleigh_number_;
    ScalarType                     prefactor_;

    BuoyancyForceAssembly(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataVec< ScalarType, 3 >& data_f,
        const Grid4DDataScalar< ScalarType >& data_T,
        const RhoFieldType&                   data_rho,
        const Grid2DDataScalar< ScalarType >& alpha,
        const ScalarType                      rayleigh_number,
        const ScalarType                      prefactor = ScalarType( 1 ) )
    : grid_( grid )
    , radii_( radii )
    , data_f_( data_f )
    , data_T_( data_T )
    , data_rho_( data_rho )
    , alpha_( alpha )
    , rayleigh_number_( rayleigh_number )
    , prefactor_( prefactor )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( id, x, y, r, grid_, radii_ );
        const auto                        n      = coords.normalized();

        // Check if data_rho_ is radial profile or 3-D field.
        // TALA compressibility requires radial profile, PDA requires full 3-D density.
        ScalarType rho_val;
        if constexpr ( std::is_same_v< RhoFieldType, Grid2DDataScalar< ScalarType > > )
            rho_val = data_rho_( id, r );
        else
            rho_val = data_rho_( id, x, y, r );

        for ( int d = 0; d < 3; d++ )
        {
            data_f_( id, x, y, r, d ) =
                prefactor_ * rayleigh_number_ * n( d ) * alpha_( id, r ) * rho_val * data_T_( id, x, y, r );
        }
    }
};

struct NoiseAdder
{
    ScalarType                                  eps_;
    ScalarType                                  T_min_;
    ScalarType                                  T_max_;
    ScalarType                                  r_min_;
    ScalarType                                  r_max_;
    bool                                        taper_near_boundaries_;
    Grid3DDataVec< ScalarType, 3 >              grid_;
    Grid2DDataScalar< ScalarType >              radii_;
    Grid4DDataScalar< ScalarType >              data_;
    Grid4DDataScalar< grid::NodeOwnershipFlag > mask_;
    Kokkos::Random_XorShift64_Pool<>            rand_pool_;

    NoiseAdder(
        const ScalarType                                   eps,
        const ScalarType                                   T_min,
        const ScalarType                                   T_max,
        const ScalarType                                   r_min,
        const ScalarType                                   r_max,
        const bool                                         taper_near_boundaries,
        const Grid3DDataVec< ScalarType, 3 >&              grid,
        const Grid2DDataScalar< ScalarType >&              radii,
        const Grid4DDataScalar< ScalarType >&              data,
        const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask )
    : eps_( eps )
    , T_min_( T_min )
    , T_max_( T_max )
    , r_min_( r_min )
    , r_max_( r_max )
    , taper_near_boundaries_( taper_near_boundaries )
    , grid_( grid )
    , radii_( radii )
    , data_( data )
    , mask_( mask )
    , rand_pool_( 12345 )
    {}

    static constexpr ScalarType taper_width_ = ScalarType( 0.05 );

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        auto generator = rand_pool_.get_state();

        const ScalarType perturbation = eps_ * ( 2.0 * generator.drand() - 1.0 );
        ScalarType       taper        = ScalarType( 1 );

        // apply tapering near top and bottom boundary
        if ( taper_near_boundaries_ )
        {
            const ScalarType radius = radii_( id, r );

            const ScalarType dist_to_boundary = Kokkos::min( radius - r_min_, r_max_ - radius );
            const ScalarType t = Kokkos::clamp( dist_to_boundary / taper_width_, ScalarType( 0 ), ScalarType( 1 ) );

            taper = t * t * ( ScalarType( 3 ) - ScalarType( 2 ) * t );
        }

        // Only write to owned nodes
        const auto process_owns_point = util::has_flag( mask_( id, x, y, r ), grid::NodeOwnershipFlag::OWNED );

        if ( process_owns_point )
        {
            data_( id, x, y, r ) = Kokkos::clamp( data_( id, x, y, r ) + taper * perturbation, T_min_, T_max_ );
        }
        else
        {
            data_( id, x, y, r ) = ScalarType( 0 );
        }

        rand_pool_.free_state( generator );
    }
};

struct SphericalHarmonicPerturbationAdder
{
    ScalarType                                  eps_;
    ScalarType                                  T_min_, T_max_;
    ScalarType                                  r_min_, r_max_;
    bool                                        taper_near_boundaries_;
    Grid2DDataScalar< ScalarType >              radii_; // Only used if taper_near_boundaries == 'true'
    Grid3DDataScalar< ScalarType >              sph_coeffs_;
    Grid4DDataScalar< ScalarType >              data_;
    Grid4DDataScalar< grid::NodeOwnershipFlag > mask_;

    static constexpr ScalarType taper_width_ = ScalarType( 0.05 );

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        ScalarType taper = ScalarType( 1 );

        // apply tapering near top and bottom boundary
        if ( taper_near_boundaries_ )
        {
            const ScalarType radius           = radii_( id, r );
            const ScalarType dist_to_boundary = Kokkos::min( radius - r_min_, r_max_ - radius );
            const ScalarType t = Kokkos::clamp( dist_to_boundary / taper_width_, ScalarType( 0 ), ScalarType( 1 ) );

            taper = t * t * ( ScalarType( 3 ) - ScalarType( 2 ) * t );
        }

        // Only write to owned nodes
        const auto process_owns_point = util::has_flag( mask_( id, x, y, r ), grid::NodeOwnershipFlag::OWNED );

        if ( process_owns_point )
        {
            data_( id, x, y, r ) =
                Kokkos::clamp( data_( id, x, y, r ) + taper * eps_ * sph_coeffs_( id, x, y ), T_min_, T_max_ );
        }
        else
        {
            data_( id, x, y, r ) = ScalarType( 0 );
        }
    }
};

/// Initial condition for FV cell-centred temperature: same radial profile as the Q1 version,
/// evaluated at the precomputed cell centres.
struct FVInitialConditionInterpolator
{
    ScalarType                     r_min_, r_max_;
    ScalarType                     T_min_, T_max_;
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
        data_( id, x, y, r )    = T_min_ + ( T_max_ - T_min_ ) * Kokkos::pow( frac, ScalarType( 5 ) );
    }
};

/// Noise adder for FV cells.  All non-ghost cells are owned by the local subdomain,
/// so no ownership mask is needed.
struct FVNoiseAdder
{
    ScalarType                       T_min_, T_max_;
    Grid4DDataScalar< ScalarType >   data_T_;
    Kokkos::Random_XorShift64_Pool<> rand_pool_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        auto             gen          = rand_pool_.get_state();
        const ScalarType eps          = 1e-1;
        const ScalarType perturbation = eps * ( 2.0 * gen.drand() - 1.0 );
        data_T_( id, x, y, r )        = Kokkos::clamp( data_T_( id, x, y, r ) + perturbation, T_min_, T_max_ );
        rand_pool_.free_state( gen );
    }
};

/// Computes viscosity from temperature according to the selected viscosity law.
struct ViscosityFromTemperature
{
    ViscosityLaw                         law_;
    Grid4DDataScalar< ScalarType >       eta_;
    const Grid4DDataScalar< ScalarType > T_;
    const Grid2DDataScalar< ScalarType > eta_profile_;
    const Grid2DDataScalar< ScalarType > coords_radii_;
    const ScalarType                     activation_energy_;
    const ScalarType                     activation_volume_;
    const ScalarType                     radius_max_;
    const ScalarType                     eta_min_;
    const ScalarType                     eta_max_;
    const ScalarType                     c1_ = 0.25;
    const ScalarType                     c2_ = 1.25;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const ScalarType T_val = T_( id, x, y, r );
        const ScalarType depth = radius_max_ - coords_radii_( id, r );

        ScalarType eta_val = 0.0;

        switch ( law_ )
        {
        case ViscosityLaw::FK_BENCHMARK:
            // Zhong et al. (2008) form: mu = rmu^(0.5 - T).
            // Total viscosity contrast (cold/hot) = rmu.
            // activation_energy here used in place of rmu.
            eta_val = Kokkos::pow( activation_energy_, ScalarType( 0.5 ) - T_val );
            break;
        case ViscosityLaw::FK_TYPE1: // uses T as input
        case ViscosityLaw::FK_TYPE3: // dT instead of T
            eta_val =
                eta_profile_( id, r ) * Kokkos::exp( -1 * activation_energy_ * T_val + activation_volume_ * depth );
            break;
        case ViscosityLaw::FK_TYPE2:
            eta_val =
                eta_profile_( id, r ) *
                Kokkos::exp( -1 * activation_energy_ * ( T_val - ScalarType( 0.5 ) ) + activation_volume_ * depth );
            break;
        case ViscosityLaw::ARRHENIUS:
            eta_val = eta_profile_( id, r ) * Kokkos::exp(
                                                  activation_energy_ * ( ScalarType( 1 ) / ( T_val + c1_ ) - c2_ ) +
                                                  activation_volume_ * depth );
            break;
        case ViscosityLaw::CONSTANT:
        default:
            // eta is already set, nothing to do.
            return;
        }
        eta_( id, x, y, r ) = Kokkos::clamp( eta_val, eta_min_, eta_max_ );
    }
};


struct InitialConditionInterpolator
{
    ScalarType                                         r_min_;
    ScalarType                                         r_max_;
    ScalarType                                         T_min_;
    ScalarType                                         T_max_;
    Grid3DDataVec< ScalarType, 3 >                     grid_;
    Grid2DDataScalar< ScalarType >                     radii_;
    Grid4DDataScalar< ScalarType >                     data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_data_;
    bool                                               only_boundary_;

    InitialConditionInterpolator(
        const ScalarType                                          r_min,
        const ScalarType                                          r_max,
        const ScalarType                                          T_min,
        const ScalarType                                          T_max,
        const Grid3DDataVec< ScalarType, 3 >&                     grid,
        const Grid2DDataScalar< ScalarType >&                     radii,
        const Grid4DDataScalar< ScalarType >&                     data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask_data,
        bool                                                      only_boundary )
    : r_min_( r_min )
    , r_max_( r_max )
    , T_min_( T_min )
    , T_max_( T_max )
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
            data_( local_subdomain_id, x, y, r ) = T_min_ + ( T_max_ - T_min_ ) * Kokkos::pow( frac, 5 );
        }
    }
};

struct DensityInit
{
    Grid4DDataScalar< ScalarType > rho_;
    Grid2DDataScalar< ScalarType > radii_;
    ScalarType                     r_max_;
    ScalarType                     surface_density_;
    ScalarType                     dissipation_number_;
    ScalarType                     grueneisen_parameter_;
    bool                           compressible_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        if ( compressible_ )
        {
            // Adiabatic compression
            const ScalarType radius = radii_( id, r );

            rho_( id, x, y, r ) =
                surface_density_ * Kokkos::exp( dissipation_number_ * ( r_max_ - radius ) / grueneisen_parameter_ );
        }
        else
        {
            rho_( id, x, y, r ) = 1.0;
        }
    }
};

/// @brief Nodal adiabatic (compression) heating source for the TALA energy
/// equation.
///
/// Fills a Q1 scalar field with
///
///     S_adiab = prefactor · Di · (u·n) · T
///
/// where n = coords.normalized() is the outward radial (anti-gravity) unit
/// vector, u·n the radial velocity, T the temperature and Di the dissipation
/// number. The nodal field is meant to be L²-projected onto the RHS via the
/// mass matrix (like the constant internal-heating source) and interpolated
/// into the entropy-viscosity residual.
///
/// The nondimensional coefficient is Di alone. Notably there is NO reference
/// density ρ̄ and NO thermal expansivity α: this matches the buoyancy force
/// used by the Stokes solve, Ra·δT·n, which is likewise ρ̄- and α-free (α is
/// folded into Di and Ra). Weighting the adiabatic term by ρ̄ while the
/// buoyancy is unweighted would break the dissipation balance ⟨Φ⟩=⟨W⟩; keeping
/// both ρ̄-free is the self-consistent choice for this branch's formulation.
///
/// The physically-correct prefactor is −1: rising material (u·n > 0) does work
/// against gravity and cools, so it must contribute negatively to DT/Dt. The
/// prefactor is left configurable to calibrate against this branch's exact
/// nondimensionalisation / temperature-offset convention.
struct AdiabaticHeatingSource
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > u_;
    Grid4DDataScalar< ScalarType > T_;
    Grid4DDataScalar< ScalarType > dst_;
    ScalarType                     dissipation_number_;
    ScalarType                     prefactor_ = ScalarType( -1 );
    // Radial alpha/cp (both normalised by their reference values, so the ratio is 1 for an
    // incompressible run). Di already carries alpha_0/cp_0; this supplies the radial shape.
    Grid2DDataScalar< ScalarType > alpha_;
    Grid2DDataScalar< ScalarType > cp_;
    bool                           use_profiles_ = false;
    /// true  -> shape = alpha/cp, the coefficient of the adiabatic SOURCE term.
    /// false -> shape = alpha, for the dissipation-conservation diagnostic, whose identity
    ///          is  Di * int rho*alpha*u_r*T  =  (Di/Ra) * int Phi  (the equation was
    ///          multiplied back by rho*cp, so the 1/cp must not appear there).
    bool                           divide_by_cp_ = true;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( id, x, y, r, grid_, radii_ );
        const auto                        n      = coords.normalized();

        ScalarType u_r = ScalarType( 0 );
        for ( int d = 0; d < 3; ++d )
            u_r += u_( id, x, y, r, d ) * n( d );

        const ScalarType shape =
            use_profiles_ ? ( divide_by_cp_ ? ( alpha_( id, r ) / cp_( id, r ) ) : alpha_( id, r ) ) : ScalarType( 1 );

        dst_( id, x, y, r ) = prefactor_ * dissipation_number_ * shape * u_r * T_( id, x, y, r );
    }
};

/// Multiply a Q1 nodal field in place by a radial profile (or its reciprocal).
/// Used to apply 1/(rho*cp) to the shear-heating source and 1/cp to internal heating.
struct ScaleByRadialProfile
{
    Grid4DDataScalar< ScalarType > data_;
    Grid2DDataScalar< ScalarType > p1_;
    Grid2DDataScalar< ScalarType > p2_;
    bool                           use_p2_ = false;
    bool                           invert_ = true;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const ScalarType p = use_p2_ ? ( p1_( id, r ) * p2_( id, r ) ) : p1_( id, r );
        if ( p > ScalarType( 0 ) )
            data_( id, x, y, r ) = invert_ ? ( data_( id, x, y, r ) / p ) : ( data_( id, x, y, r ) * p );
    }
};

} // namespace terra::mantlecirculation
