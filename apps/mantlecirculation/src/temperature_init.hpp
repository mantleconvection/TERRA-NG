#pragma once

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "interpolators.hpp"
#include "io.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_q1.hpp"
#include "parameters.hpp"
#include "shell/spherical_harmonics.hpp"
#include "util/logging.hpp"

namespace terra::mantlecirculation {

struct ComputeConductiveProfile
{
    ScalarType                     r_min_, r_max_, T_min_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid2DDataScalar< ScalarType > radial_profile_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int r ) const
    {
        // Guard against zero radius (non-owned ghost nodes may have zero coordinates).
        const ScalarType radius = radii_( id, r );
        if ( radius < ScalarType( 1e-15 ) )
        {
            radial_profile_( id, r ) = ScalarType( 0 );
            return;
        }
        radial_profile_( id, r ) = ( r_min_ * r_max_ / radius - r_min_ ) / ( r_max_ - r_min_ ) + T_min_;
    }
};

struct ComputePowerLawProfile
{
    ScalarType                     r_min_, r_max_, T_min_, T_max_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid2DDataScalar< ScalarType > radial_profile_;
    ScalarType                     exponent_ = 5.0;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int r ) const
    {
        const ScalarType radius = radii_( id, r );
        const ScalarType frac   = ( r_max_ - radius ) / ( r_max_ - r_min_ );

        radial_profile_( id, r ) = T_min_ + ( T_max_ - T_min_ ) * Kokkos::pow( frac, exponent_ );
    }
};

/// Random superposition sum_{l = 1}^{l_max} sum_{|m| <= min(l, m_max)} a_lm * Y_l^m, with a_lm uniform in [-1, 1].
/// The seed is fixed, so all ranks draw the same weights.
template < typename ScalarType >
grid::Grid3DDataScalar< ScalarType > random_sph_superposition(
    const int                                   l_max,
    const int                                   m_max,
    const grid::Grid3DDataVec< ScalarType, 3 >& coords_shell )
{
    grid::Grid3DDataScalar< ScalarType > sum(
        "sph_superposition", coords_shell.extent( 0 ), coords_shell.extent( 1 ), coords_shell.extent( 2 ) );

    std::mt19937                                 rng( 42 );
    std::uniform_real_distribution< ScalarType > weight( -1, 1 );

    for ( int l = 1; l <= l_max; ++l )
    {
        for ( int m = -std::min( l, m_max ); m <= std::min( l, m_max ); ++m )
        {
            const auto Y_lm =
                shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >( l, m, coords_shell );
            const ScalarType a_lm = weight( rng );

            Kokkos::parallel_for(
                "add Y_lm",
                Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
                    { 0, 0, 0 }, { sum.extent( 0 ), sum.extent( 1 ), sum.extent( 2 ) } ),
                KOKKOS_LAMBDA( int id, int x, int y ) { sum( id, x, y ) += a_lm * Y_lm( id, x, y ); } );
        }
    }
    Kokkos::fence();

    return sum;
}

/// Set T (Q1) from the configured initial-condition profile,
/// and add an optional noise or spherical-harmonic perturbation of maximum size
/// parameters.physics_parameters.initial_temperature.perturbation_amplitude.
///
/// Three profiles are supported:
///   * CONDUCTIVE: Spherical steady-state conduction solution.
///   * POWER_LAW: Radial exponential T_min + (T_max - T_min) * radius_frac^5,
///     with radius_frac = (r_max - radius) / (r_max - r_min).
///   * FROM_FILE: Reading custom temperature profile from csv file.
template < typename ScalarType >
void initialize_temperature_fields(
    linalg::VectorQ1Scalar< ScalarType >&                           T,
    grid::Grid2DDataScalar< ScalarType >&                           T_ref,
    const grid::shell::DistributedDomain&                           domain,
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const Parameters&                                               prm )
{
    using util::logroot;
    const auto& init_temp = prm.physics_parameters.initial_temperature;

    if ( init_temp.profile == InitialTemperatureProfile::FROM_FILE )
    {
        util::logroot << "Reading reference temperature from: '" << init_temp.Tref_profile_csv_path << "'" << std::endl;

        // Read and populate reference temperature profile
        T_ref = shell::interpolate_radial_profile_into_subdomains_from_csv(
            init_temp.Tref_profile_csv_path,
            prm.physics_parameters.radial_profiles_radii_key,
            init_temp.Tref_profile_value_key,
            coords_radii,
            ScalarType( 1 ) / prm.mesh_parameters.mantle_thickness_m,
            ScalarType( 1 ) / prm.boundary_parameters.delta_T_K );
    }

    else if ( init_temp.profile == InitialTemperatureProfile::CONDUCTIVE )
    {
        util::logroot << "Computing conductive reference temperature profile" << std::endl;

        Kokkos::parallel_for(
            "ComputeConductiveProfile",
            grid::shell::local_domain_md_range_policy_radial( domain ),
            ComputeConductiveProfile{
                prm.mesh_parameters.radius_min,
                prm.mesh_parameters.radius_max,
                prm.boundary_parameters.temperature_min,
                coords_radii,
                T_ref } );
    }

    else if ( init_temp.profile == InitialTemperatureProfile::POWER_LAW )
    {
        util::logroot << "Computing power-law reference temperature profile" << std::endl;

        Kokkos::parallel_for(
            "ComputePowerLawProfile",
            grid::shell::local_domain_md_range_policy_radial( domain ),
            ComputePowerLawProfile{
                prm.mesh_parameters.radius_min,
                prm.mesh_parameters.radius_max,
                prm.boundary_parameters.temperature_min,
                prm.boundary_parameters.temperature_max,
                coords_radii,
                T_ref } );
    }

    // Broadcast reference profile to Q1 nodes
    Kokkos::parallel_for(
        "RadialProfileToQ1",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        RadialProfileToQ1{ T.grid_data(), T_ref } );
    Kokkos::fence();

    // Add initial perturbation
    if ( init_temp.perturbation == InitialPerturbation::NOISE )
    {
        util::logroot << "Adding random noise..." << std::endl;

        // Add noise
        Kokkos::parallel_for(
            "noise to Q1 temperature",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            NoiseAdder{
                init_temp.perturbation_amplitude,
                prm.boundary_parameters.temperature_min,
                prm.boundary_parameters.temperature_max,
                prm.mesh_parameters.radius_min,
                prm.mesh_parameters.radius_max,
                true, /* taper_near_boundaries */
                coords_shell,
                coords_radii,
                T.grid_data(),
                ownership_mask } );
        Kokkos::fence();

        // Communicate to ghost nodes
        communication::shell::send_recv( domain, T.grid_data(), communication::CommunicationReduction::MAX );
    }

    else if (
        init_temp.perturbation == InitialPerturbation::SPHERICAL_HARMONICS ||
        init_temp.perturbation == InitialPerturbation::SPHERICAL_HARMONICS_SUPERPOSITION )
    {
        const bool superpose = ( init_temp.perturbation == InitialPerturbation::SPHERICAL_HARMONICS_SUPERPOSITION );
        const bool sph       = ( init_temp.sph_degree_l > 0 && init_temp.perturbation_amplitude != 0.0 );
        const bool sph_2 =
            ( !superpose && init_temp.sph_degree_l_2 > 0 && init_temp.sph_factor_2 != 0.0 &&
              init_temp.perturbation_amplitude != 0.0 );

        if ( sph )
        {
            util::logroot << "Adding spherical harmonic " << ( superpose ? "superposition " : "" ) << "perturbation..."
                          << std::endl;

            grid::Grid3DDataScalar< ScalarType > sph_coeffs =
                superpose ? random_sph_superposition( init_temp.sph_degree_l, init_temp.sph_order_m, coords_shell ) :
                            shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >(
                                init_temp.sph_degree_l, init_temp.sph_order_m, coords_shell );

            if ( sph_2 )
            {
                grid::Grid3DDataScalar< ScalarType > sph_coeffs_2;

                sph_coeffs_2 = shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >(
                    init_temp.sph_degree_l_2, init_temp.sph_order_m_2, coords_shell );
                const ScalarType factor_2 = static_cast< ScalarType >( init_temp.sph_factor_2 );

                // Combine spherical harmonics
                Kokkos::parallel_for(
                    "combine spherical harmonics",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
                        { 0, 0, 0 },
                        { static_cast< int >( sph_coeffs.extent( 0 ) ),
                          static_cast< int >( sph_coeffs.extent( 1 ) ),
                          static_cast< int >( sph_coeffs.extent( 2 ) ) } ),
                    KOKKOS_LAMBDA( int sd, int x, int y ) {
                        sph_coeffs( sd, x, y ) += factor_2 * sph_coeffs_2( sd, x, y );
                    } );
                Kokkos::fence();
            }

            // Normalize sph-coefficients to [-1, 1], so the user-chosen perturbation amplitude remains meaningful
            ScalarType max_abs_sph;

            Kokkos::parallel_reduce(
                "sph_coeffs_max_abs",
                Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
                    { 0, 0, 0 }, { sph_coeffs.extent( 0 ), sph_coeffs.extent( 1 ), sph_coeffs.extent( 2 ) } ),
                KOKKOS_LAMBDA( int id, int x, int y, ScalarType& max_tmp ) {
                    max_tmp = Kokkos::max( max_tmp, Kokkos::abs( sph_coeffs( id, x, y ) ) );
                },
                Kokkos::Max< ScalarType >( max_abs_sph ) );

            // Normalize in-place
            Kokkos::parallel_for(
                "normalize sph_coeffs",
                Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
                    { 0, 0, 0 }, { sph_coeffs.extent( 0 ), sph_coeffs.extent( 1 ), sph_coeffs.extent( 2 ) } ),
                KOKKOS_LAMBDA( int id, int x, int y ) { sph_coeffs( id, x, y ) /= max_abs_sph; } );
            Kokkos::fence();

            SphericalHarmonicPerturbationAdder adder{
                init_temp.perturbation_amplitude,
                prm.boundary_parameters.temperature_min,
                prm.boundary_parameters.temperature_max,
                prm.mesh_parameters.radius_min,
                prm.mesh_parameters.radius_max,
                true, /*taper_near_boundaries*/
                coords_radii,
                sph_coeffs,
                T.grid_data(),
                ownership_mask };

            // Optionally restrict the perturbation to a single radial layer
            if ( init_temp.sph_layer >= 0 )
            {
                const auto& radii      = domain.domain_info().radii();
                const int   num_layers = static_cast< int >( radii.size() ) - 1;

                if ( init_temp.sph_layer >= num_layers )
                {
                    Kokkos::abort( ( "--initial-temperature-sph-layer = " + std::to_string( init_temp.sph_layer ) +
                                     " is out of range: the mesh has " + std::to_string( num_layers ) +
                                     " radial layers (valid: 0 ... " + std::to_string( num_layers - 1 ) + ")." )
                                       .c_str() );
                }

                const int    i      = init_temp.sph_layer;
                const double r_lo   = radii[i];
                const double r_hi   = radii[i + 1];
                double       min_dr = r_hi - r_lo;
                for ( int k = 0; k < num_layers; ++k )
                {
                    min_dr = std::min( min_dr, radii[k + 1] - radii[k] );
                }

                adder.restrict_to_layer_ = true;
                adder.layer_r_lo_        = static_cast< ScalarType >( r_lo );
                adder.layer_r_hi_        = static_cast< ScalarType >( r_hi );
                adder.layer_tol_         = static_cast< ScalarType >( 0.25 * min_dr );

                util::logroot << "Restricting spherical harmonic perturbation to radial layer " << i << " of "
                              << num_layers << " (r in [" << r_lo << ", " << r_hi << "], nondimensional)" << std::endl;
            }

            // Add spherical harmonic perturbation
            Kokkos::parallel_for(
                "SphericalHarmonicPerturbationAdder",
                grid::shell::local_domain_md_range_policy_nodes( domain ),
                adder );
            Kokkos::fence();

            // Communicate to ghost nodes
            communication::shell::send_recv( domain, T.grid_data(), communication::CommunicationReduction::MAX );
        }
    }
}

// Read custom radial profiles (if given), nondimensionalise and fill respective 2D array.
// Fill radially constant otherwise.
template < typename ScalarType >
void radial_profile_init(
    grid::Grid2DDataScalar< ScalarType >&       rho_profile,
    grid::Grid2DDataScalar< ScalarType >&       alpha_profile,
    grid::Grid2DDataScalar< ScalarType >&       cp_profile,
    const grid::shell::DistributedDomain&       domain,
    const grid::Grid2DDataScalar< ScalarType >& coords_radii,
    const Parameters&                           prm )
{
    const auto& phys = prm.physics_parameters;

    // Density
    if ( !phys.density_profile_csv_path.empty() )
    {
        rho_profile = shell::interpolate_radial_profile_into_subdomains_from_csv(
            phys.density_profile_csv_path,
            phys.radial_profiles_radii_key,
            phys.density_profile_value_key,
            coords_radii,
            ScalarType( 1 ) / prm.mesh_parameters.mantle_thickness_m,
            ScalarType( 1 ) / phys.reference_density );
    }
    else if ( phys.compressible )
    {
        const ScalarType surface_density = phys.surface_density_nondim;
        const ScalarType dissipation_nr  = phys.dissipation_number;
        const ScalarType radius_max      = prm.mesh_parameters.radius_max;
        const ScalarType grueneisen_prm  = phys.grueneisen_parameter;

        // Adiabatic compression
        Kokkos::parallel_for(
            "adiabatic compression",
            grid::shell::local_domain_md_range_policy_radial( domain ),
            KOKKOS_LAMBDA( int id, int r ) {
                rho_profile( id, r ) =
                    surface_density *
                    Kokkos::exp( dissipation_nr * ( radius_max - coords_radii( id, r ) ) / grueneisen_prm );
            } );
        Kokkos::fence();
    }
    else // Fill with ones if incompressible
    {
        Kokkos::deep_copy( rho_profile, ScalarType( 1 ) );
    }

    // alpha
    if ( !phys.alpha_profile_csv_path.empty() )
    {
        alpha_profile = shell::interpolate_radial_profile_into_subdomains_from_csv(
            phys.alpha_profile_csv_path,
            phys.radial_profiles_radii_key,
            phys.alpha_profile_value_key,
            coords_radii,
            ScalarType( 1 ) / prm.mesh_parameters.mantle_thickness_m,
            ScalarType( 1 ) / phys.thermal_expansivity );
    }
    else // Fill with ones
    {
        Kokkos::deep_copy( alpha_profile, ScalarType( 1 ) );
    }

    // Cp
    if ( !phys.cp_profile_csv_path.empty() )
    {
        cp_profile = shell::interpolate_radial_profile_into_subdomains_from_csv(
            phys.cp_profile_csv_path,
            phys.radial_profiles_radii_key,
            phys.cp_profile_value_key,
            coords_radii,
            ScalarType( 1 ) / prm.mesh_parameters.mantle_thickness_m,
            ScalarType( 1 ) / phys.specific_heat_capacity );
    }
    else // Fill with ones
    {
        Kokkos::deep_copy( cp_profile, ScalarType( 1 ) );
    }
}

template < typename ScalarType >
void subtract_radial_profile(
    linalg::VectorQ1Scalar< ScalarType >&       dst,
    const linalg::VectorQ1Scalar< ScalarType >& src,
    const grid::Grid2DDataScalar< ScalarType >& profile,
    const grid::shell::DistributedDomain&       domain )
{
    Kokkos::parallel_for(
        "subtract_radial_profile",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        SubtractRadialProfile{ profile, src.grid_data(), dst.grid_data() } );
    Kokkos::fence();
}

/// Spherical steady-state conduction profile  T_cond(r) = r_min·r_max/r − r_min.
/// Used as the reference temperature for the Nusselt-number diagnostic and
/// added to XDMF output for visualisation.
template < typename ScalarType >
void compute_reference_conductive_profile(
    linalg::VectorQ1Scalar< ScalarType >&       T_cond,
    const grid::shell::DistributedDomain&       domain,
    const grid::Grid3DDataVec< ScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< ScalarType >& coords_radii,
    const Parameters&                           prm )
{
    Kokkos::parallel_for(
        "conductive profile T_cond",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        ConductiveProfileInterpolator{
            domain.domain_info().radii().front(),
            domain.domain_info().radii().back(),
            ScalarType( 0 ),
            prm.boundary_parameters.temperature_min,
            coords_shell,
            coords_radii,
            T_cond.grid_data(),
            {},
            false } );
    Kokkos::fence();
    // NOTE: do NOT call send_recv here.  Same rationale as in the IC kernel:
    // every subdomain copy of a shared node already gets the correct value, so
    // a SUM exchange would multiply it by the sharing count.
}

/// Load (u, T) from an XDMF checkpoint.
template < typename ScalarType >
void load_temperature_checkpoint(
    linalg::VectorQ1Vec< ScalarType, 3 >&       u_velocity,
    linalg::VectorQ1Scalar< ScalarType >&       T,
    const grid::shell::DistributedDomain&       domain,
    const grid::Grid3DDataVec< ScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< ScalarType >& coords_radii,
    const Parameters&                           prm )
{
    using util::logroot;

    logroot << "Loading checkpoint from " << prm.io_parameters.checkpoint_dir << " at simulation step "
            << prm.io_parameters.checkpoint_step << std::endl;

    // Checking if checkpoint is dimensional or nondimensional
    auto metadata_result = io::read_xdmf_checkpoint_metadata( prm.io_parameters.checkpoint_dir );
    if ( metadata_result.is_err() )
    {
        Kokkos::abort( metadata_result.error().c_str() );
    }
    const auto& metadata = metadata_result.unwrap();

    if ( metadata.is_dimensional == -1 )
    {
        // Checkpoint version 0 or 1 - no flag present.
        logroot << "\nWARNING: Checkpoint predates is_dimensional flag (version " << metadata.version
                << "). Assuming nondimensional.\n"
                << std::endl;
    }

    else if ( static_cast< bool >( metadata.is_dimensional ) != prm.devel_parameters.output_dimensional )
    {
        logroot
            << "\nWARNING: Read and write checkpoint details are inconsistent - one is dimensional, one is nondimensional.\n"
            << std::endl;
    }

    if ( static_cast< bool >( metadata.is_dimensional ) && prm.devel_parameters.nondimensional_input )
    {
        logroot
            << "\n Nondimensional input selected, but the checkpoint you are trying to read is dimensional. This will produce garbage. Exiting..."
            << std::endl;
        Kokkos::abort( "Error: Nondimensional input inconsistent with dimensional checkpoint." );
    }

    auto success_vel = io::read_xdmf_checkpoint_grid(
        prm.io_parameters.checkpoint_dir,
        std::string( "u_u" ),
        prm.io_parameters.checkpoint_step,
        domain,
        u_velocity.grid_data() );
    if ( success_vel.is_err() )
    {
        Kokkos::abort( success_vel.error().c_str() );
    }

    auto success_temp = io::read_xdmf_checkpoint_grid(
        prm.io_parameters.checkpoint_dir,
        std::string( "T" ),
        prm.io_parameters.checkpoint_step,
        domain,
        T.grid_data() );
    if ( success_temp.is_err() )
    {
        Kokkos::abort( success_temp.error().c_str() );
    }

    // Nondimensionalise checkpoint, if necessary
    if ( metadata.is_dimensional )
    {
        scale( T.grid_data(), ScalarType( 1 ) / prm.boundary_parameters.delta_T_K );
        scale( u_velocity.grid_data(), ScalarType( 1 ) / prm.physics_parameters.calc_cm_per_year );
    }
}

} // namespace terra::mantlecirculation
