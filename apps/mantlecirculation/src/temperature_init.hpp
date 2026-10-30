#pragma once

#include <string>
#include <vector>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "interpolators.hpp"
#include "io.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_w1.hpp"
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

/// Set T (W1) from the configured initial-condition profile,
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
    linalg::VectorW1Scalar< ScalarType >&                           T,
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

        // Compute conductive temperature profile
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

    // Broadcast reference temperature profile to W1 nodes
    Kokkos::parallel_for(
        "RadialProfileToW1",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        RadialProfileToW1{ T.grid_data(), T_ref } );
    Kokkos::fence();

    // Add initial perturbation
    if ( init_temp.perturbation == InitialPerturbation::NOISE )
    {
        util::logroot << "Adding random noise..." << std::endl;

        // Add noise
        Kokkos::parallel_for(
            "noise to W1 temperature",
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

    else if ( init_temp.perturbation == InitialPerturbation::SPHERICAL_HARMONICS )
    {
        const bool sph = ( init_temp.sph_degree_l > 0 && init_temp.perturbation_amplitude != 0.0 );
        const bool sph_2 =
            ( init_temp.sph_degree_l_2 > 0 && init_temp.sph_factor_2 != 0.0 &&
              init_temp.perturbation_amplitude != 0.0 );

        if ( sph )
        {
            util::logroot << "Adding spherical harmonic perturbation..." << std::endl;

            grid::Grid3DDataScalar< ScalarType > sph_coeffs;

            sph_coeffs = shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >(
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

            // Add spherical harmonic perturbation
            Kokkos::parallel_for(
                "SphericalHarmonicPerturbationAdder",
                grid::shell::local_domain_md_range_policy_nodes( domain ),
                SphericalHarmonicPerturbationAdder{
                    init_temp.perturbation_amplitude,
                    prm.boundary_parameters.temperature_min,
                    prm.boundary_parameters.temperature_max,
                    prm.mesh_parameters.radius_min,
                    prm.mesh_parameters.radius_max,
                    true, /*taper_near_boundaries*/
                    coords_radii,
                    sph_coeffs,
                    T.grid_data(),
                    ownership_mask } );
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
    grid::Grid2DDataScalar< ScalarType >&       temperature_profile,
    grid::Grid2DDataScalar< ScalarType >&       rho_profile,
    grid::Grid2DDataScalar< ScalarType >&       alpha_profile,
    grid::Grid2DDataScalar< ScalarType >&       cp_profile,
    const grid::shell::DistributedDomain&       domain,
    const grid::Grid2DDataScalar< ScalarType >& coords_radii,
    const Parameters&                           prm )
{
    const auto& phys      = prm.physics_parameters;
    const auto& init_temp = prm.physics_parameters.initial_temperature;

    // Reference temperature profile
    if ( init_temp.profile == InitialTemperatureProfile::FROM_FILE )
    {
        util::logroot << "Reading reference temperature from: '" << init_temp.Tref_profile_csv_path << "'" << std::endl;

        // Read and populate reference temperature profile
        temperature_profile = shell::interpolate_radial_profile_into_subdomains_from_csv(
            init_temp.Tref_profile_csv_path,
            phys.radial_profiles_radii_key,
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
                temperature_profile } );
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
                temperature_profile } );
    }

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
    linalg::VectorW1Scalar< ScalarType >&       dst,
    const linalg::VectorW1Scalar< ScalarType >& src,
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
    linalg::VectorW1Scalar< ScalarType >&       T_cond,
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
    linalg::VectorW1Vec< ScalarType, 3 >&       u_velocity,
    linalg::VectorW1Scalar< ScalarType >&       T,
    int&                                        return_timestep,
    ScalarType&                                 return_model_time,
    const grid::shell::DistributedDomain&       domain,
    const grid::Grid3DDataVec< ScalarType, 3 >& coords_shell,
    const grid::Grid2DDataScalar< ScalarType >& coords_radii,
    const Parameters&                           prm )
{
    using util::logroot;

    // Append xdmf-subdirectory to checkpoint_dir name
    std::string checkpoint_dir = prm.io_parameters.checkpoint_dir + "/xdmf";

    // This assumes consistency regarding param "output_per_Ma" between the
    // checkpoint and the current simulation. Otherwise, the wording of the log
    // message will be inaccurate, no further consequences.
    if ( prm.io_parameters.output_per_Ma )
    {
        logroot << "Loading checkpoint from " << checkpoint_dir << " at model age " << prm.io_parameters.checkpoint_step
                << " Ma." << std::endl;
    }
    else
    {
        logroot << "Loading checkpoint from " << checkpoint_dir << " at simulation step "
                << prm.io_parameters.checkpoint_step << "." << std::endl;
    }

    // Checking if checkpoint is dimensional or nondimensional
    auto metadata_result = io::read_xdmf_checkpoint_metadata( checkpoint_dir );
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
        checkpoint_dir, std::string( "u_u" ), prm.io_parameters.checkpoint_step, domain, u_velocity.grid_data() );
    if ( success_vel.is_err() )
    {
        Kokkos::abort( success_vel.error().c_str() );
    }

    auto success_temp = io::read_xdmf_checkpoint_grid(
        checkpoint_dir, std::string( "T" ), prm.io_parameters.checkpoint_step, domain, T.grid_data() );
    if ( success_temp.is_err() )
    {
        Kokkos::abort( success_temp.error().c_str() );
    }

    // Read checkpoint model time and timestep. We read both, because
    // prm.io_parameters.checkpoint_step might be either model_time or timestep.
    if ( prm.io_parameters.continue_simulation )
        io::read_xdmf_step_and_time(
            checkpoint_dir, prm.io_parameters.checkpoint_step, return_timestep, return_model_time );

    // Nondimensionalise checkpoint, if necessary
    if ( metadata.is_dimensional )
    {
        scale( T.grid_data(), ScalarType( 1 ) / prm.boundary_parameters.delta_T_K );
        scale( u_velocity.grid_data(), ScalarType( 1 ) / prm.physics_parameters.calc_cm_per_year );
    }
}

} // namespace terra::mantlecirculation
