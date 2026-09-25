#pragma once

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "communication/shell/fv_communication.hpp"
#include "fv/hex/conversion.hpp"
#include "fv/hex/helpers.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "interpolators.hpp"
#include "io.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_fv.hpp"
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

/// Adiabatic reference profile for TALA runs:
///   T_ad(r) = T_ad,s * exp( Di * (r_max - r) )
/// Used both as the initial condition and as the radial reference the buoyancy
/// deviation Tdev = T - T_ref(r) is taken against, so that the reference is the
/// same adiabat the buoyancy term is linearised about (canonical TALA, cf. King
/// et al. 2010). Selected with --initial-temperature-profile adiabatic.
struct ComputeAdiabaticProfile
{
    ScalarType                     r_max_, T_ad_s_, Di_;
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
        radial_profile_( id, r ) = T_ad_s_ * Kokkos::exp( Di_ * ( r_max_ - radius ) );
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

/// Set T (Q1) and T_fct (FV) from the configured initial-condition profile,
/// apply Dirichlet BCs on T_fct, exchange ghost layers, and L2-project to keep
/// both fields consistent.
///
/// Two profiles are supported:
///   * CONDUCTIVE: spherical steady-state conduction solution + optional
///     spherical-harmonic perturbation (Y_l^m + factor·Y_l2^m2).  Q1 first,
///     then projected to FV.
///   * power-law + noise: FV interpolation followed by a per-cell noise add.
///     FV first, then projected back to Q1.
///
/// In either case the post-condition is: T_fct holds the FV cell-averaged
/// initial state with Dirichlet BCs applied and ghost layers populated; T is
/// the L2-projected Q1 representation of T_fct.
template < typename ScalarType >
void initialize_temperature_fields(
    linalg::VectorQ1Scalar< ScalarType >&                           T,
    linalg::VectorFVScalar< ScalarType >&                           T_fct,
    grid::Grid2DDataScalar< ScalarType >&                           T_ref,
    const fv::hex::DirichletBCs< ScalarType >&                      fct_bcs,
    const grid::shell::DistributedDomain&                           domain,
    const grid::Grid3DDataVec< ScalarType, 3 >&                     coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                     coords_radii,
    const linalg::VectorFVVec< ScalarType, 3 >&                     fv_cell_centers,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&        ownership_mask,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
    const Parameters&                                               prm )
{
    using util::logroot;
    const auto& init_temp = prm.physics_parameters.initial_temperature;

    if ( prm.energy_solver_parameters.energy_solver != EnergySolverType::FCT )
    {
        if ( init_temp.profile == InitialTemperatureProfile::FROM_FILE )
        {
            util::logroot << "Reading reference temperature from: '" << init_temp.Tref_profile_csv_path << "'"
                          << std::endl;

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

        else if ( init_temp.profile == InitialTemperatureProfile::ADIABATIC )
        {
            util::logroot << "Computing adiabatic reference temperature profile (T_ad,s "
                          << init_temp.adiabat_surface_temperature << ", Di "
                          << prm.physics_parameters.dissipation_number << ")" << std::endl;

            Kokkos::parallel_for(
                "ComputeAdiabaticProfile",
                grid::shell::local_domain_md_range_policy_radial( domain ),
                ComputeAdiabaticProfile{
                    static_cast< ScalarType >( prm.mesh_parameters.radius_max ),
                    static_cast< ScalarType >( init_temp.adiabat_surface_temperature ),
                    static_cast< ScalarType >( prm.physics_parameters.dissipation_number ),
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

        else if ( init_temp.perturbation == InitialPerturbation::SPHERICAL_HARMONICS )
        {
            const bool sph = ( init_temp.sph_degree_l > 0 && init_temp.perturbation_amplitude != 0.0 );
            const bool sph_2 =
                ( init_temp.sph_degree_l_2 > 0 && init_temp.sph_factor_2 != 0.0 &&
                  init_temp.perturbation_amplitude != 0.0 );

            // Broadband initial perturbation: a random-coefficient sum of Y_l^m over
            // l in [sph_degree_min, sph_degree_max]; takes precedence over the single/
            // double-mode path below. plmbar is evaluated once per node for the whole
            // Legendre triangle, which is what makes large l_max tractable.
            // Shared by the broadband and the single/double-mode paths below.
            grid::Grid3DDataScalar< ScalarType > sph_coeffs;

            const bool broadband_sph =
                ( init_temp.sph_degree_max >= 1 && init_temp.sph_degree_max >= init_temp.sph_degree_min &&
                  init_temp.perturbation_amplitude != 0.0 );
        if ( broadband_sph )
        {
            const int l_min = std::max( 1, init_temp.sph_degree_min );
            const int l_max = init_temp.sph_degree_max;
            logroot << " + eps * broadband SH sum, l in [" << l_min << "," << l_max << "], seed "
                    << init_temp.sph_random_seed << std::endl;
            // Draw coefficients once on the host (reproducible), amplitude-normalise by
            // 1/sqrt(#modes). FUSED evaluation: for each shell node we call plmbar ONCE
            // (the full Legendre triangle up to l_max) and accumulate every mode, instead
            // of building one device grid + fence per mode. Cost is O(N * l_max^2) on the
            // host with a single host->device copy, versus the previous
            // O(N * l_max^4) with a device round-trip and global fence per mode -- the
            // latter made large l_max (e.g. 96 -> 9345 modes) effectively hang for hours.
            // Equivalent field: same modes, same coefficient draw order; plmbar's P_l^m
            // recurrence is independent of the max degree requested, so per-mode values
            // match (accumulation is done in double).
            std::mt19937                             rng( init_temp.sph_random_seed );
            std::uniform_real_distribution< double > dist( -1.0, 1.0 );
            int                                      nmodes = 0;
            for ( int l = l_min; l <= l_max; ++l )
                nmodes += ( 2 * l + 1 );
            const double norm = ( nmodes > 0 ) ? 1.0 / std::sqrt( static_cast< double >( nmodes ) ) : 1.0;

            // Coefficient for every (l,m), drawn in the original (l: l_min..l_max, m: -l..l) order.
            std::vector< double > coeff;
            coeff.reserve( static_cast< size_t >( nmodes > 0 ? nmodes : 0 ) );
            for ( int l = l_min; l <= l_max; ++l )
                for ( int m = -l; m <= l; ++m )
                    coeff.push_back( dist( rng ) * norm );

            sph_coeffs = grid::Grid3DDataScalar< ScalarType >(
                "sph_coeffs_broadband",
                static_cast< int >( coords_shell.extent( 0 ) ),
                static_cast< int >( coords_shell.extent( 1 ) ),
                static_cast< int >( coords_shell.extent( 2 ) ) );

            auto sph_coeffs_host   = Kokkos::create_mirror_view( Kokkos::HostSpace{}, sph_coeffs );
            auto coords_shell_host = Kokkos::create_mirror_view( Kokkos::HostSpace{}, coords_shell );
            Kokkos::deep_copy( coords_shell_host, coords_shell );

            shell::SphericalHarmonicsTool sph_tool( static_cast< unsigned int >( l_max ) );
            const int             plm_size = ( l_max + 1 ) * ( l_max + 2 ) / 2;
            std::vector< double > plm( static_cast< size_t >( plm_size ) );

            const int n_sd = static_cast< int >( sph_coeffs.extent( 0 ) );
            const int n_i  = static_cast< int >( sph_coeffs.extent( 1 ) );
            const int n_j  = static_cast< int >( sph_coeffs.extent( 2 ) );
            for ( int sd = 0; sd < n_sd; ++sd )
            {
                for ( int i = 0; i < n_i; ++i )
                {
                    for ( int j = 0; j < n_j; ++j )
                    {
                        const double x   = static_cast< double >( coords_shell_host( sd, i, j, 0 ) );
                        const double y   = static_cast< double >( coords_shell_host( sd, i, j, 1 ) );
                        const double z   = static_cast< double >( coords_shell_host( sd, i, j, 2 ) );
                        const double rad = std::sqrt( x * x + y * y + z * z );
                        const double phi = std::atan2( y, x );
                        sph_tool.plmbar( plm.data(), static_cast< unsigned int >( l_max ), z / rad );

                        double acc = 0.0;
                        int    k   = 0;
                        for ( int l = l_min; l <= l_max; ++l )
                        {
                            const int base = l * ( l + 1 ) / 2;
                            for ( int m = -l; m <= l; ++m, ++k )
                            {
                                const int    order = ( m < 0 ) ? -m : m;
                                const double trig  = ( m > 0 ) ? std::sin( order * phi ) : std::cos( order * phi );
                                acc += coeff[static_cast< size_t >( k )] * plm[static_cast< size_t >( base + order )] * trig;
                            }
                        }
                        sph_coeffs_host( sd, i, j ) = static_cast< ScalarType >( acc );
                    }
                }
            }
            Kokkos::deep_copy( sph_coeffs, sph_coeffs_host );
            }
            else if ( sph )
            {
                util::logroot << "Adding spherical harmonic perturbation..." << std::endl;


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
            }

            // Both the broadband and the single/double-mode paths above only FILL
            // sph_coeffs; the normalisation and the adder below are what actually apply
            // the perturbation to T, so they must run for either of them.
            if ( broadband_sph || sph )
            {
                ScalarType max_abs_sph = ScalarType( 0 );
                Kokkos::parallel_reduce(
                    "sph_coeffs_max_abs",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 3 > >(
                        { 0, 0, 0 }, { sph_coeffs.extent( 0 ), sph_coeffs.extent( 1 ), sph_coeffs.extent( 2 ) } ),
                    KOKKOS_LAMBDA( int id, int x, int y, ScalarType& max_tmp ) {
                        max_tmp = Kokkos::max( max_tmp, Kokkos::abs( sph_coeffs( id, x, y ) ) );
                    },
                    Kokkos::Max< ScalarType >( max_abs_sph ) );
                if ( max_abs_sph <= ScalarType( 0 ) )
                    max_abs_sph = ScalarType( 1 );

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

    // Project Q1 -> FV
    //fv::hex::l2_project_fe_to_fv( T_fct, T, domain, coords_shell, coords_radii );

    else // FCT
    {
        logroot << "Initial temperature with FCT: power-law + noise" << std::endl;

        Kokkos::parallel_for(
            "initial temp interpolation (FCT)",
            grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
            FVInitialConditionInterpolator{
                domain.domain_info().radii().front(),
                domain.domain_info().radii().back(),
                prm.boundary_parameters.temperature_min,
                prm.boundary_parameters.temperature_max,
                fv_cell_centers.grid_data(),
                T_fct.grid_data() } );
        Kokkos::fence();

        Kokkos::parallel_for(
            "adding noise to temp (FCT)",
            grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
            FVNoiseAdder{
                prm.boundary_parameters.temperature_min,
                prm.boundary_parameters.temperature_max,
                T_fct.grid_data(),
                Kokkos::Random_XorShift64_Pool<>( 12345 ) } );
        Kokkos::fence();

        // Enforce Dirichlet BCs on the FV field and exchange ghost layers.
        fv::hex::apply_dirichlet_bcs( T_fct, boundary_mask, fct_bcs, domain );
        communication::shell::update_fv_ghost_layers( domain, T_fct.grid_data() );

        // Project T_fct → Q1 T so downstream consumers (Stokes RHS, output, Nusselt
        // diagnostic) see a consistent Q1 representation.  Allocate the L2 scratch
        // locally — this is a one-shot setup call, not a hot loop.
        std::vector< linalg::VectorQ1Scalar< ScalarType > > init_l2_tmps;
        init_l2_tmps.reserve( 5 );
        for ( int i = 0; i < 5; ++i )
        {
            init_l2_tmps.emplace_back( "init_l2_tmp_" + std::to_string( i ), domain, ownership_mask );
        }
        fv::hex::l2_project_fv_to_fe_lumped( T, T_fct, domain, coords_shell, coords_radii, init_l2_tmps );
    }
}

// Read custom radial profiles (if given), nondimensionalise and fill respective 2D array.
// Fill radially constant otherwise.
template < typename ScalarType >
void radial_profile_init(
    grid::Grid2DDataScalar< ScalarType >&       rho_profile,
    grid::Grid2DDataScalar< ScalarType >&       alpha_profile,
    grid::Grid2DDataScalar< ScalarType >&       cp_profile,
    grid::Grid2DDataScalar< ScalarType >&       kappa_profile,
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

    // Kappa
    // Compute diffusivity profile from k (=1), cp_profile and rho_profile
    Kokkos::parallel_for(
        "compute kappa_profile",
        grid::shell::local_domain_md_range_policy_radial( domain ),
        KOKKOS_LAMBDA( int id, int r ) {
            kappa_profile( id, r ) = ScalarType( 1 ) / ( rho_profile( id, r ) * cp_profile( id, r ) );
        } );
    Kokkos::fence();
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

/// Load (u, T) from an XDMF checkpoint and rebuild T_fct via FE→FV projection.
template < typename ScalarType >
void load_temperature_checkpoint(
    linalg::VectorQ1Vec< ScalarType, 3 >&       u_velocity,
    linalg::VectorQ1Scalar< ScalarType >&       T,
    linalg::VectorFVScalar< ScalarType >&       T_fct,
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

    // T_fct is not stored in checkpoints (only Q1 T is).  Recover it via FE→FV
    // projection.  Ghost layers are populated inside l2_project_fe_to_fv, so
    // the result is immediately usable by FCT kernels.
    fv::hex::l2_project_fe_to_fv( T_fct, T, domain, coords_shell, coords_radii );
}

} // namespace terra::mantlecirculation
