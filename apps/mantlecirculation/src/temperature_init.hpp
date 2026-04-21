#pragma once

#include <string>
#include <vector>

#include "communication/shell/fv_communication.hpp"
#include "fv/hex/conversion.hpp"
#include "fv/hex/helpers.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"
#include "shell/spherical_harmonics.hpp"
#include "util/logging.hpp"

#include "interpolators.hpp"
#include "io.hpp"
#include "parameters.hpp"

namespace terra::mantlecirculation {

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
    linalg::VectorQ1Scalar< ScalarType >&                                   T,
    linalg::VectorFVScalar< ScalarType >&                                   T_fct,
    const fv::hex::DirichletBCs< ScalarType >&                              fct_bcs,
    const grid::shell::DistributedDomain&                                   domain,
    const grid::Grid3DDataVec< ScalarType, 3 >&                             coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                             coords_radii,
    const linalg::VectorFVVec< ScalarType, 3 >&                             fv_cell_centers,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&                ownership_mask,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >&         boundary_mask,
    const Parameters&                                                       prm )
{
    using util::logroot;
    const auto& init_temp = prm.physics_parameters.initial_temperature;

    if ( init_temp.profile == InitialTemperatureProfile::CONDUCTIVE )
    {
        const bool has_sph_2 = ( init_temp.sph_degree_l_2 > 0 && init_temp.sph_factor_2 != 0.0 &&
                                 init_temp.sph_epsilon != 0.0 );

        logroot << "Initial temperature: conductive profile";
        if ( init_temp.sph_degree_l > 0 && init_temp.sph_epsilon != 0.0 )
        {
            logroot << " + eps * (Y_" << init_temp.sph_degree_l << "^" << init_temp.sph_order_m;
            if ( has_sph_2 )
            {
                logroot << " + " << init_temp.sph_factor_2 << " * Y_" << init_temp.sph_degree_l_2 << "^"
                        << init_temp.sph_order_m_2;
            }
            logroot << ") (eps=" << init_temp.sph_epsilon << ")";
        }
        logroot << std::endl;

        // Compute spherical-harmonic coefficients on unit-sphere Q1 nodes.
        grid::Grid3DDataScalar< ScalarType > sph_coeffs;
        const bool has_sph = ( init_temp.sph_degree_l > 0 && init_temp.sph_epsilon != 0.0 );
        if ( has_sph )
        {
            sph_coeffs = shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >(
                init_temp.sph_degree_l, init_temp.sph_order_m, coords_shell );

            if ( has_sph_2 )
            {
                auto sph_coeffs_2 = shell::spherical_harmonics_coefficients_grid< ScalarType, ScalarType >(
                    init_temp.sph_degree_l_2, init_temp.sph_order_m_2, coords_shell );
                const ScalarType factor_2 = static_cast< ScalarType >( init_temp.sph_factor_2 );
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
        }

        // Fill Q1 T with the conductive profile + spherical-harmonic perturbation.
        Kokkos::parallel_for(
            "initial temp (conductive + sph. harm.)",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            ConductiveProfileInterpolator{
                domain.domain_info().radii().front(),
                domain.domain_info().radii().back(),
                init_temp.sph_epsilon,
                coords_shell,
                coords_radii,
                T.grid_data(),
                sph_coeffs,
                has_sph } );
        Kokkos::fence();
        // NOTE: do NOT call send_recv here.  The kernel writes the same value to every
        // subdomain copy of each shared node already; send_recv (SUM) would accumulate
        // them and produce N*value at shared nodes.  The downstream FE→FV→FE projection
        // re-establishes consistency.

        // Project Q1 T → FV T_fct.
        fv::hex::l2_project_fe_to_fv( T_fct, T, domain, coords_shell, coords_radii );
    }
    else
    {
        logroot << "Initial temperature: power-law + noise" << std::endl;

        Kokkos::parallel_for(
            "initial temp interpolation (FCT)",
            grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
            FVInitialConditionInterpolator{
                domain.domain_info().radii().front(),
                domain.domain_info().radii().back(),
                fv_cell_centers.grid_data(),
                T_fct.grid_data() } );
        Kokkos::fence();

        Kokkos::parallel_for(
            "adding noise to temp (FCT)",
            grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
            FVNoiseAdder{ T_fct.grid_data(), Kokkos::Random_XorShift64_Pool<>( 12345 ) } );
        Kokkos::fence();
    }

    // Enforce Dirichlet BCs on the FV field and exchange ghost layers.
    fv::hex::apply_dirichlet_bcs( T_fct, boundary_mask, fct_bcs, domain );
    communication::shell::update_fv_ghost_layers( domain, T_fct.grid_data() );

    // Project T_fct → Q1 T so downstream consumers (Stokes RHS, output, Nusselt
    // diagnostic) see a consistent Q1 representation.  Allocate the L2 scratch
    // locally — this is a one-shot setup call, not a hot loop.
    {
        std::vector< linalg::VectorQ1Scalar< ScalarType > > init_l2_tmps;
        init_l2_tmps.reserve( 5 );
        for ( int i = 0; i < 5; ++i )
        {
            init_l2_tmps.emplace_back(
                "init_l2_tmp_" + std::to_string( i ), domain, ownership_mask );
        }
        fv::hex::l2_project_fv_to_fe_lumped(
            T, T_fct, domain, coords_shell, coords_radii, init_l2_tmps );
    }
}

/// Spherical steady-state conduction profile  T_ref(r) = r_min·r_max/r − r_min.
/// Used as the reference temperature for the Nusselt-number diagnostic and
/// added to XDMF output for visualisation.
template < typename ScalarType >
void compute_reference_conductive_profile(
    linalg::VectorQ1Scalar< ScalarType >&                                   T_ref,
    const grid::shell::DistributedDomain&                                   domain,
    const grid::Grid3DDataVec< ScalarType, 3 >&                             coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                             coords_radii )
{
    Kokkos::parallel_for(
        "conductive profile T_ref",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        ConductiveProfileInterpolator{
            domain.domain_info().radii().front(),
            domain.domain_info().radii().back(),
            ScalarType( 0 ),
            coords_shell,
            coords_radii,
            T_ref.grid_data(),
            {},
            false } );
    Kokkos::fence();
    // NOTE: do NOT call send_recv here.  Same rationale as in the IC kernel:
    // every subdomain copy of a shared node already gets the correct value, so
    // a SUM exchange would multiply it by the sharing count.
}

/// Load (u, T) from an XDMF checkpoint and rebuild T_fct via FE→FV projection.
/// Returns the simulation timestep to resume from (uses
/// `checkpoint_timestep` if set, else falls back to the file step).
///
/// Pre-condition: `prm.io_parameters.checkpoint_dir` is non-empty and
/// `checkpoint_step >= 0`.
template < typename ScalarType >
int load_temperature_checkpoint(
    linalg::VectorQ1Vec< ScalarType, 3 >&                                   u_velocity,
    linalg::VectorQ1Scalar< ScalarType >&                                   T,
    linalg::VectorFVScalar< ScalarType >&                                   T_fct,
    const grid::shell::DistributedDomain&                                   domain,
    const grid::Grid3DDataVec< ScalarType, 3 >&                             coords_shell,
    const grid::Grid2DDataScalar< ScalarType >&                             coords_radii,
    const Parameters&                                                       prm )
{
    using util::logroot;

    const int  checkpoint_file_step = prm.io_parameters.checkpoint_step;
    const int  timestep_initial     = ( prm.io_parameters.checkpoint_timestep >= 0 )
                                          ? prm.io_parameters.checkpoint_timestep
                                          : checkpoint_file_step;

    logroot << "Loading checkpoint from " << prm.io_parameters.checkpoint_dir
            << " (file step " << checkpoint_file_step
            << ", simulation timestep " << timestep_initial << ")" << std::endl;

    auto success_vel = io::read_xdmf_checkpoint_grid(
        prm.io_parameters.checkpoint_dir,
        std::string( "u_u" ),
        checkpoint_file_step,
        domain,
        u_velocity.grid_data() );
    if ( success_vel.is_err() )
    {
        Kokkos::abort( success_vel.error().c_str() );
    }

    auto success_temp = io::read_xdmf_checkpoint_grid(
        prm.io_parameters.checkpoint_dir,
        std::string( "T" ),
        checkpoint_file_step,
        domain,
        T.grid_data() );
    if ( success_temp.is_err() )
    {
        Kokkos::abort( success_temp.error().c_str() );
    }

    // T_fct is not stored in checkpoints (only Q1 T is).  Recover it via FE→FV
    // projection.  Ghost layers are populated inside l2_project_fe_to_fv, so
    // the result is immediately usable by FCT kernels.
    fv::hex::l2_project_fe_to_fv( T_fct, T, domain, coords_shell, coords_radii );

    return timestep_initial;
}

} // namespace terra::mantlecirculation
