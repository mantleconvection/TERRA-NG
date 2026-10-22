#pragma once

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "io/lookup_table_2d_reader.hpp"
#include "linalg/vector_q1.hpp"
#include "parameters.hpp"
#include "util/logging.hpp"

namespace terra::mantlecirculation {

using grid::Grid2DDataScalar;
using grid::Grid4DDataScalar;
using linalg::VectorQ1Scalar;

/// Time derivative of hydrostatic density via BDF2 (non-uniform timestep), evaluated using rho^n, rho^{n-1}, rho^{n-2}.
struct ComputeDensityTimeDerivativeBDF2
{
    Grid4DDataScalar< ScalarType > time_derivative_;
    Grid4DDataScalar< ScalarType > density_;
    Grid4DDataScalar< ScalarType > density_prev_;
    Grid4DDataScalar< ScalarType > density_prev_2_;
    ScalarType                     dt_;
    ScalarType                     dt_prev_;

    ComputeDensityTimeDerivativeBDF2(
        const Grid4DDataScalar< ScalarType >& time_derivative,
        const Grid4DDataScalar< ScalarType >& density,
        const Grid4DDataScalar< ScalarType >& density_prev,
        const Grid4DDataScalar< ScalarType >& density_prev_2,
        const ScalarType                      dt,
        const ScalarType                      dt_prev )
    : time_derivative_( time_derivative )
    , density_( density )
    , density_prev_( density_prev )
    , density_prev_2_( density_prev_2 )
    , dt_( dt )
    , dt_prev_( dt_prev )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const ScalarType rhoN   = density_( id, x, y, r );
        const ScalarType rhoN1  = density_prev_( id, x, y, r );
        const ScalarType rhoN2  = density_prev_2_( id, x, y, r );
        const ScalarType dt_sum = dt_ + dt_prev_;

        time_derivative_( id, x, y, r ) =
            ( ( ( ScalarType( 2 ) * dt_ + dt_prev_ ) * rhoN / ( dt_sum ) ) - ( dt_sum * rhoN1 / dt_prev_ ) +
              ( dt_ * dt_ * rhoN2 / ( dt_prev_ * dt_sum ) ) ) /
            dt_;
    }
};

/// Time derivative of hydrostatic density via first-order backward difference, evaluated using rho^n, rho^{n-1}. Used at timestep == 1, where no rho^{n-2} history is available yet.
struct ComputeDensityTimeDerivative
{
    Grid4DDataScalar< ScalarType > time_derivative_;
    Grid4DDataScalar< ScalarType > density_;
    Grid4DDataScalar< ScalarType > density_prev_;
    ScalarType                     dt_;

    ComputeDensityTimeDerivative(
        const Grid4DDataScalar< ScalarType >& time_derivative,
        const Grid4DDataScalar< ScalarType >& density,
        const Grid4DDataScalar< ScalarType >& density_prev,
        const ScalarType                      dt )
    : time_derivative_( time_derivative )
    , density_( density )
    , density_prev_( density_prev )
    , dt_( dt )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const ScalarType rhoN  = density_( id, x, y, r );
        const ScalarType rhoN1 = density_prev_( id, x, y, r );

        time_derivative_( id, x, y, r ) = ( rhoN - rhoN1 ) / dt_;
    }
};

/// Owns the density history and lookup table needed for the "Projected Density
/// Approximation" (PDA). Callers need to:
/// 1. call update_density_from_table() once per timestep, after the energy solve,
/// 2. call compute_density_time_derivative() before assembling the Stoke rhs,
/// 3. call seed_history() once, after the first (pre-loop) Stokes solve,
/// 4. call shift_history() once per timestep, after the Picard loop.
template < typename ScalarType >
class PDAManager
{
  public:
    PDAManager(
        const grid::shell::DistributedDomain&              domain,
        const Grid2DDataScalar< ScalarType >&              coords_radii,
        const Grid4DDataScalar< grid::NodeOwnershipFlag >& ownership_mask,
        const Parameters&                                  prm )
    : domain_( domain )
    , hydrostatic_pressure_( "hydrostatic_pressure", coords_radii.extent( 0 ), coords_radii.extent( 1 ) )
    , density_( "density", domain, ownership_mask )
    , density_prev_( "density_prev", domain, ownership_mask )
    , density_prev_2_( "density_prev_2", domain, ownership_mask )
    , drho_dt_( "drho_dt", domain, ownership_mask )
    , lookup_table_( terra::io::read_lookup_table_2d(
          prm.physics_parameters.pda_parameters.mintable_path,
          prm.physics_parameters.pda_parameters.table_density_col,
          LookupTableLayout( prm ),
          "density_from_table" ) )
    , prm_( prm )
    {
        auto pda_params = prm_.physics_parameters.pda_parameters;

        // read hydrostatic pressure profile from file
        // pressure stays dimensional, since only used for table-lookup
        hydrostatic_pressure_ = shell::interpolate_radial_profile_into_subdomains_from_csv(
            pda_params.pressure_profile_csv_path,
            prm_.physics_parameters.radial_profiles_radii_key,
            pda_params.pressure_profile_value_key,
            coords_radii,
            ScalarType( 1 ) / prm_.mesh_parameters.mantle_thickness_m,
            ScalarType( 1 ) );
    }

    // Evaluate density from (p, T) via lookup table into the owned density field
    void update_density_from_table( const VectorQ1Scalar< ScalarType >& temperature )
    {
        util::logroot << "Updating density (PDA) ..." << std::endl;

        auto        density_grid      = density_.grid_data();
        const auto  temperature_grid  = temperature.grid_data();
        const auto  pressure_grid     = hydrostatic_pressure_;
        const auto  delta_T_K         = prm_.boundary_parameters.delta_T_K;
        const auto  reference_density = prm_.physics_parameters.reference_density;
        const auto& lookup_table      = lookup_table_;

        Kokkos::parallel_for(
            "read_density_from_table",
            grid::shell::local_domain_md_range_policy_nodes( domain_ ),
            KOKKOS_LAMBDA( const int id, const int x, const int y, const int r ) {
                const ScalarType local_density =
                    lookup_table( pressure_grid( id, r ), temperature_grid( id, x, y, r ) * delta_T_K );

                // Nondimensionalise
                density_grid( id, x, y, r ) = local_density / reference_density;
            } );
        Kokkos::fence();
    }

    /// Computes drho/dt into the owned field drho_dt_.
    /// timestep == 0: no history yet -- leaves drho_dt_untouched,
    /// timestep == 1: first-order backward difference,
    /// timestep >= 2: BDF2.
    void compute_density_time_derivative( const int timestep, const ScalarType dt, const ScalarType dt_prev )
    {
        if ( timestep == 0 )
            return;

        if ( timestep == 1 )
        {
            // Compute first-order time derivative
            Kokkos::parallel_for(
                "ComputeDensityTimeDerivative",
                grid::shell::local_domain_md_range_policy_nodes( domain_ ),
                ComputeDensityTimeDerivative{
                    drho_dt_.grid_data(), density_.grid_data(), density_prev_.grid_data(), dt } );
            Kokkos::fence();
        }

        else // timestep > 1
        {
            // Compute BDF2 time derivative
            Kokkos::parallel_for(
                "ComputeDensityTimeDerivativeBDF2",
                grid::shell::local_domain_md_range_policy_nodes( domain_ ),
                ComputeDensityTimeDerivativeBDF2{
                    drho_dt_.grid_data(),
                    density_.grid_data(),
                    density_prev_.grid_data(),
                    density_prev_2_.grid_data(),
                    dt,
                    dt_prev } );
            Kokkos::fence();
        }
    }

    // Call once, right after the initial (pre-loop) Stokes solve.
    void seed_history() { Kokkos::deep_copy( density_prev_.grid_data(), density_.grid_data() ); }

    // Call once per timestep, after Picard loop.
    void shift_history()
    {
        Kokkos::deep_copy( density_prev_2_.grid_data(), density_prev_.grid_data() );
        Kokkos::deep_copy( density_prev_.grid_data(), density_.grid_data() );
    }

    // Accessors
    const VectorQ1Scalar< ScalarType >& density() const { return density_; }
    VectorQ1Scalar< ScalarType >&       density() { return density_; }
    const VectorQ1Scalar< ScalarType >& drho_dt() const { return drho_dt_; }
    VectorQ1Scalar< ScalarType >&       drho_dt() { return drho_dt_; }

  private:
    grid::shell::DistributedDomain               domain_;
    Grid2DDataScalar< ScalarType >               coords_radii_;
    Grid2DDataScalar< ScalarType >               hydrostatic_pressure_;
    VectorQ1Scalar< ScalarType >                 density_;
    VectorQ1Scalar< ScalarType >                 density_prev_;
    VectorQ1Scalar< ScalarType >                 density_prev_2_;
    VectorQ1Scalar< ScalarType >                 drho_dt_;
    terra::io::ScalarLookupTable2D< ScalarType > lookup_table_;
    Parameters                                   prm_;

    static terra::io::GridLayout2D LookupTableLayout( const Parameters& prm )
    {
        // Assuming col[0] = P, col[1] = T and table varies fastest in temperature.
        return terra::io::GridLayout2D{
            prm.physics_parameters.pda_parameters.table_nP,
            prm.physics_parameters.pda_parameters.table_nT,
            prm.physics_parameters.pda_parameters.table_min_P,
            prm.physics_parameters.pda_parameters.table_min_T,
            prm.physics_parameters.pda_parameters.table_dP,
            prm.physics_parameters.pda_parameters.table_dT,
            prm.physics_parameters.pda_parameters.table_nT,
            1 };
    }
};
} // namespace terra::mantlecirculation
