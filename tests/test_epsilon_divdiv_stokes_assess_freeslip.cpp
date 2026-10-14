/// \file test_epsilon_divdiv_stokes_assess_freeslip.cpp
///
/// \brief Stokes convergence test using analytical spherical-harmonic solutions from the
/// "assess" package (Analytical Solutions for the Stokes Equations in Spherical Shells).

// Solution: SphericalStokesSolutionSmoothFreeZeroSlip(l=2, m=2, k=2, Rp=1.0, Rm=0.5, nu=1.0, g=1.0)
//   - Free-slip BC at CMB (Rm=0.5), Dirichlet (zero-slip) BC at surface (Rp=1.0)
//   - Constant viscosity nu = 1
//   - Smooth r^k forcing: f = -g (r/Rp)^k Y_lm(theta,phi) r_hat
//
// The velocity is derived from a poloidal function P(r,theta,phi) = Pl(r) * Y_22(theta,phi):
//   u = curl(r x grad(P))
// The coefficients A,B,C,D,E for Pl(r) and G,H,K for pressure are pre-computed by assess.

#include "../src/terra/communication/shell/communication.hpp"

#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/strong_algebraic_freeslip_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/identity.hpp"
#include "fe/wedge/operators/shell/kmass.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/prolongation_linear.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/restriction_linear.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"

#include "grid/shell/bit_masks.hpp"

#include "io/xdmf.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/gca/gca.hpp"
#include "linalg/solvers/gca/gca_elements_collector.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "terra/linalg/solvers/chebyshev.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/pminres.hpp"
#include "util/info.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"

#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/vtk.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/linalg/diagonally_scaled_operator.hpp"
#include "terra/linalg/solvers/diagonal_solver.hpp"
#include "terra/linalg/solvers/power_iteration.hpp"
#include "terra/shell/radial_profiles.hpp"

#include "util/init.hpp"
#include "util/table.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <tuple>
#include <vector>

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::get_shell_boundary_flag;
using grid::shell::SubdomainInfo;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::DiagonallyScaledOperator;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using linalg::solvers::DiagonalSolver;
using linalg::solvers::power_iteration;
using linalg::solvers::TwoGridGCA;
using terra::grid::shell::BoundaryConditions;

// =============================================================================
// Assess analytical solution: SphericalStokesSolutionSmoothFreeZeroSlip
//   l=2, m=2, k_forcing=2, Rp=1.0, Rm=0.5, nu=1.0, g=1.0
//   Free-slip at Rm (CMB), zero-slip at Rp (surface)
// =============================================================================

namespace assess_solution
{

// Poloidal coefficients: Pl(r) = A*r^l + B*r^(-l-1) + C*r^(l+2) + D*r^(-l+1) + E*r^(k+3)
// With l=2, k=2: Pl(r) = A*r^2 + B*r^-3 + C*r^4 + D*r^-1 + E*r^5
constexpr double coeff_A =  4.54414476717381228543e-03;
constexpr double coeff_B =  3.74596588289534304281e-05;
constexpr double coeff_C = -1.10448363301060385744e-02;
constexpr double coeff_D = -4.81212540341170963258e-04;
constexpr double coeff_E =  6.94444444444444405895e-03;

// Pressure coefficients: p(r,θ,φ) = (G*r^l + H*r^(-l-1) + K*r^(k+1)) * Y_22
constexpr double coeff_G =  4.63883125864453627063e-01;
constexpr double coeff_H =  5.77455048409405112542e-03;
constexpr double coeff_K = -6.66666666666666629659e-01;

constexpr int    l_deg      = 2;
constexpr double Rp         = 1.0;
constexpr double Rm         = 0.5;
constexpr double nu         = 1.0;
constexpr double gravity    = 1.0;

// Y_22(θ,φ) = sqrt(15/(32π)) sin²(θ) cos(2φ)
KOKKOS_INLINE_FUNCTION
double Y22( const double theta, const double phi )
{
    constexpr double c = 0.38627420202318958;  // sqrt(15/(32*pi))
    const double st = Kokkos::sin( theta );
    return c * st * st * Kokkos::cos( 2.0 * phi );
}

// dY_22/dθ = sqrt(15/(32π)) 2 sin(θ) cos(θ) cos(2φ)
KOKKOS_INLINE_FUNCTION
double dY22_dtheta( const double theta, const double phi )
{
    constexpr double c = 0.38627420202318958;
    return c * 2.0 * Kokkos::sin( theta ) * Kokkos::cos( theta ) * Kokkos::cos( 2.0 * phi );
}

// dY_22/dφ = sqrt(15/(32π)) sin²(θ) (-2 sin(2φ))
KOKKOS_INLINE_FUNCTION
double dY22_dphi( const double theta, const double phi )
{
    constexpr double c = 0.38627420202318958;
    const double st = Kokkos::sin( theta );
    return c * st * st * ( -2.0 * Kokkos::sin( 2.0 * phi ) );
}

// Pl(r) = A*r^2 + B*r^-3 + C*r^4 + D*r^-1 + E*r^5
KOKKOS_INLINE_FUNCTION
double Pl( const double r )
{
    const double r2 = r * r;
    const double r4 = r2 * r2;
    const double inv_r = 1.0 / r;
    return coeff_A * r2 + coeff_B * inv_r * inv_r * inv_r +
           coeff_C * r4 + coeff_D * inv_r + coeff_E * r4 * r;
}

// dPl/dr = 2A*r + (-3)B*r^-4 + 4C*r^3 + (-1)D*r^-2 + 5E*r^4
KOKKOS_INLINE_FUNCTION
double dPldr( const double r )
{
    const double r2 = r * r;
    const double r3 = r2 * r;
    const double inv_r = 1.0 / r;
    const double inv_r2 = inv_r * inv_r;
    return 2.0 * coeff_A * r - 3.0 * coeff_B * inv_r2 * inv_r2 +
           4.0 * coeff_C * r3 - coeff_D * inv_r2 + 5.0 * coeff_E * r2 * r2;
}

// Cartesian (cx,cy,cz) → spherical (r, theta, phi)
KOKKOS_INLINE_FUNCTION
void to_spherical( const double cx, const double cy, const double cz,
                   double& r, double& theta, double& phi )
{
    r     = Kokkos::sqrt( cx * cx + cy * cy + cz * cz );
    theta = Kokkos::acos( cz / r );
    phi   = Kokkos::atan2( cy, cx );
}

// Velocity in Cartesian coordinates from (u_r, u_theta, u_phi)
// u_r     = -l(l+1) Pl(r) Y / r
// u_theta = -(Pl/r + dPl/dr) dY/dtheta
// u_phi   = -(Pl/r + dPl/dr) / sin(theta) dY/dphi
KOKKOS_INLINE_FUNCTION
void velocity_cartesian( const double cx, const double cy, const double cz,
                         double& ux, double& uy, double& uz )
{
    double r, theta, phi;
    to_spherical( cx, cy, cz, r, theta, phi );

    const double st   = Kokkos::sin( theta );
    const double ct   = Kokkos::cos( theta );
    const double P    = Pl( r );
    const double dP   = dPldr( r );
    const double Yr   = Y22( theta, phi );
    const double dYdt = dY22_dtheta( theta, phi );
    const double dYdp = dY22_dphi( theta, phi );

    const double u_r     = -static_cast< double >( l_deg * ( l_deg + 1 ) ) * P * Yr / r;
    const double radial  = P / r + dP;
    const double u_theta = -radial * dYdt;

    // u_phi = -radial / sin(theta) * dY/dphi
    // For Y_22: dY/dphi ~ sin^2(theta), so u_phi ~ sin(theta) — removable singularity at poles
    const double req = Kokkos::sqrt( cx * cx + cy * cy );

    if ( req < 1.0e-14 * r )
    {
        // At poles: Y_22 = 0, dY/dtheta = 0, dY/dphi = 0 → velocity = 0
        ux = 0.0;
        uy = 0.0;
        uz = 0.0;
        return;
    }

    const double u_phi = -radial / st * dYdp;

    // Spherical to Cartesian:
    // u_x = (x/r)*u_r + (x/req)*cos(theta)*u_theta - (y/req)*u_phi
    // u_y = (y/r)*u_r + (y/req)*cos(theta)*u_theta + (x/req)*u_phi
    // u_z = (z/r)*u_r - sin(theta)*u_theta
    const double inv_req = 1.0 / req;
    ux = cx / r * u_r + cx * inv_req * ct * u_theta - cy * inv_req * u_phi;
    uy = cy / r * u_r + cy * inv_req * ct * u_theta + cx * inv_req * u_phi;
    uz = cz / r * u_r - st * u_theta;
}

// Pressure: p(r,θ,φ) = (G*r^2 + H*r^-3 + K*r^3) * Y_22(θ,φ)
KOKKOS_INLINE_FUNCTION
double pressure_cartesian( const double cx, const double cy, const double cz )
{
    double r, theta, phi;
    to_spherical( cx, cy, cz, r, theta, phi );

    const double r2 = r * r;
    const double r3 = r2 * r;
    return ( coeff_G * r2 + coeff_H / r3 + coeff_K * r3 ) * Y22( theta, phi );
}

// Density perturbation: delta_rho(r,θ,φ) = (r/Rp)^k * Y_22(θ,φ)
// Body force (strong form): f_i = -g * delta_rho * x_i / r
KOKKOS_INLINE_FUNCTION
void body_force_cartesian( const double cx, const double cy, const double cz,
                           double& fx, double& fy, double& fz )
{
    double r, theta, phi;
    to_spherical( cx, cy, cz, r, theta, phi );

    const double rho = ( r / Rp ) * ( r / Rp ) * Y22( theta, phi );  // (r/Rp)^2 * Y_22
    const double f_radial = -gravity * rho / r;

    fx = f_radial * cx;
    fy = f_radial * cy;
    fz = f_radial * cz;
}

} // namespace assess_solution

// =============================================================================
// Interpolators
// =============================================================================

struct AssessSolutionVelocityInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataVec< double, 3 >                         data_u_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    AssessSolutionVelocityInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataVec< double, 3 >&                         data_u,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const bool                                                only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , mask_( mask )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || on_boundary )
        {
            double ux, uy, uz;
            assess_solution::velocity_cartesian( coords( 0 ), coords( 1 ), coords( 2 ), ux, uy, uz );
            data_u_( local_subdomain_id, x, y, r, 0 ) = ux;
            data_u_( local_subdomain_id, x, y, r, 1 ) = uy;
            data_u_( local_subdomain_id, x, y, r, 2 ) = uz;
        }
    }
};

struct AssessSolutionPressureInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataScalar< double >                         data_p_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    AssessSolutionPressureInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataScalar< double >&                         data_p,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const bool                                                only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_p_( data_p )
    , mask_( mask )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || on_boundary )
        {
            data_p_( local_subdomain_id, x, y, r ) =
                assess_solution::pressure_cartesian( coords( 0 ), coords( 1 ), coords( 2 ) );
        }
    }
};

struct AssessRHSVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;

    AssessRHSVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        double fx, fy, fz;
        assess_solution::body_force_cartesian( coords( 0 ), coords( 1 ), coords( 2 ), fx, fy, fz );

        data_( local_subdomain_id, x, y, r, 0 ) = fx;
        data_( local_subdomain_id, x, y, r, 1 ) = fy;
        data_( local_subdomain_id, x, y, r, 2 ) = fz;
    }
};

struct ConstantViscosityInterpolator
{
    Grid4DDataScalar< double > data_;

    ConstantViscosityInterpolator( const Grid4DDataScalar< double >& data )
    : data_( data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        data_( local_subdomain_id, x, y, r ) = assess_solution::nu;
    }
};

// =============================================================================
// Test
// =============================================================================

std::tuple< double, double, int >
test( int    min_level,
      int    max_level,
      int    level_subdomains,
      const std::shared_ptr< util::Table >& table )
{
    using ScalarType = double;

    std::vector< DistributedDomain >                                  domains;
    std::vector< Grid3DDataVec< double, 3 > >                         coords_shell;
    std::vector< Grid2DDataScalar< double > >                         coords_radii;
    std::vector< Grid4DDataScalar< grid::NodeOwnershipFlag > >        mask_data;
    std::vector< Grid4DDataScalar< grid::shell::ShellBoundaryFlag > > boundary_mask_data;

    ScalarType r_min = assess_solution::Rm;
    ScalarType r_max = assess_solution::Rp;

    util::logroot << "Allocating domains ...\n";
    for ( int level = min_level; level <= max_level; level++ )
    {
        const int idx = level - min_level;

        domains.push_back(
            DistributedDomain::create_uniform(
                level, level, r_min, r_max, level_subdomains, level_subdomains ) );

        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( grid::setup_node_ownership_mask_data( domains[idx] ) );
        boundary_mask_data.push_back( grid::shell::setup_boundary_mask_data( domains[idx] ) );
    }

    const auto num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    std::vector< std::string >                             stok_vec_names = { "u", "f", "solution", "error" };
    constexpr int                                          num_stok_tmps  = 8;

    util::logroot << "Allocating temps ...\n";
    for ( int i = 0; i < num_stok_tmps; i++ )
    {
        stok_vec_names.push_back( "tmp_" + std::to_string( i ) );
    }

    for ( const auto& name : stok_vec_names )
    {
        stok_vecs[name] = VectorQ1IsoQ2Q1< ScalarType >(
            name,
            domains[velocity_level],
            domains[pressure_level],
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    auto& u        = stok_vecs["u"];
    auto& f        = stok_vecs["f"];
    auto& solution = stok_vecs["solution"];
    auto& error    = stok_vecs["error"];

    std::vector< VectorQ1Vec< ScalarType > > tmp_mg;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_r;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_e;

    for ( size_t level = 0; level < num_levels; level++ )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );
        }
    }

    const auto num_dofs_velocity =
        3 * kernels::common::count_masked< long >( mask_data[num_levels - 1], grid::NodeOwnershipFlag::OWNED );
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( mask_data[num_levels - 2], grid::NodeOwnershipFlag::OWNED );

    BoundaryConditions bcs = {
        { CMB, FREESLIP },
        { SURFACE, DIRICHLET },
    };
    BoundaryConditions bcs_neumann = {
        { CMB, NEUMANN },
        { SURFACE, NEUMANN },
    };

    util::logroot << "Setting operators ...\n";
    using Stokes      = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using Gradient    = Stokes::Block12Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    // Constant viscosity k = nu = 1.0
    VectorQ1Scalar< ScalarType > k( "k", domains[velocity_level], mask_data[velocity_level] );

    util::logroot << "Interpolating k (constant viscosity = " << assess_solution::nu << ") ...\n";
    Kokkos::parallel_for(
        "constant viscosity interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        ConstantViscosityInterpolator( k.grid_data() ) );

    Stokes K_op(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k.grid_data(),
        bcs,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k.grid_data(),
        bcs_neumann,
        false );

    Stokes K_neumann_diag(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        boundary_mask_data[velocity_level],
        k.grid_data(),
        bcs_neumann,
        true );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    std::vector< Viscous >      A_diag;
    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;

    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;

    util::logroot << "MG hierarchy ...\n";
    for ( size_t level = 0; level < num_levels; level++ )
    {
        VectorQ1Scalar< ScalarType > k_c( "k_c", domains[level], mask_data[level] );
        Kokkos::parallel_for(
            "constant viscosity interpolation (mg)",
            local_domain_md_range_policy_nodes( domains[level] ),
            ConstantViscosityInterpolator( k_c.grid_data() ) );

        A_diag.emplace_back(
            domains[level],
            coords_shell[level],
            coords_radii[level],
            boundary_mask_data[level],
            k_c.grid_data(),
            bcs,
            true );

        if ( level < num_levels - 1 )
        {
            A_c.emplace_back(
                domains[level],
                coords_shell[level],
                coords_radii[level],
                boundary_mask_data[level],
                k_c.grid_data(),
                bcs,
                false );

            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // Interpolate analytical solution
    Kokkos::parallel_for(
        "assess solution interpolation (velocity)",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        AssessSolutionVelocityInterpolator(
            coords_shell[velocity_level],
            coords_radii[velocity_level],
            stok_vecs["solution"].block_1().grid_data(),
            boundary_mask_data[velocity_level],
            false ) );

    Kokkos::parallel_for(
        "assess solution interpolation (pressure)",
        local_domain_md_range_policy_nodes( domains[pressure_level] ),
        AssessSolutionPressureInterpolator(
            coords_shell[pressure_level],
            coords_radii[pressure_level],
            stok_vecs["solution"].block_2().grid_data(),
            boundary_mask_data[pressure_level],
            false ) );

    // RHS: f = M * body_force
    Kokkos::parallel_for(
        "assess rhs interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        AssessRHSVelocityInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level], stok_vecs["tmp_1"].block_1().grid_data() ) );

    linalg::apply( M, stok_vecs["tmp_1"].block_1(), stok_vecs["f"].block_1() );

    // Free-slip at CMB: zero only the normal component of the RHS
    fe::strong_algebraic_freeslip_enforcement_in_place(
        stok_vecs["f"], coords_shell[velocity_level], boundary_mask_data[velocity_level], CMB );

    // Zero-slip at surface: zero all velocity RHS components
    fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
        stok_vecs["f"], boundary_mask_data[velocity_level], SURFACE );

    using Smoother = linalg::solvers::Chebyshev< Viscous >;

    std::vector< Smoother > smoothers;
    for ( size_t level = 0; level < num_levels; level++ )
    {
        inverse_diagonals.emplace_back(
            "inverse_diagonal_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( level ), domains[level], mask_data[level] );

        linalg::assign( tmp, 1.0 );

        if ( level == num_levels - 1 )
        {
            K_op.block_11().set_diagonal( true );
            linalg::apply( K_op.block_11(), tmp, inverse_diagonals.back() );
            K_op.block_11().set_diagonal( false );
        }
        else
        {
            A_c[level].set_diagonal( true );
            linalg::apply( A_c[level], tmp, inverse_diagonals.back() );
            A_c[level].set_diagonal( false );
        }

        linalg::invert_entries( inverse_diagonals.back() );

        constexpr int chebyshev_order      = 2;
        constexpr int chebyshev_iterations = 3;

        std::vector< VectorQ1Vec< ScalarType > > cheby_tmps;
        cheby_tmps.emplace_back( "cheby_tmp_0_" + std::to_string( level ), domains[level], mask_data[level] );
        cheby_tmps.emplace_back( "cheby_tmp_1_" + std::to_string( level ), domains[level], mask_data[level] );

        smoothers.emplace_back( chebyshev_order, inverse_diagonals[level], cheby_tmps, chebyshev_iterations );

        util::logroot << "Chebyshev smoother on level " << level << " (order=" << chebyshev_order
                      << ", iterations=" << chebyshev_iterations << ")\n";
    }

    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;

    std::vector< VectorQ1Vec< ScalarType > > coarse_grid_tmps;
    for ( int i = 0; i < 4; i++ )
    {
        coarse_grid_tmps.emplace_back( "tmp_coarse_grid", domains[0], mask_data[0] );
    }

    CoarseGridSolver coarse_grid_solver(
        linalg::solvers::IterativeSolverParameters{ 1000, 1e-6, 1e-16 }, table, coarse_grid_tmps );

    constexpr auto num_mg_cycles = 2;

    using PrecVisc =
        linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;

    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_grid_solver, num_mg_cycles, 1e-8 );

    // Schur complement preconditioner: inverse lumped mass with k_inv = 1/nu = 1
    VectorQ1Scalar< ScalarType > k_pm( "k_pm", domains[max_level - min_level], mask_data[max_level - min_level] );
    assign( k_pm, k );
    linalg::invert_entries( k_pm );

    using PressureMass = fe::wedge::operators::shell::KMass< ScalarType >;
    PressureMass pmass(
        domains[pressure_level], coords_shell[pressure_level], coords_radii[pressure_level], k_pm.grid_data(), false );
    pmass.set_lumped_diagonal( true );

    VectorQ1Scalar< ScalarType > lumped_diagonal_pmass(
        "lumped_diagonal_pmass", domains[pressure_level], mask_data[pressure_level] );
    {
        VectorQ1Scalar< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( pressure_level ),
            domains[pressure_level],
            mask_data[pressure_level] );
        linalg::assign( tmp, 1.0 );
        linalg::apply( pmass, tmp, lumped_diagonal_pmass );
    }

    using PrecSchur = linalg::solvers::DiagonalSolver< PressureMass >;
    PrecSchur inv_lumped_pmass( lumped_diagonal_pmass );

    using PrecStokes =
        linalg::solvers::BlockTriangularPreconditioner2x2<
            Stokes, Viscous, PressureMass, Gradient, PrecVisc, PrecSchur >;

    VectorQ1IsoQ2Q1< ScalarType > triangular_prec_tmp(
        "triangular_prec_tmp",
        domains[velocity_level],
        domains[pressure_level],
        mask_data[velocity_level],
        mask_data[pressure_level] );

    PrecStokes prec_stokes( K_op.block_11(), pmass, K_op.block_12(), triangular_prec_tmp, prec_11, inv_lumped_pmass );

    constexpr int fgmres_restart    = 30;
    constexpr int fgmres_max_iters  = 200;

    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_fgmres;
    for ( int i = 0; i < 2 * fgmres_restart + 4; ++i )
    {
        tmp_fgmres.emplace_back(
            "tmp_" + std::to_string( i ),
            domains[velocity_level],
            domains[pressure_level],
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    linalg::solvers::FGMRESOptions< ScalarType > fgmres_options;
    fgmres_options.restart                     = fgmres_restart;
    fgmres_options.max_iterations              = fgmres_max_iters;
    fgmres_options.relative_residual_tolerance = 1e-14;
    fgmres_options.absolute_residual_tolerance = 1e-16;

    auto solver_table = std::make_shared< util::Table >();
    linalg::solvers::FGMRES< Stokes, PrecStokes > fgmres( tmp_fgmres, fgmres_options, solver_table, prec_stokes );

    util::logroot << "Solve ...\n";
    assign( u, 0 );
    linalg::solvers::solve( fgmres, K_op, u, f );

    solver_table->query_rows_equals( "tag", "fgmres_solver" )
        .select_columns( { "absolute_residual", "relative_residual", "iteration" } )
        .print_pretty();

    // Subtract mean pressure (pressure is only determined up to a constant)
    const double avg_pressure_solution =
        kernels::common::masked_sum(
            solution.block_2().grid_data(), solution.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;

    const double avg_pressure_approximation =
        kernels::common::masked_sum(
            u.block_2().grid_data(), u.block_2().mask_data(), grid::NodeOwnershipFlag::OWNED ) /
        num_dofs_pressure;

    linalg::lincomb( solution.block_2(), { 1.0 }, { solution.block_2() }, -avg_pressure_solution );
    linalg::lincomb( u.block_2(), { 1.0 }, { u.block_2() }, -avg_pressure_approximation );

    // Compute residual
    linalg::apply( K_op, u, stok_vecs["tmp_6"] );
    linalg::lincomb( stok_vecs["tmp_5"], { 1.0, -1.0 }, { f, stok_vecs["tmp_6"] } );
    const auto inf_residual_vel = linalg::norm_inf( stok_vecs["tmp_5"].block_1() );
    const auto inf_residual_pre = linalg::norm_inf( stok_vecs["tmp_5"].block_2() );

    // Compute error
    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution } );
    const auto l2_error_velocity =
        std::sqrt( dot( error.block_1(), error.block_1() ) / static_cast< double >( num_dofs_velocity ) );
    const auto l2_error_pressure =
        std::sqrt( dot( error.block_2(), error.block_2() ) / static_cast< double >( num_dofs_pressure ) );

    table->add_row(
        { { "level", max_level },
          { "level_subdomains", level_subdomains },
          { "dofs_vel", num_dofs_velocity },
          { "l2_error_vel", l2_error_velocity },
          { "dofs_pre", num_dofs_pressure },
          { "l2_error_pre", l2_error_pressure },
          { "inf_res_vel", inf_residual_vel },
          { "inf_res_pre", inf_residual_pre },
          { "h_vel", ( r_max - r_min ) / std::pow( 2, max_level ) },
          { "h_p", ( r_max - r_min ) / std::pow( 2, max_level - 1 ) } } );

    // Write XDMF output for visualization
    {
        util::logroot << "Writing XDMF output (velocity level) ...\n";

        io::XDMFOutput xdmf_vel(
            "xdmf_velocity_level_" + std::to_string( max_level ),
            domains[velocity_level],
            coords_shell[velocity_level],
            coords_radii[velocity_level] );

        xdmf_vel.add( u.block_1().grid_data() );
        xdmf_vel.add( solution.block_1().grid_data() );
        xdmf_vel.add( error.block_1().grid_data() );

        xdmf_vel.write( 0 );

        util::logroot << "Writing XDMF output (pressure level) ...\n";

        io::XDMFOutput xdmf_pre(
            "xdmf_pressure_level_" + std::to_string( max_level ),
            domains[pressure_level],
            coords_shell[pressure_level],
            coords_radii[pressure_level] );

        xdmf_pre.add( u.block_2().grid_data() );
        xdmf_pre.add( solution.block_2().grid_data() );
        xdmf_pre.add( error.block_2().grid_data() );

        xdmf_pre.write( 0 );

        util::logroot << "XDMF output written.\n";
    }

    return {
        l2_error_velocity,
        l2_error_pressure,
        static_cast< int >( solver_table->query_rows_equals( "tag", "fgmres_solver" ).rows().size() )
    };
}

// =============================================================================
// main
// =============================================================================

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    const int max_level = 6;
    auto      table     = std::make_shared< util::Table >();

    std::map< int, std::map< int, double > > err_vel;
    std::map< int, std::map< int, double > > err_pre;

    for ( int minlevel = 2; minlevel <= 2; ++minlevel )
    {
        util::logroot << "=== Assess Stokes test (l=2, m=2, k=2, FreeZeroSlip) ===\n";
        util::logroot << "minlevel = " << minlevel << "\n";

        err_vel.clear();
        err_pre.clear();

        static bool   have_prev_level = false;
        static double prev_l2_vel     = 1.0;
        static double prev_l2_pre     = 1.0;

        for ( int level = minlevel + 1; level <= max_level; ++level )
        {
            for ( int level_subdomains = 0; level_subdomains <= 0; ++level_subdomains )
            {
                util::logroot << "  level=" << level << " level_subdomains=" << level_subdomains << "\n";

                Kokkos::Timer timer;
                timer.reset();

                const auto [l2_error_vel, l2_error_pre, iterations] =
                    test( minlevel, level, level_subdomains, table );

                const auto time_total = timer.seconds();

                util::logroot << "  errors: vel=" << l2_error_vel
                              << " pre=" << l2_error_pre
                              << " iters=" << iterations
                              << " time=" << time_total << "\n";

                err_vel[level][level_subdomains] = l2_error_vel;
                err_pre[level][level_subdomains] = l2_error_pre;

                // Sanity check: same global level should give same error regardless of subdomains
                if ( level_subdomains > 0 )
                {
                    const double dv = std::abs( err_vel[level][level_subdomains] - err_vel[level][level_subdomains - 1] );
                    const double dp = std::abs( err_pre[level][level_subdomains] - err_pre[level][level_subdomains - 1] );

                    if ( dv > 1e-3 || dp > 1e-3 )
                    {
                        util::logroot
                            << "ERROR: Error invariance w.r.t. subdomain refinement violated.\n"
                            << "  level=" << level
                            << " vel_diff=" << dv
                            << " pre_diff=" << dp << "\n";
                        Kokkos::abort( "Error invariance w.r.t. subdomain refinement violated." );
                    }
                }
            }

            // Convergence orders
            const double curr_l2_vel = err_vel[level][0];
            const double curr_l2_pre = err_pre[level][0];

            if ( have_prev_level )
            {
                const double order_vel = prev_l2_vel / curr_l2_vel;
                const double order_pre = prev_l2_pre / curr_l2_pre;

                util::logroot << "Level " << level
                              << ": order_vel=" << order_vel
                              << " order_pre=" << order_pre
                              << " (using level_subdomains=0)\n";

                table->add_row(
                    { { "level", level }, { "level_subdomains", 0 }, { "order_vel", order_vel }, { "order_pre", order_pre } } );
            }

            prev_l2_vel = curr_l2_vel;
            prev_l2_pre = curr_l2_pre;
            have_prev_level = true;
        }

        table->query_rows_not_none( "dofs_vel" )
            .select_columns(
                { "level", "level_subdomains", "dofs_pre", "dofs_vel", "l2_error_pre", "l2_error_vel", "h_vel", "h_p" } )
            .print_pretty();

        table->query_rows_not_none( "order_vel" )
            .select_columns( { "level", "level_subdomains", "order_pre", "order_vel" } )
            .print_pretty();
    }

    return 0;
}
