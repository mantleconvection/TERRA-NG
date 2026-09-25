// Test: solid-body rotation *with diffusion*, comparing the entropy-viscosity scheme against MMOC
// semi-Lagrangian transport plus a backward-Euler diffusion solve, on a manufactured solution.
//
// The rotating-cone test cannot be extended to diffusion directly: (1 - d/R)^2 has no closed-form heat
// kernel, so there would be nothing exact to measure against. A Gaussian does. Under
//
//     dT/dt + u . grad T = kappa laplacian T,      u = (-y, x, 0),
//
// a rigid rotation carries the bump around the circle of radius r0 in the z = 0 plane while pure diffusion
// spreads it, and the two commute because the rotation is an isometry. So with
//
//     T(x, 0) = exp( -|x - c0|^2 / (2 sigma0^2) ),      c0 = (r0, 0, 0),
//
// the exact solution in free space is the spreading Gaussian carried along the orbit,
//
//     T(x, t) = (sigma0^2 / s(t))^{3/2} exp( -|x - c(t)|^2 / (2 s(t)) ),
//     s(t)    = sigma0^2 + 2 kappa t,        c(t) = (r0 cos t, r0 sin t, 0).
//
// The domain is a shell, not free space, so this is only the exact solution while the bump stays clear of
// both boundaries. sigma0 and kappa are chosen so that at t = 2 pi the shells sit at more than three final
// standard deviations, where the analytic solution is below 1e-3 of the peak and homogeneous Dirichlet
// conditions are consistent with it to that order.
//
// Both schemes take the same manufactured solution, the same mesh, and the same FGMRES tolerance for their
// linear solves, so the comparison isolates the transport treatment. Each is run at its own natural
// timestep, and the EV scheme additionally at the MMOC timestep, since the two disagree by an order of
// magnitude about what a sensible step is.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/operators/shell/entropy_viscosity.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/mmoc_transport.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg_kerngen.hpp"
#include "fe/wedge/operators/shell/wedge_constant_div_k_grad.hpp"
#include "linalg/solvers/fgmres.hpp"
#include "linalg/solvers/solver.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::Grid5DDataScalar;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using ScalarType = double;

namespace
{
constexpr ScalarType r0     = 0.75;   // orbit radius, mid-shell
constexpr ScalarType sigma0 = 0.05;   // initial standard deviation
constexpr ScalarType kappa  = 2e-4;   // diffusivity; see the header note on the boundary clearance
constexpr ScalarType t_end  = 2.0 * M_PI;

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        util::logroot << "  FAIL: " << what << std::endl;
    }
}
} // namespace

// Solid-body rotation velocity: u = (-y, x, 0).
struct VelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const auto c    = grid::shell::coords( id, x, y, r, grid_, radii_ );
        data_( id, x, y, r, 0 ) = -c( 1 );
        data_( id, x, y, r, 1 ) = c( 0 );
        data_( id, x, y, r, 2 ) = ScalarType( 0 );
    }
};

/// Fills a field with the exact solution at time `t`.
struct ExactInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > data_;
    ScalarType                     t_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        const auto       c  = grid::shell::coords( id, x, y, r, grid_, radii_ );
        const ScalarType s  = sigma0 * sigma0 + ScalarType( 2 ) * kappa * t_;
        const ScalarType cx = r0 * Kokkos::cos( t_ );
        const ScalarType cy = r0 * Kokkos::sin( t_ );

        const ScalarType dx = c( 0 ) - cx;
        const ScalarType dy = c( 1 ) - cy;
        const ScalarType dz = c( 2 );
        const ScalarType d2 = dx * dx + dy * dy + dz * dz;

        const ScalarType amp = Kokkos::pow( sigma0 * sigma0 / s, ScalarType( 1.5 ) );
        data_( id, x, y, r ) = amp * Kokkos::exp( -d2 / ( ScalarType( 2 ) * s ) );
    }
};

ScalarType l2_relative_error( const Grid4DDataScalar< ScalarType >&              a,
                              const Grid4DDataScalar< ScalarType >&              b,
                              const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask,
                              MPI_Comm                                           comm )
{
    ScalarType num = 0, den = 0;
    Kokkos::parallel_reduce(
        "l2",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { static_cast< int >( a.extent( 0 ) ), static_cast< int >( a.extent( 1 ) ),
              static_cast< int >( a.extent( 2 ) ), static_cast< int >( a.extent( 3 ) ) } ),
        KOKKOS_LAMBDA( int s, int x, int y, int r, ScalarType& n, ScalarType& d ) {
            if ( !util::has_flag( mask( s, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                return;
            const ScalarType e = a( s, x, y, r ) - b( s, x, y, r );
            n += e * e;
            d += b( s, x, y, r ) * b( s, x, y, r );
        },
        num, den );
    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &num, 1, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( MPI_IN_PLACE, &den, 1, MPI_DOUBLE, MPI_SUM, comm );
    return ( den > 0 ) ? Kokkos::sqrt( num / den ) : ScalarType( 0 );
}

/// Zeroes the two boundary shells, which is what the manufactured solution is there to within 1e-3.
template < typename BoundaryMaskType >
void apply_homogeneous_dirichlet( VectorQ1Scalar< ScalarType >& T, const BoundaryMaskType& boundary_mask )
{
    auto d = T.grid_data();
    Kokkos::parallel_for(
        "dirichlet",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { static_cast< int >( d.extent( 0 ) ), static_cast< int >( d.extent( 1 ) ),
              static_cast< int >( d.extent( 2 ) ), static_cast< int >( d.extent( 3 ) ) } ),
        KOKKOS_LAMBDA( int s, int x, int y, int r ) {
            if ( util::has_flag( boundary_mask( s, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY ) )
                d( s, x, y, r ) = ScalarType( 0 );
        } );
    Kokkos::fence();
}

/// Outcome of one run, so the two schemes can be tabulated side by side.
struct Result
{
    ScalarType l2      = 0;
    ScalarType peak    = 0;
    int        steps   = 0;
    double     seconds = 0;
    long long  iters   = 0;   ///< FGMRES iterations summed over all timesteps
};

/// The linear solves are unpreconditioned FGMRES at a deliberately light tolerance: the point of the test is
/// the transport error, and driving the algebra to round-off would only bury it under solver cost. Both
/// schemes use exactly these options.
linalg::solvers::FGMRESOptions< ScalarType > solver_options()
{
    return linalg::solvers::FGMRESOptions< ScalarType >{
        .restart                     = 30,
        .relative_residual_tolerance = 1e-6,
        .absolute_residual_tolerance = 1e-10,
        .max_iterations              = 200,
    };
}

// ==============================================================================================================
//  Entropy viscosity
// ==============================================================================================================
Result run_ev( const int level, const ScalarType dt_factor, const std::string& tag )
{
    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto       mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto       boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );
    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Vec< ScalarType >    u( "u", domain, mask_data );
    VectorQ1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorQ1Scalar< ScalarType > T_prev( "T_prev", domain, mask_data );
    VectorQ1Scalar< ScalarType > T_exact( "T_exact", domain, mask_data );
    VectorQ1Scalar< ScalarType > f( "f", domain, mask_data );
    VectorQ1Scalar< ScalarType > rhs_ev( "rhs_ev", domain, mask_data );
    VectorQ1Scalar< ScalarType > lap_T( "lap_T", domain, mask_data );
    VectorQ1Scalar< ScalarType > M_lumped( "M_lumped", domain, mask_data );

    const auto num_sub = static_cast< long long >( domain.subdomains().size() );
    const auto nx_c    = domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1;
    const auto nr_c    = domain.domain_info().subdomain_num_nodes_radially() - 1;
    Grid5DDataScalar< ScalarType > nu_h_wedge(
        "nu_h_wedge", num_sub, nx_c, nx_c, nr_c, fe::wedge::num_wedges_per_hex_cell );
    Grid5DDataScalar< ScalarType > kappa_wedge(
        "kappa_wedge", num_sub, nx_c, nx_c, nr_c, fe::wedge::num_wedges_per_hex_cell );
    kernels::common::set_constant( kappa_wedge, kappa );

    Kokkos::parallel_for( "velocity_init", local_domain_md_range_policy_nodes( domain ),
                          VelocityInterpolator{ coords_shell, coords_radii, u.grid_data() } );
    Kokkos::parallel_for( "T_init", local_domain_md_range_policy_nodes( domain ),
                          ExactInterpolator{ coords_shell, coords_radii, T.grid_data(), ScalarType( 0 ) } );
    Kokkos::fence();
    Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() );

    const ScalarType h           = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt          = dt_factor * h;
    const int        n_timesteps = static_cast< int >( std::ceil( t_end / dt ) );

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;
    AD A( domain, coords_shell, coords_radii, boundary_mask_data, u, kappa, dt, /*treat_boundary=*/true );
    A.set_supg_enabled( false );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;
    Mass M( domain, coords_shell, coords_radii, false );

    using EVDiffOp = fe::wedge::operators::shell::WedgeConstantDivKGrad< ScalarType >;
    EVDiffOp A_kappa( domain, coords_shell, coords_radii, kappa_wedge );
    EVDiffOp A_evdiff( domain, coords_shell, coords_radii, nu_h_wedge );

    {
        VectorQ1Scalar< ScalarType > ones( "ones", domain, mask_data );
        linalg::assign( ones, ScalarType( 1 ) );
        linalg::assign( M_lumped, ScalarType( 0 ) );
        linalg::apply( M, ones, M_lumped );
    }

    const auto opts = solver_options();
    std::vector< VectorQ1Scalar< ScalarType > > tmps;
    tmps.reserve( 2 * opts.restart + 4 );
    for ( int i = 0; i < 2 * opts.restart + 4; ++i )
        tmps.emplace_back( "fgmres_tmp", domain, mask_data );
    linalg::solvers::FGMRES< AD > fgmres( tmps, opts );
    fgmres.set_tag( "ev_fgmres" );
    auto table = std::make_shared< util::Table >();

    const fe::wedge::operators::shell::EntropyViscosityParameters< ScalarType > ev_params{};

    util::logroot << "  [EV " << tag << "] level=" << level << " dt=" << dt << " steps=" << n_timesteps
                  << std::endl;

    io::XDMFOutput xdmf( "./output/gauss_ev_" + tag, domain, coords_shell, coords_radii );
    xdmf.add( T.grid_data() );
    xdmf.add( T_exact.grid_data() );
    xdmf.write( 0 );

    long long    total_iters = 0;
    const double t0          = MPI_Wtime();
    for ( int ts = 1; ts <= n_timesteps; ++ts )
    {
        linalg::apply( A_kappa, T, lap_T );
        {
            auto lap_v = lap_T.grid_data();
            auto m_v   = M_lumped.grid_data();
            auto bm    = boundary_mask_data;
            Kokkos::parallel_for(
                "lap_T_lumped",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                    { 0, 0, 0, 0 },
                    { lap_v.extent( 0 ), lap_v.extent( 1 ), lap_v.extent( 2 ), lap_v.extent( 3 ) } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                    if ( util::has_flag( bm( s, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY ) )
                    {
                        lap_v( s, x, y, r ) = ScalarType( 0 );
                        return;
                    }
                    const ScalarType m  = m_v( s, x, y, r );
                    lap_v( s, x, y, r ) = ( m > ScalarType( 0 ) ) ? ( lap_v( s, x, y, r ) / m ) : ScalarType( 0 );
                } );
            Kokkos::fence();
        }

        const auto stats = fe::wedge::operators::shell::compute_entropy_stats(
            T, mask_data, domain, coords_shell, coords_radii, ev_params );
        fe::wedge::operators::shell::compute_nu_h( nu_h_wedge, T, T_prev, u, lap_T.grid_data(), domain,
                                                   coords_shell, coords_radii, dt, stats, ev_params );

        linalg::apply( A_evdiff, T, rhs_ev );
        linalg::apply( M, T, f );
        linalg::lincomb( f, { ScalarType( 1 ), -dt }, { f, rhs_ev } );

        Kokkos::deep_copy( T_prev.grid_data(), T.grid_data() );

        fe::strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like(
            f, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );
        linalg::solvers::solve( fgmres, A, T, f );
        total_iters += fgmres.last_iterations();
        table->clear();
    }
    Kokkos::fence();
    const double t1 = MPI_Wtime();

    Kokkos::parallel_for(
        "T_exact_end", local_domain_md_range_policy_nodes( domain ),
        ExactInterpolator{ coords_shell, coords_radii, T_exact.grid_data(), ScalarType( n_timesteps ) * dt } );
    Kokkos::fence();
    xdmf.write( n_timesteps );

    Result res;
    res.l2      = l2_relative_error( T.grid_data(), T_exact.grid_data(), mask_data, domain.comm() );
    res.peak    = kernels::common::max_entry( T.grid_data() );
    res.steps   = n_timesteps;
    res.seconds = t1 - t0;
    res.iters   = total_iters;
    return res;
}

// ==============================================================================================================
//  MMOC transport plus a backward-Euler diffusion solve
// ==============================================================================================================
//
// Lie splitting: advect over the whole step, then diffuse over the whole step. The diffusion operator is the
// same discrete operator the EV scheme solves, evaluated at zero velocity, so it reduces to M + dt * K and the
// two schemes differ only in how they treat advection.
Result run_mmoc( const int level, const ScalarType dt_factor, const int width, const std::string& tag )
{
    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto       mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto       boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );
    const auto coords_shell = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Vec< ScalarType >    u( "u", domain, mask_data );
    VectorQ1Vec< ScalarType >    u_zero( "u_zero", domain, mask_data );
    VectorQ1Scalar< ScalarType > T( "T", domain, mask_data );
    VectorQ1Scalar< ScalarType > T_exact( "T_exact", domain, mask_data );
    VectorQ1Scalar< ScalarType > f( "f", domain, mask_data );

    Kokkos::parallel_for( "velocity_init", local_domain_md_range_policy_nodes( domain ),
                          VelocityInterpolator{ coords_shell, coords_radii, u.grid_data() } );
    linalg::assign( u_zero, ScalarType( 0 ) );
    Kokkos::parallel_for( "T_init", local_domain_md_range_policy_nodes( domain ),
                          ExactInterpolator{ coords_shell, coords_radii, T.grid_data(), ScalarType( 0 ) } );
    Kokkos::fence();

    const ScalarType h           = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt          = dt_factor * h;
    const int        n_timesteps = static_cast< int >( std::ceil( t_end / dt ) );
    const int        substeps    = fe::wedge::operators::shell::MMOCTransport< ScalarType >::substeps_for_accuracy(
        dt_factor );

    fe::wedge::operators::shell::MMOCTransport< ScalarType > transport(
        domain, mask_data, fe::wedge::operators::shell::TimeSteppingScheme::RK4, width );

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPGKerngen< ScalarType >;
    AD A_diff( domain, coords_shell, coords_radii, boundary_mask_data, u_zero, kappa, dt,
               /*treat_boundary=*/true );
    A_diff.set_supg_enabled( false );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;
    Mass M( domain, coords_shell, coords_radii, false );

    const auto opts = solver_options();
    std::vector< VectorQ1Scalar< ScalarType > > tmps;
    tmps.reserve( 2 * opts.restart + 4 );
    for ( int i = 0; i < 2 * opts.restart + 4; ++i )
        tmps.emplace_back( "fgmres_tmp", domain, mask_data );
    linalg::solvers::FGMRES< AD > fgmres( tmps, opts );
    fgmres.set_tag( "mmoc_fgmres" );
    auto table = std::make_shared< util::Table >();

    util::logroot << "  [MMOC " << tag << "] level=" << level << " dt=" << dt << " steps=" << n_timesteps
                  << " substeps=" << substeps << std::endl;

    io::XDMFOutput xdmf( "./output/gauss_mmoc_" + tag, domain, coords_shell, coords_radii );
    xdmf.add( T.grid_data() );
    xdmf.add( T_exact.grid_data() );
    xdmf.write( 0 );

    long long    escapes     = 0;
    long long    total_iters = 0;
    const double t0          = MPI_Wtime();
    for ( int ts = 1; ts <= n_timesteps; ++ts )
    {
        transport.step( T, u, u, dt, substeps );
        escapes += transport.last_escapes();
        apply_homogeneous_dirichlet( T, boundary_mask_data );

        linalg::apply( M, T, f );
        fe::strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like(
            f, boundary_mask_data, grid::shell::ShellBoundaryFlag::BOUNDARY );
        linalg::solvers::solve( fgmres, A_diff, T, f );
        total_iters += fgmres.last_iterations();
        table->clear();
    }
    Kokkos::fence();
    const double t1 = MPI_Wtime();

    Kokkos::parallel_for(
        "T_exact_end", local_domain_md_range_policy_nodes( domain ),
        ExactInterpolator{ coords_shell, coords_radii, T_exact.grid_data(), ScalarType( n_timesteps ) * dt } );
    Kokkos::fence();
    xdmf.write( n_timesteps );

    check( escapes == 0, "MMOC foot points escaped the ghost layer" );

    Result res;
    res.l2      = l2_relative_error( T.grid_data(), T_exact.grid_data(), mask_data, domain.comm() );
    res.peak    = kernels::common::max_entry( T.grid_data() );
    res.steps   = n_timesteps;
    res.seconds = t1 - t0;
    res.iters   = total_iters;
    return res;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int level = 5;
    if ( argc > 1 )
        level = std::atoi( argv[1] );

    {
        // Peak of the exact solution at t_end, for scale: the Gaussian loses height purely by spreading.
        const ScalarType s_end = sigma0 * sigma0 + ScalarType( 2 ) * kappa * t_end;
        util::logroot << "Rotating Gaussian with diffusion, level " << level << ": kappa=" << kappa
                      << " sigma0=" << sigma0 << " sigma_end=" << std::sqrt( s_end )
                      << " exact peak at t_end=" << std::pow( sigma0 * sigma0 / s_end, ScalarType( 1.5 ) )
                      << std::endl;
    }

    // Every scheme on the same mesh, the same timestep and the same solver tolerance, so what is left is the
    // treatment of advection.
    const Result q1      = run_mmoc( level, 0.5, 2, "Q1" );
    const Result cubic   = run_mmoc( level, 0.5, 4, "cubic" );
    const Result quintic = run_mmoc( level, 0.5, 6, "quintic" );
    const Result ev      = run_ev( level, 0.5, "EV" );

    util::logroot << std::scientific << std::setprecision( 4 );
    util::logroot << "\n  scheme            steps      L2 rel err       peak       seconds   FGMRES its  its/step" << std::endl;
    util::logroot << "  MMOC  Q1        " << std::setw( 7 ) << q1.steps << "  " << q1.l2 << "  " << q1.peak
                  << "  " << q1.seconds << std::setw( 10 ) << q1.iters << std::setprecision( 2 ) << std::fixed
                  << "   " << double( q1.iters ) / q1.steps << std::scientific << std::setprecision( 4 )
                  << std::endl;
    util::logroot << "  MMOC  cubic     " << std::setw( 7 ) << cubic.steps << "  " << cubic.l2 << "  "
                  << cubic.peak << "  " << cubic.seconds << std::setw( 10 ) << cubic.iters
                  << std::setprecision( 2 ) << std::fixed << "   " << double( cubic.iters ) / cubic.steps
                  << std::scientific << std::setprecision( 4 ) << std::endl;
    util::logroot << "  MMOC  quintic   " << std::setw( 7 ) << quintic.steps << "  " << quintic.l2 << "  "
                  << quintic.peak << "  " << quintic.seconds << std::setw( 10 ) << quintic.iters
                  << std::setprecision( 2 ) << std::fixed << "   " << double( quintic.iters ) / quintic.steps
                  << std::scientific << std::setprecision( 4 ) << std::endl;
    util::logroot << "  EV              " << std::setw( 7 ) << ev.steps << "  " << ev.l2 << "  " << ev.peak
                  << "  " << ev.seconds << std::setw( 10 ) << ev.iters << std::setprecision( 2 ) << std::fixed
                  << "   " << double( ev.iters ) / ev.steps << std::scientific << std::setprecision( 4 )
                  << std::endl;

    check( std::isfinite( quintic.l2 ) && quintic.l2 < ScalarType( 1 ), "MMOC lost the bump" );
    check( quintic.l2 < cubic.l2, "the quintic reconstruction is not better than the cubic" );
    check( cubic.l2 < q1.l2, "the cubic reconstruction is not better than Q1" );
    check( std::isfinite( ev.l2 ) && ev.l2 < ScalarType( 1 ), "EV lost the bump" );

    int failures = g_failures;
    MPI_Allreduce( MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    util::logroot << "\ntest_rotating_gaussian_diffusion: " << ( failures == 0 ? "PASSED" : "FAILED" )
                  << std::endl;

    return failures == 0 ? 0 : 1;
}
