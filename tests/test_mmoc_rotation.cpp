// Test: MMOC (semi-Lagrangian) transport under solid-body rotation on the spherical shell.
//
// Velocity u = (-y, x, 0), i.e. rigid rotation about the z axis with omega = 1. Two checks:
//
//   1. Exactness on a rotation-invariant linear field. For T = a + b*z the exact solution is stationary, and
//      because u_z vanishes identically every Runge-Kutta stage has a zero z-component, so the traced foot
//      point keeps z exactly. The field is linear, hence reproduced exactly by the Q1 interpolant. The scheme
//      must therefore preserve it to round-off -- any drift exposes a bug in the tracing, the point location,
//      the ghost layer or the interpolation.
//
//   2. Accuracy on a cone. Same setup as test_supg_rotation.cpp / test_finite_volume_rotation.cpp: a cone of
//      radius 0.2 centred at (0.75, 0, 0), advected through one full revolution, after which the exact
//      solution is the initial condition again. Reports the L2 error and the worst over/undershoot so the
//      numbers can be put next to the SUPG, entropy-viscosity and FCT results.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <mpi.h>

#include "fe/wedge/operators/shell/mmoc_transport.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using namespace terra;
using ScalarType = double;

// Subdomain refinement on the *same* mesh: lateral 1 gives 2x2 subdomains per diamond, radial 1 splits each
// into two shells. Both are decomposition, not resolution -- the answer must not depend on either.
int g_lateral_subdomain_level = 0;
int g_radial_subdomain_level  = 0;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

using fe::wedge::operators::shell::MMOCTransport;
using fe::wedge::operators::shell::TimeSteppingScheme;

namespace
{

int g_failures = 0;

void check( const bool ok, const std::string& what )
{
    if ( !ok )
    {
        ++g_failures;
        std::cout << "  FAIL: " << what << std::endl;
    }
}

struct RotationVelocity
{
    Grid3DDataVec< ScalarType, 3 > coords_;
    Grid2DDataScalar< ScalarType > radii_;
    grid::Grid4DDataVec< ScalarType, 3 > data_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int x, const int y, const int r ) const
    {
        const auto c        = grid::shell::coords( sd, x, y, r, coords_, radii_ );
        data_( sd, x, y, r, 0 ) = -c( 1 );
        data_( sd, x, y, r, 1 ) = c( 0 );
        data_( sd, x, y, r, 2 ) = 0.0;
    }
};

ScalarType l2_relative_error(
    const Grid4DDataScalar< ScalarType >&              a,
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
        KOKKOS_LAMBDA( int sd, int x, int y, int r, ScalarType& n, ScalarType& d ) {
            if ( util::has_flag( mask( sd, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
            {
                const ScalarType diff = a( sd, x, y, r ) - b( sd, x, y, r );
                n += diff * diff;
                d += b( sd, x, y, r ) * b( sd, x, y, r );
            }
        },
        num,
        den );
    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &num, 1, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( MPI_IN_PLACE, &den, 1, MPI_DOUBLE, MPI_SUM, comm );
    return std::sqrt( num ) / std::sqrt( den );
}

/// Max |a - b| restricted to radial shells in [r_lo, r_hi).
ScalarType max_abs_difference_shells(
    const Grid4DDataScalar< ScalarType >&              a,
    const Grid4DDataScalar< ScalarType >&              b,
    const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask,
    const int                                          r_lo,
    const int                                          r_hi,
    MPI_Comm                                           comm )
{
    ScalarType m = 0;
    Kokkos::parallel_reduce(
        "maxdiff_shells",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, r_lo },
            { static_cast< int >( a.extent( 0 ) ), static_cast< int >( a.extent( 1 ) ),
              static_cast< int >( a.extent( 2 ) ), r_hi } ),
        KOKKOS_LAMBDA( int sd, int x, int y, int r, ScalarType& acc ) {
            if ( util::has_flag( mask( sd, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                acc = Kokkos::max( acc, Kokkos::abs( a( sd, x, y, r ) - b( sd, x, y, r ) ) );
        },
        Kokkos::Max< ScalarType >( m ) );
    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, comm );
    return m;
}

ScalarType max_abs_difference(
    const Grid4DDataScalar< ScalarType >&              a,
    const Grid4DDataScalar< ScalarType >&              b,
    const Grid4DDataScalar< grid::NodeOwnershipFlag >& mask,
    MPI_Comm                                           comm )
{
    ScalarType m = 0;
    Kokkos::parallel_reduce(
        "maxdiff",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
            { 0, 0, 0, 0 },
            { static_cast< int >( a.extent( 0 ) ), static_cast< int >( a.extent( 1 ) ),
              static_cast< int >( a.extent( 2 ) ), static_cast< int >( a.extent( 3 ) ) } ),
        KOKKOS_LAMBDA( int sd, int x, int y, int r, ScalarType& acc ) {
            if ( util::has_flag( mask( sd, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
                acc = Kokkos::max( acc, Kokkos::abs( a( sd, x, y, r ) - b( sd, x, y, r ) ) );
        },
        Kokkos::Max< ScalarType >( m ) );
    Kokkos::fence();
    MPI_Allreduce( MPI_IN_PLACE, &m, 1, MPI_DOUBLE, MPI_MAX, comm );
    return m;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------
// 1. Rotation-invariant linear field must be preserved to round-off.
// ---------------------------------------------------------------------------------------------------------
void test_invariant_linear( const int level, const int num_steps, const ScalarType cfl,
                            const int substeps )
{
    const auto domain = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );

    auto       mask   = grid::setup_node_ownership_mask_data( domain );
    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto radii  = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Vec< ScalarType, 3 > u( "u", domain, mask );
    VectorQ1Scalar< ScalarType > T( "T", domain, mask );
    VectorQ1Scalar< ScalarType > T_ref( "T_ref", domain, mask );

    Kokkos::parallel_for(
        "u_init",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        RotationVelocity{ coords, radii, u.grid_data() } );
    Kokkos::fence();

    const auto T_data = T.grid_data();
    Kokkos::parallel_for(
        "T_init",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
            const auto c              = grid::shell::coords( sd, x, y, r, coords, radii );
            T_data( sd, x, y, r ) = 0.3 + 1.7 * c( 2 ); // a + b*z, invariant under rotation about z
        } );
    Kokkos::fence();
    Kokkos::deep_copy( T_ref.grid_data(), T.grid_data() );

    MMOCTransport< ScalarType > transport( domain, mask, TimeSteppingScheme::RK4 );

    const ScalarType h   = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType dt  = cfl * h; // |u| <= 1 on the unit-radius shell
    const int        sub = substeps;

    // Re-impose the Dirichlet values on the CMB and the surface after every step, exactly as the energy solver
    // does. This matters here: the mesh is the polyhedron inscribed in the sphere, whereas the analytic
    // rotation velocity is tangential to the *sphere*, so a characteristic starting on the outermost shell
    // leaves the mesh and is clamped back onto it, by a relative O(h^2). Without the Dirichlet reset that
    // boundary artifact -- which has nothing to do with the transport scheme -- diffuses into the interior over
    // many steps. A discrete velocity with u.n = 0 at the boundary nodes has no such artifact, because the
    // outer element face is flat and a Q1-interpolated velocity that is tangential at its three nodes is
    // tangential on the whole face.
    const int  n_rad_nodes = domain.domain_info().subdomain_num_nodes_radially();
    const auto T_grid      = T.grid_data();
    const auto T_ref_grid  = T_ref.grid_data();
    auto       reset_dirichlet_shells = [&]() {
        Kokkos::parallel_for(
            "reset_dirichlet",
            grid::shell::local_domain_md_range_policy_nodes( domain ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                if ( r == 0 || r == n_rad_nodes - 1 )
                    T_grid( sd, x, y, r ) = T_ref_grid( sd, x, y, r );
            } );
        Kokkos::fence();
    };

    long long escapes_total = 0;
    for ( int step = 0; step < num_steps; ++step )
    {
        transport.step( T, u, u, dt, sub );
        escapes_total += transport.last_escapes();
        reset_dirichlet_shells();
    }

    const auto err_l2  = l2_relative_error( T.grid_data(), T_ref.grid_data(), mask, domain.comm() );
    const auto err_max = max_abs_difference( T.grid_data(), T_ref.grid_data(), mask, domain.comm() );

    util::logroot << std::scientific << std::setprecision( 4 );
    util::logroot << "  [invariant linear] level=" << level << " steps=" << num_steps << " cfl=" << cfl
                  << " substeps=" << sub << std::endl;
    util::logroot << "    L2 rel error : " << err_l2 << std::endl;
    util::logroot << "    max abs error: " << err_max << std::endl;
    util::logroot << "    escapes      : " << escapes_total << std::endl;

    const auto err_inner = max_abs_difference_shells( T.grid_data(), T_ref.grid_data(), mask, 1,
                                                      n_rad_nodes - 1, domain.comm() );
    util::logroot << "    max abs error (interior shells only): " << err_inner << std::endl;

    check( err_inner < 1e-12, "rotation-invariant linear field is not preserved on the interior shells" );
    check( escapes_total == 0, "departure points escaped the ghost layer" );
}

// ---------------------------------------------------------------------------------------------------------
// 2. Cone through one full revolution.
// ---------------------------------------------------------------------------------------------------------
void test_cone_revolution( const int level, const ScalarType cfl )
{
    const auto domain =
        DistributedDomain::create_uniform( level, level, 0.5, 1.0, g_lateral_subdomain_level,
                                          g_radial_subdomain_level );

    auto       mask   = grid::setup_node_ownership_mask_data( domain );
    const auto coords = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto radii  = grid::shell::subdomain_shell_radii< ScalarType >( domain );

    VectorQ1Vec< ScalarType, 3 > u( "u", domain, mask );
    VectorQ1Scalar< ScalarType > T( "T", domain, mask );
    VectorQ1Scalar< ScalarType > T_ref( "T_ref", domain, mask );

    Kokkos::parallel_for(
        "u_init",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        RotationVelocity{ coords, radii, u.grid_data() } );
    Kokkos::fence();

    const auto T_data = T.grid_data();
    Kokkos::parallel_for(
        "T_init",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
            const auto                        c = grid::shell::coords( sd, x, y, r, coords, radii );
            const dense::Vec< ScalarType, 3 > center{ 0.75, 0.0, 0.0 };
            constexpr ScalarType              radius = 0.2;
            const ScalarType                  dist   = ( c - center ).norm();
            T_data( sd, x, y, r ) = dist < radius ? ( 1 - dist / radius ) * ( 1 - dist / radius ) : 0.0;
        } );
    Kokkos::fence();
    Kokkos::deep_copy( T_ref.grid_data(), T.grid_data() );

    MMOCTransport< ScalarType > transport( domain, mask, TimeSteppingScheme::RK4 );

    const ScalarType h           = grid::shell::min_radial_h( domain.domain_info().radii() );
    const ScalarType t_end       = 2.0 * M_PI;
    const int        num_steps   = static_cast< int >( std::ceil( t_end / ( cfl * h ) ) );
    const ScalarType dt          = t_end / num_steps;
    const int        sub         = MMOCTransport< ScalarType >::substeps_for_accuracy( cfl );

    util::logroot << "  [cone] level=" << level << " cfl=" << cfl << " steps=" << num_steps
                  << " dt=" << dt << " substeps=" << sub << std::endl;

    // One frame every vtk_interval steps, so the cone can be watched going round.
    constexpr int  vtk_interval = 25;
    io::XDMFOutput xdmf( "./output/test_mmoc_rotation_out", domain, coords, radii );
    xdmf.add( T.grid_data() );
    xdmf.write( 0 );

    long long escapes_total = 0;
    for ( int step = 0; step < num_steps; ++step )
    {
        transport.step( T, u, u, dt, sub );
        escapes_total += transport.last_escapes();

        if ( ( step + 1 ) % vtk_interval == 0 || step + 1 == num_steps )
            xdmf.write( step + 1 );
    }

    const auto err_l2 = l2_relative_error( T.grid_data(), T_ref.grid_data(), mask, domain.comm() );

    ScalarType t_min = 0, t_max = 0;
    {
        const auto d = T.grid_data();
        Kokkos::parallel_reduce(
            "minmax",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                { static_cast< int >( d.extent( 0 ) ), static_cast< int >( d.extent( 1 ) ),
                  static_cast< int >( d.extent( 2 ) ), static_cast< int >( d.extent( 3 ) ) } ),
            KOKKOS_LAMBDA( int sd, int x, int y, int r, ScalarType& lo, ScalarType& hi ) {
                lo = Kokkos::min( lo, d( sd, x, y, r ) );
                hi = Kokkos::max( hi, d( sd, x, y, r ) );
            },
            Kokkos::Min< ScalarType >( t_min ),
            Kokkos::Max< ScalarType >( t_max ) );
        Kokkos::fence();
        MPI_Allreduce( MPI_IN_PLACE, &t_min, 1, MPI_DOUBLE, MPI_MIN, domain.comm() );
        MPI_Allreduce( MPI_IN_PLACE, &t_max, 1, MPI_DOUBLE, MPI_MAX, domain.comm() );
    }

    util::logroot << "    L2 rel error after one revolution: " << err_l2 << std::endl;
    util::logroot << "    T range (exact [0,1])            : [" << t_min << ", " << t_max << "]" << std::endl;
    util::logroot << "    escapes                          : " << escapes_total << std::endl;

    check( err_l2 < 1.0, "cone was destroyed" );
    check( t_min > -1e-10, "undershoot below the global limiter bound" );
    check( t_max < 1.0 + 1e-10, "overshoot above the global limiter bound" );
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    int level = 7;
    if ( argc > 1 )
        level = std::atoi( argv[1] );

    // The Courant number is bounded by the ghost layer width, not by stability; 0.85 is just below the limit
    // for the width-1 layer. The substep count is varied independently to cover the multi-substep tracing.
    // test_invariant_linear( level, 20, 0.5, 1 );
    // test_invariant_linear( level, 20, 0.85, 1 );
    // test_invariant_linear( level, 20, 0.85, 4 );

    // The cone spans only a couple of cells below level 5, where any scheme destroys it; the accuracy check is
    // only meaningful on the finer grids.
    if ( level >= 5 )
    {
        if ( argc > 3 )
            g_lateral_subdomain_level = std::atoi( argv[3] );
        if ( argc > 4 )
            g_radial_subdomain_level = std::atoi( argv[4] );
        test_cone_revolution( level, ( argc > 2 ) ? std::atof( argv[2] ) : 0.5 );
        // Same timestep as test_supg_rotation.cpp / test_finite_volume_rotation.cpp (dt = 0.5 * 0.1 * h), so
        // the errors can be compared directly.
        // test_cone_revolution( level, 0.05 );
    }

    int failures = g_failures;
    MPI_Allreduce( MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    int rank = 0;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if ( rank == 0 )
        std::cout << "\ntest_mmoc_rotation: " << ( failures == 0 ? "PASSED" : "FAILED" ) << std::endl;

    return failures == 0 ? 0 : 1;
}
