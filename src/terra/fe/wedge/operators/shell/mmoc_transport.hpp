#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "communication/shell/communication.hpp"
#include "fe/wedge/sl/ghost_exchange.hpp"
#include "fe/wedge/sl/ghosted_geometry.hpp"
#include "fe/wedge/sl/point_location.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_q1.hpp"
#include "mpi/mpi.hpp"
#include "util/timer.hpp"

/// @file
///
/// Modified method of characteristics (MMOC) transport of a Q1 nodal field on the spherical shell.
///
/// Solves the pure advection problem
/// \f[
///   \partial_t T + \mathbf{u} \cdot \nabla T = 0, \qquad t \in [t^n, t^{n+1}]
/// \f]
/// exactly along characteristics: for every node \f$ x_i \f$ the characteristic is traced **backwards** by
/// integrating \f$ \mathrm{d}X/\mathrm{d}s = -\hat{\mathbf{u}}(X, s) \f$ from \f$ X(0) = x_i \f$ over
/// \f$ s \in [0, \Delta t] \f$, and the new value is the old field evaluated at the foot point,
/// \f$ T^{n+1}_i = T^n(X(\Delta t)) \f$.
///
/// The velocity is interpolated linearly in time between the two given fields. Because the integration runs
/// backwards, pseudo-time \f$ s \f$ maps to physical time \f$ t^{n+1} - s \f$, so a stage at
/// \f$ \tau = s / \Delta t \f$ uses \f$ (1 - \tau) \mathbf{u}^{n+1} + \tau \mathbf{u}^{n} \f$.
///
/// @note With more than one substep the interpolation weight is tracked in *global* pseudo-time,
///       \f$ \tau = (m + c_k) / M \f$ for substep \f$ m \f$ of \f$ M \f$ and stage \f$ k \f$. Reusing the raw
///       stage weight \f$ c_k \f$ in every substep -- as some implementations do -- would interpolate as if each
///       substep spanned the whole timestep and silently drop the temporal order back to first order.
///
/// The scheme is unconditionally stable in the advective sense: there is no CFL restriction from the
/// characteristic tracing itself. The *implementation* is bounded by the ghost layer width, since a foot point
/// must remain inside the local subdomain plus its ghost layer. \ref MMOCTransport::recommended_substeps
/// converts a Courant number into the number of substeps that keeps every substep inside that budget.
///
/// **Interpolation.** The field at the foot point is evaluated with
/// \ref terra::fe::wedge::sl::evaluate_cubic_scalar, which reconstructs it over the structured index stencil
/// rather than with the Q1 shape functions of the containing wedge: multilinear evaluation is only second order
/// and pays that error once per timestep, which is the dominant source of numerical diffusion here. The
/// transport asks for a six-node stencil, so the reconstruction is a quintic where a centred window of that
/// width fits and steps down to the centred cubic and then to the sliding cubic/quadratic/linear ladder where
/// it does not -- see \ref terra::fe::wedge::sl::stencil_window, which also says why the reduced band around
/// each diamond seam is structural. The result is bounded by the Bermejo-Staniforth clip to the located cell's
/// own eight nodes; the per-sweep PCHIP slope limiter is deliberately not used, for the reason set out at
/// \ref terra::fe::wedge::sl::evaluate_cubic_scalar. The velocity stays on
/// \ref terra::fe::wedge::sl::evaluate_q1_vec, which reproduces a linear velocity field to round-off.
///
/// Measured on `test_mmoc_rotation`, one full revolution of the cone, against the entropy-viscosity scheme on
/// the same mesh. Level 6: L2 relative error 0.090 for this scheme against 0.504 for EV, peak 0.691 of the
/// exact 1.0 against 0.252, in 1609 timesteps against EV's 16086 and about a minute against 40 on one GH200.
/// Level 5: 0.313 against 0.745. Dropping the stencil to four nodes gives 0.291 and 0.620 on the same two
/// grids, so the sixth-order stencil is worth roughly a grid refinement.
///
/// What error remains is diffusion, not displacement. At level 6 the peak sits at 0.00 degrees of azimuth and
/// the mass-weighted mean radius at 0.7449 against 0.7449 exact, while the cone is 31% short in amplitude and
/// 57% too wide along its orbit (12.4 degrees full width at half maximum against 7.9). The broadening is
/// anisotropic -- 0.5% in radius against 57% in azimuth -- because the foot-point displacement is tangential,
/// so every step commits its interpolation error along the direction of travel and almost none across it.
///
/// @warning **Do not reduce the timestep to make this scheme more accurate -- it does the opposite.** The
///          trajectory is already essentially exact (Q1 reproduces a linear velocity to round-off, and the RK4
///          error is ~1e-13 per step and cannot accumulate, since X is reset to the node every step), so there
///          is no temporal error left to reduce and each extra step only commits another interpolation error.
///          Measured at level 5, one revolution: L2 0.620 at Courant 0.5, 0.741 at 0.25, 0.809 at 0.125, with
///          the lag growing from 3 to 5.6 degrees. Run at the largest Courant number the ghost layer allows --
///          see \ref max_courant -- and refine in space, not in time.
///
///          Recovering the order properly needs a reconstruction that is exact for polynomials in the physical
///          coordinates, e.g. a least-squares fit over the stencil in a local tangent frame with the
///          coefficients precomputed per cell.
///
///
/// Diffusion is not part of this operator; combine it with an implicit diffusion solve by operator splitting.

namespace terra::fe::wedge::operators::shell
{

/// @brief Explicit Runge-Kutta scheme used to trace the characteristics.
enum class TimeSteppingScheme
{
    /// First order.
    ExplicitEuler,
    /// Third order.
    RK3,
    /// Third order, Ralston's minimum-error coefficients.
    Ralston,
    /// Fourth order.
    RK4,
};

/// @brief Butcher tableau, sized for the schemes in \ref TimeSteppingScheme.
template < typename ScalarType >
struct ButcherTableau
{
    static constexpr int max_stages = 4;

    int        stages                       = 1;
    ScalarType A[max_stages][max_stages]    = {};
    ScalarType b[max_stages]                = {};
    ScalarType c[max_stages]                = {};
};

template < typename ScalarType >
ButcherTableau< ScalarType > butcher_tableau( const TimeSteppingScheme scheme )
{
    ButcherTableau< ScalarType > t;

    switch ( scheme )
    {
    case TimeSteppingScheme::ExplicitEuler:
        t.stages = 1;
        t.b[0]   = 1;
        t.c[0]   = 0;
        break;

    case TimeSteppingScheme::RK3:
        t.stages  = 3;
        t.A[1][0] = ScalarType( 0.5 );
        t.A[2][0] = ScalarType( -1 );
        t.A[2][1] = ScalarType( 2 );
        t.b[0]    = ScalarType( 1 ) / 6;
        t.b[1]    = ScalarType( 2 ) / 3;
        t.b[2]    = ScalarType( 1 ) / 6;
        t.c[0]    = 0;
        t.c[1]    = ScalarType( 0.5 );
        t.c[2]    = 1;
        break;

    case TimeSteppingScheme::Ralston:
        t.stages  = 3;
        t.A[1][0] = ScalarType( 0.5 );
        t.A[2][1] = ScalarType( 0.75 );
        t.b[0]    = ScalarType( 2 ) / 9;
        t.b[1]    = ScalarType( 1 ) / 3;
        t.b[2]    = ScalarType( 4 ) / 9;
        t.c[0]    = 0;
        t.c[1]    = ScalarType( 0.5 );
        t.c[2]    = ScalarType( 0.75 );
        break;

    case TimeSteppingScheme::RK4:
        t.stages  = 4;
        t.A[1][0] = ScalarType( 0.5 );
        t.A[2][1] = ScalarType( 0.5 );
        t.A[3][2] = ScalarType( 1 );
        t.b[0]    = ScalarType( 1 ) / 6;
        t.b[1]    = ScalarType( 1 ) / 3;
        t.b[2]    = ScalarType( 1 ) / 3;
        t.b[3]    = ScalarType( 1 ) / 6;
        t.c[0]    = 0;
        t.c[1]    = ScalarType( 0.5 );
        t.c[2]    = ScalarType( 0.5 );
        t.c[3]    = 1;
        break;
    }

    return t;
}

/// @brief Semi-Lagrangian (MMOC) advection of a Q1 nodal scalar field.
template < typename ScalarType >
class MMOCTransport
{
  public:
    using Vec3 = dense::Vec< ScalarType, 3 >;

    MMOCTransport(
        const grid::shell::DistributedDomain&       domain,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& ownership_mask,
        const TimeSteppingScheme                    scheme = TimeSteppingScheme::RK4,
        const int                                   interpolation_width = sl::quintic_stencil_size )
    : domain_( &domain )
    , ownership_mask_( ownership_mask )
    , exchange_( domain )
    , tableau_( butcher_tableau< ScalarType >( scheme ) )
    , interp_width_( interpolation_width )
    {
        links_          = build_lateral_links( domain );
        multiplicity_   = build_multiplicity( domain, ownership_mask );
        coords_g_       = sl::ghosted_unit_sphere_coords< ScalarType >( domain, exchange_ );
        radii_g_        = sl::ghosted_shell_radii< ScalarType >( domain, exchange_ );
        lateral_valid_  = sl::ghosted_lateral_validity< ScalarType >( exchange_, coords_g_ );

        // Tolerance of the cone-coordinate containment test in locate_point, relative to |X|. mu = A^{-1} X is
        // formed through the adjugate of A = [p1 p2 p3]: its 2x2 minors cancel from O(1) operands down to
        // O(theta), theta being the angular cell size, and det A is O(theta^2), so mu carries a rounding error
        // of O(eps_mach / theta^2) relative to |X|. A fixed 1e-12 is exceeded from about refinement level 8 on;
        // a node then fails to locate its *own* position (it is a vertex of the seed cell), bounces between
        // the cells sharing that vertex until max_walk is spent, and is reported as an escape although it never
        // moved (28% of all nodes at level 10). Scale the tolerance with the mesh instead.
        {
            const int gw = sl::ghost_width;
            // coords_g_ is SoA: one 4D view per component. Owned node (0,0) and its x-neighbour of subdomain 0.
            double dot = 0;
            for ( int d = 0; d < 3; ++d )
            {
                auto p0 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, Kokkos::subview( coords_g_.comp_[d], 0, gw, gw, gw ) );
                auto p1 = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace{}, Kokkos::subview( coords_g_.comp_[d], 0, gw + 1, gw, gw ) );
                dot += static_cast< double >( p0() ) * static_cast< double >( p1() );
            }
            const double theta = std::acos( std::min( 1.0, std::max( -1.0, dot ) ) );
            const double eps   = 64.0 * std::numeric_limits< double >::epsilon() / ( theta * theta );
            locate_eps_        = std::max( ScalarType( 1e-12 ), static_cast< ScalarType >( eps ) );
        }

        const auto bounds = sl::shell_radius_bounds< ScalarType >( domain );
        r_min_            = bounds.first;
        r_max_            = bounds.second;

        T_g_     = exchange_.allocate_scalar< ScalarType >( "mmoc_T_ghosted" );
        u_g_     = exchange_.allocate_vec< ScalarType, 3 >( "mmoc_u_ghosted" );
        u_old_g_ = exchange_.allocate_vec< ScalarType, 3 >( "mmoc_u_old_ghosted" );

        T_new_ = grid::Grid4DDataScalar< ScalarType >(
            "mmoc_T_new",
            exchange_.num_subdomains(),
            exchange_.num_nodes_lateral(),
            exchange_.num_nodes_lateral(),
            exchange_.num_nodes_radial() );
    }

    /// @brief Largest Courant number this implementation supports for a single \ref step.
    ///
    /// The characteristic tracing itself is unconditionally stable -- there is no CFL condition in the scheme.
    /// What is bounded is the *distance* the departure point may travel: it has to stay inside the local
    /// subdomain plus its ghost layer, because that is all the data available to interpolate from. Substepping
    /// does **not** relax this: substeps refine the trajectory but the foot point still ends up a full Courant
    /// number away. Raising the limit means widening \ref terra::fe::wedge::sl::ghost_width.
    [[nodiscard]] static constexpr ScalarType max_courant()
    {
        // A margin below the ghost width, so that the foot point stays strictly inside the ghosted region.
        return ScalarType( 0.9 ) * sl::ghost_width;
    }

    /// @brief Substeps giving a trajectory error comparable to the interpolation error.
    ///
    /// Purely an accuracy control -- see \ref max_courant for what actually limits the timestep. One substep is
    /// enough for a Courant number below one with a fourth-order scheme; the count is raised only so that the
    /// per-substep rotation angle stays small when the flow turns sharply within a timestep.
    [[nodiscard]] static int substeps_for_accuracy( const ScalarType courant )
    {
        const int n = static_cast< int >( Kokkos::ceil( courant ) );
        return n < 1 ? 1 : n;
    }

    /// @brief Advances `T` from t^n to t^{n+1} along the characteristics of the given velocity fields.
    ///
    /// @param T             in/out, the transported field
    /// @param u             velocity at t^{n+1}
    /// @param u_old         velocity at t^n
    /// @param dt            the full timestep
    /// @param substeps      number of substeps; see \ref recommended_substeps
    /// @param global_limiter clip the interpolated value to the global range of T^n
    void step(
        linalg::VectorQ1Scalar< ScalarType >&          T,
        const linalg::VectorQ1Vec< ScalarType, 3 >&    u,
        const linalg::VectorQ1Vec< ScalarType, 3 >&    u_old,
        const ScalarType                               dt,
        const int                                      substeps,
        const bool                                     global_limiter = true )
    {
        util::Timer timer( "mmoc_transport" );

        exchange_.fill( T.grid_data(), T_g_ );
        exchange_.fill( u.grid_data(), u_g_ );
        exchange_.fill( u_old.grid_data(), u_old_g_ );

        ScalarType t_min = std::numeric_limits< ScalarType >::max();
        ScalarType t_max = std::numeric_limits< ScalarType >::lowest();

        if ( global_limiter )
        {
            const auto T_data = T.grid_data();
            Kokkos::parallel_reduce(
                "mmoc_global_range",
                grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, ScalarType& lo,
                               ScalarType& hi ) {
                    lo = Kokkos::min( lo, T_data( sd, x, y, r ) );
                    hi = Kokkos::max( hi, T_data( sd, x, y, r ) );
                },
                Kokkos::Min< ScalarType >( t_min ),
                Kokkos::Max< ScalarType >( t_max ) );
            Kokkos::fence();

            const MPI_Datatype mpi_scalar = terra::mpi::mpi_datatype< ScalarType >();
            MPI_Allreduce( MPI_IN_PLACE, &t_min, 1, mpi_scalar, MPI_MIN, domain_->comm() );
            MPI_Allreduce( MPI_IN_PLACE, &t_max, 1, mpi_scalar, MPI_MAX, domain_->comm() );
        }
        else
        {
            t_min = std::numeric_limits< ScalarType >::lowest();
            t_max = std::numeric_limits< ScalarType >::max();
        }

        Kokkos::deep_copy( num_escape_locations_, 0 );

        long long escapes = 0;
        trace( T, dt, substeps, t_min, t_max, escapes );

        Kokkos::deep_copy( T.grid_data(), T_new_ );

        // Duplicated interface nodes are computed independently by each subdomain that shares them, and the
        // results do *not* agree: the side for which the foot point lies beyond its own block reconstructs with
        // a stencil clamped against the seam, while the side that owns the point uses a centred one. Picking
        // the larger, as this did with a MAX reduction, is biased upward at every seam on every step -- worth
        // 34% of the L2 error on the level-5 cone, measured by swapping it for MIN. Average instead: an
        // additive exchange sums the contributions at a shared node, and dividing by how many subdomains share
        // it recovers the mean. `multiplicity_` is that count, assembled once the same way.
        communication::shell::send_recv( *domain_, T.grid_data(), communication::CommunicationReduction::SUM );
        {
            const auto t_v = T.grid_data();
            const auto m_v = multiplicity_;
            Kokkos::parallel_for(
                "mmoc_interface_average",
                grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
                KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r ) {
                    const ScalarType m = m_v( sd, x, y, r );
                    if ( m > ScalarType( 1 ) )
                        t_v( sd, x, y, r ) /= m;
                } );
            Kokkos::fence();
        }

        MPI_Allreduce( MPI_IN_PLACE, &escapes, 1, MPI_LONG_LONG, MPI_SUM, domain_->comm() );
        last_escapes_ = escapes;
    }

    /// @brief Number of subdomains sharing each node: 1 in the interior, more on a seam.
    ///
    /// Assembled by summing a field of ones through the same additive exchange the transport uses, so the count
    /// is exactly the number of contributions a shared node receives there.
    static grid::Grid4DDataScalar< ScalarType > build_multiplicity(
        const grid::shell::DistributedDomain&                          domain,
        const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >&       ownership_mask )
    {
        linalg::VectorQ1Scalar< ScalarType > ones( "sl_multiplicity", domain, ownership_mask );
        linalg::assign( ones, ScalarType( 1 ) );
        communication::shell::send_recv( domain, ones.grid_data(), communication::CommunicationReduction::SUM );

        grid::Grid4DDataScalar< ScalarType > m( "sl_multiplicity_data", ones.grid_data().extent( 0 ),
                                                ones.grid_data().extent( 1 ), ones.grid_data().extent( 2 ),
                                                ones.grid_data().extent( 3 ) );
        Kokkos::deep_copy( m, ones.grid_data() );
        return m;
    }

    /// @brief Builds the same-diamond, same-device lateral links; see sl::canonicalise_lateral_cell.
    static sl::LateralSubdomainLinks build_lateral_links( const grid::shell::DistributedDomain& domain )
    {
        const auto& subdomains = domain.subdomains();

        sl::LateralSubdomainLinks links;
        links.owned_nodes = domain.domain_info().subdomain_num_nodes_per_side_laterally();
        links.ghost       = sl::ghost_width;
        links.link         = Kokkos::View< int* [4] >( "sl_lateral_links", subdomains.size() );
        links.same_diamond = Kokkos::View< int* [4] >( "sl_same_diamond", subdomains.size() );

        auto host = Kokkos::create_mirror_view( links.link );
        auto host_sd = Kokkos::create_mirror_view( links.same_diamond );
        for ( size_t i = 0; i < subdomains.size(); ++i )
            for ( int f = 0; f < 4; ++f )
            {
                host( i, f )    = -1;
                host_sd( i, f ) = 0;
            }

        const int per_side = domain.domain_info().num_subdomains_per_diamond_side();

        // A neighbour qualifies only if it is in this map -- which is exactly the set resident on this rank --
        // and in the same diamond, so that the index axes are aligned and the remap is a pure translation.
        for ( const auto& [info, data] : subdomains )
        {
            const int me = static_cast< int >( std::get< 0 >( data ) );

            const int dx[4] = { -1, 1, 0, 0 };
            const int dy[4] = { 0, 0, -1, 1 };
            for ( int f = 0; f < 4; ++f )
            {
                const grid::shell::SubdomainInfo probe( info.diamond_id(), info.subdomain_x() + dx[f],
                                                        info.subdomain_y() + dy[f], info.subdomain_r() );
                // Whether a same-diamond neighbour *exists* is a property of the decomposition, not of who
                // owns it: the ghost exchange fills the ring across ranks just the same.
                const bool in_diamond = probe.subdomain_x() >= 0 && probe.subdomain_x() < per_side &&
                                        probe.subdomain_y() >= 0 && probe.subdomain_y() < per_side;
                host_sd( me, f ) = in_diamond ? 1 : 0;

                const auto it = subdomains.find( probe );
                if ( in_diamond && it != subdomains.end() )
                    host( me, f ) = static_cast< int >( std::get< 0 >( it->second ) );
            }
        }
        Kokkos::deep_copy( links.link, host );
        Kokkos::deep_copy( links.same_diamond, host_sd );
        return links;
    }

    /// @brief Indices of up to 16 nodes that escaped in the last \ref step, as (subdomain, x, y, r).
    ///
    /// Diagnostic only; the count in \ref last_escapes is authoritative.
    [[nodiscard]] std::vector< std::array< int, 4 > > last_escape_locations() const
    {
        auto host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, escape_locations_ );
        auto n    = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, num_escape_locations_ );

        std::vector< std::array< int, 4 > > result;
        for ( int i = 0; i < Kokkos::min( n(), max_escape_locations ); ++i )
            result.push_back( { host( i, 0 ), host( i, 1 ), host( i, 2 ), host( i, 3 ) } );
        return result;
    }

    /// @brief Number of nodes whose departure point could not be located in the last \ref step.
    ///
    /// Those nodes keep their previous value (no advection). A non-zero count means the per-substep Courant
    /// number exceeded the ghost layer width, or that the departure point fell into one of the degenerate
    /// corner regions at the twelve pentagonal points of the icosahedral grid.
    [[nodiscard]] long long last_escapes() const { return last_escapes_; }

    /// @brief Containment tolerance of the lateral point location, relative to |X| (mesh dependent).
    [[nodiscard]] ScalarType locate_tolerance() const { return locate_eps_; }

    /// @internal Traces the characteristics and writes the result into `T_new_`.
    ///
    /// Public only because CUDA does not permit an extended `__host__ __device__` lambda inside a private or
    /// protected member function. Not part of the interface -- use \ref step.
    void trace(
        const linalg::VectorQ1Scalar< ScalarType >& T,
        const ScalarType                            dt,
        const int                                   substeps,
        const ScalarType                            t_min,
        const ScalarType                            t_max,
        long long&                                  escapes )
    {
        const sl::IndexBounds bounds{ exchange_.num_nodes_lateral_ghosted(),
                                      exchange_.num_nodes_lateral_ghosted(),
                                      exchange_.num_nodes_radial_ghosted() };

        const sl::RadialSliceCoords< decltype( coords_g_ ) > lateral{ coords_g_, sl::ghost_width };

        const auto coords_g = coords_g_;
        const auto radii_g  = radii_g_;
        const auto T_g      = T_g_;
        const auto u_g      = u_g_;
        const auto u_old_g  = u_old_g_;
        const auto T_new    = T_new_;
        const auto tableau  = tableau_;

        const int  n_lat_g     = exchange_.num_nodes_lateral_ghosted();
        const int  n_rad_g     = exchange_.num_nodes_radial_ghosted();
        const int  n_lat_owned = exchange_.num_nodes_lateral();
        const int  n_rad_owned = exchange_.num_nodes_radial();
        const auto r_min   = r_min_;
        const auto r_max   = r_max_;

        const int        interp_width = interp_width_;
        const ScalarType h        = dt / static_cast< ScalarType >( substeps );
        const ScalarType inv_M    = ScalarType( 1 ) / static_cast< ScalarType >( substeps );
        constexpr int    max_walk = 4 * sl::ghost_width + 4;
        const ScalarType eps      = locate_eps_;

        const auto T_old         = T.grid_data();
        const auto lateral_valid = lateral_valid_;
        const auto escape_loc     = escape_locations_;
        const auto links          = links_;
        const auto escape_loc_num = num_escape_locations_;

        Kokkos::parallel_reduce(
            "mmoc_trace_characteristics",
            grid::shell::local_domain_md_range_policy_nodes( *domain_ ),
            KOKKOS_LAMBDA( const int sd, const int x, const int y, const int r, long long& esc ) {
                const int gx = sl::to_ghosted_index( x );
                const int gy = sl::to_ghosted_index( y );
                const int gr = sl::to_ghosted_index( r );

                Vec3 p;
                for ( int d = 0; d < 3; ++d )
                    p( d ) = coords_g( sd, gx, gy, gr, d );

                Vec3 X = p * radii_g( sd, gr );

                // Stencil the cubic interpolation may read from. Laterally it is the owned block only: a
                // lateral ghost row holds a neighbour's real values, but it belongs to another diamond, and
                // the index-space parametrisation kinks at that seam -- a stencil straddling it is
                // inconsistent, not just less accurate. Radially the ghost layers outside the CMB and the
                // surface exist only so that the radii array stays monotone; they hold extrapolated radii and
                // no field data at all, so trim them off too. In both directions the window then slides
                // inwards near the edge (one-sided but never extrapolating), and a foot point that lands
                // outside the range altogether falls back to the Q1 evaluation.
                sl::StencilBounds stencil{ { sl::ghost_width, n_lat_g - 1 - sl::ghost_width },
                                           { sl::ghost_width, n_lat_g - 1 - sl::ghost_width },
                                           { 0, n_rad_g - 1 } };
                {
                    const ScalarType r_tol = ScalarType( 1e-12 ) * r_max;
                    while ( stencil.r.lo < stencil.r.hi && radii_g( sd, stencil.r.lo ) < r_min - r_tol )
                        ++stencil.r.lo;
                    while ( stencil.r.hi > stencil.r.lo && radii_g( sd, stencil.r.hi ) > r_max + r_tol )
                        --stencil.r.hi;
                }

                // Seed with a cell all of whose nodes are owned: it always contains this node as a vertex,
                // and it can never be one of the degenerate ghost-corner wedges.
                sl::WedgeCell cell{ sl::to_ghosted_index( Kokkos::min( x, n_lat_owned - 2 ) ),
                                    sl::to_ghosted_index( Kokkos::min( y, n_lat_owned - 2 ) ),
                                    sl::to_ghosted_index( Kokkos::min( r, n_rad_owned - 2 ) ),
                                    0 };

                bool escaped = false;

                for ( int m = 0; m < substeps && !escaped; ++m )
                {
                    Vec3 kv[ButcherTableau< ScalarType >::max_stages];
                    const Vec3 X_start = X;

                    for ( int s = 0; s < tableau.stages; ++s )
                    {
                        Vec3 Y = X_start;
                        for ( int l = 0; l < s; ++l )
                            Y = Y + kv[l] * ( h * tableau.A[s][l] );

                        const auto res = sl::locate_point(
                            Y, sd, cell, lateral, radii_g, bounds, max_walk, eps,
                            /*clamp_radially=*/true, r_min, r_max, lateral_valid );

                        if ( !res.found )
                        {
                            escaped = true;
                            break;
                        }
                        cell = res.cell;

                        // Q1 for the velocity. An error here displaces the foot point and enters T just as
                        // directly as an error in T itself, but the cubic reconstruction buys nothing on this
                        // grid: a Stokes velocity is smooth, and in the rotation test the two are
                        // indistinguishable while Q1 is an order of magnitude cheaper.
                        const auto u_new_s =
                            sl::evaluate_q1_vec< ScalarType, 3 >( u_g, sd, res.cell, res.xi, res.eta, res.zeta );
                        const auto u_old_s = sl::evaluate_q1_vec< ScalarType, 3 >(
                            u_old_g, sd, res.cell, res.xi, res.eta, res.zeta );

                        // Global pseudo-time of this stage, in [0, 1] across the whole timestep.
                        const ScalarType tau = ( static_cast< ScalarType >( m ) + tableau.c[s] ) * inv_M;

                        for ( int d = 0; d < 3; ++d )
                            kv[s]( d ) = -( ( ScalarType( 1 ) - tau ) * u_new_s( d ) + tau * u_old_s( d ) );
                    }

                    if ( escaped )
                        break;

                    Vec3 X_next = X_start;
                    for ( int s = 0; s < tableau.stages; ++s )
                        X_next = X_next + kv[s] * ( h * tableau.b[s] );
                    X = X_next;
                }

                if ( !escaped )
                {
                    const auto res = sl::locate_point(
                        X, sd, cell, lateral, radii_g, bounds, max_walk, eps,
                        /*clamp_radially=*/true, r_min, r_max, lateral_valid );

                    if ( res.found )
                    {
                        // Sample the transported field at the departure point's *true* radius, not at the
                        // wedge's parametric one; see sl::radial_coords_from_radius for why the difference
                        // marches a radially stratified field inwards.
                        // If the walk finished in the ghost ring, hand the cell to the subdomain that owns it,
                        // so the reconstruction is centred rather than clamped against the seam.
                        int           ev_sd   = sd;
                        sl::WedgeCell ev_cell = res.cell;
                        ScalarType    ev_zeta = res.zeta;
                        // Twice, so that a ghost corner reaches the diagonal neighbour: x first, then y.
                        sl::canonicalise_lateral_cell< ScalarType >( ev_sd, ev_cell, links, bounds );
                        sl::canonicalise_lateral_cell< ScalarType >( ev_sd, ev_cell, links, bounds );
                        const auto ev_stencil = sl::widen_across_internal_seams( stencil, ev_sd, links );
                        sl::radial_coords_from_radius( ev_sd, X.norm(), radii_g, n_rad_g - 1, r_min, r_max,
                                                       ev_cell.r, ev_zeta );

                        const ScalarType value = sl::evaluate_cubic_scalar(
                            T_g, ev_sd, ev_cell, res.xi, res.eta, ev_zeta, radii_g, ev_stencil, lateral_valid,
                            /*clip_to_cell=*/true, /*limit_slopes=*/false, interp_width );
                        T_new( sd, x, y, r ) = Kokkos::clamp( value, t_min, t_max );
                        return;
                    }
                }

                // The departure point left the ghosted region, or the walk ran into one of the degenerate
                // corner wedges. `cell` still holds the last wedge that was accepted, so interpolate at the
                // point of that wedge closest to the departure point: a bounded, first-order error, rather
                // than leaving the node un-advected for a whole timestep.
                {
                    ScalarType xi = 0, eta = 0, zeta = 0;
                    sl::clamp_to_wedge( X, sd, cell, lateral, radii_g, xi, eta, zeta );

                    sl::WedgeCell ev_cell = cell;
                    sl::radial_coords_from_radius( sd, X.norm(), radii_g, n_rad_g - 1, r_min, r_max,
                                                   ev_cell.r, zeta );

                    const ScalarType value = sl::evaluate_cubic_scalar(
                        T_g, sd, ev_cell, xi, eta, zeta, radii_g, stencil, lateral_valid,
                        /*clip_to_cell=*/true, /*limit_slopes=*/false, interp_width );
                    T_new( sd, x, y, r )   = Kokkos::clamp( value, t_min, t_max );
                }
                esc += 1;

                const int slot = Kokkos::atomic_fetch_add( &escape_loc_num(), 1 );
                if ( slot < max_escape_locations )
                {
                    escape_loc( slot, 0 ) = sd;
                    escape_loc( slot, 1 ) = x;
                    escape_loc( slot, 2 ) = y;
                    escape_loc( slot, 3 ) = r;
                }
            },
            escapes );
        Kokkos::fence();
    }

  private:
    const grid::shell::DistributedDomain*                    domain_ = nullptr;
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag >        ownership_mask_;
    sl::GhostExchange                                        exchange_;
    ButcherTableau< ScalarType >                             tableau_;

    grid::Grid4DDataVec< ScalarType, 3 >    coords_g_;
    grid::Grid2DDataScalar< ScalarType >    radii_g_;
    Kokkos::View< uint8_t*** >              lateral_valid_;
    ScalarType                              r_min_ = 0;
    ScalarType                              r_max_ = 0;
    /// Containment tolerance of the lateral point location, relative to |X|; see the constructor.
    ScalarType                              locate_eps_ = ScalarType( 1e-12 );

    grid::Grid4DDataScalar< ScalarType > T_g_;
    grid::Grid4DDataVec< ScalarType, 3 > u_g_;
    grid::Grid4DDataVec< ScalarType, 3 > u_old_g_;
    grid::Grid4DDataScalar< ScalarType > T_new_;

    long long last_escapes_ = 0;

    /// Nodes per direction the foot-point reconstruction asks for; see sl::stencil_window for the ladder it
    /// steps down when a centred window of this width does not fit.
    int interp_width_ = sl::quintic_stencil_size;

    /// Lateral links to same-diamond neighbours resident on this device; see sl::canonicalise_lateral_cell.
    sl::LateralSubdomainLinks links_;

    /// Subdomains sharing each node; see build_multiplicity.
    grid::Grid4DDataScalar< ScalarType > multiplicity_;

    static constexpr int          max_escape_locations = 16;
    Kokkos::View< int* [4] >      escape_locations_{ "mmoc_escape_locations", max_escape_locations };
    Kokkos::View< int >           num_escape_locations_{ "mmoc_num_escape_locations" };
};

} // namespace terra::fe::wedge::operators::shell
