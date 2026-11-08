#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "grid/grid_types.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "util/logging.hpp"

/// @file
/// Run-to-run reproducible cell-to-node scatters (TERRA_DETERMINISTIC_SCATTER=1).
///
/// Floating-point addition is not associative, so a node that receives its contributions through concurrent
/// atomic adds gets a result that depends on the order in which the hardware happened to serialise them. Here
/// every contribution is instead rounded to an integer multiple of a quantum q and added as a 64-bit integer:
/// integer addition is associative (and wraps modulo 2^64), so the sum is bitwise identical in every order. The
/// kernels are untouched; only the atomic itself changes, and the destination's 8-byte storage is reused for the
/// integers (double fields only; other types fall back to the plain atomics).
///
/// The quantum is chosen per apply: q = B * 2^-52 with B a bound on a single contribution, B = beta * max|src| *
/// 2^margin. beta (the largest contribution per unit of input) is measured on the first apply of each operator
/// instance and every @ref remeasure_interval applies after that, by a pass that records max|contribution| with
/// an atomic max (exact and order-independent) and adds nothing. A contribution that does not fit the window
/// raises a flag; in Replace mode the apply is then redone with a 256x larger B, in Add mode (destination
/// already holds data, see @ref Scatter::run) it is an error, as is a second overflow.
///
/// The rounding error is q/2 per contribution, i.e. 2^-53 * B: of the same order as the round-off of the
/// double atomics it replaces, measured against the largest contribution instead of the running partial sum.
namespace terra::det
{

namespace detail
{
inline int& override_state()
{
    static int v = -1; // -1: from the environment, 0/1: set_requested()
    return v;
}
} // namespace detail

/// Whether the deterministic scatter is on: TERRA_DETERMINISTIC_SCATTER=1 in the environment, or set_requested().
inline bool requested()
{
    if ( detail::override_state() >= 0 )
        return detail::override_state() == 1;
    static const bool r = [] {
        const char* e = std::getenv( "TERRA_DETERMINISTIC_SCATTER" );
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return r;
}

/// Programmatic switch (tests): overrides the environment from now on.
inline void set_requested( const bool on )
{
    detail::override_state() = on ? 1 : 0;
}

/// Safety factor (powers of two) between the measured largest contribution and the fixed-point window.
inline int margin_bits()
{
    static const int m = [] {
        const char* e = std::getenv( "TERRA_DETERMINISTIC_MARGIN" );
        return e ? std::atoi( e ) : 6;
    }();
    return m;
}

/// Re-measure the contribution bound every this many applies (0: only on the first apply and on overflow).
inline int remeasure_interval()
{
    static const int n = [] {
        const char* e = std::getenv( "TERRA_DETERMINISTIC_REMEASURE" );
        return e ? std::atoi( e ) : 64;
    }();
    return n;
}

/// Device-side state of one scatter. Default-constructed: plain floating-point atomics.
struct FixedPoint
{
    double  inv_q    = 0.0;     ///< 1/q; 0 means inactive
    double* measure  = nullptr; ///< measurement pass: receives max|contribution|, nothing is added
    int*    overflow = nullptr; ///< set when a contribution does not fit the window

    KOKKOS_INLINE_FUNCTION bool active() const { return inv_q != 0.0; }

    /// Window for a sum whose total magnitude is bounded by `total` (e.g. a bin reduction): contributions and
    /// the sum both fit, no guard needed.
    static FixedPoint for_total( const double total )
    {
        FixedPoint fp;
        if ( total > 0.0 && std::isfinite( total ) )
            fp.inv_q = std::ldexp( 1.0, 61 ) / total;
        return fp;
    }
};

/// The one replacement for Kokkos::atomic_add( addr, v ) at every cell-to-node scatter site.
template < typename T, typename V >
KOKKOS_INLINE_FUNCTION void add( const FixedPoint& fp, T* addr, const V v )
{
    if constexpr ( sizeof( T ) == sizeof( long long ) && std::is_floating_point_v< T > )
    {
        if ( fp.measure != nullptr )
        {
            Kokkos::atomic_max( fp.measure, Kokkos::abs( static_cast< double >( v ) ) );
            return;
        }
        if ( fp.inv_q != 0.0 )
        {
            const double s = static_cast< double >( v ) * fp.inv_q;
            if ( !( Kokkos::abs( s ) < 9007199254740992.0 ) ) // 2^53: llround exact, 1024 of them cannot wrap
            {
                if ( fp.overflow != nullptr )
                    Kokkos::atomic_store( fp.overflow, 1 );
                return;
            }
            Kokkos::atomic_add( reinterpret_cast< long long* >( addr ), static_cast< long long >( Kokkos::round( s ) ) );
            return;
        }
    }
    Kokkos::atomic_add( addr, static_cast< T >( v ) );
}

/// max |v| over a contiguous view (any rank).
template < typename ViewT >
double max_abs( const ViewT& v )
{
    using T = typename ViewT::non_const_value_type;
    if ( v.span() == 0 )
        return 0.0;
    const T* p = v.data();
    double   m = 0.0;
    Kokkos::parallel_reduce(
        "det_max_abs",
        Kokkos::RangePolicy< Kokkos::IndexType< size_t > >( 0, v.span() ),
        KOKKOS_LAMBDA( const size_t i, double& lmax ) {
            const double a = Kokkos::abs( static_cast< double >( p[i] ) );
            lmax           = a > lmax ? a : lmax;
        },
        Kokkos::Max< double >( m ) );
    Kokkos::fence();
    return m;
}

template < typename T, int VecDim >
double max_abs( const grid::Grid4DDataVec< T, VecDim >& v )
{
    double m = 0.0;
    for ( int d = 0; d < VecDim; ++d )
        m = std::max( m, max_abs( v.comp_[d] ) );
    return m;
}

/// In-place double -> fixed point (Add mode: the destination already holds data).
template < typename ViewT >
void to_fixed( const ViewT& v, const double inv_q )
{
    double* p = v.data();
    Kokkos::parallel_for(
        "det_to_fixed", Kokkos::RangePolicy< Kokkos::IndexType< size_t > >( 0, v.span() ), KOKKOS_LAMBDA( const size_t i ) {
            const long long iv = static_cast< long long >( Kokkos::round( p[i] * inv_q ) );
            *reinterpret_cast< long long* >( p + i ) = iv;
        } );
}

/// In-place fixed point -> double.
template < typename ViewT >
void from_fixed( const ViewT& v, const double q )
{
    double* p = v.data();
    Kokkos::parallel_for(
        "det_from_fixed", Kokkos::RangePolicy< Kokkos::IndexType< size_t > >( 0, v.span() ), KOKKOS_LAMBDA( const size_t i ) {
            const long long iv = *reinterpret_cast< const long long* >( p + i );
            p[i]               = static_cast< double >( iv ) * q;
        } );
}

template < typename ViewT >
void from_fixed( const ViewT& v, const FixedPoint& fp )
{
    if ( fp.active() )
    {
        from_fixed( v, 1.0 / fp.inv_q );
        Kokkos::fence();
    }
}

/// The destination(s) of a scatter as a list of scalar views.
template < typename T >
std::vector< grid::Grid4DDataScalar< T > > views( const grid::Grid4DDataScalar< T >& v )
{
    return { v };
}
template < typename T, int VecDim >
std::vector< grid::Grid4DDataScalar< T > > views( const grid::Grid4DDataVec< T, VecDim >& v )
{
    return std::vector< grid::Grid4DDataScalar< T > >( v.comp_, v.comp_ + VecDim );
}

/// Host-side state of one operator's scatter: the measured bound and the two device flags. Trivially
/// copyable (Views), so operators that are copied into their kernels can hold one.
class Scatter
{
  public:
    /// Runs `launch( fp )`, which must perform the operator's kernel launch(es) with every scatter going through
    /// det::add( fp, ... ). `dsts` are the (double) destination views, `src_max_abs` the max norm of the input
    /// (1 for a linear form), `add_mode` whether `dsts` already hold data to be added to.
    template < typename T, typename Launch >
    void run(
        const bool                                            add_mode,
        const double                                          src_max_abs,
        const std::vector< grid::Grid4DDataScalar< T > >&    dsts,
        Launch&&                                              launch )
    {
        if ( !requested() )
        {
            launch( FixedPoint{} );
            return;
        }
        if constexpr ( sizeof( T ) != sizeof( long long ) )
        {
            if ( !warned_ )
            {
                util::logroot << "[det] fixed-point scatter needs 8-byte fields; this operator uses plain atomics"
                              << std::endl;
                warned_ = true;
            }
            launch( FixedPoint{} );
            return;
        }
        else
        {
            if ( !overflow_.is_allocated() )
            {
                overflow_ = Kokkos::View< int >( "det_overflow" );
                measure_  = Kokkos::View< double >( "det_measure" );
            }
            ++applies_;
            if ( !( src_max_abs > 0.0 ) )
            {
                launch( FixedPoint{} ); // every contribution is zero
                return;
            }
            const int  every   = remeasure_interval();
            const bool measure = beta_ < 0.0 || ( every > 0 && applies_ % every == 0 );
            if ( measure )
            {
                Kokkos::deep_copy( measure_, 0.0 );
                FixedPoint fp;
                fp.measure = measure_.data();
                launch( fp );
                Kokkos::fence();
                double m = 0.0;
                Kokkos::deep_copy( m, measure_ );
                beta_ = m / src_max_abs;
            }
            if ( !( beta_ > 0.0 ) )
            {
                launch( FixedPoint{} ); // no contribution is nonzero
                return;
            }
            for ( int attempt = 0;; ++attempt )
            {
                double bound = beta_ * src_max_abs * std::ldexp( 1.0, margin_bits() );
                if ( add_mode )
                {
                    double dmax = 0.0;
                    for ( const auto& v : dsts )
                        dmax = std::max( dmax, max_abs( v ) );
                    bound = std::max( bound, dmax * std::ldexp( 1.0, -9 ) ); // existing data must fit 2^61
                }
                FixedPoint fp;
                fp.inv_q    = std::ldexp( 1.0, 52 ) / bound;
                fp.overflow = overflow_.data();
                Kokkos::deep_copy( overflow_, 0 );
                if ( add_mode )
                    for ( const auto& v : dsts )
                        to_fixed( v, fp.inv_q );
                launch( fp );
                Kokkos::fence();
                int of = 0;
                Kokkos::deep_copy( of, overflow_ );
                if ( of == 0 )
                {
                    for ( const auto& v : dsts )
                        from_fixed( v, bound * std::ldexp( 1.0, -52 ) );
                    Kokkos::fence();
                    return;
                }
                if ( add_mode || attempt > 0 )
                    throw std::runtime_error(
                        "det: a scatter contribution exceeded the fixed-point window (bound " + std::to_string( bound ) +
                        "); raise TERRA_DETERMINISTIC_MARGIN (default 6) or lower TERRA_DETERMINISTIC_REMEASURE" );
                beta_ *= 256.0; // Replace mode: the destination was zero, start over
                for ( const auto& v : dsts )
                    Kokkos::deep_copy( v, T( 0 ) );
            }
        }
    }

  private:
    double                  beta_    = -1.0;
    long                    applies_ = 0;
    bool                    warned_  = false;
    Kokkos::View< int >     overflow_;
    Kokkos::View< double >  measure_;
};

/// splitmix64 finaliser of (seed, idx) scaled to [0, 1): a reproducible stand-in for a random draw per node.
KOKKOS_INLINE_FUNCTION double hash_to_unit( const uint64_t seed, const uint64_t idx )
{
    uint64_t z = seed * 0x9E3779B97F4A7C15ull + idx * 0xBF58476D1CE4E5B9ull;
    z          = ( z ^ ( z >> 30 ) ) * 0xBF58476D1CE4E5B9ull;
    z          = ( z ^ ( z >> 27 ) ) * 0x94D049BB133111EBull;
    z          = z ^ ( z >> 31 );
    return static_cast< double >( z >> 11 ) * ( 1.0 / 9007199254740992.0 );
}

} // namespace terra::det
