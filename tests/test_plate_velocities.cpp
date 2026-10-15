/// \file
/// Correctness checks for the plate velocity field used as the surface Dirichlet BC.
///
/// The point of this test is that it does *not* rely on a golden baseline. A regression
/// baseline only tells you the answer has not changed; the checks here have exact expected
/// values derived from the structure of the problem, so they say something about whether the
/// answer was right in the first place.
///
/// Over a single plate the velocity is a rigid rotation, v(x) = W x r, which gives three
/// invariants that hold to round-off and need no reference implementation:
///
///   1. tangency        v(p) . p     == 0     (cross product is perpendicular to its operands)
///   2. rigidity        (v(p)-v(q)) . (p-q) == 0   (rigid motion preserves distances)
///   3. averaging       the locally averaged velocity over a symmetric stencil that lies
///                      entirely on one plate is parallel to the unaveraged one, because the
///                      stencil's centroid is exactly radial
///
/// Time interpolation is linear by construction, which gives two more:
///
///   4. at an age that is exactly a plate stage, the interpolated value reproduces that stage
///   5. at any other age the value lies on the segment between the bracketing stage values
///
/// Two failure modes of the time interpolation are reported rather than asserted, because they
/// are properties of the reconstruction data and not bugs: a point whose plate ID differs
/// between the bracketing stages is being interpolated between two different rigid bodies, and
/// a point found at only one of the two stages is silently interpolated against a zero vector
/// (PlateVelocityProvider returns the not-found handler's {0,0,0}).
///
/// Requires the plate reconstruction data, which is not in the repository. Pass either a
/// directory holding one .geojson and one .rot, or the two files explicitly, as arguments or via
/// TERRA_PLATE_DATA_DIR; without it the test skips (exit code 77).

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "terra/plates/plate_velocity_provider.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using namespace terra;

using terra::util::logroot;

namespace {

constexpr int kSkipExitCode = 77;

/// Wraps a point/weight provider but overrides the reported maximum sample distance.
///
/// PlateVelocityProvider::getLocallyAveragedPointVelocity() skips averaging entirely when
/// maxDistance() is smaller than the point's distance to the plate boundary. Forcing that
/// value lets the test drive either branch for the same point, which is what makes invariant 3
/// checkable: deep inside a plate, where the exact answer is known, the averaging path is
/// normally never taken.
class ForcedReachProvider final : public plates::LocalAveragingPointWeightProvider
{
  public:
    ForcedReachProvider( const plates::LocalAveragingPointWeightProvider& inner, double reach )
    : inner_( inner )
    , reach_( reach )
    {}

    std::vector< std::pair< vec3D, double > > samplePointsAndWeightsLonLat( const vec3D& p ) const override
    {
        return inner_.samplePointsAndWeightsLonLat( p );
    }

    double maxDistance( const vec3D& ) const override { return reach_; }

  private:
    const plates::LocalAveragingPointWeightProvider& inner_;
    double                                           reach_;
};

/// Quasi-uniform point set on the unit sphere. Deterministic, and avoids the pole clustering a
/// lon/lat sweep would give.
std::vector< vec3D > fibonacci_sphere( int n )
{
    const double golden = 0.5 * ( 1.0 + std::sqrt( 5.0 ) );

    std::vector< vec3D > points;
    points.reserve( n );

    for ( int i = 0; i < n; ++i )
    {
        const double z   = 1.0 - 2.0 * ( static_cast< double >( i ) + 0.5 ) / static_cast< double >( n );
        const double rho = std::sqrt( std::max( 0.0, 1.0 - z * z ) );
        const double phi = 2.0 * plates::conversions::pi * static_cast< double >( i ) / golden;

        points.push_back( vec3D{ rho * std::cos( phi ), rho * std::sin( phi ), z } );
    }

    return points;
}

double rel( double err, double scale )
{
    return scale > 0.0 ? err / scale : err;
}

/// Accumulates the worst relative violation of one invariant.
struct Worst
{
    std::string name;
    double      tol;
    double      value{ 0.0 };
    long        samples{ 0 };

    void observe( double v )
    {
        value = std::max( value, v );
        ++samples;
    }

    bool ok() const { return value <= tol; }

    void report() const
    {
        logroot << ( ok() ? "  PASS  " : "  FAIL  " ) << name << ": worst = " << value << ", tol = " << tol << " ("
                << samples << " samples)" << std::endl;
    }
};

} // namespace

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    // ---- locate the reconstruction data, or skip ----

    std::string topologies;
    std::string reconstructions;

    if ( argc > 2 )
    {
        topologies      = argv[1];
        reconstructions = argv[2];
    }
    else
    {
        // A single argument (or the environment variable) names a directory; find the one
        // topology file and the one rotation file in it, since the names vary per dataset.
        std::string dir;
        if ( argc > 1 )
        {
            dir = argv[1];
        }
        else if ( const char* env = std::getenv( "TERRA_PLATE_DATA_DIR" ) )
        {
            dir = env;
        }

        if ( !dir.empty() && std::filesystem::is_directory( dir ) )
        {
            for ( const auto& entry : std::filesystem::directory_iterator( dir ) )
            {
                const auto ext = entry.path().extension().string();
                if ( ext == ".geojson" && topologies.empty() )
                {
                    topologies = entry.path().string();
                }
                else if ( ext == ".rot" && reconstructions.empty() )
                {
                    reconstructions = entry.path().string();
                }
            }
        }
    }

    if ( topologies.empty() || reconstructions.empty() || !std::filesystem::exists( topologies ) ||
         !std::filesystem::exists( reconstructions ) )
    {
        logroot << "SKIP: plate reconstruction data not found.\n"
                << "      Pass a directory holding one .geojson and one .rot as the first argument,\n"
                << "      or the two files explicitly, or set TERRA_PLATE_DATA_DIR." << std::endl;
        return kSkipExitCode;
    }

    logroot << "topologies      : " << topologies << "\n"
            << "reconstructions : " << reconstructions << std::endl;

    plates::PlateVelocityProvider oracle( topologies, reconstructions );

    // The stencil the mantle circulation app actually uses (plates.hpp): centre plus one ring of
    // six at 1/100 rad, Gaussian weights with sigma = 0.1.
    const plates::UniformCirclesPointWeightProvider stencil( { { 1.0 / 100.0, 6 } }, 1e-1 );

    // Reach 0 never averages (invariants 1, 2, 4, 5 see the pure rigid rotation);
    // a reach larger than any distance on the sphere always averages (invariant 3).
    const ForcedReachProvider never_average( stencil, 0.0 );
    const ForcedReachProvider always_average( stencil, 1e30 );

    plates::DefaultPlateNotFoundHandler handler;

    // The pre-port Boost path is orders of magnitude slower, so the reference run that this is
    // diffed against needs a smaller point set to finish in a sensible walltime.
    int num_points = 4000;
    if ( const char* env = std::getenv( "TERRA_PLATE_TEST_POINTS" ) )
    {
        num_points = std::atoi( env );
    }

    const auto points = fibonacci_sphere( num_points );
    logroot << "points          : " << num_points << std::endl;

    const auto& stages = oracle.getListOfPlateStages();
    if ( stages.size() < 2 )
    {
        logroot << "Need at least two plate stages to test time interpolation." << std::endl;
        return EXIT_FAILURE;
    }

    // A few ages spread across the available range, all exactly on stages.
    const std::vector< double > test_stages{
        stages.front(), stages[stages.size() / 4], stages[stages.size() / 2], stages.back() };

    // ---- invariants over a single age stage ----

    Worst tangency{ "tangency          |v.p| / |v|", 1e-12 };
    Worst rigidity{ "rigidity          |dv.dp| / (|dv||dp|)", 1e-12 };
    Worst parallel{ "averaging (dir)   |v_avg x v_dir| / (|v_avg||v_dir|)", 1e-10 };
    Worst magnitude{ "averaging (mag)   | |v_avg|/|v_dir| - 1 |", 1e-2 };

    std::map< uint_t, long > plate_histogram;

    // Optional golden-data dump: plate ID and velocity per (stage, point), at full precision.
    // Diffing this against a build of the pre-port code is what establishes that replacing
    // Boost.Geometry did not move any answers.
    std::FILE* dump = nullptr;
    if ( const char* path = std::getenv( "TERRA_PLATE_DUMP" ) )
    {
        dump = std::fopen( path, "w" );
        if ( dump )
        {
            std::fprintf( dump, "age,point,plate_id,vx,vy,vz,avx,avy,avz\n" );
        }
    }

    for ( const double age : test_stages )
    {
        vec3D prev_v{ 0, 0, 0 };
        vec3D prev_p{ 0, 0, 0 };
        uint_t prev_id  = 0;
        bool   have_prev = false;

        for ( const auto& p : points )
        {
            const uint_t id = oracle.findPlateID( p, age );
            if ( id == oracle.idWhenNoPlateFound )
            {
                have_prev = false;
                continue;
            }

            plate_histogram[id] += 1;

            const vec3D v_dir = oracle.getLocallyAveragedPointVelocity( p, age, never_average, handler );
            const double v_norm = v_dir.norm();
            if ( v_norm == 0.0 )
            {
                have_prev = false;
                continue;
            }

            // 1. tangency: v = W x r is perpendicular to r, in both branches of the provider --
            //    in the direct branch by construction, in the averaged branch only because of the
            //    explicit projection.
            tangency.observe( rel( std::abs( v_dir.dot( p ) ), v_norm * p.norm() ) );

            // 3. averaging over a symmetric stencil entirely inside one plate. The six ring
            //    points are equidistant from the centre and equivariant under rotation about it,
            //    so their centroid is exactly radial and the averaged velocity is parallel to the
            //    unaveraged one. The magnitude check is what catches a missing division by the
            //    weight sum, which would leave the direction untouched.
            bool stencil_on_one_plate = true;
            for ( const auto& [sample_lonlat, weight] : stencil.samplePointsAndWeightsLonLat(
                      plates::conversions::cart2sph( p ) ) )
            {
                const vec3D sample_cart = plates::conversions::sph2cart(
                    { sample_lonlat( 0 ), sample_lonlat( 1 ) }, sample_lonlat( 2 ) );
                if ( oracle.findPlateID( sample_cart, age ) != id )
                {
                    stencil_on_one_plate = false;
                    break;
                }
            }

            if ( stencil_on_one_plate )
            {
                const vec3D v_avg = oracle.getLocallyAveragedPointVelocity( p, age, always_average, handler );
                const double a_norm = v_avg.norm();

                if ( a_norm > 0.0 )
                {
                    parallel.observe( rel( v_avg.cross( v_dir ).norm(), a_norm * v_norm ) );
                    magnitude.observe( std::abs( a_norm / v_norm - 1.0 ) );
                }
            }

            // 2. rigidity, between consecutive points that landed on the same plate.
            if ( have_prev && prev_id == id )
            {
                const vec3D dv = v_dir - prev_v;
                const vec3D dp = p - prev_p;
                const double scale = dv.norm() * dp.norm();
                if ( scale > 0.0 )
                {
                    rigidity.observe( rel( std::abs( dv.dot( dp ) ), scale ) );
                }
            }

            if ( dump )
            {
                const vec3D v_avg = oracle.getLocallyAveragedPointVelocity( p, age, stencil, handler );
                std::fprintf( dump,
                              "%.17g,%d,%u,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                              age,
                              static_cast< int >( &p - points.data() ),
                              id,
                              v_dir( 0 ),
                              v_dir( 1 ),
                              v_dir( 2 ),
                              v_avg( 0 ),
                              v_avg( 1 ),
                              v_avg( 2 ) );
            }

            prev_v    = v_dir;
            prev_p    = p;
            prev_id   = id;
            have_prev = true;
        }
    }

    // ---- time interpolation ----

    Worst on_stage{ "interp on stage   |v_interp - v_stage| / |v_stage|", 1e-14 };
    Worst is_lerp{ "interp is lerp    |v - (v0 + f(v1-v0))| / |v|", 1e-14 };

    long id_changed_between_stages = 0;
    long found_at_one_stage_only   = 0;
    long interp_points             = 0;
    double worst_linear_error      = 0.0;
    double worst_stage_delta       = 0.0;

    // Exactly on a stage, the interpolation factor is exactly 1.0, so the result must reproduce
    // the stage value bit for bit (up to the tolerance above, which is there only to guard
    // against a compiler reassociating the lerp).
    for ( const double age : test_stages )
    {
        for ( const auto& p : points )
        {
            const vec3D v_stage  = oracle.getLocallyAveragedPointVelocity( p, age, never_average, handler );
            const vec3D v_interp =
                oracle.getLocallyAveragedPointVelocityInterpolatedInTime( p, age, never_average, handler );

            const double scale = v_stage.norm();
            if ( scale > 0.0 )
            {
                on_stage.observe( rel( ( v_interp - v_stage ).norm(), scale ) );
            }
        }
    }

    // Between stages: the result must lie on the segment joining the two bracketing values.
    for ( size_t k = 1; k + 1 < stages.size(); k += std::max< size_t >( 1, stages.size() / 8 ) )
    {
        const double age_mid = 0.5 * ( stages[k] + stages[k + 1] );

        double age_floor, age_ceil, factor;
        std::tie( age_floor, age_ceil, factor ) = oracle.getSurroundingAges( age_mid );

        for ( const auto& p : points )
        {
            const uint_t id_floor = oracle.findPlateID( p, age_floor );
            const uint_t id_ceil  = oracle.findPlateID( p, age_ceil );

            const bool found_floor = id_floor != oracle.idWhenNoPlateFound;
            const bool found_ceil  = id_ceil != oracle.idWhenNoPlateFound;

            if ( found_floor != found_ceil )
            {
                ++found_at_one_stage_only;
                continue;
            }
            if ( !found_floor )
            {
                continue;
            }
            if ( id_floor != id_ceil )
            {
                ++id_changed_between_stages;
                continue;
            }

            ++interp_points;

            const vec3D v0 = oracle.getLocallyAveragedPointVelocity( p, age_floor, never_average, handler );
            const vec3D v1 = oracle.getLocallyAveragedPointVelocity( p, age_ceil, never_average, handler );
            const vec3D v =
                oracle.getLocallyAveragedPointVelocityInterpolatedInTime( p, age_mid, never_average, handler );

            // Assert the result *is* the linear interpolant, by recomputing it. The obvious
            // alternative -- checking that v lies on the segment via a cross product -- is
            // hopelessly ill-conditioned here: this dataset's stages are 0.1 Ma apart, so
            // v1 - v0 is a difference of nearly equal vectors and is almost pure round-off.
            // Comparing v against a quantity of its own magnitude has no such problem.
            const vec3D lerp = v0 + factor * ( v1 - v0 );
            const double vnorm = v.norm();
            if ( vnorm > 0.0 )
            {
                is_lerp.observe( rel( ( v - lerp ).norm(), vnorm ) );
            }

            // How far above the round-off floor the stage-to-stage change actually is.
            if ( v0.norm() > 0.0 )
            {
                worst_stage_delta = std::max( worst_stage_delta, ( v1 - v0 ).norm() / v0.norm() );
            }

            // Not an assertion: the size of the linear-in-time approximation. The true stage pole
            // is not linear in age, so this is a modelling error to be bounded, not removed.
            const vec3D v_direct_mid = oracle.getLocallyAveragedPointVelocity( p, age_mid, never_average, handler );
            const double dnorm       = v_direct_mid.norm();
            if ( dnorm > 0.0 )
            {
                worst_linear_error = std::max( worst_linear_error, ( v - v_direct_mid ).norm() / dnorm );
            }
        }
    }

    // ---- report ----

    if ( dump )
    {
        std::fclose( dump );
    }

    logroot << "\n--- exact invariants ---" << std::endl;
    const Worst* checks[] = { &tangency, &rigidity, &parallel, &magnitude, &on_stage, &is_lerp };
    for ( const auto* c : checks )
    {
        c->report();
    }

    logroot << "\n--- diagnostics (not assertions) ---\n"
            << "  distinct plate IDs hit          : " << plate_histogram.size() << "\n"
            << "  points interpolated cleanly     : " << interp_points << "\n"
            << "  plate ID changed between stages : " << id_changed_between_stages
            << "   (interpolating between two different rigid bodies)\n"
            << "  found at one stage only         : " << found_at_one_stage_only
            << "   (interpolated against a zero vector)\n"
            << "  worst linear-in-time deviation  : " << worst_linear_error << " relative\n"
            << "  largest stage-to-stage change   : " << worst_stage_delta << " relative"
            << "   (if this is near 1e-16 the stages are too close to distinguish)" << std::endl;

    logroot << "\n--- points per plate ID (freeze as golden data) ---" << std::endl;
    for ( const auto& [id, count] : plate_histogram )
    {
        logroot << "  plate " << id << " : " << count << std::endl;
    }

    for ( const auto* c : checks )
    {
        if ( !c->ok() )
        {
            return EXIT_FAILURE;
        }
    }

    return 0;
}
