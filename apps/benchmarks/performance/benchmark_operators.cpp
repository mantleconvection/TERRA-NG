
#include <kernels/common/grid_operations.hpp>

#include <cstdlib>

#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/table.hpp"
#ifdef TERRANEO_USE_NESMIK
#include <nesmik/nesmik.hpp>
#endif

using namespace terra;

using fe::wedge::operators::shell::EpsilonDivDivKerngen;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::BOUNDARY;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using linalg::apply;
using linalg::DstOf;
using linalg::OperatorLike;
using linalg::SrcOf;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using terra::grid::shell::BoundaryConditions;
using util::logroot;

struct BenchmarkData
{
    int    level;
    long   dofs;
    double duration;
};

struct Parameters
{
    int min_level                   = 1;
    int max_level                   = 6;
    int executions                  = 5;
    int refinement_level_subdomains = 0;

    // Per-axis overrides. -1 means "not set: fall back to the scalar above".
    // When set, these directly control create_uniform's per-axis arguments.
    int lat_level = -1;
    int rad_level = -1;
    int lat_sdr   = -1;
    int rad_sdr   = -1;
};

template < OperatorLike OperatorT >
double measure_run_time( int executions, OperatorT& A, const SrcOf< OperatorT >& src, DstOf< OperatorT >& dst )
{
    Kokkos::Timer timer;

    Kokkos::fence();
    MPI_Barrier( MPI_COMM_WORLD );
    timer.reset();

    for ( int i = 0; i < executions; ++i )
    {
        apply( A, src, dst );
    }

    Kokkos::fence();

    MPI_Barrier( MPI_COMM_WORLD );
    double duration     = timer.seconds() / executions;
    double duration_max = 0.0;
    MPI_Allreduce( &duration, &duration_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return duration_max;
}

BenchmarkData
    run( const int lat_level, const int rad_level, const int executions, const int lat_sdr, const int rad_sdr )
{
    if ( lat_level < 1 || rad_level < 1 )
    {
        Kokkos::abort( "lateral and radial levels must be >= 1" );
    }

    const auto domain = grid::shell::DistributedDomain::create_uniform(
        lat_level, rad_level, 0.5, 1.0, lat_sdr, rad_sdr );
    const auto subdomain_distr = grid::shell::subdomain_distribution( domain );
    logroot << "Subdomain distribution: \n";
    logroot << " - total: " << subdomain_distr.total << "\n";
    logroot << " - min:   " << subdomain_distr.min << "\n";
    logroot << " - avg:   " << subdomain_distr.avg << "\n";
    logroot << " - max:   " << subdomain_distr.max << "\n\n";

    const auto coords_shell_double = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto coords_radii_double = grid::shell::subdomain_shell_radii< double >( domain );

    auto mask_data          = grid::setup_node_ownership_mask_data( domain );
    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto dofs_scalar = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    const auto dofs_vec    = 3 * dofs_scalar;

    VectorQ1Vec< double > src_vec_double( "src_vec_double", domain, mask_data );
    VectorQ1Vec< double > dst_vec_double( "dst_vec_double", domain, mask_data );

    VectorQ1Scalar< double > coeff_double( "coeff_double", domain, mask_data );

    linalg::assign( coeff_double, 1.0 );
    linalg::randomize( src_vec_double );

    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };

    EpsilonDivDivKerngen A(
        domain,
        coords_shell_double,
        coords_radii_double,
        boundary_mask_data,
        coeff_double.grid_data(),
        bcs,
        false );
    // Default DN path. On HIP/AMD, env vars opt into experimental paths:
    //   EPSDIVDIV_HEX=1  → hex 2x2x2 Gauss path
    //   EPSDIVDIV_WAVE=1 → wave-parallel wedge path
#ifdef __HIP_PLATFORM_AMD__
    if ( std::getenv( "EPSDIVDIV_HEX" ) != nullptr )
    {
        A.set_kernel_path( decltype( A )::KernelPath::FastDirichletNeumannHex );
    }
    else if ( std::getenv( "EPSDIVDIV_WAVE" ) != nullptr )
    {
        A.set_kernel_path( decltype( A )::KernelPath::FastDirichletNeumannWave );
    }
    else
#endif
    {
        A.set_kernel_path( decltype( A )::KernelPath::FastDirichletNeumann );
    }
    util::Timer t( "EpsDivDivKerngen - double" );
    double      duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
    long        dofs     = dofs_vec;

    return BenchmarkData{ lat_level, dofs, duration };
}

static std::string axis_tag( const std::string& prefix, int lat, int rad )
{
    if ( lat == rad )
        return prefix + std::to_string( lat );
    return prefix + std::to_string( lat ) + "x" + std::to_string( rad );
}

void run_all( const Parameters& p )
{
    logroot << "Running operator (matvec) benchmarks." << std::endl;
    logroot << "min_level:            " << p.min_level << std::endl;
    logroot << "max_level:            " << p.max_level << std::endl;
    logroot << "executions per level: " << p.executions << std::endl;
    logroot << "refinement for subdomains " << p.refinement_level_subdomains << std::endl;
    if ( p.lat_level >= 0 || p.rad_level >= 0 )
        logroot << "lat/rad level overrides: lat=" << p.lat_level << " rad=" << p.rad_level << std::endl;
    if ( p.lat_sdr >= 0 || p.rad_sdr >= 0 )
        logroot << "lat/rad sdr overrides:   lat=" << p.lat_sdr << " rad=" << p.rad_sdr << std::endl;
    logroot << std::endl;
    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    logroot << "EpsDivDivKerngen (double)" << std::endl;

    util::Table table;

    // Track last-used per-axis values so the output filenames reflect the actual
    // configuration (important when overrides fix one axis while the scalar loop varies the other).
    int last_lat_level = p.max_level;
    int last_rad_level = p.max_level;
    int last_lat_sdr   = p.refinement_level_subdomains;
    int last_rad_sdr   = p.refinement_level_subdomains;

    for ( int i = p.min_level; i <= p.max_level; ++i )
    {
        const int lat_lvl = ( p.lat_level >= 0 ) ? p.lat_level : i;
        const int rad_lvl = ( p.rad_level >= 0 ) ? p.rad_level : i;
        const int lat_sdr = ( p.lat_sdr   >= 0 ) ? p.lat_sdr   : p.refinement_level_subdomains;
        const int rad_sdr = ( p.rad_sdr   >= 0 ) ? p.rad_sdr   : p.refinement_level_subdomains;

        const auto data = run( lat_lvl, rad_lvl, p.executions, lat_sdr, rad_sdr );
        table.add_row(
            { { "lat_level", lat_lvl },
              { "rad_level", rad_lvl },
              { "dofs", data.dofs },
              { "duration (s)", data.duration },
              { "updated dofs/sec", data.dofs / data.duration } } );

        last_lat_level = lat_lvl;
        last_rad_level = rad_lvl;
        last_lat_sdr   = lat_sdr;
        last_rad_sdr   = rad_sdr;
    }

    table.print_pretty();

    const std::string suffix = "_np" + std::to_string( world_size ) + "_" +
                               axis_tag( "sdr", last_lat_sdr, last_rad_sdr ) + "_" +
                               axis_tag( "ml", last_lat_level, last_rad_level );

    if ( mpi::rank() == 0 )
    {
        std::ofstream out( "./csv/bo" + suffix + ".csv" );
        table.print_csv( out );
    }
    table.print_csv( logroot );

    logroot << std::endl;
    logroot << std::endl;

    util::TimerTree::instance().aggregate_mpi();
    if ( mpi::rank() == 0 )
    {
        std::ofstream out( "./tts/bo" + suffix + ".json" );
        out << util::TimerTree::instance().json_aggregate();
        out.close();
    }
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    #ifdef TERRANEO_USE_NESMIK
    nesmik::init();
    #endif
    Kokkos::ScopeGuard scope_guard( argc, argv );

    util::print_general_info( argc, argv );

    const auto description =
        "Operator benchmark. Runs a couple of matrix-vector multiplications for various operators to get an idea of the throughput.";
    CLI::App app{ description };

    Parameters parameters{};

    util::add_option_with_default( app, "--min-level", parameters.min_level, "Min refinement level." );
    util::add_option_with_default( app, "--max-level", parameters.max_level, "Max refinement level." );
    util::add_option_with_default(
        app,
        "--refinement-level-subdomains",
        parameters.refinement_level_subdomains,
        "Refinement level applied to form the subdomains (applied to both axes unless per-axis overrides are set)." );
    util::add_option_with_default(
        app, "--executions", parameters.executions, "Number of matrix-vector multiplications to be executed." );
    util::add_option_with_default(
        app,
        "--lat-level",
        parameters.lat_level,
        "Override the lateral diamond refinement level (otherwise the loop level is used)." );
    util::add_option_with_default(
        app,
        "--rad-level",
        parameters.rad_level,
        "Override the radial diamond refinement level (otherwise the loop level is used)." );
    util::add_option_with_default(
        app,
        "--lat-sdr",
        parameters.lat_sdr,
        "Override the lateral subdomain refinement level." );
    util::add_option_with_default(
        app,
        "--rad-sdr",
        parameters.rad_sdr,
        "Override the radial subdomain refinement level." );

    CLI11_PARSE( app, argc, argv );

    if ( parameters.min_level < 1 )
    {
        logroot << "Error: min-level must be >= 1." << std::endl;
        return 1;
    }

    logroot << "\n" << description << "\n\n";

    util::print_cli_summary( app, logroot );
    logroot << "\n\n";

    run_all( parameters );

    #ifdef TERRANEO_USE_NESMIK
    nesmik::finalize();
    #endif
    MPI_Finalize();
}
