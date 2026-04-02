
#include <kernels/common/grid_operations.hpp>

#include "fe/wedge/operators/shell/epsilon_divdiv.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_simple.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_kerngen.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v01_initial.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v02_split_dimij.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v02b_single_quadpoint.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v03_teams_precomp.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v04_shmem_coords.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v05_shmem_src_k.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v06_xy_tiling.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v07_split_paths.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v08_scalar_coalesced.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v09_separate_scatter.hpp"
#include "fe/wedge/operators/shell/performance_history/epsilon_divdiv_kerngen_v10_seq_rpasses.hpp"
#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
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

using namespace terra;

using fe::wedge::operators::shell::EpsDivDivStokes;
using fe::wedge::operators::shell::EpsilonDivDiv;
using fe::wedge::operators::shell::EpsilonDivDivSimple;
using fe::wedge::operators::shell::EpsilonDivDivKerngen;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV01Initial;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV02SplitDimij;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV02bSingleQuadpoint;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV03TeamsPrecomp;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV04ShmemCoords;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV05ShmemSrcK;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV06XyTiling;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV07SplitPaths;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV08ScalarCoalesced;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV09SeparateScatter;
using fe::wedge::operators::shell::epsdivdiv_history::EpsilonDivDivKerngenV10SeqRpasses;
using fe::wedge::operators::shell::Laplace;
using fe::wedge::operators::shell::LaplaceSimple;
using fe::wedge::operators::shell::Stokes;
using fe::wedge::operators::shell::VectorLaplaceSimple;
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
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using terra::grid::shell::BoundaryConditions;
using util::logroot;

enum class BenchmarkType : int
{
    LaplaceFloat,
    LaplaceDouble,
    LaplaceSimpleDouble,
    VectorLaplaceFloat,
    VectorLaplaceDouble,
    VectorLaplaceNeumannDouble,
    EpsDivDivSimpleDouble,
    EpsDivDivFloat,
    EpsDivDivDouble,
    EpsDivDivKerngenDouble,
    EpsDivDivKerngenV01Initial,
    EpsDivDivKerngenV02SplitDimij,
    EpsDivDivKerngenV02bSingleQuadpoint,
    EpsDivDivKerngenV03TeamsPrecomp,
    EpsDivDivKerngenV04ShmemCoords,
    EpsDivDivKerngenV05ShmemSrcK,
    EpsDivDivKerngenV06XyTiling,
    EpsDivDivKerngenV07SplitPaths,
    EpsDivDivKerngenV08ScalarCoalesced,
    EpsDivDivKerngenV09SeparateScatter,
    EpsDivDivKerngenV10SeqRpasses,
    StokesDouble,
    EpsDivDivStokesDouble
};

constexpr auto all_benchmark_types = {
    BenchmarkType::EpsDivDivSimpleDouble,
    BenchmarkType::EpsDivDivDouble,
    BenchmarkType::EpsDivDivKerngenV01Initial,
    BenchmarkType::EpsDivDivKerngenV02SplitDimij,
    BenchmarkType::EpsDivDivKerngenV02bSingleQuadpoint,
    BenchmarkType::EpsDivDivKerngenV07SplitPaths,
    BenchmarkType::EpsDivDivKerngenV09SeparateScatter,
};

const std::map< BenchmarkType, std::string > benchmark_description = {
    { BenchmarkType::LaplaceFloat, "Laplace (float)" },
    { BenchmarkType::LaplaceSimpleDouble, "LaplaceSimple (double)" },
    { BenchmarkType::LaplaceDouble, "Laplace (double)" },
    { BenchmarkType::VectorLaplaceFloat, "VectorLaplace (float)" },
    { BenchmarkType::VectorLaplaceDouble, "VectorLaplace (double)" },
    { BenchmarkType::VectorLaplaceNeumannDouble, "VectorLaplaceNeumann (double)" },
    { BenchmarkType::EpsDivDivSimpleDouble, "EpsDivDivSimple (double, naive baseline)" },
    { BenchmarkType::EpsDivDivFloat, "EpsDivDiv (float)" },
    { BenchmarkType::EpsDivDivDouble, "EpsDivDiv (double, fused matvec)" },
    { BenchmarkType::EpsDivDivKerngenDouble, "EpsDivDivKerngen (double)" },
    { BenchmarkType::EpsDivDivKerngenV01Initial, "v01 initial (1t/cell, 6qp, 3x3 dimij)" },
    { BenchmarkType::EpsDivDivKerngenV02SplitDimij, "v02 split dimij (2x3 complexity)" },
    { BenchmarkType::EpsDivDivKerngenV02bSingleQuadpoint, "v02b single quadpoint (1qp)" },
    { BenchmarkType::EpsDivDivKerngenV03TeamsPrecomp, "v03 teams + precomputation" },
    { BenchmarkType::EpsDivDivKerngenV04ShmemCoords, "v04 shmem coords, collapsed qp" },
    { BenchmarkType::EpsDivDivKerngenV05ShmemSrcK, "v05 shmem src + k dofs" },
    { BenchmarkType::EpsDivDivKerngenV06XyTiling, "v06 xy tiling" },
    { BenchmarkType::EpsDivDivKerngenV07SplitPaths, "v07 split fast/slow paths" },
    { BenchmarkType::EpsDivDivKerngenV08ScalarCoalesced, "v08 scalar coalesced access" },
    { BenchmarkType::EpsDivDivKerngenV09SeparateScatter, "v09 separate scatter (7.6 Gdofs)" },
    { BenchmarkType::EpsDivDivKerngenV10SeqRpasses, "v10 seq r_passes (7.8 Gdofs)" },
    { BenchmarkType::StokesDouble, "Stokes (double)" },
    { BenchmarkType::EpsDivDivStokesDouble, "EpsDivDivStokes (double)" } };

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

    // Ensure stuff is not optimized out?!
    // const auto mm = kernels::common::max_abs_entry( dst.grid_data() );
    // std::cout << "Printing some derived value to ensure nothing is optimized out: " << mm << std::endl;
    MPI_Barrier( MPI_COMM_WORLD );
    double duration     = timer.seconds() / executions;
    double duration_max = 0.0;
    MPI_Allreduce( &duration, &duration_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    return duration_max;
}

BenchmarkData
    run( const BenchmarkType benchmark, const int level, const int executions, const int refinement_level_subdomains )
{
    if ( level < 1 )
    {
        Kokkos::abort( "level must be >= 1" );
    }

    const auto domain = grid::shell::DistributedDomain::create_uniform(
        level, level, 0.5, 1.0, refinement_level_subdomains, refinement_level_subdomains );
    const auto subdomain_distr = grid::shell::subdomain_distribution( domain );
    logroot << "Subdomain distribution: \n";
    logroot << " - total: " << subdomain_distr.total << "\n";
    logroot << " - min:   " << subdomain_distr.min << "\n";
    logroot << " - avg:   " << subdomain_distr.avg << "\n";
    logroot << " - max:   " << subdomain_distr.max << "\n\n";

    const auto domain_coarse = grid::shell::DistributedDomain::create_uniform(
        level - 1, level - 1, 0.5, 1.0, refinement_level_subdomains, refinement_level_subdomains );

    const auto coords_shell_double = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto coords_radii_double = grid::shell::subdomain_shell_radii< double >( domain );

    const auto coords_shell_float = grid::shell::subdomain_unit_sphere_single_shell_coords< float >( domain );
    const auto coords_radii_float = grid::shell::subdomain_shell_radii< float >( domain );

    const auto coords_shell_coarse_double =
        grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain_coarse );
    const auto coords_radii_coarse_double = grid::shell::subdomain_shell_radii< double >( domain_coarse );

    const auto coords_shell_coarse_float =
        grid::shell::subdomain_unit_sphere_single_shell_coords< float >( domain_coarse );
    const auto coords_radii_coarse_float = grid::shell::subdomain_shell_radii< float >( domain_coarse );

    auto mask_data        = grid::setup_node_ownership_mask_data( domain );
    auto mask_data_coarse = grid::setup_node_ownership_mask_data( domain_coarse );

    auto boundary_mask_data = grid::shell::setup_boundary_mask_data( domain );

    const auto dofs_scalar = kernels::common::count_masked< long >( mask_data, grid::NodeOwnershipFlag::OWNED );
    const auto dofs_vec    = 3 * dofs_scalar;
    const auto dofs_scalar_coarse =
        kernels::common::count_masked< long >( mask_data_coarse, grid::NodeOwnershipFlag::OWNED );
    const auto dofs_stokes = dofs_vec + dofs_scalar_coarse;

    VectorQ1Scalar< double > src_scalar_double( "src_scalar_double", domain, mask_data );
    VectorQ1Scalar< double > dst_scalar_double( "dst_scalar_double", domain, mask_data );

    VectorQ1Scalar< float > src_scalar_float( "src_scalar_float", domain, mask_data );
    VectorQ1Scalar< float > dst_scalar_float( "dst_scalar_float", domain, mask_data );

    VectorQ1Vec< double > src_vec_double( "src_vec_double", domain, mask_data );
    VectorQ1Vec< double > dst_vec_double( "dst_vec_double", domain, mask_data );

    VectorQ1Vec< float > src_vec_float( "src_vec_float", domain, mask_data );
    VectorQ1Vec< float > dst_vec_float( "dst_vec_float", domain, mask_data );

    VectorQ1IsoQ2Q1< double > src_stokes_double(
        "src_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );
    VectorQ1IsoQ2Q1< double > dst_stokes_double(
        "dst_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );

    VectorQ1IsoQ2Q1< float > src_stokes_float(
        "src_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );
    VectorQ1IsoQ2Q1< float > dst_stokes_float(
        "dst_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );

    VectorQ1Scalar< double > coeff_double( "coeff_double", domain, mask_data );
    VectorQ1Scalar< float >  coeff_float( "coeff_float", domain, mask_data );

    linalg::assign( coeff_double, 1.0 );
    linalg::assign( coeff_float, 1.0 );

    linalg::randomize( src_scalar_double );
    linalg::randomize( src_scalar_float );
    linalg::randomize( src_vec_double );
    linalg::randomize( src_vec_float );
    linalg::randomize( src_stokes_double );
    linalg::randomize( src_stokes_float );
    BoundaryConditions bcs = {
        { CMB, DIRICHLET },
        { SURFACE, DIRICHLET },
    };
    double duration = 0.0;
    long   dofs     = 0;
    if ( benchmark == BenchmarkType::LaplaceFloat )
    {
        LaplaceSimple< float > A( domain, coords_shell_float, coords_radii_float, true, false );
        duration = measure_run_time( executions, A, src_scalar_float, dst_scalar_float );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::LaplaceSimpleDouble )
    {
        LaplaceSimple< double > A( domain, coords_shell_double, coords_radii_double, true, false );
        util::Timer             t( "Laplace - double" );
        duration = measure_run_time( executions, A, src_scalar_double, dst_scalar_double );
        dofs     = dofs_scalar;
    }
    else if ( benchmark == BenchmarkType::LaplaceDouble )
    {
        Laplace< double > A( domain, coords_shell_double, coords_radii_double, boundary_mask_data, true, false );
        util::Timer       t( "Laplace - double" );
        duration = measure_run_time( executions, A, src_scalar_double, dst_scalar_double );
        dofs     = dofs_scalar;
    }
    else if ( benchmark == BenchmarkType::VectorLaplaceFloat )
    {
        VectorLaplaceSimple< float > A( domain, coords_shell_float, coords_radii_float, true, false );
        duration = measure_run_time( executions, A, src_vec_float, dst_vec_float );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::VectorLaplaceDouble )
    {
        VectorLaplaceSimple< double > A( domain, coords_shell_double, coords_radii_double, true, false );
        util::Timer                   t( "VectorLaplace - double" );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::VectorLaplaceNeumannDouble )
    {
        VectorLaplaceSimple< double > A( domain, coords_shell_double, coords_radii_double, false, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivSimpleDouble )
    {
        EpsilonDivDivSimple< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), true, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivFloat )
    {
        EpsilonDivDiv A(
            domain, coords_shell_float, coords_radii_float, boundary_mask_data, coeff_float.grid_data(), true, false );
        util::Timer t( "EpsDivDiv - float" );
        duration = measure_run_time( executions, A, src_vec_float, dst_vec_float );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivDouble )
    {
        EpsilonDivDiv A(
            domain,
            coords_shell_double,
            coords_radii_double,
            boundary_mask_data,
            coeff_double.grid_data(),
            true,
            false );
        util::Timer t( "EpsDivDiv - double" );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenDouble )
    {
        EpsilonDivDivKerngen A(
            domain,
            coords_shell_double,
            coords_radii_double,
            boundary_mask_data,
            coeff_double.grid_data(),
            bcs,
            false );
        util::Timer t( "EpsDivDivKerngen - double" );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV01Initial )
    {
        EpsilonDivDivKerngenV01Initial< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV02SplitDimij )
    {
        EpsilonDivDivKerngenV02SplitDimij< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV02bSingleQuadpoint )
    {
        EpsilonDivDivKerngenV02bSingleQuadpoint< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV03TeamsPrecomp )
    {
        EpsilonDivDivKerngenV03TeamsPrecomp< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV04ShmemCoords )
    {
        EpsilonDivDivKerngenV04ShmemCoords< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV05ShmemSrcK )
    {
        EpsilonDivDivKerngenV05ShmemSrcK< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV06XyTiling )
    {
        EpsilonDivDivKerngenV06XyTiling< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV07SplitPaths )
    {
        EpsilonDivDivKerngenV07SplitPaths< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV08ScalarCoalesced )
    {
        EpsilonDivDivKerngenV08ScalarCoalesced< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV09SeparateScatter )
    {
        EpsilonDivDivKerngenV09SeparateScatter< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivKerngenV10SeqRpasses )
    {
        EpsilonDivDivKerngenV10SeqRpasses< double > A(
            domain, coords_shell_double, coords_radii_double, boundary_mask_data,
            coeff_double.grid_data(), bcs, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::StokesDouble )
    {
        Stokes< double > A(
            domain, domain_coarse, coords_shell_double, coords_radii_double, boundary_mask_data, bcs, false );
        duration = measure_run_time( executions, A, src_stokes_double, dst_stokes_double );
        dofs     = dofs_stokes;
    }
    else if ( benchmark == BenchmarkType::EpsDivDivStokesDouble )
    {
        EpsDivDivStokes< double > A(
            domain,
            domain_coarse,
            coords_shell_double,
            coords_radii_double,
            boundary_mask_data,
            coeff_double.grid_data(),
            bcs,
            false );
        duration = measure_run_time( executions, A, src_stokes_double, dst_stokes_double );
        dofs     = dofs_stokes;
    }
    else
    {
        Kokkos::abort( "Unknown benchmark type" );
    }

    return BenchmarkData{ level, dofs, duration };
}

void run_all( const int min_level, const int max_level, const int executions, const int refinement_level_subdomains )
{
    logroot << "Running operator (matvec) benchmarks." << std::endl;
    logroot << "min_level:            " << min_level << std::endl;
    logroot << "max_level:            " << max_level << std::endl;
    logroot << "executions per level: " << executions << std::endl;
    logroot << "refinement for subdomains " << refinement_level_subdomains << std::endl;
    logroot << std::endl;
    int world_size = 0;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size ); // total number of MPI processes

    for ( auto benchmark : all_benchmark_types )
    {
        logroot << benchmark_description.at( benchmark ) << std::endl;

        util::Table table;

        for ( int i = min_level; i <= max_level; ++i )
        {
            const auto data = run( benchmark, i, executions, refinement_level_subdomains );
            table.add_row(
                { { "level", i },
                  { "dofs", data.dofs },
                  { "duration (s)", data.duration },
                  { "updated dofs/sec", data.dofs / data.duration } } );
        }

        table.print_pretty();

        // output a csv table of results
        if ( mpi::rank() == 0 )
        {
            std::ofstream out(
                "./csv/bo_np" + std::to_string( world_size ) + "_sdr" + std::to_string( refinement_level_subdomains ) +
                "_ml" + std::to_string( max_level ) + ".csv" );
            table.print_csv( out );
        }
        table.print_csv( logroot );

        logroot << std::endl;
        logroot << std::endl;
    }

    util::TimerTree::instance().aggregate_mpi();
    if ( mpi::rank() == 0 )
    {
        std::ofstream out(
            "./tts/bo_np" + std::to_string( world_size ) + "_sdr" + std::to_string( refinement_level_subdomains ) +
            "_ml" + std::to_string( max_level ) + ".json" );
        out << util::TimerTree::instance().json_aggregate();
        out.close();
    }
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
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
        "Refinement level applied to form the subdomains." );
    util::add_option_with_default(
        app, "--executions", parameters.executions, "Number of matrix-vector multiplications to be executed." );

    CLI11_PARSE( app, argc, argv );

    if ( parameters.min_level < 1 )
    {
        logroot << "Error: min-level must be >= 1." << std::endl;
        return 1;
    }

    logroot << "\n" << description << "\n\n";

    util::print_cli_summary( app, logroot );
    logroot << "\n\n";

    run_all(
        parameters.min_level, parameters.max_level, parameters.executions, parameters.refinement_level_subdomains );

    MPI_Finalize();
}
