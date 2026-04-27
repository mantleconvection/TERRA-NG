
/// Unit test for terra::communication::shell::Redistribute.
///
/// Builds a fine DistributedDomain on MPI_COMM_WORLD and a coarse DistributedDomain
/// on a factor-F sub-communicator with the agglomerated owner map. Seeds the fine
/// field from an ownership-agnostic function test_seed(diamond,x,y,r,i,j,k,c),
/// then checks TWO things:
///   (1) After `apply` (fine -> coarse), every coarse-owned subdomain holds the
///       same value the fine owner had. Validates the pack / Alltoallv / unpack
///       pipeline in the forward direction without the transpose disguising errors.
///   (2) After `apply_transpose` (coarse -> fine), every fine-owned subdomain
///       recovers its original value. Validates the full roundtrip.
///
/// CLI: --lat-level L  --lat-sdr S  --rad-sdr R  --factor F
/// Requires world_size % factor == 0.

#include <algorithm>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "communication/shell/redistribute.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/agglomerated_distribution.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/mpi/level_comms.hpp"
#include "util/init.hpp"

using namespace terra;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;

/// Ownership-agnostic seed. Keyed on the subdomain's global coordinates (diamond,
/// subdomain_x, subdomain_y, subdomain_r) plus intra-subdomain node (i,j,k) and
/// component c. Any rank that owns this subdomain — on either the fine or coarse
/// side — can compute the expected value without knowing the ownership map.
KOKKOS_INLINE_FUNCTION double
    test_seed( int diamond, int sdr_x, int sdr_y, int sdr_r, int i, int j, int k, int c )
{
    return static_cast< double >(
        ( ( ( ( ( ( diamond * 64 + sdr_x ) * 64 + sdr_y ) * 64 + sdr_r ) * 31 + i ) * 29 + j ) * 23 + k ) * 7 + c );
}

int parse_int_arg( int argc, char** argv, const char* flag, int fallback )
{
    for ( int a = 1; a + 1 < argc; ++a )
    {
        if ( std::strcmp( argv[a], flag ) == 0 )
            return std::atoi( argv[a + 1] );
    }
    return fallback;
}

/// Build a device-resident table of (diamond, x, y, r) for each local subdomain.
/// Layout: coords(local_sdr, 0..3) = {diamond, x, y, r}.
Kokkos::View< int**, Kokkos::DefaultExecutionSpace::memory_space >
    build_subdomain_coord_table( const DistributedDomain& domain )
{
    const int num_local = static_cast< int >( domain.subdomains().size() );
    Kokkos::View< int**, Kokkos::DefaultExecutionSpace::memory_space > coords(
        Kokkos::view_alloc( Kokkos::WithoutInitializing, "sdr_coords" ), std::max( num_local, 1 ), 4 );
    auto host = Kokkos::create_mirror_view( coords );
    for ( const auto& [sdr, info] : domain.subdomains() )
    {
        const int local_idx = std::get< 0 >( info );
        host( local_idx, 0 ) = sdr.diamond_id();
        host( local_idx, 1 ) = sdr.subdomain_x();
        host( local_idx, 2 ) = sdr.subdomain_y();
        host( local_idx, 3 ) = sdr.subdomain_r();
    }
    Kokkos::deep_copy( coords, host );
    return coords;
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    const int world_rank = mpi::rank();
    const int world_size = mpi::num_processes();

    const int lat_level = parse_int_arg( argc, argv, "--lat-level", 3 );
    const int lat_sdr   = parse_int_arg( argc, argv, "--lat-sdr",   3 );
    const int rad_sdr   = parse_int_arg( argc, argv, "--rad-sdr",   0 );
    const int factor    = parse_int_arg( argc, argv, "--factor",    2 );

    if ( world_size % factor != 0 )
    {
        if ( world_rank == 0 )
            std::cerr << "world_size (" << world_size << ") must be divisible by factor (" << factor << ")\n";
        return 1;
    }

    const int num_global_subdomains = 10 * ( 1 << lat_sdr ) * ( 1 << lat_sdr ) * ( 1 << rad_sdr );
    if ( num_global_subdomains < world_size )
    {
        if ( world_rank == 0 )
            std::cerr << "need at least " << world_size << " global subdomains, have "
                      << num_global_subdomains << " (bump --lat-sdr or --rad-sdr)\n";
        return 1;
    }

    if ( world_rank == 0 )
    {
        std::cout << "[test_redistribute] config:  world_size=" << world_size << "  factor=" << factor
                  << "  coarse_size=" << ( world_size / factor )
                  << "  lat_level=" << lat_level << "  lat_sdr=" << lat_sdr << "  rad_sdr=" << rad_sdr
                  << "  num_global_sdrs=" << num_global_subdomains << std::endl;
    }

    const double r_min = 0.5;
    const double r_max = 1.0;

    const auto orig_fn = grid::shell::subdomain_to_rank_iterate_diamond_subdomains;

    auto domain_fine = DistributedDomain::create_uniform(
        lat_level, /*radial_diamond_refinement_level=*/lat_level, r_min, r_max, lat_sdr, rad_sdr, orig_fn );

    auto level_comms     = mpi::build_level_comms( MPI_COMM_WORLD, { factor } );
    MPI_Comm coarse_comm = level_comms[1];

    auto coarse_fn = grid::shell::agglomerated_subdomain_to_rank( orig_fn, factor );

    auto domain_coarse = DistributedDomain::create_uniform_on_comm(
        coarse_comm,
        lat_level,
        grid::shell::uniform_shell_radii( r_min, r_max, ( 1 << lat_level ) + 1 ),
        lat_sdr,
        rad_sdr,
        coarse_fn );

    const int ni                = domain_fine.domain_info().subdomain_num_nodes_per_side_laterally();
    const int nr                = domain_fine.domain_info().subdomain_num_nodes_radially();
    const int num_local_fine    = static_cast< int >( domain_fine.subdomains().size() );
    const int num_local_coarse  = static_cast< int >( domain_coarse.subdomains().size() );
    const long per_sdr_bytes    = long( ni ) * ni * nr * 3 * sizeof( double );

    // Diagnostic: aggregate min/max/mean of local subdomain counts per domain.
    int fine_min = num_local_fine, fine_max = num_local_fine, fine_sum = num_local_fine;
    int coarse_min = num_local_coarse, coarse_max = num_local_coarse, coarse_sum = num_local_coarse;
    MPI_Allreduce( MPI_IN_PLACE, &fine_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &fine_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &fine_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &coarse_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &coarse_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( MPI_IN_PLACE, &coarse_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    // Count active coarse ranks (non-null coarse comm).
    int active_coarse = ( coarse_comm != MPI_COMM_NULL ) ? 1 : 0;
    MPI_Allreduce( MPI_IN_PLACE, &active_coarse, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

    if ( world_rank == 0 )
    {
        std::cout << "[test_redistribute] domains: "
                  << "fine min/max/total=" << fine_min << "/" << fine_max << "/" << fine_sum
                  << "   coarse min/max/total=" << coarse_min << "/" << coarse_max << "/" << coarse_sum
                  << "   active_coarse_ranks=" << active_coarse << std::endl;
        std::cout << "[test_redistribute] mesh:    ni=" << ni << "  nr=" << nr
                  << "  per_subdomain=" << ( per_sdr_bytes / 1024.0 ) << " KB"
                  << "  bytes_moved_per_apply=" << ( fine_sum * per_sdr_bytes / 1024.0 / 1024.0 ) << " MB"
                  << std::endl;
    }

    using GridT = Grid4DDataVec< double, 3 >;
    GridT data_fine( "data_fine", std::max( num_local_fine, 1 ), ni, ni, nr );
    GridT data_fine_roundtrip( "data_fine_rt", std::max( num_local_fine, 1 ), ni, ni, nr );
    GridT data_coarse( "data_coarse", std::max( num_local_coarse, 1 ), ni, ni, nr );

    auto coords_fine   = build_subdomain_coord_table( domain_fine );
    auto coords_coarse = build_subdomain_coord_table( domain_coarse );

    // Seed the fine field with test_seed keyed on global coords.
    if ( num_local_fine > 0 )
    {
        Kokkos::parallel_for(
            "init_fine",
            Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >( { 0, 0, 0, 0, 0 }, { num_local_fine, ni, ni, nr, 3 } ),
            KOKKOS_LAMBDA( int s, int i, int j, int k, int c ) {
                data_fine( s, i, j, k, c ) =
                    test_seed( coords_fine( s, 0 ), coords_fine( s, 1 ), coords_fine( s, 2 ), coords_fine( s, 3 ),
                               i, j, k, c );
            } );
        Kokkos::fence();
    }

    MPI_Barrier( MPI_COMM_WORLD );
    const double t_build_start = MPI_Wtime();

    communication::shell::Redistribute< GridT > redistribute( domain_fine, domain_coarse, orig_fn, coarse_fn );

    MPI_Barrier( MPI_COMM_WORLD );
    const double t_build_end = MPI_Wtime();

    redistribute.apply( data_fine, data_coarse );

    MPI_Barrier( MPI_COMM_WORLD );
    const double t_fwd_end = MPI_Wtime();

    // Check (1): after forward, every coarse-owned subdomain holds the value
    // originally seeded for that subdomain by its fine owner.
    double coarse_err = 0.0;
    long   coarse_checked = 0;
    if ( num_local_coarse > 0 )
    {
        Kokkos::parallel_reduce(
            "check_after_forward",
            Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >( { 0, 0, 0, 0, 0 }, { num_local_coarse, ni, ni, nr, 3 } ),
            KOKKOS_LAMBDA( int s, int i, int j, int k, int c, double& local_max ) {
                const double expected =
                    test_seed( coords_coarse( s, 0 ), coords_coarse( s, 1 ), coords_coarse( s, 2 ),
                               coords_coarse( s, 3 ), i, j, k, c );
                const double got = data_coarse( s, i, j, k, c );
                local_max        = Kokkos::max( local_max, Kokkos::abs( got - expected ) );
            },
            Kokkos::Max< double >( coarse_err ) );
        Kokkos::fence();
        coarse_checked = long( num_local_coarse ) * ni * ni * nr * 3;
    }
    double global_coarse_err = 0.0;
    long   global_coarse_dofs = 0;
    MPI_Allreduce( &coarse_err, &global_coarse_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &coarse_checked, &global_coarse_dofs, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD );

    redistribute.apply_transpose( data_coarse, data_fine_roundtrip );

    MPI_Barrier( MPI_COMM_WORLD );
    const double t_bwd_end = MPI_Wtime();

    // Check (2): after transpose, every fine-owned subdomain recovers its original value.
    double roundtrip_err     = 0.0;
    long   roundtrip_checked = 0;
    if ( num_local_fine > 0 )
    {
        Kokkos::parallel_reduce(
            "check_roundtrip",
            Kokkos::MDRangePolicy< Kokkos::Rank< 5 > >( { 0, 0, 0, 0, 0 }, { num_local_fine, ni, ni, nr, 3 } ),
            KOKKOS_LAMBDA( int s, int i, int j, int k, int c, double& local_max ) {
                const double expected =
                    test_seed( coords_fine( s, 0 ), coords_fine( s, 1 ), coords_fine( s, 2 ), coords_fine( s, 3 ),
                               i, j, k, c );
                const double got = data_fine_roundtrip( s, i, j, k, c );
                local_max        = Kokkos::max( local_max, Kokkos::abs( got - expected ) );
            },
            Kokkos::Max< double >( roundtrip_err ) );
        Kokkos::fence();
        roundtrip_checked = long( num_local_fine ) * ni * ni * nr * 3;
    }
    double global_roundtrip_err = 0.0;
    long   global_roundtrip_dofs = 0;
    MPI_Allreduce( &roundtrip_err, &global_roundtrip_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD );
    MPI_Allreduce( &roundtrip_checked, &global_roundtrip_dofs, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD );

    if ( world_rank == 0 )
    {
        const double t_build  = ( t_build_end - t_build_start ) * 1e3;
        const double t_fwd    = ( t_fwd_end   - t_build_end )   * 1e3;
        const double t_bwd    = ( t_bwd_end   - t_fwd_end )     * 1e3;
        const double fwd_bw   = ( fine_sum * per_sdr_bytes ) / ( t_fwd / 1e3 ) / ( 1ull << 30 );
        const double bwd_bw   = ( fine_sum * per_sdr_bytes ) / ( t_bwd / 1e3 ) / ( 1ull << 30 );

        std::cout << "[test_redistribute] timings:"
                  << "  build_plan=" << t_build << " ms"
                  << "  fwd=" << t_fwd << " ms (" << fwd_bw << " GB/s aggregate)"
                  << "  transpose=" << t_bwd << " ms (" << bwd_bw << " GB/s aggregate)"
                  << std::endl;

        std::cout << "[test_redistribute] checks:"
                  << "  after-forward:   " << global_coarse_dofs << " DoFs  max_err=" << global_coarse_err << "\n"
                  << "[test_redistribute]        "
                  << "  after-transpose: " << global_roundtrip_dofs << " DoFs  max_err=" << global_roundtrip_err
                  << std::endl;
    }

    const bool pass_fwd = ( global_coarse_err == 0.0 );
    const bool pass_rt  = ( global_roundtrip_err == 0.0 );
    const bool pass     = pass_fwd && pass_rt;

    if ( world_rank == 0 )
    {
        std::cout << ( pass_fwd ? "[ OK ]" : "[FAIL]" ) << " after forward apply"        << "\n"
                  << ( pass_rt  ? "[ OK ]" : "[FAIL]" ) << " after apply_transpose"      << "\n"
                  << ( pass     ? "[ OK ]" : "[FAIL]" ) << " redistribute test overall" << std::endl;
    }

    mpi::free_level_comms( level_comms );

    return pass ? 0 : 1;
}
