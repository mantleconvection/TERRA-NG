#pragma once

#include <numeric>

#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_q1.hpp"
#include "util/interpolation.hpp"
#include "util/table.hpp"

namespace terra::shell {

/// @brief Simple struct, holding device-views for radial profiles.
///
/// See `radial_profiles()` and `radial_profiles_to_table()` for radial profiles computation.
template < typename ScalarType >
struct RadialProfiles
{
    explicit RadialProfiles( int radial_shells )
    : radial_min_( "radial_profiles_min", radial_shells )
    , radial_max_( "radial_profiles_max", radial_shells )
    , radial_sum_( "radial_profiles_sum", radial_shells )
    , radial_cnt_( "radial_profiles_cnt", radial_shells )
    , radial_avg_( "radial_profiles_avg", radial_shells )
    {}

    grid::Grid1DDataScalar< ScalarType > radial_min_;
    grid::Grid1DDataScalar< ScalarType > radial_max_;
    grid::Grid1DDataScalar< ScalarType > radial_sum_;
    grid::Grid1DDataScalar< ScalarType > radial_cnt_;
    grid::Grid1DDataScalar< ScalarType > radial_avg_;
};

/// @brief Compute radial profiles (min, max, sum, count, average) for a Q1 scalar field (on device!).
///
/// @details Computes the radial profiles for a Q1 scalar field by iterating over all radial shells.
/// The profiles include minimum, maximum, sum, and count of nodes in each shell.
/// The radial shells are defined by the radial dimension of the Q1 scalar field.
/// The output is a RadialProfiles struct with device-side arrays of size num_shells which contain:
/// - Minimum value in the shell
/// - Maximum value in the shell
/// - Sum of values in the shell
/// - Count of nodes in the shell
/// - Average of nodes in the shell (sum / count)
/// Performs reduction per process on the device and also inter-device reduction with MPI_Allreduce.
/// All processes will carry the same result.
///
/// Mask data is used internally to filter out non-owned nodes. So this really only reduces owned nodes.
///
/// @note The returned Views are still on the device.
///       To convert it to a util::Table for output, use the `radial_profiles_to_table()` function.
///       So unless you really want to compute any further data on the device using the profile, you like always want
///       to call the `radial_profiles_to_table()` function as outlined below.
///
/// To nicely format the output, use the `radial_profiles_to_table()` function, e.g., via
/// @code
/// auto radii = domain_info.radii(); // the DomainInfo is also available in the DistributedDomain
/// auto table = radial_profiles_to_table( radial_profiles( some_field_like_temperature ), radii );
/// std::ofstream out( "radial_profiles.csv" );
/// table.print_csv( out );
/// @endcode
///
/// @tparam ScalarType Scalar type of the field.
/// @param data Q1 scalar field data.
/// @param subdomain_shell_idx global shell indices array with layout
///                            @code subdomain_shell_idx( local_subdomain_id, node_r_idx ) = global_shell_idx @endcode
///                            compute with \ref terra::grid::shell::subdomain_shell_idx.
/// @param num_global_shells number of global shells
/// @return RadialProfiles struct containing [min, max, sum, count, average] for each radial shell (still on the device).
template < typename ScalarType >
RadialProfiles< ScalarType > radial_profiles(
    const linalg::VectorQ1Scalar< ScalarType >& data,
    const grid::Grid2DDataScalar< int >&        subdomain_shell_idx,
    const int                                   num_global_shells )
{
    RadialProfiles< ScalarType > radial_profiles( num_global_shells );

    const auto data_grid = data.grid_data();
    const auto data_mask = data.mask_data();

    if ( data_grid.extent( 0 ) != subdomain_shell_idx.extent( 0 ) ||
         data_grid.extent( 3 ) != subdomain_shell_idx.extent( 1 ) )
    {
        Kokkos::abort( "radial_profiles: Data and subdomain_shell_idx do not have matching dimensions." );
    }

    Kokkos::parallel_for(
        "radial profiles init", num_global_shells, KOKKOS_LAMBDA( int r ) {
            radial_profiles.radial_min_( r ) = Kokkos::Experimental::finite_max_v< ScalarType >;
            radial_profiles.radial_max_( r ) = Kokkos::Experimental::finite_min_v< ScalarType >;
            radial_profiles.radial_sum_( r ) = 0;
            radial_profiles.radial_cnt_( r ) = 0;
        } );

    Kokkos::fence();

    Kokkos::parallel_for(
        "radial profiles reduction",
        Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
            { 0, 0, 0, 0 },
            { data_grid.extent( 0 ), data_grid.extent( 1 ), data_grid.extent( 2 ), data_grid.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain_id, int x, int y, int r ) {
            if ( !util::has_flag( data_mask( local_subdomain_id, x, y, r ), grid::NodeOwnershipFlag::OWNED ) )
            {
                return;
            }

            const int global_shell_idx = subdomain_shell_idx( local_subdomain_id, r );

            Kokkos::atomic_min(
                &radial_profiles.radial_min_( global_shell_idx ), data_grid( local_subdomain_id, x, y, r ) );
            Kokkos::atomic_max(
                &radial_profiles.radial_max_( global_shell_idx ), data_grid( local_subdomain_id, x, y, r ) );
            Kokkos::atomic_add(
                &radial_profiles.radial_sum_( global_shell_idx ), data_grid( local_subdomain_id, x, y, r ) );
            Kokkos::atomic_add( &radial_profiles.radial_cnt_( global_shell_idx ), static_cast< ScalarType >( 1 ) );
        } );

    Kokkos::fence();

    // Perform MPI_Allreduce on host to avoid GPU-aware MPI issues with MPI_IN_PLACE
    // at large rank counts. The buffers are small (num_global_shells doubles), so the
    // device-to-host copy overhead is negligible.
    {
        auto host_min = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_min_ );
        auto host_max = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_max_ );
        auto host_sum = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_sum_ );
        auto host_cnt = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_cnt_ );

        MPI_Allreduce(
            MPI_IN_PLACE, host_min.data(), host_min.size(), mpi::mpi_datatype< ScalarType >(), MPI_MIN, MPI_COMM_WORLD );
        MPI_Allreduce(
            MPI_IN_PLACE, host_max.data(), host_max.size(), mpi::mpi_datatype< ScalarType >(), MPI_MAX, MPI_COMM_WORLD );
        MPI_Allreduce(
            MPI_IN_PLACE, host_sum.data(), host_sum.size(), mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );
        MPI_Allreduce(
            MPI_IN_PLACE, host_cnt.data(), host_cnt.size(), mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );

        Kokkos::deep_copy( radial_profiles.radial_min_, host_min );
        Kokkos::deep_copy( radial_profiles.radial_max_, host_max );
        Kokkos::deep_copy( radial_profiles.radial_sum_, host_sum );
        Kokkos::deep_copy( radial_profiles.radial_cnt_, host_cnt );
    }

    Kokkos::parallel_for(
        "radial profiles avg", num_global_shells, KOKKOS_LAMBDA( int r ) {
            radial_profiles.radial_avg_( r ) = radial_profiles.radial_sum_( r ) / radial_profiles.radial_cnt_( r );
        } );

    Kokkos::fence();

    return radial_profiles;
}

/// @brief Convert radial profile data to a util::Table for analysis or output.
///
/// @details Converts the radial profile data (min, max, sum, avg, count) into a util::Table.
///
/// This table can then be used for further analysis or output to CSV/JSON.
/// The table will have the following columns:
/// - tag: "radial_profiles"
/// - shell_idx: Index of the radial shell
/// - radius: Radius of the shell
/// - min: Minimum value in the shell
/// - max: Maximum value in the shell
/// - sum: Average value in the shell
/// - avg: Average value in the shell
/// - cnt: Count of nodes in the shell
///
/// To use this function, you can compute the radial profiles using `radial_profiles()` and then convert
/// the result to a table using this function:
///
/// @code
/// auto radii = domain_info.radii(); // the DomainInfo is also available in the DistributedDomain
/// auto table = radial_profiles_to_table( radial_profiles( some_field_like_temperature ), radii );
/// std::ofstream out( "radial_profiles.csv" );
/// table.print_csv( out );
/// @endcode
///
/// @tparam ScalarType Scalar type of the profile data.
/// @param radial_profiles RadialProfiles struct containing radial profile statistics.
///                        Compute this using `radial_profiles()` function. Data is expected to be on the device still.
///                        It is copied to host for table creation in this function.
/// @param radii Vector of shell radii. Can for instance be obtained from the DomainInfo.
/// @return Table with columns: tag, shell_idx, radius, min, max, avg, cnt.
template < typename ScalarType >
util::Table
    radial_profiles_to_table( const RadialProfiles< ScalarType >& radial_profiles, const std::vector< double >& radii )
{
    if ( radii.size() != radial_profiles.radial_min_.extent( 0 ) ||
         radii.size() != radial_profiles.radial_max_.extent( 0 ) ||
         radii.size() != radial_profiles.radial_sum_.extent( 0 ) ||
         radii.size() != radial_profiles.radial_cnt_.extent( 0 ) )
    {
        throw std::runtime_error( "Radial profiles and radii do not have the same number of shells." );
    }

    const auto radial_profiles_host_min =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_min_ );
    const auto radial_profiles_host_max =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_max_ );
    const auto radial_profiles_host_sum =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_sum_ );
    const auto radial_profiles_host_cnt =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_cnt_ );
    const auto radial_profiles_host_avg =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_avg_ );

    util::Table table;
    for ( int r = 0; r < radii.size(); r++ )
    {
        table.add_row(
            { { "tag", "radial_profiles" },
              { "shell_idx", r },
              { "radius", radii[r] },
              { "min", radial_profiles_host_min( r ) },
              { "max", radial_profiles_host_max( r ) },
              { "sum", radial_profiles_host_sum( r ) },
              { "avg", radial_profiles_host_avg( r ) },
              { "cnt", radial_profiles_host_cnt( r ) } } );
    }
    return table;
}

/// @brief Linearly interpolates radial data from a std::vector (host) into a 2D grid (device) with dimensions
/// (local_subdomain_id, subdomain_size_r) for straightforward use in kernels.
///
/// This function takes care of handling possibly duplicated nodes at subdomain boundaries.
///
/// @note This will clamp values outside the passed radial profile values to the first and last values in the vector
///       respectively.
///
/// @param profile_out_label label of the returned Kokkos::View
/// @param coords_radii radii of each node for all local subdomains - see \ref terra::grid::shell::subdomain_shell_radii(),
///                     the output grid data will have the same extents
/// @param profile_radii input profile: vector of radii
/// @param profile_values input profile: vector of values to interpolate
/// @return Kokkos::View with the same dimensions of coords_radii, populated with interpolated values per subdomain
template <
    std::floating_point GridDataType,
    std::floating_point ProfileInRadiiType,
    std::floating_point ProfileInValueType,
    std::floating_point ProfileOutDataType = double >
grid::Grid2DDataScalar< ProfileOutDataType > interpolate_radial_profile_into_subdomains(
    const std::string&                            profile_out_label,
    const grid::Grid2DDataScalar< GridDataType >& coords_radii,
    const std::vector< ProfileInRadiiType >&      profile_radii,
    const std::vector< ProfileInValueType >&      profile_values )
{
    grid::Grid2DDataScalar< ProfileOutDataType > profile_data(
        profile_out_label, coords_radii.extent( 0 ), coords_radii.extent( 1 ) );
    auto profile_data_host = Kokkos::create_mirror_view( Kokkos::HostSpace{}, profile_data );

    auto coords_radii_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_radii );

    // Sort ascending (by radius - from innermost to outermost shell).

    std::vector< int > idx( profile_radii.size() );
    std::iota( idx.begin(), idx.end(), 0 );

    // sort indices by comparing keys
    std::sort( idx.begin(), idx.end(), [&]( int a, int b ) { return profile_radii[a] < profile_radii[b]; } );

    // apply permutation
    std::vector< ProfileInRadiiType > profile_radii_sorted( profile_radii );
    std::vector< ProfileInValueType > profile_values_sorted( profile_values );

    for ( size_t i = 0; i < idx.size(); i++ )
    {
        profile_radii_sorted[i]  = profile_radii[idx[i]];
        profile_values_sorted[i] = profile_values[idx[i]];
    }

    for ( int local_subdomain_id = 0; local_subdomain_id < profile_data_host.extent( 0 ); local_subdomain_id++ )
    {
        const size_t subdomain_size_r   = profile_data_host.extent( 1 );
        const size_t input_profile_size = profile_radii_sorted.size();

        if ( input_profile_size == 0 )
        {
            continue;
        }

        if ( input_profile_size == 1 )
        {
            for ( int r = 0; r < subdomain_size_r; r++ )
            {
                profile_data_host( local_subdomain_id, r ) = profile_values_sorted[0];
            }
            continue;
        }

        size_t i = 0;

        for ( size_t r = 0; r < subdomain_size_r; ++r )
        {
            double x = coords_radii_host( local_subdomain_id, r );

            // Advance src index to the correct interval
            while ( i + 1 < input_profile_size && profile_radii_sorted[i + 1] < x )
            {
                ++i;
            }

            // If x is below the first point → clamp
            if ( x <= profile_radii_sorted.front() )
            {
                profile_data_host( local_subdomain_id, r ) = profile_values_sorted.front();
                continue;
            }

            // If x is above the last point → clamp
            if ( x >= profile_radii_sorted.back() )
            {
                profile_data_host( local_subdomain_id, r ) = profile_values_sorted.back();
                continue;
            }

            // Interpolate between i and i+1
            profile_data_host( local_subdomain_id, r ) = util::interpolate_linear_1D(
                profile_radii_sorted[i],
                profile_radii_sorted[i + 1],
                profile_values_sorted[i],
                profile_values_sorted[i + 1],
                x,
                true );
        }
    }

    Kokkos::deep_copy( profile_data, profile_data_host );

    return profile_data;
}

/// @brief Interpolating a radial profile from a CSV file into a grid.
///
/// This is just a convenient wrapper around \ref interpolate_radial_profile_into_subdomains() that also
/// reads the CSV file.
///
/// @note This will clamp values outside the passed radial profile values to the first and last values in the vector
///       respectively.
///
/// @param filename csv file name containing the radial profile (radii and at least one value column)
/// @param key_radii name of the radii column
/// @param key_values name of the value column
/// @param coords_radii radii of each node for all local subdomains - see \ref terra::grid::shell::subdomain_shell_radii(),
///                     the output grid data will have the same extents
/// @return Kokkos::View with the same dimensions of coords_radii, populated with interpolated values per subdomain
template < std::floating_point GridDataType, std::floating_point ProfileOutDataType = double >
grid::Grid2DDataScalar< ProfileOutDataType > interpolate_radial_profile_into_subdomains_from_csv(
    const std::string&                            filename,
    const std::string&                            key_radii,
    const std::string&                            key_values,
    const grid::Grid2DDataScalar< GridDataType >& coords_radii )
{
    auto profile_table_result = util::read_table_from_csv( filename );
    if ( profile_table_result.is_err() )
    {
        util::logroot << profile_table_result.error() << std::endl;
        Kokkos::abort( "Error reading csv file into table." );
    }
    const auto& profile_table = profile_table_result.unwrap();

    const auto profile_radii  = profile_table.column_as_vector< double >( key_radii );
    const auto profile_values = profile_table.column_as_vector< double >( key_values );

    return interpolate_radial_profile_into_subdomains(
        "radial_profile_" + key_values, coords_radii, profile_radii, profile_values );
}

/// @brief Interpolating a constant radial profile into a grid.
///
/// @param coords_radii radii of each node for all local subdomains - see \ref terra::grid::shell::subdomain_shell_radii(),
///                     the output grid data will have the same extents
/// @param value the constant value to interpolate
/// @return Kokkos::View with the same dimensions of coords_radii, populated with interpolated values per subdomain
template < std::floating_point GridDataType, std::floating_point ProfileOutDataType = double >
grid::Grid2DDataScalar< ProfileOutDataType > interpolate_constant_radial_profile(
    const grid::Grid2DDataScalar< GridDataType >& coords_radii,
    const ProfileOutDataType&                     value )
{
    grid::Grid2DDataScalar< ProfileOutDataType > profile_data(
        "radial_profile_constant", coords_radii.extent( 0 ), coords_radii.extent( 1 ) );
    kernels::common::set_constant( profile_data, value );
    return profile_data;
}

} // namespace terra::shell