
#pragma once

#include <filesystem>

namespace terra::util {
/// @brief Prepares an empty directory with the passed path.
///
/// If the directory does not exist:              will create a directory
///  - if possible                                                                      => returns true
///  - if creation fails                                                                => returns false
/// If the directory does exist and is empty:     will do nothing                       => returns true
/// If the directory does exist and is not empty: will do nothing                       => returns false
///
/// Returns true for all non-zero ranks if `root_only` == true.
///
/// Returns true even if directory exists and is non-empty if `overwrite_content` == true.
inline bool
    prepare_empty_directory( const std::string& path_str, bool root_only = true, bool overwrite_content = false )
{
    if ( root_only && mpi::rank() != 0 )
    {
        return true;
    }

    namespace fs = std::filesystem;

    fs::path        dir( path_str );
    std::error_code ec;

    if ( !fs::exists( dir, ec ) )
    {
        // Directory doesn't exist: create it, immediately return false if creation fails
        if ( !fs::create_directories( dir, ec ) )
        {
            return false;
        }
    }
    else if ( !fs::is_empty( dir, ec ) )
    {
        // Directory exists and is not empty, return false if overwrite_content is false
        if ( !overwrite_content )
        {
            return false;
        }
    }

    // Directory either newly created, exists and is empty, or exists and files are allowed to be overwritten
    return true;
}

/// @brief Like `prepare_empty_directory()` but aborts if empty directory could not be prepared successfully.
inline void prepare_empty_directory_or_abort( const std::string& path_str, bool root_only = true )
{
    if ( !prepare_empty_directory( path_str, root_only ) )
    {
        Kokkos::abort( ( "Could not prepare empty directory with path '" + path_str + "'." ).c_str() );
    }
}
} // namespace terra::util