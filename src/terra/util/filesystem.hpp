
#pragma once

#include <filesystem>

namespace terra::util {
/// @brief Prepares an empty directory with the passed path.
///
/// If the directory does not exist:              will create a directory
///  - if possible                                                                      => returns true
///  - if creation fails                                                                => returns false
/// If the directory does exist and is empty:     will do nothing                       => returns true
/// If the directory does exist and is not empty: will do nothing                       => returns false or true (see below)
///
/// \return
/// - true for all non-zero ranks, if `root_only` == true
/// - true even if directory exists and is non-empty, if `overwrite_content` == true
/// - a string describing the status of the operation
//
// IMHO this function should be marked as nodiscard. But this requires updating the XDMF writer code.
// [[nodiscard( "filesystem operation status should be checked by caller" )]]
inline std::pair< bool, std::string >
    prepare_empty_directory( const std::string& path_str, bool root_only = true, bool overwrite_content = false )
{
    if ( root_only && mpi::rank() != 0 )
    {
        return { true, "not root: no operation" };
    }

    namespace fs = std::filesystem;

    fs::path        dir( path_str );
    std::error_code ec;

    if ( !fs::exists( dir, ec ) )
    {
        // Directory doesn't exist: create it, immediately return false if creation fails
        if ( !fs::create_directories( dir, ec ) )
        {
            std::string error_descr = "Creation of directory '" + path_str + "' failed!";
            return { false, error_descr };
        }
    }
    else if ( !fs::is_empty( dir, ec ) )
    {
        // Directory exists and is not empty, return false if overwrite_content is false
        if ( !overwrite_content )
        {
            std::string error_descr = "Directory '" + path_str + "' is not empty and overwriting not allowed!";
            return { false, error_descr };
        }
    }

    // Directory either newly created, exists and is empty, or exists and files are allowed to be overwritten
    return { true, "successfully created directory'" + path_str + "'" };
}

/// @brief Like `prepare_empty_directory()` but aborts if empty directory could not be prepared successfully.
inline void prepare_empty_directory_or_abort( const std::string& path_str, bool root_only = true )
{
    const auto [status, message] = prepare_empty_directory( path_str, root_only );
    if ( status == false )
    {
        Kokkos::abort( ( "Could not prepare empty directory with path '" + path_str + "'." ).c_str() );
    }
}
} // namespace terra::util
