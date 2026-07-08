#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "terra/grid/grid_types.hpp"

namespace terra::io {

// Define VTK cell type IDs for clarity
constexpr int VTK_QUAD           = 9;
constexpr int VTK_QUADRATIC_QUAD = 23;

// Enum to specify the desired element type
enum class VtkElementType
{
    LINEAR_QUAD,
    QUADRATIC_QUAD
};

/**
 * @brief Writes a 2D grid of vertices stored in a Kokkos View to a VTK XML
 *        Unstructured Grid file (.vtu) representing a quadrilateral mesh
 *        (linear or quadratic).
 *
 * @param filename The path to the output VTK file (.vtu).
 * @param vertices A Kokkos View containing the vertex coordinates.
 *                 Assumed dimensions: (Nx, Ny, 3).
 *                 vertices(i, j, 0) = X coordinate of point (i, j)
 *                 vertices(i, j, 1) = Y coordinate of point (i, j)
 *                 vertices(i, j, 2) = Z coordinate of point (i, j)
 *                 Nx = vertices.extent(0), Ny = vertices.extent(1)
 *                 For QUADRATIC_QUAD, Nx and Ny must be odd and >= 3.
 * @param elementType Specifies whether to write linear or quadratic elements.
 */
[[deprecated( "Use XDMF output." )]]
void write_vtk_xml_quad_mesh(
    const std::string&                  filename,
    const Kokkos::View< double** [3] >& vertices,
    VtkElementType                      elementType = VtkElementType::LINEAR_QUAD )
{
    // 1. Get Dimensions and Validate
    const size_t nx         = vertices.extent( 0 );
    const size_t ny         = vertices.extent( 1 );
    const size_t num_points = nx * ny;

    size_t  num_elements       = 0;
    size_t  points_per_element = 0;
    uint8_t vtk_cell_type_id   = 0; // Use uint8_t for VTK types array

    if ( elementType == VtkElementType::LINEAR_QUAD )
    {
        if ( nx < 2 || ny < 2 )
        {
            throw std::runtime_error( "XML: Cannot create linear quads from a mesh smaller than 2x2 points." );
        }
        num_elements       = ( nx - 1 ) * ( ny - 1 );
        points_per_element = 4;
        vtk_cell_type_id   = static_cast< uint8_t >( VTK_QUAD );
    }
    else if ( elementType == VtkElementType::QUADRATIC_QUAD )
    {
        if ( nx < 3 || ny < 3 )
        {
            throw std::runtime_error( "XML: Cannot create quadratic quads from a mesh smaller than 3x3 points." );
        }
        if ( nx % 2 == 0 || ny % 2 == 0 )
        {
            throw std::runtime_error(
                "XML: For QUADRATIC_QUAD elements using the 'every second node' scheme, Nx and Ny must be odd." );
        }
        size_t num_quad_elems_x = ( nx - 1 ) / 2;
        size_t num_quad_elems_y = ( ny - 1 ) / 2;
        num_elements            = num_quad_elems_x * num_quad_elems_y;
        points_per_element      = 8;
        vtk_cell_type_id        = static_cast< uint8_t >( VTK_QUADRATIC_QUAD );
    }
    else
    {
        throw std::runtime_error( "XML: Unsupported VtkElementType." );
    }

    if ( num_elements == 0 && num_points > 0 )
    {
        // Handle cases like 2x1 or 3x1 grids where no elements can be formed
        // Write points but zero cells
        std::cout << "Warning: Input dimensions result in zero elements. Writing points only." << std::endl;
    }
    else if ( num_elements == 0 && num_points == 0 )
    {
        throw std::runtime_error( "XML: Input dimensions result in zero points and zero elements." );
    }

    // 2. Ensure data is accessible on the Host
    auto h_vertices = Kokkos::create_mirror_view( vertices );
    Kokkos::deep_copy( h_vertices, vertices );
    Kokkos::fence();

    // 3. Open the output file stream
    std::ofstream ofs( filename );
    if ( !ofs.is_open() )
    {
        throw std::runtime_error( "XML: Could not open file for writing: " + filename );
    }
    ofs << std::fixed << std::setprecision( 8 ); // Precision for coordinates

    // --- 4. Write VTK XML Header ---
    ofs << "<?xml version=\"1.0\"?>\n";
    // Use Float64 for coordinates (double), Int64 for connectivity/offsets (safer for large meshes)
    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    ofs << "  <UnstructuredGrid>\n";
    // Piece contains the main mesh data. NumberOfPoints/Cells must be correct.
    ofs << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfCells=\"" << num_elements << "\">\n";

    // --- 5. Write Points ---
    ofs << "      <Points>\n";
    // DataArray for coordinates: Float64, 3 components (XYZ)
    ofs << "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for ( size_t i = 0; i < nx; ++i )
    {
        for ( size_t j = 0; j < ny; ++j )
        {
            ofs << "          " << h_vertices( i, j, 0 ) << " " << h_vertices( i, j, 1 ) << " " << h_vertices( i, j, 2 )
                << "\n";
        }
    }
    ofs << "        </DataArray>\n";
    ofs << "      </Points>\n";

    // --- 6. Write Cells (Connectivity, Offsets, Types) ---
    ofs << "      <Cells>\n";

    // 6.a. Connectivity Array (flat list of point indices for all cells)
    // Use Int64 for indices to be safe with large meshes (size_t can exceed Int32)
    std::vector< int64_t > connectivity;                       // Use std::vector temporarily or write directly
    connectivity.reserve( num_elements * points_per_element ); // Pre-allocate roughly

    // 6.b. Offsets Array (index in connectivity where each cell ENDS)
    std::vector< int64_t > offsets;
    offsets.reserve( num_elements );
    int64_t current_offset = 0; // VTK offsets are cumulative

    // 6.c. Types Array (VTK cell type ID for each cell)
    std::vector< uint8_t > types;
    types.reserve( num_elements );

    // --- Populate Connectivity, Offsets, and Types ---
    if ( elementType == VtkElementType::LINEAR_QUAD )
    {
        for ( size_t i = 0; i < nx - 1; ++i )
        {
            for ( size_t j = 0; j < ny - 1; ++j )
            {
                // Calculate the 0-based indices
                int64_t p0_idx = static_cast< int64_t >( i * ny + j );
                int64_t p1_idx = static_cast< int64_t >( ( i + 1 ) * ny + j );
                int64_t p2_idx = static_cast< int64_t >( ( i + 1 ) * ny + ( j + 1 ) );
                int64_t p3_idx = static_cast< int64_t >( i * ny + ( j + 1 ) );

                // Append connectivity
                connectivity.push_back( p0_idx );
                connectivity.push_back( p1_idx );
                connectivity.push_back( p2_idx );
                connectivity.push_back( p3_idx );

                // Update and append offset
                current_offset += points_per_element;
                offsets.push_back( current_offset );

                // Append type
                types.push_back( vtk_cell_type_id );
            }
        }
    }
    else
    { // QUADRATIC_QUAD
        for ( size_t i = 0; i < nx - 1; i += 2 )
        {
            for ( size_t j = 0; j < ny - 1; j += 2 )
            {
                // Calculate indices (casting to int64_t for the array)
                int64_t p0_idx  = static_cast< int64_t >( i * ny + j );
                int64_t p1_idx  = static_cast< int64_t >( ( i + 2 ) * ny + j );
                int64_t p2_idx  = static_cast< int64_t >( ( i + 2 ) * ny + ( j + 2 ) );
                int64_t p3_idx  = static_cast< int64_t >( i * ny + ( j + 2 ) );
                int64_t m01_idx = static_cast< int64_t >( ( i + 1 ) * ny + j );
                int64_t m12_idx = static_cast< int64_t >( ( i + 2 ) * ny + ( j + 1 ) );
                int64_t m23_idx = static_cast< int64_t >( ( i + 1 ) * ny + ( j + 2 ) );
                int64_t m30_idx = static_cast< int64_t >( i * ny + ( j + 1 ) );

                // Append connectivity (VTK order: corners then midsides)
                connectivity.push_back( p0_idx );
                connectivity.push_back( p1_idx );
                connectivity.push_back( p2_idx );
                connectivity.push_back( p3_idx );
                connectivity.push_back( m01_idx );
                connectivity.push_back( m12_idx );
                connectivity.push_back( m23_idx );
                connectivity.push_back( m30_idx );

                // Update and append offset
                current_offset += points_per_element;
                offsets.push_back( current_offset );

                // Append type
                types.push_back( vtk_cell_type_id );
            }
        }
    }

    // --- Write the populated arrays to the file ---
    // Connectivity
    ofs << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    ofs << "          "; // Indentation
    for ( size_t i = 0; i < connectivity.size(); ++i )
    {
        ofs << connectivity[i] << ( ( i + 1 ) % 12 == 0 ? "\n          " : " " ); // Newline every 12 values
    }
    ofs << "\n        </DataArray>\n"; // Add newline before closing tag if needed

    // Offsets
    ofs << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    ofs << "          ";
    for ( size_t i = 0; i < offsets.size(); ++i )
    {
        ofs << offsets[i] << ( ( i + 1 ) % 12 == 0 ? "\n          " : " " );
    }
    ofs << "\n        </DataArray>\n";

    // Types
    ofs << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    ofs << "          ";
    for ( size_t i = 0; i < types.size(); ++i )
    {
        // Need to cast uint8_t to int for printing as number, not char
        ofs << static_cast< int >( types[i] ) << ( ( i + 1 ) % 20 == 0 ? "\n          " : " " );
    }
    ofs << "\n        </DataArray>\n";

    ofs << "      </Cells>\n";

    // --- 7. Write Empty PointData and CellData (Good Practice) ---
    ofs << "      <PointData>\n";
    // Add <DataArray> tags here if you have data associated with points
    ofs << "      </PointData>\n";
    ofs << "      <CellData>\n";
    // Add <DataArray> tags here if you have data associated with cells
    ofs << "      </CellData>\n";

    // --- 8. Write VTK XML Footer ---
    ofs << "    </Piece>\n";
    ofs << "  </UnstructuredGrid>\n";
    ofs << "</VTKFile>\n";

    // 9. Close the file
    ofs.close();
    if ( !ofs )
    {
        throw std::runtime_error( "XML: Error occurred during writing or closing file: " + filename );
    }
}

// Enum to choose the diagonal for splitting quads
enum class DiagonalSplitType
{
    FORWARD_SLASH, // Connects (i,j) with (i+1,j+1)
    BACKWARD_SLASH // Connects (i+1,j) with (i,j+1)
};

// Helper to get VTK type string from C++ type
template < typename T >
[[deprecated( "Use XDMF output." )]]
std::string get_vtk_type_string()
{
    if ( std::is_same_v< T, float > )
        return "Float32";
    if ( std::is_same_v< T, double > )
        return "Float64";
    if ( std::is_same_v< T, int > )
        return "Int32";
    if ( std::is_same_v< T, long long > )
        return "Int64";
    if ( std::is_same_v< T, int8_t > )
        return "Int8";
    if ( std::is_same_v< T, uint8_t > )
        return "UInt8";
    // Add more types as needed
    throw std::runtime_error( "Unsupported data type for VTK output" );
}

template < typename ScalarType >
[[deprecated( "Use XDMF output." )]]
void write_rectilinear_to_triangular_vtu(
    Kokkos::View< ScalarType** [3] > points_device_view,
    const std::string&               filename,
    DiagonalSplitType                split_type )
{
    auto points_host_mirror = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, points_device_view );

    const int Nx = points_host_mirror.extent( 0 ); // Number of points in 1st dim (e.g., x)
    const int Ny = points_host_mirror.extent( 1 ); // Number of points in 2nd dim (e.g., y)

    if ( Nx < 2 || Ny < 2 )
    {
        throw std::runtime_error( "Grid dimensions are too small to form cells (Nx, Ny must be >= 2)." );
    }

    const long long num_total_points   = static_cast< long long >( Nx ) * Ny;
    const long long num_quads_in_plane = static_cast< long long >( Nx - 1 ) * ( Ny - 1 );
    const long long num_cells          = num_quads_in_plane * 2; // Each quad becomes 2 triangles

    std::ofstream vtk_file( filename );
    if ( !vtk_file.is_open() )
    {
        throw std::runtime_error( "Failed to open file: " + filename );
    }

    vtk_file
        << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vtk_file << "  <UnstructuredGrid>\n";
    vtk_file << "    <Piece NumberOfPoints=\"" << num_total_points << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // --- Points ---
    // Points are written by iterating through the first index (Nx), then the second index (Ny).
    // The global point ID for point_host_mirror(i,j,*) is (i * Ny + j).
    vtk_file << "      <Points>\n";
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< ScalarType >()
             << "\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    vtk_file << std::fixed << std::setprecision( 10 );
    for ( int i = 0; i < Nx; ++i )
    { // Iterate 1st dim
        for ( int j = 0; j < Ny; ++j )
        { // Iterate 2nd dim
            vtk_file << "          " << points_host_mirror( i, j, 0 ) << " " << points_host_mirror( i, j, 1 ) << " "
                     << points_host_mirror( i, j, 2 ) << "\n";
        }
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Points>\n";

    // --- Cells (Connectivity, Offsets, Types) ---
    vtk_file << "      <Cells>\n";

    // Connectivity
    std::vector< long long > connectivity_data;
    connectivity_data.reserve( num_cells * 3 ); // 3 (indices) per triangle

    for ( int i = 0; i < Nx - 1; ++i )
    { // Iterate over quads in 1st dim
        for ( int j = 0; j < Ny - 1; ++j )
        { // Iterate over quads in 2nd dim
            // Global 0-based indices of the quad's corners
            // Point (i,j) has global ID: i * Ny + j
            long long p00 = static_cast< long long >( i ) * Ny + j;             // (i, j)
            long long p10 = static_cast< long long >( i + 1 ) * Ny + j;         // (i+1, j)
            long long p01 = static_cast< long long >( i ) * Ny + ( j + 1 );     // (i, j+1)
            long long p11 = static_cast< long long >( i + 1 ) * Ny + ( j + 1 ); // (i+1, j+1)

            if ( split_type == DiagonalSplitType::FORWARD_SLASH )
            {
                // Diagonal from (i,j) to (i+1,j+1)
                // Triangle 1: (i,j), (i+1,j), (i+1,j+1)
                connectivity_data.push_back( p00 );
                connectivity_data.push_back( p10 );
                connectivity_data.push_back( p11 );
                // Triangle 2: (i,j), (i+1,j+1), (i,j+1)
                connectivity_data.push_back( p00 );
                connectivity_data.push_back( p11 );
                connectivity_data.push_back( p01 );
            }
            else
            { // BACKWARD_SLASH
                // Diagonal from (i+1,j) to (i,j+1)
                // Triangle 1: (i,j), (i+1,j), (i,j+1)
                connectivity_data.push_back( p00 );
                connectivity_data.push_back( p10 );
                connectivity_data.push_back( p01 );
                // Triangle 2: (i+1,j), (i+1,j+1), (i,j+1)
                connectivity_data.push_back( p10 );
                connectivity_data.push_back( p11 );
                connectivity_data.push_back( p01 );
            }
        }
    }
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< long long >()
             << "\" Name=\"connectivity\" format=\"ascii\">\n";
    vtk_file << "          ";
    for ( size_t k = 0; k < connectivity_data.size(); ++k )
    {
        vtk_file << connectivity_data[k] << ( ( k == connectivity_data.size() - 1 ) ? "" : " " );
        if ( ( k + 1 ) % 3 == 0 && k < connectivity_data.size() - 1 )
        { // Newline for readability
            vtk_file << "\n          ";
        }
    }
    vtk_file << "\n";
    vtk_file << "        </DataArray>\n";

    // Offsets
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< long long >()
             << "\" Name=\"offsets\" format=\"ascii\">\n";
    long long current_offset = 0;
    for ( long long c = 0; c < num_cells; ++c )
    {
        current_offset += ( 3 ); // Each triangle: 3 points
        vtk_file << "          " << current_offset << "\n";
    }
    vtk_file << "        </DataArray>\n";

    // Types (VTK_TRIANGLE = 5)
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< uint8_t >()
             << "\" Name=\"types\" format=\"ascii\">\n";
    for ( long long c = 0; c < num_cells; ++c )
    {
        vtk_file << "          5\n";
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Cells>\n";

    vtk_file << "    </Piece>\n";
    vtk_file << "  </UnstructuredGrid>\n";
    vtk_file << "</VTKFile>\n";

    vtk_file.close();
}

template <
    typename PointRealT, // Type for surface coordinates and radii elements
    typename AttachedDataType >
[[deprecated( "Use XDMF output." )]]
void write_surface_radial_extruded_to_wedge_vtu(
    grid::Grid2DDataVec< PointRealT, 3 > surface_points_device_view,
    grid::Grid1DDataScalar< PointRealT > radii_device_view,
    std::optional< AttachedDataType >    optional_attached_data_device_view,
    const std::string& vector_data_name, // Used only if vector_data_device_view has value & NumVecComponents > 0
    const std::string& filename,
    DiagonalSplitType  split_type )
{
    static_assert(
        std::is_same_v< AttachedDataType, grid::Grid3DDataScalar< double > > ||
        std::is_same_v< AttachedDataType, grid::Grid3DDataVec< double, 3 > > );

    const bool has_attached_data  = optional_attached_data_device_view.has_value();
    const int  is_scalar_data     = std::is_same_v< AttachedDataType, grid::Grid3DDataScalar< double > >;
    const int  num_vec_components = has_attached_data ? optional_attached_data_device_view.value().extent( 3 ) : 1;

    // --- 1. Create host mirrors and copy data ---
    auto surface_points_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, surface_points_device_view );
    auto radii_host          = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, radii_device_view );

    typename AttachedDataType::host_mirror_type vector_data_host;
    if ( has_attached_data )
    {
        vector_data_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, optional_attached_data_device_view.value() );
    }
    // Kokkos::fence(); // If using non-blocking deep_copy

    // --- 2. Get dimensions ---
    const int Ns       = surface_points_host.extent( 0 ); // Num points in 1st dim of surface
    const int Nt       = surface_points_host.extent( 1 ); // Num points in 2nd dim of surface
    const int Nw_radii = radii_host.extent( 0 );          // Num of radius values (defines layers of points)

    if ( Ns < 2 || Nt < 2 )
    {
        throw std::runtime_error( "Surface grid dimensions (Ns, Nt) must be >= 2 to form base triangles." );
    }
    if ( Nw_radii < 1 )
    {
        throw std::runtime_error( "Radii view must contain at least one radius value (Nw_radii >= 1)." );
    }

    // Validate vector data dimensions if provided
    if ( has_attached_data )
    { // Implies NumVecComponents > 0
        if ( vector_data_host.extent( 0 ) != Ns || vector_data_host.extent( 1 ) != Nt ||
             vector_data_host.extent( 2 ) != Nw_radii )
        {
            throw std::runtime_error(
                "VTK: Vector data dimensions (Ns, Nt, Nw_radii) do not match generated point structure." );
        }
        // extent(3) is NumVecComponents, checked by template/view type
        if ( vector_data_name.empty() )
        {
            throw std::runtime_error( "VTK: Vector data name must be provided if vector data is present." );
        }
    }

    // --- 3. Calculate total points and cells ---
    const long long num_total_points = static_cast< long long >( Ns ) * Nt * Nw_radii;
    long long       num_cells        = 0;
    if ( Nw_radii >= 2 )
    { // Need at least 2 radial layers to form wedges
        const long long num_quads_in_surface_plane = static_cast< long long >( Ns - 1 ) * ( Nt - 1 );
        const long long num_wedge_layers           = static_cast< long long >( Nw_radii - 1 );
        num_cells                                  = num_quads_in_surface_plane * 2 * num_wedge_layers;
    }

    // --- 4. Open VTK file and write header ---
    std::ofstream vtk_file( filename );
    if ( !vtk_file.is_open() )
    {
        throw std::runtime_error( "Failed to open file: " + filename );
    }
    vtk_file
        << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    vtk_file << "  <UnstructuredGrid>\n";
    vtk_file << "    <Piece NumberOfPoints=\"" << num_total_points << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // --- 5. Points ---
    vtk_file << "      <Points>\n";
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< PointRealT >()
             << "\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    vtk_file << std::fixed << std::setprecision( 10 );

    PointRealT base_s_pt_coords[3]; // To store coords of one surface_points_host(s,t,*)
    for ( int s_idx = 0; s_idx < Ns; ++s_idx )
    {
        for ( int t_idx = 0; t_idx < Nt; ++t_idx )
        {
            // Cache the base surface point coordinates
            base_s_pt_coords[0] = surface_points_host( s_idx, t_idx, 0 );
            base_s_pt_coords[1] = surface_points_host( s_idx, t_idx, 1 );
            base_s_pt_coords[2] = surface_points_host( s_idx, t_idx, 2 );
            for ( int w_rad_idx = 0; w_rad_idx < Nw_radii; ++w_rad_idx )
            {
                PointRealT current_radius_val = radii_host( w_rad_idx );
                vtk_file << "          " << base_s_pt_coords[0] * current_radius_val << " "
                         << base_s_pt_coords[1] * current_radius_val << " " << base_s_pt_coords[2] * current_radius_val
                         << "\n";
            }
        }
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Points>\n";

    // --- 6. Cells (Connectivity, Offsets, Types) ---
    vtk_file << "      <Cells>\n";
    std::vector< long long > connectivity_data;
    if ( num_cells > 0 )
    {
        connectivity_data.reserve( num_cells * 6 ); //  6 indices per wedge

        auto pt_gid = [&]( int s_surf_idx, int t_surf_idx, int w_rad_layer_idx ) {
            return static_cast< long long >( s_surf_idx ) * ( Nt * Nw_radii ) +
                   static_cast< long long >( t_surf_idx ) * Nw_radii + static_cast< long long >( w_rad_layer_idx );
        };

        for ( int s = 0; s < Ns - 1; ++s )
        { // Iterate over quads in s-dim of surface
            for ( int t = 0; t < Nt - 1; ++t )
            { // Iterate over quads in t-dim of surface
                for ( int w_rad_layer_idx = 0; w_rad_layer_idx < Nw_radii - 1; ++w_rad_layer_idx )
                { // Iterate over wedge layers
                    int base_rad_idx = w_rad_layer_idx;
                    int top_rad_idx  = w_rad_layer_idx + 1;

                    long long p00_base = pt_gid( s, t, base_rad_idx );
                    long long p10_base = pt_gid( s + 1, t, base_rad_idx );
                    long long p01_base = pt_gid( s, t + 1, base_rad_idx );
                    long long p11_base = pt_gid( s + 1, t + 1, base_rad_idx );

                    long long p00_top = pt_gid( s, t, top_rad_idx );
                    long long p10_top = pt_gid( s + 1, t, top_rad_idx );
                    long long p01_top = pt_gid( s, t + 1, top_rad_idx );
                    long long p11_top = pt_gid( s + 1, t + 1, top_rad_idx );

                    if ( split_type == DiagonalSplitType::FORWARD_SLASH )
                    {
                        // Wedge 1 from Tri 1: (p00, p10, p11)_base -> (p00, p10, p11)_top
                        connectivity_data.push_back( p00_base );
                        connectivity_data.push_back( p10_base );
                        connectivity_data.push_back( p11_base );
                        connectivity_data.push_back( p00_top );
                        connectivity_data.push_back( p10_top );
                        connectivity_data.push_back( p11_top );
                        // Wedge 2 from Tri 2: (p00, p11, p01)_base -> (p00, p11, p01)_top
                        connectivity_data.push_back( p00_base );
                        connectivity_data.push_back( p11_base );
                        connectivity_data.push_back( p01_base );
                        connectivity_data.push_back( p00_top );
                        connectivity_data.push_back( p11_top );
                        connectivity_data.push_back( p01_top );
                    }
                    else
                    { // BACKWARD_SLASH
                        // Wedge 1 from Tri 1: (p00, p10, p01)_base -> (p00, p10, p01)_top
                        connectivity_data.push_back( p00_base );
                        connectivity_data.push_back( p10_base );
                        connectivity_data.push_back( p01_base );
                        connectivity_data.push_back( p00_top );
                        connectivity_data.push_back( p10_top );
                        connectivity_data.push_back( p01_top );
                        // Wedge 2 from Tri 2: (p10, p11, p01)_base -> (p10, p11, p01)_top
                        connectivity_data.push_back( p10_base );
                        connectivity_data.push_back( p11_base );
                        connectivity_data.push_back( p01_base );
                        connectivity_data.push_back( p10_top );
                        connectivity_data.push_back( p11_top );
                        connectivity_data.push_back( p01_top );
                    }
                }
            }
        }
    }
    // Write Connectivity
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< long long >()
             << "\" Name=\"connectivity\" format=\"ascii\">\n";
    if ( !connectivity_data.empty() )
    {
        vtk_file << "          ";
        for ( size_t k = 0; k < connectivity_data.size(); ++k )
        {
            vtk_file << connectivity_data[k] << ( ( k == connectivity_data.size() - 1 ) ? "" : " " );
            if ( ( k + 1 ) % 6 == 0 && k < connectivity_data.size() - 1 )
                vtk_file << "\n          ";
        }
        vtk_file << "\n";
    }
    vtk_file << "        </DataArray>\n";

    // Write Offsets
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< long long >()
             << "\" Name=\"offsets\" format=\"ascii\">\n";
    if ( num_cells > 0 )
    {
        long long current_offset = 0;
        for ( long long c = 0; c < num_cells; ++c )
        {
            current_offset += 6; // Each wedge: 6 points
            vtk_file << "          " << current_offset << "\n";
        }
    }
    vtk_file << "        </DataArray>\n";

    // Write Types
    vtk_file << "        <DataArray type=\"" << get_vtk_type_string< uint8_t >()
             << "\" Name=\"types\" format=\"ascii\">\n";
    if ( num_cells > 0 )
    {
        for ( long long c = 0; c < num_cells; ++c )
        {
            vtk_file << "          13\n"; // VTK_WEDGE cell type
        }
    }
    vtk_file << "        </DataArray>\n";
    vtk_file << "      </Cells>\n";

    // --- 7. PointData (if vector data is provided) ---
    if ( has_attached_data )
    {
        std::string point_data_attributes_str;
        if ( is_scalar_data )
        {
            point_data_attributes_str = " Scalars=\"" + vector_data_name + "\"";
        }
        else
        {
            point_data_attributes_str = " Vectors=\"" + vector_data_name + "\"";
        }

        vtk_file << "      <PointData" << point_data_attributes_str << ">\n";
        vtk_file << "        <DataArray type=\"" << get_vtk_type_string< double >() << "\" Name=\"" << vector_data_name
                 << "\" NumberOfComponents=\"" << num_vec_components << "\" format=\"ascii\">\n";
        vtk_file << std::fixed << std::setprecision( 10 );

        // Iterate in the SAME order as points were written: s_idx, t_idx, w_rad_idx
        for ( int s_idx = 0; s_idx < Ns; ++s_idx )
        {
            for ( int t_idx = 0; t_idx < Nt; ++t_idx )
            {
                for ( int w_rad_idx = 0; w_rad_idx < Nw_radii; ++w_rad_idx )
                {
                    vtk_file << "          ";
                    if constexpr ( is_scalar_data )
                    {
                        vtk_file << vector_data_host( s_idx, t_idx, w_rad_idx ) << " ";
                    }
                    else
                    {
                        for ( size_t comp = 0; comp < num_vec_components; ++comp )
                        {
                            vtk_file << vector_data_host( s_idx, t_idx, w_rad_idx, comp )
                                     << ( comp == num_vec_components - 1 ? "" : " " );
                        }
                    }

                    vtk_file << "\n";
                }
            }
        }
        vtk_file << "        </DataArray>\n";
        vtk_file << "      </PointData>\n";
    }
    else
    {                                      // No vector data or NumVecComponents is 0
        vtk_file << "      <PointData>\n"; // Empty PointData section
        vtk_file << "      </PointData>\n";
    }

    // --- Footer ---
    vtk_file << "    </Piece>\n";
    vtk_file << "  </UnstructuredGrid>\n";
    vtk_file << "</VTKFile>\n";
    vtk_file.close();
}

/// @tparam InputDataScalarType The scalar type of the added grids - the output type can be set later.
template < typename InputDataScalarType >
class [[deprecated( "Use XDMF output." )]] VTKOutput
{
  public:
    using ScalarFieldDeviceView = Kokkos::View< InputDataScalarType**** >;
    using VectorFieldDeviceView = Kokkos::View< InputDataScalarType**** [3] >;

    // Define Host view types for field data storage for clarity
    // Assuming input fields are double, will be cast to float on write.
    // If input fields can be float, this could be templated further or use a base class.
    using ScalarFieldHostView = ScalarFieldDeviceView::host_mirror_type;
    using VectorFieldHostView = VectorFieldDeviceView::host_mirror_type;

    template < class ShellCoordsView, class RadiiView >
    VTKOutput(
        const ShellCoordsView& shell_node_coords_device,
        const RadiiView&       radii_per_layer_device,
        bool                   generate_quadratic_elements_from_linear_input = false )
    : is_quadratic_( generate_quadratic_elements_from_linear_input )
    {
        // 1. Copy input geometry data to managed host views (as before)
        h_shell_coords_managed_ = Kokkos::create_mirror_view( shell_node_coords_device );
        Kokkos::deep_copy( h_shell_coords_managed_, shell_node_coords_device );
        h_radii_managed_ = Kokkos::create_mirror_view( radii_per_layer_device );
        Kokkos::deep_copy( h_radii_managed_, radii_per_layer_device );

        // 2. Get INPUT dimensions
        num_subdomains_      = h_shell_coords_managed_.extent( 0 );
        NX_nodes_surf_input_ = h_shell_coords_managed_.extent( 1 );
        NY_nodes_surf_input_ = h_shell_coords_managed_.extent( 2 );
        NR_nodes_rad_input_  = h_radii_managed_.extent( 1 );

        if ( NX_nodes_surf_input_ < 1 || NY_nodes_surf_input_ < 1 || NR_nodes_rad_input_ < 1 )
        { // Min 1 node
            throw std::runtime_error( "Input node counts must be at least 1 in each dimension." );
        }
        // Check for cell formation possibility
        bool can_form_cells = ( NX_nodes_surf_input_ > 1 || NY_nodes_surf_input_ > 1 || NR_nodes_rad_input_ > 1 );
        if ( !can_form_cells && ( NX_nodes_surf_input_ * NY_nodes_surf_input_ * NR_nodes_rad_input_ > 1 ) )
        {
            // Multiple points, but arranged as a line/plane that can't form 3D hex cells
            // This is fine, we might just output points.
        }
        if ( NX_nodes_surf_input_ < 2 && NY_nodes_surf_input_ < 2 && NR_nodes_rad_input_ < 2 &&
             ( NX_nodes_surf_input_ + NY_nodes_surf_input_ + NR_nodes_rad_input_ > 3 ) )
        {
            // This case implies more than one point, but not enough to form a cell.
            // E.g., 1x1xN where N>1 (line of points). VTK handles this.
        }

        // 3. Calculate OUTPUT grid dimensions and total points/cells for header
        if ( is_quadratic_ )
        {
            NX_nodes_surf_output_ =
                ( NX_nodes_surf_input_ > 1 ) ? ( 2 * ( NX_nodes_surf_input_ - 1 ) + 1 ) : NX_nodes_surf_input_;
            NY_nodes_surf_output_ =
                ( NY_nodes_surf_input_ > 1 ) ? ( 2 * ( NY_nodes_surf_input_ - 1 ) + 1 ) : NY_nodes_surf_input_;
            NR_nodes_rad_output_ =
                ( NR_nodes_rad_input_ > 1 ) ? ( 2 * ( NR_nodes_rad_input_ - 1 ) + 1 ) : NR_nodes_rad_input_;
        }
        else
        {
            NX_nodes_surf_output_ = NX_nodes_surf_input_;
            NY_nodes_surf_output_ = NY_nodes_surf_input_;
            NR_nodes_rad_output_  = NR_nodes_rad_input_;
        }

        num_total_points_ = num_subdomains_ * NX_nodes_surf_output_ * NY_nodes_surf_output_ * NR_nodes_rad_output_;

        size_t num_cells_x_surf = ( NX_nodes_surf_input_ > 1 ) ? NX_nodes_surf_input_ - 1 : 0;
        size_t num_cells_y_surf = ( NY_nodes_surf_input_ > 1 ) ? NY_nodes_surf_input_ - 1 : 0;
        size_t num_cells_r_rad  = ( NR_nodes_rad_input_ > 1 ) ? NR_nodes_rad_input_ - 1 : 0;
        num_total_cells_        = num_subdomains_ * num_cells_x_surf * num_cells_y_surf * num_cells_r_rad;
    }

    template < class ScalarFieldViewDevice >
    void add_scalar_field( const ScalarFieldViewDevice& field_data_view_device )
    {
        if ( field_data_view_device.rank() != 4 )
        {
            throw std::runtime_error( "Scalar field data view must have rank 4: (sd, x_in, y_in, r_in)." );
        }
        validate_field_view_dimensions_for_input_grid( field_data_view_device, field_data_view_device.label() );

        PointDataEntry entry;
        entry.name                   = field_data_view_device.label();
        entry.num_components         = 1;
        entry.data_type_str          = "Float32";
        entry.device_view_input_data = field_data_view_device;
        point_data_entries_.push_back( std::move( entry ) );
    }

    template < class VectorFieldViewDevice >
    void add_vector_field( const VectorFieldViewDevice& field_data_view_device )
    {
        if ( field_data_view_device.rank() != 5 )
        { // sd, x_in, y_in, r_in, comp
            throw std::runtime_error( "Vector field data view must have rank 5." );
        }
        int num_vec_components = field_data_view_device.extent( 4 );
        if ( num_vec_components <= 0 )
        {
            throw std::runtime_error( "Vector field must have at least one component." );
        }
        validate_field_view_dimensions_for_input_grid( field_data_view_device, field_data_view_device.label() );

        PointDataEntry entry;
        entry.name                   = field_data_view_device.label();
        entry.num_components         = num_vec_components;
        entry.data_type_str          = "Float32";
        entry.device_view_input_data = field_data_view_device;
        point_data_entries_.push_back( std::move( entry ) );
    }

    void write( const std::string& output_file )
    {
        std::ofstream vtk_file( output_file );
        if ( !vtk_file.is_open() )
        {
            throw std::runtime_error( "Failed to open VTK output file: " + output_file );
        }
        vtk_file << std::fixed << std::setprecision( 8 );

        vtk_file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
        vtk_file << "  <UnstructuredGrid>\n";
        vtk_file << "    <Piece NumberOfPoints=\"" << num_total_points_ << "\" NumberOfCells=\"" << num_total_cells_
                 << "\">\n";

        // --- Points ---
        vtk_file << "      <Points>\n";
        vtk_file
            << "        <DataArray type=\"Float32\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n";
        write_points_data( vtk_file );
        vtk_file << "        </DataArray>\n";
        vtk_file << "      </Points>\n";

        // --- Cells ---
        if ( num_total_cells_ > 0 )
        { // Only write cell data if there are cells
            vtk_file << "      <Cells>\n";
            vtk_file << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
            write_cell_connectivity_data( vtk_file );
            vtk_file << "        </DataArray>\n";
            vtk_file << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
            write_cell_offsets_data( vtk_file );
            vtk_file << "        </DataArray>\n";
            vtk_file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
            write_cell_types_data( vtk_file );
            vtk_file << "        </DataArray>\n";
            vtk_file << "      </Cells>\n";
        }

        // --- PointData ---
        if ( !point_data_entries_.empty() && num_total_points_ > 0 )
        {
            vtk_file << "      <PointData>\n";
            for ( const auto& entry : point_data_entries_ )
            {
                vtk_file << "        <DataArray type=\"" << entry.data_type_str << "\" Name=\"" << entry.name << "\"";
                if ( entry.num_components > 1 )
                {
                    vtk_file << " NumberOfComponents=\"" << entry.num_components << "\"";
                }
                vtk_file << " format=\"ascii\">\n";
                write_field_data( vtk_file, entry );
                vtk_file << "        </DataArray>\n";
            }
            vtk_file << "      </PointData>\n";
        }

        vtk_file << "    </Piece>\n";
        vtk_file << "  </UnstructuredGrid>\n";
        vtk_file << "</VTKFile>\n";

        vtk_file.close();
    }

  private:
    struct PointDataEntry
    {
        std::string name;
        // Store the host-mirrored INPUT grid data view directly
        std::variant< ScalarFieldDeviceView, VectorFieldDeviceView > device_view_input_data;
        int                                                          num_components;
        std::string                                                  data_type_str;
    };

    // --- Helper: Get Global Output Node ID --- (same as before)
    size_t get_global_output_node_id( size_t sd_idx, size_t ix_out, size_t iy_out, size_t ir_out ) const
    {
        size_t nodes_per_surf_layer_output = NX_nodes_surf_output_ * NY_nodes_surf_output_;
        size_t nodes_per_subdomain_output  = nodes_per_surf_layer_output * NR_nodes_rad_output_;
        size_t base_offset_sd              = sd_idx * nodes_per_subdomain_output;
        size_t offset_in_sd = ir_out * nodes_per_surf_layer_output + iy_out * NX_nodes_surf_output_ + ix_out;
        return base_offset_sd + offset_in_sd;
    }

    // --- Helper: Validate Field View Dimensions --- (same as before)
    template < class FieldView >
    void validate_field_view_dimensions_for_input_grid( const FieldView& view, const std::string& field_name )
    {
        if ( view.extent( 0 ) != num_subdomains_ )
            throw std::runtime_error( "Field '" + field_name + "' sd mismatch." );
        if ( view.extent( 1 ) != NX_nodes_surf_input_ )
            throw std::runtime_error( "Field '" + field_name + "' X_in mismatch." );
        if ( view.extent( 2 ) != NY_nodes_surf_input_ )
            throw std::runtime_error( "Field '" + field_name + "' Y_in mismatch." );
        if ( view.extent( 3 ) != NR_nodes_rad_input_ )
            throw std::runtime_error( "Field '" + field_name + "' R_in mismatch." );
    }

    // --- Helper: Write Points Data ---
    void write_points_data( std::ostream& os )
    {
        for ( size_t sd = 0; sd < num_subdomains_; ++sd )
        {
            for ( size_t ir_out = 0; ir_out < NR_nodes_rad_output_; ++ir_out )
            {
                for ( size_t iy_out = 0; iy_out < NY_nodes_surf_output_; ++iy_out )
                {
                    for ( size_t ix_out = 0; ix_out < NX_nodes_surf_output_; ++ix_out )
                    {
                        double final_px, final_py, final_pz;
                        get_interpolated_point_coordinates( sd, ix_out, iy_out, ir_out, final_px, final_py, final_pz );
                        os << "          " << static_cast< float >( final_px ) << " "
                           << static_cast< float >( final_py ) << " " << static_cast< float >( final_pz ) << "\n";
                    }
                }
            }
        }
    }

    // --- Helper: Get Interpolated Point Coordinates ---
    void get_interpolated_point_coordinates(
        size_t  sd,
        size_t  ix_out,
        size_t  iy_out,
        size_t  ir_out,
        double& final_px,
        double& final_py,
        double& final_pz ) const
    {
        if ( !is_quadratic_ || ( NX_nodes_surf_input_ <= 1 && NY_nodes_surf_input_ <= 1 && NR_nodes_rad_input_ <= 1 ) )
        { // Single point case
            size_t ix_in = ix_out;
            size_t iy_in = iy_out;
            size_t ir_in = ir_out;
            // Ensure indices are valid for non-interpolated case, should be 0 if input dim is 1
            if ( NX_nodes_surf_input_ == 1 )
                ix_in = 0;
            if ( NY_nodes_surf_input_ == 1 )
                iy_in = 0;
            if ( NR_nodes_rad_input_ == 1 )
                ir_in = 0;

            double radius = h_radii_managed_( sd, ir_in );
            final_px      = h_shell_coords_managed_( sd, ix_in, iy_in, 0 ) * radius;
            final_py      = h_shell_coords_managed_( sd, ix_in, iy_in, 1 ) * radius;
            final_pz      = h_shell_coords_managed_( sd, ix_in, iy_in, 2 ) * radius;
        }
        else
        {
            size_t ix_in0 = ix_out / 2;
            size_t iy_in0 = iy_out / 2;
            size_t ir_in0 = ir_out / 2;

            double base_x_sum = 0.0, base_y_sum = 0.0, base_z_sum = 0.0;
            double W_shell_sum = 0.0;

            double wx_param = ( NX_nodes_surf_input_ > 1 ) ? ( ix_out % 2 ) * 0.5 : 0.0;
            double wy_param = ( NY_nodes_surf_input_ > 1 ) ? ( iy_out % 2 ) * 0.5 : 0.0;

            for ( int j_off = 0; j_off <= ( ( wy_param > 0.0 && iy_in0 + 1 < NY_nodes_surf_input_ ) ? 1 : 0 ); ++j_off )
            {
                size_t iy_in = iy_in0 + j_off;
                double W_j   = ( j_off == 0 ) ? ( 1.0 - wy_param ) : wy_param;
                for ( int i_off = 0; i_off <= ( ( wx_param > 0.0 && ix_in0 + 1 < NX_nodes_surf_input_ ) ? 1 : 0 );
                      ++i_off )
                {
                    size_t ix_in = ix_in0 + i_off;
                    double W_i   = ( i_off == 0 ) ? ( 1.0 - wx_param ) : wx_param;

                    double shell_w = W_i * W_j;
                    base_x_sum += shell_w * h_shell_coords_managed_( sd, ix_in, iy_in, 0 );
                    base_y_sum += shell_w * h_shell_coords_managed_( sd, ix_in, iy_in, 1 );
                    base_z_sum += shell_w * h_shell_coords_managed_( sd, ix_in, iy_in, 2 );
                    W_shell_sum += shell_w;
                }
            }
            // Handle case where denominator might be zero (e.g. single input node line)
            double unit_px =
                ( W_shell_sum > 1e-9 ) ?
                    base_x_sum / W_shell_sum :
                    h_shell_coords_managed_(
                        sd, ( NX_nodes_surf_input_ > 1 ? ix_in0 : 0 ), ( NY_nodes_surf_input_ > 1 ? iy_in0 : 0 ), 0 );
            double unit_py =
                ( W_shell_sum > 1e-9 ) ?
                    base_y_sum / W_shell_sum :
                    h_shell_coords_managed_(
                        sd, ( NX_nodes_surf_input_ > 1 ? ix_in0 : 0 ), ( NY_nodes_surf_input_ > 1 ? iy_in0 : 0 ), 1 );
            double unit_pz =
                ( W_shell_sum > 1e-9 ) ?
                    base_z_sum / W_shell_sum :
                    h_shell_coords_managed_(
                        sd, ( NX_nodes_surf_input_ > 1 ? ix_in0 : 0 ), ( NY_nodes_surf_input_ > 1 ? iy_in0 : 0 ), 2 );

            double radius_sum   = 0.0;
            double W_radius_sum = 0.0;
            double wr_param     = ( NR_nodes_rad_input_ > 1 ) ? ( ir_out % 2 ) * 0.5 : 0.0;

            for ( int k_off = 0; k_off <= ( ( wr_param > 0.0 && ir_in0 + 1 < NR_nodes_rad_input_ ) ? 1 : 0 ); ++k_off )
            {
                size_t ir_in = ir_in0 + k_off;
                double R_k   = ( k_off == 0 ) ? ( 1.0 - wr_param ) : wr_param;
                radius_sum += R_k * h_radii_managed_( sd, ir_in );
                W_radius_sum += R_k;
            }
            double current_radius = ( W_radius_sum > 1e-9 ) ?
                                        radius_sum / W_radius_sum :
                                        h_radii_managed_( sd, ( NR_nodes_rad_input_ > 1 ? ir_in0 : 0 ) );

            final_px = unit_px * current_radius;
            final_py = unit_py * current_radius;
            final_pz = unit_pz * current_radius;
        }
    }

    // --- Helper: Write Cell Connectivity Data ---
    void write_cell_connectivity_data( std::ostream& os )
    {
        if ( num_total_cells_ == 0 )
            return;
        size_t nodes_per_cell_out = is_quadratic_ ? 20 : 8;

        size_t num_cells_x_surf = ( NX_nodes_surf_input_ > 1 ) ? NX_nodes_surf_input_ - 1 : 0;
        size_t num_cells_y_surf = ( NY_nodes_surf_input_ > 1 ) ? NY_nodes_surf_input_ - 1 : 0;
        size_t num_cells_r_rad  = ( NR_nodes_rad_input_ > 1 ) ? NR_nodes_rad_input_ - 1 : 0;

        for ( size_t sd = 0; sd < num_subdomains_; ++sd )
        {
            for ( size_t k_cell_in = 0; k_cell_in < num_cells_r_rad; ++k_cell_in )
            {
                for ( size_t j_cell_in = 0; j_cell_in < num_cells_y_surf; ++j_cell_in )
                {
                    for ( size_t i_cell_in = 0; i_cell_in < num_cells_x_surf; ++i_cell_in )
                    {
                        // ... (same logic as before to get cell_node_ids for one cell) ...
                        std::vector< size_t > cell_node_ids( nodes_per_cell_out );
                        if ( is_quadratic_ )
                        {
                            size_t ix0_out = i_cell_in * 2;
                            size_t ix1_out = ix0_out + 1;
                            size_t ix2_out = ix0_out + 2;
                            size_t iy0_out = j_cell_in * 2;
                            size_t iy1_out = iy0_out + 1;
                            size_t iy2_out = iy0_out + 2;
                            size_t ir0_out = k_cell_in * 2;
                            size_t ir1_out = ir0_out + 1;
                            size_t ir2_out = ir0_out + 2;

                            cell_node_ids[0]  = get_global_output_node_id( sd, ix0_out, iy0_out, ir0_out );
                            cell_node_ids[1]  = get_global_output_node_id( sd, ix2_out, iy0_out, ir0_out );
                            cell_node_ids[2]  = get_global_output_node_id( sd, ix2_out, iy2_out, ir0_out );
                            cell_node_ids[3]  = get_global_output_node_id( sd, ix0_out, iy2_out, ir0_out );
                            cell_node_ids[4]  = get_global_output_node_id( sd, ix0_out, iy0_out, ir2_out );
                            cell_node_ids[5]  = get_global_output_node_id( sd, ix2_out, iy0_out, ir2_out );
                            cell_node_ids[6]  = get_global_output_node_id( sd, ix2_out, iy2_out, ir2_out );
                            cell_node_ids[7]  = get_global_output_node_id( sd, ix0_out, iy2_out, ir2_out );
                            cell_node_ids[8]  = get_global_output_node_id( sd, ix1_out, iy0_out, ir0_out );
                            cell_node_ids[9]  = get_global_output_node_id( sd, ix2_out, iy1_out, ir0_out );
                            cell_node_ids[10] = get_global_output_node_id( sd, ix1_out, iy2_out, ir0_out );
                            cell_node_ids[11] = get_global_output_node_id( sd, ix0_out, iy1_out, ir0_out );
                            cell_node_ids[12] = get_global_output_node_id( sd, ix1_out, iy0_out, ir2_out );
                            cell_node_ids[13] = get_global_output_node_id( sd, ix2_out, iy1_out, ir2_out );
                            cell_node_ids[14] = get_global_output_node_id( sd, ix1_out, iy2_out, ir2_out );
                            cell_node_ids[15] = get_global_output_node_id( sd, ix0_out, iy1_out, ir2_out );
                            cell_node_ids[16] = get_global_output_node_id( sd, ix0_out, iy0_out, ir1_out );
                            cell_node_ids[17] = get_global_output_node_id( sd, ix2_out, iy0_out, ir1_out );
                            cell_node_ids[18] = get_global_output_node_id( sd, ix2_out, iy2_out, ir1_out );
                            cell_node_ids[19] = get_global_output_node_id( sd, ix0_out, iy2_out, ir1_out );
                        }
                        else
                        {
                            size_t ix0_out   = i_cell_in;
                            size_t ix1_out   = i_cell_in + 1;
                            size_t iy0_out   = j_cell_in;
                            size_t iy1_out   = j_cell_in + 1;
                            size_t ir0_out   = k_cell_in;
                            size_t ir1_out   = k_cell_in + 1;
                            cell_node_ids[0] = get_global_output_node_id( sd, ix0_out, iy0_out, ir0_out );
                            cell_node_ids[1] = get_global_output_node_id( sd, ix1_out, iy0_out, ir0_out );
                            cell_node_ids[2] = get_global_output_node_id( sd, ix1_out, iy1_out, ir0_out );
                            cell_node_ids[3] = get_global_output_node_id( sd, ix0_out, iy1_out, ir0_out );
                            cell_node_ids[4] = get_global_output_node_id( sd, ix0_out, iy0_out, ir1_out );
                            cell_node_ids[5] = get_global_output_node_id( sd, ix1_out, iy0_out, ir1_out );
                            cell_node_ids[6] = get_global_output_node_id( sd, ix1_out, iy1_out, ir1_out );
                            cell_node_ids[7] = get_global_output_node_id( sd, ix0_out, iy1_out, ir1_out );
                        }
                        os << "          ";
                        for ( size_t i = 0; i < nodes_per_cell_out; ++i )
                        {
                            os << cell_node_ids[i] << ( i == nodes_per_cell_out - 1 ? "" : " " );
                        }
                        os << "\n";
                    }
                }
            }
        }
    }

    // --- Helper: Write Cell Offsets Data ---
    void write_cell_offsets_data( std::ostream& os )
    {
        if ( num_total_cells_ == 0 )
            return;
        size_t  nodes_per_cell_out = is_quadratic_ ? 20 : 8;
        int64_t current_offset     = 0;
        for ( size_t i = 0; i < num_total_cells_; ++i )
        {
            current_offset += nodes_per_cell_out;
            os << "          " << current_offset << "\n";
        }
    }

    // --- Helper: Write Cell Types Data ---
    void write_cell_types_data( std::ostream& os )
    {
        if ( num_total_cells_ == 0 )
            return;
        uint8_t cell_type_out = is_quadratic_ ? 25 : 12;
        for ( size_t i = 0; i < num_total_cells_; ++i )
        {
            os << "          " << static_cast< int >( cell_type_out ) << "\n";
        }
    }

    // --- Helper: Interpolate and Write Field Data ---
    // Templated on the actual stored view type within the variant
    template < typename InputFieldViewType >
    void get_interpolated_field_value_at_output_node(
        const InputFieldViewType& h_field_data_input,
        size_t                    sd,
        size_t                    ix_out,
        size_t                    iy_out,
        size_t                    ir_out,
        int                       num_components,
        std::vector< double >&    interpolated_values ) const
    {
        interpolated_values.assign( num_components, 0.0 );

        bool use_direct_mapping =
            !is_quadratic_ || ( NX_nodes_surf_input_ <= 1 && NY_nodes_surf_input_ <= 1 && NR_nodes_rad_input_ <= 1 );

        if ( use_direct_mapping )
        {
            size_t ix_in = ix_out;
            size_t iy_in = iy_out;
            size_t ir_in = ir_out;
            if ( NX_nodes_surf_input_ == 1 )
                ix_in = 0;
            if ( NY_nodes_surf_input_ == 1 )
                iy_in = 0;
            if ( NR_nodes_rad_input_ == 1 )
                ir_in = 0;

            if constexpr ( std::is_same_v< InputFieldViewType, ScalarFieldHostView > )
            { // Scalar
                interpolated_values[0] = h_field_data_input( sd, ix_in, iy_in, ir_in );
            }
            else
            { // Vector
                for ( int comp = 0; comp < num_components; ++comp )
                {
                    interpolated_values[comp] = h_field_data_input( sd, ix_in, iy_in, ir_in, comp );
                }
            }
        }
        else
        {
            size_t ix_in0 = ix_out / 2;
            size_t iy_in0 = iy_out / 2;
            size_t ir_in0 = ir_out / 2;

            std::vector< double > val_sum( num_components, 0.0 );
            double                W_sum = 0.0;

            double wx_param = ( NX_nodes_surf_input_ > 1 ) ? ( ix_out % 2 ) * 0.5 : 0.0;
            double wy_param = ( NY_nodes_surf_input_ > 1 ) ? ( iy_out % 2 ) * 0.5 : 0.0;
            double wr_param = ( NR_nodes_rad_input_ > 1 ) ? ( ir_out % 2 ) * 0.5 : 0.0;

            for ( int k_off = 0; k_off <= ( ( wr_param > 0.0 && ir_in0 + 1 < NR_nodes_rad_input_ ) ? 1 : 0 ); ++k_off )
            {
                size_t ir_in = ir_in0 + k_off;
                double R_k   = ( k_off == 0 ) ? ( 1.0 - wr_param ) : wr_param;
                for ( int j_off = 0; j_off <= ( ( wy_param > 0.0 && iy_in0 + 1 < NY_nodes_surf_input_ ) ? 1 : 0 );
                      ++j_off )
                {
                    size_t iy_in = iy_in0 + j_off;
                    double W_j   = ( j_off == 0 ) ? ( 1.0 - wy_param ) : wy_param;
                    for ( int i_off = 0; i_off <= ( ( wx_param > 0.0 && ix_in0 + 1 < NX_nodes_surf_input_ ) ? 1 : 0 );
                          ++i_off )
                    {
                        size_t ix_in = ix_in0 + i_off;
                        double W_i   = ( i_off == 0 ) ? ( 1.0 - wx_param ) : wx_param;

                        double weight = W_i * W_j * R_k;
                        if constexpr ( std::is_same_v< InputFieldViewType, ScalarFieldHostView > )
                        { // Scalar
                            val_sum[0] += weight * h_field_data_input( sd, ix_in, iy_in, ir_in );
                        }
                        else
                        { // Vector
                            for ( int comp = 0; comp < num_components; ++comp )
                            {
                                val_sum[comp] += weight * h_field_data_input( sd, ix_in, iy_in, ir_in, comp );
                            }
                        }
                        W_sum += weight;
                    }
                }
            }

            if ( W_sum > 1e-9 )
            {
                for ( int comp = 0; comp < num_components; ++comp )
                {
                    interpolated_values[comp] = val_sum[comp] / W_sum;
                }
            }
            else
            { // Original node or not enough points to interpolate from
                if constexpr ( std::is_same_v< InputFieldViewType, ScalarFieldHostView > )
                {
                    interpolated_values[0] = h_field_data_input(
                        sd,
                        ( NX_nodes_surf_input_ > 1 ? ix_in0 : 0 ),
                        ( NY_nodes_surf_input_ > 1 ? iy_in0 : 0 ),
                        ( NR_nodes_rad_input_ > 1 ? ir_in0 : 0 ) );
                }
                else
                {
                    for ( int comp = 0; comp < num_components; ++comp )
                    {
                        interpolated_values[comp] = h_field_data_input(
                            sd,
                            ( NX_nodes_surf_input_ > 1 ? ix_in0 : 0 ),
                            ( NY_nodes_surf_input_ > 1 ? iy_in0 : 0 ),
                            ( NR_nodes_rad_input_ > 1 ? ir_in0 : 0 ),
                            comp );
                    }
                }
            }
        }
    }

    void write_field_data( std::ostream& os, const PointDataEntry& entry )
    {
        std::variant< ScalarFieldHostView, VectorFieldHostView > host_view;

        if ( entry.num_components == 1 )
        {
            if ( !scalar_field_host_buffer_.has_value() )
            {
                ScalarFieldHostView host_mirror = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), std::get< ScalarFieldDeviceView >( entry.device_view_input_data ) );
                scalar_field_host_buffer_ = host_mirror;
            }
            else
            {
                Kokkos::deep_copy(
                    scalar_field_host_buffer_.value(),
                    std::get< ScalarFieldDeviceView >( entry.device_view_input_data ) );
            }

            host_view = scalar_field_host_buffer_.value();
        }
        else
        {
            if ( !vector_field_host_buffer_.has_value() )
            {
                VectorFieldHostView host_mirror = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), std::get< VectorFieldDeviceView >( entry.device_view_input_data ) );
                vector_field_host_buffer_ = host_mirror;
            }
            else
            {
                Kokkos::deep_copy(
                    vector_field_host_buffer_.value(),
                    std::get< VectorFieldDeviceView >( entry.device_view_input_data ) );
            }

            host_view = vector_field_host_buffer_.value();
        }

        std::vector< double > interpolated_values_buffer( entry.num_components );
        for ( size_t sd = 0; sd < num_subdomains_; ++sd )
        {
            for ( size_t ir_out = 0; ir_out < NR_nodes_rad_output_; ++ir_out )
            {
                for ( size_t iy_out = 0; iy_out < NY_nodes_surf_output_; ++iy_out )
                {
                    for ( size_t ix_out = 0; ix_out < NX_nodes_surf_output_; ++ix_out )
                    {
                        // Use std::visit to call the interpolation helper with the correct view type
                        std::visit(
                            [&]( const auto& view_arg ) {
                                this->get_interpolated_field_value_at_output_node(
                                    view_arg,
                                    sd,
                                    ix_out,
                                    iy_out,
                                    ir_out,
                                    entry.num_components,
                                    interpolated_values_buffer );
                            },
                            host_view );

                        os << "          ";
                        for ( int comp = 0; comp < entry.num_components; ++comp )
                        {
                            os << static_cast< float >( interpolated_values_buffer[comp] )
                               << ( comp == entry.num_components - 1 ? "" : " " );
                        }
                        os << "\n";
                    }
                }
            }
        }
    }

    // --- Member Variables ---
    bool is_quadratic_;

    grid::Grid3DDataVec< double, 3 >::host_mirror_type h_shell_coords_managed_;
    grid::Grid2DDataScalar< double >::host_mirror_type h_radii_managed_;

    size_t num_subdomains_;
    size_t NX_nodes_surf_input_, NY_nodes_surf_input_, NR_nodes_rad_input_;
    size_t NX_nodes_surf_output_, NY_nodes_surf_output_, NR_nodes_rad_output_;

    size_t num_total_points_;
    size_t num_total_cells_;

    std::vector< PointDataEntry > point_data_entries_;

    std::optional< ScalarFieldHostView > scalar_field_host_buffer_;
    std::optional< VectorFieldHostView > vector_field_host_buffer_;
};

} // namespace terra::visualization