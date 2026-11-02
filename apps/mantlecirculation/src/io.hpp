
#pragma once

#include <vector>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "io/xdmf.hpp"
#include "linalg/vector_w1_iso_w2_w1.hpp"
#include "parameters.hpp"
#include "shell/radial_profiles.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/result.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorW1IsoW2W1;
using linalg::VectorW1Scalar;
using linalg::VectorW1Vec;
using util::logroot;
using util::Ok;
using util::Result;

using terra::kernels::common::scale;

using ScalarType = double;

namespace terra::mantlecirculation {

inline Result<> create_directories( const IOParameters& io_parameters )
{
    const auto xdmf_dir            = io_parameters.outdir + "/" + io_parameters.xdmf_dir;
    const auto radial_profiles_dir = io_parameters.outdir + "/" + io_parameters.radial_profiles_out_dir;
    const auto timer_trees_dir     = io_parameters.outdir + "/" + io_parameters.timer_trees_dir;

    if ( !io_parameters.overwrite && std::filesystem::exists( io_parameters.outdir ) )
    {
        return { "Will not overwrite existing directory (to not accidentally delete old simulation data). "
                 "Use -h for help and look for an overwrite option or choose a different output dir name." };
    }

    util::prepare_empty_directory( io_parameters.outdir );
    util::prepare_empty_directory( xdmf_dir );
    util::prepare_empty_directory( radial_profiles_dir );
    util::prepare_empty_directory( timer_trees_dir );

    return { Ok{} };
}

// Xdmf output bundle
template < typename FieldType >
struct ScalableField
{
    FieldType* data;
    ScalarType scale_factor;
    bool       restore;

    ScalableField( FieldType& data_, ScalarType scale_factor_, bool restore_ = true )
    : data( &data_ )
    , scale_factor( scale_factor_ )
    , restore( restore_ )
    {}
};

using ScalarXdmfField = ScalableField< Grid4DDataScalar< ScalarType > >;
using VectorXdmfField = ScalableField< Grid4DDataVec< ScalarType, 3 > >;

struct XdmfFields
{
    std::vector< ScalarXdmfField >   scalar_fields;
    std::vector< VectorXdmfField >   vector_fields;
    std::optional< ScalarXdmfField > pressure_field;
};

inline Result<> write_xdmf(
    std::optional< io::XDMFOutput< ScalarType > >& xdmf_output,
    std::optional< io::XDMFOutput< ScalarType > >& xdmf_output_pressure,
    int                                            timestep,
    ScalarType                                     simulated_time_Ma,
    bool                                           output_per_Ma,
    bool                                           output_dimensional,
    const std::vector< ScalarXdmfField >&          scalar_fields,
    const std::vector< VectorXdmfField >&          vector_fields,
    const std::optional< ScalarXdmfField >&        pressure_field )
{
    logroot << "Writing XDMF output ..." << std::endl;

    if ( output_dimensional )
    {
        // Redimensionalise ...
        for ( auto& f : scalar_fields )
            scale( *f.data, f.scale_factor );
        for ( auto& f : vector_fields )
            scale( *f.data, f.scale_factor );

        xdmf_output->write( timestep, simulated_time_Ma, output_per_Ma );

        // ... and nondimensionalise again, if required
        for ( auto& f : scalar_fields )
            if ( f.restore )
                scale( *f.data, ScalarType( 1 ) / f.scale_factor );
        for ( auto& f : vector_fields )
            if ( f.restore )
                scale( *f.data, ScalarType( 1 ) / f.scale_factor );

        // Redim, write and nondim pressure
        if ( pressure_field )
        {
            scale( *pressure_field->data, pressure_field->scale_factor );

            xdmf_output_pressure->write( timestep, simulated_time_Ma, output_per_Ma );

            if ( pressure_field->restore )
                scale( *pressure_field->data, ScalarType( 1 ) / pressure_field->scale_factor );
        }
    }
    else
    {
        xdmf_output->write( timestep );
        if ( pressure_field )
            xdmf_output_pressure->write( timestep );
    }
    return { Ok{} };
}

inline Result<> compute_and_write_radial_profiles(
    const VectorW1Scalar< ScalarType >& scalar_function,
    const Grid2DDataScalar< int >&      subdomain_shell_idx,
    const std::vector< double >&        radial_coords,
    const Parameters&                   prm,
    const int                           out_tag,
    const ScalarType                    field_scale  = ScalarType( 1 ),
    const std::string&                  radial_label = "radius" )
{
    const auto profiles = shell::radial_profiles_to_table< ScalarType >(
        shell::radial_profiles( scalar_function, subdomain_shell_idx, static_cast< int >( radial_coords.size() ) ),
        radial_coords,
        field_scale,
        radial_label );

    if ( mpi::rank() == 0 )
    {
        std::ofstream out(
            prm.io_parameters.outdir + "/" + prm.io_parameters.radial_profiles_out_dir + "/radial_profiles_" +
            scalar_function.grid_data().label() + "_" + std::to_string( out_tag ) + ".csv" );
        profiles.print_csv( out );
    }

    return { Ok{} };
}

/// Decompose a W1 velocity field into its radial (signed) and tangential (magnitude)
/// components and write radial profiles for both, reusing
/// `compute_and_write_radial_profiles`. Output files are named
/// `radial_profiles_u_r_<step>.csv` and `radial_profiles_u_t_<step>.csv`.
inline Result<> compute_and_write_velocity_radial_profiles(
    const linalg::VectorW1Vec< ScalarType, 3 >&              velocity,
    const grid::Grid3DDataVec< ScalarType, 3 >&              coords_shell,
    const grid::Grid2DDataScalar< int >&                     subdomain_shell_idx,
    const DistributedDomain&                                 domain,
    const std::vector< double >&                             radial_coords,
    const grid::Grid4DDataScalar< grid::NodeOwnershipFlag >& mask,
    const Parameters&                                        prm,
    const int                                                out_tag,
    const ScalarType                                         field_scale  = ScalarType( 1 ),
    const std::string&                                       radial_label = "radius" )
{
    VectorW1Scalar< ScalarType > u_r( "u_r", domain, mask );
    VectorW1Scalar< ScalarType > u_t( "u_t", domain, mask );

    const auto u_grid   = velocity.grid_data();
    auto       u_r_grid = u_r.grid_data();
    auto       u_t_grid = u_t.grid_data();

    Kokkos::parallel_for(
        "decompose velocity into radial and tangential",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0 }, { u_grid.extent( 0 ), u_grid.extent( 1 ), u_grid.extent( 2 ), u_grid.extent( 3 ) } ),
        KOKKOS_LAMBDA( int sd, int x, int y, int r ) {
            const ScalarType ux      = u_grid( sd, x, y, r, 0 );
            const ScalarType uy      = u_grid( sd, x, y, r, 1 );
            const ScalarType uz      = u_grid( sd, x, y, r, 2 );
            const ScalarType rx      = coords_shell( sd, x, y, 0 );
            const ScalarType ry      = coords_shell( sd, x, y, 1 );
            const ScalarType rz      = coords_shell( sd, x, y, 2 );
            const ScalarType ur_node = ux * rx + uy * ry + uz * rz;
            const ScalarType umag2   = ux * ux + uy * uy + uz * uz;
            const ScalarType ut2     = umag2 - ur_node * ur_node;
            u_r_grid( sd, x, y, r )  = ur_node;
            u_t_grid( sd, x, y, r )  = ( ut2 > ScalarType( 0 ) ) ? Kokkos::sqrt( ut2 ) : ScalarType( 0 );
        } );
    Kokkos::fence();

    auto res1 = compute_and_write_radial_profiles(
        u_r, subdomain_shell_idx, radial_coords, prm, out_tag, field_scale, radial_label );
    if ( res1.is_err() )
    {
        return res1;
    }
    return compute_and_write_radial_profiles(
        u_t, subdomain_shell_idx, radial_coords, prm, out_tag, field_scale, radial_label );
}

inline Result<> write_timer_tree( const IOParameters& io_parameters, const int out_tag )
{
    util::TimerTree::instance().aggregate_mpi();
    if ( mpi::rank() == 0 )
    {
        const auto timer_tree_file = io_parameters.outdir + "/" + io_parameters.timer_trees_dir + "/timer_tree_" +
                                     std::to_string( out_tag ) + ".json";
        logroot << "Writing timer tree to " << timer_tree_file << std::endl;
        std::ofstream out( timer_tree_file );
        out << util::TimerTree::instance().json_aggregate();
        out.close();
    }

    return { Ok{} };
}

} // namespace terra::mantlecirculation
