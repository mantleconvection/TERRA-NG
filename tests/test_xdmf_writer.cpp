#include <fstream>
#include <iomanip>
#include <linalg/vector_q1.hpp>
#include <optional>

#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/io/xdmf.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"

struct SomeInterpolator
{
    terra::grid::Grid3DDataVec< double, 3 > shell_coords_;
    terra::grid::Grid2DDataScalar< double > radii_;
    terra::grid::Grid4DDataScalar< double > scalar_data_;

    SomeInterpolator(
        const terra::grid::Grid3DDataVec< double, 3 >& shell_coords,
        const terra::grid::Grid2DDataScalar< double >& radii,
        const terra::grid::Grid4DDataScalar< double >& scalar_data,
        const double                                   t )
    : shell_coords_( shell_coords )
    , radii_( radii )
    , scalar_data_( scalar_data )
    , t_( t )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int subdomain, const int x, const int y, const int r ) const
    {
        const terra::dense::Vec< double, 3 > coords =
            terra::grid::shell::coords( subdomain, x, y, r, shell_coords_, radii_ );

        const double value = coords( 0 ) * Kokkos::sin( t_ + coords( 1 ) ) * Kokkos::cos( coords( 2 ) );

        scalar_data_( subdomain, x, y, r ) = value;
    }

    double t_;
};

struct SomeVecInterpolator
{
    terra::grid::Grid3DDataVec< double, 3 > shell_coords_;
    terra::grid::Grid2DDataScalar< double > radii_;
    terra::grid::Grid4DDataVec< double, 3 > vec_data_;

    SomeVecInterpolator(
        const terra::grid::Grid3DDataVec< double, 3 >& shell_coords,
        const terra::grid::Grid2DDataScalar< double >& radii,
        const terra::grid::Grid4DDataVec< double, 3 >& vec_data,
        const double                                   t )
    : shell_coords_( shell_coords )
    , radii_( radii )
    , vec_data_( vec_data )
    , t_( t )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int subdomain, const int x, const int y, const int r ) const
    {
        const terra::dense::Vec< double, 3 > coords =
            terra::grid::shell::coords( subdomain, x, y, r, shell_coords_, radii_ );

        const auto radius = coords.norm();
        vec_data_( subdomain, x, y, r, 0 ) =
            ( 1.0 - radius * radius ) * coords( 0 ) + Kokkos::sin( M_PI * radius ) * ( -coords( 1 ) ) * t_;
        vec_data_( subdomain, x, y, r, 1 ) =
            ( 1.0 - radius * radius ) * coords( 1 ) + Kokkos::sin( M_PI * radius ) * ( coords( 0 ) ) * t_;
        vec_data_( subdomain, x, y, r, 2 ) =
            ( 1.0 - radius * radius ) * coords( 2 ) + Kokkos::cos( M_PI * radius ) * coords( 2 ) * t_;
    }

    double t_;
};

struct RankInterpolator
{
    terra::grid::Grid3DDataVec< double, 3 > shell_coords_;
    terra::grid::Grid2DDataScalar< double > radii_;
    terra::grid::Grid4DDataScalar< double > scalar_data_;

    RankInterpolator(
        const terra::grid::Grid3DDataVec< double, 3 >& shell_coords,
        const terra::grid::Grid2DDataScalar< double >& radii,
        const terra::grid::Grid4DDataScalar< double >& scalar_data )
    : shell_coords_( shell_coords )
    , radii_( radii )
    , scalar_data_( scalar_data )
    , rank_( terra::mpi::rank() )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int subdomain, const int x, const int y, const int r ) const
    {
        scalar_data_( subdomain, x, y, r ) = static_cast< double >( rank_ );
    }

    int rank_;
};

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    constexpr int    lateral_refinement_level = 4;
    constexpr int    radial_refinement_level  = 4;
    constexpr double r_min                    = 0.5;
    constexpr double r_max                    = 1.0;

    const auto domain = terra::grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond(
        lateral_refinement_level, radial_refinement_level, r_min, r_max );

    const auto subdomain_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii< double >( domain );

    auto mask_data = terra::grid::setup_node_ownership_mask_data( domain );

    auto rank = terra::grid::shell::allocate_scalar_grid< double >( "rank_data", domain );

    terra::linalg::VectorQ1Scalar< double > data_scalar( "scalar_data", domain, mask_data );
    terra::linalg::VectorQ1Vec< double >    data_vec( "vec_data", domain, mask_data );

    terra::util::prepare_empty_directory( "test_xdmf_writer_output" );

    terra::io::XDMFOutput xdmf( "test_xdmf_writer_output", domain, subdomain_shell_coords, subdomain_radii );
    xdmf.add( data_scalar.grid_data() );
    xdmf.add( data_vec.grid_data() );
    xdmf.add( rank );

    Kokkos::parallel_for(
        "rank_interpolation",
        terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
        RankInterpolator( subdomain_shell_coords, subdomain_radii, rank ) );

    for ( int i = 0; i < 5; ++i )
    {
        Kokkos::parallel_for(
            "some_interpolation",
            terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
            SomeInterpolator( subdomain_shell_coords, subdomain_radii, data_scalar.grid_data(), 0.1 * i ) );

        Kokkos::parallel_for(
            "some_interpolation",
            terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
            SomeVecInterpolator( subdomain_shell_coords, subdomain_radii, data_vec.grid_data(), 0.1 * i ) );

        xdmf.write( i );
    }

    return 0;
}
