
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/optimized_laplace/Laplace0S0.hpp"
#include "fe/wedge/operators/shell/optimized_laplace/Laplace0S1.hpp"
#include "fe/wedge/operators/shell/optimized_laplace/Laplace0S2.hpp"
#include "fe/wedge/operators/shell/optimized_laplace/Laplace1S1.hpp"
#include "fe/wedge/operators/shell/optimized_laplace/Laplace1S2.hpp"
#include "fe/wedge/operators/shell/optimized_laplace/Laplace2S2.hpp"
#include "fe/wedge/operators/shell/optimized_laplace/quadrature.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/bit_masks.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "mpi/mpi.hpp"
#include "util/cli11_helper.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;
using grid::NodeOwnershipFlag;
using grid::shell::ShellBoundaryFlag;
using linalg::OperatorApplyMode;
using linalg::OperatorCommunicationMode;
using util::Table;
using namespace fe::wedge::quadrature;

using ScalarT           = double;
const size_t ITERATIONS = 10;
const size_t WARMUPS    = 5;

struct Context
{
    int                                         level;
    grid::shell::DistributedDomain              domain;
    grid::Grid3DDataVec< ScalarT, 3 >           grid;
    grid::Grid2DDataScalar< ScalarT >           radii;
    grid::Grid4DDataScalar< ShellBoundaryFlag > boundary_mask;
    grid::Grid4DDataScalar< NodeOwnershipFlag > ownership_mask;
    linalg::VectorQ1Scalar< ScalarT >           src;
};

struct Initializer
{
    grid::shell::DistributedDomain    domain_;
    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;
    grid::Grid4DDataScalar< ScalarT > data_;

    Initializer(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        const grid::Grid4DDataScalar< ScalarT >& data )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , data_( data )
    {}

    void initialize()
    {
        Kokkos::parallel_for( "initialize", grid::shell::local_domain_md_range_policy_nodes( domain_ ), *this );
        Kokkos::fence();
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const int subdomain, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarT, 3 > coords = grid::shell::coords( subdomain, x, y, r, grid_, radii_ );
        const ScalarT value = ( 1.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) ) *
                              Kokkos::sin( 2 * coords( 2 ) );
        data_( subdomain, x, y, r ) = value;
    }
};

template < linalg::OperatorLike OperatorT >
double measure_run_time( OperatorT& op, const linalg::SrcOf< OperatorT >& src, linalg::DstOf< OperatorT >& dst )
{
    for ( size_t i = 0; i < WARMUPS; ++i )
    {
        linalg::apply( op, src, dst );
    }
    Kokkos::fence();
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    for ( size_t i = 0; i < ITERATIONS; ++i )
    {
        linalg::apply( op, src, dst );
    }
    Kokkos::fence();
    auto end = high_resolution_clock::now();
    return duration_cast< std::chrono::duration< double > >( end - start ).count() / ITERATIONS;
}

template < linalg::OperatorLike OperatorT >
void run_benchmark( OperatorT& op, const Context& ctx, Table::Row& row, bool compare_baseline = false )
{
    linalg::VectorQ1Scalar< ScalarT > dst( "dst", ctx.domain, ctx.ownership_mask );
    const double                      seconds = measure_run_time( op, ctx.src, dst );
    const auto   n_dofs = kernels::common::count_masked< long >( ctx.ownership_mask, grid::NodeOwnershipFlag::OWNED );
    const double dofs_per_s = static_cast< double >( n_dofs ) / seconds;
    row["runtime"]          = seconds;
    row["dof_throughput"]   = dofs_per_s;
    if ( compare_baseline )
    {
        using Laplace = terra::fe::wedge::operators::shell::Laplace< ScalarT >;
        Laplace op_ref(
            ctx.domain,
            ctx.grid,
            ctx.radii,
            ctx.boundary_mask,
            true,
            false,
            OperatorApplyMode::Add,
            OperatorCommunicationMode::SkipCommunication );
        const double baseline_seconds = measure_run_time( op_ref, ctx.src, dst );
        row["speedup"]                = baseline_seconds / seconds;
    }
}

template < linalg::OperatorLike OperatorT >
void run_test( OperatorT& op, const Context& ctx, Table::Row& row )
{
    using Laplace = terra::fe::wedge::operators::shell::Laplace< ScalarT >;
    Laplace op_ref(
        ctx.domain,
        ctx.grid,
        ctx.radii,
        ctx.boundary_mask,
        true,
        false,
        OperatorApplyMode::Add,
        OperatorCommunicationMode::SkipCommunication );

    linalg::VectorQ1Scalar< ScalarT > dst( "dst", ctx.domain, ctx.ownership_mask );
    linalg::VectorQ1Scalar< ScalarT > dst_ref( "dst-ref", ctx.domain, ctx.ownership_mask );

    linalg::assign( dst, 0.0 );
    linalg::assign( dst_ref, 0.0 );

    linalg::apply( op, ctx.src, dst );
    linalg::apply( op_ref, ctx.src, dst_ref );

    linalg::VectorQ1Scalar< ScalarT > error( "error", ctx.domain, ctx.ownership_mask );
    linalg::lincomb( error, { 1.0, -1.0 }, { dst, dst_ref } );

    ScalarT    inf_error = linalg::norm_inf( error );
    const auto n_dofs    = kernels::common::count_masked< long >( ctx.ownership_mask, grid::NodeOwnershipFlag::OWNED );

    row["l2_error"]      = std::sqrt( dot( error, error ) / n_dofs );
    row["inf_error"]     = inf_error;
    row["rel_inf_error"] = inf_error / linalg::norm_inf( dst_ref );
}

template < linalg::OperatorLike OperatorT >
void run_1_quad_point( const Context& ctx, std::string op_name, Table& table )
{
    Table::Row row;
    row["operator"]    = op_name;
    row["quad_points"] = 1;
    row["precision"]   = std::is_same_v< ScalarT, double > ? "double" : "single";
    row["level"]       = ctx.level;

    OperatorT op(
        ctx.domain,
        ctx.grid,
        ctx.radii,
        ctx.boundary_mask,
        true,
        false,
        OperatorApplyMode::Add,
        OperatorCommunicationMode::SkipCommunication );
    run_benchmark( op, ctx, row );

    table.add_row( row );
}

template < linalg::OperatorLike OperatorT >
void run_6_quad_points( const Context& ctx, std::string op_name, Table& table )
{
    Table::Row row;
    row["operator"]    = op_name;
    row["quad_points"] = 6;
    row["precision"]   = std::is_same_v< ScalarT, double > ? "double" : "single";
    row["level"]       = ctx.level;

    OperatorT op(
        ctx.domain,
        ctx.grid,
        ctx.radii,
        ctx.boundary_mask,
        true,
        false,
        OperatorApplyMode::Add,
        OperatorCommunicationMode::SkipCommunication );
    run_benchmark( op, ctx, row, true );
    run_test( op, ctx, row );

    table.add_row( row );
}

int main( int argc, char** argv )
{
    int level = 8;

    CLI::App app{ "Benchmark laplace operators" };
    util::add_option_with_default( app, "-l,--level", level, "refinement level for both radial and lateral refinement" )
        ->check( CLI::Range( 0, std::numeric_limits< int >::max() ) );

    CLI11_PARSE( app, argc, argv );

    util::terra_initialize( &argc, &argv );

    const auto domain =
        grid::shell::DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, 0.5, 1.0 );
    const auto ownership_mask = grid::setup_node_ownership_mask_data( domain );
    const auto boundary_mask  = grid::shell::setup_boundary_mask_data( domain );

    const auto grid  = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarT >( domain );
    const auto radii = grid::shell::subdomain_shell_radii< ScalarT >( domain );

    const linalg::VectorQ1Scalar< ScalarT > src( "src", domain, ownership_mask );
    Initializer( domain, grid, radii, src.grid_data() ).initialize();

    Context ctx{
        .level          = level,
        .domain         = domain,
        .grid           = grid,
        .radii          = radii,
        .boundary_mask  = boundary_mask,
        .ownership_mask = ownership_mask,
        .src            = src,
    };

    using namespace terra::fe::wedge::operators::shell;

    Table table_q1;
    run_1_quad_point< Laplace0S0< ScalarT, QuadRuleWedge1Point< ScalarT > > >( ctx, "Laplace0S0", table_q1 );
    run_1_quad_point< Laplace0S1< ScalarT, QuadRuleWedge1Point< ScalarT > > >( ctx, "Laplace0S1", table_q1 );
    run_1_quad_point< Laplace1S1< ScalarT, QuadRuleWedge1Point< ScalarT > > >( ctx, "Laplace1S1", table_q1 );
    run_1_quad_point< Laplace0S2< ScalarT, QuadRuleWedge1Point< ScalarT > > >( ctx, "Laplace0S2", table_q1 );
    run_1_quad_point< Laplace1S2< ScalarT, QuadRuleWedge1Point< ScalarT > > >( ctx, "Laplace1S2", table_q1 );
    run_1_quad_point< Laplace2S2< ScalarT, QuadRuleWedge1Point< ScalarT > > >( ctx, "Laplace2S2", table_q1 );
    terra::util::logroot << "1 quad point:" << std::endl;
    table_q1.print_pretty();

    Table table_q6;
    run_6_quad_points< Laplace0S0< ScalarT, QuadRuleWedge6Points< ScalarT > > >( ctx, "Laplace0S0", table_q6 );
    run_6_quad_points< Laplace0S1< ScalarT, QuadRuleWedge6Points< ScalarT > > >( ctx, "Laplace0S1", table_q6 );
    run_6_quad_points< Laplace1S1< ScalarT, QuadRuleWedge6Points< ScalarT > > >( ctx, "Laplace1S1", table_q6 );
    run_6_quad_points< Laplace0S2< ScalarT, QuadRuleWedge6Points< ScalarT > > >( ctx, "Laplace0S2", table_q6 );
    run_6_quad_points< Laplace1S2< ScalarT, QuadRuleWedge6Points< ScalarT > > >( ctx, "Laplace1S2", table_q6 );
    run_6_quad_points< Laplace2S2< ScalarT, QuadRuleWedge6Points< ScalarT > > >( ctx, "Laplace2S2", table_q6 );
    terra::util::logroot << "\n6 quad points:" << std::endl;
    table_q6.print_pretty();
}
