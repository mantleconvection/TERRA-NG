#pragma once

/// @file  divergence_kerngen.hpp
/// @brief Team-based matrix-free Divergence operator for the spherical shell.
///
/// This is a performance-oriented variant of `Divergence`
/// (`fe/wedge/operators/shell/divergence.hpp`) that transfers the optimisation
/// techniques used in `EpsilonDivDivKerngen`:
///   - `Kokkos::TeamPolicy` with backend-aware tile sizing.
///   - Per-team shared-memory staging of coords, radii and velocity src.
///   - Host-side BC-aware kernel path dispatch (Slow / FastDirichletNeumann /
///     FastFreeslip) with no in-kernel branching on the path.
///   - `LaunchBounds<128, 5>` for occupancy tuning on CUDA.
///   - `ShellBoundaryCommPlan`-based halo exchange.
///
/// The kernel path _math_ is intentionally kept identical to the original
/// `Divergence`: every cell assembles the same local 6x18 element matrix
/// (`dense::Mat< ScalarT, 6, 18 > A[2]`), applies the same boundary mask /
/// freeslip rotation, and scatters through the same
/// `atomically_add_local_wedge_scalar_coefficients` helper onto the coarse
/// pressure grid `(x/2, y/2, r/2)`. The only difference in the fast paths is
/// that per-cell input data is read from team shared memory rather than from
/// global memory. This gives a safe, first-pass optimisation that keeps
/// correctness provable against the existing operator.

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell {

using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using grid::shell::get_boundary_condition_flag;
using terra::grid::shell::BoundaryConditionFlag;
using terra::grid::shell::BoundaryConditions;
using terra::grid::shell::ShellBoundaryFlag;
using terra::linalg::trafo::trafo_mat_cartesian_to_normal_tangential;

template < typename ScalarT >
class DivergenceKerngen
{
  public:
    using SrcVectorType = linalg::VectorQ1Vec< ScalarT, 3 >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;
    using Team          = Kokkos::TeamPolicy<>::member_type;

    enum class KernelPath
    {
        Slow,
        FastDirichletNeumann,
        FastFreeslip,
    };

  private:
    grid::shell::DistributedDomain domain_fine_;
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_fine_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_fine_;

    BoundaryConditions bcs_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT >                    recv_buffers_;
    terra::communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataScalar< ScalarT > > comm_plan_;

    // Captured by kernels
    grid::Grid4DDataVec< ScalarType, 3 > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    // Tiling (mirror of EpsilonDivDivKerngen)
    int local_subdomains_;
    int hex_lat_;   // fine cells per side
    int hex_rad_;   // fine cells radially
    int lat_tile_;
    int r_tile_;
    int r_passes_;
    int r_tile_block_;
    int lat_tiles_;
    int r_tiles_;
    int team_size_;
    int blocks_;

    KernelPath kernel_path_ = KernelPath::FastDirichletNeumann;

    /// Host-side path selection, updated whenever BCs change.
    void update_kernel_path_flag_host_only()
    {
        // Serial backend: keep the slow path (fast paths assume a real team of cooperative threads).
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > )
        {
            kernel_path_ = KernelPath::Slow;
            return;
        }

        const BoundaryConditionFlag cmb_bc     = get_boundary_condition_flag( bcs_, CMB );
        const BoundaryConditionFlag surface_bc = get_boundary_condition_flag( bcs_, SURFACE );
        const bool                  has_freeslip = ( cmb_bc == FREESLIP ) || ( surface_bc == FREESLIP );

        kernel_path_ = has_freeslip ? KernelPath::FastFreeslip : KernelPath::FastDirichletNeumann;
    }

  public:
    DivergenceKerngen(
        const grid::shell::DistributedDomain&                           domain_fine,
        const grid::shell::DistributedDomain&                           domain_coarse,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid_fine,
        const grid::Grid2DDataScalar< ScalarT >&                        radii_fine,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_fine,
        BoundaryConditions                                              bcs,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_fine_( domain_fine )
    , domain_coarse_( domain_coarse )
    , grid_fine_( grid_fine )
    , radii_( radii_fine )
    , boundary_mask_fine_( boundary_mask_fine )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , recv_buffers_( domain_coarse )
    , comm_plan_( domain_coarse )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];

        const grid::shell::DomainInfo& domain_info = domain_fine_.domain_info();
        local_subdomains_ = domain_fine_.subdomains().size();
        hex_lat_          = domain_info.subdomain_num_nodes_per_side_laterally() - 1;
        hex_rad_          = domain_info.subdomain_num_nodes_radially() - 1;

        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > )
        {
            lat_tile_ = 1; r_tile_ = 1; r_passes_ = 1;
        }
#ifdef KOKKOS_ENABLE_OPENMP
        else if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::OpenMP > )
        {
            const int max_team = std::min( Kokkos::OpenMP().concurrency(),
                                           static_cast< int >( Kokkos::Impl::HostThreadTeamData::max_team_members ) );
            if ( max_team >= 64 )      { lat_tile_ = 4; r_tile_ = 4; r_passes_ = 4; }
            else if ( max_team >= 16 ) { lat_tile_ = 4; r_tile_ = 1; r_passes_ = 16; }
            else                        { lat_tile_ = 1; r_tile_ = 1; r_passes_ = 1; }
        }
#endif
        else
        {
            lat_tile_ = 4; r_tile_ = 8; r_passes_ = 2;
        }
        r_tile_block_ = r_tile_ * r_passes_;
        lat_tiles_    = ( hex_lat_ + lat_tile_ - 1 ) / lat_tile_;
        r_tiles_      = ( hex_rad_ + r_tile_block_ - 1 ) / r_tile_block_;
        team_size_    = lat_tile_ * lat_tile_ * r_tile_;
        blocks_       = local_subdomains_ * lat_tiles_ * lat_tiles_ * r_tiles_;

        update_kernel_path_flag_host_only();

        util::logroot << "[DivergenceKerngen] tile=(" << lat_tile_ << "," << lat_tile_ << "," << r_tile_
                      << "), r_passes=" << r_passes_ << ", team=" << team_size_ << ", blocks=" << blocks_
                      << ", path=" << path_name() << std::endl;
    }

    const char* path_name() const
    {
        switch ( kernel_path_ )
        {
        case KernelPath::Slow: return "slow";
        case KernelPath::FastFreeslip: return "fast-freeslip";
        default: return "fast-dirichlet-neumann";
        }
    }

    KernelPath kernel_path() const { return kernel_path_; }

    /// Force the slow path. Useful for validation/testing the fast paths against a reference.
    void force_slow_path() { kernel_path_ = KernelPath::Slow; }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    // -------------------------------------------------------------------------
    // Apply
    // -------------------------------------------------------------------------
    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "divergence_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        util::Timer          timer_kernel( "divergence_kernel" );
        Kokkos::TeamPolicy<> policy( blocks_, team_size_ );
        if ( kernel_path_ != KernelPath::Slow )
        {
            policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );
        }

        if ( kernel_path_ == KernelPath::Slow )
        {
            Kokkos::parallel_for(
                "divergence_apply_kernel_slow", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->run_team_slow( team );
                } );
        }
        else if ( kernel_path_ == KernelPath::FastFreeslip )
        {
            Kokkos::parallel_for(
                "divergence_apply_kernel_fast_fs", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->template run_team_fast< /*Freeslip=*/true >( team );
                } );
        }
        else
        {
            Kokkos::TeamPolicy< Kokkos::LaunchBounds< 128, 5 > > dn_policy( blocks_, team_size_ );
            dn_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );
            Kokkos::parallel_for(
                "divergence_apply_kernel_fast_dn", dn_policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->template run_team_fast< /*Freeslip=*/false >( team );
                } );
        }

        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "divergence_comm" );
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

    // -------------------------------------------------------------------------
    // Team shmem sizing & decode helpers
    // -------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size( const int /*ts*/ ) const
    {
        const int nlev = r_tile_block_ + 1;
        const int n    = lat_tile_ + 1;
        const int nxy  = n * n;
        // coords_sh(nxy,3) + src_sh(nxy,3,nlev) + r_sh(nlev)
        const size_t nscalars = size_t( nxy ) * 3 + size_t( nxy ) * 3 * nlev + size_t( nlev );
        return sizeof( ScalarType ) * nscalars;
    }

  private:
    KOKKOS_INLINE_FUNCTION
    void decode_team_indices(
        const Team& team,
        int&        local_subdomain_id,
        int&        x0,
        int&        y0,
        int&        r0,
        int&        tx,
        int&        ty,
        int&        tr,
        int&        x_cell,
        int&        y_cell,
        int&        r_cell ) const
    {
        int       tmp      = team.league_rank();
        const int r_tile_id = tmp % r_tiles_;
        tmp /= r_tiles_;
        const int lat_y_id = tmp % lat_tiles_;
        tmp /= lat_tiles_;
        const int lat_x_id = tmp % lat_tiles_;
        tmp /= lat_tiles_;
        local_subdomain_id = tmp;

        x0 = lat_x_id * lat_tile_;
        y0 = lat_y_id * lat_tile_;
        r0 = r_tile_id * r_tile_block_;

        const int tid = team.team_rank();
        tr            = tid % r_tile_;
        tx            = ( tid / r_tile_ ) % lat_tile_;
        ty            = tid / ( r_tile_ * lat_tile_ );

        x_cell = x0 + tx;
        y_cell = y0 + ty;
        r_cell = r0 + tr;
    }

    KOKKOS_INLINE_FUNCTION
    bool has_flag( int s, int x, int y, int r, grid::shell::ShellBoundaryFlag flag ) const
    {
        return util::has_flag( boundary_mask_fine_( s, x, y, r ), flag );
    }

    // -------------------------------------------------------------------------
    // Slow path: team-policy wrapper that calls the original per-cell kernel.
    // Kept verbatim (same math) as divergence.hpp so it serves as the reference.
    // -------------------------------------------------------------------------
    KOKKOS_INLINE_FUNCTION
    void run_team_slow( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        for ( int pass = 0; pass < r_passes_; ++pass )
        {
            const int r_cell_pass = r0 + pass * r_tile_ + tr;
            if ( r_cell_pass >= hex_rad_ )
                break;
            if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ )
                continue;

            process_cell_from_global( local_subdomain_id, x_cell, y_cell, r_cell_pass );
        }
    }

    // Per-cell compute: reads src/coords/radii directly from global memory.
    // Bit-identical (modulo FMA reordering) to Divergence::operator() in divergence.hpp.
    KOKKOS_INLINE_FUNCTION
    void process_cell_from_global( int s, int x_cell, int y_cell, int r_cell ) const
    {
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_fine_, s, x_cell, y_cell );

        const ScalarT r_1 = radii_( s, r_cell );
        const ScalarT r_2 = radii_( s, r_cell + 1 );

        dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
        ScalarT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];
        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );

        dense::Vec< ScalarT, 18 > src_loc[num_wedges_per_hex_cell];
        for ( int d = 0; d < 3; d++ )
        {
            dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
            extract_local_wedge_vector_coefficients( src_d, s, x_cell, y_cell, r_cell, d, src_ );
            for ( int w = 0; w < num_wedges_per_hex_cell; ++w )
                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                    src_loc[w]( d * 6 + i ) = src_d[w]( i );
        }

        eval_cell( s, x_cell, y_cell, r_cell, r_1, r_2, wedge_phy_surf, quad_points, quad_weights, src_loc );
    }

    // Shared core: given per-cell inputs already gathered, assemble local 6x18 A matrices,
    // apply boundary treatment, matvec, scatter to coarse grid. Used by both slow and fast paths.
    KOKKOS_INLINE_FUNCTION
    void eval_cell(
        const int                                        s,
        const int                                        x_cell,
        const int                                        y_cell,
        const int                                        r_cell,
        const ScalarT                                    r_1,
        const ScalarT                                    r_2,
        const dense::Vec< ScalarT, 3 > ( &wedge_phy_surf )[num_wedges_per_hex_cell][num_nodes_per_wedge_surface],
        const dense::Vec< ScalarT, 3 > ( &quad_points )[quadrature::quad_felippa_1x1_num_quad_points],
        const ScalarT ( &quad_weights )[quadrature::quad_felippa_1x1_num_quad_points],
        const dense::Vec< ScalarT, 18 > ( &src_in )[num_wedges_per_hex_cell] ) const
    {
        constexpr int num_quad_points     = quadrature::quad_felippa_1x1_num_quad_points;
        const int     fine_radial_wedge_index = r_cell % 2;

        dense::Mat< ScalarT, 6, 18 > A[num_wedges_per_hex_cell] = {};

        for ( int q = 0; q < num_quad_points; q++ )
        {
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const int fine_lateral_wedge_index = fine_lateral_wedge_idx( x_cell, y_cell, wedge );
                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i =
                        shape_coarse( i, fine_radial_wedge_index, fine_lateral_wedge_index, quad_points[q] );

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto grad_j = grad_shape( j, quad_points[q] );
                        for ( int d = 0; d < 3; d++ )
                        {
                            A[wedge]( i, d * 6 + j ) +=
                                quad_weights[q] * ( -( J_inv_transposed * grad_j )( d ) * shape_i * det );
                        }
                    }
                }
            }
        }

        bool at_cmb     = util::has_flag( boundary_mask_fine_( s, x_cell, y_cell, r_cell ), CMB );
        bool at_surface = util::has_flag( boundary_mask_fine_( s, x_cell, y_cell, r_cell + 1 ), SURFACE );

        // Locally modify src + A for FreeSlip / Dirichlet.
        dense::Vec< ScalarT, 18 >    src_loc[num_wedges_per_hex_cell];
        for ( int w = 0; w < num_wedges_per_hex_cell; ++w )
            src_loc[w] = src_in[w];

        dense::Mat< ScalarT, 6, 18 > boundary_mask;
        boundary_mask.fill( 1.0 );

        if ( at_cmb || at_surface )
        {
            ShellBoundaryFlag     sbf = at_cmb ? CMB : SURFACE;
            BoundaryConditionFlag bcf = get_boundary_condition_flag( bcs_, sbf );

            if ( bcf == DIRICHLET )
            {
                for ( int dimj = 0; dimj < 3; ++dimj )
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                            if ( ( at_cmb && j < 3 ) || ( at_surface && j >= 3 ) )
                                boundary_mask( i, dimj * num_nodes_per_wedge + j ) = 0.0;
            }
            else if ( bcf == FREESLIP )
            {
                dense::Mat< ScalarT, 6, 18 > A_tmp[num_wedges_per_hex_cell] = { 0 };

                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    for ( int node_idxi = 0; node_idxi < num_nodes_per_wedge; ++node_idxi )
                        for ( int dimj = 0; dimj < 3; ++dimj )
                            for ( int node_idxj = 0; node_idxj < num_nodes_per_wedge; ++node_idxj )
                                A_tmp[wedge]( node_idxi, node_idxj * 3 + dimj ) =
                                    A[wedge]( node_idxi, node_idxj + dimj * num_nodes_per_wedge );
                    reorder_local_dofs( DoFOrdering::DIMENSIONWISE, DoFOrdering::NODEWISE, src_loc[wedge] );
                }

                constexpr int layer_hex_offset_x[2][3] = { { 0, 1, 0 }, { 1, 0, 1 } };
                constexpr int layer_hex_offset_y[2][3] = { { 0, 0, 1 }, { 1, 1, 0 } };

                dense::Mat< ScalarT, 18, 18 > R[num_wedges_per_hex_cell];
                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    for ( int i = 0; i < 18; ++i )
                        R[wedge]( i, i ) = 1.0;

                    for ( int bn = 0; bn < 3; ++bn )
                    {
                        dense::Vec< double, 3 > normal = grid::shell::coords(
                            s,
                            x_cell + layer_hex_offset_x[wedge][bn],
                            y_cell + layer_hex_offset_y[wedge][bn],
                            r_cell + ( at_cmb ? 0 : 1 ),
                            grid_fine_,
                            radii_ );

                        auto      R_i          = trafo_mat_cartesian_to_normal_tangential( normal );
                        const int offset_in_R  = at_cmb ? 0 : 9;
                        for ( int dimi = 0; dimi < 3; ++dimi )
                            for ( int dimj = 0; dimj < 3; ++dimj )
                                R[wedge]( offset_in_R + bn * 3 + dimi, offset_in_R + bn * 3 + dimj ) = R_i( dimi, dimj );
                    }

                    A[wedge] = A_tmp[wedge] * R[wedge].transposed();

                    auto src_tmp = R[wedge] * src_loc[wedge];
                    for ( int i = 0; i < 18; ++i )
                        src_loc[wedge]( i ) = src_tmp( i );

                    const int node_start = at_surface ? 3 : 0;
                    const int node_end   = at_surface ? 6 : 3;
                    for ( int node_idx = node_start; node_idx < node_end; ++node_idx )
                    {
                        const int idx = node_idx * 3;
                        for ( int k = 0; k < 6; ++k )
                            boundary_mask( k, idx ) = 0.0;
                    }
                }
            }
            else if ( bcf == NEUMANN )
            {
                // No mask modification for Neumann (matches legacy Divergence).
            }
        }

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            A[wedge].hadamard_product( boundary_mask );

        dense::Vec< ScalarT, 6 > dst_loc[num_wedges_per_hex_cell];
        dst_loc[0] = A[0] * src_loc[0];
        dst_loc[1] = A[1] * src_loc[1];

        atomically_add_local_wedge_scalar_coefficients(
            dst_, s, x_cell / 2, y_cell / 2, r_cell / 2, dst_loc );
    }

    // -------------------------------------------------------------------------
    // Fast paths: team-policy with shmem-cached coords / radii / src velocity.
    // Templated on Freeslip so the rotation code is dead-eliminated on the DN path.
    // -------------------------------------------------------------------------
    template < bool Freeslip >
    KOKKOS_INLINE_FUNCTION
    void run_team_fast( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        const int nlev = r_tile_block_ + 1;
        const int n    = lat_tile_ + 1;
        const int nxy  = n * n;

        double* shmem =
            reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size( team.team_size() ) ) );

        using ScratchCoords =
            Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchSrc =
            Kokkos::View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchR =
            Kokkos::View< double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;

        ScratchCoords coords_sh( shmem, nxy, 3 );
        shmem += nxy * 3;
        ScratchSrc src_sh( shmem, nxy, 3, nlev );
        shmem += nxy * 3 * nlev;
        ScratchR r_sh( shmem, nlev );

        auto node_id = [&]( int nx, int ny ) -> int { return nx + n * ny; };

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy ), [&]( int id ) {
            const int dxn = id % n;
            const int dyn = id / n;
            const int xi  = x0 + dxn;
            const int yi  = y0 + dyn;
            if ( xi <= hex_lat_ && yi <= hex_lat_ )
            {
                coords_sh( id, 0 ) = grid_fine_( local_subdomain_id, xi, yi, 0 );
                coords_sh( id, 1 ) = grid_fine_( local_subdomain_id, xi, yi, 1 );
                coords_sh( id, 2 ) = grid_fine_( local_subdomain_id, xi, yi, 2 );
            }
            else
            {
                coords_sh( id, 0 ) = coords_sh( id, 1 ) = coords_sh( id, 2 ) = 0.0;
            }
        } );

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nlev ), [&]( int lvl ) {
            const int rr = r0 + lvl;
            r_sh( lvl )  = ( rr <= hex_rad_ ) ? radii_( local_subdomain_id, rr ) : 0.0;
        } );

        const int total_src = nxy * nlev;
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, total_src ), [&]( int t ) {
            const int node = t / nlev;
            const int lvl  = t - node * nlev;
            const int dxn  = node % n;
            const int dyn  = node / n;
            const int xi   = x0 + dxn;
            const int yi   = y0 + dyn;
            const int rr   = r0 + lvl;
            if ( xi <= hex_lat_ && yi <= hex_lat_ && rr <= hex_rad_ )
            {
                src_sh( node, 0, lvl ) = src_( local_subdomain_id, xi, yi, rr, 0 );
                src_sh( node, 1, lvl ) = src_( local_subdomain_id, xi, yi, rr, 1 );
                src_sh( node, 2, lvl ) = src_( local_subdomain_id, xi, yi, rr, 2 );
            }
            else
            {
                src_sh( node, 0, lvl ) = src_sh( node, 1, lvl ) = src_sh( node, 2, lvl ) = 0.0;
            }
        } );

        team.team_barrier();

        if ( tr >= r_tile_ )
            return;
        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ )
            return;

        // --- Fused arithmetic kernel (same optimisations as EpsilonDivDivKerngen's fast paths) ---
        //
        // The divergence operator's 6x18 element matrix collapses to
        //     p_i = -qw * |det J| * shape_coarse_i * (∇·u)_phys
        // so we never materialise A. Per wedge we compute one Jacobian, one div_u at the quadrature
        // point (18 FMAs), then scatter `prefactor · shape_coarse_i` to 6 coarse nodes.
        //
        // Dirichlet: zero-out the boundary nodes' velocity contribution.
        // Freeslip:  project the boundary nodes' velocity onto the tangent plane before use
        //            (the outward unit normal on a spherical shell is just the unit-sphere
        //             lateral coord already cached in coords_sh).
        //
        // Quadrature: Felippa 1x1 single point. Weights + points fetched once.

        dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
        ScalarT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];
        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );
        const ScalarT qw = quad_weights[0];

        // Wedge node offset table: (ddx, ddy, ddr) per (wedge, local node 0..5). Same layout used
        // by atomically_add_local_wedge_scalar_coefficients for the coarse scatter.
        constexpr int WEDGE_NODE_OFF[2][6][3] = {
            { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
            { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

        // Same lat hex-corner layout as the surface-coord gather: v0 is the "first" corner of each
        // sub-wedge, v1 and v2 the other two. Only lat nodes matter for the Jacobian block.
        const int n00 = node_id( tx,     ty     );
        const int n01 = node_id( tx,     ty + 1 );
        const int n10 = node_id( tx + 1, ty     );
        const int n11 = node_id( tx + 1, ty + 1 );

        for ( int pass = 0; pass < r_passes_; ++pass )
        {
            const int lvl0       = pass * r_tile_ + tr;
            const int r_cell_abs = r0 + lvl0;
            if ( r_cell_abs >= hex_rad_ )
                break;

            const ScalarT r_1 = r_sh( lvl0     );
            const ScalarT r_2 = r_sh( lvl0 + 1 );

            const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_abs,     CMB     );
            const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_abs + 1, SURFACE );

            // BC classification, done once per cell.
            bool treat_dirichlet = false;
            bool treat_freeslip  = false;
            if ( at_cmb || at_surface )
            {
                const ShellBoundaryFlag     sbf = at_cmb ? CMB : SURFACE;
                const BoundaryConditionFlag bcf = get_boundary_condition_flag( bcs_, sbf );
                treat_dirichlet = ( bcf == DIRICHLET );
                if constexpr ( Freeslip )
                    treat_freeslip = ( bcf == FREESLIP );
            }

            // Boundary nodes live either at the cell's bottom (ddr == 0, CMB) or top (ddr == 1, SURFACE).
            const int boundary_ddr = at_cmb ? 0 : 1;

            const int fine_rad_wedge = r_cell_abs % 2;

            for ( int w = 0; w < num_wedges_per_hex_cell; ++w )
            {
                const int v0 = ( w == 0 ? n00 : n11 );
                const int v1 = ( w == 0 ? n10 : n01 );
                const int v2 = ( w == 0 ? n01 : n10 );

                // --- Jacobian (same formulas as EpsilonDivDivKerngen) ---
                constexpr double ONE_THIRD = 1.0 / 3.0;
                const double     half_dr   = 0.5 * ( r_2 - r_1 );
                const double     r_mid     = 0.5 * ( r_1 + r_2 );

                const double J_0_0 = r_mid  * ( -coords_sh( v0, 0 ) + coords_sh( v1, 0 ) );
                const double J_0_1 = r_mid  * ( -coords_sh( v0, 0 ) + coords_sh( v2, 0 ) );
                const double J_0_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 0 ) + coords_sh( v1, 0 ) + coords_sh( v2, 0 ) ) );
                const double J_1_0 = r_mid  * ( -coords_sh( v0, 1 ) + coords_sh( v1, 1 ) );
                const double J_1_1 = r_mid  * ( -coords_sh( v0, 1 ) + coords_sh( v2, 1 ) );
                const double J_1_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 1 ) + coords_sh( v1, 1 ) + coords_sh( v2, 1 ) ) );
                const double J_2_0 = r_mid  * ( -coords_sh( v0, 2 ) + coords_sh( v1, 2 ) );
                const double J_2_1 = r_mid  * ( -coords_sh( v0, 2 ) + coords_sh( v2, 2 ) );
                const double J_2_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 2 ) + coords_sh( v1, 2 ) + coords_sh( v2, 2 ) ) );

                const double J_det  = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1
                                    - J_0_1 * J_1_0 * J_2_2 + J_0_1 * J_1_2 * J_2_0
                                    + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;
                const double abs_det = Kokkos::abs( J_det );
                const double inv_det = 1.0 / J_det;

                // J^{-T} (row d dot dN_ref = physical gradient component d)
                const double i00 = inv_det * (  J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                const double i01 = inv_det * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                const double i02 = inv_det * (  J_1_0 * J_2_1 - J_1_1 * J_2_0 );
                const double i10 = inv_det * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                const double i11 = inv_det * (  J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                const double i12 = inv_det * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
                const double i20 = inv_det * (  J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                const double i21 = inv_det * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                const double i22 = inv_det * (  J_0_0 * J_1_1 - J_0_1 * J_1_0 );

                // dN_ref[j] on the reference wedge at the Felippa centroid. Matches EpsilonDivDivKerngen.
                constexpr double ONE_SIXTH = 1.0 / 6.0;
                static constexpr double dN_ref[6][3] = {
                    { -0.5, -0.5, -ONE_SIXTH },
                    {  0.5,  0.0, -ONE_SIXTH },
                    {  0.0,  0.5, -ONE_SIXTH },
                    { -0.5, -0.5,  ONE_SIXTH },
                    {  0.5,  0.0,  ONE_SIXTH },
                    {  0.0,  0.5,  ONE_SIXTH } };

                // --- Fused gather: accumulate (∇·u)_phys at the single quadrature point ---
                double div_u = 0.0;
#pragma unroll
                for ( int j = 0; j < num_nodes_per_wedge; ++j )
                {
                    const int ddx = WEDGE_NODE_OFF[w][j][0];
                    const int ddy = WEDGE_NODE_OFF[w][j][1];
                    const int ddr = WEDGE_NODE_OFF[w][j][2];

                    const bool is_boundary_node = ( at_cmb || at_surface ) && ( ddr == boundary_ddr );

                    // Dirichlet: boundary-node velocity contributes nothing to div_u.
                    if ( treat_dirichlet && is_boundary_node )
                        continue;

                    const int nid = node_id( tx + ddx, ty + ddy );
                    const int lvl = lvl0 + ddr;

                    double u0 = src_sh( nid, 0, lvl );
                    double u1 = src_sh( nid, 1, lvl );
                    double u2 = src_sh( nid, 2, lvl );

                    // Freeslip: project boundary-node velocity onto the tangent plane.
                    // On the spherical shell the outward normal is the unit-sphere coord itself.
                    if constexpr ( Freeslip )
                    {
                        if ( treat_freeslip && is_boundary_node )
                        {
                            const double nx  = coords_sh( nid, 0 );
                            const double ny  = coords_sh( nid, 1 );
                            const double nz  = coords_sh( nid, 2 );
                            const double un  = u0 * nx + u1 * ny + u2 * nz;
                            u0 -= un * nx;
                            u1 -= un * ny;
                            u2 -= un * nz;
                        }
                    }

                    // Physical gradient of reference shape function j.
                    const double gx = dN_ref[j][0];
                    const double gy = dN_ref[j][1];
                    const double gz = dN_ref[j][2];
                    const double g0 = i00 * gx + i01 * gy + i02 * gz;
                    const double g1 = i10 * gx + i11 * gy + i12 * gz;
                    const double g2 = i20 * gx + i21 * gy + i22 * gz;

                    div_u += g0 * u0 + g1 * u1 + g2 * u2;
                }

                // --- Scatter: p_i += -qw * |det J| * shape_coarse_i * div_u ---
                const int    fine_lat_wedge = fine_lateral_wedge_idx( x_cell, y_cell, w );
                const double prefactor      = -qw * abs_det * div_u;

                const int xc = x_cell / 2;
                const int yc = y_cell / 2;
                const int rc = r_cell_abs / 2;

#pragma unroll
                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                {
                    const double shape_i = shape_coarse( i, fine_rad_wedge, fine_lat_wedge, quad_points[0] );
                    const int    ddx     = WEDGE_NODE_OFF[w][i][0];
                    const int    ddy     = WEDGE_NODE_OFF[w][i][1];
                    const int    ddr     = WEDGE_NODE_OFF[w][i][2];
                    Kokkos::atomic_add( &dst_( local_subdomain_id, xc + ddx, yc + ddy, rc + ddr ),
                                         prefactor * shape_i );
                }
            }
        }
    }
};

static_assert( linalg::OperatorLike< DivergenceKerngen< double > > );

} // namespace terra::fe::wedge::operators::shell
