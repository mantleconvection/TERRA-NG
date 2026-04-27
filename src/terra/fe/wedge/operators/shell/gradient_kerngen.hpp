#pragma once

/// @file  gradient_kerngen.hpp
/// @brief Team-based matrix-free Gradient operator for the spherical shell,
///        with the same fused-arithmetic optimisations applied to
///        `DivergenceKerngen` / `EpsilonDivDivKerngen`.
///
/// Structure mirrors `DivergenceKerngen` — but Gradient is the transpose of
/// Divergence:
///    - src: scalar pressure on the coarse grid  (Grid4DDataScalar)
///    - dst: vec3 velocity on the fine grid      (Grid4DDataVec<..,3>)
///
/// Fused arithmetic derivation. The original element-matrix form is
///   A[wedge](d·6+i, j) = -qw · |det J| · (J⁻ᵀ ∇N_i)_d · shape_coarse_j
/// so
///   dst_{d,i} = Σⱼ A(d·6+i, j) · p_j
///             = -qw · |det J| · (J⁻ᵀ ∇N_i)_d · Σⱼ shape_coarse_j · p_j
///             = -qw · |det J| · (J⁻ᵀ ∇N_i)_d · p_interp,
/// where p_interp is the interpolated coarse pressure at the quadrature point.
/// We never materialise A.
///
/// Per wedge:
///  1. Compute J, |det J|, J⁻¹.
///  2. Compute p_interp = Σⱼ shape_coarse_j(q) · p_j  (6 muls).
///  3. prefactor = -qw · |det J| · p_interp.
///  4. For each fine node i ∈ 0..5:
///       g = J⁻ᵀ · dN_ref_i  (9 FMAs)
///       contribution_d = prefactor · g_d  (3 muls)
///       Dirichlet: skip boundary-node scatter.
///       Freeslip:  project contribution onto tangent plane at boundary nodes.
///       atomic_add into dst_(s, x+ddx, y+ddy, r+ddr, d) for d=0,1,2.
///
/// The freeslip projection uses the fact that on a spherical shell the outward
/// normal at a lateral node equals the unit-sphere coord already cached in
/// coords_sh — same trick as DivergenceKerngen, applied to the output side here.

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
class GradientKerngen
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Vec< ScalarT, 3 >;
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

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, 3 >                   recv_buffers_;
    terra::communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataVec< ScalarT, 3 > >   comm_plan_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataVec< ScalarType, 3 > dst_;

    int local_subdomains_;
    int hex_lat_;
    int hex_rad_;
    int lat_tile_;
    int r_tile_;
    int r_passes_;
    int r_tile_block_;
    int lat_tiles_;
    int r_tiles_;
    int team_size_;
    int blocks_;

    KernelPath kernel_path_ = KernelPath::FastDirichletNeumann;

    void update_kernel_path_flag_host_only()
    {
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
    GradientKerngen(
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
    , recv_buffers_( domain_fine )
    , comm_plan_( domain_fine )
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

        util::logroot << "[GradientKerngen] tile=(" << lat_tile_ << "," << lat_tile_ << "," << r_tile_
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

    void force_slow_path() { kernel_path_ = KernelPath::Slow; }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "gradient_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        util::Timer          timer_kernel( "gradient_kernel" );
        Kokkos::TeamPolicy<> policy( blocks_, team_size_ );
        if ( kernel_path_ != KernelPath::Slow )
        {
            policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );
        }

        if ( kernel_path_ == KernelPath::Slow )
        {
            Kokkos::parallel_for(
                "gradient_apply_kernel_slow", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->run_team_slow( team );
                } );
        }
        else if ( kernel_path_ == KernelPath::FastFreeslip )
        {
            Kokkos::parallel_for(
                "gradient_apply_kernel_fast_fs", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->template run_team_fast< /*Freeslip=*/true >( team );
                } );
        }
        else
        {
            Kokkos::TeamPolicy< Kokkos::LaunchBounds< 128, 5 > > dn_policy( blocks_, team_size_ );
            dn_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );
            Kokkos::parallel_for(
                "gradient_apply_kernel_fast_dn", dn_policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->template run_team_fast< /*Freeslip=*/false >( team );
                } );
        }

        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "gradient_comm" );
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size( const int /*ts*/ ) const
    {
        const int nlev   = r_tile_block_ + 1;
        const int n      = lat_tile_ + 1;
        const int nxy    = n * n;
        const int n_c    = ( lat_tile_ / 2 ) + 1;
        const int nxy_c  = n_c * n_c;
        const int nlev_c = ( r_tile_block_ / 2 ) + 1;
        // coords_sh(nxy,3) + r_sh(nlev) + p_sh(nxy_c, nlev_c)
        const size_t nscalars = size_t( nxy ) * 3 + size_t( nlev ) + size_t( nxy_c ) * nlev_c;
        return sizeof( ScalarType ) * nscalars;
    }

  private:
    KOKKOS_INLINE_FUNCTION
    void decode_team_indices(
        const Team& team,
        int& local_subdomain_id,
        int& x0, int& y0, int& r0,
        int& tx, int& ty, int& tr,
        int& x_cell, int& y_cell, int& r_cell ) const
    {
        int       tmp       = team.league_rank();
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

    // ----------------------------------------------------------
    // Slow (reference) path — team wrapper around the legacy per-cell logic.
    // Kept math-identical to Gradient (gradient.hpp).
    // ----------------------------------------------------------
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

            process_cell_legacy( local_subdomain_id, x_cell, y_cell, r_cell_pass );
        }
    }

    // Per-cell legacy Gradient kernel (same math as gradient.hpp), used by the slow path.
    KOKKOS_INLINE_FUNCTION
    void process_cell_legacy( int s, int x_cell, int y_cell, int r_cell ) const
    {
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_fine_, s, x_cell, y_cell );

        const ScalarT r_1 = radii_( s, r_cell );
        const ScalarT r_2 = radii_( s, r_cell + 1 );

        dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
        ScalarT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];
        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );

        constexpr int num_quad_points        = quadrature::quad_felippa_1x1_num_quad_points;
        const int     fine_radial_wedge_index = r_cell % 2;

        dense::Mat< ScalarT, 18, 6 > A[num_wedges_per_hex_cell] = {};

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
                    const auto grad_i = grad_shape( i, quad_points[q] );
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto shape_j =
                            shape_coarse( j, fine_radial_wedge_index, fine_lateral_wedge_index, quad_points[q] );
                        for ( int d = 0; d < 3; d++ )
                        {
                            A[wedge]( d * 6 + i, j ) +=
                                quad_weights[q] * ( -( ( J_inv_transposed * grad_i )( d ) * shape_j ) * det );
                        }
                    }
                }
            }
        }

        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, s, x_cell / 2, y_cell / 2, r_cell / 2, src_ );

        bool at_cmb     = util::has_flag( boundary_mask_fine_( s, x_cell, y_cell, r_cell ), CMB );
        bool at_surface = util::has_flag( boundary_mask_fine_( s, x_cell, y_cell, r_cell + 1 ), SURFACE );

        dense::Mat< ScalarT, 18, 6 > boundary_mask;
        boundary_mask.fill( 1.0 );

        dense::Mat< ScalarT, 18, 18 > R[num_wedges_per_hex_cell];
        bool freeslip_reorder = false;

        if ( at_cmb || at_surface )
        {
            ShellBoundaryFlag     sbf = at_cmb ? CMB : SURFACE;
            BoundaryConditionFlag bcf = get_boundary_condition_flag( bcs_, sbf );

            if ( bcf == DIRICHLET )
            {
                for ( int dimi = 0; dimi < 3; ++dimi )
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                            if ( ( at_cmb && i < 3 ) || ( at_surface && i >= 3 ) )
                                boundary_mask( dimi * num_nodes_per_wedge + i, j ) = 0.0;
            }
            else if ( bcf == FREESLIP )
            {
                freeslip_reorder = true;
                dense::Mat< ScalarT, 18, 6 > A_tmp[num_wedges_per_hex_cell] = { 0 };

                for ( int wedge = 0; wedge < 2; ++wedge )
                    for ( int node_idxi = 0; node_idxi < num_nodes_per_wedge; ++node_idxi )
                        for ( int dimi = 0; dimi < 3; ++dimi )
                            for ( int node_idxj = 0; node_idxj < num_nodes_per_wedge; ++node_idxj )
                                A_tmp[wedge]( node_idxi * 3 + dimi, node_idxj ) =
                                    A[wedge]( node_idxi + dimi * num_nodes_per_wedge, node_idxj );

                constexpr int layer_hex_offset_x[2][3] = { { 0, 1, 0 }, { 1, 0, 1 } };
                constexpr int layer_hex_offset_y[2][3] = { { 0, 0, 1 }, { 1, 1, 0 } };

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

                        auto      R_i         = trafo_mat_cartesian_to_normal_tangential( normal );
                        const int offset_in_R = at_cmb ? 0 : 9;
                        for ( int dimi = 0; dimi < 3; ++dimi )
                            for ( int dimj = 0; dimj < 3; ++dimj )
                                R[wedge]( offset_in_R + bn * 3 + dimi, offset_in_R + bn * 3 + dimj ) = R_i( dimi, dimj );
                    }

                    A[wedge] = R[wedge] * A_tmp[wedge];

                    const int node_start = at_surface ? 3 : 0;
                    const int node_end   = at_surface ? 6 : 3;
                    for ( int node_idx = node_start; node_idx < node_end; ++node_idx )
                    {
                        const int idx = node_idx * 3;
                        for ( int k = 0; k < 6; ++k )
                            boundary_mask( idx, k ) = 0.0;
                    }
                }
            }
            else if ( bcf == NEUMANN )
            {
                // No mask modification — matches legacy Gradient.
            }
        }

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            A[wedge].hadamard_product( boundary_mask );

        dense::Vec< ScalarT, 18 > dst_loc[num_wedges_per_hex_cell];
        dst_loc[0] = A[0] * src[0];
        dst_loc[1] = A[1] * src[1];

        if ( freeslip_reorder )
        {
            dense::Vec< ScalarT, 18 > dst_tmp[num_wedges_per_hex_cell];
            dst_tmp[0] = R[0].transposed() * dst_loc[0];
            dst_tmp[1] = R[1].transposed() * dst_loc[1];
            for ( int i = 0; i < 18; ++i )
            {
                dst_loc[0]( i ) = dst_tmp[0]( i );
                dst_loc[1]( i ) = dst_tmp[1]( i );
            }
            reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst_loc[0] );
            reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst_loc[1] );
        }

        for ( int d = 0; d < 3; d++ )
        {
            dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
            dst_d[0] = dst_loc[0].template slice< 6 >( d * 6 );
            dst_d[1] = dst_loc[1].template slice< 6 >( d * 6 );
            atomically_add_local_wedge_vector_coefficients(
                dst_, s, x_cell, y_cell, r_cell, d, dst_d );
        }
    }

    // ----------------------------------------------------------
    // Fast fused-arithmetic path. Templated on Freeslip so the projection code
    // dead-eliminates on the Dirichlet/Neumann path.
    // ----------------------------------------------------------
    template < bool Freeslip >
    KOKKOS_INLINE_FUNCTION
    void run_team_fast( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        const int nlev   = r_tile_block_ + 1;
        const int n      = lat_tile_ + 1;
        const int nxy    = n * n;
        const int n_c    = ( lat_tile_ / 2 ) + 1;
        const int nxy_c  = n_c * n_c;
        const int nlev_c = ( r_tile_block_ / 2 ) + 1;

        double* shmem =
            reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size( team.team_size() ) ) );

        using ScratchCoords =
            Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchR =
            Kokkos::View< double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchP =
            Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;

        ScratchCoords coords_sh( shmem, nxy, 3 );
        shmem += nxy * 3;
        ScratchR r_sh( shmem, nlev );
        shmem += nlev;
        ScratchP p_sh( shmem, nxy_c, nlev_c );  // coarse pressure

        auto node_id   = [&]( int nx, int ny ) -> int { return nx + n   * ny; };
        auto node_id_c = [&]( int nx, int ny ) -> int { return nx + n_c * ny; };

        // Coarse-grid starting point for the tile: since x0, y0 are multiples of
        // lat_tile_ (≥2 if fast path active), x0/2 / y0/2 are integer coarse indices.
        const int x0_c = x0 / 2;
        const int y0_c = y0 / 2;
        const int r0_c = r0 / 2;

        // Preload lat coords.
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

        // Preload radii.
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nlev ), [&]( int lvl ) {
            const int rr = r0 + lvl;
            r_sh( lvl )  = ( rr <= hex_rad_ ) ? radii_( local_subdomain_id, rr ) : 0.0;
        } );

        // Preload coarse pressure.
        const int total_coarse = nxy_c * nlev_c;
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, total_coarse ), [&]( int t ) {
            const int node_c = t / nlev_c;
            const int lvl_c  = t - node_c * nlev_c;
            const int dxn    = node_c % n_c;
            const int dyn    = node_c / n_c;
            const int xi_c   = x0_c + dxn;
            const int yi_c   = y0_c + dyn;
            const int rr_c   = r0_c + lvl_c;
            // Coarse index bounds: coarse hex_lat ≈ hex_lat/2 nodes. We just bounds-check using the
            // src_ extents (guaranteed sized for the coarse domain by the caller).
            if ( xi_c < static_cast< int >( src_.extent( 1 ) ) &&
                 yi_c < static_cast< int >( src_.extent( 2 ) ) &&
                 rr_c < static_cast< int >( src_.extent( 3 ) ) )
            {
                p_sh( node_c, lvl_c ) = src_( local_subdomain_id, xi_c, yi_c, rr_c );
            }
            else
            {
                p_sh( node_c, lvl_c ) = 0.0;
            }
        } );

        team.team_barrier();

        if ( tr >= r_tile_ )
            return;
        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ )
            return;

        dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
        ScalarT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];
        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );
        const ScalarT qw = quad_weights[0];

        constexpr int WEDGE_NODE_OFF[2][6][3] = {
            { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
            { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

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
            const int boundary_ddr = at_cmb ? 0 : 1;

            const int fine_rad_wedge = r_cell_abs % 2;

            // Coarse indices for this fine cell within the tile's coarse slab.
            const int cxc_in_tile = ( x_cell - x0 ) / 2;     // 0..(lat_tile/2-1)
            const int cyc_in_tile = ( y_cell - y0 ) / 2;
            const int crc_in_tile = lvl0 / 2;                // 0..(r_tile_block/2-1)

            for ( int w = 0; w < num_wedges_per_hex_cell; ++w )
            {
                const int v0 = ( w == 0 ? n00 : n11 );
                const int v1 = ( w == 0 ? n10 : n01 );
                const int v2 = ( w == 0 ? n01 : n10 );

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

                const double i00 = inv_det * (  J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                const double i01 = inv_det * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                const double i02 = inv_det * (  J_1_0 * J_2_1 - J_1_1 * J_2_0 );
                const double i10 = inv_det * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                const double i11 = inv_det * (  J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                const double i12 = inv_det * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
                const double i20 = inv_det * (  J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                const double i21 = inv_det * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                const double i22 = inv_det * (  J_0_0 * J_1_1 - J_0_1 * J_1_0 );

                constexpr double ONE_SIXTH = 1.0 / 6.0;
                static constexpr double dN_ref[6][3] = {
                    { -0.5, -0.5, -ONE_SIXTH },
                    {  0.5,  0.0, -ONE_SIXTH },
                    {  0.0,  0.5, -ONE_SIXTH },
                    { -0.5, -0.5,  ONE_SIXTH },
                    {  0.5,  0.0,  ONE_SIXTH },
                    {  0.0,  0.5,  ONE_SIXTH } };

                const int fine_lat_wedge = fine_lateral_wedge_idx( x_cell, y_cell, w );

                // --- Interpolate coarse pressure at the quadrature point. ---
                // The 6 coarse nodes of this fine cell live at the 8 corners of one coarse hex
                // (wedge layout via WEDGE_NODE_OFF on the coarse grid).
                double p_interp = 0.0;
#pragma unroll
                for ( int j = 0; j < num_nodes_per_wedge; ++j )
                {
                    const int cddx = WEDGE_NODE_OFF[w][j][0];
                    const int cddy = WEDGE_NODE_OFF[w][j][1];
                    const int cddr = WEDGE_NODE_OFF[w][j][2];
                    const int nidc = node_id_c( cxc_in_tile + cddx, cyc_in_tile + cddy );
                    const int lvlc = crc_in_tile + cddr;
                    const double pj = p_sh( nidc, lvlc );
                    const double sj = shape_coarse( j, fine_rad_wedge, fine_lat_wedge, quad_points[0] );
                    p_interp += sj * pj;
                }

                const double prefactor = -qw * abs_det * p_interp;

                // --- Scatter contributions to 6 fine nodes (3 dims each). ---
                const int xc_base = x_cell;
                const int yc_base = y_cell;
                const int rc_base = r_cell_abs;

#pragma unroll
                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                {
                    const int ddx = WEDGE_NODE_OFF[w][i][0];
                    const int ddy = WEDGE_NODE_OFF[w][i][1];
                    const int ddr = WEDGE_NODE_OFF[w][i][2];

                    const bool is_boundary_node = ( at_cmb || at_surface ) && ( ddr == boundary_ddr );

                    // Dirichlet: no scatter at boundary nodes.
                    if ( treat_dirichlet && is_boundary_node )
                        continue;

                    // Physical gradient of fine basis function i.
                    const double gx = dN_ref[i][0];
                    const double gy = dN_ref[i][1];
                    const double gz = dN_ref[i][2];
                    const double g0 = i00 * gx + i01 * gy + i02 * gz;
                    const double g1 = i10 * gx + i11 * gy + i12 * gz;
                    const double g2 = i20 * gx + i21 * gy + i22 * gz;

                    double c0 = prefactor * g0;
                    double c1 = prefactor * g1;
                    double c2 = prefactor * g2;

                    // Freeslip: project the 3-component contribution onto the tangent plane at boundary nodes.
                    if constexpr ( Freeslip )
                    {
                        if ( treat_freeslip && is_boundary_node )
                        {
                            const int    nid = node_id( tx + ddx, ty + ddy );
                            const double nx  = coords_sh( nid, 0 );
                            const double ny  = coords_sh( nid, 1 );
                            const double nz  = coords_sh( nid, 2 );
                            const double cn  = c0 * nx + c1 * ny + c2 * nz;
                            c0 -= cn * nx;
                            c1 -= cn * ny;
                            c2 -= cn * nz;
                        }
                    }

                    Kokkos::atomic_add( &dst_( local_subdomain_id, xc_base + ddx, yc_base + ddy, rc_base + ddr, 0 ), c0 );
                    Kokkos::atomic_add( &dst_( local_subdomain_id, xc_base + ddx, yc_base + ddy, rc_base + ddr, 1 ), c1 );
                    Kokkos::atomic_add( &dst_( local_subdomain_id, xc_base + ddx, yc_base + ddy, rc_base + ddr, 2 ), c2 );
                }
            }
        }
    }
};

static_assert( linalg::OperatorLike< GradientKerngen< double > > );

} // namespace terra::fe::wedge::operators::shell
