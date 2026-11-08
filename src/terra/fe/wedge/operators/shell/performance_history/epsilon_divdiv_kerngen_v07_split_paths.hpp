#pragma once

#include "../../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "impl/Kokkos_Profiling.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/gca/local_matrix_storage.hpp"
#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_w1.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell::epsdivdiv_history {

using grid::shell::get_boundary_condition_flag;
using grid::shell::BoundaryConditionFlag::DIRICHLET;
using grid::shell::BoundaryConditionFlag::FREESLIP;
using grid::shell::BoundaryConditionFlag::NEUMANN;
using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
using terra::grid::shell::BoundaryConditionFlag;
using terra::grid::shell::BoundaryConditions;
using terra::grid::shell::ShellBoundaryFlag;
using terra::linalg::trafo::trafo_mat_cartesian_to_normal_tangential;

/**
 * @brief Matrix-free / matrix-based epsilon-div-div operator on wedge elements in a spherical shell.
 *
 * This functor supports two execution modes:
 *
 * 1) FAST PATH (shared-memory fused local matvec)
 *    - Used when both shell boundaries are only DIRICHLET/NEUMANN
 *    - Used only when no stored local matrices are enabled
 *    - This is the high-throughput path that loads a tile slab into team scratch memory
 *
 * 2) SLOW PATH (local matrix path)
 *    - Used when FREESLIP is present on either boundary, because the local basis needs
 *      a per-boundary-node rotation (normal/tangential transform)
 *    - Used whenever local matrices are stored (full/selective)
 *
 * The path decision is computed on the host and cached in `use_slow_path_`, so the kernel
 * launch itself is specialized (we do not branch per thread/team inside the hot path).
 */
template < typename ScalarT, int VecDim = 3 >
class EpsilonDivDivKerngenV07SplitPaths
{
  public:
    using SrcVectorType                 = linalg::VectorW1Vec< ScalarT, VecDim >;
    using DstVectorType                 = linalg::VectorW1Vec< ScalarT, VecDim >;
    using ScalarType                    = ScalarT;
    static constexpr int LocalMatrixDim = 18;
    using Grid4DDataLocalMatrices =
        terra::grid::Grid4DDataMatrices< ScalarType, LocalMatrixDim, LocalMatrixDim, 2 >;
    using LocalMatrixStorage = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >;
    using Team              = Kokkos::TeamPolicy<>::member_type;

  private:
    // Optional storage for element-local matrices (used by GCA/coarsening or explicit local mat path)
    LocalMatrixStorage local_matrix_storage_;

    // Domain and geometry / coefficient data
    grid::shell::DistributedDomain domain_;
    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;   // lateral shell geometry (unit sphere coords)
    grid::Grid2DDataScalar< ScalarT >                        radii_;  // radial coordinates per local subdomain
    grid::Grid4DDataScalar< ScalarType >                     k_;      // scalar coefficient field
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;   // boundary flags per cell/node
    BoundaryConditions                                       bcs_;     // CMB/SURFACE BC types

    bool diagonal_; // if true, apply only diagonal of local operator (or diagonalized fast approx)

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;
    linalg::OperatorStoredMatrixMode  operator_stored_matrix_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buffers_;
    terra::communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataVec< ScalarType, VecDim > > comm_plan_;

    // Views captured by device kernels during apply_impl
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;
    grid::Grid4DDataVec< ScalarType, VecDim > src_;

    // Quadrature data (Felippa 1x1 on wedge)
    const int num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;
    dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
    ScalarT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];

    // Domain extents (cells) for one local subdomain
    int local_subdomains_;
    int hex_lat_;
    int hex_rad_;
    int lat_refinement_level_;

    // 3D tile decomposition for TeamPolicy
    // A team handles a slab: lat_tile_ x lat_tile_ x r_tile_ cells
    int lat_tile_;
    int r_tile_;
    int lat_tiles_;
    int r_tiles_;
    int team_size_;
    int blocks_;

    ScalarT r_max_;
    ScalarT r_min_;

    /**
     * @brief Cached host-side dispatch flag.
     *
     * true  -> launch slow kernel
     * false -> launch fast kernel
     *
     * Recomputed whenever boundary conditions or stored-matrix mode changes.
     */
    bool use_slow_path_ = false;

  private:
    /**
     * @brief Recompute whether the operator must use the slow path.
     *
     * Slow path is required when:
     * - local matrices are stored (full/selective), OR
     * - either shell boundary uses FREESLIP (needs local basis rotation)
     *
     * This function is intended to be called on the host only (constructor/setters).
     */
    void update_kernel_path_flag_host_only()
    {
        const BoundaryConditionFlag cmb_bc     = get_boundary_condition_flag( bcs_, CMB );
        const BoundaryConditionFlag surface_bc = get_boundary_condition_flag( bcs_, SURFACE );

        const bool has_freeslip_bc     = ( cmb_bc == FREESLIP ) || ( surface_bc == FREESLIP );
        const bool has_stored_matrices = ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off );

        use_slow_path_ = has_freeslip_bc || has_stored_matrices;
    }

  public:
    EpsilonDivDivKerngenV07SplitPaths(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const grid::Grid4DDataScalar< ScalarT >&                        k,
        BoundaryConditions                                              bcs,
        bool                                                            diagonal,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode operator_stored_matrix_mode = linalg::OperatorStoredMatrixMode::Off )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , k_( k )
    , mask_( mask )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , operator_stored_matrix_mode_( operator_stored_matrix_mode )
    , recv_buffers_( domain )
    , comm_plan_( domain )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];

        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );

        const grid::shell::DomainInfo& domain_info = domain_.domain_info();
        local_subdomains_                          = domain_.subdomains().size();
        hex_lat_                                   = domain_info.subdomain_num_nodes_per_side_laterally() - 1;
        hex_rad_                                   = domain_info.subdomain_num_nodes_radially() - 1;
        lat_refinement_level_                      = domain_info.diamond_lateral_refinement_level();

        // Tile dimensions: tune for backend occupancy / scratch usage
        lat_tile_ = 4;
        r_tile_   = 8;

        lat_tiles_ = ( hex_lat_ + lat_tile_ - 1 ) / lat_tile_;
        r_tiles_   = ( hex_rad_ + r_tile_ - 1 ) / r_tile_;

        team_size_ = lat_tile_ * lat_tile_ * r_tile_;
        blocks_    = local_subdomains_ * lat_tiles_ * lat_tiles_ * r_tiles_;

        r_min_ = domain_info.radii()[0];
        r_max_ = domain_info.radii()[domain_info.radii().size() - 1];

        update_kernel_path_flag_host_only();

        util::logroot << "[EpsilonDivDiv] tile size (x,y,r)=(" << lat_tile_ << "," << lat_tile_ << "," << r_tile_ << ")"
                      << std::endl;
        util::logroot << "[EpsilonDivDiv] number of tiles (x,y,r)=(" << lat_tiles_ << "," << lat_tiles_ << ","
                      << r_tiles_ << "), team_size=" << team_size_ << ", blocks=" << blocks_ << std::endl;
        util::logroot << "[EpsilonDivDiv] kernel path = " << ( use_slow_path_ ? "slow" : "fast" ) << std::endl;
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void set_diagonal( bool v ) { diagonal_ = v; }

    /// Optional runtime BC update; also refreshes fast/slow dispatch decision.
    void set_boundary_conditions( BoundaryConditions bcs )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];
        update_kernel_path_flag_host_only();
    }

    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }
    const grid::shell::DistributedDomain&       get_domain() const { return domain_; }
    grid::Grid2DDataScalar< ScalarT >           get_radii() const { return radii_; }
    grid::Grid3DDataVec< ScalarT, 3 >           get_grid() { return grid_; }

    /// Convenience wrapper for shell boundary mask checks.
    KOKKOS_INLINE_FUNCTION
    bool has_flag(
        const int                      local_subdomain_id,
        const int                      x_cell,
        const int                      y_cell,
        const int                      r_cell,
        grid::shell::ShellBoundaryFlag flag ) const
    {
        return util::has_flag( mask_( local_subdomain_id, x_cell, y_cell, r_cell ), flag );
    }

    /**
     * @brief Configure local matrix storage mode (Off / Selective / Full).
     *
     * This may allocate storage immediately and changes the kernel dispatch:
     * any non-Off mode forces the slow path.
     */
    void set_stored_matrix_mode(
        linalg::OperatorStoredMatrixMode     operator_stored_matrix_mode,
        int                                  level_range,
        grid::Grid4DDataScalar< ScalarType > GCAElements )
    {
        operator_stored_matrix_mode_ = operator_stored_matrix_mode;

        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            local_matrix_storage_ = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >(
                domain_, operator_stored_matrix_mode_, level_range, GCAElements );
        }

        update_kernel_path_flag_host_only();
    }

    linalg::OperatorStoredMatrixMode get_stored_matrix_mode() { return operator_stored_matrix_mode_; }

    /// Store a local element matrix (used in GCA/coarsening workflows).
    KOKKOS_INLINE_FUNCTION
    void set_local_matrix(
        const int                                                    local_subdomain_id,
        const int                                                    x_cell,
        const int                                                    y_cell,
        const int                                                    r_cell,
        const int                                                    wedge,
        const dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim >& mat ) const
    {
        KOKKOS_ASSERT( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off );
        local_matrix_storage_.set_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge, mat );
    }

    /**
     * @brief Get a local matrix, either from storage or assembled on-the-fly.
     */
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > get_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            if ( !local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) )
            {
                Kokkos::abort( "No matrix found at that spatial index." );
            }
            return local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
        }
        return assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
    }

    /**
     * @brief Apply operator to src and accumulate/replace into dst.
     *
     * Key design point:
     * - The slow/fast path decision is made here (host side), before kernel launch.
     * - This avoids branching on boundary/storage mode in the hot kernel body.
     */
    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "epsilon_divdiv_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        // Cache input/output grid views into members so the device functor sees them.
        dst_ = dst.grid_data();
        src_ = src.grid_data();

        util::Timer          timer_kernel( "epsilon_divdiv_kernel" );
        Kokkos::TeamPolicy<> policy( blocks_, team_size_ );

        // Fast path uses team scratch; slow path does not need it.
        if ( !use_slow_path_ )
        {
            policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );
        }

        // Host-side dispatch to specialized kernel
        if ( use_slow_path_ )
        {
            Kokkos::parallel_for(
                "matvec_slow",
                policy,
                KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->operator_slow_kernel( team );
                } );
        }
        else
        {
            Kokkos::parallel_for(
                "matvec_fast",
                policy,
                KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->operator_fast_kernel( team );
                } );
        }

        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "epsilon_divdiv_comm" );
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

    /**
     * @brief Convert one gradient column (for vector component dim) into the symmetric-gradient entries.
     *
     * The operator is assembled/applied using sym(grad u). For basis vector component `dim`,
     * only one gradient column is populated, so this helper computes:
     * - diagonal symmetric entries E00,E11,E22
     * - off-diagonal symmetric entries sym01,sym02,sym12
     * - divergence contribution gdd (= corresponding diagonal gradient entry)
     */
    KOKKOS_INLINE_FUNCTION
    void column_grad_to_sym(
        const int    dim,
        const double g0,
        const double g1,
        const double g2,
        double&      E00,
        double&      E11,
        double&      E22,
        double&      sym01,
        double&      sym02,
        double&      sym12,
        double&      gdd ) const
    {
        E00 = E11 = E22 = sym01 = sym02 = sym12 = gdd = 0.0;

        switch ( dim )
        {
        case 0:
            E00   = g0;
            gdd   = g0;
            sym01 = 0.5 * g1;
            sym02 = 0.5 * g2;
            break;
        case 1:
            E11   = g1;
            gdd   = g1;
            sym01 = 0.5 * g0;
            sym12 = 0.5 * g2;
            break;
        default:
            E22   = g2;
            gdd   = g2;
            sym02 = 0.5 * g0;
            sym12 = 0.5 * g1;
            break;
        }
    }

    /**
     * @brief Team scratch requirement for the fast path.
     *
     * Per team we store a slab covering:
     * - (lat_tile_+1)x(lat_tile_+1) surface geometry nodes
     * - (r_tile_+1) radial levels
     * - src dofs and coefficient values for the slab
     */
    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size( const int /*ts*/ ) const
    {
        const int nlev = r_tile_ + 1;
        const int n    = lat_tile_ + 1;
        const int nxy  = n * n;

        const size_t nscalars =
            size_t( nxy ) * 3 +               // coords_sh
            size_t( nxy ) * 3 * nlev +        // src_sh
            size_t( nxy ) * nlev +            // k_sh
            size_t( nlev ) +                  // r_sh
            1;                                

        return sizeof( ScalarType ) * nscalars;
    }

  private:
    /**
     * @brief Decode TeamPolicy league/team rank into subdomain/tile/cell indices.
     *
     * Mapping:
     *   league_rank -> (local_subdomain_id, lat_x_tile, lat_y_tile, r_tile)
     *   team_rank   -> (tx, ty, tr) within the tile
     */
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
        int tmp = team.league_rank();

        const int r_tile_id = tmp % r_tiles_;
        tmp /= r_tiles_;

        const int lat_y_id = tmp % lat_tiles_;
        tmp /= lat_tiles_;

        const int lat_x_id = tmp % lat_tiles_;
        tmp /= lat_tiles_;

        local_subdomain_id = tmp;

        x0 = lat_x_id * lat_tile_;
        y0 = lat_y_id * lat_tile_;
        r0 = r_tile_id * r_tile_;

        const int tid = team.team_rank();
        tx            = tid % lat_tile_;
        ty            = ( tid / lat_tile_ ) % lat_tile_;
        tr            = tid / ( lat_tile_ * lat_tile_ );

        x_cell = x0 + tx;
        y_cell = y0 + ty;
        r_cell = r0 + tr;
    }

    /**
     * @brief Device wrapper for slow path launch.
     *
     * Only does index decode + boundary flag fetch + dispatch into operator_slow_path().
     */
    KOKKOS_INLINE_FUNCTION
    void operator_slow_kernel( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
        const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

        operator_slow_path(
            team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell, at_cmb, at_surface );
    }

    /**
     * @brief Device wrapper for fast path launch.
     *
     * Only does index decode + boundary flag fetch + dispatch into operator_fast_path().
     */
    KOKKOS_INLINE_FUNCTION
    void operator_fast_kernel( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
        const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

        operator_fast_path(
            team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell, at_cmb, at_surface );
    }

    /**
     * @brief Slow path: local matrix-based application.
     *
     * Used for:
     * - FREESLIP (normal/tangential transform required)
     * - stored matrix mode (Full/Selective)
     *
     * Steps per element:
     * 1) fetch/assemble wedge-local matrices A[2]
     * 2) gather local src dofs
     * 3) apply boundary treatment:
     *    - DIRICHLET: zero corresponding rows/cols via boundary mask
     *    - FREESLIP: rotate boundary dofs into n/t coordinates, apply mask, rotate back
     *    - NEUMANN: no modification
     * 4) local matvec and atomic scatter to global dst
     */
    KOKKOS_INLINE_FUNCTION
    void operator_slow_path(
        const Team& team,
        const int   local_subdomain_id,
        const int   x0,
        const int   y0,
        const int   r0,
        const int   tx,
        const int   ty,
        const int   tr,
        const int   x_cell,
        const int   y_cell,
        const int   r_cell,
        const bool  at_cmb,
        const bool  at_surface ) const
    {
        // Unused in slow path body (kept in signature for symmetry with fast path)
        (void)team;
        (void)x0;
        (void)y0;
        (void)r0;
        (void)tx;
        (void)ty;
        (void)tr;

        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ || r_cell >= hex_rad_ )
            return;

        // ---- local matrix acquisition (stored or on-the-fly) ----
        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A[num_wedges_per_hex_cell] = { 0 };

        if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Full )
        {
            A[0] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
            A[1] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
        }
        else if ( operator_stored_matrix_mode_ == linalg::OperatorStoredMatrixMode::Selective )
        {
            if ( local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 ) &&
                 local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 ) )
            {
                A[0] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                A[1] = local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
            }
            else
            {
                A[0] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                A[1] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
            }
        }
        else
        {
            A[0] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
            A[1] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
        }

        // ---- gather local source dofs (dimension-wise layout) ----
        dense::Vec< ScalarT, 18 > src[num_wedges_per_hex_cell];
        for ( int dimj = 0; dimj < 3; dimj++ )
        {
            dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
            extract_local_wedge_vector_coefficients( src_d, local_subdomain_id, x_cell, y_cell, r_cell, dimj, src_ );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    src[wedge]( dimj * num_nodes_per_wedge + i ) = src_d[wedge]( i );
                }
            }
        }

        // Boundary masking is applied multiplicatively to local matrices.
        // Starts as all ones; DIRICHLET/FREESLIP zero out constrained couplings.
        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > boundary_mask;
        boundary_mask.fill( 1.0 );

        bool                                                  freeslip_reorder = false;
        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > R[num_wedges_per_hex_cell];

        if ( at_cmb || at_surface )
        {
            ShellBoundaryFlag     sbf = at_cmb ? CMB : SURFACE;
            BoundaryConditionFlag bcf = get_boundary_condition_flag( bcs_, sbf );

            if ( bcf == DIRICHLET )
            {
                // Zero couplings involving constrained boundary nodes.
                for ( int dimi = 0; dimi < 3; ++dimi )
                {
                    for ( int dimj = 0; dimj < 3; ++dimj )
                    {
                        for ( int i = 0; i < num_nodes_per_wedge; i++ )
                        {
                            for ( int j = 0; j < num_nodes_per_wedge; j++ )
                            {
                                if ( ( at_cmb && ( ( dimi == dimj && i != j && ( i < 3 || j < 3 ) ) ||
                                                   ( dimi != dimj && ( i < 3 || j < 3 ) ) ) ) ||
                                     ( at_surface && ( ( dimi == dimj && i != j && ( i >= 3 || j >= 3 ) ) ||
                                                       ( dimi != dimj && ( i >= 3 || j >= 3 ) ) ) ) )
                                {
                                    boundary_mask(
                                        i + dimi * num_nodes_per_wedge,
                                        j + dimj * num_nodes_per_wedge ) = 0.0;
                                }
                            }
                        }
                    }
                }
            }
            else if ( bcf == FREESLIP )
            {
                // FREESLIP treatment:
                // rotate boundary dofs (per node) to [normal, tangential1, tangential2],
                // constrain the normal component, then rotate back.
                freeslip_reorder = true;
                dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A_tmp[num_wedges_per_hex_cell] = { 0 };

                // Reorder A into node-wise layout because FREESLIP rotation is node-local 3x3 blocks.
                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    for ( int dimi = 0; dimi < 3; ++dimi )
                    {
                        for ( int node_idxi = 0; node_idxi < num_nodes_per_wedge; node_idxi++ )
                        {
                            for ( int dimj = 0; dimj < 3; ++dimj )
                            {
                                for ( int node_idxj = 0; node_idxj < num_nodes_per_wedge; node_idxj++ )
                                {
                                    A_tmp[wedge]( node_idxi * 3 + dimi, node_idxj * 3 + dimj ) = A[wedge](
                                        node_idxi + dimi * num_nodes_per_wedge,
                                        node_idxj + dimj * num_nodes_per_wedge );
                                }
                            }
                        }
                    }
                    reorder_local_dofs( DoFOrdering::DIMENSIONWISE, DoFOrdering::NODEWISE, src[wedge] );
                }

                // Local node offsets (in the hex cell) for the 3 nodes on the boundary face of each wedge
                constexpr int layer_hex_offset_x[2][3] = { { 0, 1, 0 }, { 1, 0, 1 } };
                constexpr int layer_hex_offset_y[2][3] = { { 0, 0, 1 }, { 1, 1, 0 } };

                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    // Start as identity
                    for ( int i = 0; i < LocalMatrixDim; ++i ) { R[wedge]( i, i ) = 1.0; }

                    // Build 3x3 rotation blocks for the boundary nodes
                    for ( int boundary_node_idx = 0; boundary_node_idx < 3; boundary_node_idx++ )
                    {
                        dense::Vec< double, 3 > normal = grid::shell::coords(
                            local_subdomain_id,
                            x_cell + layer_hex_offset_x[wedge][boundary_node_idx],
                            y_cell + layer_hex_offset_y[wedge][boundary_node_idx],
                            r_cell + ( at_cmb ? 0 : 1 ),
                            grid_,
                            radii_ );

                        auto R_i = trafo_mat_cartesian_to_normal_tangential( normal );

                        const int offset_in_R = at_cmb ? 0 : 9; // first 3 nodes (CMB) or top 3 nodes (SURFACE)
                        for ( int dimi = 0; dimi < 3; ++dimi )
                        {
                            for ( int dimj = 0; dimj < 3; ++dimj )
                            {
                                R[wedge](
                                    offset_in_R + boundary_node_idx * 3 + dimi,
                                    offset_in_R + boundary_node_idx * 3 + dimj ) = R_i( dimi, dimj );
                            }
                        }
                    }

                    // Rotate matrix and vector into n/t frame
                    A[wedge] = R[wedge] * A_tmp[wedge] * R[wedge].transposed();

                    auto src_tmp = R[wedge] * src[wedge];
                    for ( int i = 0; i < 18; ++i ) { src[wedge]( i ) = src_tmp( i ); }

                    // Constrain only the normal component at boundary nodes (node-wise layout => index = node*3 + 0)
                    const int node_start = at_surface ? 3 : 0;
                    const int node_end   = at_surface ? 6 : 3;

                    for ( int node_idx = node_start; node_idx < node_end; node_idx++ )
                    {
                        const int idx = node_idx * 3; // normal component
                        for ( int k = 0; k < 18; ++k )
                        {
                            if ( k != idx )
                            {
                                boundary_mask( idx, k ) = 0.0;
                                boundary_mask( k, idx ) = 0.0;
                            }
                        }
                    }
                }
            }
            else if ( bcf == NEUMANN )
            {
                // Natural BC => no extra masking
            }
        }

        // Apply boundary masking to local matrices
        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            A[wedge].hadamard_product( boundary_mask );
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        // Local matvec
        dense::Vec< ScalarT, LocalMatrixDim > dst[num_wedges_per_hex_cell];
        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        // Rotate back from n/t frame and reorder back to dimension-wise layout
        if ( freeslip_reorder )
        {
            dense::Vec< ScalarT, LocalMatrixDim > dst_tmp[num_wedges_per_hex_cell];
            dst_tmp[0] = R[0].transposed() * dst[0];
            dst_tmp[1] = R[1].transposed() * dst[1];

            for ( int i = 0; i < 18; ++i )
            {
                dst[0]( i ) = dst_tmp[0]( i );
                dst[1]( i ) = dst_tmp[1]( i );
            }

            reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst[0] );
            reorder_local_dofs( DoFOrdering::NODEWISE, DoFOrdering::DIMENSIONWISE, dst[1] );
        }

        // Scatter back to global vector (atomic because neighboring cells share nodes)
        for ( int dimi = 0; dimi < 3; dimi++ )
        {
            dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
            dst_d[0] = dst[0].template slice< 6 >( dimi * num_nodes_per_wedge );
            dst_d[1] = dst[1].template slice< 6 >( dimi * num_nodes_per_wedge );

            atomically_add_local_wedge_vector_coefficients(
                dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst_d );
        }
    }

    /**
     * @brief Fast path: fused shared-memory matrix-free local matvec.
     *
     * Assumptions:
     * - no stored matrices
     * - no FREESLIP boundaries
     *
     * Team-level workflow:
     * 1) cooperatively load a (lat_tile_+1)x(lat_tile_+1)x(r_tile_+1) slab into scratch
     * 2) each thread computes one hex cell in the tile
     * 3) each cell is split into 2 wedges and integrated in fused form
     * 4) atomic scatter to dst_
     *
     * Boundary handling in this path supports:
     * - DIRICHLET (via local node-range shifts / diagonal treatment)
     * - NEUMANN (natural, no modification)
     */
    KOKKOS_INLINE_FUNCTION
    void operator_fast_path(
        const Team& team,
        const int   local_subdomain_id,
        const int   x0,
        const int   y0,
        const int   r0,
        const int   tx,
        const int   ty,
        const int   tr,
        const int   x_cell,
        const int   y_cell,
        const int   r_cell,
        const bool  at_cmb,
        const bool  at_surface ) const
    {
        // ---- scratch slab dimensions ----
        const int nlev = r_tile_ + 1;
        const int nxy  = ( lat_tile_ + 1 ) * ( lat_tile_ + 1 );

        // Team scratch memory layout:
        // [coords_sh | src_sh | k_sh | r_sh]
        double* shmem = reinterpret_cast< double* >(
            team.team_shmem().get_shmem( team_shmem_size( team.team_size() ) ) );

        using ScratchCoords =
            Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchSrc =
            Kokkos::View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchK =
            Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;

        ScratchCoords coords_sh( shmem, nxy, 3 );
        shmem += nxy * 3;

        ScratchSrc src_sh( shmem, nxy, 3, nlev );
        shmem += nxy * 3 * nlev;

        ScratchK k_sh( shmem, nxy, nlev );
        shmem += nxy * nlev;

        auto r_sh =
            Kokkos::View< double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >(
                shmem, nlev );

        auto node_id = [&]( int nx, int ny ) -> int { return nx + ( lat_tile_ + 1 ) * ny; };

        // ---- cooperative tile loads ----
        // surface geometry coords
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy ), [&]( int n ) {
            const int dxn = n % ( lat_tile_ + 1 );
            const int dyn = n / ( lat_tile_ + 1 );
            const int xi  = x0 + dxn;
            const int yi  = y0 + dyn;

            if ( xi <= hex_lat_ && yi <= hex_lat_ )
            {
                coords_sh( n, 0 ) = grid_( local_subdomain_id, xi, yi, 0 );
                coords_sh( n, 1 ) = grid_( local_subdomain_id, xi, yi, 1 );
                coords_sh( n, 2 ) = grid_( local_subdomain_id, xi, yi, 2 );
            }
            else
            {
                coords_sh( n, 0 ) = coords_sh( n, 1 ) = coords_sh( n, 2 ) = 0.0;
            }
        } );

        // radial coordinates
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nlev ), [&]( int lvl ) {
            const int rr = r0 + lvl;
            r_sh( lvl )  = ( rr <= hex_rad_ ) ? radii_( local_subdomain_id, rr ) : 0.0;
        } );

        // coefficient and source values
        const int total_pairs = nxy * nlev;
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, total_pairs ), [&]( int t ) {
            const int lvl  = t / nxy;
            const int node = t - lvl * nxy;

            const int dxn = node % ( lat_tile_ + 1 );
            const int dyn = node / ( lat_tile_ + 1 );

            const int xi = x0 + dxn;
            const int yi = y0 + dyn;
            const int rr = r0 + lvl;

            if ( xi <= hex_lat_ && yi <= hex_lat_ && rr <= hex_rad_ )
            {
                k_sh( node, lvl )      = k_( local_subdomain_id, xi, yi, rr );
                src_sh( node, 0, lvl ) = src_( local_subdomain_id, xi, yi, rr, 0 );
                src_sh( node, 1, lvl ) = src_( local_subdomain_id, xi, yi, rr, 1 );
                src_sh( node, 2, lvl ) = src_( local_subdomain_id, xi, yi, rr, 2 );
            }
            else
            {
                k_sh( node, lvl )      = 0.0;
                src_sh( node, 0, lvl ) = src_sh( node, 1, lvl ) = src_sh( node, 2, lvl ) = 0.0;
            }
        } );

        team.team_barrier();

        // Each logical thread computes one hex cell in the tile
        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ || r_cell >= hex_rad_ )
            return;

        const int    lvl0 = tr;
        const double r_0  = r_sh( lvl0 );
        const double r_1  = r_sh( lvl0 + 1 );

        // In the fast path we only treat DIRICHLET specially.
        // FREESLIP is excluded by host-side dispatch.
        const bool at_boundary = at_cmb || at_surface;
        bool       treat_boundary_dirichlet = false;
        if ( at_boundary )
        {
            const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
            treat_boundary_dirichlet    = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
        }

        // For full (non-diagonal) application, skip constrained face nodes.
        // For diagonal mode, the diagonal term can still be accumulated with boundary handling below.
        const int cmb_shift = ( ( at_boundary && treat_boundary_dirichlet && ( !diagonal_ ) && at_cmb ) ? 3 : 0 );
        const int surface_shift =
            ( ( at_boundary && treat_boundary_dirichlet && ( !diagonal_ ) && at_surface ) ? 3 : 0 );

        // Wedge connectivity / local constants
        static constexpr int WEDGE_NODE_OFF[2][6][3] = {
            { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
            { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

        // Map wedge-local nodes into the 8 unique hex nodes used for scatter accumulation
        static constexpr int WEDGE_TO_UNIQUE[2][6] = {
            { 0, 1, 2, 3, 4, 5 }, // wedge 0
            { 6, 2, 1, 7, 5, 4 }  // wedge 1
        };

        constexpr double ONE_THIRD      = 1.0 / 3.0;
        constexpr double ONE_SIXTH      = 1.0 / 6.0;
        constexpr double NEG_TWO_THIRDS = -0.66666666666666663;

        // Reference gradients of wedge basis functions at the single Felippa 1x1 point
        static constexpr double dN_ref[6][3] = {
            { -0.5, -0.5, -ONE_SIXTH },
            { 0.5, 0.0, -ONE_SIXTH },
            { 0.0, 0.5, -ONE_SIXTH },
            { -0.5, -0.5, ONE_SIXTH },
            { 0.5, 0.0, ONE_SIXTH },
            { 0.0, 0.5, ONE_SIXTH } };

        // Four lateral nodes of the hex footprint inside this tile
        const int n00 = node_id( tx, ty );
        const int n01 = node_id( tx, ty + 1 );
        const int n10 = node_id( tx + 1, ty );
        const int n11 = node_id( tx + 1, ty + 1 );

        // Surface coordinates for the 2 wedges (3 points each)
        double ws[2][3][3];

        // wedge 0: (q00,q10,q01)
        ws[0][0][0] = coords_sh( n00, 0 );
        ws[0][0][1] = coords_sh( n00, 1 );
        ws[0][0][2] = coords_sh( n00, 2 );
        ws[0][1][0] = coords_sh( n10, 0 );
        ws[0][1][1] = coords_sh( n10, 1 );
        ws[0][1][2] = coords_sh( n10, 2 );
        ws[0][2][0] = coords_sh( n01, 0 );
        ws[0][2][1] = coords_sh( n01, 1 );
        ws[0][2][2] = coords_sh( n01, 2 );

        // wedge 1: (q11,q01,q10)
        ws[1][0][0] = coords_sh( n11, 0 );
        ws[1][0][1] = coords_sh( n11, 1 );
        ws[1][0][2] = coords_sh( n11, 2 );
        ws[1][1][0] = coords_sh( n01, 0 );
        ws[1][1][1] = coords_sh( n01, 1 );
        ws[1][1][2] = coords_sh( n01, 2 );
        ws[1][2][0] = coords_sh( n10, 0 );
        ws[1][2][1] = coords_sh( n10, 1 );
        ws[1][2][2] = coords_sh( n10, 2 );

        // Per-thread accumulation into 8 hex nodes x 3 vector components.
        // We accumulate locally in registers and atomically scatter once at the end.
        double dst8[3][8] = { 0.0 };

        for ( int w = 0; w < 2; ++w )
        {
            // Coefficient k evaluated at quadrature point via average of 6 wedge nodes for 1x1 rule
            double k_sum = 0.0;
#pragma unroll
            for ( int node = 0; node < 6; ++node )
            {
                const int ddx = WEDGE_NODE_OFF[w][node][0];
                const int ddy = WEDGE_NODE_OFF[w][node][1];
                const int ddr = WEDGE_NODE_OFF[w][node][2];

                const int nid = node_id( tx + ddx, ty + ddy );
                const int lvl = lvl0 + ddr;

                k_sum += k_sh( nid, lvl );
            }
            const double k_eval = ONE_SIXTH * k_sum;

            // Compute inverse Jacobian and |det J| for this wedge element
            double wJ = 0.0;
            double i00, i01, i02;
            double i10, i11, i12;
            double i20, i21, i22;

            {
                const double half_dr = 0.5 * ( r_1 - r_0 );
                const double r_mid   = 0.5 * ( r_0 + r_1 );

                const double J_0_0 = r_mid * ( -ws[w][0][0] + ws[w][1][0] );
                const double J_0_1 = r_mid * ( -ws[w][0][0] + ws[w][2][0] );
                const double J_0_2 = half_dr * ( ONE_THIRD * ( ws[w][0][0] + ws[w][1][0] + ws[w][2][0] ) );

                const double J_1_0 = r_mid * ( -ws[w][0][1] + ws[w][1][1] );
                const double J_1_1 = r_mid * ( -ws[w][0][1] + ws[w][2][1] );
                const double J_1_2 = half_dr * ( ONE_THIRD * ( ws[w][0][1] + ws[w][1][1] + ws[w][2][1] ) );

                const double J_2_0 = r_mid * ( -ws[w][0][2] + ws[w][1][2] );
                const double J_2_1 = r_mid * ( -ws[w][0][2] + ws[w][2][2] );
                const double J_2_2 = half_dr * ( ONE_THIRD * ( ws[w][0][2] + ws[w][1][2] + ws[w][2][2] ) );

                const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                     J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;

                const double invJ = 1.0 / J_det;

                i00 = invJ * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                i01 = invJ * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                i02 = invJ * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );
                i10 = invJ * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                i11 = invJ * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                i12 = invJ * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
                i20 = invJ * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                i21 = invJ * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                i22 = invJ * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

                wJ = Kokkos::abs( J_det );
            }

            const double kwJ = k_eval * wJ;

            // Fused operator action:
            // - first pass builds grad(u) / div(u)-dependent accumulators
            // - second pass tests each basis function and accumulates to dst8
            double gu00 = 0.0;
            double gu10 = 0.0, gu11 = 0.0;
            double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
            double div_u = 0.0;

            if ( !diagonal_ )
            {
                // Build sym(grad u) and div(u) at quadrature point
                for ( int dimj = 0; dimj < 3; ++dimj )
                {
#pragma unroll
                    for ( int node_idx = cmb_shift; node_idx < 6 - surface_shift; ++node_idx )
                    {
                        const double gx = dN_ref[node_idx][0];
                        const double gy = dN_ref[node_idx][1];
                        const double gz = dN_ref[node_idx][2];

                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;

                        double E00, E11, E22, sym01, sym02, sym12, gdd;
                        column_grad_to_sym( dimj, g0, g1, g2, E00, E11, E22, sym01, sym02, sym12, gdd );

                        const int ddx = WEDGE_NODE_OFF[w][node_idx][0];
                        const int ddy = WEDGE_NODE_OFF[w][node_idx][1];
                        const int ddr = WEDGE_NODE_OFF[w][node_idx][2];

                        const int nid = node_id( tx + ddx, ty + ddy );
                        const int lvl = lvl0 + ddr;

                        const double s = src_sh( nid, dimj, lvl );

                        gu00 += E00 * s;
                        gu10 += sym01 * s;
                        gu11 += E11 * s;
                        gu20 += sym02 * s;
                        gu21 += sym12 * s;
                        gu22 += E22 * s;
                        div_u += gdd * s;
                    }
                }

                // Test against each basis function and accumulate local contributions
                for ( int dimi = 0; dimi < 3; ++dimi )
                {
#pragma unroll
                    for ( int node_idx = cmb_shift; node_idx < 6 - surface_shift; ++node_idx )
                    {
                        const double gx = dN_ref[node_idx][0];
                        const double gy = dN_ref[node_idx][1];
                        const double gz = dN_ref[node_idx][2];

                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;

                        double E00, E11, E22, sym01, sym02, sym12, gdd;
                        column_grad_to_sym( dimi, g0, g1, g2, E00, E11, E22, sym01, sym02, sym12, gdd );

                        const int u = WEDGE_TO_UNIQUE[w][node_idx];

                        dst8[dimi][u] += kwJ * ( NEG_TWO_THIRDS * div_u * gdd +
                                                 4.0 * sym01 * gu10 +
                                                 4.0 * sym02 * gu20 +
                                                 4.0 * sym12 * gu21 +
                                                 2.0 * E00 * gu00 +
                                                 2.0 * E11 * gu11 +
                                                 2.0 * E22 * gu22 );
                    }
                }
            }

            // Diagonal-only mode (or diagonal correction on DIRICHLET boundaries)
            if ( diagonal_ || ( treat_boundary_dirichlet && at_boundary ) )
            {
                for ( int dim_diagBC = 0; dim_diagBC < 3; ++dim_diagBC )
                {
#pragma unroll
                    for ( int node_idx = surface_shift; node_idx < 6 - cmb_shift; ++node_idx )
                    {
                        const double gx = dN_ref[node_idx][0];
                        const double gy = dN_ref[node_idx][1];
                        const double gz = dN_ref[node_idx][2];

                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;

                        double E00, E11, E22, sym01, sym02, sym12, gdd;
                        column_grad_to_sym( dim_diagBC, g0, g1, g2, E00, E11, E22, sym01, sym02, sym12, gdd );

                        const int ddx = WEDGE_NODE_OFF[w][node_idx][0];
                        const int ddy = WEDGE_NODE_OFF[w][node_idx][1];
                        const int ddr = WEDGE_NODE_OFF[w][node_idx][2];

                        const int nid = node_id( tx + ddx, ty + ddy );
                        const int lvl = lvl0 + ddr;

                        const double s = src_sh( nid, dim_diagBC, lvl );
                        const int    u = WEDGE_TO_UNIQUE[w][node_idx];

                        dst8[dim_diagBC][u] +=
                            kwJ * ( 4.0 * s * ( sym01 * sym01 + sym02 * sym02 + sym12 * sym12 ) +
                                    2.0 * s * ( E00 * E00 + E11 * E11 + E22 * E22 ) +
                                    NEG_TWO_THIRDS * ( gdd * gdd ) * s );
                    }
                }
            }
        } // wedge loop

        // Final atomic scatter to global dst vector (8 hex nodes x 3 components)
        for ( int dim_add = 0; dim_add < 3; ++dim_add )
        {
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell, dim_add ), dst8[dim_add][0] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell, dim_add ), dst8[dim_add][1] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell, dim_add ), dst8[dim_add][2] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1, dim_add ), dst8[dim_add][3] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim_add ), dst8[dim_add][4] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][5] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, dim_add ), dst8[dim_add][6] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][7] );
        }
    }

  public:
    /**
     * @brief Fallback functor entry point.
     *
     * Not used by apply_impl anymore (which launches specialized kernels directly),
     * but kept for compatibility/debug.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()( const Team& team ) const
    {
        if ( use_slow_path_ ) { operator_slow_kernel( team ); }
        else                  { operator_fast_kernel( team ); }
    }

    // -------------------------------------------------------------------------
    // The remaining methods are unchanged numerically; only comments added.
    // -------------------------------------------------------------------------

    /**
     * @brief Build trial/test symmetric-gradient vectors at one quadrature point and wedge.
     *
     * For a given pair of vector components (dimi,d
imj), this computes:
     * - sym_grad_i / sym_grad_j for all local wedge nodes
     * - scalar quadrature factor = weight * k(x_q) * |det J|
     *
     * These vectors are reused both for:
     * - fused local matvecs, and
     * - explicit local matrix assembly (outer products)
     */
    KOKKOS_INLINE_FUNCTION void assemble_trial_test_vecs(
        const int                               wedge,
        const dense::Vec< ScalarType, VecDim >& quad_point,
        const ScalarType                        quad_weight,
        const ScalarT                           r_1,
        const ScalarT                           r_2,
        dense::Vec< ScalarT, 3 > ( *wedge_phy_surf )[3],
        const dense::Vec< ScalarT, 6 >*           k_local_hex,
        const int                                 dimi,
        const int                                 dimj,
        dense::Mat< ScalarType, VecDim, VecDim >* sym_grad_i,
        dense::Mat< ScalarType, VecDim, VecDim >* sym_grad_j,
        ScalarType&                               jdet_keval_quadweight ) const
    {
        dense::Mat< ScalarType, VecDim, VecDim >       J       = jac( wedge_phy_surf[wedge], r_1, r_2, quad_point );
        const auto                                     det     = J.det();
        const auto                                     abs_det = Kokkos::abs( det );
        const dense::Mat< ScalarType, VecDim, VecDim > J_inv_transposed = J.inv_transposed( det );

        ScalarType k_eval = 0.0;
        for ( int k = 0; k < num_nodes_per_wedge; k++ )
        {
            k_eval += shape( k, quad_point ) * k_local_hex[wedge]( k );
        }

        for ( int k = 0; k < num_nodes_per_wedge; k++ )
        {
            sym_grad_i[k] = symmetric_grad( J_inv_transposed, quad_point, k, dimi );
            sym_grad_j[k] = symmetric_grad( J_inv_transposed, quad_point, k, dimj );
        }

        jdet_keval_quadweight = quad_weight * k_eval * abs_det;
    }

    /**
     * @brief Assemble one wedge-local 18x18 matrix on-the-fly.
     *
     * This is used by the slow path when local matrices are not stored, and by GCA workflows.
     */
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > assemble_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        dense::Vec< ScalarT, 6 > k_local_hex[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A = {};

        for ( int dimi = 0; dimi < 3; ++dimi )
        {
            for ( int dimj = 0; dimj < 3; ++dimj )
            {
                for ( int q = 0; q < num_quad_points; q++ )
                {
                    dense::Mat< ScalarType, VecDim, VecDim > sym_grad_i[num_nodes_per_wedge];
                    dense::Mat< ScalarType, VecDim, VecDim > sym_grad_j[num_nodes_per_wedge];
                    ScalarType                               jdet_keval_quadweight = 0;

                    assemble_trial_test_vecs(
                        wedge,
                        quad_points[q],
                        quad_weights[q],
                        r_1,
                        r_2,
                        wedge_phy_surf,
                        k_local_hex,
                        dimi,
                        dimj,
                        sym_grad_i,
                        sym_grad_j,
                        jdet_keval_quadweight );

                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            A( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) +=
                                jdet_keval_quadweight *
                                ( 2 * sym_grad_j[j].double_contract( sym_grad_i[i] ) -
                                  2.0 / 3.0 * sym_grad_j[j]( dimj, dimj ) * sym_grad_i[i]( dimi, dimi ) );
                        }
                    }
                }
            }
        }

        return A;
    }
};

static_assert( linalg::GCACapable< EpsilonDivDivKerngenV07SplitPaths< float > > );
static_assert( linalg::GCACapable< EpsilonDivDivKerngenV07SplitPaths< double > > );

} // namespace terra::fe::wedge::operators::shell::epsdivdiv_history