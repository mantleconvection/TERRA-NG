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
#include "linalg/vector_q1.hpp"
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
 * This class supports three execution paths:
 *
 * 1) Slow path (local matrix path)
 *    - Used if local matrices are stored (full or selective)
 *    - Reuses assembled/stored 18x18 local matrices
 *
 * 2) Fast path: Dirichlet/Neumann
 *    - Matrix-free
 *    - Handles Dirichlet by skipping constrained face-node couplings (and diagonal treatment if requested)
 *    - Neumann is naturally included
 *
 * 3) Fast path: Free-slip
 *    - Matrix-free
 *    - Applies trial and test-side tangential projection on free-slip boundaries
 *    - Preserves behavior consistent with the slow path for iterative solves
 *
 * IMPORTANT DESIGN CHANGE:
 * ------------------------
 * The path decision is made on the HOST and cached in `kernel_path_`.
 * `apply_impl()` dispatches to a different kernel launch per path.
 * This avoids a runtime branch inside the hot device kernel.
 */
template < typename ScalarT, int VecDim = 3 >
class EpsilonDivDivKerngenV08ScalarCoalesced
{
  public:
    using SrcVectorType                 = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType                 = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType                    = ScalarT;
    static constexpr int LocalMatrixDim = 18;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, LocalMatrixDim, LocalMatrixDim, 2 >;
    using LocalMatrixStorage      = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >;
    using Team                    = Kokkos::TeamPolicy<>::member_type;

  private:
    // Optional element-local matrix storage (GCA/coarsening/explicit local-matrix mode)
    LocalMatrixStorage local_matrix_storage_;

    // Domain and geometry / coefficients
    grid::shell::DistributedDomain                           domain_;
    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;   ///< Lateral shell geometry (unit sphere coords)
    grid::Grid2DDataScalar< ScalarT >                        radii_;  ///< Radial coordinates per local subdomain
    grid::Grid4DDataScalar< ScalarType >                     k_;      ///< Scalar coefficient field
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;   ///< Boundary flags per cell/node
    BoundaryConditions                                       bcs_;    ///< CMB and SURFACE boundary conditions

    bool diagonal_; ///< If true, apply diagonal-only action

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;
    linalg::OperatorStoredMatrixMode  operator_stored_matrix_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim >                    recv_buffers_;
    terra::communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataVec< ScalarType, VecDim > > comm_plan_;

    // Device views captured by kernels
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;
    grid::Grid4DDataVec< ScalarType, VecDim > src_;

    // Quadrature (Felippa 1x1 on wedge)
    const int                num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;
    dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
    ScalarT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];

    // Local subdomain extents (in cells)
    int local_subdomains_;
    int hex_lat_;
    int hex_rad_;
    int lat_refinement_level_;

    // Team-policy tiling (one team handles a slab lat_tile x lat_tile x r_tile cells)
    int lat_tile_;
    int r_tile_;
    int lat_tiles_;
    int r_tiles_;
    int team_size_;
    int blocks_;

    ScalarT r_max_;
    ScalarT r_min_;

    /**
     * @brief Kernel path selected on host.
     *
     * This value is computed whenever BCs or stored-matrix mode change and is then
     * used by `apply_impl()` to select which kernel launch to issue.
     */
    enum class KernelPath
    {
        Slow,
        FastDirichletNeumann,
        FastFreeslip,
    };
    KernelPath kernel_path_ = KernelPath::FastDirichletNeumann;

  private:
    /**
     * @brief Recompute the kernel path on host.
     *
     * Rules:
     * - Any stored local matrix mode => slow path
     * - Else if any free-slip BC on CMB or SURFACE => fast free-slip path
     * - Else => fast Dirichlet/Neumann path
     */
    void update_kernel_path_flag_host_only()
    {
        const BoundaryConditionFlag cmb_bc     = get_boundary_condition_flag( bcs_, CMB );
        const BoundaryConditionFlag surface_bc = get_boundary_condition_flag( bcs_, SURFACE );

        const bool has_freeslip        = ( cmb_bc == FREESLIP ) || ( surface_bc == FREESLIP );
        const bool has_stored_matrices = ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off );

        if ( has_stored_matrices )
            kernel_path_ = KernelPath::Slow;
        else if ( has_freeslip )
            kernel_path_ = KernelPath::FastFreeslip;
        else
            kernel_path_ = KernelPath::FastDirichletNeumann;
    }

  public:
    EpsilonDivDivKerngenV08ScalarCoalesced(
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

        lat_tile_ = 4;
        r_tile_   = 16;

        lat_tiles_ = ( hex_lat_ + lat_tile_ - 1 ) / lat_tile_;
        r_tiles_   = ( hex_rad_ + r_tile_ - 1 ) / r_tile_;

        team_size_ = lat_tile_ * lat_tile_ * r_tile_;
        blocks_    = local_subdomains_ * lat_tiles_ * lat_tiles_ * r_tiles_;

        r_min_ = domain_info.radii()[0];
        r_max_ = domain_info.radii()[domain_info.radii().size() - 1];

        // Host-side path selection (no in-kernel path branching)
        update_kernel_path_flag_host_only();

        util::logroot << "[EpsilonDivDiv] tile size (x,y,r)=(" << lat_tile_ << "," << lat_tile_ << "," << r_tile_ << ")"
                      << std::endl;
        util::logroot << "[EpsilonDivDiv] number of tiles (x,y,r)=(" << lat_tiles_ << "," << lat_tiles_ << ","
                      << r_tiles_ << "), team_size=" << team_size_ << ", blocks=" << blocks_ << std::endl;
        const char* path_name = ( kernel_path_ == KernelPath::Slow )         ? "slow" :
                                ( kernel_path_ == KernelPath::FastFreeslip ) ? "fast-freeslip" :
                                                                               "fast-dirichlet-neumann";
        util::logroot << "[EpsilonDivDiv] kernel path = " << path_name << std::endl;
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void set_diagonal( bool v ) { diagonal_ = v; }

    void set_boundary_conditions( BoundaryConditions bcs )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];

        // Recompute host-side dispatch mode whenever BCs change
        update_kernel_path_flag_host_only();
    }

    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }
    const grid::shell::DistributedDomain&       get_domain() const { return domain_; }
    grid::Grid2DDataScalar< ScalarT >           get_radii() const { return radii_; }
    grid::Grid3DDataVec< ScalarT, 3 >           get_grid() { return grid_; }

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

        // Recompute host-side dispatch mode whenever storage mode changes
        update_kernel_path_flag_host_only();

        util::logroot << "[EpsilonDivDiv] (set_stored_matrix_mode) kernel path = "
                      << (( kernel_path_ == KernelPath::Slow ) ? "slow" :
                          ( kernel_path_ == KernelPath::FastFreeslip ) ? "fast-freeslip" :
                                                                         "fast-dirichlet-neumann")
                      << std::endl;
    }

    linalg::OperatorStoredMatrixMode get_stored_matrix_mode() { return operator_stored_matrix_mode_; }

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
     * @brief Apply the operator: dst <- Op(src) (or additive, depending on mode).
     *
     * Host-side responsibilities:
     * - Handle replace/add mode initialization
     * - Cache src/dst views for kernel capture
     * - Build team policy
     * - Dispatch to exactly one kernel variant based on `kernel_path_`
     * - Optional halo communication accumulation
     *
     * This is where the path decision now lives (host), so device code does not branch
     * on `kernel_path_` in the hot path.
     */
    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "epsilon_divdiv_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        dst_ = dst.grid_data();
        src_ = src.grid_data();

        util::Timer          timer_kernel( "epsilon_divdiv_kernel" );
        Kokkos::TeamPolicy<> policy( blocks_, team_size_ );

        // Fast paths use shared-memory slab staging
        if ( kernel_path_ != KernelPath::Slow )
        {
            policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );
        }

        // Host-side dispatch => no per-team path branching in device code
        if ( kernel_path_ == KernelPath::Slow )
        {
            Kokkos::parallel_for(
                "epsilon_divdiv_apply_kernel_slow", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->run_team_slow( team );
                } );
        }
        else if ( kernel_path_ == KernelPath::FastFreeslip )
        {
            Kokkos::parallel_for(
                "epsilon_divdiv_apply_kernel_fast_freeslip", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                    this->run_team_fast_freeslip( team );
                } );
        }
        else
        {
            // Launch bounds: 256 threads/block, min 3 blocks/SM to reduce register pressure
            Kokkos::TeamPolicy<  > dn_policy( blocks_, team_size_ );
            dn_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );
            Kokkos::parallel_for(
                "epsilon_divdiv_apply_kernel_fast_dirichlet_neumann",
                dn_policy,
                KOKKOS_CLASS_LAMBDA( const Team& team ) { this->run_team_fast_dirichlet_neumann( team ); } );
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
     * @brief Convert one gradient column into symmetric-gradient and div contributions.
     *
     * Given grad(phi e_dim), this helper computes:
     * - diagonal entries (E00,E11,E22)
     * - off-diagonal symmetric entries (sym01,sym02,sym12)
     * - divergence contribution `gdd`
     *
     * The fast kernels use this repeatedly in fused operator application.
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
     * @brief Team scratch memory size for fast paths.
     *
     * Layout per team:
     *   [coords_sh | src_sh | k_sh | r_sh | padding]
     */
    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size( const int ts ) const
    {
        const int nlev = r_tile_ + 1;
        const int n    = lat_tile_ + 1;
        const int nxy  = n * n;

        const size_t nscalars =
            size_t( nxy ) * 3 + size_t( nxy ) * 3 * nlev + size_t( nxy ) * nlev + size_t( nlev ) + 1;

        return sizeof( ScalarType ) * nscalars;
    }

  private:
    /**
     * @brief Decode a team league rank / team rank into:
     * - tile origin (x0,y0,r0)
     * - intra-tile thread coordinates (tx,ty,tr)
     * - target cell (x_cell,y_cell,r_cell)
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
        tr            = tid % r_tile_;
        tx            = ( tid / r_tile_ ) % lat_tile_;
        ty            = tid / ( r_tile_ * lat_tile_ );

        x_cell = x0 + tx;
        y_cell = y0 + ty;
        r_cell = r0 + tr;
    }

    // -------------------------------------------------------------------------
    // Path-specific team entry points (NO path branching)
    // These are the functions called directly by host-dispatched kernel launches.
    // -------------------------------------------------------------------------

    /**
     * @brief Team entry for slow (local-matrix) path.
     *
     * This wrapper performs only:
     * - team index decoding
     * - boundary flag queries
     * - forwarding to operator_slow_path()
     *
     * No dispatch branching happens here.
     */
    KOKKOS_INLINE_FUNCTION
    void run_team_slow( const Team& team ) const
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
     * @brief Team entry for fast Dirichlet/Neumann matrix-free path.
     */
    KOKKOS_INLINE_FUNCTION
    void run_team_fast_dirichlet_neumann( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
        const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

        operator_fast_dirichlet_neumann_path(
            team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell, at_cmb, at_surface );
    }

    /**
     * @brief Team entry for fast free-slip matrix-free path.
     */
    KOKKOS_INLINE_FUNCTION
    void run_team_fast_freeslip( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
        const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

        operator_fast_freeslip_path(
            team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell, at_cmb, at_surface );
    }

    // ===================== SLOW PATH =====================
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
        (void) team;
        (void) x0;
        (void) y0;
        (void) r0;
        (void) tx;
        (void) ty;
        (void) tr;

        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ || r_cell >= hex_rad_ )
            return;

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
                                    boundary_mask( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) =
                                        0.0;
                                }
                            }
                        }
                    }
                }
            }
            else if ( bcf == FREESLIP )
            {
                freeslip_reorder                                                                     = true;
                dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A_tmp[num_wedges_per_hex_cell] = { 0 };

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

                constexpr int layer_hex_offset_x[2][3] = { { 0, 1, 0 }, { 1, 0, 1 } };
                constexpr int layer_hex_offset_y[2][3] = { { 0, 0, 1 }, { 1, 1, 0 } };

                for ( int wedge = 0; wedge < 2; ++wedge )
                {
                    for ( int i = 0; i < LocalMatrixDim; ++i )
                    {
                        R[wedge]( i, i ) = 1.0;
                    }

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

                        const int offset_in_R = at_cmb ? 0 : 9;
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

                    A[wedge] = R[wedge] * A_tmp[wedge] * R[wedge].transposed();

                    auto src_tmp = R[wedge] * src[wedge];
                    for ( int i = 0; i < 18; ++i )
                    {
                        src[wedge]( i ) = src_tmp( i );
                    }

                    const int node_start = at_surface ? 3 : 0;
                    const int node_end   = at_surface ? 6 : 3;

                    for ( int node_idx = node_start; node_idx < node_end; node_idx++ )
                    {
                        const int idx = node_idx * 3;
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
        }

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            A[wedge].hadamard_product( boundary_mask );
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        dense::Vec< ScalarT, LocalMatrixDim > dst[num_wedges_per_hex_cell];
        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

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

        for ( int dimi = 0; dimi < 3; dimi++ )
        {
            dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
            dst_d[0] = dst[0].template slice< 6 >( dimi * num_nodes_per_wedge );
            dst_d[1] = dst[1].template slice< 6 >( dimi * num_nodes_per_wedge );

            atomically_add_local_wedge_vector_coefficients(
                dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst_d );
        }
    }

    // ===================== FAST DIRICHLET/NEUMANN PATH =====================
    KOKKOS_INLINE_FUNCTION
    void operator_fast_dirichlet_neumann_path(
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
        const int nlev = r_tile_ + 1;
        const int nxy  = ( lat_tile_ + 1 ) * ( lat_tile_ + 1 );

        double* shmem =
            reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size( team.team_size() ) ) );

        using ScratchCoords =
            Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchSrc = Kokkos::
            View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
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

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nlev ), [&]( int lvl ) {
            const int rr = r0 + lvl;
            r_sh( lvl )  = ( rr <= hex_rad_ ) ? radii_( local_subdomain_id, rr ) : 0.0;
        } );

        const int total_pairs = nxy * nlev;
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, total_pairs ), [&]( int t ) {
            const int node = t / nlev;
            const int lvl  = t - node * nlev;

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

        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ || r_cell >= hex_rad_ )
            return;

        const int    lvl0 = tr;
        const double r_0  = r_sh( lvl0 );
        const double r_1  = r_sh( lvl0 + 1 );

        const bool at_boundary              = at_cmb || at_surface;
        bool       treat_boundary_dirichlet = false;
        if ( at_boundary )
        {
            const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
            treat_boundary_dirichlet    = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
        }

        const int cmb_shift = ( ( at_boundary && treat_boundary_dirichlet && ( !diagonal_ ) && at_cmb ) ? 3 : 0 );
        const int surface_shift =
            ( ( at_boundary && treat_boundary_dirichlet && ( !diagonal_ ) && at_surface ) ? 3 : 0 );

        static constexpr int WEDGE_NODE_OFF[2][6][3] = {
            { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
            { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

        constexpr double ONE_THIRD      = 1.0 / 3.0;
        constexpr double ONE_SIXTH      = 1.0 / 6.0;
        constexpr double NEG_TWO_THIRDS = -0.66666666666666663;

        static constexpr double dN_ref[6][3] = {
            { -0.5, -0.5, -ONE_SIXTH },
            { 0.5, 0.0, -ONE_SIXTH },
            { 0.0, 0.5, -ONE_SIXTH },
            { -0.5, -0.5, ONE_SIXTH },
            { 0.5, 0.0, ONE_SIXTH },
            { 0.0, 0.5, ONE_SIXTH } };

        const int n00 = node_id( tx, ty );
        const int n01 = node_id( tx, ty + 1 );
        const int n10 = node_id( tx + 1, ty );
        const int n11 = node_id( tx + 1, ty + 1 );

        for ( int w = 0; w < 2; ++w )
        {
            const int v0 = w == 0 ? n00 : n11;
            const int v1 = w == 0 ? n10 : n01;
            const int v2 = w == 0 ? n01 : n10;

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

            double wJ = 0.0;
            double i00, i01, i02;
            double i10, i11, i12;
            double i20, i21, i22;

            {
                const double half_dr = 0.5 * ( r_1 - r_0 );
                const double r_mid   = 0.5 * ( r_0 + r_1 );

                const double J_0_0 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v1, 0 ) );
                const double J_0_1 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v2, 0 ) );
                const double J_0_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 0 ) + coords_sh( v1, 0 ) + coords_sh( v2, 0 ) ) );

                const double J_1_0 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v1, 1 ) );
                const double J_1_1 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v2, 1 ) );
                const double J_1_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 1 ) + coords_sh( v1, 1 ) + coords_sh( v2, 1 ) ) );

                const double J_2_0 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v1, 2 ) );
                const double J_2_1 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v2, 2 ) );
                const double J_2_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 2 ) + coords_sh( v1, 2 ) + coords_sh( v2, 2 ) ) );

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

            // Per-wedge output accumulator (scatter after each wedge to reduce register pressure)
            double dst_w[3][6] = { 0.0 };

            double gu00 = 0.0;
            double gu10 = 0.0, gu11 = 0.0;
            double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
            double div_u = 0.0;

            if ( !diagonal_ )
            {
                // Trial side: accumulate symmetric gradient of u (fused dim loops).
                // Physical gradients computed inline to avoid local memory spills.
#pragma unroll
                for ( int n = cmb_shift; n < 6 - surface_shift; ++n )
                {
                    const double gx = dN_ref[n][0];
                    const double gy = dN_ref[n][1];
                    const double gz = dN_ref[n][2];
                    const double g0 = i00 * gx + i01 * gy + i02 * gz;
                    const double g1 = i10 * gx + i11 * gy + i12 * gz;
                    const double g2 = i20 * gx + i21 * gy + i22 * gz;

                    const int ddx = WEDGE_NODE_OFF[w][n][0];
                    const int ddy = WEDGE_NODE_OFF[w][n][1];
                    const int ddr = WEDGE_NODE_OFF[w][n][2];
                    const int nid = node_id( tx + ddx, ty + ddy );
                    const int lvl = lvl0 + ddr;

                    const double s0 = src_sh( nid, 0, lvl );
                    const double s1 = src_sh( nid, 1, lvl );
                    const double s2 = src_sh( nid, 2, lvl );

                    gu00  += g0 * s0;
                    gu11  += g1 * s1;
                    gu22  += g2 * s2;
                    gu10  += 0.5 * ( g1 * s0 + g0 * s1 );
                    gu20  += 0.5 * ( g2 * s0 + g0 * s2 );
                    gu21  += 0.5 * ( g2 * s1 + g1 * s2 );
                    div_u += g0 * s0 + g1 * s1 + g2 * s2;
                }

                // Test side: compute dst for all 3 dims at once (fused).
                // 2*epsilon(v):epsilon(u) - (2/3)*div(v)*div(u)
#pragma unroll
                for ( int n = cmb_shift; n < 6 - surface_shift; ++n )
                {
                    const double gx = dN_ref[n][0];
                    const double gy = dN_ref[n][1];
                    const double gz = dN_ref[n][2];
                    const double g0 = i00 * gx + i01 * gy + i02 * gz;
                    const double g1 = i10 * gx + i11 * gy + i12 * gz;
                    const double g2 = i20 * gx + i21 * gy + i22 * gz;

                    dst_w[0][n] +=
                        kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 ) + NEG_TWO_THIRDS * g0 * div_u );
                    dst_w[1][n] +=
                        kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 ) + NEG_TWO_THIRDS * g1 * div_u );
                    dst_w[2][n] +=
                        kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 ) + NEG_TWO_THIRDS * g2 * div_u );
                }
            }

            if ( diagonal_ || ( treat_boundary_dirichlet && at_boundary ) )
            {
                // Diagonal action: kwJ * s_d * (|g|^2 + (1/3) * g_d^2)
#pragma unroll
                for ( int n = surface_shift; n < 6 - cmb_shift; ++n )
                {
                    const double gx = dN_ref[n][0];
                    const double gy = dN_ref[n][1];
                    const double gz = dN_ref[n][2];
                    const double g0 = i00 * gx + i01 * gy + i02 * gz;
                    const double g1 = i10 * gx + i11 * gy + i12 * gz;
                    const double g2 = i20 * gx + i21 * gy + i22 * gz;
                    const double gg = g0 * g0 + g1 * g1 + g2 * g2;

                    const int ddx = WEDGE_NODE_OFF[w][n][0];
                    const int ddy = WEDGE_NODE_OFF[w][n][1];
                    const int ddr = WEDGE_NODE_OFF[w][n][2];
                    const int nid = node_id( tx + ddx, ty + ddy );
                    const int lvl = lvl0 + ddr;

                    const double s0 = src_sh( nid, 0, lvl );
                    const double s1 = src_sh( nid, 1, lvl );
                    const double s2 = src_sh( nid, 2, lvl );
                    dst_w[0][n] += kwJ * s0 * ( gg + ONE_THIRD * g0 * g0 );
                    dst_w[1][n] += kwJ * s1 * ( gg + ONE_THIRD * g1 * g1 );
                    dst_w[2][n] += kwJ * s2 * ( gg + ONE_THIRD * g2 * g2 );
                }
            }

            // Scatter this wedge's contributions immediately
#pragma unroll
            for ( int n = 0; n < 6; ++n )
            {
                const int ddx = WEDGE_NODE_OFF[w][n][0];
                const int ddy = WEDGE_NODE_OFF[w][n][1];
                const int ddr = WEDGE_NODE_OFF[w][n][2];
                for ( int dim_add = 0; dim_add < 3; ++dim_add )
                {
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, dim_add ),
                        dst_w[dim_add][n] );
                }
            }

        }
    }

    // ===================== FAST FREESLIP PATH =====================
    KOKKOS_INLINE_FUNCTION
    void normalize3( double& x, double& y, double& z ) const
    {
        const double n2 = x * x + y * y + z * z;
        if ( n2 > 0.0 )
        {
            const double invn = 1.0 / Kokkos::sqrt( n2 );
            x *= invn;
            y *= invn;
            z *= invn;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void project_tangential_inplace(
        const double nx,
        const double ny,
        const double nz,
        double&      ux,
        double&      uy,
        double&      uz ) const
    {
        const double dot = nx * ux + ny * uy + nz * uz;
        ux -= dot * nx;
        uy -= dot * ny;
        uz -= dot * nz;
    }

    KOKKOS_INLINE_FUNCTION
    void operator_fast_freeslip_path(
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
        const int nlev = r_tile_ + 1;
        const int nxy  = ( lat_tile_ + 1 ) * ( lat_tile_ + 1 );

        double* shmem =
            reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size( team.team_size() ) ) );

        using ScratchCoords =
            Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchSrc = Kokkos::
            View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
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

        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nlev ), [&]( int lvl ) {
            const int rr = r0 + lvl;
            r_sh( lvl )  = ( rr <= hex_rad_ ) ? radii_( local_subdomain_id, rr ) : 0.0;
        } );

        const int total_pairs = nxy * nlev;
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, total_pairs ), [&]( int t ) {
            const int node = t / nlev;
            const int lvl  = t - node * nlev;

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

        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ || r_cell >= hex_rad_ )
            return;

        const int    lvl0 = tr;
        const double r_0  = r_sh( lvl0 );
        const double r_1  = r_sh( lvl0 + 1 );

        const BoundaryConditionFlag cmb_bc     = get_boundary_condition_flag( bcs_, CMB );
        const BoundaryConditionFlag surface_bc = get_boundary_condition_flag( bcs_, SURFACE );

        const bool cmb_freeslip      = at_cmb && ( cmb_bc == FREESLIP );
        const bool surf_freeslip     = at_surface && ( surface_bc == FREESLIP );
        const bool cmb_dirichlet     = at_cmb && ( cmb_bc == DIRICHLET );
        const bool surface_dirichlet = at_surface && ( surface_bc == DIRICHLET );

        const int cmb_shift     = ( ( cmb_dirichlet && ( !diagonal_ ) ) ? 3 : 0 );
        const int surface_shift = ( ( surface_dirichlet && ( !diagonal_ ) ) ? 3 : 0 );

        static constexpr int WEDGE_NODE_OFF[2][6][3] = {
            { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
            { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

        static constexpr int WEDGE_TO_UNIQUE[2][6] = {
            { 0, 1, 2, 3, 4, 5 },
            { 6, 2, 1, 7, 5, 4 }
        };

        constexpr double ONE_THIRD      = 1.0 / 3.0;
        constexpr double ONE_SIXTH      = 1.0 / 6.0;
        constexpr double NEG_TWO_THIRDS = -0.66666666666666663;

        static constexpr double dN_ref[6][3] = {
            { -0.5, -0.5, -ONE_SIXTH },
            { 0.5, 0.0, -ONE_SIXTH },
            { 0.0, 0.5, -ONE_SIXTH },
            { -0.5, -0.5, ONE_SIXTH },
            { 0.5, 0.0, ONE_SIXTH },
            { 0.0, 0.5, ONE_SIXTH } };

        const int n00 = node_id( tx, ty );
        const int n01 = node_id( tx, ty + 1 );
        const int n10 = node_id( tx + 1, ty );
        const int n11 = node_id( tx + 1, ty + 1 );

        double src8[3][8];

        src8[0][0] = src_sh( n00, 0, lvl0 );
        src8[1][0] = src_sh( n00, 1, lvl0 );
        src8[2][0] = src_sh( n00, 2, lvl0 );
        src8[0][1] = src_sh( n10, 0, lvl0 );
        src8[1][1] = src_sh( n10, 1, lvl0 );
        src8[2][1] = src_sh( n10, 2, lvl0 );
        src8[0][2] = src_sh( n01, 0, lvl0 );
        src8[1][2] = src_sh( n01, 1, lvl0 );
        src8[2][2] = src_sh( n01, 2, lvl0 );
        src8[0][6] = src_sh( n11, 0, lvl0 );
        src8[1][6] = src_sh( n11, 1, lvl0 );
        src8[2][6] = src_sh( n11, 2, lvl0 );

        src8[0][3] = src_sh( n00, 0, lvl0 + 1 );
        src8[1][3] = src_sh( n00, 1, lvl0 + 1 );
        src8[2][3] = src_sh( n00, 2, lvl0 + 1 );
        src8[0][4] = src_sh( n10, 0, lvl0 + 1 );
        src8[1][4] = src_sh( n10, 1, lvl0 + 1 );
        src8[2][4] = src_sh( n10, 2, lvl0 + 1 );
        src8[0][5] = src_sh( n01, 0, lvl0 + 1 );
        src8[1][5] = src_sh( n01, 1, lvl0 + 1 );
        src8[2][5] = src_sh( n01, 2, lvl0 + 1 );
        src8[0][7] = src_sh( n11, 0, lvl0 + 1 );
        src8[1][7] = src_sh( n11, 1, lvl0 + 1 );
        src8[2][7] = src_sh( n11, 2, lvl0 + 1 );

        double nx00 = coords_sh( n00, 0 ), ny00 = coords_sh( n00, 1 ), nz00 = coords_sh( n00, 2 );
        double nx10 = coords_sh( n10, 0 ), ny10 = coords_sh( n10, 1 ), nz10 = coords_sh( n10, 2 );
        double nx01 = coords_sh( n01, 0 ), ny01 = coords_sh( n01, 1 ), nz01 = coords_sh( n01, 2 );
        double nx11 = coords_sh( n11, 0 ), ny11 = coords_sh( n11, 1 ), nz11 = coords_sh( n11, 2 );

        normalize3( nx00, ny00, nz00 );
        normalize3( nx10, ny10, nz10 );
        normalize3( nx01, ny01, nz01 );
        normalize3( nx11, ny11, nz11 );

        // Save normal components of src at free-slip boundary nodes before trial-side projection.
        // Used in two places:
        //  - non-diagonal mode: to restore A_nn * u_n * n after the test-side projection
        //    (normal_corr mechanism).
        //  - diagonal mode: to reconstruct the unprojected src inside apply_rotated_diag so
        //    that R^T diag(RAR^T) R src can be evaluated.
        // Corner indexing: 0=n00, 1=n10, 2=n01, 3=n11.
        double u_n_cmb[4]  = {};
        double u_n_surf[4] = {};
        if ( cmb_freeslip )
        {
            u_n_cmb[0] = nx00 * src8[0][0] + ny00 * src8[1][0] + nz00 * src8[2][0];
            u_n_cmb[1] = nx10 * src8[0][1] + ny10 * src8[1][1] + nz10 * src8[2][1];
            u_n_cmb[2] = nx01 * src8[0][2] + ny01 * src8[1][2] + nz01 * src8[2][2];
            u_n_cmb[3] = nx11 * src8[0][6] + ny11 * src8[1][6] + nz11 * src8[2][6];
        }
        if ( surf_freeslip )
        {
            u_n_surf[0] = nx00 * src8[0][3] + ny00 * src8[1][3] + nz00 * src8[2][3];
            u_n_surf[1] = nx10 * src8[0][4] + ny10 * src8[1][4] + nz10 * src8[2][4];
            u_n_surf[2] = nx01 * src8[0][5] + ny01 * src8[1][5] + nz01 * src8[2][5];
            u_n_surf[3] = nx11 * src8[0][7] + ny11 * src8[1][7] + nz11 * src8[2][7];
        }

        if ( cmb_freeslip )
        {
            project_tangential_inplace( nx00, ny00, nz00, src8[0][0], src8[1][0], src8[2][0] );
            project_tangential_inplace( nx10, ny10, nz10, src8[0][1], src8[1][1], src8[2][1] );
            project_tangential_inplace( nx01, ny01, nz01, src8[0][2], src8[1][2], src8[2][2] );
            project_tangential_inplace( nx11, ny11, nz11, src8[0][6], src8[1][6], src8[2][6] );
        }
        if ( surf_freeslip )
        {
            project_tangential_inplace( nx00, ny00, nz00, src8[0][3], src8[1][3], src8[2][3] );
            project_tangential_inplace( nx10, ny10, nz10, src8[0][4], src8[1][4], src8[2][4] );
            project_tangential_inplace( nx01, ny01, nz01, src8[0][5], src8[1][5], src8[2][5] );
            project_tangential_inplace( nx11, ny11, nz11, src8[0][7], src8[1][7], src8[2][7] );
        }

        double dst8[3][8]        = { { 0.0 } };
        double normal_corr[3][8] = {};

        

        for ( int w = 0; w < 2; ++w )
        {
            const int v0 = w == 0 ? n00 : n11;
            const int v1 = w == 0 ? n10 : n01;
            const int v2 = w == 0 ? n01 : n10;

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

            double wJ = 0.0;
            double i00, i01, i02;
            double i10, i11, i12;
            double i20, i21, i22;

            {
                const double half_dr = 0.5 * ( r_1 - r_0 );
                const double r_mid   = 0.5 * ( r_0 + r_1 );

                const double J_0_0 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v1, 0 ) );
                const double J_0_1 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v2, 0 ) );
                const double J_0_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 0 ) + coords_sh( v1, 0 ) + coords_sh( v2, 0 ) ) );

                const double J_1_0 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v1, 1 ) );
                const double J_1_1 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v2, 1 ) );
                const double J_1_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 1 ) + coords_sh( v1, 1 ) + coords_sh( v2, 1 ) ) );

                const double J_2_0 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v1, 2 ) );
                const double J_2_1 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v2, 2 ) );
                const double J_2_2 = half_dr * ( ONE_THIRD * ( coords_sh( v0, 2 ) + coords_sh( v1, 2 ) + coords_sh( v2, 2 ) ) );

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

            // Accumulate normal-normal diagonal correction for free-slip boundary nodes.
            // For each boundary node u with outward unit normal n_u and shape-function physical
            // gradient g_u = J^{-T} dN_ref[node_idx]:
            //   A_nn_u += kwJ * ( |g_u|^2 + (1/3) * (n_u · g_u)^2 )
            // This equals n_u^T · A_3x3_self_u · n_u — the (0,0) entry of R_u A R_u^T in the
            // normal-tangential frame — which the slow path preserves on its diagonal.
            // Map: (w, in-boundary index ni in 0..2) -> corner index {0=n00,1=n10,2=n01,3=n11}
            // normal_corr is only needed for the non-diagonal operator apply (P A P + A_nn u_n n).
            // In diagonal mode the rotated-frame diagonal is used directly (see below).
            if ( !diagonal_ && ( cmb_freeslip || surf_freeslip ) )
            {
                static constexpr int CMB_NODE_TO_CORNER[2][3] = { { 0, 1, 2 }, { 3, 2, 1 } };
                const double         cn[4][3]                 = { { nx00, ny00, nz00 },
                                                   { nx10, ny10, nz10 },
                                                   { nx01, ny01, nz01 },
                                                   { nx11, ny11, nz11 } };
                if ( cmb_freeslip )
                {
                    for ( int ni = 0; ni < 3; ++ni )
                    {
                        const int    corner = CMB_NODE_TO_CORNER[w][ni];
                        const double nxu = cn[corner][0], nyu = cn[corner][1], nzu = cn[corner][2];

                        const double gx  = dN_ref[ni][0];
                        const double gy  = dN_ref[ni][1];
                        const double gz  = dN_ref[ni][2];
                        const double g0  = i00 * gx + i01 * gy + i02 * gz;
                        const double g1  = i10 * gx + i11 * gy + i12 * gz;
                        const double g2  = i20 * gx + i21 * gy + i22 * gz;
                        const double gg  = g0 * g0 + g1 * g1 + g2 * g2;
                        const double ng  = nxu * g0 + nyu * g1 + nzu * g2;
                        const double Ann = kwJ * ( gg + ONE_THIRD * ng * ng );
                        const double c   = Ann * u_n_cmb[corner];
                        const int    u   = WEDGE_TO_UNIQUE[w][ni];

                        normal_corr[0][u] += c * nxu;
                        normal_corr[1][u] += c * nyu;
                        normal_corr[2][u] += c * nzu;
                    }
                }
                if ( surf_freeslip )
                {
                    for ( int ni = 0; ni < 3; ++ni )
                    {
                        const int    corner = CMB_NODE_TO_CORNER[w][ni];
                        const double nxu = cn[corner][0], nyu = cn[corner][1], nzu = cn[corner][2];

                        const double gx  = dN_ref[ni + 3][0];
                        const double gy  = dN_ref[ni + 3][1];
                        const double gz  = dN_ref[ni + 3][2];
                        const double g0  = i00 * gx + i01 * gy + i02 * gz;
                        const double g1  = i10 * gx + i11 * gy + i12 * gz;
                        const double g2  = i20 * gx + i21 * gy + i22 * gz;
                        const double gg  = g0 * g0 + g1 * g1 + g2 * g2;
                        const double ng  = nxu * g0 + nyu * g1 + nzu * g2;
                        const double Ann = kwJ * ( gg + ONE_THIRD * ng * ng );
                        const double c   = Ann * u_n_surf[corner];
                        const int    u   = WEDGE_TO_UNIQUE[w][ni + 3];

                        normal_corr[0][u] += c * nxu;
                        normal_corr[1][u] += c * nyu;
                        normal_corr[2][u] += c * nzu;
                    }
                }
            }

            double gu00 = 0.0;
            double gu10 = 0.0, gu11 = 0.0;
            double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
            double div_u = 0.0;

            if ( !diagonal_ )
            {
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

                        const int    u = WEDGE_TO_UNIQUE[w][node_idx];
                        const double s = src8[dimj][u];

                        gu00 += E00 * s;
                        gu10 += sym01 * s;
                        gu11 += E11 * s;
                        gu20 += sym02 * s;
                        gu21 += sym12 * s;
                        gu22 += E22 * s;
                        div_u += gdd * s;
                    }
                }

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

                        dst8[dimi][u] +=
                            kwJ * ( NEG_TWO_THIRDS * div_u * gdd + 4.0 * sym01 * gu10 + 4.0 * sym02 * gu20 +
                                    4.0 * sym12 * gu21 + 2.0 * E00 * gu00 + 2.0 * E11 * gu11 + 2.0 * E22 * gu22 );
                    }
                }
            }


            if ( diagonal_ || cmb_dirichlet || surface_dirichlet )
            {
                for ( int dim_diagBC = 0; dim_diagBC < 3; ++dim_diagBC )
                {
                    for ( int node_idx = surface_shift; node_idx < 6 - cmb_shift; ++node_idx )
                    {
                        // In diagonal mode, free-slip boundary nodes are handled below with the
                        // rotated-frame diagonal (R^T diag(RAR^T) R) to match the slow path.
                        if ( diagonal_ && cmb_freeslip && node_idx < 3 )
                            continue;
                        if ( diagonal_ && surf_freeslip && node_idx >= 3 )
                            continue;

                        const double gx = dN_ref[node_idx][0];
                        const double gy = dN_ref[node_idx][1];
                        const double gz = dN_ref[node_idx][2];

                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;

                        double E00, E11, E22, sym01, sym02, sym12, gdd;
                        column_grad_to_sym( dim_diagBC, g0, g1, g2, E00, E11, E22, sym01, sym02, sym12, gdd );

                        const int    u = WEDGE_TO_UNIQUE[w][node_idx];
                        const double s = src8[dim_diagBC][u];

                        dst8[dim_diagBC][u] += kwJ * ( 4.0 * s * ( sym01 * sym01 + sym02 * sym02 + sym12 * sym12 ) +
                                                       2.0 * s * ( E00 * E00 + E11 * E11 + E22 * E22 ) +
                                                       NEG_TWO_THIRDS * ( gdd * gdd ) * s );
                    }
                }

                // For free-slip boundary nodes in diagonal mode: compute R^T diag(R A_3x3 R^T) R src.
                // The slow path takes the diagonal of the 18x18 matrix *after* rotating to the
                // normal-tangential frame, giving diag entries kwJ*(gg + (r_alpha . g)^2 / 3) for
                // alpha in {n, t1, t2}.  The Cartesian diagonal used above differs from this by a
                // rotation, so it must not be used for free-slip boundary nodes.
                if ( diagonal_ )
                {
                    static constexpr int FS_CORNER_MAP[2][3] = { { 0, 1, 2 }, { 3, 2, 1 } };
                    const double ncoords[4][3]               = { { nx00, ny00, nz00 },
                                                   { nx10, ny10, nz10 },
                                                   { nx01, ny01, nz01 },
                                                   { nx11, ny11, nz11 } };

                    auto apply_rotated_diag =
                        [&]( const int ni, const int node_idx, const double* u_n_arr ) {
                            const int    corner = FS_CORNER_MAP[w][ni];
                            const double nxu    = ncoords[corner][0];
                            const double nyu    = ncoords[corner][1];
                            const double nzu    = ncoords[corner][2];
                            const int    u      = WEDGE_TO_UNIQUE[w][node_idx];

                            const double gx = dN_ref[node_idx][0];
                            const double gy = dN_ref[node_idx][1];
                            const double gz = dN_ref[node_idx][2];

                            const double g0     = i00 * gx + i01 * gy + i02 * gz;
                            const double g1     = i10 * gx + i11 * gy + i12 * gz;
                            const double g2     = i20 * gx + i21 * gy + i22 * gz;
                            const double gg_loc = g0 * g0 + g1 * g1 + g2 * g2;

                            dense::Vec< double, 3 > n_vec;
                            n_vec( 0 )      = nxu;
                            n_vec( 1 )      = nyu;
                            n_vec( 2 )      = nzu;
                            const auto R_rot = trafo_mat_cartesian_to_normal_tangential< double >( n_vec );

                            // Reconstruct full (unprojected) src at this node.
                            // src8 was tangentially projected; u_n_arr holds the saved normal component.
                            const double s0 = src8[0][u] + u_n_arr[corner] * nxu;
                            const double s1 = src8[1][u] + u_n_arr[corner] * nyu;
                            const double s2 = src8[2][u] + u_n_arr[corner] * nzu;

                            // Physical gradient and src in normal-tangential frame
                            const double Rg0 = R_rot( 0, 0 ) * g0 + R_rot( 0, 1 ) * g1 + R_rot( 0, 2 ) * g2;
                            const double Rg1 = R_rot( 1, 0 ) * g0 + R_rot( 1, 1 ) * g1 + R_rot( 1, 2 ) * g2;
                            const double Rg2 = R_rot( 2, 0 ) * g0 + R_rot( 2, 1 ) * g1 + R_rot( 2, 2 ) * g2;
                            const double Rs0 = R_rot( 0, 0 ) * s0 + R_rot( 0, 1 ) * s1 + R_rot( 0, 2 ) * s2;
                            const double Rs1 = R_rot( 1, 0 ) * s0 + R_rot( 1, 1 ) * s1 + R_rot( 1, 2 ) * s2;
                            const double Rs2 = R_rot( 2, 0 ) * s0 + R_rot( 2, 1 ) * s1 + R_rot( 2, 2 ) * s2;

                            // Rotated-frame diagonal applied to Rsrc:
                            //   diag_alpha = kwJ * (gg_loc + (1/3) * Rg_alpha^2)
                            const double v0 = kwJ * ( gg_loc + ONE_THIRD * Rg0 * Rg0 ) * Rs0;
                            const double v1 = kwJ * ( gg_loc + ONE_THIRD * Rg1 * Rg1 ) * Rs1;
                            const double v2 = kwJ * ( gg_loc + ONE_THIRD * Rg2 * Rg2 ) * Rs2;

                            // Transform back to Cartesian: R^T [v0, v1, v2]
                            dst8[0][u] += R_rot( 0, 0 ) * v0 + R_rot( 1, 0 ) * v1 + R_rot( 2, 0 ) * v2;
                            dst8[1][u] += R_rot( 0, 1 ) * v0 + R_rot( 1, 1 ) * v1 + R_rot( 2, 1 ) * v2;
                            dst8[2][u] += R_rot( 0, 2 ) * v0 + R_rot( 1, 2 ) * v1 + R_rot( 2, 2 ) * v2;
                        };

                    if ( cmb_freeslip )
                    {
                        for ( int ni = 0; ni < 3; ++ni )
                            apply_rotated_diag( ni, ni, u_n_cmb );
                    }
                    if ( surf_freeslip )
                    {
                        for ( int ni = 0; ni < 3; ++ni )
                            apply_rotated_diag( ni, ni + 3, u_n_surf );
                    }
                }
            }
        }

        // Test-side projection for free-slip (P A P).
        // In diagonal mode the rotated-frame diagonal is used, which already encodes the correct
        // normal/tangential split — no post-projection is needed or wanted.
        if ( !diagonal_ && cmb_freeslip )
        {
            project_tangential_inplace( nx00, ny00, nz00, dst8[0][0], dst8[1][0], dst8[2][0] );
            project_tangential_inplace( nx10, ny10, nz10, dst8[0][1], dst8[1][1], dst8[2][1] );
            project_tangential_inplace( nx01, ny01, nz01, dst8[0][2], dst8[1][2], dst8[2][2] );
            project_tangential_inplace( nx11, ny11, nz11, dst8[0][6], dst8[1][6], dst8[2][6] );
        }
        if ( !diagonal_ && surf_freeslip )
        {
            project_tangential_inplace( nx00, ny00, nz00, dst8[0][3], dst8[1][3], dst8[2][3] );
            project_tangential_inplace( nx10, ny10, nz10, dst8[0][4], dst8[1][4], dst8[2][4] );
            project_tangential_inplace( nx01, ny01, nz01, dst8[0][5], dst8[1][5], dst8[2][5] );
            project_tangential_inplace( nx11, ny11, nz11, dst8[0][7], dst8[1][7], dst8[2][7] );
        }

        // Add back the normal-normal diagonal contribution removed by the test-side projection.
        // This makes the fast path equivalent to the slow path for arbitrary input vectors.
        // Not needed in diagonal mode: the rotated-frame diagonal already includes A_nn * u_n.
        if ( !diagonal_ && ( cmb_freeslip || surf_freeslip ) )
        {
            for ( int dim = 0; dim < 3; ++dim )
                for ( int u = 0; u < 8; ++u )
                    dst8[dim][u] += normal_corr[dim][u];
        }

        for ( int dim_add = 0; dim_add < 3; ++dim_add )
        {
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell, dim_add ), dst8[dim_add][0] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell, dim_add ), dst8[dim_add][1] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell, dim_add ), dst8[dim_add][2] );
            Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1, dim_add ), dst8[dim_add][3] );
            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim_add ), dst8[dim_add][4] );
            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][5] );
            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, dim_add ), dst8[dim_add][6] );
            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][7] );
        }
    }

  public:
    /**
     * @brief Legacy generic team operator.
     *
     * Kept for compatibility/debugging, but no longer used by apply_impl().
     * The host now dispatches directly to path-specific kernels.
     *
     * This function still works, but it reintroduces a branch on `kernel_path_`
     * and should therefore be avoided in performance-critical use.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
        const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

        if ( kernel_path_ == KernelPath::Slow )
        {
            operator_slow_path(
                team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell, at_cmb, at_surface );
        }
        else if ( kernel_path_ == KernelPath::FastFreeslip )
        {
            operator_fast_freeslip_path(
                team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell, at_cmb, at_surface );
        }
        else
        {
            operator_fast_dirichlet_neumann_path(
                team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell, at_cmb, at_surface );
        }
    }

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
     * @brief Assemble one wedge-local 18x18 matrix (slow path / on-demand assembly).
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

static_assert( linalg::GCACapable< EpsilonDivDivKerngenV08ScalarCoalesced< float > > );
static_assert( linalg::GCACapable< EpsilonDivDivKerngenV08ScalarCoalesced< double > > );

} // namespace terra::fe::wedge::operators::shell::epsdivdiv_history
