#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "impl/Kokkos_Profiling.hpp"
#ifdef KOKKOS_ENABLE_OPENMP
#include "impl/Kokkos_HostThreadTeam.hpp"
#endif
#include "grid/bit_masks.hpp"
#include "kernels/common/grid_operations.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/gca/local_matrix_storage.hpp"
#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

#include <cstdlib>   // for std::getenv (wave-parallel DN path opt-in)

namespace terra::fe::wedge::operators::shell {

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

// Optional sweep-harness overrides for the EpsDivDivKerngen GPU tile config.
// 0 = use the hardcoded production default (lat=4, r=16, r_passes=2). The
// benchmark binary sets these from CLI args (--lat-tile, --r-tile, --r-passes).
inline int g_epsdivdiv_lat_tile_override = 0;
inline int g_epsdivdiv_r_tile_override   = 0;
inline int g_epsdivdiv_r_passes_override = 0;

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
class EpsilonDivDivKerngen
{
  public:
    using SrcVectorType                 = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType                 = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType                    = ScalarT;
    static constexpr int LocalMatrixDim = 18;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, LocalMatrixDim, LocalMatrixDim, 2 >;
    using LocalMatrixStorage      = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >;
    using Team                    = Kokkos::TeamPolicy<>::member_type;

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
#ifdef __HIP_PLATFORM_AMD__
        FastDirichletNeumannWave,   // wave-parallel experimental DN path (HIP/AMD layout)
        FastDirichletNeumannHex,    // hex 1-pt Gauss experimental DN path (HIP only)
#endif
    };

  private:
    // Optional element-local matrix storage (GCA/coarsening/explicit local-matrix mode)
    LocalMatrixStorage local_matrix_storage_;

    // Domain and geometry / coefficients
    grid::shell::DistributedDomain                           domain_;
    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;  ///< Lateral shell geometry (unit sphere coords)
    grid::Grid2DDataScalar< ScalarT >                        radii_; ///< Radial coordinates per local subdomain
    grid::Grid4DDataScalar< ScalarType >                     k_;     ///< Scalar coefficient field
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;  ///< Boundary flags per cell/node
    BoundaryConditions                                       bcs_;   ///< CMB and SURFACE boundary conditions

    bool diagonal_; ///< If true, apply diagonal-only action
    /// When true, replace the per-cell coefficient with a per-wedge constant.
    /// Mean kind chosen by eta_homogenization_mean_:
    ///   1 = harmonic mean of eta  (k_eval = harmonic mean of k=eta)
    ///   2 = arithmetic mean of eta (k_eval = arithmetic mean of k=eta)
    ///   3 = geometric mean of eta  (k_eval = geometric mean of k=eta)
    /// All match ASPECT MaterialAveraging::{harmonic,arithmetic,geometric}.
    /// Default mode = harmonic.  Default off.
    bool homogenize_eta_per_cell_ = false;
    int  eta_homogenization_mean_ = 1;

    /// When true, read a per-cell-constant coefficient from k_q0_ at the
    /// operator's own cell index (subdomain, x_cell, y_cell, r_cell) —
    /// direct lookup, no level translation.  Caller supplies a Q0 field at
    /// the operator's own MG level.
    bool                                 use_q0_coefficient_ = false;
    grid::Grid4DDataScalar< ScalarType > k_q0_;

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
    int r_passes_;
    int r_tile_block_;
    int lat_tiles_;
    int r_tiles_;
    int team_size_;
    int blocks_;

    ScalarT r_max_;
    ScalarT r_min_;

    KernelPath kernel_path_ = KernelPath::FastDirichletNeumann;

    // Rotation null-space penalty (active only for fs/fs)
    bool                                              penalty_active_  = false;
    ScalarT                                           penalty_epsilon_ = 1e-7;
    grid::Grid4DDataVec< ScalarType, VecDim >         null_modes_[3];
    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > ownership_mask_;

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
        // Serial backend: always use slow path (fast paths require shared memory / larger teams).
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > )
        {
            kernel_path_ = KernelPath::Slow;
            return;
        }

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
    EpsilonDivDivKerngen(
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

        // On Serial backend, team_size must be 1 => use 1x1x1 tiles.
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > )
        {
            lat_tile_ = 1;
            r_tile_   = 1;
            r_passes_ = 1;
        }
#ifdef KOKKOS_ENABLE_OPENMP
        // OpenMP host teams are capped at min(num_threads, 64) by Kokkos.
        // Choose tile sizes so that team_size = lat^2 * r_tile fits.
        else if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::OpenMP > )
        {
            const int max_team = std::min(
                Kokkos::OpenMP().concurrency(),
                static_cast< int >( Kokkos::Impl::HostThreadTeamData::max_team_members ) );
            if ( max_team >= 64 )
            {
                lat_tile_ = 4;
                r_tile_   = 4;
                r_passes_ = 4; // team_size = 64
            }
            else if ( max_team >= 16 )
            {
                lat_tile_ = 4;
                r_tile_   = 1;
                r_passes_ = 16; // team_size = 16
            }
            else
            {
                lat_tile_ = 1;
                r_tile_   = 1;
                r_passes_ = 1; // team_size = 1 (serial-like)
            }
        }
#endif
        else
        {
            // A/B test: cross-branch peak tile (was 4/8/2 on mt-sweeps default).
            // Sweep-harness overrides via CLI; 0 = keep default.
            lat_tile_ = g_epsdivdiv_lat_tile_override > 0 ? g_epsdivdiv_lat_tile_override : 4;
            r_tile_   = g_epsdivdiv_r_tile_override   > 0 ? g_epsdivdiv_r_tile_override   : 16;
            r_passes_ = g_epsdivdiv_r_passes_override > 0 ? g_epsdivdiv_r_passes_override : 2;
        }
        r_tile_block_ = r_tile_ * r_passes_;

        lat_tiles_ = ( hex_lat_ + lat_tile_ - 1 ) / lat_tile_;
        r_tiles_   = ( hex_rad_ + r_tile_block_ - 1 ) / r_tile_block_;

        team_size_ = lat_tile_ * lat_tile_ * r_tile_;
        blocks_    = local_subdomains_ * lat_tiles_ * lat_tiles_ * r_tiles_;

        r_min_ = domain_info.radii()[0];
        r_max_ = domain_info.radii()[domain_info.radii().size() - 1];

        // Host-side path selection (no in-kernel path branching)
        update_kernel_path_flag_host_only();
#if 0
        util::logroot << "[EpsilonDivDiv] tile size (x,y,r)=(" << lat_tile_ << "," << lat_tile_ << "," << r_tile_
                      << "), r_passes=" << r_passes_ << std::endl;
        util::logroot << "[EpsilonDivDiv] number of tiles (x,y,r)=(" << lat_tiles_ << "," << lat_tiles_ << ","
                      << r_tiles_ << "), team_size=" << team_size_ << ", blocks=" << blocks_ << std::endl;
        const char* path_name =
            ( kernel_path_ == KernelPath::Slow )                     ? "slow" :
            ( kernel_path_ == KernelPath::FastFreeslip )             ? "fast-freeslip" :
#ifdef __HIP_PLATFORM_AMD__
            ( kernel_path_ == KernelPath::FastDirichletNeumannWave ) ? "fast-dirichlet-neumann-wave" :
            ( kernel_path_ == KernelPath::FastDirichletNeumannHex )  ? "fast-dirichlet-neumann-hex" :
#endif
                                                                       "fast-dirichlet-neumann";
        util::logroot << "[EpsilonDivDiv] kernel path = " << path_name << std::endl;
#endif
        init_penalty();
    }

    void set_penalty_epsilon( ScalarT eps ) { penalty_epsilon_ = eps; }

    /**
     * @brief Initialize penalty for fs/fs null-space regularization.
     *
     * Builds 3 orthonormal rotation-mode vectors (ê_i × r) with free-slip
     * enforcement, then Gram-Schmidt orthonormalizes them.
     * Only called when both CMB and SURFACE have FREESLIP BCs.
     */
    void init_penalty()
    {
        const BoundaryConditionFlag cmb_bc     = get_boundary_condition_flag( bcs_, CMB );
        const BoundaryConditionFlag surface_bc = get_boundary_condition_flag( bcs_, SURFACE );

        penalty_active_ = ( cmb_bc == FREESLIP ) && ( surface_bc == FREESLIP );
        if ( !penalty_active_ )
            return;

        util::logroot << "[EpsilonDivDiv] Initializing fs/fs null-space penalty (epsilon=" << penalty_epsilon_ << ")"
                      << std::endl;

        ownership_mask_ = grid::setup_node_ownership_mask_data( domain_ );

        const int nsub = domain_.subdomains().size();
        const int nlat = domain_.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nrad = domain_.domain_info().subdomain_num_nodes_radially();

        auto grid_local  = grid_;
        auto radii_local = radii_;
        auto mask_local  = mask_;

        // Build 3 rotation modes: axis i => ê_i × r
        for ( int axis = 0; axis < 3; ++axis )
        {
            null_modes_[axis] = grid::Grid4DDataVec< ScalarType, VecDim >(
                "penalty_null_mode_" + std::to_string( axis ), nsub, nlat, nlat, nrad );

            auto nm = null_modes_[axis];

            Kokkos::parallel_for(
                "penalty_fill_mode_" + std::to_string( axis ),
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { nsub, nlat, nlat, nrad } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                    const auto c = grid::shell::coords( s, x, y, r, grid_local, radii_local );
                    // ê_axis × r: cyclic cross product
                    // axis=0: (0, z, -y), axis=1: (-z, 0, x), axis=2: (y, -x, 0)
                    const int a1 = ( axis + 1 ) % 3;
                    const int a2 = ( axis + 2 ) % 3;
                    for ( int d = 0; d < VecDim; ++d )
                        nm( s, x, y, r, d ) = 0.0;
                    nm( s, x, y, r, a1 ) = c( a2 );
                    nm( s, x, y, r, a2 ) = -c( a1 );
                } );
            Kokkos::fence();

            // Free-slip enforce: transform to n-t, zero normal, transform back
            // Do this for both CMB and SURFACE boundaries
            auto freeslip_boundary_mask = static_cast< grid::shell::ShellBoundaryFlag >(
                static_cast< uint8_t >( CMB ) | static_cast< uint8_t >( SURFACE ) );

            Kokkos::parallel_for(
                "penalty_fs_enforce_" + std::to_string( axis ),
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { nsub, nlat, nlat, nrad } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                    if ( !util::has_flag( mask_local( s, x, y, r ), freeslip_boundary_mask ) )
                        return;

                    dense::Vec< ScalarType, 3 > v;
                    for ( int d = 0; d < 3; ++d )
                        v( d ) = nm( s, x, y, r, d );

                    dense::Vec< ScalarType, 3 > normal;
                    for ( int d = 0; d < 3; ++d )
                        normal( d ) = grid_local( s, x, y, d );

                    // R * v -> normal-tangential
                    const auto R  = trafo_mat_cartesian_to_normal_tangential( normal );
                    auto       nt = R * v;

                    // Zero normal component
                    nt( 0 ) = 0.0;

                    // R^T * nt -> back to Cartesian
                    const auto Rt   = R.transposed();
                    const auto cart = Rt * nt;

                    for ( int d = 0; d < 3; ++d )
                        nm( s, x, y, r, d ) = cart( d );
                } );
            Kokkos::fence();
        }

        // Gram-Schmidt orthonormalization
        for ( int i = 0; i < 3; ++i )
        {
            // Subtract projections of previous modes
            for ( int j = 0; j < i; ++j )
            {
                ScalarType dot_ij = kernels::common::masked_dot_product(
                    null_modes_[i], null_modes_[j], ownership_mask_, grid::NodeOwnershipFlag::OWNED );

                auto ni = null_modes_[i];
                auto nj = null_modes_[j];
                Kokkos::parallel_for(
                    "penalty_gs_subtract",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { nsub, nlat, nlat, nrad } ),
                    KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                        for ( int d = 0; d < VecDim; ++d )
                            ni( s, x, y, r, d ) -= dot_ij * nj( s, x, y, r, d );
                    } );
                Kokkos::fence();
            }

            // Normalize
            ScalarType norm_sq = kernels::common::masked_dot_product(
                null_modes_[i], null_modes_[i], ownership_mask_, grid::NodeOwnershipFlag::OWNED );
            ScalarType inv_norm = 1.0 / Kokkos::sqrt( norm_sq );

            auto ni = null_modes_[i];
            Kokkos::parallel_for(
                "penalty_gs_normalize",
                Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >( { 0, 0, 0, 0 }, { nsub, nlat, nlat, nrad } ),
                KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                    for ( int d = 0; d < VecDim; ++d )
                        ni( s, x, y, r, d ) *= inv_norm;
                } );
            Kokkos::fence();
        }

        util::logroot << "[EpsilonDivDiv] Null-space penalty initialized (3 modes)" << std::endl;
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void set_diagonal( bool v ) { diagonal_ = v; }

    /// Toggle per-cell coefficient homogenization (mean kind via set_eta_homogenization_mean).
    void set_homogenize_eta_per_cell( bool v ) { homogenize_eta_per_cell_ = v; }
    /// Set the per-cell mean kind: 1=harmonic-of-eta, 2=arithmetic-of-eta, 3=geometric-of-eta.
    void set_eta_homogenization_mean( int m ) { eta_homogenization_mean_ = m; }
    /// Configure the operator to read a per-cell-constant coefficient from a
    /// Q0 cell-centered viscosity field at the operator's own MG level
    /// (direct index lookup, no translation).  Caller must build a Q0 field
    /// sized for this operator's mesh; for a velocity-level operator that
    /// means broadcasting the pressure-level Q0 into a velocity-level Q0 so
    /// each pressure cell's 8 velocity sub-cells share the value.
    void set_q0_coefficient_field( const grid::Grid4DDataScalar< ScalarType >& k_q0 )
    {
        k_q0_              = k_q0;
        use_q0_coefficient_ = true;
    }
    void clear_q0_coefficient_field() { use_q0_coefficient_ = false; }

    void set_boundary_conditions( BoundaryConditions bcs )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];

        // Recompute host-side dispatch mode whenever BCs change
        update_kernel_path_flag_host_only();
    }

    /// @brief Override the kernel path. Bypasses the host-side rules in
    /// update_kernel_path_flag_host_only(); will be re-derived if BCs or
    /// stored-matrix mode change.
    void set_kernel_path( KernelPath p ) { kernel_path_ = p; }
    KernelPath kernel_path() const { return kernel_path_; }

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
    { return util::has_flag( mask_( local_subdomain_id, x_cell, y_cell, r_cell ), flag ); }

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
                      << ( ( kernel_path_ == KernelPath::Slow )                     ? "slow" :
                           ( kernel_path_ == KernelPath::FastFreeslip )             ? "fast-freeslip" :
#ifdef __HIP_PLATFORM_AMD__
                           ( kernel_path_ == KernelPath::FastDirichletNeumannWave ) ? "fast-dirichlet-neumann-wave" :
                           ( kernel_path_ == KernelPath::FastDirichletNeumannHex )  ? "fast-dirichlet-neumann-hex" :
#endif
                                                                                      "fast-dirichlet-neumann" )
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
            util::Timer timer_zero( "epsilon_divdiv_zero" );
            assign( dst, 0 );
            Kokkos::fence();
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
            if ( diagonal_ )
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_fs_diag", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_freeslip< true >( team );
                    } );
            }
            else
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_fs_matvec", policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_freeslip< false >( team );
                    } );
            }
        }
#ifdef __HIP_PLATFORM_AMD__
        else if ( kernel_path_ == KernelPath::FastDirichletNeumannWave )
        {
            // Wave-parallel DN path: each wave (60 of 64 lanes active) processes
            // 10 radial cells via team_size=10 threads × vector_length=6 vector
            // lanes (one lane per active wedge node).
            const int r_stacks = ( hex_rad_ + kWaveCellsPerWave - 1 ) / kWaveCellsPerWave;
            const int wave_blocks =
                local_subdomains_ * ( hex_lat_ ) * ( hex_lat_ ) * r_stacks;
            using LB = Kokkos::LaunchBounds< 64, 1 >;
            Kokkos::TeamPolicy< LB > wv_policy(
                wave_blocks, /*team_size=*/10, /*vector_length=*/6 );
            wv_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size_dn_wave() ) );
            if ( diagonal_ )
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_dn_wave_diag", wv_policy,
                    KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_dirichlet_neumann_wave< true >( team );
                    } );
            }
            else
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_dn_wave_matvec", wv_policy,
                    KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_dirichlet_neumann_wave< false >( team );
                    } );
            }
        }
        else if ( kernel_path_ == KernelPath::FastDirichletNeumannHex )
        {
            // Hex 1-pt Gauss DN path: each wave (64 lanes) processes 16 radial cells
            // (2 blocks × 8 cells, 8 lanes per hex). league_size: one team per
            // (subdomain, x_cell, y_cell, r_stack).
            const int r_stacks = ( hex_rad_ + kHexCellsPerWave - 1 ) / kHexCellsPerWave;
            const int hex_blocks =
                local_subdomains_ * ( hex_lat_ ) * ( hex_lat_ ) * r_stacks;
            using LB = Kokkos::LaunchBounds< 64, 1 >;
            Kokkos::TeamPolicy< LB > hx_policy(
                hex_blocks, /*team_size=*/1, /*vector_length=*/64 );
            hx_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size_dn_hex() ) );
            if ( diagonal_ )
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_dn_hex_diag", hx_policy,
                    KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_dirichlet_neumann_hex< true >( team );
                    } );
            }
            else
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_dn_hex_matvec", hx_policy,
                    KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_dirichlet_neumann_hex< false >( team );
                    } );
            }
        }
#endif
        else
        {
            // A/B test: full revert to cross-branch dispatch (LB<512,2>, no team doubling).
            // Paired with run_team_fast_dirichlet_neumann calling operator_fast_dirichlet_neumann_path
            // with w_filter=-1 so each thread sequentially computes both wedges (compile-time loop).
            Kokkos::TeamPolicy< Kokkos::LaunchBounds< 512, 2 > > dn_policy( blocks_, team_size_ );
            dn_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size_dn( team_size_ ) ) );
            if ( diagonal_ )
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_dn_diag", dn_policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_dirichlet_neumann< true >( team );
                    } );
            }
            else
            {
                Kokkos::parallel_for(
                    "epsilon_divdiv_apply_kernel_fast_dn_matvec", dn_policy, KOKKOS_CLASS_LAMBDA( const Team& team ) {
                        this->template run_team_fast_dirichlet_neumann< false >( team );
                    } );
            }
        }

        Kokkos::fence();
        timer_kernel.stop();

        // Null-space penalty for fs/fs: dst += epsilon * sum_i (n_i^T src) n_i.
        //
        // Fused implementation:
        //  - 1 GPU reduction kernel computing all 3 dot products (vs 3x3=9 before)
        //  - 1 MPI_Allreduce of length 3 (vs 9 separate 1-element reductions before)
        //  - 1 fused AXPY pass over dst (vs 3 separate passes before)
        if ( penalty_active_ && !diagonal_ )
        {
            util::Timer timer_penalty( "epsilon_divdiv_penalty" );

            ScalarType alpha0 = 0, alpha1 = 0, alpha2 = 0;
            {
                auto       src_local = src_;
                auto       nm0       = null_modes_[0];
                auto       nm1       = null_modes_[1];
                auto       nm2       = null_modes_[2];
                auto       mask      = ownership_mask_;
                const auto mask_val_owned = grid::NodeOwnershipFlag::OWNED;
                Kokkos::parallel_reduce(
                    "epsilon_divdiv_penalty_dots",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                        { 0, 0, 0, 0 },
                        { src_local.extent( 0 ),
                          src_local.extent( 1 ),
                          src_local.extent( 2 ),
                          src_local.extent( 3 ) } ),
                    KOKKOS_LAMBDA(
                        int s, int x, int y, int r, ScalarType& a0, ScalarType& a1, ScalarType& a2 ) {
                        const ScalarType m =
                            util::has_flag( mask( s, x, y, r ), mask_val_owned ) ? ScalarType( 1 ) : ScalarType( 0 );
                        ScalarType s0 = 0, s1 = 0, s2 = 0;
                        for ( int d = 0; d < VecDim; ++d )
                        {
                            const ScalarType sv = src_local( s, x, y, r, d ) * m;
                            s0 += nm0( s, x, y, r, d ) * sv;
                            s1 += nm1( s, x, y, r, d ) * sv;
                            s2 += nm2( s, x, y, r, d ) * sv;
                        }
                        a0 += s0;
                        a1 += s1;
                        a2 += s2;
                    },
                    Kokkos::Sum< ScalarType >( alpha0 ),
                    Kokkos::Sum< ScalarType >( alpha1 ),
                    Kokkos::Sum< ScalarType >( alpha2 ) );
                Kokkos::fence( "epsilon_divdiv_penalty_dots" );
            }

            ScalarType partial[3] = { alpha0, alpha1, alpha2 };
            MPI_Allreduce( MPI_IN_PLACE, partial, 3, mpi::mpi_datatype< ScalarType >(), MPI_SUM, MPI_COMM_WORLD );
            const ScalarType a0 = penalty_epsilon_ * partial[0];
            const ScalarType a1 = penalty_epsilon_ * partial[1];
            const ScalarType a2 = penalty_epsilon_ * partial[2];

            {
                auto dst_local = dst_;
                auto nm0       = null_modes_[0];
                auto nm1       = null_modes_[1];
                auto nm2       = null_modes_[2];
                Kokkos::parallel_for(
                    "epsilon_divdiv_penalty_axpy_fused",
                    Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                        { 0, 0, 0, 0 },
                        { dst_local.extent( 0 ),
                          dst_local.extent( 1 ),
                          dst_local.extent( 2 ),
                          dst_local.extent( 3 ) } ),
                    KOKKOS_LAMBDA( int s, int x, int y, int r ) {
                        for ( int d = 0; d < VecDim; ++d )
                            dst_local( s, x, y, r, d ) += a0 * nm0( s, x, y, r, d ) + a1 * nm1( s, x, y, r, d ) +
                                                          a2 * nm2( s, x, y, r, d );
                    } );
            }
            Kokkos::fence( "penalty" );
        }

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
     *   [coords_sh | src_sh | k_sh | r_sh]
     */
    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size( const int ts ) const
    {
        const int nlev = r_tile_block_ + 1;
        const int n    = lat_tile_ + 1;
        const int nxy  = n * n;

        // coords_sh(nxy,3) + normals_sh(nxy,3) + src_sh(nxy,3,nlev) + k_sh(nxy,nlev) + r_sh(nlev) + 1
        const size_t nscalars = size_t( nxy ) * 3 + size_t( nxy ) * 3 + size_t( nxy ) * 3 * nlev +
                                size_t( nxy ) * nlev + size_t( nlev ) + 1;

        // The team scratch tiles (coords_sh/src_sh/k_sh/r_sh) are stored in DOUBLE
        // regardless of ScalarType, so the byte size must use sizeof(double). Using
        // sizeof(ScalarType) under-allocates by 2x in a single-precision build, so the
        // double tiles overrun/overlap and the operator apply reads garbage -> NaN.
        return sizeof( double ) * nscalars;
    }

    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size_dn( const int /* ts */ ) const
    {
        const int nlev = r_tile_block_ + 1;
        const int n    = lat_tile_ + 1;
        const int nxy  = n * n;

        // coords_sh(nxy,3) + src_sh(nxy,3,nlev) + k_sh(nxy,nlev) + r_sh(nlev)
        const size_t nscalars = size_t( nxy ) * 3 + size_t( nxy ) * 3 * nlev + size_t( nxy ) * nlev + size_t( nlev );

        // The team scratch tiles (coords_sh/src_sh/k_sh/r_sh) are stored in DOUBLE
        // regardless of ScalarType, so the byte size must use sizeof(double). Using
        // sizeof(ScalarType) under-allocates by 2x in a single-precision build, so the
        // double tiles overrun/overlap and the operator apply reads garbage -> NaN.
        return sizeof( double ) * nscalars;
    }

    // HIP/AMD-characteristic experimental DN paths: wave-parallel (60-of-64
    // active wave-lane layout) and hex 1-pt Gauss (uses __shfl_xor). Kept in
    // their own subdirectory and gated on __HIP_PLATFORM_AMD__ so the CUDA
    // backend does not try to compile them.
#ifdef __HIP_PLATFORM_AMD__
#include "hip/epsilon_divdiv_kerngen_wave.hpp"
#include "hip/epsilon_divdiv_hex.hpp"
#endif

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
        r0 = r_tile_id * r_tile_block_;

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

        for ( int pass = 0; pass < r_passes_; ++pass )
        {
            const int r_cell_pass = r0 + pass * r_tile_ + tr;
            if ( r_cell_pass >= hex_rad_ )
                break;

            const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_pass, CMB );
            const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_pass + 1, SURFACE );

            operator_slow_path(
                team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell_pass, at_cmb, at_surface );
        }
    }

    /**
     * @brief Team entry for fast Dirichlet/Neumann matrix-free path.
     *
     * Templated on Diagonal so the compiler can dead-code-eliminate the
     * unused matvec or diagonal-only path, reducing register pressure.
     */
    template < bool Diagonal >
    KOKKOS_INLINE_FUNCTION void run_team_fast_dirichlet_neumann( const Team& team ) const
    {
        // A/B test: legacy thread-per-cell layout. Use decode_team_indices like
        // the cross-branch version and pass w_filter=-1 so the inner kernel runs
        // both wedges in the compile-time `for (w=0;w<2;++w)` loop.
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        operator_fast_dirichlet_neumann_path< Diagonal >(
            team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, /*w_filter=*/-1 );
    }


    /**
     * @brief Team entry for fast free-slip matrix-free path.
     *
     * Templated on Diagonal so the compiler can dead-code-eliminate the
     * unused matvec or diagonal-only path, reducing register pressure.
     */
    template < bool Diagonal >
    KOKKOS_INLINE_FUNCTION void run_team_fast_freeslip( const Team& team ) const
    {
        int local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell;
        decode_team_indices( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell );

        if ( tr >= r_tile_ )
            return;

        operator_fast_freeslip_path< Diagonal >( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
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

        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A[num_wedges_per_hex_cell] = {};

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
                dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A_tmp[num_wedges_per_hex_cell] = {};

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
                        dense::Vec< ScalarType, 3 > normal = grid::shell::coords(
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
    template < bool Diagonal >
    KOKKOS_INLINE_FUNCTION void operator_fast_dirichlet_neumann_path(
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
        const int   w_filter = -1 ) const
    {
        const int nlev = r_tile_block_ + 1;
        const int nxy  = ( lat_tile_ + 1 ) * ( lat_tile_ + 1 );

        double* shmem =
            reinterpret_cast< double* >( team.team_shmem().get_shmem( team_shmem_size_dn( team.team_size() ) ) );

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

        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ )
            return;

        constexpr double ONE_THIRD = 1.0 / 3.0;
        constexpr double ONE_SIXTH = 1.0 / 6.0;

        static constexpr double dN_ref[6][3] = {
            { -0.5, -0.5, -ONE_SIXTH },
            { 0.5, 0.0, -ONE_SIXTH },
            { 0.0, 0.5, -ONE_SIXTH },
            { -0.5, -0.5, ONE_SIXTH },
            { 0.5, 0.0, ONE_SIXTH },
            { 0.0, 0.5, ONE_SIXTH } };

        static constexpr int WEDGE_NODE_OFF[2][6][3] = {
            { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
            { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

        const int n00 = node_id( tx, ty );
        const int n01 = node_id( tx, ty + 1 );
        const int n10 = node_id( tx + 1, ty );
        const int n11 = node_id( tx + 1, ty + 1 );

        for ( int pass = 0; pass < r_passes_; ++pass )
        {
            const int lvl0   = pass * r_tile_ + tr;
            const int r_cell = r0 + lvl0;

            if ( r_cell >= hex_rad_ )
                break;

            const double r_0 = r_sh( lvl0 );
            const double r_1 = r_sh( lvl0 + 1 );

            const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
            const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

            const bool at_boundary              = at_cmb || at_surface;
            bool       treat_boundary_dirichlet = false;
            if ( at_boundary )
            {
                const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
                treat_boundary_dirichlet    = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
            }

            const int cmb_shift = ( ( at_boundary && treat_boundary_dirichlet && ( !Diagonal ) && at_cmb ) ? 3 : 0 );
            const int surface_shift =
                ( ( at_boundary && treat_boundary_dirichlet && ( !Diagonal ) && at_surface ) ? 3 : 0 );

            // w_filter == -1 → run both wedges (legacy behaviour);
            // w_filter == 0 or 1 → run only that wedge (thread-per-wedge layout).
            const int w_start = ( w_filter < 0 ) ? 0 : w_filter;
            const int w_end   = ( w_filter < 0 ) ? 2 : w_filter + 1;
            for ( int w = w_start; w < w_end; ++w )
            {
                const int v0 = w == 0 ? n00 : n11;
                const int v1 = w == 0 ? n10 : n01;
                const int v2 = w == 0 ? n01 : n10;

                double k_eval;
                if ( use_q0_coefficient_ )
                {
                    // Q0 cell-centered eta on the pressure mesh; parent of
                    // velocity-cell (x_cell, y_cell, r_cell) is half-index.
                    k_eval = k_q0_( local_subdomain_id, x_cell, y_cell, r_cell );
                }
                else
                {
                    double k_sum     = 0.0;
                    double inv_k_sum = 0.0;
                    double log_k_sum = 0.0;
#pragma unroll
                    for ( int node = 0; node < 6; ++node )
                    {
                        const int nid = node_id( tx + WEDGE_NODE_OFF[w][node][0], ty + WEDGE_NODE_OFF[w][node][1] );
                        const double k_corner = k_sh( nid, lvl0 + WEDGE_NODE_OFF[w][node][2] );
                        k_sum += k_corner;
                        inv_k_sum += 1.0 / k_corner;
                        log_k_sum += Kokkos::log( k_corner );
                    }
                    if ( !homogenize_eta_per_cell_ )
                        k_eval = ONE_SIXTH * k_sum;
                    else if ( eta_homogenization_mean_ == 2 )
                        k_eval = ONE_SIXTH * k_sum;
                    else if ( eta_homogenization_mean_ == 3 )
                        k_eval = Kokkos::exp( log_k_sum * ONE_SIXTH );
                    else
                        k_eval = 6.0 / inv_k_sum;
                }

                double kwJ;

                // ==== Phase 1: Jacobian + Gather (gu tensor) ====
                // invJ lives only in this scope so the compiler can reclaim its registers.
                double gu00 = 0.0;
                double gu10 = 0.0, gu11 = 0.0;
                double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
                double div_u = 0.0;
                {
                    const double half_dr = 0.5 * ( r_1 - r_0 );
                    const double r_mid   = 0.5 * ( r_0 + r_1 );

                    const double J_0_0 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v1, 0 ) );
                    const double J_0_1 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v2, 0 ) );
                    const double J_0_2 =
                        half_dr * ( ONE_THIRD * ( coords_sh( v0, 0 ) + coords_sh( v1, 0 ) + coords_sh( v2, 0 ) ) );

                    const double J_1_0 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v1, 1 ) );
                    const double J_1_1 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v2, 1 ) );
                    const double J_1_2 =
                        half_dr * ( ONE_THIRD * ( coords_sh( v0, 1 ) + coords_sh( v1, 1 ) + coords_sh( v2, 1 ) ) );

                    const double J_2_0 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v1, 2 ) );
                    const double J_2_1 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v2, 2 ) );
                    const double J_2_2 =
                        half_dr * ( ONE_THIRD * ( coords_sh( v0, 2 ) + coords_sh( v1, 2 ) + coords_sh( v2, 2 ) ) );

                    const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                         J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;

                    kwJ = k_eval * Kokkos::abs( J_det );

                    const double inv_det = 1.0 / J_det;

                    const double i00 = inv_det * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                    const double i01 = inv_det * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                    const double i02 = inv_det * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );
                    const double i10 = inv_det * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                    const double i11 = inv_det * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                    const double i12 = inv_det * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
                    const double i20 = inv_det * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                    const double i21 = inv_det * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                    const double i22 = inv_det * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

                    if ( !Diagonal )
                    {
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

                            gu00 += g0 * s0;
                            gu11 += g1 * s1;
                            gu22 += g2 * s2;
                            gu10 += 0.5 * ( g1 * s0 + g0 * s1 );
                            gu20 += 0.5 * ( g2 * s0 + g0 * s2 );
                            gu21 += 0.5 * ( g2 * s1 + g1 * s2 );
                            div_u += g0 * s0 + g1 * s1 + g2 * s2;
                        }
                    }
                }
                // invJ (i00..i22) is now out of scope — registers can be reclaimed.

                // ==== Phase 2: Recompute Jacobian + Scatter ====
                {
                    const double half_dr = 0.5 * ( r_1 - r_0 );
                    const double r_mid   = 0.5 * ( r_0 + r_1 );

                    const double J_0_0 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v1, 0 ) );
                    const double J_0_1 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v2, 0 ) );
                    const double J_0_2 =
                        half_dr * ( ONE_THIRD * ( coords_sh( v0, 0 ) + coords_sh( v1, 0 ) + coords_sh( v2, 0 ) ) );

                    const double J_1_0 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v1, 1 ) );
                    const double J_1_1 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v2, 1 ) );
                    const double J_1_2 =
                        half_dr * ( ONE_THIRD * ( coords_sh( v0, 1 ) + coords_sh( v1, 1 ) + coords_sh( v2, 1 ) ) );

                    const double J_2_0 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v1, 2 ) );
                    const double J_2_1 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v2, 2 ) );
                    const double J_2_2 =
                        half_dr * ( ONE_THIRD * ( coords_sh( v0, 2 ) + coords_sh( v1, 2 ) + coords_sh( v2, 2 ) ) );

                    const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                         J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;

                    const double inv_det = 1.0 / J_det;

                    const double i00 = inv_det * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                    const double i01 = inv_det * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                    const double i02 = inv_det * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );
                    const double i10 = inv_det * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                    const double i11 = inv_det * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                    const double i12 = inv_det * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
                    const double i20 = inv_det * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                    const double i21 = inv_det * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                    const double i22 = inv_det * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );

                    if ( !Diagonal )
                    {
                        constexpr double NEG_TWO_THIRDS = -0.66666666666666663;
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
                            Kokkos::atomic_add(
                                &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 0 ),
                                kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 ) + NEG_TWO_THIRDS * g0 * div_u ) );
                            Kokkos::atomic_add(
                                &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 1 ),
                                kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 ) + NEG_TWO_THIRDS * g1 * div_u ) );
                            Kokkos::atomic_add(
                                &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 2 ),
                                kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 ) + NEG_TWO_THIRDS * g2 * div_u ) );
                        }
                    }

                    if ( Diagonal || ( treat_boundary_dirichlet && at_boundary ) )
                    {
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

                            const int nid = node_id( tx + WEDGE_NODE_OFF[w][n][0], ty + WEDGE_NODE_OFF[w][n][1] );
                            const int lvl = lvl0 + WEDGE_NODE_OFF[w][n][2];

                            const double sv0 = src_sh( nid, 0, lvl );
                            const double sv1 = src_sh( nid, 1, lvl );
                            const double sv2 = src_sh( nid, 2, lvl );

                            const int ddx = WEDGE_NODE_OFF[w][n][0];
                            const int ddy = WEDGE_NODE_OFF[w][n][1];
                            const int ddr = WEDGE_NODE_OFF[w][n][2];
                            Kokkos::atomic_add(
                                &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 0 ),
                                kwJ * sv0 * ( gg + ONE_THIRD * g0 * g0 ) );
                            Kokkos::atomic_add(
                                &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 1 ),
                                kwJ * sv1 * ( gg + ONE_THIRD * g1 * g1 ) );
                            Kokkos::atomic_add(
                                &dst_( local_subdomain_id, x_cell + ddx, y_cell + ddy, r_cell + ddr, 2 ),
                                kwJ * sv2 * ( gg + ONE_THIRD * g2 * g2 ) );
                        }
                    }
                }

            } // end wedge loop
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

    template < bool Diagonal >
    KOKKOS_INLINE_FUNCTION void operator_fast_freeslip_path(
        const Team& team,
        const int   local_subdomain_id,
        const int   x0,
        const int   y0,
        const int   r0,
        const int   tx,
        const int   ty,
        const int   tr,
        const int   x_cell,
        const int   y_cell ) const
    {
        const int nlev = r_tile_block_ + 1;
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

        // Normalized normals in shared memory — avoids 12 persistent register-doubles.
        ScratchCoords normals_sh( shmem, nxy, 3 );
        shmem += nxy * 3;

        ScratchSrc src_sh( shmem, nxy, 3, nlev );
        shmem += nxy * 3 * nlev;

        ScratchK k_sh( shmem, nxy, nlev );
        shmem += nxy * nlev;

        auto r_sh =
            Kokkos::View< double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >(
                shmem, nlev );

        auto node_id = [&]( int nx, int ny ) -> int { return nx + ( lat_tile_ + 1 ) * ny; };

        // Preload coords + compute normalized normals cooperatively.
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy ), [&]( int n ) {
            const int dxn = n % ( lat_tile_ + 1 );
            const int dyn = n / ( lat_tile_ + 1 );
            const int xi  = x0 + dxn;
            const int yi  = y0 + dyn;

            if ( xi <= hex_lat_ && yi <= hex_lat_ )
            {
                const double cx   = grid_( local_subdomain_id, xi, yi, 0 );
                const double cy   = grid_( local_subdomain_id, xi, yi, 1 );
                const double cz   = grid_( local_subdomain_id, xi, yi, 2 );
                coords_sh( n, 0 ) = cx;
                coords_sh( n, 1 ) = cy;
                coords_sh( n, 2 ) = cz;

                const double n2 = cx * cx + cy * cy + cz * cz;
                if ( n2 > 0.0 )
                {
                    const double invn  = 1.0 / Kokkos::sqrt( n2 );
                    normals_sh( n, 0 ) = cx * invn;
                    normals_sh( n, 1 ) = cy * invn;
                    normals_sh( n, 2 ) = cz * invn;
                }
                else
                {
                    normals_sh( n, 0 ) = normals_sh( n, 1 ) = normals_sh( n, 2 ) = 0.0;
                }
            }
            else
            {
                coords_sh( n, 0 ) = coords_sh( n, 1 ) = coords_sh( n, 2 ) = 0.0;
                normals_sh( n, 0 ) = normals_sh( n, 1 ) = normals_sh( n, 2 ) = 0.0;
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

        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ )
            return;

        for ( int pass = 0; pass < r_passes_; ++pass )
        {
            const int lvl0   = pass * r_tile_ + tr;
            const int r_cell = r0 + lvl0;

            if ( r_cell >= hex_rad_ )
                break;

            const double r_0 = r_sh( lvl0 );
            const double r_1 = r_sh( lvl0 + 1 );

            const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
            const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

            const BoundaryConditionFlag cmb_bc     = get_boundary_condition_flag( bcs_, CMB );
            const BoundaryConditionFlag surface_bc = get_boundary_condition_flag( bcs_, SURFACE );

            const bool cmb_freeslip      = at_cmb && ( cmb_bc == FREESLIP );
            const bool surf_freeslip     = at_surface && ( surface_bc == FREESLIP );
            const bool cmb_dirichlet     = at_cmb && ( cmb_bc == DIRICHLET );
            const bool surface_dirichlet = at_surface && ( surface_bc == DIRICHLET );

            const int cmb_shift     = ( ( cmb_dirichlet && ( !Diagonal ) ) ? 3 : 0 );
            const int surface_shift = ( ( surface_dirichlet && ( !Diagonal ) ) ? 3 : 0 );

            static constexpr int WEDGE_NODE_OFF[2][6][3] = {
                { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
                { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

            static constexpr int WEDGE_TO_UNIQUE[2][6] = { { 0, 1, 2, 3, 4, 5 }, { 6, 2, 1, 7, 5, 4 } };

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

            // Corner-to-shared-memory-node mapping: 0→n00, 1→n10, 2→n01, 3→n11.
            const int corner_node[4] = { n00, n10, n01, n11 };
            // Corner-to-unique-node for CMB (r=0) and surface (r=1) layers.
            static constexpr int CMB_CORNER_TO_UNIQUE[4]  = { 0, 1, 2, 6 };
            static constexpr int SURF_CORNER_TO_UNIQUE[4] = { 3, 4, 5, 7 };

            double dst8[3][8] = { { 0.0 } };

            // Scalar Ann accumulators per corner.  Accumulated inside the test-side loop
            // (merged from the former separate Ann loop to reuse already-computed gradients).
            double Ann_acc_cmb[4]  = {};
            double Ann_acc_surf[4] = {};

            for ( int w = 0; w < 2; ++w )
            {
                const int v0 = w == 0 ? n00 : n11;
                const int v1 = w == 0 ? n10 : n01;
                const int v2 = w == 0 ? n01 : n10;

                double k_eval;
                if ( use_q0_coefficient_ )
                {
                    k_eval = k_q0_( local_subdomain_id, x_cell, y_cell, r_cell );
                }
                else
                {
                    double k_sum     = 0.0;
                    double inv_k_sum = 0.0;
                    double log_k_sum = 0.0;
#pragma unroll
                    for ( int node = 0; node < 6; ++node )
                    {
                        const int ddx = WEDGE_NODE_OFF[w][node][0];
                        const int ddy = WEDGE_NODE_OFF[w][node][1];
                        const int ddr = WEDGE_NODE_OFF[w][node][2];

                        const int    nid      = node_id( tx + ddx, ty + ddy );
                        const int    lvl      = lvl0 + ddr;
                        const double k_corner = k_sh( nid, lvl );
                        k_sum += k_corner;
                        inv_k_sum += 1.0 / k_corner;
                        log_k_sum += Kokkos::log( k_corner );
                    }
                    if ( !homogenize_eta_per_cell_ )
                        k_eval = ONE_SIXTH * k_sum;
                    else if ( eta_homogenization_mean_ == 2 )
                        k_eval = ONE_SIXTH * k_sum;
                    else if ( eta_homogenization_mean_ == 3 )
                        k_eval = Kokkos::exp( log_k_sum * ONE_SIXTH );
                    else
                        k_eval = 6.0 / inv_k_sum;
                }

                double wJ = 0.0;
                double i00, i01, i02;
                double i10, i11, i12;
                double i20, i21, i22;

                {
                    const double half_dr = 0.5 * ( r_1 - r_0 );
                    const double r_mid   = 0.5 * ( r_0 + r_1 );

                    const double J_0_0 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v1, 0 ) );
                    const double J_0_1 = r_mid * ( -coords_sh( v0, 0 ) + coords_sh( v2, 0 ) );
                    const double J_0_2 =
                        half_dr * ( ONE_THIRD * ( coords_sh( v0, 0 ) + coords_sh( v1, 0 ) + coords_sh( v2, 0 ) ) );

                    const double J_1_0 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v1, 1 ) );
                    const double J_1_1 = r_mid * ( -coords_sh( v0, 1 ) + coords_sh( v2, 1 ) );
                    const double J_1_2 =
                        half_dr * ( ONE_THIRD * ( coords_sh( v0, 1 ) + coords_sh( v1, 1 ) + coords_sh( v2, 1 ) ) );

                    const double J_2_0 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v1, 2 ) );
                    const double J_2_1 = r_mid * ( -coords_sh( v0, 2 ) + coords_sh( v2, 2 ) );
                    const double J_2_2 =
                        half_dr * ( ONE_THIRD * ( coords_sh( v0, 2 ) + coords_sh( v1, 2 ) + coords_sh( v2, 2 ) ) );

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

                // ---- Fused trial + test side with merged Ann accumulation ----
                static constexpr int CMB_NODE_TO_CORNER[2][3] = { { 0, 1, 2 }, { 3, 2, 1 } };

                double gu00 = 0.0;
                double gu10 = 0.0, gu11 = 0.0;
                double gu20 = 0.0, gu21 = 0.0, gu22 = 0.0;
                double div_u = 0.0;

                if ( !Diagonal )
                {
                    // Trial side: accumulate symmetric gradient of u (fused dim loops).
                    // Read directly from shared memory with inline tangential projection for freeslip nodes.
#pragma unroll
                    for ( int n = cmb_shift; n < 6 - surface_shift; ++n )
                    {
                        const double gx = dN_ref[n][0];
                        const double gy = dN_ref[n][1];
                        const double gz = dN_ref[n][2];
                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;

                        const int nid = node_id( tx + WEDGE_NODE_OFF[w][n][0], ty + WEDGE_NODE_OFF[w][n][1] );
                        const int lvl = lvl0 + WEDGE_NODE_OFF[w][n][2];

                        double s0 = src_sh( nid, 0, lvl );
                        double s1 = src_sh( nid, 1, lvl );
                        double s2 = src_sh( nid, 2, lvl );

                        // Inline tangential projection for freeslip boundary nodes.
                        if ( cmb_freeslip && n < 3 )
                        {
                            const double nx  = normals_sh( nid, 0 );
                            const double ny  = normals_sh( nid, 1 );
                            const double nz  = normals_sh( nid, 2 );
                            const double dot = nx * s0 + ny * s1 + nz * s2;
                            s0 -= dot * nx;
                            s1 -= dot * ny;
                            s2 -= dot * nz;
                        }
                        if ( surf_freeslip && n >= 3 )
                        {
                            const double nx  = normals_sh( nid, 0 );
                            const double ny  = normals_sh( nid, 1 );
                            const double nz  = normals_sh( nid, 2 );
                            const double dot = nx * s0 + ny * s1 + nz * s2;
                            s0 -= dot * nx;
                            s1 -= dot * ny;
                            s2 -= dot * nz;
                        }

                        gu00 += g0 * s0;
                        gu11 += g1 * s1;
                        gu22 += g2 * s2;
                        gu10 += 0.5 * ( g1 * s0 + g0 * s1 );
                        gu20 += 0.5 * ( g2 * s0 + g0 * s2 );
                        gu21 += 0.5 * ( g2 * s1 + g1 * s2 );
                        div_u += g0 * s0 + g1 * s1 + g2 * s2;
                    }

                    // Test side + merged Ann accumulation.
                    // Ann uses the same gradient already computed — no separate loop needed.
#pragma unroll
                    for ( int n = cmb_shift; n < 6 - surface_shift; ++n )
                    {
                        const double gx = dN_ref[n][0];
                        const double gy = dN_ref[n][1];
                        const double gz = dN_ref[n][2];
                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;

                        const int uid = WEDGE_TO_UNIQUE[w][n];
                        dst8[0][uid] +=
                            kwJ * ( 2.0 * ( g0 * gu00 + g1 * gu10 + g2 * gu20 ) + NEG_TWO_THIRDS * g0 * div_u );
                        dst8[1][uid] +=
                            kwJ * ( 2.0 * ( g0 * gu10 + g1 * gu11 + g2 * gu21 ) + NEG_TWO_THIRDS * g1 * div_u );
                        dst8[2][uid] +=
                            kwJ * ( 2.0 * ( g0 * gu20 + g1 * gu21 + g2 * gu22 ) + NEG_TWO_THIRDS * g2 * div_u );

                        // Accumulate Ann for freeslip CMB nodes (n < 3) and surface nodes (n >= 3).
                        if ( cmb_freeslip && n < 3 )
                        {
                            const int    corner = CMB_NODE_TO_CORNER[w][n];
                            const int    cn     = corner_node[corner];
                            const double nxu    = normals_sh( cn, 0 );
                            const double nyu    = normals_sh( cn, 1 );
                            const double nzu    = normals_sh( cn, 2 );
                            const double gg     = g0 * g0 + g1 * g1 + g2 * g2;
                            const double ng     = nxu * g0 + nyu * g1 + nzu * g2;
                            Ann_acc_cmb[corner] += kwJ * ( gg + ONE_THIRD * ng * ng );
                        }
                        if ( surf_freeslip && n >= 3 )
                        {
                            const int    corner = CMB_NODE_TO_CORNER[w][n - 3];
                            const int    cn     = corner_node[corner];
                            const double nxu    = normals_sh( cn, 0 );
                            const double nyu    = normals_sh( cn, 1 );
                            const double nzu    = normals_sh( cn, 2 );
                            const double gg     = g0 * g0 + g1 * g1 + g2 * g2;
                            const double ng     = nxu * g0 + nyu * g1 + nzu * g2;
                            Ann_acc_surf[corner] += kwJ * ( gg + ONE_THIRD * ng * ng );
                        }
                    }
                }

                // ---- Diagonal / Dirichlet boundary handling (fused) ----
                if ( Diagonal || cmb_dirichlet || surface_dirichlet )
                {
                    // Fused diagonal: kwJ * s_d * (|g|^2 + (1/3) * g_d^2)
#pragma unroll
                    for ( int n = surface_shift; n < 6 - cmb_shift; ++n )
                    {
                        if ( Diagonal && cmb_freeslip && n < 3 )
                            continue;
                        if ( Diagonal && surf_freeslip && n >= 3 )
                            continue;

                        const double gx = dN_ref[n][0];
                        const double gy = dN_ref[n][1];
                        const double gz = dN_ref[n][2];
                        const double g0 = i00 * gx + i01 * gy + i02 * gz;
                        const double g1 = i10 * gx + i11 * gy + i12 * gz;
                        const double g2 = i20 * gx + i21 * gy + i22 * gz;
                        const double gg = g0 * g0 + g1 * g1 + g2 * g2;

                        const int    uid = WEDGE_TO_UNIQUE[w][n];
                        const int    nid = node_id( tx + WEDGE_NODE_OFF[w][n][0], ty + WEDGE_NODE_OFF[w][n][1] );
                        const int    lvl = lvl0 + WEDGE_NODE_OFF[w][n][2];
                        const double s0  = src_sh( nid, 0, lvl );
                        const double s1  = src_sh( nid, 1, lvl );
                        const double s2  = src_sh( nid, 2, lvl );

                        dst8[0][uid] += kwJ * s0 * ( gg + ONE_THIRD * g0 * g0 );
                        dst8[1][uid] += kwJ * s1 * ( gg + ONE_THIRD * g1 * g1 );
                        dst8[2][uid] += kwJ * s2 * ( gg + ONE_THIRD * g2 * g2 );
                    }

                    // For free-slip boundary nodes in diagonal mode: compute R^T diag(R A_3x3 R^T) R src.
                    // Normals loaded from shared memory, u_n recomputed from src_sh (original, unprojected).
                    if ( Diagonal )
                    {
                        static constexpr int FS_CORNER_MAP[2][3] = { { 0, 1, 2 }, { 3, 2, 1 } };

                        auto apply_rotated_diag = [&]( const int ni, const int node_idx, const int src_lvl ) {
                            const int    corner = FS_CORNER_MAP[w][ni];
                            const int    cn     = corner_node[corner];
                            const double nxu    = normals_sh( cn, 0 );
                            const double nyu    = normals_sh( cn, 1 );
                            const double nzu    = normals_sh( cn, 2 );
                            const int    u      = WEDGE_TO_UNIQUE[w][node_idx];

                            const double gx = dN_ref[node_idx][0];
                            const double gy = dN_ref[node_idx][1];
                            const double gz = dN_ref[node_idx][2];

                            const double g0     = i00 * gx + i01 * gy + i02 * gz;
                            const double g1     = i10 * gx + i11 * gy + i12 * gz;
                            const double g2     = i20 * gx + i21 * gy + i22 * gz;
                            const double gg_loc = g0 * g0 + g1 * g1 + g2 * g2;

                            dense::Vec< double, 3 > n_vec;
                            n_vec( 0 )       = nxu;
                            n_vec( 1 )       = nyu;
                            n_vec( 2 )       = nzu;
                            const auto R_rot = trafo_mat_cartesian_to_normal_tangential< double >( n_vec );

                            // Read original (unprojected) src directly from shared memory.
                            // (Previously: projected src8 + u_n*n = original, now simplified.)
                            const double s0 = src_sh( cn, 0, src_lvl );
                            const double s1 = src_sh( cn, 1, src_lvl );
                            const double s2 = src_sh( cn, 2, src_lvl );

                            const double Rg0 = R_rot( 0, 0 ) * g0 + R_rot( 0, 1 ) * g1 + R_rot( 0, 2 ) * g2;
                            const double Rg1 = R_rot( 1, 0 ) * g0 + R_rot( 1, 1 ) * g1 + R_rot( 1, 2 ) * g2;
                            const double Rg2 = R_rot( 2, 0 ) * g0 + R_rot( 2, 1 ) * g1 + R_rot( 2, 2 ) * g2;
                            const double Rs0 = R_rot( 0, 0 ) * s0 + R_rot( 0, 1 ) * s1 + R_rot( 0, 2 ) * s2;
                            const double Rs1 = R_rot( 1, 0 ) * s0 + R_rot( 1, 1 ) * s1 + R_rot( 1, 2 ) * s2;
                            const double Rs2 = R_rot( 2, 0 ) * s0 + R_rot( 2, 1 ) * s1 + R_rot( 2, 2 ) * s2;

                            const double v0 = kwJ * ( gg_loc + ONE_THIRD * Rg0 * Rg0 ) * Rs0;
                            const double v1 = kwJ * ( gg_loc + ONE_THIRD * Rg1 * Rg1 ) * Rs1;
                            const double v2 = kwJ * ( gg_loc + ONE_THIRD * Rg2 * Rg2 ) * Rs2;

                            dst8[0][u] += R_rot( 0, 0 ) * v0 + R_rot( 1, 0 ) * v1 + R_rot( 2, 0 ) * v2;
                            dst8[1][u] += R_rot( 0, 1 ) * v0 + R_rot( 1, 1 ) * v1 + R_rot( 2, 1 ) * v2;
                            dst8[2][u] += R_rot( 0, 2 ) * v0 + R_rot( 1, 2 ) * v1 + R_rot( 2, 2 ) * v2;
                        };

                        if ( cmb_freeslip )
                        {
                            for ( int ni = 0; ni < 3; ++ni )
                                apply_rotated_diag( ni, ni, lvl0 );
                        }
                        if ( surf_freeslip )
                        {
                            for ( int ni = 0; ni < 3; ++ni )
                                apply_rotated_diag( ni, ni + 3, lvl0 + 1 );
                        }
                    }
                }
            }

            // Test-side projection for free-slip (P A P) — normals loaded from shared memory.
            if ( !Diagonal && cmb_freeslip )
            {
                for ( int c = 0; c < 4; ++c )
                {
                    const double nx  = normals_sh( corner_node[c], 0 );
                    const double ny  = normals_sh( corner_node[c], 1 );
                    const double nz  = normals_sh( corner_node[c], 2 );
                    const int    u   = CMB_CORNER_TO_UNIQUE[c];
                    const double dot = nx * dst8[0][u] + ny * dst8[1][u] + nz * dst8[2][u];
                    dst8[0][u] -= dot * nx;
                    dst8[1][u] -= dot * ny;
                    dst8[2][u] -= dot * nz;
                }
            }
            if ( !Diagonal && surf_freeslip )
            {
                for ( int c = 0; c < 4; ++c )
                {
                    const double nx  = normals_sh( corner_node[c], 0 );
                    const double ny  = normals_sh( corner_node[c], 1 );
                    const double nz  = normals_sh( corner_node[c], 2 );
                    const int    u   = SURF_CORNER_TO_UNIQUE[c];
                    const double dot = nx * dst8[0][u] + ny * dst8[1][u] + nz * dst8[2][u];
                    dst8[0][u] -= dot * nx;
                    dst8[1][u] -= dot * ny;
                    dst8[2][u] -= dot * nz;
                }
            }

            // Add back normal correction: Ann_acc[c] * u_n[c] * n_c.
            // u_n recomputed from original (unprojected) src in shared memory.
            if ( !Diagonal && cmb_freeslip )
            {
                for ( int c = 0; c < 4; ++c )
                {
                    const int    cn      = corner_node[c];
                    const double nx      = normals_sh( cn, 0 );
                    const double ny      = normals_sh( cn, 1 );
                    const double nz      = normals_sh( cn, 2 );
                    const double os0     = src_sh( cn, 0, lvl0 );
                    const double os1     = src_sh( cn, 1, lvl0 );
                    const double os2     = src_sh( cn, 2, lvl0 );
                    const double u_n_val = nx * os0 + ny * os1 + nz * os2;
                    const double corr    = Ann_acc_cmb[c] * u_n_val;
                    const int    u       = CMB_CORNER_TO_UNIQUE[c];
                    dst8[0][u] += corr * nx;
                    dst8[1][u] += corr * ny;
                    dst8[2][u] += corr * nz;
                }
            }
            if ( !Diagonal && surf_freeslip )
            {
                for ( int c = 0; c < 4; ++c )
                {
                    const int    cn      = corner_node[c];
                    const double nx      = normals_sh( cn, 0 );
                    const double ny      = normals_sh( cn, 1 );
                    const double nz      = normals_sh( cn, 2 );
                    const double os0     = src_sh( cn, 0, lvl0 + 1 );
                    const double os1     = src_sh( cn, 1, lvl0 + 1 );
                    const double os2     = src_sh( cn, 2, lvl0 + 1 );
                    const double u_n_val = nx * os0 + ny * os1 + nz * os2;
                    const double corr    = Ann_acc_surf[c] * u_n_val;
                    const int    u       = SURF_CORNER_TO_UNIQUE[c];
                    dst8[0][u] += corr * nx;
                    dst8[1][u] += corr * ny;
                    dst8[2][u] += corr * nz;
                }
            }

            // Scatter accumulated hex-cell contributions to global memory.
#pragma unroll
            for ( int dim_add = 0; dim_add < 3; ++dim_add )
            {
                Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell, dim_add ), dst8[dim_add][0] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell, dim_add ), dst8[dim_add][1] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell, dim_add ), dst8[dim_add][2] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1, dim_add ), dst8[dim_add][3] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim_add ), dst8[dim_add][4] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][5] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, dim_add ), dst8[dim_add][6] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, dim_add ), dst8[dim_add][7] );
            }

        } // end r_passes loop
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

        if ( kernel_path_ == KernelPath::Slow )
        {
            for ( int pass = 0; pass < r_passes_; ++pass )
            {
                const int r_cell_pass = r0 + pass * r_tile_ + tr;
                if ( r_cell_pass >= hex_rad_ )
                    break;
                const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_pass, CMB );
                const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell_pass + 1, SURFACE );
                operator_slow_path(
                    team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell, r_cell_pass, at_cmb, at_surface );
            }
        }
        else if ( kernel_path_ == KernelPath::FastFreeslip )
        {
            if ( diagonal_ )
                operator_fast_freeslip_path< true >( team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
            else
                operator_fast_freeslip_path< false >(
                    team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
        }
        else
        {
            if ( diagonal_ )
                operator_fast_dirichlet_neumann_path< true >(
                    team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
            else
                operator_fast_dirichlet_neumann_path< false >(
                    team, local_subdomain_id, x0, y0, r0, tx, ty, tr, x_cell, y_cell );
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
        if ( homogenize_eta_per_cell_ )
        {
            // ASPECT-style: harmonic/arithmetic/geometric mean of k=eta over
            // the 6 wedge corners, applied as a constant at every QP.
            const ScalarType inv_N = ScalarType( 1 ) / ScalarType( num_nodes_per_wedge );
            if ( eta_homogenization_mean_ == 2 )
            {
                ScalarType s = 0;
                for ( int k = 0; k < num_nodes_per_wedge; k++ ) s += k_local_hex[wedge]( k );
                k_eval = s * inv_N;
            }
            else if ( eta_homogenization_mean_ == 3 )
            {
                ScalarType log_sum = 0;
                for ( int k = 0; k < num_nodes_per_wedge; k++ ) log_sum += Kokkos::log( k_local_hex[wedge]( k ) );
                k_eval = Kokkos::exp( log_sum * inv_N );
            }
            else
            {
                ScalarType inv_sum = 0;
                for ( int k = 0; k < num_nodes_per_wedge; k++ ) inv_sum += ScalarType( 1 ) / k_local_hex[wedge]( k );
                k_eval = ScalarType( num_nodes_per_wedge ) / inv_sum;
            }
        }
        else
        {
            for ( int k = 0; k < num_nodes_per_wedge; k++ )
            {
                k_eval += shape( k, quad_point ) * k_local_hex[wedge]( k );
            }
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
        if ( use_q0_coefficient_ )
        {
            // Q0 cell-centered eta: override the nodal-extracted corners with
            // the per-cell constant.  Both wedges share the same value.  This
            // makes any downstream per-QP shape interpolation produce the
            // same constant, so the slow path (and GCA assembly which uses
            // it) sees the Q0 eta consistently with the fast paths.
            const ScalarT k_cell = k_q0_( local_subdomain_id, x_cell, y_cell, r_cell );
            for ( int w = 0; w < num_wedges_per_hex_cell; ++w )
            {
                for ( int n = 0; n < num_nodes_per_wedge; ++n )
                {
                    k_local_hex[w]( n ) = k_cell;
                }
            }
        }
        else
        {
            extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );
        }

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

static_assert( linalg::GCACapable< EpsilonDivDivKerngen< float > > );
static_assert( linalg::GCACapable< EpsilonDivDivKerngen< double > > );

} // namespace terra::fe::wedge::operators::shell