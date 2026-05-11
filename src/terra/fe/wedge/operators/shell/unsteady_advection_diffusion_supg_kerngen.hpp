#pragma once

/// @file  unsteady_advection_diffusion_supg_kerngen.hpp
/// @brief Team-based matrix-free unsteady advection-diffusion with SUPG stabilization,
///        with the fused-arithmetic optimizations transferred from EpsilonDivDivKerngen
///        and DivergenceKerngen/GradientKerngen.
///
/// Math identical to `UnsteadyAdvectionDiffusionSUPG`, but the 6×6 local mass
/// matrix M and operator matrix A (advection + diffusion + streamline-diffusion)
/// are never materialised. The standard bilinear form
///
///     dst_i = Σⱼ (M_ij + dt·A_ij) · src_j
///
/// with
///   M_ij = Σ_q w_q |det J_q| · m · φ_i(q) · φ_j(q)
///   A_ij = Σ_q w_q |det J_q| · [ κ · ∇φ_i(q)·∇φ_j(q)
///                              + φ_i(q) · u(q)·∇φ_j(q)
///                              + τ · (u(q)·∇φ_i(q)) · (u(q)·∇φ_j(q)) ]
///
/// collapses to
///     dst_i = Σ_q w_q |det J_q| · { φ_i(q) · A_scalar(q)
///                                  + ∇φ_i(q) · B_vec(q) }
/// where, for each quadrature point,
///     T̂(q)      = Σⱼ φ_j(q) · T_j
///     ∇T(q)     = Σⱼ ∇φ_j(q) · T_j
///     u(q)      = Σⱼ φ_j(q) · u_j
///     A_scalar(q) = m · T̂(q) + dt · (u(q)·∇T(q))
///     B_vec(q)    = dt·κ·∇T(q) + dt·τ·(u(q)·∇T(q)) · u(q)
///
/// τ is kept **exact**: volume-averaged over all 6 quadrature points (matching the
/// legacy code) via a pre-pass that reuses the `u(q)` values.
///
/// Dirichlet boundary handling:
///   * **column elimination** — when accumulating T̂(q) and ∇T(q), skip
///     boundary-node T_j contributions.
///   * **row elimination** — for boundary node i, only the diagonal term
///     survives. A separate inline diagonal compute handles this.
///
/// Lumped mass:  mass term replaced by row-sum diagonal M_ii T_i, where
///   M_ii = m · Σ_q w_q |det J_q| · φ_i(q)
/// (because Σⱼ φ_j ≡ 1 for a partition-of-unity basis).
///
/// Diagonal mode: full diagonal-only of (M + dt·A).
///
/// Transferred structural optimisations:
///   - Kokkos::TeamPolicy with backend-aware tiling (4,4,8) × r_passes=2 on CUDA.
///   - Host-side KernelPath dispatch (Slow / Fast) + `template<bool LumpedMass,
///     bool Diagonal, bool TreatBoundary>` so the compiler dead-eliminates unused
///     branches.
///   - LaunchBounds<128, 5>.
///   - Shared-memory staging: coords, radii, T, velocity.
///   - `ShellBoundaryCommPlan` for halo exchange.

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "communication/shell/communication_plan.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"  // for supg_tau + legacy comparison
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell {

// NB: `supg_tau` is defined in the legacy header; we #include it to reuse.
// Re-declaration here would collide, so we rely on include order / placement.

template < typename ScalarT, int VelocityVecDim = 3 >
class UnsteadyAdvectionDiffusionSUPGKerngen
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;
    using Team          = Kokkos::TeamPolicy<>::member_type;

    enum class KernelPath
    {
        Slow,
        Fast,
    };

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    linalg::VectorQ1Vec< ScalarT, VelocityVecDim > velocity_;

    ScalarT diffusivity_;
    ScalarT dt_;

    bool    treat_boundary_;
    bool    diagonal_;
    ScalarT mass_scaling_;
    bool    lumped_mass_;

    /// When false, the SUPG streamline-diffusion term τ·(u·∇φ_i)(u·∇φ_j) is
    /// dropped from the local matrix, turning this into a pure Galerkin
    /// advection-diffusion operator.  Mirrors the legacy SUPG operator's
    /// `set_supg_enabled(false)` toggle so the EV energy solver can route
    /// through the optimized kerngen path.
    bool    supg_enabled_ = true;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT >                   recv_buffers_;
    terra::communication::shell::ShellBoundaryCommPlan< grid::Grid4DDataScalar< ScalarT > > comm_plan_;

    grid::Grid4DDataScalar< ScalarType >              src_;
    grid::Grid4DDataScalar< ScalarType >              dst_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

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

    KernelPath kernel_path_ = KernelPath::Fast;

    void update_kernel_path_flag_host_only()
    {
        if constexpr ( std::is_same_v< Kokkos::DefaultExecutionSpace, Kokkos::Serial > )
        {
            kernel_path_ = KernelPath::Slow;
            return;
        }
        kernel_path_ = KernelPath::Fast;
    }

  public:
    UnsteadyAdvectionDiffusionSUPGKerngen(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const linalg::VectorQ1Vec< ScalarT, VelocityVecDim >&           velocity,
        const ScalarT                                                   diffusivity,
        const ScalarT                                                   dt,
        bool                                                            treat_boundary,
        bool                                                            diagonal     = false,
        ScalarT                                                         mass_scaling = 1.0,
        bool                                                            lumped_mass  = false,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , boundary_mask_( boundary_mask )
    , velocity_( velocity )
    , diffusivity_( diffusivity )
    , dt_( dt )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , mass_scaling_( mass_scaling )
    , lumped_mass_( lumped_mass )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , recv_buffers_( domain )
    , comm_plan_( domain )
    {
        const grid::shell::DomainInfo& domain_info = domain_.domain_info();
        local_subdomains_ = domain_.subdomains().size();
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

        util::logroot << "[SUPGKerngen] tile=(" << lat_tile_ << "," << lat_tile_ << "," << r_tile_
                      << "), r_passes=" << r_passes_ << ", team=" << team_size_ << ", blocks=" << blocks_
                      << ", path=" << path_name()
                      << ", lumped=" << lumped_mass_ << ", diagonal=" << diagonal_
                      << ", treat_boundary=" << treat_boundary_ << std::endl;
    }

    ScalarT&       dt() { return dt_; }
    const ScalarT& dt() const { return dt_; }

    void set_supg_enabled( bool on ) { supg_enabled_ = on; }
    bool supg_enabled() const { return supg_enabled_; }

    const char* path_name() const { return kernel_path_ == KernelPath::Slow ? "slow" : "fast"; }
    KernelPath  kernel_path() const { return kernel_path_; }
    void        force_slow_path() { kernel_path_ = KernelPath::Slow; }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "ad_supg_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
            assign( dst, 0 );

        src_      = src.grid_data();
        dst_      = dst.grid_data();
        vel_grid_ = velocity_.grid_data();

        util::Timer          timer_kernel( "ad_supg_kernel" );
        Kokkos::TeamPolicy<> policy( blocks_, team_size_ );
        if ( kernel_path_ == KernelPath::Fast )
            policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );

        if ( kernel_path_ == KernelPath::Slow )
        {
            Kokkos::parallel_for( "ad_supg_apply_kernel_slow", policy,
                                  KOKKOS_CLASS_LAMBDA( const Team& team ) { this->run_team_slow( team ); } );
        }
        else
        {
            Kokkos::TeamPolicy< Kokkos::LaunchBounds< 128, 5 > > fast_policy( blocks_, team_size_ );
            fast_policy.set_scratch_size( 0, Kokkos::PerTeam( team_shmem_size( team_size_ ) ) );

            // 2x2x2 = 8 template variants; host-side branch on runtime flags.
            if ( diagonal_ )
            {
                if ( treat_boundary_ && lumped_mass_ )
                    Kokkos::parallel_for( "ad_supg_fast_D_L_TB", fast_policy,
                        KOKKOS_CLASS_LAMBDA( const Team& t ) { this->template run_team_fast< true, true, true >( t ); } );
                else if ( treat_boundary_ )
                    Kokkos::parallel_for( "ad_supg_fast_D_TB", fast_policy,
                        KOKKOS_CLASS_LAMBDA( const Team& t ) { this->template run_team_fast< false, true, true >( t ); } );
                else if ( lumped_mass_ )
                    Kokkos::parallel_for( "ad_supg_fast_D_L", fast_policy,
                        KOKKOS_CLASS_LAMBDA( const Team& t ) { this->template run_team_fast< true, true, false >( t ); } );
                else
                    Kokkos::parallel_for( "ad_supg_fast_D", fast_policy,
                        KOKKOS_CLASS_LAMBDA( const Team& t ) { this->template run_team_fast< false, true, false >( t ); } );
            }
            else
            {
                if ( treat_boundary_ && lumped_mass_ )
                    Kokkos::parallel_for( "ad_supg_fast_L_TB", fast_policy,
                        KOKKOS_CLASS_LAMBDA( const Team& t ) { this->template run_team_fast< true, false, true >( t ); } );
                else if ( treat_boundary_ )
                    Kokkos::parallel_for( "ad_supg_fast_TB", fast_policy,
                        KOKKOS_CLASS_LAMBDA( const Team& t ) { this->template run_team_fast< false, false, true >( t ); } );
                else if ( lumped_mass_ )
                    Kokkos::parallel_for( "ad_supg_fast_L", fast_policy,
                        KOKKOS_CLASS_LAMBDA( const Team& t ) { this->template run_team_fast< true, false, false >( t ); } );
                else
                    Kokkos::parallel_for( "ad_supg_fast", fast_policy,
                        KOKKOS_CLASS_LAMBDA( const Team& t ) { this->template run_team_fast< false, false, false >( t ); } );
            }
        }

        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "ad_supg_comm" );
            terra::communication::shell::send_recv_with_plan( comm_plan_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size( const int /*ts*/ ) const
    {
        const int nlev = r_tile_block_ + 1;
        const int n    = lat_tile_ + 1;
        const int nxy  = n * n;
        // coords_sh(nxy,3) + r_sh(nlev) + T_sh(nxy, nlev) + vel_sh(nxy, 3, nlev)
        const size_t nscalars = size_t( nxy ) * 3 + size_t( nlev ) + size_t( nxy ) * nlev + size_t( nxy ) * 3 * nlev;
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

    // ----------------------------------------------------------
    // Slow (reference) path — team wrapper that calls the legacy per-cell kernel.
    // Bit-identical to UnsteadyAdvectionDiffusionSUPG::operator().
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

    // Per-cell legacy math (replica of UnsteadyAdvectionDiffusionSUPG::operator()).
    KOKKOS_INLINE_FUNCTION
    void process_cell_legacy( int s, int x_cell, int y_cell, int r_cell ) const
    {
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, s, x_cell, y_cell );

        const ScalarT r_1 = radii_( s, r_cell );
        const ScalarT r_2 = radii_( s, r_cell + 1 );

        constexpr auto num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;
        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];
        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        dense::Vec< ScalarT, VelocityVecDim > vel_interp[num_wedges_per_hex_cell][num_quad_points] = {};
        dense::Vec< ScalarT, 6 >              vel_coeffs[VelocityVecDim][num_wedges_per_hex_cell];
        for ( int d = 0; d < VelocityVecDim; ++d )
            extract_local_wedge_vector_coefficients( vel_coeffs[d], s, x_cell, y_cell, r_cell, d, vel_grid_ );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            for ( int q = 0; q < num_quad_points; ++q )
                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    for ( int d = 0; d < VelocityVecDim; ++d )
                        vel_interp[wedge][q]( d ) += vel_coeffs[d][wedge]( i ) * shape_i;
                }

        ScalarT streamline_diffusivity[num_wedges_per_hex_cell];
        const auto h = r_2 - r_1;
        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
        {
            ScalarT tau_accum = 0.0, waccum = 0.0;
            for ( int q = 0; q < num_quad_points; ++q )
            {
                const auto&   uq         = vel_interp[wedge][q];
                const ScalarT vel_norm_q = uq.norm();
                const ScalarT tau_q      = supg_tau< ScalarT >( vel_norm_q, diffusivity_, h, 1e-08 );
                tau_accum += tau_q * quad_weights[q];
                waccum    += quad_weights[q];
            }
            streamline_diffusivity[wedge] = ( supg_enabled_ && waccum > 0.0 ) ? ( tau_accum / waccum ) : 0.0;
        }

        dense::Mat< ScalarT, 6, 6 > A[num_wedges_per_hex_cell] = {};
        dense::Mat< ScalarT, 6, 6 > M[num_wedges_per_hex_cell] = {};

        for ( int q = 0; q < num_quad_points; ++q )
        {
            const auto w = quad_weights[q];
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();
                const auto vel              = vel_interp[wedge][q];

                for ( int i = 0; i < num_nodes_per_wedge; ++i )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    const auto grad_i  = J_inv_transposed * grad_shape( i, quad_points[q] );
                    for ( int j = 0; j < num_nodes_per_wedge; ++j )
                    {
                        const auto shape_j = shape( j, quad_points[q] );
                        const auto grad_j  = J_inv_transposed * grad_shape( j, quad_points[q] );

                        const auto mass       = shape_i * shape_j;
                        const auto diffusion  = diffusivity_ * grad_i.dot( grad_j );
                        const auto advection  = vel.dot( grad_j ) * shape_i;
                        const auto streamline = streamline_diffusivity[wedge] * vel.dot( grad_j ) * vel.dot( grad_i );

                        M[wedge]( i, j ) += w * mass_scaling_ * mass * det;
                        A[wedge]( i, j ) += w * dt_ * ( diffusion + advection + streamline ) * det;
                    }
                }
            }
        }

        if ( lumped_mass_ )
        {
            dense::Vec< ScalarT, 6 > ones;
            ones.fill( 1.0 );
            M[0] = dense::Mat< ScalarT, 6, 6 >::diagonal_from_vec( M[0] * ones );
            M[1] = dense::Mat< ScalarT, 6, 6 >::diagonal_from_vec( M[1] * ones );
        }

        if ( treat_boundary_ )
        {
            const int at_cmb_boundary     = util::has_flag(
                boundary_mask_( s, x_cell, y_cell, r_cell ), grid::shell::ShellBoundaryFlag::CMB );
            const int at_surface_boundary = util::has_flag(
                boundary_mask_( s, x_cell, y_cell, r_cell + 1 ), grid::shell::ShellBoundaryFlag::SURFACE );

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; ++wedge )
            {
                dense::Mat< ScalarT, 6, 6 > boundary_mask;
                boundary_mask.fill( 1.0 );
                if ( at_cmb_boundary )
                    for ( int i = 0; i < 6; ++i )
                        for ( int j = 0; j < 6; ++j )
                            if ( i != j && ( i < 3 || j < 3 ) )
                                boundary_mask( i, j ) = 0.0;
                if ( at_surface_boundary )
                    for ( int i = 0; i < 6; ++i )
                        for ( int j = 0; j < 6; ++j )
                            if ( i != j && ( i >= 3 || j >= 3 ) )
                                boundary_mask( i, j ) = 0.0;
                M[wedge].hadamard_product( boundary_mask );
                A[wedge].hadamard_product( boundary_mask );
            }
        }

        if ( diagonal_ )
        {
            M[0] = M[0].diagonal();
            M[1] = M[1].diagonal();
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, s, x_cell, y_cell, r_cell, src_ );

        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];
        dst[0] = ( M[0] + A[0] ) * src[0];
        dst[1] = ( M[1] + A[1] ) * src[1];

        atomically_add_local_wedge_scalar_coefficients( dst_, s, x_cell, y_cell, r_cell, dst );
    }

    // ----------------------------------------------------------
    // Fast path: fused arithmetic, shmem-cached inputs.
    // Templated on LumpedMass, Diagonal, TreatBoundary so the compiler
    // dead-eliminates inapplicable branches.
    // ----------------------------------------------------------
    template < bool LumpedMass, bool Diagonal, bool TreatBoundary >
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
        using ScratchR =
            Kokkos::View< double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchT =
            Kokkos::View< double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;
        using ScratchVel =
            Kokkos::View< double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged >;

        ScratchCoords coords_sh( shmem, nxy, 3 );
        shmem += nxy * 3;
        ScratchR      r_sh( shmem, nlev );
        shmem += nlev;
        ScratchT      T_sh( shmem, nxy, nlev );
        shmem += nxy * nlev;
        ScratchVel    vel_sh( shmem, nxy, 3, nlev );

        auto node_id = [&]( int nx, int ny ) -> int { return nx + n * ny; };

        // Preload coords, radii, T, velocity.
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, nxy ), [&]( int id ) {
            const int dxn = id % n;
            const int dyn = id / n;
            const int xi  = x0 + dxn;
            const int yi  = y0 + dyn;
            if ( xi <= hex_lat_ && yi <= hex_lat_ )
            {
                coords_sh( id, 0 ) = grid_( local_subdomain_id, xi, yi, 0 );
                coords_sh( id, 1 ) = grid_( local_subdomain_id, xi, yi, 1 );
                coords_sh( id, 2 ) = grid_( local_subdomain_id, xi, yi, 2 );
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
        const int total = nxy * nlev;
        Kokkos::parallel_for( Kokkos::TeamThreadRange( team, total ), [&]( int t ) {
            const int node = t / nlev;
            const int lvl  = t - node * nlev;
            const int dxn  = node % n;
            const int dyn  = node / n;
            const int xi   = x0 + dxn;
            const int yi   = y0 + dyn;
            const int rr   = r0 + lvl;
            if ( xi <= hex_lat_ && yi <= hex_lat_ && rr <= hex_rad_ )
            {
                T_sh( node, lvl )      = src_( local_subdomain_id, xi, yi, rr );
                vel_sh( node, 0, lvl ) = vel_grid_( local_subdomain_id, xi, yi, rr, 0 );
                vel_sh( node, 1, lvl ) = vel_grid_( local_subdomain_id, xi, yi, rr, 1 );
                vel_sh( node, 2, lvl ) = vel_grid_( local_subdomain_id, xi, yi, rr, 2 );
            }
            else
            {
                T_sh( node, lvl ) = 0.0;
                vel_sh( node, 0, lvl ) = vel_sh( node, 1, lvl ) = vel_sh( node, 2, lvl ) = 0.0;
            }
        } );

        team.team_barrier();

        if ( tr >= r_tile_ )
            return;
        if ( x_cell >= hex_lat_ || y_cell >= hex_lat_ )
            return;

        // Wedge-node offset table.
        constexpr int WEDGE_NODE_OFF[2][6][3] = {
            { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
            { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

        // Felippa 3x2 quadrature points & weights, hoisted for compile-time propagation.
        constexpr int NQ = 6;
        constexpr double QUAD_W = 1.0 / 6.0;
        //   (xi, eta, zeta)
        //   points 0..2 share zeta = -1/sqrt(3); points 3..5 share zeta = +1/sqrt(3).
        constexpr double ONE_OVER_SQRT3 = 0.5773502691896257;
        //   Lateral barycentric coordinates per quadrature point:
        //       (1-xi-eta, xi, eta)
        constexpr double BARY[NQ][3] = {
            { 1.0/6.0, 2.0/3.0, 1.0/6.0 },   // q=0: xi=2/3, eta=1/6
            { 1.0/6.0, 1.0/6.0, 2.0/3.0 },   // q=1: xi=1/6, eta=2/3
            { 2.0/3.0, 1.0/6.0, 1.0/6.0 },   // q=2: xi=1/6, eta=1/6
            { 1.0/6.0, 2.0/3.0, 1.0/6.0 },   // q=3: same lat as 0
            { 1.0/6.0, 1.0/6.0, 2.0/3.0 },   // q=4: same lat as 1
            { 2.0/3.0, 1.0/6.0, 1.0/6.0 }    // q=5: same lat as 2
        };
        // Zeta half-width factor: grad_shape_rad returns ±0.5; shape_rad(j, zeta) = (1 ∓ zeta)/2.
        //     j<3 (lower layer): shape_rad = (1 - zeta)/2
        //     j>=3 (upper layer): shape_rad = (1 + zeta)/2
        constexpr double ZETA_Q[NQ] = {
            -ONE_OVER_SQRT3, -ONE_OVER_SQRT3, -ONE_OVER_SQRT3,
            +ONE_OVER_SQRT3, +ONE_OVER_SQRT3, +ONE_OVER_SQRT3 };

        // PHI[j][q] = shape_lat(j, xi, eta) * shape_rad(j, zeta)
        // shape_lat(j=0|3, xi, eta) = 1 - xi - eta = BARY[q][0]
        // shape_lat(j=1|4, ...) = xi                = BARY[q][1]
        // shape_lat(j=2|5, ...) = eta               = BARY[q][2]
        // shape_rad(j<3, zeta) = 0.5 (1 - zeta)
        // shape_rad(j>=3,zeta) = 0.5 (1 + zeta)
        auto phi = [&]( int j, int q ) -> double {
            const double sl = BARY[q][j % 3];
            const double sr = ( j < 3 ) ? 0.5 * ( 1.0 - ZETA_Q[q] ) : 0.5 * ( 1.0 + ZETA_Q[q] );
            return sl * sr;
        };
        //   grad_shape_lat_xi(j=0|3) = -1, (1|4) = +1, (2|5) = 0
        //   grad_shape_lat_eta(j=0|3) = -1, (1|4) = 0,  (2|5) = +1
        //   grad_shape_rad(j<3) = -0.5, (j>=3) = +0.5
        //   grad_shape(j, xi, eta, zeta)
        //     = ( grad_lat_xi(j) * shape_rad(j, zeta),
        //         grad_lat_eta(j) * shape_rad(j, zeta),
        //         shape_lat(j, xi, eta) * grad_shape_rad(j) )
        auto gref = [&]( int j, int q, int d ) -> double {
            const double sr = ( j < 3 ) ? 0.5 * ( 1.0 - ZETA_Q[q] ) : 0.5 * ( 1.0 + ZETA_Q[q] );
            const double grad_rad = ( j < 3 ) ? -0.5 : 0.5;
            const int jmod = j % 3;
            const double glat_xi  = ( jmod == 0 ) ? -1.0 : ( jmod == 1 ) ? 1.0 : 0.0;
            const double glat_eta = ( jmod == 0 ) ? -1.0 : ( jmod == 1 ) ? 0.0 : 1.0;
            const double sl = BARY[q][jmod];
            if ( d == 0 ) return glat_xi * sr;
            if ( d == 1 ) return glat_eta * sr;
            return sl * grad_rad;
        };

        // Node neighbours for Jacobian surface-coord gather (lat-only).
        const int n00 = node_id( tx,     ty     );
        const int n01 = node_id( tx,     ty + 1 );
        const int n10 = node_id( tx + 1, ty     );
        const int n11 = node_id( tx + 1, ty + 1 );

        // Per-pass radial loop.
        for ( int pass = 0; pass < r_passes_; ++pass )
        {
            const int lvl0       = pass * r_tile_ + tr;
            const int r_cell_abs = r0 + lvl0;
            if ( r_cell_abs >= hex_rad_ )
                break;

            const double r_lower = r_sh( lvl0     );
            const double r_upper = r_sh( lvl0 + 1 );
            const double half_dr = 0.5 * ( r_upper - r_lower );
            const double r_mid   = 0.5 * ( r_lower + r_upper );
            const double h_cell  = r_upper - r_lower;

            // Boundary flags (determine Dirichlet row/col elimination masks).
            bool at_cmb = false, at_surface = false;
            if constexpr ( TreatBoundary )
            {
                at_cmb     = util::has_flag( boundary_mask_( local_subdomain_id, x_cell, y_cell, r_cell_abs     ),
                                             grid::shell::ShellBoundaryFlag::CMB );
                at_surface = util::has_flag( boundary_mask_( local_subdomain_id, x_cell, y_cell, r_cell_abs + 1 ),
                                             grid::shell::ShellBoundaryFlag::SURFACE );
            }

            // Dirichlet treatment helper: node j is a "boundary node" if
            //   at_cmb     AND ddr_j == 0   (inner layer)   -> j in {0,1,2}
            //   at_surface AND ddr_j == 1   (outer layer)   -> j in {3,4,5}
            // We keep the diagonal (j == i) so row-elimination is handled in the scatter branch.
            auto boundary_j_col = [&]( int j ) -> bool {
                if constexpr ( !TreatBoundary ) return false;
                return ( at_cmb && j < 3 ) || ( at_surface && j >= 3 );
            };

            for ( int w = 0; w < num_wedges_per_hex_cell; ++w )
            {
                // Wedge surface vertices in the lateral plane (shmem lookup).
                const int v0 = ( w == 0 ? n00 : n11 );
                const int v1 = ( w == 0 ? n10 : n01 );
                const int v2 = ( w == 0 ? n01 : n10 );

                // Surface coords P0, P1, P2 (3 doubles each).
                const double P0[3] = { coords_sh( v0, 0 ), coords_sh( v0, 1 ), coords_sh( v0, 2 ) };
                const double P1[3] = { coords_sh( v1, 0 ), coords_sh( v1, 1 ), coords_sh( v1, 2 ) };
                const double P2[3] = { coords_sh( v2, 0 ), coords_sh( v2, 1 ), coords_sh( v2, 2 ) };

                // Differences, hoisted (used per quad).
                const double dP1_P0[3] = { P1[0] - P0[0], P1[1] - P0[1], P1[2] - P0[2] };
                const double dP2_P0[3] = { P2[0] - P0[0], P2[1] - P0[1], P2[2] - P0[2] };

                // Pre-pass 1: interpolate velocity at each quad point; compute tau_wedge.
                // We cache vel_q[q][d] for reuse in main pass + boundary-diagonal pass.
                double vel_q[NQ][3] = { { 0.0 } };
                {
                    double tau_sum = 0.0;
                    double w_sum   = 0.0;
                    for ( int q = 0; q < NQ; ++q )
                    {
                        double ux = 0.0, uy = 0.0, uz = 0.0;
#pragma unroll
                        for ( int j = 0; j < num_nodes_per_wedge; ++j )
                        {
                            const int    ddx = WEDGE_NODE_OFF[w][j][0];
                            const int    ddy = WEDGE_NODE_OFF[w][j][1];
                            const int    ddr = WEDGE_NODE_OFF[w][j][2];
                            const int    nid = node_id( tx + ddx, ty + ddy );
                            const int    lvl = lvl0 + ddr;
                            const double p   = phi( j, q );
                            ux += p * vel_sh( nid, 0, lvl );
                            uy += p * vel_sh( nid, 1, lvl );
                            uz += p * vel_sh( nid, 2, lvl );
                        }
                        vel_q[q][0] = ux;
                        vel_q[q][1] = uy;
                        vel_q[q][2] = uz;

                        const double vn   = Kokkos::sqrt( ux * ux + uy * uy + uz * uz );
                        const double tauq = supg_tau< double >( vn, double( diffusivity_ ), h_cell, 1e-08 );
                        tau_sum += tauq * QUAD_W;
                        w_sum   += QUAD_W;
                    }
                    // volume-averaged tau (unchanged from legacy); zero out
                    // when SUPG is disabled (EV pipeline) so the streamline
                    // contributions degenerate cleanly.
                    const double tau_wedge = ( supg_enabled_ && w_sum > 0.0 ) ? ( tau_sum / w_sum ) : 0.0;

                    // Per-i accumulators for the 6 output nodes of this wedge.
                    double dst_acc[num_nodes_per_wedge] = { 0.0 };
                    // Lumped mass diagonal accumulator (per output node i); used only if LumpedMass.
                    double lumped_Mii[num_nodes_per_wedge] = { 0.0 };

                    // Main fused loop over quad points.
                    for ( int q = 0; q < NQ; ++q )
                    {
                        // Assemble J at this quad point.
                        //   J_*_0 = r(zeta_q) * dP1_P0
                        //   J_*_1 = r(zeta_q) * dP2_P0
                        //   J_*_2 = half_dr * (BARY[q][0]*P0 + BARY[q][1]*P1 + BARY[q][2]*P2)
                        const double r_at_q = r_mid + ZETA_Q[q] * half_dr;
                        const double b0 = BARY[q][0], b1 = BARY[q][1], b2 = BARY[q][2];

                        const double J00 = r_at_q * dP1_P0[0];
                        const double J10 = r_at_q * dP1_P0[1];
                        const double J20 = r_at_q * dP1_P0[2];
                        const double J01 = r_at_q * dP2_P0[0];
                        const double J11 = r_at_q * dP2_P0[1];
                        const double J21 = r_at_q * dP2_P0[2];
                        const double J02 = half_dr * ( b0 * P0[0] + b1 * P1[0] + b2 * P2[0] );
                        const double J12 = half_dr * ( b0 * P0[1] + b1 * P1[1] + b2 * P2[1] );
                        const double J22 = half_dr * ( b0 * P0[2] + b1 * P1[2] + b2 * P2[2] );

                        const double J_det  = J00 * J11 * J22 - J00 * J12 * J21
                                            - J01 * J10 * J22 + J01 * J12 * J20
                                            + J02 * J10 * J21 - J02 * J11 * J20;
                        const double abs_det = Kokkos::abs( J_det );
                        const double inv_det = 1.0 / J_det;

                        // invJ_T entries (row d of J^{-T} dot ∇_ref = physical gradient component d)
                        const double i00 = inv_det * (  J11 * J22 - J12 * J21 );
                        const double i01 = inv_det * ( -J10 * J22 + J12 * J20 );
                        const double i02 = inv_det * (  J10 * J21 - J11 * J20 );
                        const double i10 = inv_det * ( -J01 * J22 + J02 * J21 );
                        const double i11 = inv_det * (  J00 * J22 - J02 * J20 );
                        const double i12 = inv_det * ( -J00 * J21 + J01 * J20 );
                        const double i20 = inv_det * (  J01 * J12 - J02 * J11 );
                        const double i21 = inv_det * ( -J00 * J12 + J02 * J10 );
                        const double i22 = inv_det * (  J00 * J11 - J01 * J10 );

                        const double wdet = QUAD_W * abs_det;

                        // Gather: T̂(q), ∇T(q). Column-elimination at Dirichlet boundary nodes.
                        double T_hat = 0.0;
                        double gT0 = 0.0, gT1 = 0.0, gT2 = 0.0;
#pragma unroll
                        for ( int j = 0; j < num_nodes_per_wedge; ++j )
                        {
                            if constexpr ( TreatBoundary )
                                if ( boundary_j_col( j ) )
                                    continue;

                            const int    ddx = WEDGE_NODE_OFF[w][j][0];
                            const int    ddy = WEDGE_NODE_OFF[w][j][1];
                            const int    ddr = WEDGE_NODE_OFF[w][j][2];
                            const int    nid = node_id( tx + ddx, ty + ddy );
                            const int    lvl = lvl0 + ddr;
                            const double Tj  = T_sh( nid, lvl );

                            const double pj = phi( j, q );
                            T_hat += pj * Tj;

                            const double gx = gref( j, q, 0 );
                            const double gy = gref( j, q, 1 );
                            const double gz = gref( j, q, 2 );
                            const double g0 = i00 * gx + i01 * gy + i02 * gz;
                            const double g1 = i10 * gx + i11 * gy + i12 * gz;
                            const double g2 = i20 * gx + i21 * gy + i22 * gz;
                            gT0 += g0 * Tj;
                            gT1 += g1 * Tj;
                            gT2 += g2 * Tj;
                        }

                        const double ux = vel_q[q][0], uy = vel_q[q][1], uz = vel_q[q][2];
                        const double u_dot_gT = ux * gT0 + uy * gT1 + uz * gT2;

                        // A_scalar and B_vec, with LumpedMass handling.
                        // For LumpedMass: exclude the mass term from A_scalar (handled in diagonal acc below).
                        const double mass_in_scalar = LumpedMass ? 0.0 : ( double( mass_scaling_ ) * T_hat );
                        const double A_scalar = mass_in_scalar + double( dt_ ) * u_dot_gT;
                        const double dkappa   = double( dt_ ) * double( diffusivity_ );
                        const double dtau     = double( dt_ ) * tau_wedge * u_dot_gT;
                        const double Bx = dkappa * gT0 + dtau * ux;
                        const double By = dkappa * gT1 + dtau * uy;
                        const double Bz = dkappa * gT2 + dtau * uz;

                        // Scatter accumulate into each output node's dst_acc.
#pragma unroll
                        for ( int i = 0; i < num_nodes_per_wedge; ++i )
                        {
                            const double pi  = phi( i, q );
                            const double gxi = gref( i, q, 0 );
                            const double gyi = gref( i, q, 1 );
                            const double gzi = gref( i, q, 2 );
                            const double gi0 = i00 * gxi + i01 * gyi + i02 * gzi;
                            const double gi1 = i10 * gxi + i11 * gyi + i12 * gzi;
                            const double gi2 = i20 * gxi + i21 * gyi + i22 * gzi;

                            // Boundary row: only the i=j diagonal term contributes.
                            bool is_boundary_i = false;
                            if constexpr ( TreatBoundary )
                            {
                                is_boundary_i = ( at_cmb && i < 3 ) || ( at_surface && i >= 3 );
                            }

                            // Diagonal accumulators (used when is_boundary_i or Diagonal==true).
                            // When LumpedMass is true, the mass contribution uses the row-sum form
                            //   M_ii_lumped = m · Σ_q wdet · φ_i(q)        (since Σⱼ φ_j ≡ 1)
                            // When LumpedMass is false, the mass diagonal is the unlumped one:
                            //   M_ii        = m · Σ_q wdet · φ_i(q) · φ_i(q)
                            // A's diagonal (diff + adv + supg) is identical for lumped and unlumped.
                            auto add_diagonal_contribution = [&]() {
                                const int    ddx = WEDGE_NODE_OFF[w][i][0];
                                const int    ddy = WEDGE_NODE_OFF[w][i][1];
                                const int    ddr = WEDGE_NODE_OFF[w][i][2];
                                const int    nid = node_id( tx + ddx, ty + ddy );
                                const int    lvl = lvl0 + ddr;
                                const double Ti  = T_sh( nid, lvl );

                                const double u_dot_gi = ux * gi0 + uy * gi1 + uz * gi2;
                                const double diff_ii  = double( diffusivity_ ) * ( gi0 * gi0 + gi1 * gi1 + gi2 * gi2 );
                                const double adv_ii   = pi * u_dot_gi;
                                const double supg_ii  = tau_wedge * u_dot_gi * u_dot_gi;
                                const double A_ii     = diff_ii + adv_ii + supg_ii;

                                if constexpr ( LumpedMass )
                                {
                                    // Row-sum lumped mass (partition of unity).
                                    lumped_Mii[i] += wdet * double( mass_scaling_ ) * pi;
                                    dst_acc[i]    += wdet * double( dt_ ) * A_ii * Ti;
                                }
                                else
                                {
                                    const double M_ii_q = double( mass_scaling_ ) * pi * pi;
                                    dst_acc[i] += wdet * ( M_ii_q + double( dt_ ) * A_ii ) * Ti;
                                }
                            };

                            if ( is_boundary_i )
                            {
                                add_diagonal_contribution();
                            }
                            else
                            {
                                if constexpr ( Diagonal )
                                {
                                    add_diagonal_contribution();
                                }
                                else
                                {
                                    // Standard fused interior contribution.
                                    const double contrib = wdet * ( pi * A_scalar + gi0 * Bx + gi1 * By + gi2 * Bz );
                                    dst_acc[i] += contrib;
                                    if constexpr ( LumpedMass )
                                    {
                                        // Row-sum lumped mass (partition of unity).
                                        lumped_Mii[i] += wdet * double( mass_scaling_ ) * pi;
                                    }
                                }
                            }
                        }
                    } // end quad loop

                    // Apply lumped mass: dst_i += M_ii_lumped · T_i
                    if constexpr ( LumpedMass )
                    {
#pragma unroll
                        for ( int i = 0; i < num_nodes_per_wedge; ++i )
                        {
                            // For interior nodes: lumped_Mii[i] is Σ_q wdet · m · φ_i(q).
                            // For boundary nodes: lumped_Mii[i] is Σ_q wdet · m · φ_i(q)·φ_i(q).
                            // Both are diagonal contributions to be multiplied by T_i.
                            const int    ddx = WEDGE_NODE_OFF[w][i][0];
                            const int    ddy = WEDGE_NODE_OFF[w][i][1];
                            const int    ddr = WEDGE_NODE_OFF[w][i][2];
                            const int    nid = node_id( tx + ddx, ty + ddy );
                            const int    lvl = lvl0 + ddr;
                            const double Ti  = T_sh( nid, lvl );
                            dst_acc[i] += lumped_Mii[i] * Ti;
                        }
                    }

                    // Final atomic scatter to global dst_.
                    // Use the scalar-wedge helper by materialising the two-wedge buffer in order.
                    // atomically_add_local_wedge_scalar_coefficients expects a [2][6] layout
                    // for both wedges at once, so we accumulate per-wedge and emit inline atomics.
                    const int xc = x_cell;
                    const int yc = y_cell;
                    const int rc = r_cell_abs;
#pragma unroll
                    for ( int i = 0; i < num_nodes_per_wedge; ++i )
                    {
                        const int ddx = WEDGE_NODE_OFF[w][i][0];
                        const int ddy = WEDGE_NODE_OFF[w][i][1];
                        const int ddr = WEDGE_NODE_OFF[w][i][2];
                        Kokkos::atomic_add( &dst_( local_subdomain_id, xc + ddx, yc + ddy, rc + ddr ), dst_acc[i] );
                    }
                }
            } // end wedge loop
        } // end pass loop
    }
};

static_assert( linalg::OperatorLike< UnsteadyAdvectionDiffusionSUPGKerngen< double > > );

} // namespace terra::fe::wedge::operators::shell
