
#pragma once

#include "communication/shell/fv_communication.hpp"
#include "fv/hex/helpers.hpp"
#include "fv/hex/operators/geometry_helper.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"
#include "mpi/mpi.hpp"
#include "util/timer.hpp"

namespace terra::fv::hex::operators {

// ============================================================================
// FVFCTBuffers — pre-allocated working arrays for one FCT step
// ============================================================================

/// @brief All working storage for a single FCT timestep, allocated once and reused every step.
///
/// Avoids heap allocation in the time loop.  Pass a single `FVFCTBuffers` instance to all
/// stage functions (`fct_predictor`, `fct_limiter`, `fct_correction`) and to the convenience
/// wrappers (`fct_explicit_step`, `upwind_explicit_step`, `fct_semiimplicit_step`).
///
/// @tparam ScalarType  Floating-point type used for all fields.
template < typename ScalarType >
struct FVFCTBuffers
{
    /// Low-order (upwind) predictor \f$T^L\f$; same cell layout as the transported scalar.
    grid::Grid4DDataScalar< ScalarType > T_L;
    /// Pre-scaled antidiffusive fluxes \f$\tilde{f}_{ij} = (\Delta t / M_{ii})\,f_{ij}\f$,
    /// sixth dimension indexes the 6 faces in neighbour order (0=x−1, 1=x+1, …, 5=r+1).
    grid::Grid5DDataScalar< ScalarType > antidiff;
    /// Zalesak correction factor \f$R_i^+\f$ — limits incoming positive antidiff flux.
    grid::Grid4DDataScalar< ScalarType > R_plus;
    /// Zalesak correction factor \f$R_i^-\f$ — limits incoming negative antidiff flux.
    grid::Grid4DDataScalar< ScalarType > R_minus;

    /// MPI ghost-layer buffers for \f$T\f$ (reused for `T_L` as well via the same slot).
    communication::shell::FVGhostLayerBuffers< ScalarType > ghost_T;
    /// MPI ghost-layer buffers for \f$R^+\f$ (needed so the correction kernel reads neighbours).
    communication::shell::FVGhostLayerBuffers< ScalarType > ghost_R_plus;
    /// MPI ghost-layer buffers for \f$R^-\f$.
    communication::shell::FVGhostLayerBuffers< ScalarType > ghost_R_minus;

    explicit FVFCTBuffers( const grid::shell::DistributedDomain& domain )
    : T_L( "fct_T_L",
           domain.subdomains().size(),
           domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
           domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
           domain.domain_info().subdomain_num_nodes_radially() + 1 )
    , antidiff(
          "fct_antidiff",
          domain.subdomains().size(),
          domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
          domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
          domain.domain_info().subdomain_num_nodes_radially() + 1,
          6 )
    , R_plus(
          "fct_R_plus",
          domain.subdomains().size(),
          domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
          domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
          domain.domain_info().subdomain_num_nodes_radially() + 1 )
    , R_minus(
          "fct_R_minus",
          domain.subdomains().size(),
          domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
          domain.domain_info().subdomain_num_nodes_per_side_laterally() + 1,
          domain.domain_info().subdomain_num_nodes_radially() + 1 )
    , ghost_T( domain )
    , ghost_R_plus( domain )
    , ghost_R_minus( domain )
    {}
};

namespace fct_detail {

// Alias into the shared geometry helper.
template < typename ScalarT >
using GeometryHelper = operators::detail::GeometryHelper< ScalarT >;

} // namespace fct_detail

// ============================================================================
// Stable timestep computation
// ============================================================================

/// @brief Kokkos kernel that computes the local maximum stable explicit dt for each FV cell.
///
/// The low-order predictor \f$T^L_i\f$ is stable if and only if
/// \f[
///     \frac{\Delta t}{M_{ii}}\,\lambda_i \leq 1, \qquad
///     \lambda_i = \sum_{j:\,\beta_{ij}<0} |\beta_{ij}|
///                 + \sum_j \kappa\,\frac{|\mathbf{S}_{f,j}|^2}
///                                       {(\mathbf{x}_j-\mathbf{x}_i)\cdot\mathbf{S}_{f,j}}
/// \f]
/// (assumes `subtract_divergence = true`; the formula naturally coincides with the advective
/// stability limit \f$\sum_{j:\,\beta>0}\beta_{ij}\f$ for exactly divergence-free fields.)
///
/// For each cell the kernel outputs \f$M_{ii}/\lambda_i\f$ (a time scale), so the global
/// minimum is the largest dt that satisfies the stability criterion everywhere.
///
/// This accounts for:
///   - Lateral face fluxes on irregular/small cells near pentagon vertices of the icosahedral
///     grid — these are missed by the simpler \f$h_\text{min,radial}/u_\text{max}\f$ estimate.
///   - Non-orthogonal diffusion stencils, where \f$|\mathbf{S}_f|^2/(\mathbf{dx}\cdot\mathbf{S}_f)\f$
///     can be much larger than \f$1/h^2\f$.
template < typename ScalarT >
struct ComputeDtStableKernel
{
    using ScalarType = ScalarT;
    using Vec3       = dense::Vec< ScalarT, 3 >;
    using GH         = fct_detail::GeometryHelper< ScalarT >;

    static constexpr int num_neighbors = GH::num_neighbors;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;
    grid::Grid4DDataVec< ScalarT, 3 > cell_centers_;
    grid::Grid4DDataVec< ScalarT, 3 > vel_grid_;
    ScalarT                           diffusivity_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r, ScalarT& local_min ) const
    {
        constexpr int cell_offset_x[num_neighbors] = { -1, 1, 0, 0, 0, 0 };
        constexpr int cell_offset_y[num_neighbors] = { 0, 0, -1, 1, 0, 0 };
        constexpr int cell_offset_r[num_neighbors] = { 0, 0, 0, 0, -1, 1 };

        ScalarT beta[num_neighbors];
        ScalarT M_ii = ScalarT( 0 );
        Vec3    S_f[num_neighbors];
        GH::compute_geometry( grid_, radii_, cell_centers_, vel_grid_, id, x, y, r, beta, M_ii, S_f );

        // Effective diagonal of the low-order predictor (subtract_divergence = true):
        //   lambda_i = -sum_{beta<0} beta_ij  +  sum_j kappa * |S_f|^2 / (dx . S_f)
        // For divergence-free velocity: this equals sum_{beta>0} beta_ij.
        ScalarT lambda = ScalarT( 0 );

        for ( int n = 0; n < num_neighbors; ++n )
        {
            if ( beta[n] < ScalarT( 0 ) )
                lambda -= beta[n]; // accumulate inflow (beta<0 => outflow from neighbour, inflow to i)

            if ( diffusivity_ > ScalarT( 0 ) )
            {
                const int   nx = x + GH::cell_offset_x( n );
                const int   ny = y + GH::cell_offset_y( n );
                const int   nr = r + GH::cell_offset_r( n );
                const Vec3  dx{
                    cell_centers_( id, nx, ny, nr, 0 ) - cell_centers_( id, x, y, r, 0 ),
                    cell_centers_( id, nx, ny, nr, 1 ) - cell_centers_( id, x, y, r, 1 ),
                    cell_centers_( id, nx, ny, nr, 2 ) - cell_centers_( id, x, y, r, 2 ) };
                const ScalarT denom = dx.dot( S_f[n] );
                if ( denom > ScalarT( 0 ) )
                    lambda += diffusivity_ * S_f[n].dot( S_f[n] ) / denom;
            }
        }

        const ScalarT dt_cell = ( lambda > ScalarT( 0 ) ) ? ( M_ii / lambda ) : ScalarT( 1e30 );
        local_min             = Kokkos::min( local_min, dt_cell );
    }
};

/// @brief Compute the largest explicit time step that keeps the FCT low-order predictor stable.
///
/// Performs a Kokkos parallel reduction over all non-ghost FV cells followed by an
/// MPI_Allreduce to obtain the global minimum across all MPI ranks.
///
/// The result is exact (derived from the actual face-normal velocity fluxes and cell volumes),
/// unlike the approximate estimate \f$h_\text{min,radial} / u_\text{max}\f$ which ignores
/// lateral cell sizes and diffusion stiffness on non-orthogonal cells.
///
/// @note The resulting time step size can still be too large to ensure an "accurate" time discretization.
///       It can be a good idea to scale it down further in practice.
///
/// **Typical usage:**
/// @code
///   const ScalarType dt = dt_scaling * fv::hex::operators::compute_dt_stable(
///       domain, u, cell_centers.grid_data(), coords_shell, coords_radii, diffusivity);
/// @endcode
///
/// @param domain       Distributed domain.
/// @param vel          Q1 nodal velocity (read-only; no ghost-layer update required).
/// @param cell_centers Pre-computed cell centres with ghost layers (from `initialize_cell_centers`).
/// @param grid         Lateral node coordinates of the unit-sphere surface.
/// @param radii        Radial shell radii.
/// @param diffusivity  Physical diffusivity \f$\kappa\f$ (default 0 = pure advection).
/// @returns The minimum over all cells of \f$M_{ii}/\lambda_i\f$.
template < typename ScalarT >
ScalarT compute_dt_stable(
    const grid::shell::DistributedDomain&    domain,
    const linalg::VectorQ1Vec< ScalarT, 3 >& vel,
    const grid::Grid4DDataVec< ScalarT, 3 >& cell_centers,
    const grid::Grid3DDataVec< ScalarT, 3 >& grid,
    const grid::Grid2DDataScalar< ScalarT >& radii,
    const ScalarT                            diffusivity = ScalarT( 0 ) )
{
    ScalarT local_min;

    Kokkos::parallel_reduce(
        "compute_dt_stable",
        grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ),
        ComputeDtStableKernel< ScalarT >{
            .grid_         = grid,
            .radii_        = radii,
            .cell_centers_ = cell_centers,
            .vel_grid_     = vel.grid_data(),
            .diffusivity_  = diffusivity,
        },
        Kokkos::Min< ScalarT >( local_min ) );

    Kokkos::fence();

    ScalarT global_min = local_min;
    MPI_Allreduce( &local_min, &global_min, 1, mpi::mpi_datatype< ScalarT >(), MPI_MIN, MPI_COMM_WORLD );

    return global_min;
}

// ============================================================================
// Stage 1: Predictor — low-order upwind + antidiffusive fluxes
// ============================================================================

/// @brief Kokkos kernel for the explicit low-order predictor and antidiffusive flux assembly.
///
/// In a single pass over each cell \f$i\f$ this kernel:
///
///   1. Calls `GeometryHelper::compute_geometry` to obtain the face-normal velocity fluxes
///      \f$\beta_{ij}\f$, the cell volume \f$M_{ii}\f$, and the area-weighted face normals
///      \f$\mathbf{S}_f^{(j)}\f$ for all 6 neighbours.
///
///   2. Computes the **low-order predictor** (first-order upwind + two-point diffusion):
///      \f[
///          T_i^L = T_i^n
///          - \frac{\Delta t}{M_{ii}}
///            \left[
///              \beta_{ii}^+\,T_i^n
///              + \sum_j \beta_{ij}^-\,T_j^n
///              + \sum_j \kappa\,
///                \frac{|\mathbf{S}_f^{(j)}|^2}
///                     {(\mathbf{x}_j-\mathbf{x}_i)\cdot\mathbf{S}_f^{(j)}}
///                (T_i^n - T_j^n)
///            \right],
///      \f]
///      where \f$\beta^+ = \max(\beta,0)\f$ and \f$\beta^- = \min(\beta,0)\f$.
///
///   3. Stores the **pre-scaled antidiffusive flux** for face \f$j\f$:
///      \f[
///          \tilde{f}_{ij} = \frac{\Delta t}{M_{ii}}\,\frac{|\beta_{ij}|}{2}\,(T_i^n - T_j^n).
///      \f]
///      Physical diffusion is **not** included in \f$\tilde{f}_{ij}\f$ — only purely advective
///      antidiffusion enters the Zalesak limiter.
///
/// @note Ghost layers of \f$T^n\f$ (and of `cell_centers_`) must be filled before launch.
/// @note The kernel is Kokkos-portable (CUDA/HIP/OpenMP/Serial).
///
/// @tparam ScalarT  Floating-point scalar type.
template < typename ScalarT >
struct FCTPredictorKernel
{
    using ScalarType = ScalarT;
    using Vec3       = dense::Vec< ScalarT, 3 >;
    using GH         = fct_detail::GeometryHelper< ScalarT >;

    static constexpr int num_neighbors = GH::num_neighbors;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;
    grid::Grid4DDataVec< ScalarT, 3 > cell_centers_;
    grid::Grid4DDataVec< ScalarT, 3 > vel_grid_;

    grid::Grid4DDataScalar< ScalarT >
        T_old_;                             ///< \f$T^n\f$: scalar at time level \f$n\f$ (ghost layers must be filled).
    grid::Grid4DDataScalar< ScalarT > T_L_; ///< \f$T^L\f$: low-order predictor output.
    grid::Grid5DDataScalar< ScalarT >
        antidiff_; ///< \f$\tilde{f}_{ij}\f$: pre-scaled antidiff flux per face, shape \f$[\ldots, 6]\f$.

    ScalarT dt_;          ///< Time step \f$\Delta t\f$.
    ScalarT diffusivity_; ///< Physical diffusivity \f$\kappa \geq 0\f$; set to 0 for pure advection.

    /// @name Optional volumetric source term \f$f\f$ [T/time]
    ///@{
    /// Source values per cell.  A null (default-constructed) view means no source.
    /// The explicit predictor adds \f$\Delta t \cdot f_i\f$ to \f$T_i^L\f$ after the
    /// upwind/diffusion update.  Physical units: same as \f$T\f$ per unit time.
    grid::Grid4DDataScalar< ScalarT > source_;
    bool                              has_source_ = false;
    ///@}

    /// @brief Whether to subtract the discrete divergence error \f$T_i\,\nabla\cdot\mathbf{u}\f$.
    ///
    /// When `true` (default), the correction \f$-T_i\,(\sum_j \beta_{ij})/M_{ii}\f$ is applied,
    /// converting the conservative form \f$\nabla\cdot(\mathbf{u}T)\f$ into the advective form
    /// \f$\mathbf{u}\cdot\nabla T\f$.  This is the physically correct equation for temperature.
    /// When \f$\nabla\cdot\mathbf{u} = 0\f$ the correction is exactly zero and costs nothing.
    bool subtract_divergence_ = true;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        constexpr int cell_offset_x[num_neighbors] = { -1, 1, 0, 0, 0, 0 };
        constexpr int cell_offset_y[num_neighbors] = { 0, 0, -1, 1, 0, 0 };
        constexpr int cell_offset_r[num_neighbors] = { 0, 0, 0, 0, -1, 1 };

        ScalarT beta[num_neighbors];
        ScalarT M_ii;
        Vec3    S_f[num_neighbors];
        GH::compute_geometry( grid_, radii_, cell_centers_, vel_grid_, id, x, y, r, beta, M_ii, S_f );

        const ScalarT T_i = T_old_( id, x, y, r );

        // Low-order predictor: first-order upwind advection + explicit diffusion.
        //
        //   T_L_i = T_i - (dt/M_ii) * [A_upwind*T + D*T]_i
        //
        // Advection (upwind):   contributes beta^+*T_i (diagonal) and beta^-*T_j (off-diagonal).
        // Diffusion (two-point flux): κ*(S_f·S_f / dx·S_f)*(T_i - T_j), not FCT-limited.
        //
        // Antidiffusive fluxes remain purely advective — physical diffusion is not FCT-corrected.

        ScalarT lo_diag   = ScalarT( 0 );
        ScalarT lo_update = ScalarT( 0 );
        ScalarT beta_sum  = ScalarT( 0 ); // Σ_j β_ij = discrete divergence * M_ii

        for ( int n = 0; n < num_neighbors; ++n )
        {
            const int     nx  = x + GH::cell_offset_x( n );
            const int     ny  = y + GH::cell_offset_y( n );
            const int     nr  = r + GH::cell_offset_r( n );
            const ScalarT T_j = T_old_( id, nx, ny, nr );

            // Upwind advection.
            if ( beta[n] >= ScalarT( 0 ) )
                lo_diag += beta[n];
            else
                lo_update += beta[n] * T_j;

            beta_sum += beta[n]; // accumulate for optional divergence correction

            // Explicit diffusion (two-point flux).
            const Vec3 neighbor_center{
                cell_centers_( id, nx, ny, nr, 0 ),
                cell_centers_( id, nx, ny, nr, 1 ),
                cell_centers_( id, nx, ny, nr, 2 ) };
            const Vec3 cell_center{
                cell_centers_( id, x, y, r, 0 ), cell_centers_( id, x, y, r, 1 ), cell_centers_( id, x, y, r, 2 ) };
            const Vec3    dx         = neighbor_center - cell_center;
            const ScalarT denom      = dx.dot( S_f[n] );
            const ScalarT diff_coeff = diffusivity_ * ( S_f[n].dot( S_f[n] ) / denom );
            lo_diag += diff_coeff;         // diagonal: κ*(S·S/dx·S)*T_i
            lo_update -= diff_coeff * T_j; // off-diagonal: -κ*(S·S/dx·S)*T_j  (note: lo_update is subtracted below)

            // Antidiffusive flux: purely advective, not affected by physical diffusion.
            const ScalarT abs_beta      = Kokkos::abs( beta[n] );
            antidiff_( id, x, y, r, n ) = ( dt_ / M_ii ) * ( abs_beta / ScalarT( 2 ) ) * ( T_i - T_j );
        }

        // Divergence correction: subtract T_i * (Σ_j β_ij) / M_ii * dt from T_L.
        // This converts the conservative form ∇·(uT) to the advective form u·∇T.
        // Equivalent to reducing lo_diag by beta_sum (the inflow part cancels with lo_update).
        if ( subtract_divergence_ )
            lo_diag -= beta_sum;

        ScalarT T_L_val = T_i - ( dt_ / M_ii ) * ( lo_diag * T_i + lo_update );

        // Volumetric source term: T^L += dt * f_i  (f_i in [T/time]).
        if ( has_source_ )
        {
            T_L_val += dt_ * source_( id, x, y, r );
        }

        T_L_( id, x, y, r ) = T_L_val;
    }
};

/// @brief Stage 1: low-order predictor + antidiffusive flux computation.
///
/// Ghost layers of `T_old` are exchanged via MPI before the kernel runs.
/// On exit:
///   - `bufs.T_L`      holds \f$T^L\f$ (first-order upwind + explicit diffusion at \f$t+\Delta t\f$).
///   - `bufs.antidiff` holds \f$\tilde{f}_{ij} = (\Delta t / M_{ii})\,f_{ij}\f$ for every face.
///
/// @param domain       Distributed domain (used for MPI ghost-layer exchange).
/// @param T_old        Transported scalar \f$T^n\f$ at time level \f$n\f$.
/// @param vel          Q1 velocity field \f$\mathbf{u}\f$ (nodal, read-only).
/// @param cell_centers Pre-computed cell centres (with ghost layers filled via `initialize_cell_centers`).
/// @param grid         Lateral node coordinates of the unit-sphere surface.
/// @param radii        Radial shell radii.
/// @param dt           Time step \f$\Delta t\f$.
/// @param bufs         Pre-allocated FCT scratch arrays (updated in-place).
/// @param diffusivity          Physical diffusivity \f$\kappa\f$ (default 0 = pure advection).
/// @param source               Volumetric source term \f$f\f$ [T/time] per cell.  A
///                             default-constructed (null) view means no source.  Added as
///                             \f$\Delta t \cdot f_i\f$ to the low-order predictor.
/// @param subtract_divergence  When `true` (default), subtract \f$T_i\,(\sum_j \beta_{ij})/M_{ii}\f$
///                             from the predictor, converting \f$\nabla\cdot(\mathbf{u}T)\f$ to
///                             \f$\mathbf{u}\cdot\nabla T\f$.  This is always correct for
///                             temperature: when \f$\nabla\cdot\mathbf{u}=0\f$ the correction
///                             vanishes; when \f$\nabla\cdot\mathbf{u}\neq 0\f$ it removes an
///                             unphysical source term.
template < typename ScalarT >
void fct_predictor(
    const grid::shell::DistributedDomain&    domain,
    const linalg::VectorFVScalar< ScalarT >& T_old,
    const linalg::VectorQ1Vec< ScalarT, 3 >& vel,
    const grid::Grid4DDataVec< ScalarT, 3 >& cell_centers,
    const grid::Grid3DDataVec< ScalarT, 3 >& grid,
    const grid::Grid2DDataScalar< ScalarT >& radii,
    const ScalarT                            dt,
    FVFCTBuffers< ScalarT >&                 bufs,
    const ScalarT                            diffusivity         = ScalarT( 0 ),
    const grid::Grid4DDataScalar< ScalarT >& source              = {},
    const bool                               subtract_divergence = true )
{
    util::Timer timer_predictor( "fct_predictor" );

    {
        util::Timer timer_comm( "fct_predictor_comm" );
        communication::shell::update_fv_ghost_layers( domain, T_old.grid_data(), bufs.ghost_T );
    }

    FCTPredictorKernel< ScalarT > kernel{
        .grid_                = grid,
        .radii_               = radii,
        .cell_centers_        = cell_centers,
        .vel_grid_            = vel.grid_data(),
        .T_old_               = T_old.grid_data(),
        .T_L_                 = bufs.T_L,
        .antidiff_            = bufs.antidiff,
        .dt_                  = dt,
        .diffusivity_         = diffusivity,
        .source_              = source,
        .has_source_          = ( source.data() != nullptr ),
        .subtract_divergence_ = subtract_divergence,
    };

    {
        util::Timer timer_kernel( "fct_predictor_kernel" );
        Kokkos::parallel_for(
            "fct_predictor", grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ), kernel );
        Kokkos::fence();
    }
}

// ============================================================================
// Stage 2: Limiter — compute Zalesak R+ / R- correction factors
// ============================================================================

/// @brief Kokkos kernel that computes the Zalesak nodal correction factors \f$R_i^+\f$ and
///        \f$R_i^-\f$ from the low-order predictor \f$T^L\f$ and the antidiffusive fluxes.
///
/// Per cell \f$i\f$ (summing over all 6 neighbours \f$j \in \mathcal{N}(i)\f$):
///
/// **Positive/negative flux sums:**
/// \f[
///     P_i^+ = \sum_{j:\,\tilde{f}_{ij} > 0} \tilde{f}_{ij}, \qquad
///     P_i^- = \sum_{j:\,\tilde{f}_{ij} < 0} \tilde{f}_{ij}.
/// \f]
///
/// **Local extrema of the low-order solution:**
/// \f[
///     T_i^{\max} = \max(T_i^L,\;T_j^L\;\forall j \in \mathcal{N}(i)), \qquad
///     T_i^{\min} = \min(T_i^L,\;T_j^L\;\forall j \in \mathcal{N}(i)).
/// \f]
///
/// **Room to grow/shrink:**
/// \f[
///     Q_i^+ = T_i^{\max} - T_i^L \geq 0, \qquad
///     Q_i^- = T_i^{\min} - T_i^L \leq 0.
/// \f]
///
/// **Correction factors** (Zalesak 1979, eq. 13–14):
/// \f[
///     R_i^+ = \begin{cases} \min\!\left(1,\;\dfrac{Q_i^+}{P_i^+}\right) & P_i^+ > 0 \\ 1 & \text{otherwise} \end{cases},
///     \qquad
///     R_i^- = \begin{cases} \min\!\left(1,\;\dfrac{Q_i^-}{P_i^-}\right) & P_i^- < 0 \\ 1 & \text{otherwise} \end{cases}.
/// \f]
///
/// @note Ghost layers of \f$T^L\f$ must be filled before launch so that
///       \f$T_j^L\f$ for boundary-adjacent cells is available.
/// @tparam ScalarT  Floating-point scalar type.
template < typename ScalarT >
struct FCTLimiterKernel
{
    using GH                           = fct_detail::GeometryHelper< ScalarT >;
    static constexpr int num_neighbors = GH::num_neighbors;

    grid::Grid4DDataScalar< ScalarT > T_L_;      ///< \f$T^L\f$ with ghost layers filled.
    grid::Grid5DDataScalar< ScalarT > antidiff_; ///< \f$\tilde{f}_{ij}\f$: pre-scaled antidiff fluxes.
    grid::Grid4DDataScalar< ScalarT > R_plus_;   ///< Output: \f$R_i^+\f$ correction factor.
    grid::Grid4DDataScalar< ScalarT > R_minus_;  ///< Output: \f$R_i^-\f$ correction factor.

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        constexpr int cell_offset_x[num_neighbors] = { -1, 1, 0, 0, 0, 0 };
        constexpr int cell_offset_y[num_neighbors] = { 0, 0, -1, 1, 0, 0 };
        constexpr int cell_offset_r[num_neighbors] = { 0, 0, 0, 0, -1, 1 };

        const ScalarT T_L_i = T_L_( id, x, y, r );

        ScalarT P_plus  = ScalarT( 0 );
        ScalarT P_minus = ScalarT( 0 );
        ScalarT T_max   = T_L_i;
        ScalarT T_min   = T_L_i;

        for ( int n = 0; n < num_neighbors; ++n )
        {
            const ScalarT f_ij = antidiff_( id, x, y, r, n );
            if ( f_ij > ScalarT( 0 ) ) P_plus  += f_ij;
            else                        P_minus += f_ij;

            const ScalarT T_L_j = T_L_(
                id,
                x + GH::cell_offset_x( n ),
                y + GH::cell_offset_y( n ),
                r + GH::cell_offset_r( n ) );
            T_max = Kokkos::max( T_max, T_L_j );
            T_min = Kokkos::min( T_min, T_L_j );
        }

        const ScalarT Q_plus  = T_max - T_L_i; // ≥ 0
        const ScalarT Q_minus = T_min - T_L_i; // ≤ 0

        R_plus_( id, x, y, r ) =
            ( P_plus > ScalarT( 0 ) ) ? Kokkos::min( ScalarT( 1 ), Q_plus / P_plus ) : ScalarT( 1 );
        R_minus_( id, x, y, r ) =
            ( P_minus < ScalarT( 0 ) ) ? Kokkos::min( ScalarT( 1 ), Q_minus / P_minus ) : ScalarT( 1 );
    }
};

/// @brief Stage 2: compute Zalesak \f$R^+\f$/\f$R^-\f$ correction factors.
///
/// Ghost layers of `bufs.T_L` are exchanged via MPI so that the neighbourhood
/// min/max stencil is correct for subdomain-boundary cells.  Then
/// `FCTLimiterKernel` is launched, and finally ghost layers of `bufs.R_plus`
/// and `bufs.R_minus` are exchanged so that Stage 3 can read neighbour factors.
///
/// This function can be called **multiple times** in a non-linear iteration loop
/// with a fixed `bufs.antidiff` computed once from \f$T^n\f$ but an updated
/// `bufs.T_L` iterate.
///
/// @param domain  Distributed domain (used for ghost-layer exchange).
/// @param bufs    FCT scratch arrays; reads `T_L` and `antidiff`, writes `R_plus` and `R_minus`.
template < typename ScalarT >
void fct_limiter( const grid::shell::DistributedDomain& domain, FVFCTBuffers< ScalarT >& bufs )
{
    util::Timer timer_limiter( "fct_limiter" );

    {
        util::Timer timer_comm( "fct_limiter_comm_tl" );
        communication::shell::update_fv_ghost_layers( domain, bufs.T_L, bufs.ghost_T );
    }

    FCTLimiterKernel< ScalarT > kernel{
        .T_L_      = bufs.T_L,
        .antidiff_ = bufs.antidiff,
        .R_plus_   = bufs.R_plus,
        .R_minus_  = bufs.R_minus,
    };

    {
        util::Timer timer_kernel( "fct_limiter_kernel" );
        Kokkos::parallel_for(
            "fct_limiter", grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ), kernel );
        Kokkos::fence();
    }

    // R+/R- ghost layers must be filled before the correction kernel reads neighbours.
    {
        util::Timer timer_comm_r( "fct_limiter_comm_r" );
        communication::shell::update_fv_ghost_layers( domain, bufs.R_plus, bufs.ghost_R_plus );
        communication::shell::update_fv_ghost_layers( domain, bufs.R_minus, bufs.ghost_R_minus );
    }
}

// ============================================================================
// Stage 3: Correction — apply limited antidiffusive fluxes to T_L
// ============================================================================

/// @brief Kokkos kernel that applies the Zalesak-limited antidiffusive correction to \f$T^L\f$.
///
/// For each cell \f$i\f$ and face \f$j\f$, the symmetric limited flux factor is
/// \f[
///     \alpha_{ij} =
///     \begin{cases}
///       \min(R_i^+,\; R_j^-) & \tilde{f}_{ij} > 0 \quad (\text{flux increases } T_i), \\
///       \min(R_i^-,\; R_j^+) & \tilde{f}_{ij} < 0 \quad (\text{flux decreases } T_i).
///     \end{cases}
/// \f]
/// Note that \f$\alpha_{ji} = \alpha_{ij}\f$ by construction (symmetry), which ensures
/// conservation: what is added to cell \f$i\f$ from face \f$j\f$ is subtracted from
/// cell \f$j\f$ on the same face.
///
/// The corrected solution is
/// \f[
///     T_i^{n+1} = T_i^L + \sum_j \alpha_{ij}\,\tilde{f}_{ij}.
/// \f]
///
/// @note Ghost layers of \f$R^+\f$ and \f$R^-\f$ must be filled (done by `fct_limiter`).
/// @tparam ScalarT  Floating-point scalar type.
template < typename ScalarT >
struct FCTCorrectionKernel
{
    using GH                           = fct_detail::GeometryHelper< ScalarT >;
    static constexpr int num_neighbors = GH::num_neighbors;

    grid::Grid4DDataScalar< ScalarT > T_L_;      ///< \f$T^L\f$: low-order predictor.
    grid::Grid5DDataScalar< ScalarT > antidiff_; ///< \f$\tilde{f}_{ij}\f$: pre-scaled antidiff fluxes.
    grid::Grid4DDataScalar< ScalarT > R_plus_;   ///< \f$R^+\f$ correction factor (ghost layers filled).
    grid::Grid4DDataScalar< ScalarT > R_minus_;  ///< \f$R^-\f$ correction factor (ghost layers filled).
    grid::Grid4DDataScalar< ScalarT > T_new_;    ///< Output: \f$T^{n+1}\f$.

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        constexpr int cell_offset_x[num_neighbors] = { -1, 1, 0, 0, 0, 0 };
        constexpr int cell_offset_y[num_neighbors] = { 0, 0, -1, 1, 0, 0 };
        constexpr int cell_offset_r[num_neighbors] = { 0, 0, 0, 0, -1, 1 };

        ScalarT correction = ScalarT( 0 );

        const ScalarT R_plus_i  = R_plus_( id, x, y, r );
        const ScalarT R_minus_i = R_minus_( id, x, y, r );

        for ( int n = 0; n < num_neighbors; ++n )
        {
            const int jx = x + GH::cell_offset_x( n );
            const int jy = y + GH::cell_offset_y( n );
            const int jr = r + GH::cell_offset_r( n );

            const ScalarT f_ij      = antidiff_( id, x, y, r, n );
            const ScalarT R_plus_j  = R_plus_( id, jx, jy, jr );
            const ScalarT R_minus_j = R_minus_( id, jx, jy, jr );

            ScalarT alpha;
            if ( f_ij > ScalarT( 0 ) )
                alpha = Kokkos::min( R_plus_i, R_minus_j );
            else
                alpha = Kokkos::min( R_minus_i, R_plus_j );

            correction += alpha * f_ij;
        }

        T_new_( id, x, y, r ) = T_L_( id, x, y, r ) + correction;
    }
};

/// @brief Stage 3: apply the Zalesak-limited antidiffusive correction.
///
/// Reads `bufs.T_L`, `bufs.antidiff`, `bufs.R_plus`, `bufs.R_minus` (all read-only).
/// Writes \f$T^{n+1} = T^L + \sum_j \alpha_{ij}\,\tilde{f}_{ij}\f$ into `T_new`.
///
/// @pre   `fct_limiter` must have been called first so that ghost layers of \f$R^\pm\f$ are current.
/// @param domain  Distributed domain.
/// @param T_new   Output: high-order corrected scalar at \f$t_{n+1}\f$.
/// @param bufs    FCT scratch arrays (reads `T_L`, `antidiff`, `R_plus`, `R_minus`).
template < typename ScalarT >
void fct_correction(
    const grid::shell::DistributedDomain& domain,
    linalg::VectorFVScalar< ScalarT >&    T_new,
    FVFCTBuffers< ScalarT >&              bufs )
{
    util::Timer timer_correction( "fct_correction" );

    FCTCorrectionKernel< ScalarT > kernel{
        .T_L_      = bufs.T_L,
        .antidiff_ = bufs.antidiff,
        .R_plus_   = bufs.R_plus,
        .R_minus_  = bufs.R_minus,
        .T_new_    = T_new.grid_data(),
    };

    {
        util::Timer timer_kernel( "fct_correction_kernel" );
        Kokkos::parallel_for(
            "fct_correction", grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ), kernel );
        Kokkos::fence();
    }
}

// ============================================================================
// Convenience wrapper: single explicit FCT step
// ============================================================================

/// @brief One complete explicit FCT advection–diffusion timestep.
///
/// Executes the three stages in sequence:
///   1. `fct_predictor`  — low-order upwind + diffusion predictor \f$T^L\f$ and antidiff fluxes.
///   2. `fct_limiter`    — Zalesak \f$R^\pm\f$ factors.
///   3. `fct_correction` — limited antidiffusive correction \f$T^{n+1} = T^L + \sum_j \alpha_{ij}\,\tilde{f}_{ij}\f$.
///
/// The scheme is **LED** (local extremum diminishing) and **conservation-consistent**:
/// the limited fluxes are antisymmetric, so cell integrals are preserved up to boundary terms.
///
/// For a non-linear iteration (e.g. defect-correction), call the three stages individually
/// and repeat stages 2–3 with an updated \f$T^L\f$.
///
/// **Stability:** requires \f$\mathrm{CFL} = \Delta t\,\max_i(\sum_j \beta_{ij}^+)/M_{ii} < 1\f$.
///
/// @param domain       Distributed domain.
/// @param T            Transported scalar — \f$T^n\f$ on entry, \f$T^{n+1}\f$ on exit.
/// @param vel          Q1 nodal velocity field \f$\mathbf{u}\f$.
/// @param cell_centers Pre-computed cell centres (ghost layers populated via `initialize_cell_centers`).
/// @param grid         Lateral node coordinates of the unit-sphere surface.
/// @param radii        Radial shell radii.
/// @param dt           Time step \f$\Delta t\f$.
/// @param bufs         Pre-allocated FCT scratch arrays.
/// @param diffusivity          Physical diffusivity \f$\kappa \geq 0\f$ (default 0 = pure advection).
/// @param source               Volumetric source term \f$f\f$ [T/time]; null view = no source.
/// @param subtract_divergence  Subtract discrete divergence error (default `true`); see `fct_predictor`.
/// @param boundary_mask        Node-based boundary flag array (Q1 layout).  When provided (non-null),
///                             Dirichlet BCs are enforced on \f$T^L\f$ **between the predictor and
///                             limiter** so the Zalesak \f$R^\pm\f$ factors see the correct boundary
///                             values.  Default: null (no enforcement).
/// @param bcs                  Prescribed boundary values.  Ignored when `boundary_mask` is null.
template < typename ScalarT >
void fct_explicit_step(
    const grid::shell::DistributedDomain&                           domain,
    linalg::VectorFVScalar< ScalarT >&                              T,
    const linalg::VectorQ1Vec< ScalarT, 3 >&                        vel,
    const grid::Grid4DDataVec< ScalarT, 3 >&                        cell_centers,
    const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
    const grid::Grid2DDataScalar< ScalarT >&                        radii,
    const ScalarT                                                   dt,
    FVFCTBuffers< ScalarT >&                                        bufs,
    const ScalarT                                                   diffusivity         = ScalarT( 0 ),
    const grid::Grid4DDataScalar< ScalarT >&                        source              = {},
    const bool                                                      subtract_divergence = true,
    const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask       = {},
    const DirichletBCs< ScalarT >&                                  bcs                 = {} )
{
    util::Timer timer_fct( "fct_explicit_step" );
    fct_predictor( domain, T, vel, cell_centers, grid, radii, dt, bufs, diffusivity, source, subtract_divergence );

    // Enforce Dirichlet BCs on T_L before the limiter so that R+/R- are computed
    // relative to the correct boundary values.  Without this, the predictor can
    // move boundary cells away from the prescribed value (due to discrete divergence
    // error in the velocity), and the antidiffusive correction will then "confirm"
    // that wrong value — leading to oscillations near the boundary.
    if ( boundary_mask.extent( 0 ) > 0 )
        apply_dirichlet_bcs( bufs.T_L, boundary_mask, bcs, domain );

    fct_limiter( domain, bufs );
    fct_correction( domain, T, bufs );
}

// ============================================================================
// Convenience wrapper: plain explicit first-order upwind (no FCT correction)
// ============================================================================

/// @brief One explicit first-order upwind advection–diffusion timestep (no FCT correction).
///
/// Equivalent to calling `fct_predictor` and then copying \f$T^L \to T\f$.
/// The result is the maximally diffusive (but stable) low-order solution.  Use this as a
/// baseline or whenever sharp-gradient preservation is not required.
///
/// Physical diffusion (two-point flux) is included when `diffusivity > 0`.
///
/// **Stability:** requires \f$\mathrm{CFL} = \Delta t\,\max_i(\sum_j \beta_{ij}^+)/M_{ii} < 1\f$.
///
/// @param domain       Distributed domain.
/// @param T            Transported scalar — \f$T^n\f$ on entry, \f$T^{n+1}\f$ on exit.
/// @param vel          Q1 nodal velocity field \f$\mathbf{u}\f$.
/// @param cell_centers Pre-computed cell centres (ghost layers populated via `initialize_cell_centers`).
/// @param grid         Lateral node coordinates of the unit-sphere surface.
/// @param radii        Radial shell radii.
/// @param dt           Time step \f$\Delta t\f$.
/// @param bufs         Pre-allocated FCT scratch arrays (uses `T_L` and `ghost_T`).
/// @param diffusivity          Physical diffusivity \f$\kappa \geq 0\f$ (default 0 = pure advection).
/// @param source               Optional volumetric source term \f$f\f$ [T/time] (default: none).
/// @param subtract_divergence  Subtract discrete divergence error (default `true`); see `fct_predictor`.
template < typename ScalarT >
void upwind_explicit_step(
    const grid::shell::DistributedDomain&    domain,
    linalg::VectorFVScalar< ScalarT >&       T,
    const linalg::VectorQ1Vec< ScalarT, 3 >& vel,
    const grid::Grid4DDataVec< ScalarT, 3 >& cell_centers,
    const grid::Grid3DDataVec< ScalarT, 3 >& grid,
    const grid::Grid2DDataScalar< ScalarT >& radii,
    const ScalarT                            dt,
    FVFCTBuffers< ScalarT >&                 bufs,
    const ScalarT                            diffusivity         = ScalarT( 0 ),
    const grid::Grid4DDataScalar< ScalarT >& source              = {},
    const bool                               subtract_divergence = true )
{
    util::Timer timer_upwind( "upwind_explicit_step" );
    fct_predictor( domain, T, vel, cell_centers, grid, radii, dt, bufs, diffusivity, source, subtract_divergence );
    Kokkos::deep_copy( T.grid_data(), bufs.T_L );
}

// ============================================================================
// Stage 1 (semi-implicit variant): antidiffusive fluxes only
// ============================================================================

/// @brief Kokkos kernel that computes only the pre-scaled antidiffusive fluxes from \f$T^n\f$.
///
/// In the semi-implicit FCT scheme the low-order predictor \f$T^L\f$ is provided by an
/// external implicit solve (see `fct_semiimplicit_step`).  This kernel therefore skips the
/// upwind update and only stores the antidiffusive fluxes needed by the Zalesak limiter:
/// \f[
///     \tilde{f}_{ij} = \frac{\Delta t}{M_{ii}}\,\frac{|\beta_{ij}|}{2}\,(T_i^n - T_j^n).
/// \f]
/// The formula is identical to the antidiffusive part of `FCTPredictorKernel`.
///
/// @note Ghost layers of \f$T^n\f$ must be filled before launch.
/// @tparam ScalarT  Floating-point scalar type.
template < typename ScalarT >
struct FCTAntidiffKernel
{
    using ScalarType = ScalarT;
    using GH         = fct_detail::GeometryHelper< ScalarT >;

    static constexpr int num_neighbors = GH::num_neighbors;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;
    grid::Grid4DDataVec< ScalarT, 3 > cell_centers_;
    grid::Grid4DDataVec< ScalarT, 3 > vel_grid_;
    grid::Grid4DDataScalar< ScalarT > T_old_;    ///< \f$T^n\f$ with ghost layers filled.
    grid::Grid5DDataScalar< ScalarT > antidiff_; ///< Output: \f$\tilde{f}_{ij}\f$, shape \f$[\ldots, 6]\f$.
    ScalarT                           dt_;       ///< Time step \f$\Delta t\f$.

    KOKKOS_INLINE_FUNCTION
    void operator()( const int id, const int x, const int y, const int r ) const
    {
        ScalarT           beta[num_neighbors];
        ScalarT           M_ii;
        typename GH::Vec3 S_f[num_neighbors]; // not used here; required by shared compute_geometry
        GH::compute_geometry( grid_, radii_, cell_centers_, vel_grid_, id, x, y, r, beta, M_ii, S_f );

        const ScalarT T_i = T_old_( id, x, y, r );
        for ( int n = 0; n < num_neighbors; ++n )
        {
            const ScalarT T_j = T_old_(
                id,
                x + GH::cell_offset_x( n ),
                y + GH::cell_offset_y( n ),
                r + GH::cell_offset_r( n ) );
            const ScalarT abs_beta          = Kokkos::abs( beta[n] );
            antidiff_( id, x, y, r, n ) = ( dt_ / M_ii ) * ( abs_beta / ScalarT( 2 ) ) * ( T_i - T_j );
        }
    }
};

/// @brief Computes pre-scaled antidiffusive fluxes \f$\tilde{f}_{ij}\f$ from \f$T^n\f$
///        (semi-implicit FCT stage 1).
///
/// Ghost layers of `T_old` are exchanged via MPI before the kernel runs.
/// On exit `bufs.antidiff` holds \f$\tilde{f}_{ij} = (\Delta t / M_{ii})\,f_{ij}\f$ for
/// every real cell and all 6 faces.
///
/// @param domain       Distributed domain (used for MPI ghost-layer exchange).
/// @param T_old        Transported scalar \f$T^n\f$ at time level \f$n\f$.
/// @param vel          Q1 nodal velocity field \f$\mathbf{u}\f$.
/// @param cell_centers Pre-computed cell centres (ghost layers populated via `initialize_cell_centers`).
/// @param grid         Lateral node coordinates of the unit-sphere surface.
/// @param radii        Radial shell radii.
/// @param dt           Time step \f$\Delta t\f$.
/// @param bufs         FCT scratch arrays; only `antidiff` and `ghost_T` are written.
template < typename ScalarT >
void fct_antidiff(
    const grid::shell::DistributedDomain&    domain,
    const linalg::VectorFVScalar< ScalarT >& T_old,
    const linalg::VectorQ1Vec< ScalarT, 3 >& vel,
    const grid::Grid4DDataVec< ScalarT, 3 >& cell_centers,
    const grid::Grid3DDataVec< ScalarT, 3 >& grid,
    const grid::Grid2DDataScalar< ScalarT >& radii,
    const ScalarT                            dt,
    FVFCTBuffers< ScalarT >&                 bufs )
{
    communication::shell::update_fv_ghost_layers( domain, T_old.grid_data(), bufs.ghost_T );

    FCTAntidiffKernel< ScalarT > kernel{
        .grid_         = grid,
        .radii_        = radii,
        .cell_centers_ = cell_centers,
        .vel_grid_     = vel.grid_data(),
        .T_old_        = T_old.grid_data(),
        .antidiff_     = bufs.antidiff,
        .dt_           = dt,
    };

    Kokkos::parallel_for(
        "fct_antidiff", grid::shell::local_domain_md_range_policy_cells_fv_skip_ghost_layers( domain ), kernel );
    Kokkos::fence();
}

// ============================================================================
// Semi-implicit FCT step
// ============================================================================

/// @brief One semi-implicit FCT advection–diffusion timestep.
///
/// The semi-implicit scheme decouples the implicit solve from the flux-correction step,
/// removing the CFL restriction while retaining the LED property.  The caller is
/// responsible for providing the implicit low-order predictor \f$T^L\f$, obtained by
/// solving the backward-Euler system
/// \f[
///     \bigl(M + \Delta t\,A_{\mathrm{upw}} + \Delta t\,D\bigr)\,T^L = M\,T^n,
/// \f]
/// where \f$A_{\mathrm{upw}}\f$ is the upwind advection matrix and \f$D\f$ is the
/// two-point diffusion matrix.  See `UnsteadyAdvectionDiffusion` in
/// `advection_diffusion.hpp` for the concrete operator and FGMRES for the solver.
///
/// Steps executed internally:
///   1. `fct_antidiff`: ghost-update \f$T^n\f$, compute \f$\tilde{f}_{ij}\f$ from \f$T^n\f$.
///   2. Copy the externally provided \f$T^L\f$ into `bufs.T_L`.
///   3. `fct_limiter`: ghost-update \f$T^L\f$, compute \f$R^\pm\f$, ghost-update \f$R^\pm\f$.
///   4. `fct_correction`: \f$T^{n+1} = T^L + \sum_j \alpha_{ij}\,\tilde{f}_{ij}\f$ → written into \f$T\f$.
///
/// **Stability:** unconditionally stable for the low-order predictor; the Zalesak correction
/// cannot introduce new extrema beyond those already present in \f$T^L\f$.
///
/// @param domain       Distributed domain.
/// @param T            Transported scalar — \f$T^n\f$ on entry, \f$T^{n+1}\f$ on exit.
/// @param T_L          Implicit low-order predictor satisfying the backward-Euler system above.
/// @param vel          Q1 nodal velocity field \f$\mathbf{u}\f$ (needed only for antidiff geometry).
/// @param cell_centers Pre-computed cell centres (ghost layers populated via `initialize_cell_centers`).
/// @param grid         Lateral node coordinates of the unit-sphere surface.
/// @param radii        Radial shell radii.
/// @param dt           Time step \f$\Delta t\f$.
/// @param bufs         Pre-allocated FCT scratch arrays.
template < typename ScalarT >
void fct_semiimplicit_step(
    const grid::shell::DistributedDomain&    domain,
    linalg::VectorFVScalar< ScalarT >&       T,
    const linalg::VectorFVScalar< ScalarT >& T_L,
    const linalg::VectorQ1Vec< ScalarT, 3 >& vel,
    const grid::Grid4DDataVec< ScalarT, 3 >& cell_centers,
    const grid::Grid3DDataVec< ScalarT, 3 >& grid,
    const grid::Grid2DDataScalar< ScalarT >& radii,
    const ScalarT                            dt,
    FVFCTBuffers< ScalarT >&                 bufs )
{
    // Stage 1: antidiffusive fluxes from T^n.
    fct_antidiff( domain, T, vel, cell_centers, grid, radii, dt, bufs );

    // Stage 2: install the implicit predictor as the low-order solution.
    Kokkos::deep_copy( bufs.T_L, T_L.grid_data() );

    // Stage 3: Zalesak limiter — ghost T_L, compute R+/R-, ghost R+/R-.
    fct_limiter( domain, bufs );

    // Stage 4: apply limited correction.  T^{n+1} is written into T.
    fct_correction( domain, T, bufs );
}

} // namespace terra::fv::hex::operators
