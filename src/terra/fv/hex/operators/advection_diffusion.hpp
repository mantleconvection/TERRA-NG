
#pragma once
#include "communication/shell/fv_communication.hpp"
#include "fv/hex/operators/geometry_helper.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fv::hex::operators {

template < typename ScalarT >
class UnsteadyAdvectionDiffusion
{
  public:
    using SrcVectorType = linalg::VectorFVScalar< ScalarT >;
    using DstVectorType = linalg::VectorFVScalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    using GH   = detail::GeometryHelper< ScalarT >;
    using Vec3 = dense::Vec< ScalarT, 3 >;

    static constexpr int num_velocity_components = 3;
    static constexpr int num_neighbors           = GH::num_neighbors;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataVec< ScalarT, 3 >                        cell_centers_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > boundary_mask_;

    linalg::VectorQ1Vec< ScalarT, 3 > velocity_;

    ScalarT diffusivity_;
    ScalarT dt_;
    bool    subtract_divergence_;

    communication::shell::FVGhostLayerBuffers< ScalarType > ghost_bufs_;

    grid::Grid4DDataScalar< ScalarType >                       src_;
    grid::Grid4DDataScalar< ScalarType >                       dst_;
    grid::Grid4DDataVec< ScalarType, num_velocity_components > vel_grid_;

  public:
    UnsteadyAdvectionDiffusion(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataVec< ScalarT, 3 >&                        cell_centers,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask,
        const linalg::VectorQ1Vec< ScalarT, num_velocity_components >&  velocity,
        const ScalarT                                                   diffusivity,
        const ScalarT                                                   dt,
        const bool                                                      subtract_divergence = true )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , cell_centers_( cell_centers )
    , boundary_mask_( boundary_mask )
    , velocity_( velocity )
    , diffusivity_( diffusivity )
    , dt_( dt )
    , subtract_divergence_( subtract_divergence )
    , ghost_bufs_( domain )
    {}

    ScalarT&       dt() { return dt_; }
    const ScalarT& dt() const { return dt_; }

    /// @brief Compute the implicit-step right-hand side \f$b = M\,T^n + \Delta t\,M\,f\f$.
    ///
    /// For the semi-implicit FCT loop the linear system to solve is
    /// \f$(M + \Delta t\,A)\,T^L = M\,T^n\f$ (no source), or
    /// \f$(M + \Delta t\,A)\,T^L = M\,T^n + \Delta t\,M\,f\f$ with a source term.
    /// This method forms the right-hand side using the same Felippa 3×2 quadrature as
    /// `Mass::apply_impl`, so no separate `Mass` operator is needed when a source is present.
    ///
    /// @param T_old  Current scalar field \f$T^n\f$.
    /// @param rhs    [out] Receives \f$M\,T^n\f$ (or \f$M\,T^n + \Delta t\,M\,f\f$).
    /// @param source Optional volumetric source term \f$f\f$ [T/time] (default: none).
    void compute_rhs(
        const SrcVectorType&                     T_old,
        DstVectorType&                           rhs,
        const grid::Grid4DDataScalar< ScalarT >& source = {} )
    {
        const bool has_source = ( source.data() != nullptr );

        const auto T_old_grid = T_old.grid_data();
        const auto rhs_grid   = rhs.grid_data();
        const auto grid       = grid_;
        const auto radii      = radii_;
        const auto dt         = dt_;

        Kokkos::parallel_for(
            "compute_rhs",
            Kokkos::MDRangePolicy(
                { 0, 1, 1, 1 },
                { T_old_grid.extent( 0 ),
                  T_old_grid.extent( 1 ) - 1,
                  T_old_grid.extent( 2 ) - 1,
                  T_old_grid.extent( 3 ) - 1 } ),
            KOKKOS_LAMBDA( const int id, const int x, const int y, const int r ) {
                dense::Vec< ScalarT, 3 > wedge_phy_surf[fe::wedge::num_wedges_per_hex_cell]
                                                       [fe::wedge::num_nodes_per_wedge_surface] = {};
                fe::wedge::wedge_surface_physical_coords( wedge_phy_surf, grid, id, x - 1, y - 1 );

                const ScalarT r_1 = radii( id, r - 1 );
                const ScalarT r_2 = radii( id, r );

                constexpr auto           num_quad_points = fe::wedge::quadrature::quad_felippa_3x2_num_quad_points;
                dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
                ScalarT                  quad_weights[num_quad_points];
                fe::wedge::quadrature::quad_felippa_3x2_quad_points( quad_points );
                fe::wedge::quadrature::quad_felippa_3x2_quad_weights( quad_weights );

                ScalarT M_ii = ScalarT( 0 );
                for ( int wedge = 0; wedge < fe::wedge::num_wedges_per_hex_cell; ++wedge )
                    for ( int q = 0; q < num_quad_points; ++q )
                        M_ii += Kokkos::abs( fe::wedge::jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] ).det() ) *
                                quad_weights[q];

                ScalarT val = M_ii * T_old_grid( id, x, y, r );
                if ( has_source )
                    val += dt * M_ii * source( id, x, y, r );
                rhs_grid( id, x, y, r ) = val;
            } );

        Kokkos::fence();
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        communication::shell::update_fv_ghost_layers( domain_, src.grid_data(), ghost_bufs_ );

        src_      = src.grid_data();
        dst_      = dst.grid_data();
        vel_grid_ = velocity_.grid_data();

        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy(
                { 0, 1, 1, 1 },
                { src.grid_data().extent( 0 ),
                  src.grid_data().extent( 1 ) - 1,
                  src.grid_data().extent( 2 ) - 1,
                  src.grid_data().extent( 3 ) - 1 } ),
            *this );

        Kokkos::fence();
    }

  public:
    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        constexpr int cell_offset_x[num_neighbors] = { -1, 1, 0, 0, 0, 0 };
        constexpr int cell_offset_y[num_neighbors] = { 0, 0, -1, 1, 0, 0 };
        constexpr int cell_offset_r[num_neighbors] = { 0, 0, 0, 0, -1, 1 };

        ScalarT beta[num_neighbors];
        ScalarT M_ii;
        Vec3    S_f[num_neighbors];
        GH::compute_geometry(
            grid_, radii_, cell_centers_, vel_grid_, local_subdomain_id, x_cell, y_cell, r_cell, beta, M_ii, S_f );

        // Assemble (M + dt*A_upwind) stencil.
        ScalarT AD_ii                = ScalarT( 0 );
        ScalarT AD_ij[num_neighbors] = {};
        ScalarT beta_sum             = ScalarT( 0 ); // Σ_j β_ij, for optional divergence correction

        for ( int n = 0; n < num_neighbors; ++n )
        {
            // Advection (first-order upwind).
            if ( beta[n] >= ScalarT( 0 ) )
                AD_ii += beta[n];
            else
                AD_ij[n] += beta[n]; // inflow: negative beta → neighbour contribution

            beta_sum += beta[n];

            // Diffusion (two-point flux, cell-to-cell vector from precomputed centers).
            if ( diffusivity_ != ScalarT( 0 ) )
            {
                const int  nx = x_cell + GH::cell_offset_x( n );
                const int  ny = y_cell + GH::cell_offset_y( n );
                const int  nr = r_cell + GH::cell_offset_r( n );
                const Vec3 neighbor_center{
                    cell_centers_( local_subdomain_id, nx, ny, nr, 0 ),
                    cell_centers_( local_subdomain_id, nx, ny, nr, 1 ),
                    cell_centers_( local_subdomain_id, nx, ny, nr, 2 ) };
                const Vec3 cell_center{
                    cell_centers_( local_subdomain_id, x_cell, y_cell, r_cell, 0 ),
                    cell_centers_( local_subdomain_id, x_cell, y_cell, r_cell, 1 ),
                    cell_centers_( local_subdomain_id, x_cell, y_cell, r_cell, 2 ) };
                const Vec3       dx                = neighbor_center - cell_center;
                const ScalarType denom             = dx.dot( S_f[n] );
                const ScalarType offdiag_diffusion = -diffusivity_ * ( S_f[n].dot( S_f[n] ) / denom );
                AD_ii -= offdiag_diffusion; // diagonal gets the negative of offdiag
                AD_ij[n] += offdiag_diffusion;
            }
        }

        // Divergence correction: subtract Σ_j β_ij from the diagonal so the operator
        // represents u·∇T instead of ∇·(uT).
        if ( subtract_divergence_ )
            AD_ii -= beta_sum;

        // Apply: (M_ii + dt*AD_ii)*T_i + dt*sum_n AD_ij[n]*T_j
        ScalarType result = ( M_ii + dt_ * AD_ii ) * src_( local_subdomain_id, x_cell, y_cell, r_cell );
        for ( int n = 0; n < num_neighbors; ++n )
        {
            result += dt_ * AD_ij[n] * src_(
                local_subdomain_id,
                x_cell + GH::cell_offset_x( n ),
                y_cell + GH::cell_offset_y( n ),
                r_cell + GH::cell_offset_r( n ) );
        }
        dst_( local_subdomain_id, x_cell, y_cell, r_cell ) = result;
    }
};

} // namespace terra::fv::hex::operators
