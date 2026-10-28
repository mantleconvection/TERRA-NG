#pragma once

#include "../../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
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

template < typename ScalarT, int VecDim = 3 >
class EpsilonDivDivKerngenV06XyTiling
{
  public:
    using SrcVectorType                 = linalg::VectorW1Vec< ScalarT, VecDim >;
    using DstVectorType                 = linalg::VectorW1Vec< ScalarT, VecDim >;
    using ScalarType                    = ScalarT;
    static constexpr int LocalMatrixDim = 18;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, LocalMatrixDim, LocalMatrixDim, 2 >;
    using LocalMatrixStorage      = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >;

  private:
    LocalMatrixStorage local_matrix_storage_;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< ScalarType >                     k_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    BoundaryConditions                                       bcs_;

    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;
    linalg::OperatorStoredMatrixMode  operator_stored_matrix_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > dst_;
    grid::Grid4DDataVec< ScalarType, VecDim > src_;

    // Quadrature points.
    const int num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;

    dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
    ScalarT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];

     int local_subdomains_;
    int hex_lat_;
    int hex_rad_;
    int lat_refinement_level_;

    // --- NEW: 3D tiling parameters ---
     int lat_tile_;    // slab size in x and y (same)
  int r_tile_;      // slab size in r (team's r dimension)
  int lat_tiles_;   // number of tiles per lateral dimension (x AND y)
  int r_tiles_;     // number of tiles in r

  int team_size_;   // = lat_tile_*lat_tile_*r_tile_
  int blocks_;      // league size = local_subdomains_ * lat_tiles_^2 * r_tiles_

    ScalarT r_max_;
    ScalarT r_min_;

  public:
    EpsilonDivDivKerngenV06XyTiling(
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
    , send_buffers_( domain )
    , recv_buffers_( domain )
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

        // ---- choose tiles (tune) ----
        // must keep team_size_ reasonable for backend (GPU often <= 1024)
        lat_tile_ = 4;   // x=y slab size
    r_tile_   = 8;   // r slab size

    lat_tiles_ = (hex_lat_ + lat_tile_ - 1) / lat_tile_;
    r_tiles_   = (hex_rad_ + r_tile_  - 1) / r_tile_;

    team_size_ = lat_tile_ * lat_tile_ * r_tile_;
    blocks_    = local_subdomains_ * lat_tiles_ * lat_tiles_ * r_tiles_;

        r_min_ = domain_info.radii()[0];
        r_max_ = domain_info.radii()[domain_info.radii().size() - 1];

        util::logroot << "[EpsilonDivDiv] tile size (x,y,r)=(" << lat_tile_ << "," << lat_tile_ << "," << r_tile_
                      << ")" << std::endl;
        util::logroot << "[EpsilonDivDiv] number of tiles (x,y,r)=(" << lat_tiles_ << "," << lat_tiles_ << "," << r_tiles_
                      << "), team_size=" << team_size_ << ", blocks=" << blocks_ << std::endl;
   
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    /// @brief S/Getter for diagonal member
    void set_diagonal( bool v ) { diagonal_ = v; }

    /// @brief Getter for coefficient
    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }

    /// @brief Getter for domain member
    const grid::shell::DistributedDomain& get_domain() const { return domain_; }

    /// @brief Getter for radii member
    grid::Grid2DDataScalar< ScalarT > get_radii() const { return radii_; }

    /// @brief Getter for grid member
    grid::Grid3DDataVec< ScalarT, 3 > get_grid() { return grid_; }

    /// @brief Getter for mask member
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

    /// @brief allocates memory for the local matrices
    void set_stored_matrix_mode(
        linalg::OperatorStoredMatrixMode     operator_stored_matrix_mode,
        int                                  level_range,
        grid::Grid4DDataScalar< ScalarType > GCAElements )
    {
        operator_stored_matrix_mode_ = operator_stored_matrix_mode;

        // allocate storage if necessary
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            local_matrix_storage_ = linalg::solvers::LocalMatrixStorage< ScalarType, LocalMatrixDim >(
                domain_, operator_stored_matrix_mode_, level_range, GCAElements );
        }
    }

    linalg::OperatorStoredMatrixMode get_stored_matrix_mode() { return operator_stored_matrix_mode_; }

    /// @brief Set the local matrix stored in the operator
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

    /// @brief Retrives the local matrix
    /// if there is stored local matrices, the desired local matrix is loaded and returned
    /// if not, the local matrix is assembled on-the-fly
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > get_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        // request from storage
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
            if ( !local_matrix_storage_.has_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge ) )
            {
                Kokkos::abort( "No matrix found at that spatial index." );
            }
            return local_matrix_storage_.get_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
        }
        else
        {
            return assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
        }
    }

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
        Kokkos::parallel_for( "matvec", policy, *this );
        //   grid::shell::local_domain_md_range_policy_cells( domain_ ),
        //s   *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "epsilon_divdiv_comm" );

            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

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
        E00   = 0.0;
        E11   = 0.0;
        E22   = 0.0;
        sym01 = 0.0;
        sym02 = 0.0;
        sym12 = 0.0;
        gdd   = 0.0;

        // dim selects which COLUMN is populated:
        // dim==0: E[0][0]=g0, E[1][0]=g1, E[2][0]=g2
        // dim==1: E[0][1]=g0, E[1][1]=g1, E[2][1]=g2
        // dim==2: E[0][2]=g0, E[1][2]=g1, E[2][2]=g2
        switch ( dim )
        {
        case 0:
            E00   = g0;
            gdd   = g0;       // E[0][0]
            sym01 = 0.5 * g1; // 0.5*(E[0][1]=0 + E[1][0]=g1)
            sym02 = 0.5 * g2; // 0.5*(E[0][2]=0 + E[2][0]=g2)
            sym12 = 0.0;
            break;

        case 1:
            E11   = g1;
            gdd   = g1;       // E[1][1]
            sym01 = 0.5 * g0; // 0.5*(E[0][1]=g0 + E[1][0]=0)
            sym02 = 0.0;
            sym12 = 0.5 * g2; // 0.5*(E[1][2]=0 + E[2][1]=g2)
            break;

        default: // case 2
            E22   = g2;
            gdd   = g2; // E[2][2]
            sym01 = 0.0;
            sym02 = 0.5 * g0; // 0.5*(E[0][2]=g0 + E[2][0]=0)
            sym12 = 0.5 * g1; // 0.5*(E[1][2]=g1 + E[2][1]=0)
            break;
        }
    }

    using Team = Kokkos::TeamPolicy<>::member_type;

    // scratch size for the WHOLE (x,y,r) tile slab
    KOKKOS_INLINE_FUNCTION
    size_t team_shmem_size( const int /*ts*/ ) const
    {
          const int nlev = r_tile_ + 1;
    const int n  = lat_tile_ + 1;
    const int nxy  = n * n;

    // coords_sh: nxy * 3
    // src_sh:    nxy * 3 * nlev
    // k_sh:      nxy * nlev
    // r_sh:      nlev
    const size_t ndoubles =
      size_t(nxy) * 3 + size_t(nxy) * 3 * nlev + size_t(nxy) * nlev + size_t(nlev);

    return sizeof(double) * ndoubles;
    }

   KOKKOS_INLINE_FUNCTION
  void operator()(const Team& team) const
  {
    // league_rank -> (subdomain, lat_y_id, lat_x_id, r_tile_id)
    int tmp = team.league_rank();

    const int r_tile_id = tmp % r_tiles_;
    tmp /= r_tiles_;

    const int lat_y_id = tmp % lat_tiles_;
    tmp /= lat_tiles_;

    const int lat_x_id = tmp % lat_tiles_;
    tmp /= lat_tiles_;

    const int local_subdomain_id = tmp;

    const int x0 = lat_x_id * lat_tile_;
    const int y0 = lat_y_id * lat_tile_;
    const int r0 = r_tile_id * r_tile_;

    // team_rank -> (tx, ty, tr) where tx,ty in [0..lat_tile_-1], tr in [0..r_tile_-1]
    const int tid = team.team_rank();
    const int tx  = tid % lat_tile_;
    const int ty  = (tid / lat_tile_) % lat_tile_;
    const int tr  = tid / (lat_tile_ * lat_tile_);

    if (tr >= r_tile_) return;

    const int x_cell = x0 + tx;
    const int y_cell = y0 + ty;
    const int r_cell = r0 + tr;

    // Each element needs x_cell+1, y_cell+1 and r_cell+1
  
    // ---- shared slab dims ----
    const int nlev = r_tile_ + 1;
    const int nxy  = (lat_tile_ + 1) * (lat_tile_ + 1);

    double* shmem = reinterpret_cast<double*>(
      team.team_shmem().get_shmem(team_shmem_size(team.team_size())));

    using ScratchCoords =
      Kokkos::View<double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    using ScratchSrc =
      Kokkos::View<double***, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    using ScratchK =
      Kokkos::View<double**, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged>;

    ScratchCoords coords_sh(shmem, nxy, 3); shmem += nxy * 3;
    ScratchSrc    src_sh   (shmem, nxy, 3, nlev); shmem += nxy * 3 * nlev;
    ScratchK      k_sh     (shmem, nxy, nlev);    shmem += nxy * nlev;

    auto r_sh =
      Kokkos::View<double*, Kokkos::LayoutRight, typename Team::scratch_memory_space, Kokkos::MemoryUnmanaged>(
        shmem, nlev);

    auto node_id = [&](int nx, int ny) -> int { return nx + (lat_tile_ + 1) * ny; };

    // ---- cooperative loads for whole tile slab ----
    // coords
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nxy), [&](int n) {
      const int dxn = n % (lat_tile_ + 1) ;
      const int dyn = n / (lat_tile_ + 1) ;
      const int xi  = x0 + dxn;
      const int yi  = y0 + dyn;

      if (xi <= hex_lat_ && yi <= hex_lat_) {
        coords_sh(n,0) = grid_(local_subdomain_id, xi, yi, 0);
        coords_sh(n,1) = grid_(local_subdomain_id, xi, yi, 1);
        coords_sh(n,2) = grid_(local_subdomain_id, xi, yi, 2);
      } else {
        coords_sh(n,0) = coords_sh(n,1) = coords_sh(n,2) = 0.0;
      }
    });

    // radii
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nlev), [&](int lvl) {
      const int rr = r0 + lvl;
      r_sh(lvl) = (rr <= hex_rad_) ? radii_(local_subdomain_id, rr) : 0.0;
    });

    // src/k dofs
    const int total_pairs = nxy * nlev;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, total_pairs), [&](int t) {
      const int lvl  = t / nxy;
      const int node = t - lvl * nxy;

      const int dxn = node % (lat_tile_ + 1) ;
      const int dyn = node / (lat_tile_ + 1) ;

      const int xi = x0 + dxn;
      const int yi = y0 + dyn;
      const int rr = r0 + lvl;

      if (xi <= hex_lat_ && yi <= hex_lat_ && rr <= hex_rad_) {
        k_sh(node, lvl)      = k_(local_subdomain_id, xi, yi, rr);
        src_sh(node, 0, lvl) = src_(local_subdomain_id, xi, yi, rr, 0);
        src_sh(node, 1, lvl) = src_(local_subdomain_id, xi, yi, rr, 1);
        src_sh(node, 2, lvl) = src_(local_subdomain_id, xi, yi, rr, 2);
      } else {
        k_sh(node, lvl) = 0.0;
        src_sh(node,0,lvl) = src_sh(node,1,lvl) = src_sh(node,2,lvl) = 0.0;
      }
    });

    team.team_barrier();

        // ---------------- per-thread element compute (ONE element)
        // radii for this element's r (local slab index = tr)

          if (x_cell < hex_lat_ && y_cell < hex_lat_ && r_cell < hex_rad_) {

        const int    lvl0 = tr;
        const double r_0  = r_sh( lvl0 );
        const double r_1  = r_sh( lvl0 + 1 );

        const bool at_cmb     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
        const bool at_surface = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );

        const bool at_boundary    = at_cmb || at_surface;
        bool       treat_boundary = false;
        if ( at_boundary )
        {
            const ShellBoundaryFlag sbf = at_cmb ? CMB : SURFACE;
            treat_boundary              = ( get_boundary_condition_flag( bcs_, sbf ) == DIRICHLET );
        }

        const int cmb_shift     = ( ( at_boundary && treat_boundary && ( !diagonal_ ) && at_cmb ) ? 3 : 0 );
        const int surface_shift = ( ( at_boundary && treat_boundary && ( !diagonal_ ) && at_surface ) ? 3 : 0 );

        // ---- your existing constants (unchanged) ----
        static constexpr int WEDGE_NODE_OFF[2][6][3] = {
            { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 0, 1, 1 } },
            { { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 1, 1 }, { 0, 1, 1 }, { 1, 0, 1 } } };

        static constexpr int WEDGE_TO_UNIQUE[2][6] = {
            { 0, 1, 2, 3, 4, 5 }, // wedge 0
            { 6, 2, 1, 7, 5, 4 }  // wedge 1
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

      

        // 4 surface nodes ids (within tile footprint)
        const int n00 = node_id( tx, ty );
        const int n01 = node_id( tx, ty + 1 );
        const int n10 = node_id( tx + 1, ty );
        const int n11 = node_id( tx + 1, ty + 1 );

        // wedge surface coords in registers
        double ws[2][3][3];

        // wedge 0: (q00,q10,q01)
        ws[0][0][0] = coords_sh( n00, 0 ); ws[0][0][1] = coords_sh( n00, 1 ); ws[0][0][2] = coords_sh( n00, 2 );
        ws[0][1][0] = coords_sh( n10, 0 ); ws[0][1][1] = coords_sh( n10, 1 ); ws[0][1][2] = coords_sh( n10, 2 );
        ws[0][2][0] = coords_sh( n01, 0 ); ws[0][2][1] = coords_sh( n01, 1 ); ws[0][2][2] = coords_sh( n01, 2 );

        // wedge 1: (q11,q01,q10)
        ws[1][0][0] = coords_sh( n11, 0 ); ws[1][0][1] = coords_sh( n11, 1 ); ws[1][0][2] = coords_sh( n11, 2 );
        ws[1][1][0] = coords_sh( n01, 0 ); ws[1][1][1] = coords_sh( n01, 1 ); ws[1][1][2] = coords_sh( n01, 2 );
        ws[1][2][0] = coords_sh( n10, 0 ); ws[1][2][1] = coords_sh( n10, 1 ); ws[1][2][2] = coords_sh( n10, 2 );

        // per-thread accumulation (same as your dst8 approach)
        double dst8[3][8] = { 0.0 };

        for ( int w = 0; w < 2; ++w )
        {

            // k_eval from shared slab
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

                const double J_0_0 = r_mid * ( -ws[w][0][0] + ws[w][1][0] );
                const double J_0_1 = r_mid * ( -ws[w][0][0] + ws[w][2][0] );
                const double J_0_2 = half_dr * ( ONE_THIRD * ( ws[w][0][0] + ws[w][1][0] + ws[w][2][0] ) );

                const double J_1_0 = r_mid * ( -ws[w][0][1] + ws[w][1][1] );
                const double J_1_1 = r_mid * ( -ws[w][0][1] + ws[w][2][1] );
                const double J_1_2 = half_dr * ( ONE_THIRD * ( ws[w][0][1] + ws[w][1][1] + ws[w][2][1] ) );

                const double J_2_0 = r_mid * ( -ws[w][0][2] + ws[w][1][2] );
                const double J_2_1 = r_mid * ( -ws[w][0][2] + ws[w][2][2] );
                const double J_2_2 = half_dr * ( ONE_THIRD * ( ws[w][0][2] + ws[w][1][2] + ws[w][2][2] ) );

                const double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 -
                                     J_0_1 * J_1_0 * J_2_2 + J_0_1 * J_1_2 * J_2_0 +
                                     J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;

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

                        const double pairing0 = 2.0 * sym01;
                        const double pairing1 = 2.0 * sym02;
                        const double pairing2 = 2.0 * sym12;

                        const int u = WEDGE_TO_UNIQUE[w][node_idx];

                        dst8[dimi][u] += kwJ * ( NEG_TWO_THIRDS * div_u * gdd + 2.0 * pairing0 * gu10 +
                                                 2.0 * pairing1 * gu20 + 2.0 * pairing2 * gu21 + 2.0 * E00 * gu00 +
                                                 2.0 * E11 * gu11 + 2.0 * E22 * gu22 );
                    }
                }
            }

            if ( diagonal_ || ( treat_boundary && at_boundary ) )
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

                        const double pairing0 = 4.0 * s;
                        const double pairing1 = 2.0 * s;

                        const int u = WEDGE_TO_UNIQUE[w][node_idx];

                        dst8[dim_diagBC][u] +=
                            kwJ * ( pairing0 * ( sym01 * sym01 ) + pairing0 * ( sym02 * sym02 ) +
                                    pairing0 * ( sym12 * sym12 ) + pairing1 * ( E00 * E00 ) + pairing1 * ( E11 * E11 ) +
                                    pairing1 * ( E22 * E22 ) + NEG_TWO_THIRDS * ( gdd * gdd ) * s );
                    }
                }
            }
        } // w

        // scatter
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
    }

    /// @brief: For both trial and test space this function sets up a vector:
    /// each vector element holds the symmetric gradient (a 3x3 matrix) of the shape function of the corresponding dof
    /// (if dimi == dimj, these are the same and we are on the diagonal of the vectorial diffusion operator)
    /// Additionally, we compute the scalar factor for the numerical integral comp: determinant of the jacobian,
    /// evaluation of the coefficient k on the element and the quadrature weight of the current quad-point.

    /// The idea of this function is that the two vectors can be:
    /// - accumulated to the result of the local matvec with 2 * num_nodes_per_wedge complexity
    ///   by scaling the dot product of the trial vec and local src dofs with each element of the test vec
    ///   (and adding to the dst dofs, this is the fused local matvec).
    /// - propagated to the local matrix by an outer product of the two vectors
    ///   (without applying it to dofs). This is e.g. required to assemble the finest grid local
    ///   matrix on-the-fly during GCA/Galerkin coarsening.

    ///
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

        // dot of coeff dofs and element-local shape functions to evaluate the coefficent on the current element
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

    /// @brief assemble the local matrix and return it for a given element, wedge, and vectorial component
    /// (determined by dimi, dimj)
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > assemble_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        // Gather surface points for each wedge.
        // TODO gather this for only 1 wedge
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        dense::Vec< ScalarT, 6 > k_local_hex[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

        // Compute the local element matrix.
        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A = {};
        for ( int dimi = 0; dimi < 3; ++dimi )
        {
            for ( int dimj = 0; dimj < 3; ++dimj )
            {
                // spatial dimensions: quadrature points and wedge
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

                    // propagate on local matrix by outer product of test and trial vecs
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            A( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) +=
                                jdet_keval_quadweight *
                                ( 2 * sym_grad_j[j].double_contract( sym_grad_i[i] ) -
                                  2.0 / 3.0 * sym_grad_j[j]( dimj, dimj ) * sym_grad_i[i]( dimi, dimi ) );
                            // for the div, we just extract the component from the gradient vector
                        }
                    }
                }
            }
        }

        return A;
    }
};

static_assert( linalg::GCACapable< EpsilonDivDivKerngenV06XyTiling< float > > );
static_assert( linalg::GCACapable< EpsilonDivDivKerngenV06XyTiling< double > > );

} // namespace terra::fe::wedge::operators::shell::epsdivdiv_history
