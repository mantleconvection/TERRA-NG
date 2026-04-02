#pragma once

#include "../../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/gca/local_matrix_storage.hpp"
#include "linalg/trafo/local_basis_trafo_normal_tangential.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell::epsdivdiv_history {

using terra::grid::shell::BoundaryConditionFlag;
using terra::grid::shell::BoundaryConditions;
using terra::grid::shell::ShellBoundaryFlag;

template < typename ScalarT, int VecDim = 3 >
class EpsilonDivDivKerngenV02bSingleQuadpoint
{
  public:
    using SrcVectorType           = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType           = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType              = ScalarT;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, 18, 18, 2 >;

  private:
    bool storeLMatrices_ =
        false; // set to let apply_impl() know, that it should store the local matrices after assembling them
    bool applyStoredLMatrices_ =
        false; // set to make apply_impl() load and use the stored LMatrices for the operator application
    Grid4DDataLocalMatrices lmatrices_;
    bool                    single_quadpoint_ = true;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< ScalarType >                     k_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    BoundaryConditions                                       bcs_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;
    linalg::OperatorStoredMatrixMode  operator_stored_matrix_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

  public:
    EpsilonDivDivKerngenV02bSingleQuadpoint(
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
    , treat_boundary_( true )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , operator_stored_matrix_mode_( operator_stored_matrix_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {
        bcs_[0] = bcs[0];
        bcs_[1] = bcs[1];
    }

    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }

    /// @brief Getter for domain member
    const grid::shell::DistributedDomain& get_domain() const { return domain_; }

    /// @brief Getter for radii member
    grid::Grid2DDataScalar< ScalarT > get_radii() const { return radii_; }

    /// @brief Getter for grid member
    grid::Grid3DDataVec< ScalarT, 3 > get_grid() const { return grid_; }

    /// @brief Retrives the local matrix stored in the operator
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, 6, 6 > get_lmatrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge,
        const int dimi,
        const int dimj ) const
    {
        assert( lmatrices_.data() != nullptr );
        dense::Mat< ScalarT, 6, 6 > ijslice;
        for ( int i = 0; i < 6; ++i )
        {
            for ( int j = 0; j < 6; ++j )
            {
                ijslice( i, j ) =
                    lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge )( i + dimi * 6, j + dimj * 6 );
            }
        }
        return ijslice;
    }

    /// @brief Set the local matrix stored in the operator
    KOKKOS_INLINE_FUNCTION
    void set_lmatrix(
        const int                   local_subdomain_id,
        const int                   x_cell,
        const int                   y_cell,
        const int                   r_cell,
        const int                   wedge,
        const int                   dimi,
        const int                   dimj,
        dense::Mat< ScalarT, 6, 6 > mat ) const
    {
        assert( lmatrices_.data() != nullptr );
        for ( int i = 0; i < 6; ++i )
        {
            for ( int j = 0; j < 6; ++j )
            {
                lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge )( i + dimi * 6, j + dimj * 6 ) =
                    mat( i, j );
            }
        }
    }

    /// @brief Setter/Getter for app applyStoredLMatrices_: usage of stored local matrices during apply
    void setApplyStoredLMatrices( bool v ) { applyStoredLMatrices_ = v; }

    /// @brief S/Getter for diagonal member
    void set_diagonal( bool v ) { diagonal_ = v; }

    /// @brief
    /// allocates memory for the local matrices
    /// calls kernel with storeLMatrices_ = true to assemble and store the local matrices
    /// sets applyStoredLMatrices_, such that future applies use the stored local matrices
    void store_lmatrices()
    {
        storeLMatrices_ = true;
        if ( lmatrices_.data() == nullptr )
        {
            lmatrices_ = Grid4DDataLocalMatrices(
                "LaplaceSimple::lmatrices_",
                domain_.subdomains().size(),
                domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain_.domain_info().subdomain_num_nodes_radially() - 1 );
            Kokkos::parallel_for(
                "assemble_store_lmatrices", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
            Kokkos::fence();
        }
        storeLMatrices_       = false;
        applyStoredLMatrices_ = true;
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "vector_laplace_apply" );
        if ( storeLMatrices_ or applyStoredLMatrices_ )
            assert( lmatrices_.data() != nullptr );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "VectorLaplace: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "VectorLaplace: src/dst mismatch" );
        }

        util::Timer timer_kernel( "vector_laplace_kernel" );
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "vector_laplace_comm" );

            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Compute the local element matrix.
        dense::Mat< ScalarT, 18, 18 > A[num_wedges_per_hex_cell] = {};

        if ( !applyStoredLMatrices_ )
        {

double wedge_surf_phy_coords[2][3][3];
double quad_surface_coords[2][2][3];
quad_surface_coords[0][0][0] = grid_(local_subdomain_id, x_cell, y_cell, 0);
quad_surface_coords[0][0][1] = grid_(local_subdomain_id, x_cell, y_cell, 1);
quad_surface_coords[0][0][2] = grid_(local_subdomain_id, x_cell, y_cell, 2);
quad_surface_coords[0][1][0] = grid_(local_subdomain_id, x_cell, y_cell + 1, 0);
quad_surface_coords[0][1][1] = grid_(local_subdomain_id, x_cell, y_cell + 1, 1);
quad_surface_coords[0][1][2] = grid_(local_subdomain_id, x_cell, y_cell + 1, 2);
quad_surface_coords[1][0][0] = grid_(local_subdomain_id, x_cell + 1, y_cell, 0);
quad_surface_coords[1][0][1] = grid_(local_subdomain_id, x_cell + 1, y_cell, 1);
quad_surface_coords[1][0][2] = grid_(local_subdomain_id, x_cell + 1, y_cell, 2);
quad_surface_coords[1][1][0] = grid_(local_subdomain_id, x_cell + 1, y_cell + 1, 0);
quad_surface_coords[1][1][1] = grid_(local_subdomain_id, x_cell + 1, y_cell + 1, 1);
quad_surface_coords[1][1][2] = grid_(local_subdomain_id, x_cell + 1, y_cell + 1, 2);
wedge_surf_phy_coords[0][0][0] = quad_surface_coords[0][0][0];
wedge_surf_phy_coords[0][0][1] = quad_surface_coords[0][0][1];
wedge_surf_phy_coords[0][0][2] = quad_surface_coords[0][0][2];
wedge_surf_phy_coords[0][1][0] = quad_surface_coords[1][0][0];
wedge_surf_phy_coords[0][1][1] = quad_surface_coords[1][0][1];
wedge_surf_phy_coords[0][1][2] = quad_surface_coords[1][0][2];
wedge_surf_phy_coords[0][2][0] = quad_surface_coords[0][1][0];
wedge_surf_phy_coords[0][2][1] = quad_surface_coords[0][1][1];
wedge_surf_phy_coords[0][2][2] = quad_surface_coords[0][1][2];
wedge_surf_phy_coords[1][0][0] = quad_surface_coords[1][1][0];
wedge_surf_phy_coords[1][0][1] = quad_surface_coords[1][1][1];
wedge_surf_phy_coords[1][0][2] = quad_surface_coords[1][1][2];
wedge_surf_phy_coords[1][1][0] = quad_surface_coords[0][1][0];
wedge_surf_phy_coords[1][1][1] = quad_surface_coords[0][1][1];
wedge_surf_phy_coords[1][1][2] = quad_surface_coords[0][1][2];
wedge_surf_phy_coords[1][2][0] = quad_surface_coords[1][0][0];
wedge_surf_phy_coords[1][2][1] = quad_surface_coords[1][0][1];
wedge_surf_phy_coords[1][2][2] = quad_surface_coords[1][0][2];
double r_0 = radii_(local_subdomain_id, r_cell);
double r_1 = radii_(local_subdomain_id, r_cell + 1);
double src_local_hex[3][2][6];
int dim;
for (dim = 0; dim < 3; dim += 1) {
   src_local_hex[dim][0][0] = src_(local_subdomain_id, x_cell, y_cell, r_cell, dim);
   src_local_hex[dim][0][1] = src_(local_subdomain_id, x_cell + 1, y_cell, r_cell, dim);
   src_local_hex[dim][0][2] = src_(local_subdomain_id, x_cell, y_cell + 1, r_cell, dim);
   src_local_hex[dim][0][3] = src_(local_subdomain_id, x_cell, y_cell, r_cell + 1, dim);
   src_local_hex[dim][0][4] = src_(local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim);
   src_local_hex[dim][0][5] = src_(local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim);
   src_local_hex[dim][1][0] = src_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, dim);
   src_local_hex[dim][1][1] = src_(local_subdomain_id, x_cell, y_cell + 1, r_cell, dim);
   src_local_hex[dim][1][2] = src_(local_subdomain_id, x_cell + 1, y_cell, r_cell, dim);
   src_local_hex[dim][1][3] = src_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, dim);
   src_local_hex[dim][1][4] = src_(local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim);
   src_local_hex[dim][1][5] = src_(local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim);
};
double k_local_hex[2][6];
k_local_hex[0][0] = k_(local_subdomain_id, x_cell, y_cell, r_cell);
k_local_hex[0][1] = k_(local_subdomain_id, x_cell + 1, y_cell, r_cell);
k_local_hex[0][2] = k_(local_subdomain_id, x_cell, y_cell + 1, r_cell);
k_local_hex[0][3] = k_(local_subdomain_id, x_cell, y_cell, r_cell + 1);
k_local_hex[0][4] = k_(local_subdomain_id, x_cell + 1, y_cell, r_cell + 1);
k_local_hex[0][5] = k_(local_subdomain_id, x_cell, y_cell + 1, r_cell + 1);
k_local_hex[1][0] = k_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell);
k_local_hex[1][1] = k_(local_subdomain_id, x_cell, y_cell + 1, r_cell);
k_local_hex[1][2] = k_(local_subdomain_id, x_cell + 1, y_cell, r_cell);
k_local_hex[1][3] = k_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1);
k_local_hex[1][4] = k_(local_subdomain_id, x_cell, y_cell + 1, r_cell + 1);
k_local_hex[1][5] = k_(local_subdomain_id, x_cell + 1, y_cell, r_cell + 1);
double qp_array[1][3];
double qw_array[1];
qp_array[0][0] = 0.33333333333333333;
qp_array[0][1] = 0.33333333333333333;
qp_array[0][2] = 0.0;
qw_array[0] = 1.0;
int cmb_shift = ((treat_boundary_ && diagonal_ == false && r_cell == 0) ? (
   3
)
: (
   0
));
int max_rad = radii_.extent(1) - 1;
int surface_shift = ((treat_boundary_ && diagonal_ == false && max_rad == r_cell + 1) ? (
   3
)
: (
   0
));
double dst_array[3][2][6] = {0};
int w = 0;
for (w = 0; w < 2; w += 1) {
   int q = 0;
   for (q = 0; q < 1; q += 1) {
      /* Coefficient evaluation on current wedge w */;
      double tmpcse_k_eval_0 = (1.0/2.0)*qp_array[q][2];
      double tmpcse_k_eval_1 = 1.0/2.0 - tmpcse_k_eval_0;
      double tmpcse_k_eval_2 = tmpcse_k_eval_0 + 1.0/2.0;
      double tmpcse_k_eval_3 = -qp_array[q][0] - qp_array[q][1] + 1;
      double k_eval = tmpcse_k_eval_1*tmpcse_k_eval_3*k_local_hex[w][0] + tmpcse_k_eval_1*k_local_hex[w][1]*qp_array[q][0] + tmpcse_k_eval_1*k_local_hex[w][2]*qp_array[q][1] + tmpcse_k_eval_2*tmpcse_k_eval_3*k_local_hex[w][3] + tmpcse_k_eval_2*k_local_hex[w][4]*qp_array[q][0] + tmpcse_k_eval_2*k_local_hex[w][5]*qp_array[q][1];
      /* Computation + Inversion of the Jacobian */;
      double tmpcse_J_0 = -1.0/2.0*r_0 + (1.0/2.0)*r_1;
      double tmpcse_J_1 = r_0 + tmpcse_J_0*(qp_array[q][2] + 1);
      double tmpcse_J_2 = -qp_array[q][0] - qp_array[q][1] + 1;
      double J_0_0 = tmpcse_J_1*(-wedge_surf_phy_coords[w][0][0] + wedge_surf_phy_coords[w][1][0]);
      double J_0_1 = tmpcse_J_1*(-wedge_surf_phy_coords[w][0][0] + wedge_surf_phy_coords[w][2][0]);
      double J_0_2 = tmpcse_J_0*(tmpcse_J_2*wedge_surf_phy_coords[w][0][0] + qp_array[q][0]*wedge_surf_phy_coords[w][1][0] + qp_array[q][1]*wedge_surf_phy_coords[w][2][0]);
      double J_1_0 = tmpcse_J_1*(-wedge_surf_phy_coords[w][0][1] + wedge_surf_phy_coords[w][1][1]);
      double J_1_1 = tmpcse_J_1*(-wedge_surf_phy_coords[w][0][1] + wedge_surf_phy_coords[w][2][1]);
      double J_1_2 = tmpcse_J_0*(tmpcse_J_2*wedge_surf_phy_coords[w][0][1] + qp_array[q][0]*wedge_surf_phy_coords[w][1][1] + qp_array[q][1]*wedge_surf_phy_coords[w][2][1]);
      double J_2_0 = tmpcse_J_1*(-wedge_surf_phy_coords[w][0][2] + wedge_surf_phy_coords[w][1][2]);
      double J_2_1 = tmpcse_J_1*(-wedge_surf_phy_coords[w][0][2] + wedge_surf_phy_coords[w][2][2]);
      double J_2_2 = tmpcse_J_0*(tmpcse_J_2*wedge_surf_phy_coords[w][0][2] + qp_array[q][0]*wedge_surf_phy_coords[w][1][2] + qp_array[q][1]*wedge_surf_phy_coords[w][2][2]);
      double J_det = J_0_0*J_1_1*J_2_2 - J_0_0*J_1_2*J_2_1 - J_0_1*J_1_0*J_2_2 + J_0_1*J_1_2*J_2_0 + J_0_2*J_1_0*J_2_1 - J_0_2*J_1_1*J_2_0;
      double tmpcse_J_invT_0 = 1.0/J_det;
      double J_invT_cse_0_0 = tmpcse_J_invT_0*(J_1_1*J_2_2 - J_1_2*J_2_1);
      double J_invT_cse_0_1 = tmpcse_J_invT_0*(-J_1_0*J_2_2 + J_1_2*J_2_0);
      double J_invT_cse_0_2 = tmpcse_J_invT_0*(J_1_0*J_2_1 - J_1_1*J_2_0);
      double J_invT_cse_1_0 = tmpcse_J_invT_0*(-J_0_1*J_2_2 + J_0_2*J_2_1);
      double J_invT_cse_1_1 = tmpcse_J_invT_0*(J_0_0*J_2_2 - J_0_2*J_2_0);
      double J_invT_cse_1_2 = tmpcse_J_invT_0*(-J_0_0*J_2_1 + J_0_1*J_2_0);
      double J_invT_cse_2_0 = tmpcse_J_invT_0*(J_0_1*J_1_2 - J_0_2*J_1_1);
      double J_invT_cse_2_1 = tmpcse_J_invT_0*(-J_0_0*J_1_2 + J_0_2*J_1_0);
      double J_invT_cse_2_2 = tmpcse_J_invT_0*(J_0_0*J_1_1 - J_0_1*J_1_0);

      double scalar_grad[6][3] = {0};
      double tmpcse_grad_i_0 = (1.0/2.0)*qp_array[q][2];
      double tmpcse_grad_i_1 = tmpcse_grad_i_0 - 1.0/2.0;
      double tmpcse_grad_i_2 = (1.0/2.0)*qp_array[q][0];
      double tmpcse_grad_i_3 = (1.0/2.0)*qp_array[q][1];
      double tmpcse_grad_i_4 = tmpcse_grad_i_2 + tmpcse_grad_i_3 - 1.0/2.0;
      double tmpcse_grad_i_5 = J_invT_cse_0_2*tmpcse_grad_i_2;
      double tmpcse_grad_i_6 = -tmpcse_grad_i_1;
      double tmpcse_grad_i_7 = J_invT_cse_1_2*tmpcse_grad_i_2;
      double tmpcse_grad_i_8 = J_invT_cse_2_2*tmpcse_grad_i_2;
      double tmpcse_grad_i_9 = J_invT_cse_0_2*tmpcse_grad_i_3;
      double tmpcse_grad_i_10 = J_invT_cse_1_2*tmpcse_grad_i_3;
      double tmpcse_grad_i_11 = J_invT_cse_2_2*tmpcse_grad_i_3;
      double tmpcse_grad_i_12 = tmpcse_grad_i_0 + 1.0/2.0;
      double tmpcse_grad_i_13 = -tmpcse_grad_i_12;
      double tmpcse_grad_i_14 = -tmpcse_grad_i_4;
      scalar_grad[0][0] = J_invT_cse_0_0*tmpcse_grad_i_1 + J_invT_cse_0_1*tmpcse_grad_i_1 + J_invT_cse_0_2*tmpcse_grad_i_4;
      scalar_grad[0][1] = J_invT_cse_1_0*tmpcse_grad_i_1 + J_invT_cse_1_1*tmpcse_grad_i_1 + J_invT_cse_1_2*tmpcse_grad_i_4;
      scalar_grad[0][2] = J_invT_cse_2_0*tmpcse_grad_i_1 + J_invT_cse_2_1*tmpcse_grad_i_1 + J_invT_cse_2_2*tmpcse_grad_i_4;
      scalar_grad[1][0] = J_invT_cse_0_0*tmpcse_grad_i_6 - tmpcse_grad_i_5;
      scalar_grad[1][1] = J_invT_cse_1_0*tmpcse_grad_i_6 - tmpcse_grad_i_7;
      scalar_grad[1][2] = J_invT_cse_2_0*tmpcse_grad_i_6 - tmpcse_grad_i_8;
      scalar_grad[2][0] = J_invT_cse_0_1*tmpcse_grad_i_6 - tmpcse_grad_i_9;
      scalar_grad[2][1] = J_invT_cse_1_1*tmpcse_grad_i_6 - tmpcse_grad_i_10;
      scalar_grad[2][2] = J_invT_cse_2_1*tmpcse_grad_i_6 - tmpcse_grad_i_11;
      scalar_grad[3][0] = J_invT_cse_0_0*tmpcse_grad_i_13 + J_invT_cse_0_1*tmpcse_grad_i_13 + J_invT_cse_0_2*tmpcse_grad_i_14;
      scalar_grad[3][1] = J_invT_cse_1_0*tmpcse_grad_i_13 + J_invT_cse_1_1*tmpcse_grad_i_13 + J_invT_cse_1_2*tmpcse_grad_i_14;
      scalar_grad[3][2] = J_invT_cse_2_0*tmpcse_grad_i_13 + J_invT_cse_2_1*tmpcse_grad_i_13 + J_invT_cse_2_2*tmpcse_grad_i_14;
      scalar_grad[4][0] = J_invT_cse_0_0*tmpcse_grad_i_12 + tmpcse_grad_i_5;
      scalar_grad[4][1] = J_invT_cse_1_0*tmpcse_grad_i_12 + tmpcse_grad_i_7;
      scalar_grad[4][2] = J_invT_cse_2_0*tmpcse_grad_i_12 + tmpcse_grad_i_8;
      scalar_grad[5][0] = J_invT_cse_0_1*tmpcse_grad_i_12 + tmpcse_grad_i_9;
      scalar_grad[5][1] = J_invT_cse_1_1*tmpcse_grad_i_12 + tmpcse_grad_i_10;
      scalar_grad[5][2] = J_invT_cse_2_1*tmpcse_grad_i_12 + tmpcse_grad_i_11;
      int dimj;

      double grad_u[3][3] = {0};
      double div_u = 0.0;
      int node_idx;
      for (dimj = 0; dimj < 3; dimj += 1) {
         if (diagonal_ == false) {
            for (node_idx = cmb_shift; node_idx < 6 - surface_shift; node_idx += 1) {

               double E_grad_trial[3][3] = {0};
               E_grad_trial[0][dimj] = scalar_grad[node_idx][0];
               E_grad_trial[1][dimj] = scalar_grad[node_idx][1];
               E_grad_trial[2][dimj] = scalar_grad[node_idx][2];
               double tmpcse_symgrad_trial_0 = 0.5*E_grad_trial[0][1] + 0.5*E_grad_trial[1][0];
               double tmpcse_symgrad_trial_1 = 0.5*E_grad_trial[0][2] + 0.5*E_grad_trial[2][0];
               double tmpcse_symgrad_trial_2 = 0.5*E_grad_trial[1][2] + 0.5*E_grad_trial[2][1];
               grad_u[0][0] = 1.0*E_grad_trial[0][0]*src_local_hex[dimj][w][node_idx] + grad_u[0][0];
               grad_u[0][1] = tmpcse_symgrad_trial_0*src_local_hex[dimj][w][node_idx] + grad_u[0][1];
               grad_u[0][2] = tmpcse_symgrad_trial_1*src_local_hex[dimj][w][node_idx] + grad_u[0][2];
               grad_u[1][0] = tmpcse_symgrad_trial_0*src_local_hex[dimj][w][node_idx] + grad_u[1][0];
               grad_u[1][1] = 1.0*E_grad_trial[1][1]*src_local_hex[dimj][w][node_idx] + grad_u[1][1];
               grad_u[1][2] = tmpcse_symgrad_trial_2*src_local_hex[dimj][w][node_idx] + grad_u[1][2];
               grad_u[2][0] = tmpcse_symgrad_trial_1*src_local_hex[dimj][w][node_idx] + grad_u[2][0];
               grad_u[2][1] = tmpcse_symgrad_trial_2*src_local_hex[dimj][w][node_idx] + grad_u[2][1];
               grad_u[2][2] = 1.0*E_grad_trial[2][2]*src_local_hex[dimj][w][node_idx] + grad_u[2][2];
               div_u = div_u + E_grad_trial[dimj][dimj]*src_local_hex[dimj][w][node_idx];
            };
         };
      };
      int dimi;
      for (dimi = 0; dimi < 3; dimi += 1) {
         if (diagonal_ == false) {
            for (node_idx = cmb_shift; node_idx < 6 - surface_shift; node_idx += 1) {

               double E_grad_test[3][3] = {0};
               E_grad_test[0][dimi] = scalar_grad[node_idx][0];
               E_grad_test[1][dimi] = scalar_grad[node_idx][1];
               E_grad_test[2][dimi] = scalar_grad[node_idx][2];
               double tmpcse_symgrad_test_0 = 0.5*E_grad_test[0][1] + 0.5*E_grad_test[1][0];
               double tmpcse_symgrad_test_1 = 0.5*E_grad_test[0][2] + 0.5*E_grad_test[2][0];
               double tmpcse_symgrad_test_2 = 0.5*E_grad_test[1][2] + 0.5*E_grad_test[2][1];
               double tmpcse_pairing_0 = 2*tmpcse_symgrad_test_0;
               double tmpcse_pairing_1 = 2*tmpcse_symgrad_test_1;
               double tmpcse_pairing_2 = 2*tmpcse_symgrad_test_2;
               dst_array[dimi][w][node_idx] = k_eval*(-0.66666666666666663*div_u*E_grad_test[dimi][dimi] + tmpcse_pairing_0*grad_u[0][1] + tmpcse_pairing_0*grad_u[1][0] + tmpcse_pairing_1*grad_u[0][2] + tmpcse_pairing_1*grad_u[2][0] + tmpcse_pairing_2*grad_u[1][2] + tmpcse_pairing_2*grad_u[2][1] + 2.0*E_grad_test[0][0]*grad_u[0][0] + 2.0*E_grad_test[1][1]*grad_u[1][1] + 2.0*E_grad_test[2][2]*grad_u[2][2])*fabs(J_det)*qw_array[q] + dst_array[dimi][w][node_idx];
            };
         };
      };
      int dim_diagBC;
      for (dim_diagBC = 0; dim_diagBC < 3; dim_diagBC += 1) {
         if (diagonal_ || treat_boundary_ && (r_cell == 0 || r_cell + 1 == max_rad)) {
            int node_idx;
            for (node_idx = surface_shift; node_idx < 6 - cmb_shift; node_idx += 1) {

               double E_grad_test[3][3] = {0};
               E_grad_test[0][dim_diagBC] = scalar_grad[node_idx][0];
               E_grad_test[1][dim_diagBC] = scalar_grad[node_idx][1];
               E_grad_test[2][dim_diagBC] = scalar_grad[node_idx][2];

               double grad_u_diag[3][3] = {0};
               double tmpcse_symgrad_test_0 = 0.5*E_grad_test[0][1] + 0.5*E_grad_test[1][0];
               double tmpcse_symgrad_test_1 = 0.5*E_grad_test[0][2] + 0.5*E_grad_test[2][0];
               double tmpcse_symgrad_test_2 = 0.5*E_grad_test[1][2] + 0.5*E_grad_test[2][1];
               grad_u_diag[0][0] = 1.0*E_grad_test[0][0]*src_local_hex[dim_diagBC][w][node_idx];
               grad_u_diag[0][1] = tmpcse_symgrad_test_0*src_local_hex[dim_diagBC][w][node_idx];
               grad_u_diag[0][2] = tmpcse_symgrad_test_1*src_local_hex[dim_diagBC][w][node_idx];
               grad_u_diag[1][0] = tmpcse_symgrad_test_0*src_local_hex[dim_diagBC][w][node_idx];
               grad_u_diag[1][1] = 1.0*E_grad_test[1][1]*src_local_hex[dim_diagBC][w][node_idx];
               grad_u_diag[1][2] = tmpcse_symgrad_test_2*src_local_hex[dim_diagBC][w][node_idx];
               grad_u_diag[2][0] = tmpcse_symgrad_test_1*src_local_hex[dim_diagBC][w][node_idx];
               grad_u_diag[2][1] = tmpcse_symgrad_test_2*src_local_hex[dim_diagBC][w][node_idx];
               grad_u_diag[2][2] = 1.0*E_grad_test[2][2]*src_local_hex[dim_diagBC][w][node_idx];
               double tmpcse_pairing_0 = 4*src_local_hex[dim_diagBC][w][node_idx];
               double tmpcse_pairing_1 = 2.0*src_local_hex[dim_diagBC][w][node_idx];
               dst_array[dim_diagBC][w][node_idx] = k_eval*(tmpcse_pairing_0*pow(tmpcse_symgrad_test_0, 2) + tmpcse_pairing_0*pow(tmpcse_symgrad_test_1, 2) + tmpcse_pairing_0*pow(tmpcse_symgrad_test_2, 2) + tmpcse_pairing_1*pow(E_grad_test[0][0], 2) + tmpcse_pairing_1*pow(E_grad_test[1][1], 2) + tmpcse_pairing_1*pow(E_grad_test[2][2], 2) - 0.66666666666666663*pow(E_grad_test[dim_diagBC][dim_diagBC], 2)*src_local_hex[dim_diagBC][w][node_idx])*fabs(J_det)*qw_array[q] + dst_array[dim_diagBC][w][node_idx];
            };
         };
      };
   };
};
int dim_add;
for (dim_add = 0; dim_add < 3; dim_add += 1) {
   Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell, r_cell, dim_add), dst_array[dim_add][0][0]);
   Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell, r_cell, dim_add), dst_array[dim_add][0][1] + dst_array[dim_add][1][2]);
   Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell + 1, r_cell, dim_add), dst_array[dim_add][0][2] + dst_array[dim_add][1][1]);
   Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell, r_cell + 1, dim_add), dst_array[dim_add][0][3]);
   Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim_add), dst_array[dim_add][0][4] + dst_array[dim_add][1][5]);
   Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim_add), dst_array[dim_add][0][5] + dst_array[dim_add][1][4]);
   Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, dim_add), dst_array[dim_add][1][0]);
   Kokkos::atomic_add(&dst_(local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, dim_add), dst_array[dim_add][1][3]);
};
        }
        else
        {
            // load LMatrix for both local wedges
            A[0] = lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
            A[1] = lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        if ( storeLMatrices_ )
        {
            // write local matrices to mem
            lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 0 ) = A[0];
            lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 1 ) = A[1];
        }
        else
        {
            dense::Vec< ScalarT, 18 > src[num_wedges_per_hex_cell];
            for ( int dimj = 0; dimj < 3; dimj++ )
            {
                dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
                extract_local_wedge_vector_coefficients(
                    src_d, local_subdomain_id, x_cell, y_cell, r_cell, dimj, src_ );

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        src[wedge]( dimj * num_nodes_per_wedge + i ) = src_d[wedge]( i );
                    }
                }
            }
            //extract_local_wedge_vector_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, dimj, src_ );

            dense::Vec< ScalarT, 18 > dst[num_wedges_per_hex_cell];

            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            //atomically_add_local_wedge_vector_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst );
            for ( int dimi = 0; dimi < 3; dimi++ )
            {
                dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
                dst_d[0] = dst[0].template slice< 6 >( dimi * num_nodes_per_wedge );
                dst_d[1] = dst[1].template slice< 6 >( dimi * num_nodes_per_wedge );

                atomically_add_local_wedge_vector_coefficients(
                    dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst_d );
            }
        }
    }
};

static_assert( linalg::OperatorLike< EpsilonDivDivKerngenV02bSingleQuadpoint< float > > );
static_assert( linalg::OperatorLike< EpsilonDivDivKerngenV02bSingleQuadpoint< double > > );

} // namespace terra::fe::wedge::operators::shell::epsdivdiv_history
