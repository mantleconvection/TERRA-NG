#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "impl/Kokkos_Profiling.hpp"
#include "linalg/operator.hpp"
#include "linalg/solvers/gca/local_matrix_storage.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell {

using grid::shell::ShellBoundaryFlag::CMB;
using grid::shell::ShellBoundaryFlag::SURFACE;
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

  private:
    LocalMatrixStorage local_matrix_storage_;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< ScalarType >                     k_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;
    linalg::OperatorStoredMatrixMode  operator_stored_matrix_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

    // Quadrature points.
    const int num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;

    dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_1x1_num_quad_points];
    ScalarT                  quad_weights[quadrature::quad_felippa_1x1_num_quad_points];

  public:
    EpsilonDivDivKerngen(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        const grid::Grid4DDataScalar< ScalarT >&                        k,
        bool                                                            treat_boundary,
        bool                                                            diagonal,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively,
        linalg::OperatorStoredMatrixMode operator_stored_matrix_mode = linalg::OperatorStoredMatrixMode::Off )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , mask_( mask )
    , k_( k )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    , operator_stored_matrix_mode_( operator_stored_matrix_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {
        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );
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

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "EpsilonDivDiv: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "EpsilonDivDiv: src/dst mismatch" );
        }

        util::Timer timer_kernel( "epsilon_divdiv_kernel" );
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
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

    using Team = Kokkos::TeamPolicy<>::member_type;

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        if ( operator_stored_matrix_mode_ != linalg::OperatorStoredMatrixMode::Off )
        {
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
                    // Kokkos::abort("Matrix not found.");
                    A[0] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
                    A[1] = assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
                }
            }

            if ( treat_boundary_ )
            {
                dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > boundary_mask;
                boundary_mask.fill( 1.0 );

                for ( int dimi = 0; dimi < 3; ++dimi )
                {
                    for ( int dimj = 0; dimj < 3; ++dimj )
                    {
                        if ( has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB ) )
                        {
                            // Inner boundary (CMB).
                            for ( int i = 0; i < num_nodes_per_wedge; i++ )
                            {
                                for ( int j = 0; j < num_nodes_per_wedge; j++ )
                                {
                                    if ( ( dimi == dimj && i != j && ( i < 3 || j < 3 ) ) or
                                         ( dimi != dimj && ( i < 3 || j < 3 ) ) )
                                    {
                                        boundary_mask(
                                            i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) = 0.0;
                                    }
                                }
                            }
                        }

                        if ( has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE ) )
                        {
                            // Outer boundary (surface).
                            for ( int i = 0; i < num_nodes_per_wedge; i++ )
                            {
                                for ( int j = 0; j < num_nodes_per_wedge; j++ )
                                {
                                    if ( ( dimi == dimj && i != j && ( i >= 3 || j >= 3 ) ) or
                                         ( dimi != dimj && ( i >= 3 || j >= 3 ) ) )
                                    {
                                        boundary_mask(
                                            i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) = 0.0;
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
            }

            if ( diagonal_ )
            {
                A[0] = A[0].diagonal();
                A[1] = A[1].diagonal();
            }

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

            dense::Vec< ScalarT, LocalMatrixDim > dst[num_wedges_per_hex_cell];

            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            for ( int dimi = 0; dimi < 3; dimi++ )
            {
                dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
                dst_d[0] = dst[0].template slice< 6 >( dimi * num_nodes_per_wedge );
                dst_d[1] = dst[1].template slice< 6 >( dimi * num_nodes_per_wedge );

                atomically_add_local_wedge_vector_coefficients(
                    dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst_d );
            }
        }
        else
        {
            double wedge_surf_phy_coords[2][3][3];
            {
                double quad_surface_coords[2][2][3];
                ;
                quad_surface_coords[0][0][0]   = grid_( local_subdomain_id, x_cell, y_cell, 0 );
                quad_surface_coords[0][0][1]   = grid_( local_subdomain_id, x_cell, y_cell, 1 );
                quad_surface_coords[0][0][2]   = grid_( local_subdomain_id, x_cell, y_cell, 2 );
                quad_surface_coords[0][1][0]   = grid_( local_subdomain_id, x_cell, y_cell + 1, 0 );
                quad_surface_coords[0][1][1]   = grid_( local_subdomain_id, x_cell, y_cell + 1, 1 );
                quad_surface_coords[0][1][2]   = grid_( local_subdomain_id, x_cell, y_cell + 1, 2 );
                quad_surface_coords[1][0][0]   = grid_( local_subdomain_id, x_cell + 1, y_cell, 0 );
                quad_surface_coords[1][0][1]   = grid_( local_subdomain_id, x_cell + 1, y_cell, 1 );
                quad_surface_coords[1][0][2]   = grid_( local_subdomain_id, x_cell + 1, y_cell, 2 );
                quad_surface_coords[1][1][0]   = grid_( local_subdomain_id, x_cell + 1, y_cell + 1, 0 );
                quad_surface_coords[1][1][1]   = grid_( local_subdomain_id, x_cell + 1, y_cell + 1, 1 );
                quad_surface_coords[1][1][2]   = grid_( local_subdomain_id, x_cell + 1, y_cell + 1, 2 );
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
            }
            double r_0 = radii_( local_subdomain_id, r_cell );
            double r_1 = radii_( local_subdomain_id, r_cell + 1 );
            double src_local_hex[3][2][6];
            int    dim;
            for ( dim = 0; dim < 3; dim += 1 )
            {
                src_local_hex[dim][0][0] = src_( local_subdomain_id, x_cell, y_cell, r_cell, dim );
                src_local_hex[dim][0][1] = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell, dim );
                src_local_hex[dim][0][2] = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell, dim );
                src_local_hex[dim][0][3] = src_( local_subdomain_id, x_cell, y_cell, r_cell + 1, dim );
                src_local_hex[dim][0][4] = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim );
                src_local_hex[dim][0][5] = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim );
                src_local_hex[dim][1][0] = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, dim );
                src_local_hex[dim][1][1] = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell, dim );
                src_local_hex[dim][1][2] = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell, dim );
                src_local_hex[dim][1][3] = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, dim );
                src_local_hex[dim][1][4] = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim );
                src_local_hex[dim][1][5] = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim );
            };
            double k_local_hex[2][6];
            k_local_hex[0][0] = k_( local_subdomain_id, x_cell, y_cell, r_cell );
            k_local_hex[0][1] = k_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
            k_local_hex[0][2] = k_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
            k_local_hex[0][3] = k_( local_subdomain_id, x_cell, y_cell, r_cell + 1 );
            k_local_hex[0][4] = k_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
            k_local_hex[0][5] = k_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
            k_local_hex[1][0] = k_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell );
            k_local_hex[1][1] = k_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
            k_local_hex[1][2] = k_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
            k_local_hex[1][3] = k_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 );
            k_local_hex[1][4] = k_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
            k_local_hex[1][5] = k_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
            double qp_array[1][3];
            double qw_array[1];
            qp_array[0][0]          = 0.33333333333333331;
            qp_array[0][1]          = 0.33333333333333331;
            qp_array[0][2]          = 0.0;
            qw_array[0]             = 1.0;
            int at_cmb_boundary     = has_flag( local_subdomain_id, x_cell, y_cell, r_cell, CMB );
            int at_surface_boundary = has_flag( local_subdomain_id, x_cell, y_cell, r_cell + 1, SURFACE );
            int cmb_shift = ( ( treat_boundary_ && diagonal_ == false && at_cmb_boundary != 0 ) ? ( 3 ) : ( 0 ) );
            int max_rad   = radii_.extent( 1 ) - 1;
            int surface_shift =
                ( ( treat_boundary_ && diagonal_ == false && at_surface_boundary != 0 ) ? ( 3 ) : ( 0 ) );
            double dst_array[3][2][6] = { 0 };
            int    w                  = 0;
            /* Apply local matrix for both wedges and accumulated for all quadrature points. */;
            for ( w = 0; w < 2; w += 1 )
            {
                int q = 0;
                for ( q = 0; q < 1; q += 1 )
                {
                    /* Coefficient evaluation on current wedge w */;
                    double tmpcse_k_eval_0 = ( 1.0 / 2.0 ) * qp_array[q][2];
                    double tmpcse_k_eval_1 = 1.0 / 2.0 - tmpcse_k_eval_0;
                    double tmpcse_k_eval_2 = tmpcse_k_eval_0 + 1.0 / 2.0;
                    double tmpcse_k_eval_3 = -qp_array[q][0] - qp_array[q][1] + 1;
                    double k_eval          = tmpcse_k_eval_1 * tmpcse_k_eval_3 * k_local_hex[w][0] +
                                    tmpcse_k_eval_1 * k_local_hex[w][1] * qp_array[q][0] +
                                    tmpcse_k_eval_1 * k_local_hex[w][2] * qp_array[q][1] +
                                    tmpcse_k_eval_2 * tmpcse_k_eval_3 * k_local_hex[w][3] +
                                    tmpcse_k_eval_2 * k_local_hex[w][4] * qp_array[q][0] +
                                    tmpcse_k_eval_2 * k_local_hex[w][5] * qp_array[q][1];

                    double scalar_grad[6][3] = { 0 };
                    /* Computation + Inversion of the Jacobian */;
                    double tmpcse_J_0 = -1.0 / 2.0 * r_0 + ( 1.0 / 2.0 ) * r_1;
                    double tmpcse_J_1 = r_0 + tmpcse_J_0 * ( qp_array[q][2] + 1 );
                    double tmpcse_J_2 = -qp_array[q][0] - qp_array[q][1] + 1;
                    double J_0_0 = tmpcse_J_1 * ( -wedge_surf_phy_coords[w][0][0] + wedge_surf_phy_coords[w][1][0] );
                    double J_0_1 = tmpcse_J_1 * ( -wedge_surf_phy_coords[w][0][0] + wedge_surf_phy_coords[w][2][0] );
                    double J_0_2 = tmpcse_J_0 * ( tmpcse_J_2 * wedge_surf_phy_coords[w][0][0] +
                                                  qp_array[q][0] * wedge_surf_phy_coords[w][1][0] +
                                                  qp_array[q][1] * wedge_surf_phy_coords[w][2][0] );
                    double J_1_0 = tmpcse_J_1 * ( -wedge_surf_phy_coords[w][0][1] + wedge_surf_phy_coords[w][1][1] );
                    double J_1_1 = tmpcse_J_1 * ( -wedge_surf_phy_coords[w][0][1] + wedge_surf_phy_coords[w][2][1] );
                    double J_1_2 = tmpcse_J_0 * ( tmpcse_J_2 * wedge_surf_phy_coords[w][0][1] +
                                                  qp_array[q][0] * wedge_surf_phy_coords[w][1][1] +
                                                  qp_array[q][1] * wedge_surf_phy_coords[w][2][1] );
                    double J_2_0 = tmpcse_J_1 * ( -wedge_surf_phy_coords[w][0][2] + wedge_surf_phy_coords[w][1][2] );
                    double J_2_1 = tmpcse_J_1 * ( -wedge_surf_phy_coords[w][0][2] + wedge_surf_phy_coords[w][2][2] );
                    double J_2_2 = tmpcse_J_0 * ( tmpcse_J_2 * wedge_surf_phy_coords[w][0][2] +
                                                  qp_array[q][0] * wedge_surf_phy_coords[w][1][2] +
                                                  qp_array[q][1] * wedge_surf_phy_coords[w][2][2] );
                    double J_det = J_0_0 * J_1_1 * J_2_2 - J_0_0 * J_1_2 * J_2_1 - J_0_1 * J_1_0 * J_2_2 +
                                   J_0_1 * J_1_2 * J_2_0 + J_0_2 * J_1_0 * J_2_1 - J_0_2 * J_1_1 * J_2_0;
                    double tmpcse_J_invT_0 = 1.0 / J_det;
                    double J_invT_cse_0_0  = tmpcse_J_invT_0 * ( J_1_1 * J_2_2 - J_1_2 * J_2_1 );
                    double J_invT_cse_0_1  = tmpcse_J_invT_0 * ( -J_1_0 * J_2_2 + J_1_2 * J_2_0 );
                    double J_invT_cse_0_2  = tmpcse_J_invT_0 * ( J_1_0 * J_2_1 - J_1_1 * J_2_0 );
                    double J_invT_cse_1_0  = tmpcse_J_invT_0 * ( -J_0_1 * J_2_2 + J_0_2 * J_2_1 );
                    double J_invT_cse_1_1  = tmpcse_J_invT_0 * ( J_0_0 * J_2_2 - J_0_2 * J_2_0 );
                    double J_invT_cse_1_2  = tmpcse_J_invT_0 * ( -J_0_0 * J_2_1 + J_0_1 * J_2_0 );
                    double J_invT_cse_2_0  = tmpcse_J_invT_0 * ( J_0_1 * J_1_2 - J_0_2 * J_1_1 );
                    double J_invT_cse_2_1  = tmpcse_J_invT_0 * ( -J_0_0 * J_1_2 + J_0_2 * J_1_0 );
                    double J_invT_cse_2_2  = tmpcse_J_invT_0 * ( J_0_0 * J_1_1 - J_0_1 * J_1_0 );
                    /* Computation of the gradient of the scalar shape functions belonging to each DoF.
      In the Eps-component-loops, we insert the gradient at the entry of the
      vectorial gradient matrix corresponding to the Eps-component. */
                    ;
                    double tmpcse_grad_i_0  = ( 1.0 / 2.0 ) * qp_array[q][2];
                    double tmpcse_grad_i_1  = tmpcse_grad_i_0 - 1.0 / 2.0;
                    double tmpcse_grad_i_2  = ( 1.0 / 2.0 ) * qp_array[q][0];
                    double tmpcse_grad_i_3  = ( 1.0 / 2.0 ) * qp_array[q][1];
                    double tmpcse_grad_i_4  = tmpcse_grad_i_2 + tmpcse_grad_i_3 - 1.0 / 2.0;
                    double tmpcse_grad_i_5  = J_invT_cse_0_2 * tmpcse_grad_i_2;
                    double tmpcse_grad_i_6  = -tmpcse_grad_i_1;
                    double tmpcse_grad_i_7  = J_invT_cse_1_2 * tmpcse_grad_i_2;
                    double tmpcse_grad_i_8  = J_invT_cse_2_2 * tmpcse_grad_i_2;
                    double tmpcse_grad_i_9  = J_invT_cse_0_2 * tmpcse_grad_i_3;
                    double tmpcse_grad_i_10 = J_invT_cse_1_2 * tmpcse_grad_i_3;
                    double tmpcse_grad_i_11 = J_invT_cse_2_2 * tmpcse_grad_i_3;
                    double tmpcse_grad_i_12 = tmpcse_grad_i_0 + 1.0 / 2.0;
                    double tmpcse_grad_i_13 = -tmpcse_grad_i_12;
                    double tmpcse_grad_i_14 = -tmpcse_grad_i_4;
                    scalar_grad[0][0]       = J_invT_cse_0_0 * tmpcse_grad_i_1 + J_invT_cse_0_1 * tmpcse_grad_i_1 +
                                        J_invT_cse_0_2 * tmpcse_grad_i_4;
                    scalar_grad[0][1] = J_invT_cse_1_0 * tmpcse_grad_i_1 + J_invT_cse_1_1 * tmpcse_grad_i_1 +
                                        J_invT_cse_1_2 * tmpcse_grad_i_4;
                    scalar_grad[0][2] = J_invT_cse_2_0 * tmpcse_grad_i_1 + J_invT_cse_2_1 * tmpcse_grad_i_1 +
                                        J_invT_cse_2_2 * tmpcse_grad_i_4;
                    scalar_grad[1][0] = J_invT_cse_0_0 * tmpcse_grad_i_6 - tmpcse_grad_i_5;
                    scalar_grad[1][1] = J_invT_cse_1_0 * tmpcse_grad_i_6 - tmpcse_grad_i_7;
                    scalar_grad[1][2] = J_invT_cse_2_0 * tmpcse_grad_i_6 - tmpcse_grad_i_8;
                    scalar_grad[2][0] = J_invT_cse_0_1 * tmpcse_grad_i_6 - tmpcse_grad_i_9;
                    scalar_grad[2][1] = J_invT_cse_1_1 * tmpcse_grad_i_6 - tmpcse_grad_i_10;
                    scalar_grad[2][2] = J_invT_cse_2_1 * tmpcse_grad_i_6 - tmpcse_grad_i_11;
                    scalar_grad[3][0] = J_invT_cse_0_0 * tmpcse_grad_i_13 + J_invT_cse_0_1 * tmpcse_grad_i_13 +
                                        J_invT_cse_0_2 * tmpcse_grad_i_14;
                    scalar_grad[3][1] = J_invT_cse_1_0 * tmpcse_grad_i_13 + J_invT_cse_1_1 * tmpcse_grad_i_13 +
                                        J_invT_cse_1_2 * tmpcse_grad_i_14;
                    scalar_grad[3][2] = J_invT_cse_2_0 * tmpcse_grad_i_13 + J_invT_cse_2_1 * tmpcse_grad_i_13 +
                                        J_invT_cse_2_2 * tmpcse_grad_i_14;
                    scalar_grad[4][0] = J_invT_cse_0_0 * tmpcse_grad_i_12 + tmpcse_grad_i_5;
                    scalar_grad[4][1] = J_invT_cse_1_0 * tmpcse_grad_i_12 + tmpcse_grad_i_7;
                    scalar_grad[4][2] = J_invT_cse_2_0 * tmpcse_grad_i_12 + tmpcse_grad_i_8;
                    scalar_grad[5][0] = J_invT_cse_0_1 * tmpcse_grad_i_12 + tmpcse_grad_i_9;
                    scalar_grad[5][1] = J_invT_cse_1_1 * tmpcse_grad_i_12 + tmpcse_grad_i_10;
                    scalar_grad[5][2] = J_invT_cse_2_1 * tmpcse_grad_i_12 + tmpcse_grad_i_11;
                    int dimj;

                    double grad_u[3][3] = { 0 };
                    double div_u        = 0.0;
                    /* In the following, we exploit the outer-product-structure of the local MV both in 
      the components of the Epsilon operators and in the local DoFs. */
                    ;
                    /* Loop to assemble the trial gradient. */;
                    for ( dimj = 0; dimj < 3; dimj += 1 )
                    {
                        if ( diagonal_ == false )
                        {
                            int node_idx;
                            for ( node_idx = cmb_shift; node_idx < 6 - surface_shift; node_idx += 1 )
                            {
                                double E_grad_trial[3][3]     = { 0 };
                                E_grad_trial[0][dimj]         = scalar_grad[node_idx][0];
                                E_grad_trial[1][dimj]         = scalar_grad[node_idx][1];
                                E_grad_trial[2][dimj]         = scalar_grad[node_idx][2];
                                double tmpcse_symgrad_trial_0 = 0.5 * E_grad_trial[0][1] + 0.5 * E_grad_trial[1][0];
                                double tmpcse_symgrad_trial_1 = 0.5 * E_grad_trial[0][2] + 0.5 * E_grad_trial[2][0];
                                double tmpcse_symgrad_trial_2 = 0.5 * E_grad_trial[1][2] + 0.5 * E_grad_trial[2][1];
                                grad_u[0][0] =
                                    1.0 * E_grad_trial[0][0] * src_local_hex[dimj][w][node_idx] + grad_u[0][0];
                                grad_u[0][1] = tmpcse_symgrad_trial_0 * src_local_hex[dimj][w][node_idx] + grad_u[0][1];
                                grad_u[0][2] = tmpcse_symgrad_trial_1 * src_local_hex[dimj][w][node_idx] + grad_u[0][2];
                                grad_u[1][0] = tmpcse_symgrad_trial_0 * src_local_hex[dimj][w][node_idx] + grad_u[1][0];
                                grad_u[1][1] =
                                    1.0 * E_grad_trial[1][1] * src_local_hex[dimj][w][node_idx] + grad_u[1][1];
                                grad_u[1][2] = tmpcse_symgrad_trial_2 * src_local_hex[dimj][w][node_idx] + grad_u[1][2];
                                grad_u[2][0] = tmpcse_symgrad_trial_1 * src_local_hex[dimj][w][node_idx] + grad_u[2][0];
                                grad_u[2][1] = tmpcse_symgrad_trial_2 * src_local_hex[dimj][w][node_idx] + grad_u[2][1];
                                grad_u[2][2] =
                                    1.0 * E_grad_trial[2][2] * src_local_hex[dimj][w][node_idx] + grad_u[2][2];
                                div_u = div_u + E_grad_trial[dimj][dimj] * src_local_hex[dimj][w][node_idx];
                            };
                        };
                    };
                    int dimi;
                    /* Loop to pair the assembled trial gradient with the test gradients. */;
                    for ( dimi = 0; dimi < 3; dimi += 1 )
                    {
                        if ( diagonal_ == false )
                        {
                            int node_idx;
                            for ( node_idx = cmb_shift; node_idx < 6 - surface_shift; node_idx += 1 )
                            {
                                double E_grad_test[3][3]     = { 0 };
                                E_grad_test[0][dimi]         = scalar_grad[node_idx][0];
                                E_grad_test[1][dimi]         = scalar_grad[node_idx][1];
                                E_grad_test[2][dimi]         = scalar_grad[node_idx][2];
                                double tmpcse_symgrad_test_0 = 0.5 * E_grad_test[0][1] + 0.5 * E_grad_test[1][0];
                                double tmpcse_symgrad_test_1 = 0.5 * E_grad_test[0][2] + 0.5 * E_grad_test[2][0];
                                double tmpcse_symgrad_test_2 = 0.5 * E_grad_test[1][2] + 0.5 * E_grad_test[2][1];
                                double tmpcse_pairing_0      = 2 * tmpcse_symgrad_test_0;
                                double tmpcse_pairing_1      = 2 * tmpcse_symgrad_test_1;
                                double tmpcse_pairing_2      = 2 * tmpcse_symgrad_test_2;
                                dst_array[dimi][w][node_idx] =
                                    k_eval *
                                        ( -0.66666666666666663 * div_u * E_grad_test[dimi][dimi] +
                                          tmpcse_pairing_0 * grad_u[0][1] + tmpcse_pairing_0 * grad_u[1][0] +
                                          tmpcse_pairing_1 * grad_u[0][2] + tmpcse_pairing_1 * grad_u[2][0] +
                                          tmpcse_pairing_2 * grad_u[1][2] + tmpcse_pairing_2 * grad_u[2][1] +
                                          2.0 * E_grad_test[0][0] * grad_u[0][0] +
                                          2.0 * E_grad_test[1][1] * grad_u[1][1] +
                                          2.0 * E_grad_test[2][2] * grad_u[2][2] ) *
                                        fabs( J_det ) * qw_array[q] +
                                    dst_array[dimi][w][node_idx];
                            };
                        };
                    };
                    int dim_diagBC;
                    /* Loop to apply BCs or only the diagonal of the operator. */;
                    for ( dim_diagBC = 0; dim_diagBC < 3; dim_diagBC += 1 )
                    {
                        if ( diagonal_ || treat_boundary_ && ( at_cmb_boundary != 0 || at_surface_boundary != 0 ) )
                        {
                            int node_idx;
                            for ( node_idx = surface_shift; node_idx < 6 - cmb_shift; node_idx += 1 )
                            {
                                double E_grad_test[3][3]   = { 0 };
                                E_grad_test[0][dim_diagBC] = scalar_grad[node_idx][0];
                                E_grad_test[1][dim_diagBC] = scalar_grad[node_idx][1];
                                E_grad_test[2][dim_diagBC] = scalar_grad[node_idx][2];

                                double grad_u_diag[3][3]     = { 0 };
                                double tmpcse_symgrad_test_0 = 0.5 * E_grad_test[0][1] + 0.5 * E_grad_test[1][0];
                                double tmpcse_symgrad_test_1 = 0.5 * E_grad_test[0][2] + 0.5 * E_grad_test[2][0];
                                double tmpcse_symgrad_test_2 = 0.5 * E_grad_test[1][2] + 0.5 * E_grad_test[2][1];
                                grad_u_diag[0][0] = 1.0 * E_grad_test[0][0] * src_local_hex[dim_diagBC][w][node_idx];
                                grad_u_diag[0][1] = tmpcse_symgrad_test_0 * src_local_hex[dim_diagBC][w][node_idx];
                                grad_u_diag[0][2] = tmpcse_symgrad_test_1 * src_local_hex[dim_diagBC][w][node_idx];
                                grad_u_diag[1][0] = tmpcse_symgrad_test_0 * src_local_hex[dim_diagBC][w][node_idx];
                                grad_u_diag[1][1] = 1.0 * E_grad_test[1][1] * src_local_hex[dim_diagBC][w][node_idx];
                                grad_u_diag[1][2] = tmpcse_symgrad_test_2 * src_local_hex[dim_diagBC][w][node_idx];
                                grad_u_diag[2][0] = tmpcse_symgrad_test_1 * src_local_hex[dim_diagBC][w][node_idx];
                                grad_u_diag[2][1] = tmpcse_symgrad_test_2 * src_local_hex[dim_diagBC][w][node_idx];
                                grad_u_diag[2][2] = 1.0 * E_grad_test[2][2] * src_local_hex[dim_diagBC][w][node_idx];
                                double tmpcse_pairing_0 = 4 * src_local_hex[dim_diagBC][w][node_idx];
                                double tmpcse_pairing_1 = 2.0 * src_local_hex[dim_diagBC][w][node_idx];
                                dst_array[dim_diagBC][w][node_idx] =
                                    k_eval *
                                        ( tmpcse_pairing_0 * pow( tmpcse_symgrad_test_0, 2 ) +
                                          tmpcse_pairing_0 * pow( tmpcse_symgrad_test_1, 2 ) +
                                          tmpcse_pairing_0 * pow( tmpcse_symgrad_test_2, 2 ) +
                                          tmpcse_pairing_1 * pow( E_grad_test[0][0], 2 ) +
                                          tmpcse_pairing_1 * pow( E_grad_test[1][1], 2 ) +
                                          tmpcse_pairing_1 * pow( E_grad_test[2][2], 2 ) -
                                          0.66666666666666663 * pow( E_grad_test[dim_diagBC][dim_diagBC], 2 ) *
                                              src_local_hex[dim_diagBC][w][node_idx] ) *
                                        fabs( J_det ) * qw_array[q] +
                                    dst_array[dim_diagBC][w][node_idx];
                            };
                        };
                    };
                };
            };
            int dim_add;
            for ( dim_add = 0; dim_add < 3; dim_add += 1 )
            {
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell, r_cell, dim_add ), dst_array[dim_add][0][0] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell, dim_add ),
                    dst_array[dim_add][0][1] + dst_array[dim_add][1][2] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell, dim_add ),
                    dst_array[dim_add][0][2] + dst_array[dim_add][1][1] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1, dim_add ), dst_array[dim_add][0][3] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1, dim_add ),
                    dst_array[dim_add][0][4] + dst_array[dim_add][1][5] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1, dim_add ),
                    dst_array[dim_add][0][5] + dst_array[dim_add][1][4] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell, dim_add ), dst_array[dim_add][1][0] );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1, dim_add ),
                    dst_array[dim_add][1][3] );
            };
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

        if ( treat_boundary_ )
        {
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > boundary_mask;
            boundary_mask.fill( 1.0 );

            for ( int dimi = 0; dimi < 3; ++dimi )
            {
                for ( int dimj = 0; dimj < 3; ++dimj )
                {
                    if ( r_cell == 0 )
                    {
                        // Inner boundary (CMB).
                        for ( int i = 0; i < 6; i++ )
                        {
                            for ( int j = 0; j < 6; j++ )
                            {
                                // on diagonal components of the vectorial diffusion operator, we exclude the diagonal entries from elimination
                                if ( ( dimi == dimj && i != j && ( i < 3 || j < 3 ) ) or
                                     ( dimi != dimj && ( i < 3 || j < 3 ) ) )
                                {
                                    boundary_mask( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) =
                                        0.0;
                                }
                            }
                        }
                    }

                    if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
                    {
                        // Outer boundary (surface).
                        for ( int i = 0; i < 6; i++ )
                        {
                            for ( int j = 0; j < 6; j++ )
                            {
                                // on diagonal components of the vectorial diffusion operator, we exclude the diagonal entries from elimination
                                if ( ( dimi == dimj && i != j && ( i >= 3 || j >= 3 ) ) or
                                     ( dimi != dimj && ( i >= 3 || j >= 3 ) ) )
                                {
                                    boundary_mask( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) =
                                        0.0;
                                }
                            }
                        }
                    }
                }
            }
            A.hadamard_product( boundary_mask );
        }

        return A;
    }

    // executes the fused local matvec on an element, given the assembled trial and test vectors
    KOKKOS_INLINE_FUNCTION void fused_local_mv(
        ScalarType                      src_local_hex[8],
        ScalarType                      dst_local_hex[8],
        const int                       wedge,
        const ScalarType                jdet_keval_quadweight,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_i,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_j,
        const int                       dimi,
        const int                       dimj,
        int                             r_cell ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        dense::Mat< ScalarType, 3, 3 > grad_u;
        ScalarType                     divu = 0.0;
        grad_u.fill( 0.0 );

        const bool at_cmb        = r_cell == 0;
        const bool at_surface    = r_cell + 1 == radii_.extent( 1 ) - 1;
        int        cmb_shift     = 0;
        int        surface_shift = 0;

        // Compute ∇u at this quadrature point.
        if ( !diagonal_ )
        {
            if ( treat_boundary_ && at_cmb )
            {
                // at the core-mantle boundary, we exclude dofs that are lower-indexed than the dof on the boundary
                cmb_shift = 3;
            }
            else if ( treat_boundary_ && at_surface )
            {
                // at the surface boundary, we exclude dofs that are higher-indexed than the dof on the boundary
                surface_shift = 3;
            }

            // accumulate the element-local gradient/divergence of the trial function (loop over columns of local matrix/local dofs)
            // by dot of trial vec and src dofs
            for ( int i = 0 + cmb_shift; i < num_nodes_per_wedge - surface_shift; i++ )
            {
                grad_u =
                    grad_u +
                    sym_grad_i[i] * src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];
                divu += sym_grad_i[i]( dimi, dimi ) *
                        src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];
            }

            // Add the test function contributions.
            // for each row of the local matrix (test-functions):
            // multiply trial part (fully assembled for the current element from loop above) with test part corresponding to the current row/dof
            // += due to contributions from other elements
            for ( int j = 0 + cmb_shift; j < num_nodes_per_wedge - surface_shift; j++ )
            {
                dst_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]] +=
                    jdet_keval_quadweight * ( 2 * ( sym_grad_j[j] ).double_contract( grad_u ) -
                                              2.0 / 3.0 * sym_grad_j[j]( dimj, dimj ) * divu );
                // for the div, we just extract the component from the gradient vector
            }
        }

        // Dirichlet DoFs are only to be eliminated on diagonal blocks of epsilon
        if ( diagonal_ || ( dimi == dimj && ( treat_boundary_ && ( at_cmb || at_surface ) ) ) )
        {
            // for the diagonal elements at the boundary, we switch the shifts
            for ( int i = 0 + surface_shift; i < num_nodes_per_wedge - cmb_shift; i++ )
            {
                const auto grad_u_diag =
                    sym_grad_i[i] * src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];
                const auto div_u_diag =
                    sym_grad_i[i]( dimi, dimi ) *
                    src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];

                dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                    jdet_keval_quadweight * ( 2 * ( sym_grad_j[i] ).double_contract( grad_u_diag ) -
                                              2.0 / 3.0 * sym_grad_j[i]( dimj, dimj ) * div_u_diag );
            }
        }
    }
};

static_assert( linalg::GCACapable< EpsilonDivDivKerngen< float > > );
static_assert( linalg::GCACapable< EpsilonDivDivKerngen< double > > );

} // namespace terra::fe::wedge::operators::shell
