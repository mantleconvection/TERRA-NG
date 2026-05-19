
#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector_q1.hpp"
#include "quadrature.hpp"
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, typename QuadRule = quadrature::QuadRuleWedge6Points< ScalarT > >
class Laplace0S1
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >                        grid_;
    grid::Grid2DDataScalar< ScalarT >                        radii_;
    grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    int local_subdomains_;
    int hex_lat_;
    int hex_rad_;
    int lat_refinement_level_;
    int block_size_;
    int blocks_per_column_;
    int blocks_;

    ScalarT r_max_;
    ScalarT r_min_;

  public:
    Laplace0S1(
        const grid::shell::DistributedDomain&                           domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&                        grid,
        const grid::Grid2DDataScalar< ScalarT >&                        radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        bool                                                            treat_boundary,
        bool                                                            diagonal,
        linalg::OperatorApplyMode         operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , mask_( mask )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {
        const grid::shell::DomainInfo& domain_info = domain_.domain_info();
        local_subdomains_                          = domain_.subdomains().size();
        hex_lat_                                   = domain_info.subdomain_num_nodes_per_side_laterally() - 1;
        hex_rad_                                   = domain_info.subdomain_num_nodes_radially() - 1;
        lat_refinement_level_                      = domain_info.diamond_lateral_refinement_level();
        const int threads_per_column               = hex_rad_;
        block_size_                                = std::min( 128, threads_per_column );
        blocks_per_column_                         = ( threads_per_column + block_size_ - 1 ) / block_size_;
        blocks_                                    = local_subdomains_ * hex_lat_ * hex_lat_ * blocks_per_column_;
        r_min_                                     = domain_info.radii()[0];
        r_max_                                     = domain_info.radii()[domain_info.radii().size() - 1];
    }

    size_t team_shmem_size( int team_size ) const
    {
        // 4 grid points with 3 components each
        constexpr int size_coords = 4 * 3 * sizeof( ScalarT );
        constexpr int size_g1     = num_wedges_per_hex_cell * QuadRule::num_quad_points * 9 * sizeof( ScalarT );
        constexpr int size_g2     = num_wedges_per_hex_cell * QuadRule::num_quad_points * sizeof( ScalarT );
        return size_coords + size_g1 + size_g2;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "laplace_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>( blocks_, block_size_ );

        util::Timer timer_kernel( "laplace_kernel" );

        Kokkos::parallel_for( "matvec", policy, *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "laplace_comm" );
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    using Team = Kokkos::TeamPolicy<>::member_type;

    KOKKOS_INLINE_FUNCTION void operator()( const Team& team ) const
    {
        int local_subdomain_id, x_cell, y_cell, r_cell;

        {
            int       tmp           = team.league_rank();
            const int r_block_index = tmp % blocks_per_column_;
            tmp /= blocks_per_column_;
            y_cell = tmp & ( hex_lat_ - 1 );
            tmp >>= lat_refinement_level_;
            x_cell = tmp & ( hex_lat_ - 1 );
            tmp >>= lat_refinement_level_;
            local_subdomain_id = tmp;

            r_cell = r_block_index * team.team_size() + team.team_rank();
        }

        constexpr int N_q         = QuadRule::num_quad_points;
        constexpr int size_coords = 4 * 3;
        constexpr int size_g1     = num_wedges_per_hex_cell * N_q * 9;
        constexpr int size_g2     = num_wedges_per_hex_cell * N_q;

        ScalarT* shmem_coords = (ScalarT*) team.team_shmem().get_shmem( size_coords * sizeof( ScalarT ) );
        ScalarT* shmem_g1     = (ScalarT*) team.team_shmem().get_shmem( size_g1 * sizeof( ScalarT ) );
        ScalarT* shmem_g2     = (ScalarT*) team.team_shmem().get_shmem( size_g2 * sizeof( ScalarT ) );

        if ( team.team_rank() == 0 )
        {
            for ( int x_offset = 0; x_offset < 2; ++x_offset )
            {
                for ( int y_offset = 0; y_offset < 2; ++y_offset )
                {
                    for ( int d = 0; d < 3; ++d )
                    {
                        shmem_coords[6 * x_offset + 3 * y_offset + d] =
                            grid_( local_subdomain_id, x_cell + x_offset, y_cell + y_offset, d );
                    }
                }
            }
        }
        team.team_barrier();

        for ( int i = team.team_rank(); i < num_wedges_per_hex_cell * N_q; i += team.team_size() )
        {
            int       j = i;
            const int q = j % N_q;
            j /= N_q;
            const int                wedge = j;
            dense::Vec< ScalarT, 3 > p1_phy, p2_phy, p3_phy;
            const int                egdew = 1 - wedge;
            for ( int d = 0; d < 3; ++d )
            {
                p1_phy( d ) = shmem_coords[6 * wedge + 3 * wedge + d];
                p2_phy( d ) = shmem_coords[6 * egdew + 3 * wedge + d];
                p3_phy( d ) = shmem_coords[6 * wedge + 3 * egdew + d];
            }
            ScalarT xi, eta, zeta;
            QuadRule::get_quad_point( q, xi, eta, zeta );
            const auto    b           = jac_lat( p1_phy, p2_phy, p3_phy, xi, eta );
            const ScalarT det_b       = b.det();
            shmem_g2[wedge * N_q + q] = Kokkos::abs( det_b );
            const auto b_inv_t = b.inv_transposed( det_b ); // this fails saying "singular matrix", i.e. det_b == 0
            for ( int d1 = 0; d1 < 3; ++d1 )
            {
                for ( int d2 = 0; d2 < 3; ++d2 )
                {
                    shmem_g1[wedge * N_q * 9 + q * 9 + 3 * d1 + d2] = b_inv_t( d1, d2 );
                }
            }
        }
        team.team_barrier();

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        const ScalarT grad_r     = grad_forward_map_rad( r_1, r_2 );
        const ScalarT grad_r_inv = 1.0 / grad_r;

        // Compute the local element matrix.
        ScalarType src_local_hex[8] = { 0 };
        ScalarType dst_local_hex[8] = { 0 };

        for ( int i = 0; i < 8; i++ )
        {
            constexpr int hex_offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
            constexpr int hex_offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
            constexpr int hex_offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

            src_local_hex[i] = src_(
                local_subdomain_id, x_cell + hex_offset_x[i], y_cell + hex_offset_y[i], r_cell + hex_offset_r[i] );
        }

        const bool at_bot_boundary = r_1 == r_min_;
        const bool at_top_boundary = r_2 == r_max_;

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < QuadRule::num_quad_points; q++ )
            {
                ScalarT xi, eta, zeta;
                QuadRule::get_quad_point( q, xi, eta, zeta );
                const ScalarT               r          = forward_map_rad< ScalarT >( r_1, r_2, zeta );
                const ScalarT               r_inv      = 1.0 / r;
                const ScalarT               abs_det    = r * r * grad_r * shmem_g2[wedge * N_q + q];
                const ScalarT               factors[3] = { r_inv, r_inv, grad_r_inv };
                dense::Mat< ScalarT, 3, 3 > J_inv_transposed;

                for ( int d1 = 0; d1 < 3; ++d1 )
                {
                    for ( int d2 = 0; d2 < 3; ++d2 )
                    {
                        J_inv_transposed( d1, d2 ) = factors[d2] * shmem_g1[wedge * N_q * 9 + q * 9 + d1 * 3 + d2];
                    }
                }

                // 2. Compute physical gradients for all nodes at this quadrature point.
                dense::Vec< ScalarType, 3 > grad_phy[num_nodes_per_wedge];
                for ( int k = 0; k < num_nodes_per_wedge; k++ )
                {
                    grad_phy[k] = J_inv_transposed * grad_shape( k, xi, eta, zeta );
                }

                if ( diagonal_ )
                {
                    diagonal( src_local_hex, dst_local_hex, wedge, abs_det, grad_phy );
                }
                else if ( treat_boundary_ && at_bot_boundary )
                {
                    // Bottom boundary dirichlet
                    dirichlet_bot( src_local_hex, dst_local_hex, wedge, abs_det, grad_phy );
                }
                else if ( treat_boundary_ && at_top_boundary )
                {
                    // Top boundary dirichlet
                    dirichlet_top( src_local_hex, dst_local_hex, wedge, abs_det, grad_phy );
                }
                else
                {
                    neumann( src_local_hex, dst_local_hex, wedge, abs_det, grad_phy );
                }
            }
        }

        for ( int i = 0; i < 8; i++ )
        {
            constexpr int hex_offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
            constexpr int hex_offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
            constexpr int hex_offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

            Kokkos::atomic_add(
                &dst_(
                    local_subdomain_id, x_cell + hex_offset_x[i], y_cell + hex_offset_y[i], r_cell + hex_offset_r[i] ),
                dst_local_hex[i] );
        }
    }

    KOKKOS_INLINE_FUNCTION void neumann(
        ScalarType*                        src_local_hex,
        ScalarType*                        dst_local_hex,
        const int                          wedge,
        const ScalarType                   abs_det,
        const dense::Vec< ScalarType, 3 >* grad_phy ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        dense::Vec< ScalarType, 3 > grad_u;
        grad_u.fill( 0.0 );
        for ( int j = 0; j < num_nodes_per_wedge; j++ )
        {
            grad_u = grad_u +
                     src_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]] * grad_phy[j];
        }

        // 4. Add the test function contributions.
        for ( int i = 0; i < num_nodes_per_wedge; i++ )
        {
            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                QuadRule::single_quad_weight * grad_phy[i].dot( grad_u ) * abs_det;
        }
    }

    KOKKOS_INLINE_FUNCTION void dirichlet_bot(
        ScalarType*                        src_local_hex,
        ScalarType*                        dst_local_hex,
        const int                          wedge,
        const ScalarType                   abs_det,
        const dense::Vec< ScalarType, 3 >* grad_phy ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        dense::Vec< ScalarType, 3 > grad_u;
        grad_u.fill( 0.0 );
        for ( int j = 3; j < num_nodes_per_wedge; j++ )
        {
            grad_u = grad_u +
                     src_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]] * grad_phy[j];
        }

        // 4. Add the test function contributions.
        for ( int i = 3; i < num_nodes_per_wedge; i++ )
        {
            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                QuadRule::single_quad_weight * grad_phy[i].dot( grad_u ) * abs_det;
        }

        // Diagonal for top part
        for ( int i = 0; i < 3; i++ )
        {
            const auto grad_u_diag =
                src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] * grad_phy[i];

            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                QuadRule::single_quad_weight * grad_phy[i].dot( grad_u_diag ) * abs_det;
        }
    }

    KOKKOS_INLINE_FUNCTION void dirichlet_top(
        ScalarType*                        src_local_hex,
        ScalarType*                        dst_local_hex,
        const int                          wedge,
        const ScalarType                   abs_det,
        const dense::Vec< ScalarType, 3 >* grad_phy ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        dense::Vec< ScalarType, 3 > grad_u;
        grad_u.fill( 0.0 );
        for ( int j = 0; j < 3; j++ )
        {
            grad_u = grad_u +
                     src_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]] * grad_phy[j];
        }

        // 4. Add the test function contributions.
        for ( int i = 0; i < 3; i++ )
        {
            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                QuadRule::single_quad_weight * grad_phy[i].dot( grad_u ) * abs_det;
        }

        // Diagonal for top part
        for ( int i = 3; i < num_nodes_per_wedge; i++ )
        {
            const auto grad_u_diag =
                src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] * grad_phy[i];

            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                QuadRule::single_quad_weight * grad_phy[i].dot( grad_u_diag ) * abs_det;
        }
    }

    KOKKOS_INLINE_FUNCTION void diagonal(
        ScalarType*                        src_local_hex,
        ScalarType*                        dst_local_hex,
        const int                          wedge,
        const ScalarType                   abs_det,
        const dense::Vec< ScalarType, 3 >* grad_phy ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        // 3. Compute ∇u at this quadrature point.
        // 4. Add the test function contributions.
        for ( int i = 0; i < num_nodes_per_wedge; i++ )
        {
            const auto grad_u =
                src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] * grad_phy[i];

            dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                QuadRule::single_quad_weight * grad_phy[i].dot( grad_u ) * abs_det;
        }
    }
};

static_assert( linalg::OperatorLike< Laplace0S1< float, quadrature::QuadRuleWedge1Point< float > > > );
static_assert( linalg::OperatorLike< Laplace0S1< float, quadrature::QuadRuleWedge6Points< float > > > );
static_assert( linalg::OperatorLike< Laplace0S1< double, quadrature::QuadRuleWedge1Point< double > > > );
static_assert( linalg::OperatorLike< Laplace0S1< double, quadrature::QuadRuleWedge6Points< double > > > );

} // namespace terra::fe::wedge::operators::shell
