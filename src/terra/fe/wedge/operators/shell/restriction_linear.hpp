

#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/shell/grid_transfer_linear.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class RestrictionLinear
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< double >;
    using DstVectorType = linalg::VectorQ1Scalar< double >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid3DDataVec< ScalarType, 3 > grid_fine_;
    grid::Grid2DDataScalar< ScalarType > radii_fine_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< double > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< double > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > mask_src_;

  public:
    RestrictionLinear(
        const grid::shell::DistributedDomain&       domain_coarse,
        const grid::Grid3DDataVec< ScalarType, 3 >& grid_fine,
        const grid::Grid2DDataScalar< ScalarType >& radii_fine )
    : domain_coarse_( domain_coarse )
    , grid_fine_( grid_fine )
    , radii_fine_( radii_fine )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain_coarse )
    , recv_buffers_( domain_coarse )
    {
        if ( 2 * ( domain_coarse.domain_info().subdomain_num_nodes_per_side_laterally() - 1 ) !=
             grid_fine.extent( 1 ) - 1 )
        {
            throw std::runtime_error(
                "Restriction: domain_coarse and grid_fine must have compatible number of cells." );
        }

        if ( 2 * ( domain_coarse.domain_info().subdomain_num_nodes_radially() - 1 ) != radii_fine.extent( 1 ) - 1 )
        {
            throw std::runtime_error(
                "Restriction: domain_coarse and radii_fine must have compatible number of cells." );
        }
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        // Not quite sure currently how to implement additive update (like r += R * f).
        // For now, only implementing replacement update: r = R * f.
        assign( dst, 0 );

        src_      = src.grid_data();
        dst_      = dst.grid_data();
        mask_src_ = src.mask_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) )
        {
            throw std::runtime_error( "Restriction: src and dst must have the same number of subdomains." );
        }

        for ( int i = 1; i <= 3; i++ )
        {
            if ( src_.extent( i ) - 1 != 2 * ( dst_.extent( i ) - 1 ) )
            {
                throw std::runtime_error( "Restriction: src and dst must have a compatible number of cells." );
            }
        }

        // Looping over the fine grid.
        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 0, 0, 0 },
                {
                    src_.extent( 0 ),
                    src_.extent( 1 ),
                    src_.extent( 2 ),
                    src_.extent( 3 ),
                } ),
            *this );

        Kokkos::fence();

        // Additive communication.

        communication::shell::pack_send_and_recv_local_subdomain_boundaries(
            domain_coarse_, dst_, send_buffers_, recv_buffers_ );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_coarse_, dst_, recv_buffers_ );
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_fine, const int y_fine, const int r_fine ) const
    {
        // Only pushing from owned nodes.
        if ( !util::has_flag(
                 mask_src_( local_subdomain_id, x_fine, y_fine, r_fine ), grid::NodeOwnershipFlag::OWNED ) )
        {
            return;
        }

        if ( x_fine % 2 == 0 && y_fine % 2 == 0 && r_fine % 2 == 0 )
        {
            const auto x_coarse = x_fine / 2;
            const auto y_coarse = y_fine / 2;
            const auto r_coarse = r_fine / 2;

            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse ),
                src_( local_subdomain_id, x_fine, y_fine, r_fine ) );

            return;
        }

        const auto r_coarse_bot = r_fine < src_.extent( 3 ) - 1 ? r_fine / 2 : r_fine / 2 - 1;
        const auto r_coarse_top = r_coarse_bot + 1;

        if ( x_fine % 2 == 0 && y_fine % 2 == 0 )
        {
            const auto x_coarse = x_fine / 2;
            const auto y_coarse = y_fine / 2;

            const auto weights = wedge::shell::prolongation_linear_weights(
                dense::Vec< int, 4 >{ local_subdomain_id, x_fine, y_fine, r_fine },
                dense::Vec< int, 4 >{ local_subdomain_id, x_coarse, y_coarse, r_coarse_bot },
                grid_fine_,
                radii_fine_ );

            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse_bot ),
                weights( 0 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine ) );
            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse_top ),
                weights( 1 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine ) );

            return;
        }

        int x_coarse_0 = -1;
        int x_coarse_1 = -1;

        int y_coarse_0 = -1;
        int y_coarse_1 = -1;

        if ( x_fine % 2 == 0 )
        {
            // "Vertical" edge.
            x_coarse_0 = x_fine / 2;
            x_coarse_1 = x_fine / 2;

            y_coarse_0 = y_fine / 2;
            y_coarse_1 = y_fine / 2 + 1;
        }
        else if ( y_fine % 2 == 0 )
        {
            // "Horizontal" edge.
            x_coarse_0 = x_fine / 2;
            x_coarse_1 = x_fine / 2 + 1;

            y_coarse_0 = y_fine / 2;
            y_coarse_1 = y_fine / 2;
        }
        else
        {
            // "Diagonal" edge.
            x_coarse_0 = x_fine / 2 + 1;
            x_coarse_1 = x_fine / 2;

            y_coarse_0 = y_fine / 2;
            y_coarse_1 = y_fine / 2 + 1;
        }

        const auto weights = wedge::shell::prolongation_linear_weights(
            dense::Vec< int, 4 >{ local_subdomain_id, x_fine, y_fine, r_fine },
            dense::Vec< int, 4 >{ local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_bot },
            dense::Vec< int, 4 >{ local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_bot },
            grid_fine_,
            radii_fine_ );

        Kokkos::atomic_add(
            &dst_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_bot ),
            weights( 0 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine ) );
        Kokkos::atomic_add(
            &dst_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_bot ),
            weights( 0 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine ) );
        Kokkos::atomic_add(
            &dst_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_top ),
            weights( 1 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine ) );
        Kokkos::atomic_add(
            &dst_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_top ),
            weights( 1 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine ) );
    }

#if 0
    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_coarse, const int y_coarse, const int r_coarse ) const
    {
        const auto x_fine = 2 * x_coarse;
        const auto y_fine = 2 * y_coarse;
        const auto r_fine = 2 * r_coarse;

        dense::Vec< int, 3 > offsets[21];
        wedge::shell::prolongation_fine_grid_stencil_offsets_at_coarse_vertex( offsets );

        for ( const auto& offset : offsets )
        {
            const auto fine_stencil_x = x_fine + offset( 0 );
            const auto fine_stencil_y = y_fine + offset( 1 );
            const auto fine_stencil_r = r_fine + offset( 2 );

            if ( fine_stencil_x >= 0 && fine_stencil_x < src_.extent( 1 ) && fine_stencil_y >= 0 &&
                 fine_stencil_y < src_.extent( 2 ) && fine_stencil_r >= 0 && fine_stencil_r < src_.extent( 3 ) )
            {
                const auto weight = wedge::shell::prolongation_weight< ScalarType >(
                    fine_stencil_x, fine_stencil_y, fine_stencil_r, x_coarse, y_coarse, r_coarse );

                const auto mask_weight =
                    util::check_bits(
                        mask_src_( local_subdomain_id, fine_stencil_x, fine_stencil_y, fine_stencil_r ),
                        grid::mask_owned() ) ?
                        1.0 :
                        0.0;

                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse ),
                    weight * mask_weight * src_( local_subdomain_id, fine_stencil_x, fine_stencil_y, fine_stencil_r ) );
            }
        }
    }
#endif
};

template < typename ScalarT >
class RestrictionVecLinear
{
  public:
    using SrcVectorType = linalg::VectorQ1Vec< double >;
    using DstVectorType = linalg::VectorQ1Vec< double >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid3DDataVec< ScalarType, 3 > grid_fine_;
    grid::Grid2DDataScalar< ScalarType > radii_fine_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< double, 3 > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< double, 3 > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, 3 > src_;
    grid::Grid4DDataVec< ScalarType, 3 > dst_;

    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > mask_src_;

  public:
    RestrictionVecLinear(
        const grid::shell::DistributedDomain&       domain_coarse,
        const grid::Grid3DDataVec< ScalarType, 3 >& grid_fine,
        const grid::Grid2DDataScalar< ScalarType >& radii_fine )
    : domain_coarse_( domain_coarse )
    , grid_fine_( grid_fine )
    , radii_fine_( radii_fine )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain_coarse )
    , recv_buffers_( domain_coarse )
    {
        if ( 2 * ( domain_coarse.domain_info().subdomain_num_nodes_per_side_laterally() - 1 ) !=
             grid_fine.extent( 1 ) - 1 )
        {
            throw std::runtime_error(
                "Restriction: domain_coarse and grid_fine must have compatible number of cells." );
        }

        if ( 2 * ( domain_coarse.domain_info().subdomain_num_nodes_radially() - 1 ) != radii_fine.extent( 1 ) - 1 )
        {
            throw std::runtime_error(
                "Restriction: domain_coarse and radii_fine must have compatible number of cells." );
        }
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        // Not quite sure currently how to implement additive update (like r += R * f).
        // For now, only implementing replacement update: r = R * f.
        assign( dst, 0 );

        src_      = src.grid_data();
        dst_      = dst.grid_data();
        mask_src_ = src.mask_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) )
        {
            throw std::runtime_error( "Restriction: src and dst must have the same number of subdomains." );
        }

        for ( int i = 1; i <= 3; i++ )
        {
            if ( src_.extent( i ) - 1 != 2 * ( dst_.extent( i ) - 1 ) )
            {
                throw std::runtime_error( "Restriction: src and dst must have a compatible number of cells." );
            }
        }

        // Looping over the fine grid.
        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 0, 0, 0 },
                {
                    src_.extent( 0 ),
                    src_.extent( 1 ),
                    src_.extent( 2 ),
                    src_.extent( 3 ),
                } ),
            *this );

        Kokkos::fence();

        // Additive communication.

        communication::shell::pack_send_and_recv_local_subdomain_boundaries(
            domain_coarse_, dst_, send_buffers_, recv_buffers_ );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_coarse_, dst_, recv_buffers_ );
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_fine, const int y_fine, const int r_fine ) const
    {
        for ( int dim = 0; dim < 3; ++dim )
        {
            // Only pushing from owned nodes.
            if ( !util::has_flag(
                     mask_src_( local_subdomain_id, x_fine, y_fine, r_fine ), grid::NodeOwnershipFlag::OWNED ) )
            {
                return;
            }

            if ( x_fine % 2 == 0 && y_fine % 2 == 0 && r_fine % 2 == 0 )
            {
                const auto x_coarse = x_fine / 2;
                const auto y_coarse = y_fine / 2;
                const auto r_coarse = r_fine / 2;

                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse, dim ),
                    src_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) );

                return;
            }

            const auto r_coarse_bot = r_fine < src_.extent( 3 ) - 1 ? r_fine / 2 : r_fine / 2 - 1;
            const auto r_coarse_top = r_coarse_bot + 1;

            if ( x_fine % 2 == 0 && y_fine % 2 == 0 )
            {
                const auto x_coarse = x_fine / 2;
                const auto y_coarse = y_fine / 2;

                const auto weights = wedge::shell::prolongation_linear_weights(
                    dense::Vec< int, 4 >{ local_subdomain_id, x_fine, y_fine, r_fine },
                    dense::Vec< int, 4 >{ local_subdomain_id, x_coarse, y_coarse, r_coarse_bot },
                    grid_fine_,
                    radii_fine_ );

                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse_bot, dim ),
                    weights( 0 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) );
                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse_top, dim ),
                    weights( 1 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) );

                return;
            }

            int x_coarse_0 = -1;
            int x_coarse_1 = -1;

            int y_coarse_0 = -1;
            int y_coarse_1 = -1;

            if ( x_fine % 2 == 0 )
            {
                // "Vertical" edge.
                x_coarse_0 = x_fine / 2;
                x_coarse_1 = x_fine / 2;

                y_coarse_0 = y_fine / 2;
                y_coarse_1 = y_fine / 2 + 1;
            }
            else if ( y_fine % 2 == 0 )
            {
                // "Horizontal" edge.
                x_coarse_0 = x_fine / 2;
                x_coarse_1 = x_fine / 2 + 1;

                y_coarse_0 = y_fine / 2;
                y_coarse_1 = y_fine / 2;
            }
            else
            {
                // "Diagonal" edge.
                x_coarse_0 = x_fine / 2 + 1;
                x_coarse_1 = x_fine / 2;

                y_coarse_0 = y_fine / 2;
                y_coarse_1 = y_fine / 2 + 1;
            }

            const auto weights = wedge::shell::prolongation_linear_weights(
                dense::Vec< int, 4 >{ local_subdomain_id, x_fine, y_fine, r_fine },
                dense::Vec< int, 4 >{ local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_bot },
                dense::Vec< int, 4 >{ local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_bot },
                grid_fine_,
                radii_fine_ );

            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_bot, dim ),
                weights( 0 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) );
            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_bot, dim ),
                weights( 0 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) );
            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_top, dim ),
                weights( 1 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) );
            Kokkos::atomic_add(
                &dst_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_top, dim ),
                weights( 1 ) * src_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) );
        }
    }
};
} // namespace terra::fe::wedge::operators::shell