

#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/shell/grid_transfer_linear.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class RestrictionConstant
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_coarse_;

    linalg::OperatorApplyMode operator_apply_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > mask_src_;

  public:
    RestrictionConstant(
        const grid::shell::DistributedDomain& domain_coarse,
        linalg::OperatorApplyMode             operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : domain_coarse_( domain_coarse )
    , operator_apply_mode_( operator_apply_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain_coarse )
    , recv_buffers_( domain_coarse )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

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

        // Looping over the coarse grid.
        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 0, 0, 0 },
                {
                    dst_.extent( 0 ),
                    dst_.extent( 1 ),
                    dst_.extent( 2 ),
                    dst_.extent( 3 ),
                } ),
            *this );

        Kokkos::fence();

        // Additive communication.

        communication::shell::pack_send_and_recv_local_subdomain_boundaries(
            domain_coarse_, dst_, send_buffers_, recv_buffers_ );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_coarse_, dst_, recv_buffers_ );
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_coarse, const int y_coarse, const int r_coarse ) const
    {
        const auto x_fine = 2 * x_coarse;
        const auto y_fine = 2 * y_coarse;
        const auto r_fine = 2 * r_coarse;

        dense::Vec< int, 3 > offsets[21];
        wedge::shell::prolongation_constant_fine_grid_stencil_offsets_at_coarse_vertex( offsets );

        for ( const auto& offset : offsets )
        {
            const auto fine_stencil_x = x_fine + offset( 0 );
            const auto fine_stencil_y = y_fine + offset( 1 );
            const auto fine_stencil_r = r_fine + offset( 2 );

            if ( fine_stencil_x >= 0 && fine_stencil_x < src_.extent( 1 ) && fine_stencil_y >= 0 &&
                 fine_stencil_y < src_.extent( 2 ) && fine_stencil_r >= 0 && fine_stencil_r < src_.extent( 3 ) )
            {
                const auto weight = wedge::shell::prolongation_constant_weight< ScalarType >(
                    fine_stencil_x, fine_stencil_y, fine_stencil_r, x_coarse, y_coarse, r_coarse );

                const auto mask_weight =
                    util::has_flag(
                        mask_src_( local_subdomain_id, fine_stencil_x, fine_stencil_y, fine_stencil_r ),
                        grid::NodeOwnershipFlag::OWNED ) ?
                        1.0 :
                        0.0;

                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse ),
                    weight * mask_weight * src_( local_subdomain_id, fine_stencil_x, fine_stencil_y, fine_stencil_r ) );
            }
        }
    }
};

template < typename ScalarT, int VecDim = 3 >
class RestrictionVecConstant
{
  public:
    using SrcVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_coarse_;

    linalg::OperatorApplyMode operator_apply_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, VecDim > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarType, VecDim > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

    grid::Grid4DDataScalar< grid::NodeOwnershipFlag > mask_src_;

  public:
    RestrictionVecConstant(
        const grid::shell::DistributedDomain& domain_coarse,
        linalg::OperatorApplyMode             operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : domain_coarse_( domain_coarse )
    , operator_apply_mode_( operator_apply_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain_coarse )
    , recv_buffers_( domain_coarse )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

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

        // Looping over the coarse grid.
        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4, Kokkos::Iterate::Right, Kokkos::Iterate::Right > >(
                { 0, 0, 0, 0 },
                {
                    dst_.extent( 0 ),
                    dst_.extent( 1 ),
                    dst_.extent( 2 ),
                    dst_.extent( 3 ),
                } ),
            *this );

        Kokkos::fence();

        // Additive communication.

        communication::shell::pack_send_and_recv_local_subdomain_boundaries(
            domain_coarse_, dst_, send_buffers_, recv_buffers_ );
        communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_coarse_, dst_, recv_buffers_ );
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_coarse, const int y_coarse, const int r_coarse ) const
    {
        const auto x_fine = 2 * x_coarse;
        const auto y_fine = 2 * y_coarse;
        const auto r_fine = 2 * r_coarse;

        dense::Vec< int, 3 > offsets[21];
        wedge::shell::prolongation_constant_fine_grid_stencil_offsets_at_coarse_vertex( offsets );

        for ( const auto& offset : offsets )
        {
            const auto fine_stencil_x = x_fine + offset( 0 );
            const auto fine_stencil_y = y_fine + offset( 1 );
            const auto fine_stencil_r = r_fine + offset( 2 );

            if ( fine_stencil_x >= 0 && fine_stencil_x < src_.extent( 1 ) && fine_stencil_y >= 0 &&
                 fine_stencil_y < src_.extent( 2 ) && fine_stencil_r >= 0 && fine_stencil_r < src_.extent( 3 ) )
            {
                const auto weight = wedge::shell::prolongation_constant_weight< ScalarType >(
                    fine_stencil_x, fine_stencil_y, fine_stencil_r, x_coarse, y_coarse, r_coarse );

                const auto mask_weight =
                    util::has_flag(
                        mask_src_( local_subdomain_id, fine_stencil_x, fine_stencil_y, fine_stencil_r ),
                        grid::NodeOwnershipFlag::OWNED ) ?
                        1.0 :
                        0.0;

                for ( int d = 0; d < VecDim; ++d )
                {
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse, d ),
                        weight * mask_weight *
                            src_( local_subdomain_id, fine_stencil_x, fine_stencil_y, fine_stencil_r, d ) );
                }
            }
        }
    }
};
} // namespace terra::fe::wedge::operators::shell