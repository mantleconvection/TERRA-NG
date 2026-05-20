

#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/shell/grid_transfer_constant.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class ProlongationConstant
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    linalg::OperatorApplyMode operator_apply_mode_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    explicit ProlongationConstant( linalg::OperatorApplyMode operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : operator_apply_mode_( operator_apply_mode )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) )
        {
            throw std::runtime_error( "Prolongation: src and dst must have the same number of subdomains." );
        }

        for ( int i = 1; i <= 3; i++ )
        {
            if ( 2 * ( src_.extent( i ) - 1 ) != dst_.extent( i ) - 1 )
            {
                throw std::runtime_error( "Prolongation: src and dst must have a compatible number of cells." );
            }
        }

        // Looping over the coarse grid.
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

            if ( fine_stencil_x >= 0 && fine_stencil_x < dst_.extent( 1 ) && fine_stencil_y >= 0 &&
                 fine_stencil_y < dst_.extent( 2 ) && fine_stencil_r >= 0 && fine_stencil_r < dst_.extent( 3 ) )
            {
                const auto weight = wedge::shell::prolongation_constant_weight< ScalarType >(
                    fine_stencil_x, fine_stencil_y, fine_stencil_r, x_coarse, y_coarse, r_coarse );

                Kokkos::atomic_add(
                    &dst_( local_subdomain_id, fine_stencil_x, fine_stencil_y, fine_stencil_r ),
                    weight * src_( local_subdomain_id, x_coarse, y_coarse, r_coarse ) );
            }
        }
    }
};

template < typename ScalarT, int VecDim = 3 >
class ProlongationVecConstant
{
  public:
    using SrcVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

  private:
    linalg::OperatorApplyMode operator_apply_mode_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

  public:
    explicit ProlongationVecConstant(
        linalg::OperatorApplyMode operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : operator_apply_mode_( operator_apply_mode )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) )
        {
            throw std::runtime_error( "Prolongation: src and dst must have the same number of subdomains." );
        }

        for ( int i = 1; i <= 3; i++ )
        {
            if ( 2 * ( src_.extent( i ) - 1 ) != dst_.extent( i ) - 1 )
            {
                throw std::runtime_error( "Prolongation: src and dst must have a compatible number of cells." );
            }
        }

        // Looping over the coarse grid.
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

            if ( fine_stencil_x >= 0 && fine_stencil_x < dst_.extent( 1 ) && fine_stencil_y >= 0 &&
                 fine_stencil_y < dst_.extent( 2 ) && fine_stencil_r >= 0 && fine_stencil_r < dst_.extent( 3 ) )
            {
                const auto weight = wedge::shell::prolongation_constant_weight< ScalarType >(
                    fine_stencil_x, fine_stencil_y, fine_stencil_r, x_coarse, y_coarse, r_coarse );

                for ( int d = 0; d < VecDim; ++d )
                {
                    Kokkos::atomic_add(
                        &dst_( local_subdomain_id, fine_stencil_x, fine_stencil_y, fine_stencil_r, d ),
                        weight * src_( local_subdomain_id, x_coarse, y_coarse, r_coarse, d ) );
                }
            }
        }
    }
};

} // namespace terra::fe::wedge::operators::shell