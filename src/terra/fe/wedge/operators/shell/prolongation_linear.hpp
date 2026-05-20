

#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/shell/grid_transfer_linear.hpp"
#include "grid/grid_types.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class ProlongationLinear
{
  public:
    using SrcVectorType           = linalg::VectorQ1Scalar< double >;
    using DstVectorType           = linalg::VectorQ1Scalar< double >;
    using ScalarType              = ScalarT;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, 6, 6, 2 >;

  private:
    bool storeLMatrices_ =
        false; // set to let apply_impl() know, that it should store the local matrices after assembling them
    bool applyStoredLMatrices_ =
        false; // set to make apply_impl() load and use the stored LMatrices for the operator application

    Grid4DDataLocalMatrices LMatrices_;

    grid::Grid3DDataVec< ScalarType, 3 > grid_fine_;
    grid::Grid2DDataScalar< ScalarType > radii_fine_;

    linalg::OperatorApplyMode operator_apply_mode_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    explicit ProlongationLinear(
        const grid::Grid3DDataVec< ScalarType, 3 >& grid_fine,
        const grid::Grid2DDataScalar< ScalarType >& radii_fine,
        linalg::OperatorApplyMode                   operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : grid_fine_( grid_fine )
    , radii_fine_( radii_fine )
    , operator_apply_mode_( operator_apply_mode )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( storeLMatrices_ or applyStoredLMatrices_ )
            assert( LMatrices_.data() != nullptr );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( dst_.extent( 1 ) != grid_fine_.extent( 1 ) )
        {
            throw std::runtime_error(
                "Prolongation: dst and grid_fine must have the same number of cells in the x direction." );
        }

        if ( dst_.extent( 3 ) != radii_fine_.extent( 1 ) )
        {
            throw std::runtime_error(
                "Prolongation: dst and radii_fine must have the same number of cells in the r direction." );
        }

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

        // Looping over the fine grid.
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
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_fine, const int y_fine, const int r_fine ) const
    {
        dense::Vec< int, 4 > fine_hex_fine = { local_subdomain_id, x_fine, y_fine, r_fine };
        (void) fine_hex_fine; // unused

        if ( x_fine % 2 == 0 && y_fine % 2 == 0 && r_fine % 2 == 0 )
        {
            const auto x_coarse = x_fine / 2;
            const auto y_coarse = y_fine / 2;
            const auto r_coarse = r_fine / 2;

            dst_( local_subdomain_id, x_fine, y_fine, r_fine ) +=
                src_( local_subdomain_id, x_coarse, y_coarse, r_coarse );

            return;
        }

        const auto r_coarse_bot = r_fine < dst_.extent( 3 ) - 1 ? r_fine / 2 : r_fine / 2 - 1;
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

            dst_( local_subdomain_id, x_fine, y_fine, r_fine ) +=
                weights( 0 ) * src_( local_subdomain_id, x_coarse, y_coarse, r_coarse_bot ) +
                weights( 1 ) * src_( local_subdomain_id, x_coarse, y_coarse, r_coarse_top );

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

        dst_( local_subdomain_id, x_fine, y_fine, r_fine ) +=
            weights( 0 ) * src_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_bot ) +
            weights( 0 ) * src_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_bot ) +
            weights( 1 ) * src_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_top ) +
            weights( 1 ) * src_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_top );
    }
};

template < typename ScalarT >
class ProlongationVecLinear
{
  public:
    using SrcVectorType           = linalg::VectorQ1Vec< double >;
    using DstVectorType           = linalg::VectorQ1Vec< double >;
    using ScalarType              = ScalarT;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, 6, 6, 2 >;

  private:
    bool storeLMatrices_ =
        false; // set to let apply_impl() know, that it should store the local matrices after assembling them
    bool applyStoredLMatrices_ =
        false; // set to make apply_impl() load and use the stored LMatrices for the operator application

    Grid4DDataLocalMatrices LMatrices_;

    grid::Grid3DDataVec< ScalarType, 3 > grid_fine_;
    grid::Grid2DDataScalar< ScalarType > radii_fine_;

    linalg::OperatorApplyMode operator_apply_mode_;

    grid::Grid4DDataVec< ScalarType, 3 > src_;
    grid::Grid4DDataVec< ScalarType, 3 > dst_;

  public:
    explicit ProlongationVecLinear(
        const grid::Grid3DDataVec< ScalarType, 3 >& grid_fine,
        const grid::Grid2DDataScalar< ScalarType >& radii_fine,
        linalg::OperatorApplyMode                   operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : grid_fine_( grid_fine )
    , radii_fine_( radii_fine )
    , operator_apply_mode_( operator_apply_mode )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( storeLMatrices_ or applyStoredLMatrices_ )
            assert( LMatrices_.data() != nullptr );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( dst_.extent( 1 ) != grid_fine_.extent( 1 ) )
        {
            throw std::runtime_error(
                "Prolongation: dst and grid_fine must have the same number of cells in the x direction." );
        }

        if ( dst_.extent( 3 ) != radii_fine_.extent( 1 ) )
        {
            throw std::runtime_error(
                "Prolongation: dst and radii_fine must have the same number of cells in the r direction." );
        }

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

        // Looping over the fine grid.
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
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_fine, const int y_fine, const int r_fine ) const
    {
        for ( int dim = 0; dim < 3; ++dim )
        {
            dense::Vec< int, 4 > fine_hex_fine = { local_subdomain_id, x_fine, y_fine, r_fine };
            (void) fine_hex_fine; // unused

            if ( x_fine % 2 == 0 && y_fine % 2 == 0 && r_fine % 2 == 0 )
            {
                const auto x_coarse = x_fine / 2;
                const auto y_coarse = y_fine / 2;
                const auto r_coarse = r_fine / 2;

                dst_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) +=
                    src_( local_subdomain_id, x_coarse, y_coarse, r_coarse, dim );

                return;
            }

            const auto r_coarse_bot = r_fine < dst_.extent( 3 ) - 1 ? r_fine / 2 : r_fine / 2 - 1;
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

                dst_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) +=
                    weights( 0 ) * src_( local_subdomain_id, x_coarse, y_coarse, r_coarse_bot, dim ) +
                    weights( 1 ) * src_( local_subdomain_id, x_coarse, y_coarse, r_coarse_top, dim );

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

            dst_( local_subdomain_id, x_fine, y_fine, r_fine, dim ) +=
                weights( 0 ) * src_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_bot, dim ) +
                weights( 0 ) * src_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_bot, dim ) +
                weights( 1 ) * src_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_top, dim ) +
                weights( 1 ) * src_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_top, dim );
        }
    }
};
} // namespace terra::fe::wedge::operators::shell