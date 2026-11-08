

#pragma once

#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_w1.hpp"
#include "terra/kernels/common/grid_operations.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class Injection
{
  public:
    using SrcVectorType = linalg::VectorW1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorW1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    void apply_impl_grid_data( const grid::Grid4DDataScalar< ScalarT >& src, grid::Grid4DDataScalar< ScalarT >& dst )
    {
        src_ = src;
        dst_ = dst;

        if ( src_.extent( 0 ) != dst_.extent( 0 ) )
        {
            throw std::runtime_error( "Injection: src and dst must have the same number of subdomains." );
        }

        for ( int i = 1; i <= 3; i++ )
        {
            if ( src_.extent( i ) - 1 != 2 * ( dst_.extent( i ) - 1 ) )
            {
                throw std::runtime_error( "Injection: src and dst must have a compatible number of cells." );
            }
        }

        // Looping over the coarse grid.
        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
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

  public:
    Injection( const grid::shell::DistributedDomain& domain_coarse )
    : domain_coarse_( domain_coarse )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        assign( dst, 0 );

        apply_impl_grid_data( src.grid_data(), dst.grid_data() );
    }

    void apply_impl( const grid::Grid4DDataScalar< ScalarT >& src, grid::Grid4DDataScalar< ScalarT >& dst )
    {
        kernels::common::set_constant( dst, 0.0 );

        apply_impl_grid_data( src, dst );
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_coarse, const int y_coarse, const int r_coarse ) const
    {
        const auto x_fine = 2 * x_coarse;
        const auto y_fine = 2 * y_coarse;
        const auto r_fine = 2 * r_coarse;

        dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse ) = src_( local_subdomain_id, x_fine, y_fine, r_fine );
    }
};

// Vector injection is implemented by applying the scalar Injection operator once per component.
// See inject() overload below.

template < typename ScalarT >
inline void inject(
    const grid::shell::DistributedDomain&    domain_coarse,
    const linalg::VectorW1Scalar< ScalarT >& src,
    linalg::VectorW1Scalar< ScalarT >&       dst )
{
    Injection< ScalarT > inj( domain_coarse );
    inj.apply_impl( src, dst );
}

template < typename ScalarT, int VecDim = 3 >
inline void inject(
    const grid::shell::DistributedDomain&         domain_coarse,
    const linalg::VectorW1Vec< ScalarT, VecDim >& src,
    linalg::VectorW1Vec< ScalarT, VecDim >&       dst )
{
    Injection< ScalarT > inj( domain_coarse );

    for ( int d = 0; d < VecDim; ++d )
    {
        inj.apply_impl( src.grid_data().comp_[d], dst.grid_data().comp_[d] );
    }
}
} // namespace terra::fe::wedge::operators::shell
