

#pragma once

#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class Injection
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    Injection( const grid::shell::DistributedDomain& domain_coarse )
    : domain_coarse_( domain_coarse )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        assign( dst, 0 );

        src_ = src.grid_data();
        dst_ = dst.grid_data();

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

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_coarse, const int y_coarse, const int r_coarse ) const
    {
        const auto x_fine = 2 * x_coarse;
        const auto y_fine = 2 * y_coarse;
        const auto r_fine = 2 * r_coarse;

        dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse ) = src_( local_subdomain_id, x_fine, y_fine, r_fine );
    }
};

template < typename ScalarT, int VecDim = 3 >
class InjectionVec
{
  public:
    using SrcVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

  public:
    InjectionVec( const grid::shell::DistributedDomain& domain_coarse )
    : domain_coarse_( domain_coarse )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        assign( dst, 0 );

        src_ = src.grid_data();
        dst_ = dst.grid_data();

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

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_coarse, const int y_coarse, const int r_coarse ) const
    {
        const auto x_fine = 2 * x_coarse;
        const auto y_fine = 2 * y_coarse;
        const auto r_fine = 2 * r_coarse;

        for ( int d = 0; d < VecDim; ++d )
        {
            dst_( local_subdomain_id, x_coarse, y_coarse, r_coarse, d ) =
                src_( local_subdomain_id, x_fine, y_fine, r_fine, d );
        }
    }
};

template < typename ScalarT >
inline void inject(
    Injection< ScalarT >                     inj,
    const linalg::VectorQ1Scalar< ScalarT >& src,
    linalg::VectorQ1Scalar< ScalarT >&       dst )
{
    inj.apply_impl( src, dst );
}

template < typename ScalarT, int VecDim >
inline void inject(
    InjectionVec< ScalarT, VecDim >               inj,
    const linalg::VectorQ1Vec< ScalarT, VecDim >& src,
    linalg::VectorQ1Vec< ScalarT, VecDim >&       dst )
{
    inj.apply_impl( src, dst );
}
} // namespace terra::fe::wedge::operators::shell
