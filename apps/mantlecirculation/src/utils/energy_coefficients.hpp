#pragma once

#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "grid/grid_types.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_fv.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"

namespace terra::mantlecirculation {

template < typename T >
concept CoefficientC = requires( T t, int id, int x, int y, int r, int w, dense::Vec< double, 3 > q ) {
    { t( id, x, y, r, w, q ) } -> std::convertible_to< double >;
};

struct DiffusionCoefficient
{
    double Pe;

    grid::Grid2DDataScalar< double > density_;
    grid::Grid2DDataScalar< double > cp_;
    grid::Grid2DDataScalar< double > radii_;

    KOKKOS_INLINE_FUNCTION double operator()(
        const int                     local_subdomain_id,
        const int                     x_cell,
        const int                     y_cell,
        const int                     r_cell,
        const int                     wedge,
        const dense::Vec< double, 3 > qp ) const
    {
        const double r_1 = radii_( local_subdomain_id, r_cell );
        const double r_2 = radii_( local_subdomain_id, r_cell + 1 );

        const double r_phys = terra::fe::wedge::forward_map_rad( r_1, r_2, qp( 2 ) );

        const double rho_1 = density_( local_subdomain_id, r_cell );
        const double rho_2 = density_( local_subdomain_id, r_cell + 1 );

        const double t     = ( r_phys - r_1 ) / ( r_2 - r_1 );
        const double rho_q = rho_1 + t * ( rho_2 - rho_1 );

        double density_val = rho_q;

        const double cp_1 = cp_( local_subdomain_id, r_cell );
        const double cp_2 = cp_( local_subdomain_id, r_cell + 1 );

        const double cp_q = cp_1 + t * ( cp_2 - cp_1 );

        double cp_val = cp_q;

        return 1.0 / ( Pe * density_val * cp_val );
    }
};

struct InternalHeatingCoefficient
{
    bool internal_heating;

    double h_tilde;

    grid::Grid2DDataScalar< double > cp_;
    grid::Grid2DDataScalar< double > radii_;

    KOKKOS_INLINE_FUNCTION double operator()(
        const int                     local_subdomain_id,
        const int                     x_cell,
        const int                     y_cell,
        const int                     r_cell,
        const int                     wedge,
        const dense::Vec< double, 3 > qp ) const
    {
        if( internal_heating )
        {
            const double r_1 = radii_( local_subdomain_id, r_cell );
            const double r_2 = radii_( local_subdomain_id, r_cell + 1 );

            const double r_phys = terra::fe::wedge::forward_map_rad( r_1, r_2, qp( 2 ) );

            const double cp_1 = cp_( local_subdomain_id, r_cell );
            const double cp_2 = cp_( local_subdomain_id, r_cell + 1 );

            const double t    = ( r_phys - r_1 ) / ( r_2 - r_1 );
            const double cp_q = cp_1 + t * ( cp_2 - cp_1 );

            return h_tilde / ( cp_q );
        }
        else
        {
            return 0.0;
        }
    }
};

struct AdiabaticCoefficient
{
    bool adiabatic_heating;

    double                           Di;
    grid::Grid2DDataScalar< double > alpha_;
    grid::Grid2DDataScalar< double > cp_;
    grid::Grid2DDataScalar< double > radii_;

    KOKKOS_INLINE_FUNCTION double operator()(
        const int                     local_subdomain_id,
        const int                     x_cell,
        const int                     y_cell,
        const int                     r_cell,
        const int                     wedge,
        const dense::Vec< double, 3 > qp ) const
    {
        if( adiabatic_heating )
        {
            const double r_1 = radii_( local_subdomain_id, r_cell );
            const double r_2 = radii_( local_subdomain_id, r_cell + 1 );

            const double r_phys = terra::fe::wedge::forward_map_rad( r_1, r_2, qp( 2 ) );

            const double cp_1 = cp_( local_subdomain_id, r_cell );
            const double cp_2 = cp_( local_subdomain_id, r_cell + 1 );

            const double t = ( r_phys - r_1 ) / ( r_2 - r_1 );

            double cp_val = cp_1 + t * ( cp_2 - cp_1 );

            const double alpha_1 = alpha_( local_subdomain_id, r_cell );
            const double alpha_2 = alpha_( local_subdomain_id, r_cell + 1 );

            double alpha_val = alpha_1 + t * ( alpha_2 - alpha_1 );

            return Di * alpha_val / cp_val;
        }
        else
        {
            return 0.0;
        }
    }
};

struct ShearHeatingCoefficient
{
    bool shear_heating;

    double Di;
    double Pe;
    double Ra;

    grid::Grid2DDataScalar< double > density_;
    grid::Grid2DDataScalar< double > cp_;
    grid::Grid2DDataScalar< double > radii_;

    KOKKOS_INLINE_FUNCTION double operator()(
        const int                     local_subdomain_id,
        const int                     x_cell,
        const int                     y_cell,
        const int                     r_cell,
        const int                     wedge,
        const dense::Vec< double, 3 > qp ) const
    {
        if( shear_heating )
        {
            const double r_1 = radii_( local_subdomain_id, r_cell );
            const double r_2 = radii_( local_subdomain_id, r_cell + 1 );

            const double r_phys = terra::fe::wedge::forward_map_rad( r_1, r_2, qp( 2 ) );

            const double rho_1 = density_( local_subdomain_id, r_cell );
            const double rho_2 = density_( local_subdomain_id, r_cell + 1 );

            const double t     = ( r_phys - r_1 ) / ( r_2 - r_1 );
            const double rho_q = rho_1 + t * ( rho_2 - rho_1 );

            double density_val = rho_q;

            const double cp_1 = cp_( local_subdomain_id, r_cell );
            const double cp_2 = cp_( local_subdomain_id, r_cell + 1 );

            const double cp_q = cp_1 + t * ( cp_2 - cp_1 );

            double cp_val = cp_q;

            return (Di * Pe / Ra) * (1.0 / (density_val * cp_val));
        }
        else
        {
            return 0.0;
        }
    }
};

template <
    CoefficientC DiffusionCoeff,
    CoefficientC InternalHeatingCoeff,
    CoefficientC AdiabaticHeatingCoeff,
    CoefficientC ShearHeatingCoeff >
struct EnergyEquationCoeffT
{
    using DiffusionCoeffT        = DiffusionCoeff;
    using InternalHeatingCoeffT  = InternalHeatingCoeff;
    using AdiabaticHeatingCoeffT = AdiabaticHeatingCoeff;
    using ShearHeatingCoeffT     = ShearHeatingCoeff;
};
} // namespace terra::mantlecirculation
