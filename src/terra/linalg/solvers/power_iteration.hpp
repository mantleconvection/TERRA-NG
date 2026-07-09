#pragma once

#include "terra/linalg/operator.hpp"

#include <limits>

namespace terra::linalg::solvers {

/// @brief Power iteration to estimate the largest eigenvalue of a
/// row-normalized operator D^{-1}A, where D^{-1} is the inverted diagonal of A.
///
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
template < terra::linalg::OperatorLike OperatorT >
//template<typename OperatorT>
double power_iteration(
    OperatorT&          op,
    SrcOf< OperatorT >& tmpIt,
    SrcOf< OperatorT >& tmpAux,
    //VectorQ1Scalar< double>& tmpIt,
    //VectorQ1Scalar< double>& tmpAux,
    const int iterations )
{
    // TODO typecheck on src dst

    // randomize start
    randomize( tmpIt );

    // normalize
    auto norm = linalg::norm_2( tmpIt );
    if ( norm < std::numeric_limits< double >::min() )
    {
        return 1.0; // safe fallback
    }
    lincomb( tmpIt, { typename std::remove_reference_t< decltype( tmpIt ) >::ScalarType( 1.0 / norm ) }, { tmpIt }, 0.0 );

    // apply operator
    apply( op, tmpIt, tmpAux );

    auto radius = 0.0;
    for ( int iteration = 0; iteration < iterations; ++iteration )
    {
        // normalize
        norm = linalg::norm_2( tmpAux );
        if ( norm < std::numeric_limits< double >::min() )
        {
            return radius > 0.0 ? radius : 1.0;
        }
        lincomb( tmpIt, { typename std::remove_reference_t< decltype( tmpIt ) >::ScalarType( 1.0 / norm ) }, { tmpAux }, 0.0 );

        // apply operator
        apply( op, tmpIt, tmpAux );

        // compute radius
        radius = dot( tmpIt, tmpAux );
    }
    return radius;
}

} // namespace terra::linalg::solvers