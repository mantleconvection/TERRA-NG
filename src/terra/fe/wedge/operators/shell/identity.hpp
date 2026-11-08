
#pragma once

#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_w1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class Identity
{
  private:
    linalg::OperatorApplyMode operator_apply_mode_;

  public:
    using SrcVectorType = linalg::VectorW1Scalar< double >;
    using DstVectorType = linalg::VectorW1Scalar< double >;
    using ScalarType    = ScalarT;

    explicit Identity( linalg::OperatorApplyMode operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : operator_apply_mode_( operator_apply_mode ) {};

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, src );
        }
        else if ( operator_apply_mode_ == linalg::OperatorApplyMode::Add )
        {
            lincomb( dst, { static_cast< ScalarT >( 1.0 ), static_cast< ScalarT >( 1.0 ) }, { dst, src } );
        }
    }
};

static_assert( linalg::OperatorLike< Identity< double > > );

} // namespace terra::fe::wedge::operators::shell