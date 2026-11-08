
#pragma once

#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_w1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class Zero
{
  private:
    linalg::OperatorApplyMode operator_apply_mode_;

  public:
    using SrcVectorType = linalg::VectorW1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorW1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

    explicit Zero( linalg::OperatorApplyMode operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : operator_apply_mode_( operator_apply_mode ) {};

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }
    }
};

static_assert( linalg::OperatorLike< Zero< double > > );

} // namespace terra::fe::wedge::operators::shell