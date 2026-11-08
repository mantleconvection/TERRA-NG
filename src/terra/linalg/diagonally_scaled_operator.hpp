#pragma once

#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_w1.hpp"

namespace terra::linalg {

/// @brief Given some operator \f$K\f$ and a vector \f$v\f$, this operator is equivalent to an operator \f$A\f$ defined as
///
/// \f[ A = \mathrm{diag}(v) K \f]
///
template < OperatorLike OperatorT >
class DiagonallyScaledOperator
{
  public:
    using OperatorType  = OperatorT;
    using SrcVectorType = SrcOf< OperatorT >;
    using DstVectorType = DstOf< OperatorT >;
    using ScalarType    = ScalarOf< DstVectorType >;

  private:
    OperatorT&     op_;
    SrcVectorType& diag_;

  public:
    explicit DiagonallyScaledOperator( OperatorT& op, SrcVectorType& diag )
    : op_( op )
    , diag_( diag ) {};

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        apply( op_, src, dst );
        scale_in_place( dst, diag_ );
    }
};

static_assert(
    linalg::OperatorLike< DiagonallyScaledOperator< fe::wedge::operators::shell::LaplaceSimple< double > > > );

} // namespace terra::linalg