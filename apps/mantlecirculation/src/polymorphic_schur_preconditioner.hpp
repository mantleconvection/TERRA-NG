#pragma once

#include <memory>
#include <utility>

#include "linalg/operator.hpp"
#include "linalg/solvers/solver.hpp"

namespace terra::mantlecirculation {

/// @brief Type-erased wrapper that satisfies `SolverLike` for any concrete
///        solver whose `solve_impl(OperatorType&, SolutionVectorType&, const
///        RHSVectorType&)` matches our pressure-mass Schur slot.
///
/// Lets `StokesContext` instantiate `BlockTriangularPreconditioner2x2` once
/// and decide between the mass-Schur and w-BFBT Schur preconditioners at
/// runtime.  Held by `shared_ptr` so the wrapper itself is cheaply copyable
/// (required by `BlockTriangularPreconditioner2x2` which stores its block
/// preconditioner by value).
template < linalg::OperatorLike OperatorT >
class PolymorphicSchurPreconditioner
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = linalg::SrcOf< OperatorType >;
    using RHSVectorType      = linalg::DstOf< OperatorType >;

  private:
    struct ImplBase
    {
        virtual ~ImplBase() = default;
        virtual void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b ) = 0;
    };

    template < typename Concrete >
    struct ImplHolder final : ImplBase
    {
        Concrete solver;
        explicit ImplHolder( Concrete s ) : solver( std::move( s ) ) {}
        void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b ) override
        {
            solver.solve_impl( A, x, b );
        }
    };

  public:
    PolymorphicSchurPreconditioner() = default;

    template < typename Concrete >
    static PolymorphicSchurPreconditioner make( Concrete solver )
    {
        PolymorphicSchurPreconditioner p;
        p.impl_ = std::make_shared< ImplHolder< Concrete > >( std::move( solver ) );
        return p;
    }

    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        impl_->solve_impl( A, x, b );
    }

  private:
    std::shared_ptr< ImplBase > impl_;
};

} // namespace terra::mantlecirculation
