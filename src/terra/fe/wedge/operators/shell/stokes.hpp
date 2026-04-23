
#pragma once

#include "communication/shell/communication.hpp"
#include "divergence_kerngen.hpp"
#include "gradient_kerngen.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "util/timer.hpp"
#include "vector_laplace.hpp"
#include "zero.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT, int VecDim = 3 >
class Stokes
{
  public:
    using SrcVectorType = linalg::VectorQ1IsoQ2Q1< ScalarT, VecDim >;
    using DstVectorType = linalg::VectorQ1IsoQ2Q1< ScalarT, VecDim >;
    using ScalarType    = ScalarT;

    using Block11Type = VectorLaplace< ScalarType, VecDim >;
    using Block12Type = GradientKerngen< ScalarType >;
    using Block21Type = DivergenceKerngen< ScalarType >;
    using Block22Type = Zero< ScalarType >;

  private:
    VectorLaplace< ScalarType, VecDim > A_;
    GradientKerngen< ScalarType >              B_T_;
    DivergenceKerngen< ScalarType >            B_;
    Zero< ScalarType >                  O_;

    bool diagonal_;

  public:
    Stokes(
        const grid::shell::DistributedDomain&                           domain_fine,
        const grid::shell::DistributedDomain&                           domain_coarse,
        const grid::Grid3DDataVec< ScalarType, 3 >&                     grid,
        const grid::Grid2DDataScalar< ScalarType >&                     radii,
        const grid::Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& boundary_mask_data_fine,
        BoundaryConditions                                              bcs,
        bool                                                            diagonal )
     : A_( domain_fine, grid, radii, boundary_mask_data_fine, bcs, diagonal )
    , B_T_( domain_fine, domain_coarse, grid, radii, boundary_mask_data_fine, bcs )
    , B_( domain_fine, domain_coarse, grid, radii, boundary_mask_data_fine, bcs )
    , diagonal_( diagonal )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "stokes_apply" );

        if ( !diagonal_ )
        {
            A_.set_operator_apply_and_communication_modes(
                linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::SkipCommunication );
        }

        B_T_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Add, linalg::OperatorCommunicationMode::CommunicateAdditively );
        B_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );

        apply( A_, src.block_1(), dst.block_1() );

        if ( !diagonal_ )
        {
            apply( B_T_, src.block_2(), dst.block_1() );
            apply( B_, src.block_1(), dst.block_2() );
        }

        A_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );
        B_T_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );
        B_.set_operator_apply_and_communication_modes(
            linalg::OperatorApplyMode::Replace, linalg::OperatorCommunicationMode::CommunicateAdditively );
    }

    const Block11Type& block_11() const { return A_; }

    const Block12Type& block_12() const
    {
        if ( diagonal_ )
        {
            throw std::runtime_error( "block_12() is not implemented for diagonal Stokes operator" );
        }

        return B_T_;
    }

    const Block21Type& block_21() const
    {
        if ( diagonal_ )
        {
            throw std::runtime_error( "block_21() is not implemented for diagonal Stokes operator" );
        }

        return B_;
    }

    const Block22Type& block_22() const { return O_; }

    Block11Type& block_11() { return A_; }
    Block12Type& block_12()
    {
        if ( diagonal_ )
        {
            throw std::runtime_error( "block_12() is not implemented for diagonal Stokes operator" );
        }

        return B_T_;
    }
    Block21Type& block_21()
    {
        if ( diagonal_ )
        {
            throw std::runtime_error( "block_21() is not implemented for diagonal Stokes operator" );
        }

        return B_;
    }
    Block22Type& block_22() { return O_; }
};

static_assert( linalg::Block2x2OperatorLike< Stokes< double, 3 > > );

} // namespace terra::fe::wedge::operators::shell