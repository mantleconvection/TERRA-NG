#pragma once

namespace terra::linalg {

/// @brief Concept for types that behave like vectors.
/// Requires exposing ScalarType and implementations for linear algebra operations.
template < typename T >
concept VectorLike = requires(
    const T&                                     self_const,
    T&                                           self,
    const std::vector< typename T::ScalarType >& c,
    const T&                                     x,
    T&                                           x_non_const,
    const std::vector< T >&                      xx,
    const typename T::ScalarType                 c0 )
{
    // Requires exposing the scalar type.
    typename T::ScalarType;

    // Required lincomb overload
    {
        self.lincomb_impl( c, xx, c0 )
        } -> std::same_as< void >;

    // Required dot product
    {
        self_const.dot_impl( x )
        } -> std::same_as< typename T::ScalarType >;

    // Required entries inversion
    {
        self.invert_entries_impl()
        } -> std::same_as< void >;

    // Required scale with vector
    {
        self.scale_with_vector_impl( x )
        } -> std::same_as< void >;

    // Required randomization
    {
        self.randomize_impl()
        } -> std::same_as< void >;

    // Required min magnitude
    {
        self_const.min_entry_impl()
        } -> std::same_as< typename T::ScalarType >;

    // Required min abs magnitude
    {
        self_const.min_abs_entry_impl()
        } -> std::same_as< typename T::ScalarType >;

    // Required max magnitude
    {
        self_const.max_entry_impl()
        } -> std::same_as< typename T::ScalarType >;

    // Required max abs magnitude
    {
        self_const.max_abs_entry_impl()
        } -> std::same_as< typename T::ScalarType >;

    // Required nan/inf check
    {
        self_const.has_nan_or_inf_impl()
        } -> std::same_as< bool >;

    // Required swap operation
    {
        self.swap_impl( x_non_const )
        } -> std::same_as< void >;
};

/// @brief Concept for types that behave like block 2-vectors.
/// Extends VectorLike and requires block types and accessors.
template < typename T >
concept Block2VectorLike = VectorLike< T > && requires( const T& self_const, T& self )
{
    typename T::Block1Type;
    typename T::Block2Type;

    requires VectorLike< typename T::Block1Type >;
    requires VectorLike< typename T::Block2Type >;

    {
        self_const.block_1()
        } -> std::same_as< const typename T::Block1Type& >;
    {
        self_const.block_2()
        } -> std::same_as< const typename T::Block2Type& >;

    {
        self.block_1()
        } -> std::same_as< typename T::Block1Type& >;
    {
        self.block_2()
        } -> std::same_as< typename T::Block2Type& >;
};

/// @brief Alias for the scalar type of a vector.
template < VectorLike Vector >
using ScalarOf = typename Vector::ScalarType;

/// @brief Compute a linear combination of vectors.
/// Implements: \f$ y = \sum_{i} c_i x_i + c_0 \f$
/// @param y Output vector.
/// @param c Coefficients \f$ c_i \f$.
/// @param x Input vectors \f$ x_i \f$.
/// @param c0 Scalar to add \f$ c_0 \f$.
template < VectorLike Vector >
void lincomb(
    Vector&                                  y,
    const std::vector< ScalarOf< Vector > >& c,
    const std::vector< Vector >&             x,
    const ScalarOf< Vector >&                c0 )
{
    y.lincomb_impl( c, x, c0 );
}

/// @brief Compute a linear combination of vectors with zero scalar.
/// Implements: \f$ y = \sum_{i} c_i x_i \f$
/// @param y Output vector.
/// @param c Coefficients \f$ c_i \f$.
/// @param x Input vectors \f$ x_i \f$.
template < VectorLike Vector >
void lincomb( Vector& y, const std::vector< ScalarOf< Vector > >& c, const std::vector< Vector >& x )
{
    lincomb( y, c, x, static_cast< ScalarOf< Vector > >( 0 ) );
}

/// @brief Assign a scalar value to a vector.
/// Implements: \f$ y \gets c_0 \f$
/// @param y Output vector.
/// @param c0 Scalar value \f$ c_0 \f$.
template < VectorLike Vector >
void assign( Vector& y, const ScalarOf< Vector >& c0 )
{
    lincomb( y, {}, {}, c0 );
}

/// @brief Assign one vector to another.
/// Implements: \f$ y \gets x \f$
/// @param y Output vector.
/// @param x Input vector.
template < VectorLike Vector >
void assign( Vector& y, const Vector& x )
{
    lincomb( y, { static_cast< ScalarOf< Vector > >( 1 ) }, { x } );
}

/// @brief Compute the dot product of two vectors.
/// Implements: \f$ y \cdot x = \sum_{i} y_i x_i \f$
/// @param y First vector.
/// @param x Second vector.
/// @return Dot product value.
template < VectorLike Vector >
ScalarOf< Vector > dot( const Vector& y, const Vector& x )
{
    return y.dot_impl( x );
}

/// @brief Invert the entries of a vector.
/// For each entry \f$ y_i \f$, computes \f$ y_i = 1 / y_i \f$.
/// @param y Vector to invert.
template < VectorLike Vector >
void invert_entries( Vector& y )
{
    y.invert_entries_impl();
}

/// @brief Scale a vector in place with another vector.
/// For each entry \f$ y_i \f$, computes \f$ y_i = y_i \cdot x_i \f$.
/// @param y Vector to scale.
/// @param x Scaling vector.
template < VectorLike Vector >
void scale_in_place( Vector& y, const Vector& x )
{
    y.scale_with_vector_impl( x );
}

/// @brief Randomize the entries of a vector.
/// Sets each entry of \f$ y \f$ to a random value.
/// @param y Vector to randomize.
template < VectorLike Vector >
void randomize( Vector& y )
{
    y.randomize_impl();
}

/// @brief Compute the minimum entry of a vector.
/// Computes: \f$ \min_i y_i \f$
/// @param y Input vector.
/// @return Minimum value
template < VectorLike Vector >
ScalarOf< Vector > min_entry( const Vector& y )
{
    return y.min_entry_impl();
}

/// @brief Compute the minimum absolute entry of a vector.
/// Computes: \f$ \min_i |y_i| \f$
/// @param y Input vector.
/// @return Minimum absolute value
template < VectorLike Vector >
ScalarOf< Vector > min_abs_entry( const Vector& y )
{
    return y.min_abs_entry_impl();
}

/// @brief Compute the maximum entry of a vector.
/// Computes: \f$ \max_i y_i \f$
/// @param y Input vector.
/// @return Maximum value
template < VectorLike Vector >
ScalarOf< Vector > max_entry( const Vector& y )
{
    return y.max_entry_impl();
}

/// @brief Compute the maximum absolute entry of a vector.
/// Computes: \f$ \max_i |y_i| \f$
/// @param y Input vector.
/// @return Maximum absolute value
template < VectorLike Vector >
ScalarOf< Vector > max_abs_entry( const Vector& y )
{
    return y.max_abs_entry_impl();
}

/// @brief Compute the infinity norm (max absolute entry) of a vector.
/// Implements: \f$ \|y\|_\infty = \max_i |y_i| \f$
/// @param y Input vector.
/// @return Infinity norm value.
template < VectorLike Vector >
ScalarOf< Vector > norm_inf( const Vector& y )
{
    return y.max_abs_entry_impl();
}

/// @brief Compute the 2-norm (Euclidean norm) of a vector.
/// Implements: \f$ \|y\|_2 = \sqrt{ \sum_i y_i^2 } \f$
/// @param y Input vector.
/// @return 2-norm value.
template < VectorLike Vector >
ScalarOf< Vector > norm_2( const Vector& y )
{
    const auto dot_prod = dot( y, y );
    return std::sqrt( dot_prod );
}

/// @brief Compute the scaled 2-norm of a vector.
/// Implements: \f$ \|y\|_2^{\text{scaled}} = \sqrt{ (\sum_i y_i^2) \cdot s } \f$
/// @param y Input vector.
/// @param scaling_factor_under_the_root Scaling factor \f$ s \f$ under the square root.
/// @return Scaled 2-norm value.
template < VectorLike Vector >
ScalarOf< Vector > norm_2_scaled( const Vector& y, const ScalarOf< Vector >& scaling_factor_under_the_root )
{
    const auto dot_prod = dot( y, y );
    return std::sqrt( dot_prod * scaling_factor_under_the_root );
}

/// @brief Check if a vector contains NaN or inf entries.
/// Returns true if any entry of \f$ y \f$ is NaN or inf.
/// @param y Input vector.
/// @return True if NaN or inf is present, false otherwise.
template < VectorLike Vector >
bool has_nan_or_inf( const Vector& y )
{
    return y.has_nan_or_inf_impl();
}

/// @brief Swap the contents of two vectors.
/// Exchanges the entries of \f$ x \f$ and \f$ y \f$.
/// @param x First vector.
/// @param y Second vector.
template < VectorLike Vector >
void swap( Vector& x, Vector& y )
{
    y.swap_impl( x );
}

namespace detail {

/// @brief Dummy vector class for concept checks and testing.
/// Implements required vector operations as no-ops.
template < typename ScalarT >
class DummyVector
{
  public:
    /// @brief Scalar type used by the vector.
    using ScalarType = ScalarT;

    /// @brief Dummy implementation of linear combination.
    void lincomb_impl( const std::vector< ScalarType >& c, const std::vector< DummyVector >& x, const ScalarType c0 )
    {
        (void) c;
        (void) x;
        (void) c0;
    }

    /// @brief Dummy implementation of dot product.
    ScalarType dot_impl( const DummyVector& x ) const
    {
        (void) x;
        return 0;
    }

    /// @brief Dummy implementation of invert entries.
    void invert_entries_impl() {}

    /// @brief Dummy implementation of scale with vector.
    void scale_with_vector_impl( const DummyVector& x ) { (void) x; }

    /// @brief Dummy implementation of randomize.
    void randomize_impl() {}

    /// @brief Dummy implementation of min entry.
    ScalarType min_entry_impl() const { return 0; }

    /// @brief Dummy implementation of min absolute entry.
    ScalarType min_abs_entry_impl() const { return 0; }

    /// @brief Dummy implementation of max entry.
    ScalarType max_entry_impl() const { return 0; }

    /// @brief Dummy implementation of max absolute entry.
    ScalarType max_abs_entry_impl() const { return 0; }

    /// @brief Dummy implementation of NaN check.
    bool has_nan_or_inf_impl() const { return false; }

    /// @brief Dummy implementation of swap.
    void swap_impl( DummyVector< ScalarType >& other ) { (void) other; }
};

/// @brief Dummy block 2-vector class for concept checks and testing.
/// Contains two DummyVector blocks.
template < typename ScalarT >
class DummyBlock2Vector
{
  public:
    /// @brief Scalar type used by the block vector.
    using ScalarType = ScalarT;

    /// @brief Type of the first block.
    using Block1Type = DummyVector< ScalarType >;
    /// @brief Type of the second block.
    using Block2Type = DummyVector< ScalarType >;

    /// @brief Dummy implementation of linear combination.
    void lincomb_impl(
        const std::vector< ScalarType >&        c,
        const std::vector< DummyBlock2Vector >& x,
        const ScalarType                        c0 )
    {
        (void) c;
        (void) x;
        (void) c0;
    }

    /// @brief Dummy implementation of dot product.
    ScalarType dot_impl( const DummyBlock2Vector& x ) const
    {
        (void) x;
        return 0;
    }

    /// @brief Dummy implementation of invert entries.
    void invert_entries_impl() {}

    /// @brief Dummy implementation of scale with vector.
    void scale_with_vector_impl( const DummyBlock2Vector& x ) { (void) x; }

    /// @brief Dummy implementation of randomize.
    void randomize_impl() {}

    /// @brief Dummy implementation of min entry.
    ScalarType min_entry_impl() const { return 0; }

    /// @brief Dummy implementation of min absolute entry.
    ScalarType min_abs_entry_impl() const { return 0; }

    /// @brief Dummy implementation of max entry.
    ScalarType max_entry_impl() const { return 0; }

    /// @brief Dummy implementation of max absolute entry.
    ScalarType max_abs_entry_impl() const { return 0; }

    /// @brief Dummy implementation of NaN check.
    bool has_nan_or_inf_impl() const { return false; }

    /// @brief Dummy implementation of swap.
    void swap_impl( DummyBlock2Vector& other ) { (void) other; }

    /// @brief Get const reference to block 1.
    const DummyVector< ScalarType >& block_1() const { return block_1_; }
    /// @brief Get const reference to block 2.
    const DummyVector< ScalarType >& block_2() const { return block_2_; }

    /// @brief Get mutable reference to block 1.
    DummyVector< ScalarType >& block_1() { return block_1_; }
    /// @brief Get mutable reference to block 2.
    DummyVector< ScalarType >& block_2() { return block_2_; }

  private:
    DummyVector< ScalarType > block_1_;
    DummyVector< ScalarType > block_2_;
};

/// @brief Static assertion to check VectorLike concept for DummyVector.
static_assert( VectorLike< DummyVector< double > > );
/// @brief Static assertion to check Block2VectorLike concept for DummyBlock2Vector.
static_assert( Block2VectorLike< DummyBlock2Vector< double > > );

} // namespace detail

} // namespace terra::linalg
