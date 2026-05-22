
# Optimized Laplace Operators

*main ideas:*
- precompute a precursor to the Jacobians that only depends on the lateral position
- use shared memory to share this data across threads within a thread block ("team")

*implementation:*
- precomputation is done in the constructors of the operators
- each thread block processes hexahedra in the same column
- these hexahedra have the same lateral position and thus share the same precomputed data
- at kernel runtime, each thread block fetches the precomputed data associated to its column from global memory and stores it in shared memory
- this can be done by either one thread or multiple threads
- while this is done, all other threads in the thread block have to wait (`team.team_barrier()`)
- once this is done, all threads in the thread block can use the data from shared memory
- optionally, we perform an additional precomputation step to go from a lower precomputation strategy to a higher one at kernel runtime
- the quadrature rule and floating-point precision can be controlled using template parameters

*precomputation strategies:*
- `0`: precompute the lateral positions (Cartesian coordinates) of the vertices
  - this is done in the original laplace implementation as well
- `1`: precompute the inverse-transpose of the lateral Jacobian
  - conceptually, we split the Jacobian matrix (which depends on both the lateral position and the radial position) into a product of two matrices
  - the lateral Jacobian only depends on the lateral position (subdomain, x and y index), this is what we precompute
  - the radial Jacobian only depends on the radial position (subdomain and radial index), this is computed at runtime
- `2`: apply the inverse transpose of the lateral Jacobian to the gradient of the shape functions
- in `1` and `2` we also precompute the determinant of the lateral Jacobian
- the theory behind `1` and `2` can be read in [Nils' notes](../../../../../../../doc/other_documents/math/precomputing_gradients.tex)

*naming of the operators:*
- e.g. `Laplace1S2`
- the first number is the precomputation strategy used before kernel launch (this data is stored on the GPU's DRAM)
- the second number is the precomputation strategy used at kernel runtime before the data is stored in shared memory
- if both numbers are equal, there is no additional precomputation step at runtime (i.e. the data from the DRAM is stored directly in shared memory)
- if the second number is higher, there is an additional precomputation step at runtime (i.e. the data from the DRAM is transformed before being stored in shared memory)

*performance aspects:*

- the higher the number of the precomputation strategy, the larger the amount of precomputed data but the more it reduces the computational work (floating-point operations)
- the additional precomputation step at runtime allows us to use a lower number at first to reduce the memory volume that we have to load from the DRAM, and then switch to a higher number at runtime (since this happens locally in the thread blocks and the available block-local space in shared memory is more than large enough)
- the best performance depends on the quadrature rule and the floating-point precision:
  - 1-point quadrature rule:
    - double precision: `Laplace0S1`
    - single precision: `Laplace1S1`
  - 6-point quadrature rule:
    - double precision: `Laplace0S2`
    - single precision: `Laplace1S2`
- generally, either one of `Laplace0S1`, `Laplace0S2`, `Laplace1S1` and `Laplace1S2` yields "good" performance in all of those configurations