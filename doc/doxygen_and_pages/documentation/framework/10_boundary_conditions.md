# Boundary conditions {#boundary-conditions}

## Strong Dirichlet boundary condition enforcement

\note
See
also: [strong_algebraic_dirichlet_enforcement_poisson_like()](@ref terra::fe::strong_algebraic_dirichlet_enforcement_poisson_like)
and similar functions in that file for the treatment of right-hand sides.

<p></p>

\note
A great explanation is given in [Wolfgang Bangerth's
lectures](https://www.math.colostate.edu/~bangerth/videos.676.21.65.html).

### Poisson-like problems

#### Algebraic elimination

Consider the linear system

\f[
A x = b ,
\f]

arising from a Poisson-like operator without any boundary treatment (\f$A\f$ is the "Neumann" operator), with Dirichlet
conditions prescribed on a subset of degrees of freedom \f$d\f$:

\f[
x_d = g_d .
\f]

Reorder unknowns as

\f[
x =
\begin{pmatrix} x_i \\ x_d \end{pmatrix}, \quad
A =
\begin{pmatrix}
A_{ii} & A_{id} \\
A_{di} & A_{dd}
\end{pmatrix}, \quad
b =
\begin{pmatrix} b_i \\ b_d \end{pmatrix}.
\f]

Define the lifting vector

\f[
g =
\begin{pmatrix}
0 \\ g_d
\end{pmatrix},
\f]

i.e. zero on interior degrees of freedom and equal to the prescribed
Dirichlet values on boundary degrees of freedom. Also define

\f[
\hat x =
\begin{pmatrix}
x_{i} \\ 0
\end{pmatrix},
\f]

i.e., \f$\hat x_d = 0\f$, and \f$x = \hat x + g\f$.
Substitution into the linear system gives the lifted equation

\f[
\begin{aligned}
Ax &= A(\hat x + g) = A \hat x + A g = b \\
A \hat x &= b - A g .
\end{aligned}
\f]

To enforce \f$\hat x_d=0\f$ strongly while preserving symmetry, eliminate all
off-diagonal couplings to Dirichlet degrees of freedom:

\f[
A_{\mathrm{elim}} =
\begin{pmatrix}
A_{ii} & 0 \\
0 & A_{dd}
\end{pmatrix}.
\f]

Define the modified right-hand side

\f[
b_{\mathrm{elim}} = b - A g ,
\f]

and then overwrite its Dirichlet components by

\f[
b_{\mathrm{elim},d} = A_{dd} g_d,
\f]

s.t., overall
\f[
b_{\mathrm{elim}} =
\begin{pmatrix} b_{\mathrm{elim},i} \\ b_{\mathrm{elim},d} \end{pmatrix}
=
\begin{pmatrix} b_i - A_{id} g_d \\ A_{dd} g_d \end{pmatrix}.
\f]

The eliminated system is

\f[
A_{\mathrm{elim}} x = b_{\mathrm{elim}}.
\f]

The interior equations read

\f[
A_{ii} x_i = b_i - A_{id} g_d ,
\f]

while the boundary equations enforce

\f[
A_{dd} x_d = A_{dd} g_d \;\Rightarrow\; x_d = g_d .
\f]

Thus, the Dirichlet conditions are imposed strongly, and the interior solution
satisfies the standard reduced system.

This procedure is equivalent to a symmetric Gaussian elimination of Dirichlet
degrees of freedom. It preserves:

- symmetry of the operator,
- positive definiteness for Poisson-like problems,
- exact strong enforcement of Dirichlet data.

The diagonal entries corresponding to Dirichlet nodes are retained, which is
advantageous for conditioning and iterative solvers.

Let \f$g\f$ be the lifting vector. Then:

\f[
y = A g ,
\f]
\f[
b_{\mathrm{elim}} = b - y ,
\f]
\f[
b_{\mathrm{elim},d} = \mathrm{diag}(A)\, g_d .
\f]

The operator \f$A_{\mathrm{elim}}\f$ is applied by symmetrically zeroing all
off-diagonal entries in rows and columns corresponding to Dirichlet degrees
of freedom. In a matrix-free context, this symmetric zeroing can be implemented
by applying it to each local element matrix. In fact, this is exactly what is
happening in the operators implemented in this framework (more details below).

Solving \f$A_{\mathrm{elim}} x = b_{\mathrm{elim}}\f$ yields the solution of
the original Dirichlet problem.

#### Realization in the framework

To solve the problem
\f[
A_{\mathrm{elim}} x = b_{\mathrm{elim}}
\f]
that has been introduced above, we need several ingredients:

- the original ("Neumann") operator \f$A\f$ (evaluating volume integrals everywhere),
- the operator \f$A_{\mathrm{elim}}\f$ (that one is typically already implemented in the framework and elimination is
  enabled by setting a flag in the constructor that switches between the "Neumann" and "eliminated" versions),
- the diagonal entries of \f$A\f$, \f$\mathrm{diag}(A)\f$ (note that
  \f$\mathrm{diag}(A) = \mathrm{diag}(A_{\mathrm{elim}})\f$ by design),
- the interpolated boundary conditions \f$g\f$ (the "lifting vector" - zero on interior degrees of freedom and equal
  to the prescribed Dirichlet values on boundary degrees of freedom),
- the original forcing vector \f$b\f$ (without any boundary treatment, but already in the form of a coefficient vector,
  typically the result of evaluating a linear form on some analytical function).

Having the operators instantiated, it only remains to build the right-hand side vector \f$b_{\mathrm{elim}}\f$ and
solve the system. To build the vector, helper functions such as
@ref terra::fe::strong_algebraic_dirichlet_enforcement_poisson_like should be used that perform all the necessary steps.
Concretely, the steps are:

1. \f$ g_A \gets A g \f$,
2. \f$ g_D \gets \mathrm{diag}(A) g \f$,
3. \f$ b_\text{elim} \gets b - g_A \f$,
4. \f$ b_{\text{elim},d} \gets g_D \f$, i.e., setting the RHS \f$ b_\mathrm{elim} \f$ to \f$ g_D \f$ at the boundary
   nodes.

It then remains to pass the system:
\f[
A_{\mathrm{elim}} x = b_{\mathrm{elim}}
\f]
to a linear solver.

\f$ x \f$ is the solution of the original problem. No further boundary corrections are necessary.

#### Matrix-free implementation

The elimination procedure above is implemented in the framework by zeroing out the off-diagonal entries in the local
element matrices. Let, for instance, \f$W\f$ be a wedge element with 6 nodes. Of those 6 nodes, 3 are interior
(\f$j = 1, 2, 3\f$), and the remaining 3 are on the Dirichlet boundary (\f$j = 4, 5, 6\f$).
The local element matrix \f$A^W\f$ is of size \f$6\times 6\f$, and has the following block-structure (similar to the
global matrix after rearranging the degrees of freedom above):
\f[
A^W =
\begin{pmatrix}
A^W_{ii} & A^W_{id} \\
A^W_{di} & A^W_{dd}
\end{pmatrix}.
\f]
The symmetric elimination process simply zeroes out the off-diagonal entries in rows and columns corresponding to the
boundary nodes:
\f[
A_{\mathrm{elim}}^W =
\begin{pmatrix}
A^W_{ii} & 0 \\
0 & \mathrm{diag}(A^W_{dd})
\end{pmatrix}.
\f]
This approach can be realized on-the-fly in a matrix-free context, without the need to store the full local element
matrices.

### Stokes-like problems

Often we need to enforce Dirichlet boundary conditions on the velocity field in the context of a Stokes-like problem.
The approach is very similar to the one described above, but with a few subtleties:

Assuming an inf-sup stable discretization, we consider the discrete Stokes problem
\f[
Kx = \begin{pmatrix}
A & B \\
C & 0
\end{pmatrix}
\begin{pmatrix}
u \\
p
\end{pmatrix}
= \begin{pmatrix}
f_u \\
f_p
\end{pmatrix}
=b.
\f]
and a lifting vector \f$g\f$ (with non-zero entries on the Dirichlet boundary nodes in the velocity component).
In this definition, we explicitly allow for non-symmetric versions of \f$K\f$ due to the possible compressibility of the
fluid.

The treatment of the RHS vector, i.e., the computation of \f$b_\mathrm{elim}\f$, works just as in the Poisson-like
case. Note that although only velocity components are prescribed, the elimination may also alter the values in the
pressure component of the RHS.

We perform the steps just as above:

1. \f$ g_K \gets K g \f$,
2. \f$ g_D \gets \mathrm{diag}(K) g \f$,
3. \f$ b_\text{elim} \gets b - g_K \f$,
4. \f$ b_{\text{elim},d} \gets g_D \f$, i.e., setting the RHS \f$ b_\mathrm{elim} \f$ to \f$ g_D \f$ at the boundary
   nodes (only velocity components are prescribed).

The other subtlety is that of course the elimination process must also be carried out on the gradient and
divergence-like operators \f$B\f$ and \f$C\f$. Specifically, we need to zero out rows in \f$B\f$ and columns in \f$C\f$
corresponding to the boundary nodes. Just as in the Poisson-like case, this can be realized on-the-fly in a matrix-free
context.

## Plate boundaries

Plate boundary conditions are just Dirichlet boundary conditions for the velocity on the plate surface.
Nothing special here from the algebraic point of view. Refer to the Stokes-like case above.

## Free-slip

Free-slip boundary conditions model fluid flow in which the velocity components at a boundary are not constrained in
the tangential direction, while the normal velocity is fixed to zero.

Concretely, at free-slip boundaries, we want to enforce the following condition:
\f[
\begin{aligned}
u_n &= 0, &\qquad \text{no normal flow} \\
(\sigma n)_t &= 0 &\qquad \text{no tangential stress}
\end{aligned}
\f]
where \f$u_n\f$ is the velocity component in the normal direction, and
\f[
\sigma = 2 \eta \dot \varepsilon(u) - p I
\f]
with
\f[
\dot \varepsilon (u) = \frac{1}{2}(\nabla u + (\nabla u)^T).
\f]

To implement the free-slip boundary condition, we only enforce the first condition on the velocity component in the
normal direction, since the second one is naturally satisfied by the weak formulation.

Let's look at the simple case of the isoviscous Stokes system:
\f[
\begin{aligned}

- \Delta u + \nabla p &= 0, \\
  \nabla \cdot u &= 0
  \end{aligned}
  \f]

Multiplying the momentum equation by a test function \f$ v \f$ and integrating by parts yields
\f[
\int_{\Omega} \nabla u : \nabla v - \int_{\Omega} p \nabla \cdot v - \int_{\partial \Omega} (\partial_n u - pn) \cdot v = 0.
\f]
Since we impose \f$u \cdot n = 0\f$ we also have vanishing normal components of the test functions, i.e.,
\f$v \cdot n = 0\f$,
and therefore on the free-slip boundary
\f[
(\partial_n u - pn) \cdot v_t = (\partial_n u)_t \cdot v_t
\f]
because \f$n \cdot v_t = 0\f$.
Since we prescribe no tangential traction, the weak formulation drops that boundary term and thus enforces that the
traction is really 0.

### Implementation

\note See \ref terra::fe::strong_algebraic_freeslip_enforcement_in_place for a helper function that sets up the 
right-hand side.

We implement the free-slip condition by only strongly imposing zero normal flow at the boundary.
To do this, the implementation follows the idea of Engelman (1982): "The implementation of normal and/or tangential
boundary conditions in finite element codes for incompressible fluid flow".
The imposition is accomplished by pre- and post-multiplication of the entire linear system with rotation matrices
that transform between the canonical and a local normal-tangential coordinate system.
In that latter coordinate system, we can modify the normal velocity component directly.
Let \f$R\f$ be the rotation matrix that transforms from the canonical to the local normal-tangential coordinate system,
and \f$R^{-1} = R^T\f$ the inverse. Then, the modified Stokes system reads
\f[
\begin{pmatrix}
RAR^T & RB^T \\
BR^T & 0
\end{pmatrix}
\begin{pmatrix}
Ru \\
p
\end{pmatrix} =
\begin{pmatrix}
Rf_u \\
f_p
\end{pmatrix}
\f]
Note that this system is equivalent to the original Stokes system. However, we can now easily access the normal velocity
components of the vectors and therefore eliminate them strongly, just as in the sections above.
Particularly, we want to set them to zero.

For efficiency, the rotations are only executed at the boundary nodes. So, technically \f$R\f$ is a block-diagonal
matrix
with blocks of size \f$3\times 3\f$ that are either the local rotation matrices or identity matrices.

To enhance usability, the rotation of \f$ u \f$ is executed on-the-fly in a matrix-free context, such that before and
after the solution step, the solution vector \f$ u \f$ is stored in canonical coordinates.


