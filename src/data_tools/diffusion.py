# -*- coding: utf-8 -*-

'''
data_tools.diffusion
====================

Diffusion solvers module.

Introduction
------------

The following functions provide tools to solve the diffusion problem for
any number of spatial dimensions with different explicit and implicit
methods. The problem is defined as follows:

.. math::
  \\frac{\\partial u}{\\partial t} = D\\nabla^2u

Where :math:`u` is the diffusing component, :math:`D` is the diffusion
coefficient and :math:`\\nabla^2` is the Laplace operator
(in Euclidean space :math:`\\nabla^2f=\\nabla\\cdot\\nabla f=\\sum_{i=1\
}^n\\frac{\\partial^2 f}{\\partial x_i^2}`).

Some numerical methods to solve this probelm will be explained in
the following subsections. These are based in approximation of the
derivatives by finite-differences. Therefore we define the
discretization step sizes as :math:`\\Delta t` for the time derivative
and :math:`\\Delta x`, :math:`\\Delta y` and so on for first, second and
subsequent spatial dimensions respectively.

From here on we will assume that space is homogeneously discretized
(e.g.: :math:`\\Delta x=\\Delta y`). All second-order partial
derivatives in the spatial dimension are discretized using central
finite differences. For instance, in one dimension:

.. math::
  \\frac{\\partial^2u}{\\partial x^2}\\approx\\frac{u_{i-1}-2u_i+u_{i+1\
  }}{\\Delta x^2}

Euler explicit method
^^^^^^^^^^^^^^^^^^^^^

"Classic" method, first-order accurate, uses forward difference over
time:

.. math::
  \\frac{\\partial u}{\\partial t}\\approx\\frac{u^{k+1}-u^k}{\\Delta t}

Where :math:`k` is the current time-step. Applied to the diffusion
problem for one dimension and rearranging terms:

.. math::
  u_i^{k+1}=\\frac{D\\Delta t}{\\Delta x^2}\\left(u_{i+1}^k+u_{i-1}^k\
  \\right)+\\left(1-2\\frac{D\\Delta t}{\\Delta x^2}\\right)u_i^k

Let us define from here on :math:`\\lambda\\equiv\\frac{D\\Delta t}{\\D\
elta x^2}` for simplicity. Rewriting the equation above in terms of
linear algebra, each time-step the next state of :math:`u` is:

.. math::
  u^{k+1}=\\mathbf{A}u^{k}

Where :math:`\\mathbf{A}` is the tri-diagonal coefficient matrix whose
central element is :math:`(1-2\\lambda)` and its adjacent diagonals are
:math:`\\lambda`. Note that for :math:`n`-dimensional case, central
element will then be :math:`(1-2n\\lambda)` and :math:`u` must be
flattened (coerced into one dimension) and :math:`\\mathbf{A}` becomes
a block tri-diagonal matrix.

**NOTE:** Explicit methods are conditionally stable. This means that in
order to keep numerical stability of the solution (and obtain an
accurate result), these methods need to fulfill the
Courant–Friedrichs–Lewy (CFL) condition. For any :math:`n`-dimensional
case:

.. math::
  D\\frac{\\Delta t}{\\Delta x^2}\\leq\\frac{1}{2n}

The implicit methods are (theoretically) unconditionally stable, hence
are more permissive in terms of discretization step-size.

Euler implicit method
^^^^^^^^^^^^^^^^^^^^^

Similar to Euler explicit (first-order accurate) but uses backward
difference over time (theoretically, unconditionally stable):

.. math::
  \\frac{\\partial u}{\\partial t}\\approx\\frac{u^k-u^{k-1}}{\\Delta t}

Applied to the diffusion problem in one dimension and taking one step
forward over discrete time (:math:`k\\rightarrow k+1`):

.. math::
  u_i^k=-\\lambda\\left(u_{i+1}^{k+1}+u_{i-1}^{k+1}\\right)+\\left(1+2\
  \\lambda\\right)u_i^{k+1}

Posed as a linear algebra problem:

.. math::
  u^{k}=\\mathbf{A}u^{k+1}

Where :math:`\\mathbf{A}` is the tri-diagonal coefficient matrix whose
central element is :math:`(1+2\\lambda)` and its adjacent diagonals are
:math:`-\\lambda`. For :math:`n`-dimensional case, central element will
then be :math:`(1+2n\\lambda)` and :math:`u` must be flattened (coerced
into one dimension) and :math:`\\mathbf{A}` becomes a block tri-diagonal
matrix.

Crank-Nicolson method
^^^^^^^^^^^^^^^^^^^^^

Implicit method, second-order accurate that uses trapezoidal rule for
integration time between forward and backward differences. Therefore,
assuming :math:`u_t=f(u,t)` then:

.. math::
  \\frac{u^{k+1}-u^k}{\\Delta t}=\\frac{1}{2}\\left(f(u^k,t^k)+f(u^{k+1\
  },t^{k+1})\\right)

Applied to the diffusion problem in one dimension:

.. math::
  \\frac{\\lambda}{2}\\left(u_{i+1}^k+u_{i-1}^k\\right)+\\left(1-\
  \\lambda\\right)u_i^k=-\\frac{\\lambda}{2}\\left(u_{i+1}^{k+1}+\
  u_{i-1}^{k+1}\\right)+\\left(1+\\lambda\\right)u_i^{k+1}

Posed as a linear algebra problem:

.. math::
  \\mathbf{A}u^{k+1}=\\mathbf{B}u^k

Where :math:`\\mathbf{A}` is the tri-diagonal coefficient matrix for
:math:`k+1` whose central element is :math:`(1+\\lambda)` and its
adjacent diagonals are :math:`-\\frac{\\lambda}{2}`. Similarly,
:math:`\\mathbf{B}` is the tri-diagonal matrix for :math:`k` whose
central element is :math:`(1-\\lambda)` and its adjacent diagonals are
:math:`\\frac{\\lambda}{2}`. For :math:`n`-dimensional case, central
elements will then be :math:`(1+n\\lambda)` and :math:`(1-n\\lambda)`
for :math:`\\mathbf{A}` and :math:`\\mathbf{B}` respectively and
:math:`u` must be flattened (coerced into one dimension) as well as that
coefficient matrices become block tri-diagonal matrices.

------------------------------------------------------------------------

Independently of the numerical method, it is assumed that the problem is
posed in terms of linear algebra. This is, the current and next state
of the diffusing field :math:`u` can be expressed with matrix
multiplication(s) as shown above.

Currently only the coefficient matrix construction is available. To
solve the diffusion problem user can use any of the available linear
algebra solvers by providing the current diffusing field state
(flattened) and the coefficient matrix on each time-step (or simple
matrix multiplication for time explicit methods).

* Simplest options are either ``numpy.linalg.solve()`` or
  ``scipy.linalg.solve()`` (both not very fast).
* If the coefficient matrix is positive-definite (it is most of the
  times, but can be double-checked, specially if errors arise) and
  symmetric, a good option is Choleski's factorization. This is already
  implemented in ``scipy.linalg.cholesky()`` which factorizes the
  coefficient matrix and that can be passed to the
  ``scipy.linalg.cho_solve()`` which is way faster than the option above.
* Another option (but don't tell anyone) is to invert the coefficient
  matrix and just solve the equation with a matrix multiplication. This
  is way faster but your coefficient matrix has to be invertible. If
  the determinant is close to zero, may cause numerical instability.

Contents
--------
'''

__all__ = ['build_mat']

import numpy as np
from scipy.sparse import block_diag

from data_tools.spatial import get_boundaries

# XXX: implement wrappers for the different numerical methods?
# XXX: implement solver wrapper?


def build_mat(cent, neigh, dims, bcs='dirichlet'):
    '''
    Builds a (block) tri-diagonal coefficient matrix to solve a
    n-dimensional diffusion problem as a linear algebraic system.

    * Arguments:
        - *cent* [float]: The coefficient corresponding to the central
          element of the stencil.
        - *neigh* [float]: The coefficient corresponding to the direct
          neighbors of the central element in the stencil.
        - *dims* [list]: Or [tuple], contains the size of finite
          elements [int] for each dimension. Note that the order is
          first dimension first (e.g.: ``[x, y, z]``) as opposed to
          numpy's indexing order (last dimension first, e.g.:
          ``[z, y, x]``).
        - *bcs* [str]: Optional, ``'dirichlet'`` by default. Determines
          the boundary conditions. Available options are ``'periodic'``,
          ``'dirichlet'`` or ``'neumann'``. Note that Dirichlet BCs do
          not hold mass conservation.

    * Returns:
        - [numpy.ndarray]: The (block) tri-diagonal coefficient matrix.
          Matrix will be square with size equal to the product of all
          dimension sizes.
    '''

    try:
        assert type(dims) in [list, tuple]

    except AssertionError:
        dims = [dims]

    for i, n in enumerate(dims):

        try:
            n = int(n)

        except (ValueError, TypeError) as e:
            raise(e.__class__('Invalid dimension %d with value %s of type %s'
                  % (i, n, type(n))))

        if i == 0:
            mat = (np.eye(n) * cent
                   + np.eye(n, k=1) * neigh
                   + np.eye(n, k=-1) * neigh)

            if bcs == 'periodic':
                # Make matrix circular
                mat[[0, -1], [-1, 0]] = neigh

        else:
            prev = mat.shape[0]
            curr = prev * n

            mat = block_diag([mat] * n).toarray()
            mat += np.eye(curr, k=prev) * neigh + np.eye(curr, k=-prev) * neigh

            if bcs == 'periodic':
                mat += (np.eye(curr, k=curr - prev) * neigh
                        + np.eye(curr, k=prev - curr * neigh))

    if bcs == 'neumann':
        factor = get_boundaries(np.ndarray(dims), counts=True).flatten()
        mat += np.diag(factor) * neigh

    return mat
