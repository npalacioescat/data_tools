# -*- coding: utf-8 -*-

'''
data_tools.diffusion
====================

Diffusion solvers module.

The following functions provide tools to compute the diffusion on one or
two dimensions with different explicit or implicit methods.

**NOTE:** Explicit methods (``'euler_explicit1D'`` and
``'euler_explicit2D'``) are conditionally stable. This means that in
order to keep numerical stability of the solution (and obtain an
accurate result), these methods need to fulfill the
Courant–Friedrichs–Lewy (CFL) condition. For the one-dimensional case:

.. math::
  D\\frac{\\Delta t}{\\Delta x^2}\\leq\\frac{1}{2}

For the two-dimensional case (and assuming :math:`\\Delta x=\\Delta y`):

.. math::
  D\\frac{\\Delta t}{\\Delta x^2}\\leq\\frac{1}{4}

The implicit methods are (theoretically) unconditionally stable, hence
are more permissive in terms of discretization step-size.

Currently for the implicit methods only the coefficient matrix
contruction is available. To solve the diffusion problem user can use
any of the available linear algebra solvers by providing the current
diffusing field state and the matrix on each time-step.

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
'''

__all__ = ['euler_explicit1D', 'euler_explicit2D', 'euler_implicit_coef_mat',
           'crank_nicolson_coef_mats', 'build_coef_mat']

import numpy as np
from scipy.linalg import toeplitz

# TODO: Join methods and simplify your life with dimension-parametrized
#       slicing e.g.: x[x.ndim * (slice(1, -1),)]

# TODO: Euler explicit coefficient matrix wrapper?

# TODO: Add 3D versions


def euler_explicit1D(x0, dt, dx2, d=1, bcs='periodic'):
    '''
    Computes diffusion on a 1D space over a time-step using Euler
    explicit method.

    * Arguments:
        - *x0* [numpy.ndarray]: Initial state of a 1D array from which
          the difusion is to be computed.
        - *dt* [float]: Discretization time-step.
        - *dx2* [float]: Discretization spatial-step (squared).
        - *d* [float]: Optional, ``1`` by default. The diffusion
          coefficient.
        - *bcs* [str]: Optional, ``'periodic'`` by default. Determines
          the boundary conditions. Available options are ``'periodic'``,
          ``'dirichlet''`` or ``'neumann'``. Note that Dirichlet BCs do
          not hold mass conservation.

    * Returns:
        - [numpy.ndarray]: Computed state array (1D) after one time-step
          according to the parameters and conditions selected.
    '''

    c = d * dt / dx2

    if bcs == 'periodic':
        xpad = np.pad(x0, 1, mode='wrap')
        x = (1 - 2 * c) * x0 + c * (xpad[:-2] + xpad[2:])

    elif bcs == 'dirichlet':
        x = np.copy(x0)
        x[1:-1] = (1 - 2 * c) * x0[1:-1] + c * (x0[:-2] + x0[2:])

    elif bcs == 'neumann':
        xpad = np.pad(x0, 1, mode='edge')
        x = (1 - 2 * c) * x0 + c * (xpad[:-2] + xpad[2:])

    return x


def euler_explicit2D(x0, dt, dx2, d=1, bcs='periodic'):
    '''
    Computes diffusion on a 2D space over a time-step using Euler
    explicit method.

    * Arguments:
        - *x0* [numpy.ndarray]: Initial state of a 2D array from which
          the difusion is to be computed.
        - *dt* [float]: Discretization time-step.
        - *dx2* [float]: Discretization spatial-step (squared). It is
          assumed that is the same in both dimensions (dx = dy).
        - *d* [float]: Optional, ``1`` by default. The diffusion
          coefficient.
        - *bcs* [str]: Optional, ``'periodic'`` by default. Determines
          the boundary conditions. Available options are ``'periodic'``,
          ``'dirichlet''`` or ``'neumann'``. Note that Dirichlet BCs do
          not hold mass conservation.

    * Returns:
        - [numpy.ndarray]: Computed state array (2D) after one time-step
          according to the parameters and conditions selected.
    '''

    c = d * dt / dx2

    if bcs == 'periodic':
        xpad = np.pad(x0, 1, mode='wrap')
        x = (1 - 4 * c) * x0 + c * (xpad[1:-1, :-2] + xpad[1:-1, 2:]
                                    + xpad[:-2, 1:-1] + xpad[2:, 1:-1])

    elif bcs == 'dirichlet':
        x = np.copy(x0)
        x[1:-1, 1:-1] = (1 - 4 * c) * x0[1:-1, 1:-1] + c * (x0[1:-1, :-2]
                                                            + x0[1:-1, 2:]
                                                            + x0[:-2, 1:-1]
                                                            + x0[2:, 1:-1])

    elif bcs == 'neumann':
        xpad = np.pad(x0, 1, mode='edge')
        x = (1 - 4 * c) * x0 + c * (xpad[1:-1, :-2] + xpad[1:-1, 2:]
                                    + xpad[:-2, 1:-1] + xpad[2:, 1:-1])

    return x


def euler_implicit_coef_mat(dx2, dt, nx, ny=None, d=1, bcs='periodic'):
    '''
    Computes the coefficient matrix to solve the diffusion problem with
    the Euler implicit method such that:

    .. math::
       \mathbf{A}\cdot u^{n+1}=u^{n}

    Where **A** is the coefficient matrix and *u* the diffusing field.
    Note that for a 2D space, it is considered that *u* has been
    vectorized beforehand (e.g.: `u.reshape(-1)` assuming `u` is a 2D
    [`numpy.ndarray`]).

    * Arguments:
        - *dx2* [float]: Discretization spatial-step (squared). It is
          assumed that is the same in both dimensions (dx = dy).
        - *dt* [float]: Discretization time-step.
        - *nx* [int]: The number of discrete steps in the first spatial
          dimension.
        - *ny* [int]: Optional, ``None`` by default. The number of
          discrete steps in the second spatial dimension (if any).
        - *d* [float]: Optional, ``1`` by default. The diffusion
          coefficient.
        - *bcs* [str]: Optional, ``'periodic'`` by default. Determines
          the boundary conditions. Available options are ``'periodic'``,
          ``'dirichlet''`` or ``'neumann'``. Note that Dirichlet BCs do
          not hold mass conservation.

    * Returns:
        - [numpy.ndarray]: The coefficient matrix. Shape is `[nx, nx]`
          for one-dimensional problem and `[nx*ny, nx*ny]` for the
          two-dimensional case.
    '''

    lbd = d * dt / dx2

    a = 1 + 4 * lbd if ny else 1 + 2 * lbd # Central element coef
    b = -lbd # Neighbor element coefficient

    return build_coef_mat(a, b, nx, ny=ny, bcs=bcs)


def crank_nicolson_coef_mats(dx2, dt, nx, ny=None, d=1, bcs='periodic'):
    '''
    Computes the coefficient matrices to solve the diffusion problem
    with the Crank-Nicolson method such that:

    .. math::
       \mathbf{B}\cdot u^{n+1}=\mathbf{D}\cdot u^{n}

    Where **B** and **D** are the coefficient matrices and *u* the
    diffusing field. Note that for a 2D space, it is considered that *u*
    has been vectorized beforehand (e.g.: `u.reshape(-1)` assuming `u`
    is a 2D [`numpy.ndarray`]).

    * Arguments:
        - *dx2* [float]: Discretization spatial-step (squared). It is
          assumed that is the same in both dimensions (dx = dy).
        - *dt* [float]: Discretization time-step.
        - *nx* [int]: The number of discrete steps in the first spatial
          dimension.
        - *ny* [int]: Optional, ``None`` by default. The number of
          discrete steps in the second spatial dimension (if any).
        - *d* [float]: Optional, ``1`` by default. The diffusion
          coefficient.
        - *bcs* [str]: Optional, ``'periodic'`` by default. Determines
          the boundary conditions. Available options are ``'periodic'``,
          ``'dirichlet''`` or ``'neumann'``. Note that Dirichlet BCs do
          not hold mass conservation.

    * Returns:
        - [numpy.ndarray]: The coefficient matrix **B**. Shape is
          `[nx, nx]` for one-dimensional problem and `[nx*ny, nx*ny]`
          for the two-dimensional case.
        - [numpy.ndarray]: The coefficient matrix **D**. Shape is the
          same as **B**.
    '''

    lbd = d * dt / dx2

    lhsa = 1 + 2 * lbd if ny else 1 + lbd # Central element coef
    lhsb = -lbd / 2. # Neighbor element coefficient

    rhsa = 1 - 2 * lbd if ny else 1 - lbd # Central element coef
    rhsb = lbd / 2. # Neighbor element coefficient

    return (build_coef_mat(lhsa, lhsb, nx, ny=ny, bcs=bcs),
            build_coef_mat(rhsa, rhsb, nx, ny=ny, bcs=bcs))


def build_coef_mat(a, b, nx, ny=None, bcs='periodic'):
    '''
    Builds a coefficient matrix according to the central and neighbor
    coefficients, system size and boundary conditions.

    * Arguments:
        - *a* [float]: The central element coefficient.
        - *b* [float]: The neighbor element coefficient.
        - *nx* [int]: The number of discrete steps in the first spatial
          dimension.
        - *ny* [int]: Optional, ``None`` by default. The number of
          discrete steps in the second spatial dimension (if any).
        - *bcs* [str]: Optional, ``'periodic'`` by default. Determines
          the boundary conditions. Available options are ``'periodic'``,
          ``'dirichlet''`` or ``'neumann'``. Note that Dirichlet BCs do
          not hold mass conservation.

    * Returns:
        - [numpy.ndarray]: The coefficient matrix. Shape is `[nx, nx]`
          for one-dimensional problem and `[nx*ny, nx*ny]` for the
          two-dimensional case.
    '''

    if ny: # 2D coefficient matrix
        vec = np.zeros(int(nx * ny))
        vec[0] = a
        vec[1], vec[nx] = b, b

        if bcs == 'periodic':
            vec[-nx] = b

        mat = toeplitz(vec)

        for n in xrange(nx, nx * ny, nx):
            mat[n, n - 1] = 0
            mat[n - 1, n] = 0

        if bcs == 'periodic':

            for n in xrange(0, nx * ny, nx):
                mat[n, n + (nx - 1)] = b
                mat[n + (nx - 1), n] = b

    else: # 1D coefficient matrix
        vec = np.zeros(int(nx))

        vec[:2] = a, b

        if bcs == 'periodic':
            vec[-1] = b

        mat = toeplitz(vec)

    if bcs == 'neumann':
        bounds = get_boundaries(np.ones((nx, ny) if ny else (nx)),
                                counts=True).flatten()
        mat += b * np.diag(bounds)

    return mat
