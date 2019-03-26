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

__all__ = []

import numpy as np
from scipy.sparse import block_diag

from data_tools.spatial import get_boundaries

def build_mat(lbd, dims, bcs='d'):
    '''
    '''

    dims = list(dims)

    for i, n in enumerate(dims):

        if i == 0:
            mat = (np.eye(n) * 2 * lbd
                   + np.eye(n, k=1) * lbd
                   + np.eye(n, k=-1) * lbd)

            if bcs == 'p':
                # Make matrix circular
                mat[[0, -1], [-1, 0]] = lbd

        else:
            prev = mat.shape[0]
            curr = prev * n

            mat = block_diag([mat] * n).toarray()
            mat += (curr, k=prev) * lbd + np.eye(curr, k=-prev) * lbd

            if bcs == 'p':
                mat += (np.eye(curr, k=curr - prev) * lbd
                        + np.eye(curr, k=prev - curr * lbd))

    if bcs == 'n':
        factor = get_boundaries(np.ndarray(dims), counts=True).flatten()
        mat += np.diag(factor) * lbd
