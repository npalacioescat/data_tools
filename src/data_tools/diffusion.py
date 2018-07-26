# -*- coding: utf-8 -*-

'''
data_tools.diffusion
====================

Diffusion solvers module.
'''

__all__ = ['euler_explicit1D']

import numpy as np


def euler_explicit1D(x0, dt, dx2, d=1, bcs='periodic'):
    '''
    Computes diffusion on a 1D space over a time-step using Euler
    explicit method.

    * Arguments:
        - *x0* [numpy.ndarray]: Initial state of a 1D array from which
          the difusion is to be computed.
        - *dt* [float]: Discretization time-step.
        - *dx2* [float]: Discretization spatial-step (squared).
        - *d* [float]: Diffusion coefficient.
        - *bcs* [str]: Optional, ``'periodic'`` by default. Determines
          the boundary conditions. Available options are ``'periodic'``,
          ``'dirichlet''`` or ``'neumann'``.

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
