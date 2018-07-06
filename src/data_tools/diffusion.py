# -*- coding: utf-8 -*-

'''
data_tools.diffusion
====================

Diffusion solvers module.

Reference
---------
'''

import numpy as np


def euler_explicit1D(x0, dt, dx2, d=1, bcs='periodic'):
    '''
    Computes diffusion on a 1D space over a time-step using Euler
    explicit method

    * Arguments:
        - *x0* []: .
        - *dt* []: .
        - *dx2* []: .
        - *d* []: .
        - *bcs* []: .

    * Returns:
        - []: .
    '''

    c = d * dt / dx2

    if bcs == 'periodic':
        xpad = np.pad(x0, 1, mode='wrap')
        #x = x0 + d * dt * ((xpad[:-2] - 2 * xpad[1:-1] + xpad[2:]) / dx2)
        x = (1 - 2 * c) * x0 + c * (xpad[:-2] + xpad[2:])

    elif bcs == 'dirichlet':
        x = np.copy(x0)
        #x[1:-1] = x0[1:-1] + d * dt * ((x0[:-2] - 2 * x0[1:-1] + x0[2:]) / dx2)
        x[1:-1] = (1 - 2 * c) * x0[1:-1] + c * (x0[:-2] + x0[2:])
        #x[0] = x0[0]
        #x[-1] = x0[-1]

    elif bcs == 'neumann':
        xpad = np.pad(x0, 1, mode='edge')
        #x = x0 + d * dt * ((xpad[:-2] - 2 * xpad[1:-1] + xpad[2:]) / dx2)
        x = (1 - 2 * c) * x0 + c * (xpad[:-2] + xpad[2:])

    return x
