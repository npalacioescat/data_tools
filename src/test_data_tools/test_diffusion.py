# -*- coding: utf-8 -*-

__all__ = ['EulerExplicit1DTestCase']

import unittest

import numpy as np
from scipy.linalg import norm

from data_tools import diffusion

class EulerExplicit1DTestCase(unittest.TestCase):
    def setUp(self):
        # Spatial limits
        a = -np.pi
        b = np.pi
        # Discrete space points
        Nx = 200
        # Space step-size
        dx = (b - a) / (Nx - 1)
        self.dx2 = dx ** 2
        # Time step-size
        self.dt = self.dx2 / 4
        # Space vector [a, b]
        X = np.arange(a, b + dx, dx)
        # Time vector [0, T]
        T = np.arange(0, 1 + self.dt, self.dt)
        # Time points
        self.Nt = len(T)
        # U(t) vector w/ initial condition
        self.Ui = np.cos(X) ** 3
        # U(t+1) vector initialization
        self.U = np.copy(self.Ui)

        self.analytic = (np.exp(-1) * (3.0 / 4.0) * np.cos(X) + (1.0 / 4.0)
                         * np.exp(-9) * np.cos(3 * X))

    def test_solver(self):
        for n in range(self.Nt):
            self.U = diffusion.euler_explicit1D(self.Ui, self.dt, self.dx2)
            self.Ui = self.U

        # Check error with L^inf norm
        l_inf = norm(self.U, np.inf) - norm(self.analytic, np.inf)
        self.assertLess(l_inf, 0.01)
