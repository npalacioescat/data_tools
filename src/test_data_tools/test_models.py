# -*- coding: utf-8 -*-

__all__ = ['DoseResponseTestCase', 'LassoTestCase', 'LinearTestCase',
           'PowerLawTestCase']

import unittest

import numpy as np
from scipy.linalg import norm

from data_tools import models


class DoseResponseTestCase(unittest.TestCase):
    def __hill(self, x, k, m, n):

        return m * x ** n / (k ** n + x ** n)

    def setUp(self):
        self.xdata = np.linspace(1, 1e3, 100)
        self.params = [100, 1, 1]
        # Generate response data with additive white noise
        self.ydata = (self.__hill(self.xdata, *self.params)
                      + np.random.normal(0, 0.1, len(self.xdata)))

        # Fit the model
        self.model = models.DoseResponse(self.xdata, self.ydata)

    def test_fit(self):
        l_2 = abs(norm(self.ydata, 2)
                  - norm(self.__hill(self.xdata, *self.model.params), 2))
        self.assertLess(l_2, 0.1)

    def test_ec(self):
        self.assertEqual(round(1 - self.model.ec() / self.params[0], 0), 0)


class LassoTestCase(unittest.TestCase):
    @unittest.skip('** NOTE **: data_tools.models.Lasso test unit is not yet'
                   + ' implemented.')
    def test_null(self):
        pass


class LinearTestCase(unittest.TestCase):
    def setUp(self):
        # Fit y = x
        self.x = [0, 1, 2, 3, 4, 5]
        self.y = [0, 1, 2, 3, 4, 5]

        self.model = models.Linear(self.x, self.y)

    def test_intercept(self):
        self.assertEqual(self.model.intercept, 0.0)

    def test_slope(self):
        self.assertEqual(self.model.slope, 1.0)


class PowerLawTestCase(unittest.TestCase):
    def setUp(self):
        # Fit y = x^-2
        self.x = [1, 2, 4]
        self.y = [1, 0.25, 0.0625]

        self.model = models.PowerLaw(self.x, self.y)

    def test_exponent(self):
        self.assertEqual(self.model.k, -2.0)

    def test_const(self):
        self.assertEqual(self.model.a, 1.0)
