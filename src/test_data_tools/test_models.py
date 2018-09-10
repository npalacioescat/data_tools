# -*- coding: utf-8 -*-

__all__ = ['DoseResponseTestCase', 'LassoTestCase']

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
