# -*- coding: utf-8 -*-

__all__ = ['DoseResponseTestCase', 'LassoTestCase']

import unittest

import numpy as np

from data_tools import models


class DoseResponseTestCase(unittest.TestCase):
    def __hill(self, x, k, m, n):

        return m * x ** n / (k ** n + x ** n)

    def setUp(self):
        self.xdata = np.linspace(1, 1e3, 100)
        self.params = [100, 1, 1]
        self.ydata = self.__hill(self.xdata, *self.params)

    def test_fit(self):
        self.model = models.DoseResponse()
