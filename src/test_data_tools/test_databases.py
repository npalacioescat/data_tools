# -*- coding: utf-8 -*-

__all__ = ['UpQueryTestCase']

import unittest

import pandas as pd

from data_tools import databases

class UpQueryTestCase(unittest.TestCase):
    def setUp(self):
        self.result = databases.up_map(['P00533', 'P31749', 'P16220'])
        self.expected = pd.DataFrame([['P00533', 'EGFR'],
                                      ['P31749', 'AKT1'],
                                      ['P16220', 'CREB1']],
                                     columns=['ACC', 'GENENAME'])

    def test_instance(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    def test_content(self):
        pd.testing.assert_frame_equal(self.result, self.expected)
