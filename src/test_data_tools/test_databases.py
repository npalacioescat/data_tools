# -*- coding: utf-8 -*-

__all__ = ['KeggLinkTestCase', 'UpQueryTestCase']

import unittest

import pandas as pd

from data_tools import databases

class KeggLinkTestCase(unittest.TestCase):
    def setUp(self):
        self.result = databases.kegg_link(['hsa:10458', 'ece:Z5100'])
        self.expected = pd.DataFrame([['hsa:10458', 'path:hsa04520'],
                                      ['hsa:10458', 'path:hsa04810'],
                                      ['ece:Z5100', 'path:ece05130']],
                                     columns=['query', 'pathway'])

    def test_instance(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    def test_content(self):
        pd.testing.assert_frame_equal(self.result, self.expected)

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
