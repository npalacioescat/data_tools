# -*- coding: utf-8 -*-

__all__ = ['KeggLinkTestCase', 'KeggPathwayMappingTestCase',
           'OpKinaseSubstrateTestCase', 'UpMapTestCase']

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
        self.expected.sort_values(by='query', inplace=True)

    def test_instance(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    def test_content(self):
        pd.testing.assert_frame_equal(self.result, self.expected)


class KeggPathwayMappingTestCase(unittest.TestCase):
    @unittest.skip('** NOTE **: data_tools.databases.kegg_pathway_mapping'
                   + ' test unit is not implemented.')
    def test_null(self):
        pass


class OpKinaseSubstrateTestCase(unittest.TestCase):
    def setUp(self):
        self.result = databases.op_kinase_substrate(incl_phosphatases=True)

    def test_instance(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    def test_size(self):
        self.assertGreater(len(self.result), 0)

    def test_kinases(self):
        self.assertIn('phosphorylation',
                      self.result[self.result.columns[-1]].values)

    def test_phosphatases(self):
        self.assertIn('dephosphorylation',
                      self.result[self.result.columns[-1]].values)


class UpMapTestCase(unittest.TestCase):
    def setUp(self):
        self.result = databases.up_map(['P00533', 'P31749', 'P16220'])
        self.expected = pd.DataFrame([['P00533', 'EGFR'],
                                      ['P31749', 'AKT1'],
                                      ['P16220', 'CREB1']],
                                     columns=['ACC', 'GENENAME'])
        self.expected.sort_values(by='ACC', inplace=True)

    def test_instance(self):
        self.assertIsInstance(self.result, pd.DataFrame)

    def test_content(self):
        pd.testing.assert_frame_equal(self.result, self.expected)
