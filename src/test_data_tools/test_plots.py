# -*- coding: utf-8 -*-

__all__ = ['DensityTestCase', 'PianoConsensusTestCase', 'VennTestCase',
           'VolcanoTestCase']

import unittest

from data_tools import plots


class DensityTestCase(unittest.TestCase):
    @unittest.skip('** NOTE **: data_tools.plots.density test unit is not'
                   + ' implemented.')
    def test_null(self):
        pass


class PianoConsensusTestCase(unittest.TestCase):
    @unittest.skip('** NOTE **: data_tools.plots.piano_consensus test unit is'
                   + ' not implemented.')
    def test_null(self):
        pass


class VennTestCase(unittest.TestCase):
    @unittest.skip('** NOTE **: data_tools.plots.venn test unit is not'
                   + ' implemented.')
    def test_null(self):
        pass


class VolcanoTestCase(unittest.TestCase):
    @unittest.skip('** NOTE **: data_tools.plots.volcano test unit is not'
                   + ' implemented.')
    def test_null(self):
        pass
