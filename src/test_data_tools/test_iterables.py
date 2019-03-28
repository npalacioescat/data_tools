# -*- coding: utf-8 -*-

__all__ = ['BitOrTestCase', 'ChunkThisTestCase', 'FindMinTestCase',
           'InAllTestCase', 'SimilarityTestCase', 'SubsetsTestCase',
           'UnzipDictsTestCase']

import unittest

import numpy as np

from data_tools import iterables


class BitOrTestCase(unittest.TestCase):
    def test_int_list(self):
        self.assertTupleEqual(iterables.bit_or([0, 0, 1],
                                               [1, 0, 1]),
                              (1, 0, 1))

    def test_bool_list(self):
        self.assertTupleEqual(iterables.bit_or([False, False, True],
                                               [True, False, True]),
                              (True, False, True))


class ChunkThisTestCase(unittest.TestCase):
    def setUp(self):
        self.aux = range(6)

    def test_even_slicing(self):
        self.assertEqual(iterables.chunk_this(self.aux, 2),
                         [[0, 1], [2, 3], [4, 5]])

    def test_uneven_slicing(self):
        self.assertEqual(iterables.chunk_this(self.aux, 4),
                         [[0, 1, 2, 3], [4, 5]])


class FindMinTestCase(unittest.TestCase):
    def test_int_tuple(self):
        self.assertSetEqual(iterables.find_min({(0, 1, 1),
                                                (0, 1, 0),
                                                (1, 0, 0),
                                                (1, 1, 1)}),
                            {(0, 1, 0),
                             (1, 0, 0)})

    def test_bool_tuple(self):
        self.assertSetEqual(iterables.find_min({(False, True, True),
                                                (False, True, False),
                                                (True, False, False),
                                                (True, True, True)}),
                            {(False, True, False),
                             (True, False, False)})

    def test_diff_size_tuple(self):
        self.assertSetEqual(iterables.find_min({(0, 1, 1, 1),
                                                (1, 0),
                                                (1, 0, 0),
                                                (1, 1, 1, 1)}),
                            {(1, 0, 0),
                             (1, 0)})


class InAllTestCase(unittest.TestCase):
    def test_tuple(self):
        N = [{(0, 0), (0, 1)},
             {(0, 0), (1, 0)}]
        self.assertTrue(iterables.in_all((0, 0), N))
        self.assertFalse(iterables.in_all((0, 1), N))

    def test_int(self):
        N = [{0, 1, 2},
             {0, 2, 3},
             {2, 4, 5, 8}]
        self.assertTrue(iterables.in_all(2, N))
        self.assertFalse(iterables.in_all(8, N))


class SimilarityTestCase(unittest.TestCase):
    def setUp(self):
        self.a = {0, 1, 2, 3}
        self.b = {0, 1}

    def test_jaccard(self):
        self.assertEqual(iterables.similarity(self.a, self.b), 0.5)

    def test_sorensen_dice(self):
        self.assertEqual(iterables.similarity(self.a, self.b, mode='sd'), 2./3)

    def test_szymkiewicz_simpson(self):
        self.assertEqual(iterables.similarity(self.a, self.b, mode='ss'), 1.)

    def test_empty_set(self):
        self.assertTrue(np.isnan(iterables.similarity(self.a, set())))


class SubsetsTestCase(unittest.TestCase):
    def test_two_sets_int(self):
        self.assertDictEqual(iterables.subsets([{0, 1, 2},
                                                {2, 3, 4}]),
                             {'11': set([2]),
                              '10': set([0, 1]),
                              '01': set([3, 4])})

    def test_three_sets_int(self):
        self.assertDictEqual(iterables.subsets([{0, 1},
                                               {2, 3},
                                               {1, 3, 4}]),
                             {'010': set([2]),
                              '011': set([3]),
                              '001': set([4]),
                              '111': set([]),
                              '110': set([]),
                              '100': set([0]),
                              '101': set([1])})


class UnzipDictsTestCase(unittest.TestCase):
    def setUp(self):
        self.a = dict([('x_a', 2), ('y_a', 3)])
        self.b = dict([('x_b', 1), ('y_b', -1)])

    def test_unzip(self):
        self.assertEqual(iterables.unzip_dicts(self.a, self.b),
                         [('y_a', 'x_a', 'x_b', 'y_b'), (3, 2, 1, -1)])
