# -*- coding: utf-8 -*-

__all__ = ['BitOrTestCase', 'FindMinTestCase', 'InAllTestCase',
           'SubsetsTestCase']

import unittest

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
