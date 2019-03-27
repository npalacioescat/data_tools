# -*- coding: utf-8 -*-

__all__ = ['IsNumericTestCase', 'JoinStrListsTestCase']

import unittest

from data_tools import strings


class IsNumericTestCase(unittest.TestCase):
    def test_int(self):
        self.assertTrue(strings.is_numeric('1'))
        self.assertTrue(strings.is_numeric('-1'))

    def test_float(self):
        self.assertTrue(strings.is_numeric('1.0'))
        self.assertTrue(strings.is_numeric('-1.0'))

    def test_nan(self):
        self.assertTrue(strings.is_numeric('NaN'))

    def test_nonum(self):
        self.assertFalse(strings.is_numeric('Hello world'))


class JoinStrListsTestCase(unittest.TestCase):
    def test_same_size(self):
        self.assertEqual(strings.join_str_lists(['a', 'b'], ['1', '2']),
                         ['a1', 'b2'])

    def test_diff_size(self):
        self.assertEqual(strings.join_str_lists(['a', 'b'], ['1', '2', '3']),
                         ['a1', 'b2'])

    def test_sep(self):
        self.assertEqual(strings.join_str_lists(['a', 'b'], ['1', '2'],
                         sep='_'), ['a_1', 'b_2'])

    def test_non_str(self):
        with self.assertRaises(TypeError):
            strings.join_str_lists(['a', 'b'], [1, 2])
