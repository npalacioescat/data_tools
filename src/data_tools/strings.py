# -*- coding: utf-8 -*-

'''
Data tools - String operations module

Copyright (C) 2018 Nicolàs Palacio

Contact: nicolaspalacio91@gmail.com

GNU-GLPv3:
This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
Public License for more details.

A full copy of the GNU General Public License can be found on file
"LICENSE.md". If not, see <http://www.gnu.org/licenses/>.
'''

__all__ = ['is_numeric', 'join_str_lists']

def is_numeric(s):
    '''
    Determines if a string can be considered a numeric value.
    NaN is also considered, since it is float type.

    * Arguments:
        - s [str]: String to be evaluated.

    * Returns:
        - [bool]: True/False depending if the condition is satisfied.

    * Examples:
        >>> is_numeric('4')
        True
        >>> is_numeric('-3.2')
        True
        >>> is_numeric('number')
        False
        >>> is_numeric('NaN')
        True
    '''

    try:
        float(s)
        return True

    except ValueError:
        return False

def join_str_lists(a, b, sep=''):
    '''
    Joins element-wise two lists (or any 1D iterable) of strings with a
    given separator (if provided). Length of the input lists must be
    equal.

    * Arguments:
        - a [list]: Contains the first elements [str] of the joint
          strings.
        - b [list]: Contains the second elements [str] of the joint
          strings.
        - sep [str]: Optional '' (non separated) by default. Determines
          the separator between the joint strings.

    * Returns:
        - [list]: List of the joint strings.

    * Example:
        >>> a = ['a', 'b']
        >>> b = ['1', '2']
        >>> join_str_lists(a, b, sep='_')
        ['a_1', 'b_2']
    '''

    return map(sep.join, zip(a, b))
