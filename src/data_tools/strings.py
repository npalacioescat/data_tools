# -*- coding: utf-8 -*-

'''
data_tools.strings
==================

String operations module.
'''

__all__ = ['is_numeric', 'join_str_lists']

def is_numeric(s):
    '''
    Determines if a string can be considered a numeric value. NaN is
    also considered, since it is float type.

    * Arguments:
        - *s* [str]: String to be evaluated.

    * Returns:
        - [bool]: ``True``/``False`` depending if the condition is
          satisfied.

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
        - *a* [list]: Contains the first elements [str] of the joint
          strings.
        - *b* [list]: Contains the second elements [str] of the joint
          strings.
        - *sep* [str]: Optional ``''`` (non separated) by default.
          Determines the separator between the joint strings.

    * Returns:
        - [list]: List of the joint strings.

    * Example:
        >>> a = ['a', 'b']
        >>> b = ['1', '2']
        >>> join_str_lists(a, b, sep='_')
        ['a_1', 'b_2']
    '''

    return map(sep.join, zip(a, b))
