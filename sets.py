# -*- coding: utf-8 -*-

'''
Data tools - Set operations module

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

__all__ = ['in_all', 'bit_or', 'multi_union', 'find_min']

def in_all(x, N):
    '''
    Checks if a vector x is present in all sets contained in a list N.

    * Arguments:
        - x [tuple]: Or any hashable type as long as is the same
          contained in the sets of N.
        - N [list]: Or any iterable type containing [set] objects.

    * Returns:
        - [bool]: True if x is found in all sets of N, False otherwise.

    * Examples:
        >>> N = [{(0, 0), (0, 1)}, # <- set A
        ...      {(0, 0), (1, 1), (1, 0)}] # <- set B
        >>> x = (0, 0)
        >>> in_all(x, N)
        True
        >>> y = (0, 1)
        >>> in_all(y, N)
        False
    '''

    for s in N:
        if x in s:
            pass

        else:
            return False

    return True

def bit_or(a, b):
    '''
    Returns the bit operation OR between two bit-strings a and b.
    NOTE: a and b must have the same size.

    * Arguments:
        - a [tuple]: Or any iterable type.
        - b [tuple]: Or any iterable type.

    * Returns:
        - [tuple]: OR operation between a and b element-wise.

    * Examples:
        >>> a, b = (0, 0, 1), (1, 0, 1)
        >>> bit_or(a, b)
        (1, 0, 1)
    '''

    if a == b:
        return a

    else:
        return tuple([el_a | el_b for (el_a, el_b) in zip(a, b)])

def multi_union(N):
    '''
    Returns the union set of all sets contained in a list N.

    * Arguments:
        - N [list]: Or any iterable type containing [set] objects.

    * Returns:
        - [set]: The union of all sets contained in N.

    * Examples:
        >>> A = {1, 3, 5}
        >>> B = {0, 1, 2}
        >>> C = {0, 2, 5}
        >>> multi_union([A, B, C])
        set([0, 1, 2, 3, 5])
    '''

    return reduce(set.union, N)

def find_min(A):
    '''
    Finds and returns the subset of vectors whose sum is minimum from a
    given set A.

    * Arguments:
        - A [set]: Set of vectors ([tuple] or any iterable).

    * Returns:
        - [set]: Subset of vectors in A whose sum is minimum.

    * Examples:
        >>> A = {(0, 1, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)}
        >>> find_min(A)
        set([(0, 1, 0), (1, 0, 0)])
    '''

    A = list(A)
    sums = map(sum, A)
    idx_mins = np.where(sums == min(sums))[0]

    return {A[i] for i in idx_mins}
