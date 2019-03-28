# -*- coding: utf-8 -*-

'''
data_tools.iterables
====================

Iterable-type operations module.

Contents
--------
'''

from __future__ import division, print_function

__all__ = ['bit_or', 'chunk_this', 'find_min', 'in_all', 'similarity',
           'subsets', 'unzip_dicts']

import sys
import itertools
from functools import reduce

import numpy as np

if sys.version_info < (3,):
    range = xrange


def bit_or(a, b):
    '''
    Returns the bit operation OR between two bit-strings *a* and *b*.
    **NOTE:** *a* and *b* must have the same size.

    * Arguments:
        - *a* [tuple]: Or any iterable type.
        - *b* [tuple]: Or any iterable type.

    * Returns:
        - [tuple]: OR operation between *a* and *b* element-wise.

    * Examples:
        >>> a, b = (0, 0, 1), (1, 0, 1)
        >>> bit_or(a, b)
        (1, 0, 1)
    '''

    if a == b:
        return a

    else:
        return tuple([el_a | el_b for (el_a, el_b) in zip(a, b)])


def chunk_this(L, n):
    '''
    For a given list *L*, returns another list of *n*-sized chunks from
    it (in the same order).

    * Arguments:
        - *L* [list]: The list to be sliced into sublists of the
          definded size.
        - *n* [int]: The size of the chunks.

    * Returns:
        - [list]: List of *n*-sized chunks from *L*. **NOTE:** If the
          number of items in *L* is not divisible by *n*, the last
          element returned will have an inferior size.

    * Examples:
        >>> L = range(6)
        >>> chunk_this(L, 2)
        [[0, 1], [2, 3], [4, 5]]
        >>> chunk_this(L, 4)
        [[0, 1, 2, 3], [4, 5]]
    '''

    return [L[i:i + n] for i in range(0, len(L), n)]


def find_min(A):
    '''
    Finds and returns the subset of vectors whose sum is minimum from a
    given set *A*.

    * Arguments:
        - *A* [set]: Set of vectors ([tuple] or any iterable).

    * Returns:
        - [set]: Subset of vectors in *A* whose sum is minimum.

    * Examples:
        >>> A = {(0, 1, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)}
        >>> find_min(A)
        set([(0, 1, 0), (1, 0, 0)])
    '''

    A = list(A)
    sums = np.array(list(map(sum, A)))
    idx_mins = np.where(sums == min(sums))[0]

    return {A[i] for i in idx_mins}


def in_all(x, N):
    '''
    Checks if a element *x* is present in all collections contained in a
    list *N*.

    * Arguments:
        - *x* [object]: Any type of object, it is assumed to be the same
          type as the objects contained in the elements of *N*.
        - *N* [list]: Or any iterable type containing a collection of
          other iterables containing the objects.

    * Returns:
        - [bool]: ``True`` if *x* is found in all elements of *N*,
          ``False`` otherwise.

    * Examples:
        >>> N = [{(0, 0), (0, 1)}, # <- set A
        ...      {(0, 0), (1, 1), (1, 0)}] # <- set B
        >>> x = (0, 0)
        >>> in_all(x, N)
        True
        >>> y = (0, 1)
        >>> in_all(y, N)
        False
        >>> N = [['Hello', 'world', '!'],
        ...      ['Hello', 'user']]
        >>> x = 'Hello'
        >>> in_all(x, N)
        True
    '''

    for s in N:

        if x in s:
            pass

        else:
            return False

    return True


def similarity(a, b, mode='j'):
    '''
    Computes the similarity index between two sets. There are three
    options available:

    Jaccard (``mode='j'``):

    .. math::
        s_J(A,B) = \\frac{|A\\cap B|}{|A\\cup B|}

    Sorensen-Dice (``mode='sd'``):

    .. math::
        s_{SD}(A,B) = \\frac{2|A\\cap B|}{|A|+|B|}

    Szymkiewicz–Simpson (``mode='ss'``):

    .. math::
        s_{SS}(A,B) = \\frac{|A\\cap B|}{\\min(|A|,|B|)}

    * Arguments:
        - *a* [set]: One of the two sets to compute the similarity
          index.
        - *b* [set]: The other set to compute the similarity index.
        - *mode* [str]: Optional, ``'j'`` (Jaccard) by default.
          Indicates which type of similarity index/coefficient is to be
          computed. Available options are: ``'j'`` for Jaccard, ``'sd'``
          for Sorensen-Dice and ``'ss'`` for Szymkiewicz–Simpson.

    * Returns:
        - [float]: The corresponding similarity index/coefficient
          according to the specified *mode*.
    '''

    sa, sb = map(set, (a, b))

    if len(sa) == 0 or len(sb) == 0:
        print("WARNING: at least one of the sets' size is 0")
        return np.nan

    inter = len(sa.intersection(sb))

    if mode == 'j':
        num = inter
        den = len(sa.union(sb))

    elif mode == 'sd':
        num = 2 * inter
        den = sum(map(len, (sa, sb)))

    elif mode == 'ss':
        num = inter
        den = min(map(len, (sa, sb)))

    num, den = map(float, (num, den))

    return num / den



def subsets(N):
    '''
    Function that computes all possible logical relations between all
    sets on a list *N* and returns all subsets. This is, the subsets
    that would represent each intersecting area on a Venn diagram.

    * Arguments:
        - *N* [list]: Or any iterable type containing [set] objects.

    * Returns:
        - [dict]: Collection of subsets according to the logical
          relations between the sets in *N*. The keys are binary codes
          that denote the logical relation (see examples below).

    * Examples:
        >>> N = [{0, 1, 2}, {2, 3, 4}]
        >>> subsets(N)
        {'11': set([2]), '10': set([0, 1]), '01': set([3, 4])}
        >>> N = [{0, 1}, {2, 3}, {1, 3, 4}]
        >>> subsets(N)
        {'010': set([2]), '011': set([3]), '001': set([4]), '111': set([
        ]), '110': set([]), '100': set([0]), '101': set([1])}
    '''

    combinations = list(itertools.product(['0', '1'], repeat=len(N)))[1:]

    result = dict()

    for c in combinations:
        intersect = [N[n] for n in [i for i, v in enumerate(c) if v == '1']]
        unite = [N[n] for n in [i for i, v in enumerate(c) if v == '0']]

        lhs = reduce(set.intersection, intersect)
        rhs = reduce(set.union, unite) if unite else set()

        result[''.join(c)] = lhs.difference(rhs)

    return result


def unzip_dicts(*dicts):
    '''
    Unzips the keys and values for any number of dictionaries passed as
    arguments (see below for examples).

    * Arguments:
        - *\*dicts* [dict]: Dictionaries from which key/value pairs are
          to be unzipped.

    * Returns:
        - [list]: Two-element list contianing all keys and all values
          respectively from the dictionaries in *\*dicts*.

    * Example:
        >>> a = dict([('x_a', 2), ('y_a', 3)])
        >>> b = dict([('x_b', 1), ('y_b', -1)])
        >>> unzip_dicts(a, b)
        [('y_a', 'x_a', 'x_b', 'y_b'), (3, 2, 1, -1)]
    '''

    return list(zip(*[(k, v) for d in dicts for (k, v) in d.items()]))
