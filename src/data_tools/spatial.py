# -*- coding: utf-8 -*-

'''
data_tools.spatial
====================

Spatial matrix module.
'''

__all__ = ['get_boundaries', 'neighbour_count']

import numpy as np


def get_boundaries(x, counts=False):
    '''
    Given an array, returns either the mask where the boundary edges are
    or their counts if specified.

    * Arguments:
        - x [numpy.ndarray]: The array where boundaries are to be
          identified or counted. Data type of its elements is totally
          irrelevant.
        - count [bool]: Optional, ``False`` by default. Whether to
          return the number of boundary edges or just their mask.

    * Returns:
        - [numpy.ndarray]: Same shape as *x*. If ``counts=False``,
          contains ``True`` on any cell that is on the boundary,
          ``False`` otherwise. If ``counts=True``, will return a
          similar array but instead of [bool], there will be [int]
          denoting the number of boundary edges of the cells.

    * Examples:
        >>> x = numpy.ones((3, 3, 3))
        >>> get_boundaries(x)
        array([[[ True,  True,  True],
                [ True,  True,  True],
                [ True,  True,  True]],
               [[ True,  True,  True],
                [ True, False,  True],
                [ True,  True,  True]],
               [[ True,  True,  True],
                [ True,  True,  True],
                [ True,  True,  True]]])
        >>> get_boundaries(x, counts=True)
        array([[[3, 2, 3],
                [2, 1, 2],
                [3, 2, 3]],
               [[2, 1, 2],
                [1, 0, 1],
                [2, 1, 2]],
               [[3, 2, 3],
                [2, 1, 2],
                [3, 2, 3]]])
    '''

    bounds = np.ones(x.shape, dtype=int)

    if counts:
        aux = np.pad(bounds, 1, 'constant', constant_values=0)
        counts = neighbour_count(aux)[aux.ndim * (slice(1, -1), )]

        return counts

    else:
        bounds[x.ndim * (slice(1, -1),)] = 0

        return bounds.astype(bool)

def neighbour_count(x):
    '''
    Given an array (up to three dimensions), returns another array with
    the same shape containing the counts of cells' neighbours whose
    value is zero.

    * Arguments:
        - *x* [numpy.ndarray]: The array where to count the neighbours
          (zero-valued cells). Note that the cells can have any value
          or data type. As long as they be converted to [bool], the
          function will count all ``False`` around all ``True`` cells.

    * Returns:
        - [numpy.ndarray]: Array with same shape as *x* containing the
          neighbour count.

    * Examples:
        >>> x = numpy.random.randint(2, size=(5, 5))
        >>> x
        array([[0, 0, 1, 1, 0],
               [0, 0, 0, 1, 1],
               [1, 0, 1, 1, 1],
               [0, 0, 0, 0, 0],
               [1, 0, 0, 1, 1]])
        >>> neighbour_count(x)
        array([[0, 0, 2, 1, 0],
               [0, 0, 0, 1, 1],
               [3, 0, 3, 1, 1],
               [0, 0, 0, 0, 0],
               [2, 0, 0, 2, 1]])
    '''

    assert type(x) is np.ndarray, 'Please provide a NumPy array object'

    dim = x.ndim
    mask = x.astype(bool).astype(int)

    pad_mask = np.pad(mask, 1, 'edge')

    if dim == 1:
        counts = - pad_mask[2:] - pad_mask[:-2]

    elif dim == 2:
        counts = (- pad_mask[1:-1, 2:] - pad_mask[1:-1, :-2]
                  - pad_mask[2:, 1:-1] - pad_mask[:-2, 1:-1])

    elif dim == 3:
        counts = (- pad_mask[1:-1, 1:-1, 2:] - pad_mask[1:-1, 1:-1, :-2]
                  - pad_mask[1:-1, 2:, 1:-1] - pad_mask[1:-1, :-2, 1:-1]
                  - pad_mask[2:, 1:-1, 1:-1] - pad_mask[:-2, 1:-1, 1:-1])

    counts += 2 * dim * mask

    masked_counts = np.ma.masked_array(counts, mask=~mask.astype(bool),
                                       fill_value=0, dtype=int)

    return masked_counts.filled()
