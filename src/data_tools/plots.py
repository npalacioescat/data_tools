# -*- coding: utf-8 -*-

'''
Module plots
============

Plotting functions module.
'''

__all__ = ['volcano']

import numpy as np
import matplotlib.pyplot as plt # formatting docs

def volcano(logfc, logpval, thr_pval=0.05, thr_fc=2., c=('C0', 'C1'),
            legend=True, title=None, filename=None, figsize=None):
    '''
    Generates a volcano plot from the differential expression data
    provided.

    * Arguments:
        - logfc [list]: Or any iterable type. Contains the log 
          (usually base 2) fold-change values. Must have the same length
          as logpval.
        - logpval [list]: Or any iterable type. Contains the -log
          p-values (usually base 10). Must have the same length as
          logfc.
        - thr_pval [float]: Optional, 0.05 by default. Specifies the
          p-value (non log-transformed) threshold to consider a
          measurement as significantly differentially expressed.
        - thr_fc [float]: Optional, 2. by default. Specifies the FC
          (non log-transformed) threshold to consider a measurement as
          significantly differentially expressed.
        - c [tuple]: Optional, ('C0', 'C1') by default (matplotlib
          default colors). Any iterable containing two color arguments
          tolerated by matplotlib (e.g.: ['r', 'b'] for red and blue).
          First one is used for non-significant points, second for the
          significant ones.
        - legend [bool]: Optional, True by default. Indicates wether to
          show the plot legend or not.
        - title [str]: Optional, None by default. Defines the plot
          title.
        - filename [str]: Optional, None by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - figsize [tuple]: Optional, None by default (default matplotlib
          size). Any iterable containing two values denoting the figure
          size (in inches) as [width, height].

    * Returns:
        - [*matplotlib.figure.Figure*]: Figure object containing the
          volcano plot.

    * Examples:
        >>> volcano(my_log_fc, my_log_pval)

        .. image:: ../figures/volcano_example.png
    '''

    thr_logpval = - np.log10(thr_pval)
    thr_logfc = np.log2(thr_fc)

    max_x, max_y = map(max, [logfc, logpval])
    min_x = min(logfc)

    fig, ax = plt.subplots(figsize=figsize)

    # Boolean vector indicating which measurements are significant
    sig = [p >= thr_logpval and f >= thr_logfc
           for (p, f) in zip(logpval, abs(logfc))]

    # Plotting non-significant points
    ax.scatter([v for (i, v) in enumerate(logfc) if not sig[i]],
               [v for (i, v) in enumerate(logpval) if not sig[i]],
               color=c[0], marker='.', alpha=0.1, label='Non-significant')
    # Plotting significant points
    ax.scatter([v for (i, v) in enumerate(logfc) if sig[i]],
               [v for (i, v) in enumerate(logpval) if sig[i]],
               color=c[1], marker='.', alpha=0.2, label='Significant')

    # Dashed lines denoting thresholds
    ax.plot([min_x - 1, max_x + 1], [thr_logpval, thr_logpval],
            'k--', alpha=0.2) # -log(p-val) threshold line
    ax.plot([-thr_logfc, -thr_logfc], [-1, max_y + 1],
            'k--', alpha=0.2) # log(fc) threshold line (left)
    ax.plot([thr_logfc, thr_logfc], [-1, max_y + 1],
            'k--', alpha=0.2) # log(fc) threshold line (right)

    ax.set_xlim(1.2 * min_x, 1.2 * max_x)
    ax.set_ylim(-0.25, 1.1 * max_y)

    ax.set_xlabel(r'$\log_2(FC)$')
    ax.set_ylabel(r'$-\log_{10}(p$-val$)$')

    if title:
        ax.set_title(title)

    if legend:
        ax.legend()

    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    return fig
