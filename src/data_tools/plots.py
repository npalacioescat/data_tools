# -*- coding: utf-8 -*-

'''
data_tools.plots
================

Plotting functions module.
'''

__all__ = ['density', 'piano_consensus', 'volcano']

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


def density(df, cvf=0.25, title=None, filename=None, figsize=None):
    '''
    Generates a density plot of the values on a data frame (row-wise).

    * Arguments:
        - *df* [pandas.DataFrame]: Contains the values to generate the
          plot. Each row is considered as an individual sample while
          each column contains a measured value.
        - *cvf* [float]: Optional, ``0.25`` by default. Co-variance
          factor used in the gaussian kernel estimation. A higher value
          increases the smoothness.
        - *title* [str]: Optional, ``None`` by default. Defines the plot
          title.
        - *filename* [str]: Optional, ``None`` by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - *figsize* [tuple]: Optional, ``None`` by default (default
          matplotlib size). Any iterable containing two values denoting
          the figure size (in inches) as [width, height].

    * Returns:
        - [*matplotlib.figure.Figure*]: the figure object containing the
          density plot.
    '''

    cmap = matplotlib.cm.get_cmap('rainbow')
    colors = map(cmap, np.linspace(1, 0, len(df.index)))

    fig, ax = plt.subplots(figsize=figsize)

    for i in xrange(len(df)):
        ys = df.iloc[i, :].dropna()
        xs = np.linspace(min(ys), max(ys), 1000)

        dsty = stats.kde.gaussian_kde(ys)
        dsty.covariance_factor = lambda : cvf
        dsty._compute_covariance()

        ax.plot(xs, dsty(xs), c=colors[i], label=df.index[i])
        ax.fill(xs, dsty(xs), c=colors[i], alpha=0.05)

    if title:
        ax.set_title(title)

    ax.legend(ncol=2, fontsize=10, loc=0)
    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    return fig


def piano_consensus(df, nchar=40, boxes=True, title=None, filename=None,
                    figsize=None):
    '''
    Generates a GSEA consensus score plot like R package ``piano``'s
    ``consensusScores`` function, but prettier.
    The main input is assumed to be a ``pandas.DataFrame`` whose data
    is the same as the ``rankMat`` from the result of
    ``consensusScores``.

    * Arguments:
        - *df* [pandas.DataFrame]: Values contained correspond to the
          scores of the gene-sets (consensus and each individual
          methods). Index must contain the gene-set labels. Columns are
          assumed to be ``ConsRank`` (ignored), ``ConsScore`` followed
          by the individual methods (e.g.: ``mean``, ``median``,
          ``sum``, etc).
        - *nchar* [int]: Optional, ``40`` by default. Number of string
          characters of the gene-set labels of the plot.
        - *boxes* [bool]: Optional, ``True`` by default. Determines
          whether to show the boxplots of the gene-sets or not.
        - *title* [str]: Optional, ``None`` by default. Defines the plot
          title.
        - *filename* [str]: Optional, ``None`` by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - *figsize* [tuple]: Optional, ``None`` by default (default
          matplotlib size). Any iterable containing two values denoting
          the figure size (in inches) as [width, height].

    * Returns:
        - [*matplotlib.figure.Figure*]: the figure object containing a
          combination of box and scatter plots of the gene-set scores.

    * Examples:
        >>> piano_consensus(df, figsize=[7, 8])

        .. image:: ../figures/piano_consensus_example.png
           :align: center
           :scale: 60
    '''

    # List of equidistant colors according to a colormap
    cmap = matplotlib.cm.get_cmap('rainbow')
    colors = map(cmap, np.linspace(1, 0, len(df.columns[2:])))

    y = range(len(df))[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    # Gene-set scores for each individual method
    for i, col in enumerate(df.columns[2:]):
        ax.scatter(df[col], y, c=colors[i], alpha=.5, label=col)

    # Box plot of the gene-sets
    if boxes:
        ax.boxplot(df.iloc[:, 2:], positions=y, widths=.75, vert=False,
                   zorder=1, medianprops={'linewidth':0},
                   flierprops={'markersize':7})

    # Consensus score across methods (substitutes the median of the
    # boxplot for visibility)
    col = 'ConsScore'
    ax.scatter(df[col], y, label=col, c='r', zorder=2, s=94,
               marker=[(-.1, 1), (.1, 1),
                       (.1, -1), (-.1, -1)])

    # Gene-set labels
    ax.set_yticks(y)
    ax.set_yticklabels([s[:nchar] for s in df.index])

    # Axes properties
    ax.set_xlabel('Score')
    ax.set_ylabel('Gene-set', rotation=90, ha='center')
    ax.set_ylim(-1, len(df))

    if title:
        ax.set_title(title)

    ax.legend(loc=0)
    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    return fig


def volcano(logfc, logpval, thr_pval=0.05, thr_fc=2., c=('C0', 'C1'),
            legend=True, title=None, filename=None, figsize=None):
    '''
    Generates a volcano plot from the differential expression data
    provided.

    * Arguments:
        - *logfc* [list]: Or any iterable type. Contains the log
          (usually base 2) fold-change values. Must have the same length
          as *logpval*.
        - *logpval* [list]: Or any iterable type. Contains the -log
          p-values (usually base 10). Must have the same length as
          *logfc*.
        - *thr_pval* [float]: Optional, ``0.05`` by default. Specifies
          the p-value (non log-transformed) threshold to consider a
          measurement as significantly differentially expressed.
        - *thr_fc* [float]: Optional, ``2``. by default. Specifies the
          FC (non log-transformed) threshold to consider a measurement
          as significantly differentially expressed.
        - *c* [tuple]: Optional, ``('C0', 'C1')`` by default (matplotlib
          default colors). Any iterable containing two color arguments
          tolerated by matplotlib (e.g.: ``['r', 'b']`` for red and
          blue). First one is used for non-significant points, second
          for the significant ones.
        - *legend* [bool]: Optional, ``True`` by default. Indicates
          whether to show the plot legend or not.
        - *title* [str]: Optional, ``None`` by default. Defines the plot
          title.
        - *filename* [str]: Optional, ``None`` by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - *figsize* [tuple]: Optional, ``None`` by default (default
          matplotlib size). Any iterable containing two values denoting
          the figure size (in inches) as [width, height].

    * Returns:
        - [matplotlib.figure.Figure]: Figure object containing the
          volcano plot.

    * Examples:
        >>> volcano(my_log_fc, my_log_pval)

        .. image:: ../figures/volcano_example.png
           :align: center
           :scale: 60
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
