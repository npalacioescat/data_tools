# -*- coding: utf-8 -*-

'''
data_tools.plots
================

Plotting functions module.

Contents
--------
'''

from __future__ import absolute_import

__all__ = ['cmap_bkgr', 'cmap_bkrd','cmap_rdbkgr', 'density',
           'piano_consensus', 'venn', 'volcano']

import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from data_tools.iterables import subsets

if sys.version_info < (3,):
    range = xrange

#: Custom colormap, gradient from black (lowest) to lime green (highest).
cmap_bkgr = LinearSegmentedColormap.from_list(name='BkGr',
                                              colors=['#000000', '#00FF00'],
                                              N=256)

#: Custom colormap, gradient from black (lowest) to red (highest).
cmap_bkrd = LinearSegmentedColormap.from_list(name='BkRd',
                                              colors=['#000000', '#FF0000'],
                                              N=256)

#: Custom colormap, gradient from red (lowest) to black (middle) to lime
#: green (highest).
cmap_rdbkgr = LinearSegmentedColormap.from_list(name='RdBkGr',
                                                colors=['#FF0000', '#000000',
                                                        '#00FF00'],
                                                N=256)

# TODO: Add example figure
def density(df, cvf=0.25, sample_col=False, title=None, filename=None,
            figsize=None):
    '''
    Generates a density plot of the values on a data frame (row-wise).

    * Arguments:
        - *df* [pandas.DataFrame]: Contains the values to generate the
          plot. Each row is considered as an individual sample while
          each column contains a measured value unless otherwise stated
          by keyword argument *sample_col*.
        - *cvf* [float]: Optional, ``0.25`` by default. Co-variance
          factor used in the gaussian kernel estimation. A higher value
          increases the smoothness.
        - *sample_col* [bool]: Optional, ``False`` by default. Specifies
          whether the samples are column-wise or not.
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
          density plot, unless *filename* is provided.
    '''

    df = df.T if sample_col else df

    cmap = matplotlib.cm.get_cmap('rainbow')
    colors = map(cmap, np.linspace(1, 0, len(df.index)))

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(df)):
        ys = df.iloc[i, :].dropna()
        xs = np.linspace(min(ys), max(ys), 1000)

        dsty = stats.kde.gaussian_kde(ys)
        dsty.covariance_factor = lambda : cvf
        dsty._compute_covariance()

        y = dsty(xs)

        ax.plot(xs, y, c=colors[i], label=df.index[i])
        ax.fill(xs, y, c=colors[i], alpha=0.05)

    if title:
        ax.set_title(title)

    ax.legend(ncol=2, fontsize=10, loc=0)
    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    else:
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
          combination of box and scatter plots of the gene-set scores,
          unless *filename* is provided.

    * Example:
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

    else:
        return fig


def venn(N, labels=['A', 'B', 'C', 'D', 'E'], c=['C0', 'C1', 'C2', 'C3', 'C4'],
         pct=False, sizes=False, title=None, filename=None, figsize=None):
    '''
    Plots a Venn diagram from a list of sets *N*. Number of sets must be
    between 2 and 5 (inclusive).

    * Arguments:
        - *N* [list]: Or any iterable type containing [set] objects.
        - *labels* [list]: Optional, ``['A', 'B', 'C', 'D', 'E']`` by
          default. Labels for the sets following the same order as
          provided in *N*.
        - *c* [list]: Optional, ``['C0', 'C1' 'C2', 'C3', 'C4']`` by
          default (matplotlib default colors). Any iterable containing
          color arguments tolerated by matplotlib (e.g.: ``['r', 'b']``
          for red and blue). Must contain at least the same number of
          elements as *N* (if more are provided, they will be ignored).
        - *pct* [bool]: Optional, ``False`` by default. Indicates
          whether to show percentages instead of absolute counts.
        - *sizes* [bool]: Optional, ``False`` by default. Whether to
          include the size of the sets in the legend or not.
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
          combination of box and scatter plots of the gene-set scores,
          unless *filename* is provided.

    * Example:
        >>> N = [{0, 1}, {2, 3}, {1, 3, 4}] # Sets A, B, C
        >>> venn(N)

        .. image:: ../figures/venn_example.png
           :align: center
           :scale: 100
    '''

    if len(N) == 2:
        # Ellipse parameters
        x = [-.25, .25]
        y = [0, 0]
        w = [1, 1]
        h = [1.5, 1.5]
        a = [0, 0]

        # Text (counts) parameters
        xt = [-.5, .5, 0]
        yt = [0, 0, 0]
        keys = ['10', '01', '11']

    elif len(N) == 3:
        # Ellipse parameters
        x = [0, -.25, .25]
        y = [.33, -.33, -.33]
        w = [1, 1, 1]
        h = [1.5, 1.5, 1.5]
        a = [0, 0, 0]

        # Text (counts) parameters
        xt = [0, -.5, .5, -.33, .33, 0, 0]
        yt = [.6, -.5, -.5, .15, .15, -.6, -.1]
        keys = ['100', '010', '001', '110', '101', '011', '111']

    elif len(N) == 4:
        # Ellipse parameters
        x = [-.15, -.35, .15, .35]
        y = [.15, -.25, .15, -.25]
        w = [2, 2, 2, 2]
        h = [1, 1, 1, 1]
        a = [-60, -60, 60, 60]

        # Text (counts) parameters
        xt = [-.5, -.8, .5, .8, -.6, 0, .4, -.4, 0, .6, -.3, .2, .3, -.2, 0]
        yt = [.8, -.1, .8, -.1, .33, .4, -.55, -.55, -.9, .33, 0, -.68, 0,
              -.68, -.33]
        keys = ['1000', '0100', '0010', '0001', '1100', '1010', '1001', '0110',
                '0101', '0011', '1110', '1101', '1011', '0111', '1111']

    elif len(N) == 5:
        # Ellipse parameters
        x = [0, -.2125, -.2375, -.03125, .125]
        y = [0, -.05, -.275, -.35, -.1875]
        w = [1.25, 1.25, 1.25, 1.25, 1.25]
        h = [2, 2, 2, 2, 2]
        a = [0, 71, 154, 37, 108]

        # Text (counts) parameters
        xt = [0, -1, -.7, .5, .9, -.41, .1, .2, .51, -.87, -.67, .69, -.25,
              -.69, .6, -.1, -.51, .54, -.06, .33, .3, -.72, -.79, .63, -.48,
              -.38, .3, .49, -.13, -.69, -.1]
        yt = [.8, .1, -1, -1, .05, .55, .55, -.91, .3, -.3, .3, -.25, -1, -.65,
              -.65, .5, .43, 0, -.95, .4, -.78, 0, -.41, -.47, -.76, .33, .27,
              -.4, -.82, -.25, -.2]
        keys = ['10000', '01000', '00100', '00010', '00001', '11000', '10100',
                '10010', '10001', '01100', '01010', '01001', '00110', '00101',
                '00011', '11100', '11010', '11001', '10110', '10101', '10011',
                '01110', '01101', '01011', '00111', '11110', '11101', '11011',
                '10111', '01111', '11111']

    else:
        return 'The maximum number of sets supported is 5.'

    ssets = subsets(N)
    # Subset counts
    text = dict(zip(ssets.keys(), map(len, ssets.values())))

    if pct:
        total = float(sum(text.values()))
        text = dict(zip(text.keys(),
                          np.round(100 * np.array(text.values()) / total,
                                   decimals=2)))

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(N)):
        ellipse(ax, x[i], y[i], w[i], h[i], a[i], alpha=.25, color=c[i],
                label='%s (%d)' %(labels[i], len(N[i])) if sizes
                else labels[i])

    for i in range(len(text)):
        ax.text(xt[i], yt[i], text[keys[i]], fontdict={'ha':'center'})

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    if title:
        ax.set_title(title)

    ax.legend()

    ax.axis('off')
    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    else:
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
          volcano plot, unless *filename* is provided.

    * Example:
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
               color=c[0], marker='.', alpha=0.2, label='Non-significant')
    # Plotting significant points
    ax.scatter([v for (i, v) in enumerate(logfc) if sig[i]],
               [v for (i, v) in enumerate(logpval) if sig[i]],
               color=c[1], marker='.', alpha=0.3, label='Significant')

    # Dashed lines denoting thresholds
    ax.plot([min_x - 1, max_x + 1], [thr_logpval, thr_logpval],
            'k--', alpha=0.3) # -log(p-val) threshold line
    ax.plot([-thr_logfc, -thr_logfc], [-1, max_y + 1],
            'k--', alpha=0.3) # log(fc) threshold line (left)
    ax.plot([thr_logfc, thr_logfc], [-1, max_y + 1],
            'k--', alpha=0.3) # log(fc) threshold line (right)

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

    else:
        return fig


###############################################################################


def ellipse(ax, x, y, w, h, a, color, alpha=1, label=None):
    e = matplotlib.patches.Ellipse(xy=(x, y), width=w, height=h, angle=a,
                                   color=color, alpha=alpha, label=label)

    ax.add_patch(e)
