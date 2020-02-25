# -*- coding: utf-8 -*-

'''
data_tools.plots
================

Plotting functions module.

Contents
--------
'''

from __future__ import absolute_import

__all__ = ['cmap_bkgr', 'cmap_bkrd', 'cmap_rdbkgr', 'chordplot',
           'cluster_hmap', 'density', 'pca', 'phase_portrait',
           'piano_consensus', 'similarity_heatmap', 'similarity_histogram',
           'upset_wrap', 'venn', 'volcano']

import sys
import itertools

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import upsetplot as usp
from sklearn.decomposition import PCA
from adjustText import adjust_text

from data_tools.iterables import subsets, similarity

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


def chordplot(nodes, edges, alpha=0.2, plot_lines=False, labels=False,
              label_sizes=False, colors=None, title=None, filename=None,
              figsize=None):
    '''
    Generates a chord plot from a collection of nodes and edges (and
    their sizes). **NOTE**: Make sure that all nodes are involved in at
    least one edge (with size > 0).

    * Arguments:
        - *nodes* [dict]: Can also be [pandas.DataFrame] or
          [pandas.Series]. Values contain the nodes sizes and the keys/
          indices the node names (must correspond to the ones in
          *edges*).
        - *edges* [pd.DataFrame]: Can also be [numpy.ndarray] or [list]
          of [list] as long as contains *n* by 3 elements where *n* is
          the number of edges and their elements describe the source and
          target nodes and the size of that edge.
        - *alpha* [float]: Optional, ``0.2`` by default. Sets the
          transparency of the edges.
        - *plot_lines* [bool]: Optional, ``False`` by default. Whether
          to plot the edge borders or not.
        - *labels* [bool]: Optional, ``False`` by default. If ``True``
          will label de nodes according to their index/key inputed in
          the first argument. Otherwise can be [list] or other iterable
          containing other labels (same order as provided in *nodes*).
        - *label_sizes* [bool]: Optional, ``False`` by default. Sets
          whether to append the node sizes when labelling the plot.
        - *colors* [list]: Optional, ``None`` by default (matplotlib
          default color sequence). Any iterable containing color
          arguments tolerated by matplotlib (e.g.: ``['r', 'b']`` for
          red and blue). Must contain at least the same number of
          elements as *nodes* (if more are provided, they will be
          ignored).
        - *title* [str]: Optional, ``None`` by default. Defines the plot
          title.
        - *filename* [str]: Optional, ``None`` by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - *figsize* [tuple]: Optional, ``None`` by default (default
          matplotlib size). Any iterable containing two values denoting
          the figure size (in inches) as [width, height].

    * Returns:
        - [matplotlib.figure.Figure]: The figure object containing the
          chord plot, unless *filename* is provided.

    * Example:
        >>> nodes = {'A':5, 'B':10, 'C':20, 'D':15, 'E':15}
        >>> edges = [['A', 'B', 10],
        ...          ['A', 'C', 25],
        ...          ['B', 'C', 50],
        ...          ['D', 'E', 5],
        ...          ['B', 'E', 30],
        ...          ['C', 'D', 20]]
        >>> chordplot(nodes, edges, plot_lines=True)

        .. image:: ../figures/chordplot_example.png
           :align: center
           :scale: 80
    '''

    # Preparing inputs
    # Edge properties table
    if type(edges) is pd.DataFrame:
        edges.columns = ['source', 'target', 'size']
    else:
        edges = pd.DataFrame(edges, columns=['source', 'target', 'size'])

    # Node properties table
    if type(nodes) is dict:
        nodes = pd.Series(nodes)

    nodes = pd.DataFrame(nodes, columns=['size'])

    # Checking color list
    if colors:
        msg = ('List of colors (%d) is not the same length as nodes (%d)'
               % (len(colors), len(nodes)))
        assert len(colors) == len(nodes), msg

    colors = colors or ['C%d' % i for i in range(len(nodes))]

    # Checking label list
    if labels and type(labels) is not bool:
        msg = ('List of labels (%d) is not the same length as nodes (%d)'
               % (len(labels), len(nodes)))
        assert len(labels) == len(nodes), msg

    elif labels is True:
        labels = nodes.index.to_list()

    else:
        labels = None

    if label_sizes and labels:
        labels = [lab + ' (%s)' % str(nodes['size'][i]) for (i, lab)
                  in enumerate(labels)]

    # Node relative sizes (wrt. sum of all node sizes) - int/float
    nodes['rel_size'] = [s / nodes['size'].sum() for s in nodes['size']]
    # Edge sizes for each node (involved in them) - list
    nodes['e_sizes'] = [edges.loc[(edges.source == n) | (edges.target == n),
                                  'size'].values for n in nodes.index.values]
    # Total edge sizes involving a node - int/float
    nodes['tot_e_size'] = [sum(x) for x in nodes['e_sizes']]
    # Relative edge sizes (wrt. total edges involving that node) - list
    nodes['rel_e_sizes'] = nodes['e_sizes'].values / nodes['tot_e_size'].values
    # Global relative edge sizes (relative edge * relative node size) - list
    nodes['glob_e_sizes'] = nodes['rel_e_sizes'] * nodes['rel_size']

    # Flattened list of global edge sizes (ordered by node positions) - sum = 1
    rel_e_pos = np.concatenate(nodes['glob_e_sizes'])
    # Cummulative edge positions (starting from 0) - list [0, 1]
    cum_rel_e_pos = [0]

    for i in range(len(rel_e_pos)):
        cum_rel_e_pos.append(cum_rel_e_pos[i] + rel_e_pos[i])

    # List of edge positions - len = len(nodes)
    # Each element is an array of shape (n, 2) where n is the edges involving
    # that node and 2 are the (start, end) positions of such edge
    counter = 0
    e_pos = []
    for n in nodes.index:
        es = len(nodes.loc[n, 'glob_e_sizes'])
        aux = np.array([[0, 0] * es], dtype=float).reshape(es, 2)

        for e in range(es):
            aux[e, :] = np.array(cum_rel_e_pos[counter:counter + 2])
            counter += 1

        e_pos.append(aux)

    # Storing those edge positions to each related node
    nodes['e_pos'] = e_pos

    # Keep a counter of added edges on each node
    counter = dict((n, 0) for n in nodes.index)

    # Plotting the donut and edges
    fig, ax = plt.subplots(figsize=figsize)

    for i, e in edges.iterrows():
        # Source/target node names
        s = e['source']
        t = e['target']
        # Color according to source node
        c = colors[nodes.index.to_list().index(s)]

        # Retrieve relative positions of edges on a circle
        if s != t:
            # - Source points of edge
            s1, s2 = nodes.loc[s, 'e_pos'][counter[s]]
            ps1, ps2 = map(get_rel_pos_circ, [s1, s2])
            # - Target points of edge (swapped)
            t1, t2 = nodes.loc[t, 'e_pos'][counter[t]][::-1]
            pt1, pt2 = map(get_rel_pos_circ, [t1, t2])
            # Count the nodes' edges
            counter[s] += 1
            counter[t] += 1

            # Generate borders of edge as Bézier curves
            curve1 = bezier_quad(ps1, pt1)
            curve2 = bezier_quad(ps2, pt2)

            # Filling the gaps (arcs between borders of edge)
            sarcr = np.linspace(2 * np.pi * s1, 2 * np.pi * s2, 100)
            sarc = np.vstack([np.cos(sarcr), np.sin(sarcr)])

            tarcr = np.linspace(2 * np.pi * t1, 2 * np.pi * t2, 100)
            tarc = np.vstack([np.cos(tarcr), np.sin(tarcr)])

            # Plotting the edge borders
            if plot_lines:
                ax.plot(*curve1, c=c, zorder=0)
                ax.plot(*curve2, c=c, zorder=0)

            # Filling the edge
            ax.fill(np.concatenate([curve1[0], tarc[0][::-1],
                                    curve2[0][::-1], sarc[0][::-1]],
                                   axis=None),
                    np.concatenate([curve1[1], tarc[1][::-1],
                                    curve2[1][::-1], sarc[1][::-1]],
                                   axis=None),
                    color=c, alpha=alpha, zorder=0)

        else:
            # - Points of edge
            s1, s2 = nodes.loc[s, 'e_pos'][counter[s]]
            ps1, ps2 = map(get_rel_pos_circ, [s1, s2])

            # Count the nodes' edges
            counter[s] += 1

            # Generate borders of edge as Bézier curves
            curve1 = bezier_quad(ps1, ps2)

            # Filling the gaps (arcs between borders of edge)
            sarcr = np.linspace(2 * np.pi * s1, 2 * np.pi * s2, 100)
            sarc = np.vstack([np.cos(sarcr), np.sin(sarcr)])

            # Plotting the edge borders
            if plot_lines:
                ax.plot(*curve1, c=c, zorder=0)

            # Filling the edge
            ax.fill(np.concatenate([curve1[0], sarc[0][::-1]], axis=None),
                    np.concatenate([curve1[1], sarc[1][::-1]], axis=None),
                    color=c, alpha=alpha, zorder=0)

    # Plotting the donut on top (slightly bigger radius to cover edge tips)
    ax.pie(nodes['size'], wedgeprops=dict([('width', 0.1)]), radius=1.01,
           labels=labels, rotatelabels=True, colors=colors)

    ax.set_title(title)

    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    else:
        return fig


def cluster_hmap(matrix, xlabels=None, ylabels=None, title=None, filename=None,
                 figsize=None, cmap='viridis', link_kwargs={},
                 dendo_kwargs={}):
    '''
    Generates a heatmap with hierarchical clustering dendrograms
    attached. The linkage matrix and dendrogram are computed using the
    module :py:mod:`scipy.cluster.hierarchy`, you may check the
    corresponding documentation for available and default methods.

    * Arguments:
        - *matrix* [numpy.ndarray]: Contains the values to generate the
          plot. It is assumed to be a 2-dimensional matrix.
        - *xlabels* [list]: Optional, ``None`` by default. Labels for
          the x-axis of the matrix following the same order as provided
          (e.g. sample names).
        - *ylabels* [list]: Optional, ``None`` by default. Labels for
          the y-axis of the matrix following the same order as provided
          (e.g. measurements).
        - *title* [str]: Optional, ``None`` by default. Defines the plot
          title.
        - *filename* [str]: Optional, ``None`` by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - *figsize* [tuple]: Optional, ``None`` by default (default
          matplotlib size). Any iterable containing two values denoting
          the figure size (in inches) as [width, height].
        - *cmap* [str]: Optional, ``'viridis'`` by default. The colormap
          used for the plot (can also be a [matplotlib.colors.Colormap]
          object). See other [str] options available in `Matplotlib's
          reference manual`_.
        - *link_kwargs* [dict]: Optional, ``{}`` by default. Dictionary
          containing the key-value pairs for keyword arguments passed to
          `scipy.cluster.hierarchy.linkage`_.
        - *dendo_kwargs* [dict]: Optional, ``{}`` by default. Dictionary
          containing the key-value pairs for keyword arguments passed to
          `scipy.cluster.hierarchy.dendrogram`_.

    .. _`Matplotlib's reference manual`:
        https://matplotlib.org/examples/color/colormaps_reference.html
    .. _`scipy.cluster.hierarchy.linkage`:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/sci\
        py.cluster.hierarchy.linkage.html
    .. _`scipy.cluster.hierarchy.dendrogram`:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/sci\
        py.cluster.hierarchy.dendrogram.html

    * Returns:
        - [matplotlib.figure.Figure]: The figure object containing the
          density plot, unless *filename* is provided.
    '''

    # TODO: matrix must be complete cases, find major axis and remove NaNs
    # TODO: fix title (overlaps with dendogram)

    xlinked = linkage(matrix.T, **link_kwargs)
    ylinked = linkage(matrix, **link_kwargs)

    fig, ax = plt.subplots(figsize=figsize or (7, 7), nrows=2, ncols=2,
                           gridspec_kw={'height_ratios': [1, 7],
                                        'width_ratios': [7, 1]})
    # Upper dendrogram + store info from clustering
    xdendo = dendrogram(xlinked, ax=ax[0, 0], link_color_func=lambda k: 'k',
                        **dendo_kwargs)
    # Right-hand dendrogram
    ydendo = dendrogram(ylinked, ax=ax[1, 1], link_color_func=lambda k: 'k',
                        orientation='right', **dendo_kwargs)

    # Fixing order of side dendrogram (invert y-axes)
    ax[1, 1].set_ylim(10 * matrix.shape[0], 0)

    ord_mat = matrix[:, xdendo['leaves']][ydendo['leaves'], :]

    im = ax[1, 0].imshow(ord_mat, interpolation='none', cmap=cmap,
                         aspect='auto')

    # Share x/y axes with dendrograms
    ax[1, 0].get_shared_x_axes().join(ax[0, 0], ax[1, 0])
    ax[1, 0].get_shared_y_axes().join(ax[1, 1], ax[1, 0])

    # Remove axis from dendrograms
    ax[0, 0].set_axis_off()
    ax[1, 1].set_axis_off()

    if xlabels is not None:
        ord_xlab = [xlabels[i] for i in xdendo['leaves']]
        rng = range(len(ord_xlab))
        ax[1, 0].set_xticks(rng)
        ax[1, 0].set_xticklabels(ord_xlab, rotation=90)
    if ylabels is not None:
        ord_ylab = [ylabels[i] for i in ydendo['leaves']]
        rng = range(len(ord_ylab))
        ax[1, 0].set_yticks(rng)
        ax[1, 0].set_yticklabels(ord_ylab)

    fig.suptitle(title)

    fig.colorbar(im)
    fig.delaxes(ax[0, 1])

    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    if filename:
        fig.savefig(filename)

    else:
        return fig


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
        - [matplotlib.figure.Figure]: The figure object containing the
          density plot, unless *filename* is provided.
    '''

    df = df.T if sample_col else df

    cmap = matplotlib.cm.get_cmap('rainbow')
    colors = list(map(cmap, np.linspace(1, 0, len(df.index))))

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(df)):
        ys = df.iloc[i, :].dropna()
        xs = np.linspace(min(ys), max(ys), 1000)

        dsty = stats.kde.gaussian_kde(ys)
        dsty.covariance_factor = lambda: cvf
        dsty._compute_covariance()

        y = dsty(xs)

        ax.plot(xs, y, c=colors[i], label=df.index[i])
        ax.fill(xs, y, c=colors[i], alpha=0.05)

    ax.set_title(title)

    ax.legend(ncol=2, fontsize=10, loc=0)
    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    else:
        return fig


def pca(data, n_comp=2, groups=None, cmap='rainbow', title=None,
        filename=None, figsize=None):
    '''
    Computes the principal component analysis (PCA) and plots the
    results.

    * Arguments:
        - *data* [pandas.DataFrame]: Contains the high-dimensional data
          to compute the PCA. The samples (data points) are assumed to
          be rows and the measurements (features) in the columns.
        - *n_comp* [int]: Optional, ``2`` by default. The number of
          first components to plot. Maximum is three (otherwise defaults
          to two).
        - *groups* [dict]: Optional, ``None`` by default. Can also be a
          [pandas.Series]. Defines which samples (keys/index
          corresponding to row names in *data*) belong to a group (e.g.
          condition, treatment, ...) to color the data points. If none
          is provided, all points are colored in black.
        - *cmap* [str]: Optional, ``'rainbow'`` by default. The colormap
          used to draw the groups' colors (can also be a user-defined
          [matplotlib.colors.Colormap] object). See other [str] options
          available in `Matplotlib's reference manual`_. If no *groups*
          are provided, the argument is ignored.
        - *title* [str]: Optional, ``None`` by default. Defines the plot
          title.
        - *filename* [str]: Optional, ``None`` by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - *figsize* [tuple]: Optional, ``None`` by default (default
          matplotlib size). Any iterable containing two values denoting
          the figure size (in inches) as [width, height].

    .. _`Matplotlib's reference manual`:
        https://matplotlib.org/examples/color/colormaps_reference.html

    * Returns:
        - [matplotlib.figure.Figure]: The figure object containing the
          scatter plot of the PC's unless *filename* is provided.
    '''

    # Preparing the figure and axes
    fig = plt.figure(figsize=figsize)
    ax = (fig.add_axes(([0.1, 0.1, 0.65, 0.8] if groups is not None
                        else [0.1, 0.1, 0.8, 0.8]),
                       projection='3d' if n_comp == 3 else None),
          fig.add_axes([0.75, 0.1, 0.25, 0.8]) if groups is not None
          else None)

    # Removing NaN's from data
    print('Data contains %d rows and %d columns' % data.shape)
    data.dropna(axis=1, inplace=True)
    print('After removing NaNs %d rows and %d columns remain' % data.shape)

    if groups is not None:
        # Generating color palette
        assert (type(groups) == dict or type(groups) == pd.Series),\
               'Please, provide a dict or pandas.Series for groups'
        groups = pd.Series(groups)
        # Colormap instance
        icmp = matplotlib.cm.get_cmap(cmap)
        unique_groups = pd.unique(groups)
        palette = dict(zip(unique_groups,
                           list(map(icmp,
                                    np.linspace(1, 0, len(unique_groups))))))

        colors = [palette[groups[dtp]] for dtp in data.index]

    else:
        colors = 'black'

    # Reducing dimensions
    pca = PCA(n_components=n_comp)
    pca.fit(data)

    x = pca.transform(data)

    # Variance explained per component
    exp_var = pca.explained_variance_ratio_
    ax[0].set_xlabel('PC1 (%.1f%%)' % (exp_var[0] * 100))
    ax[0].set_ylabel('PC2 (%.1f%%)' % (exp_var[1] * 100))

    # In case of 3D
    if n_comp == 3:
        ax[0].scatter(x[:, 0], x[:, 1], x[:, 2], c=colors)
        ax[0].set_zlabel('PC3 (%.1f%%)' % (exp_var[2] * 100))

    else:
        ax[0].scatter(x[:, 0], x[:, 1], c=colors)
        ax[0].axis('square')

    # If groups are provided, add legend
    if groups is not None:
        ax[1].legend([matplotlib.lines.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=c)
                      for c in palette.values()],
                     palette.keys(), loc='center left')
        ax[1].set_axis_off()

    ax[0].set_title(title)

    if filename:
        fig.savefig(filename)

    else:
        return fig


def phase_portrait(f, x=(0, 1), y=(0, 1), ics=None, dt=0.1, title=None,
                   filename=None, figsize=None):
    '''
    Generates a phase portrait of a ODE system given the nullclines.
    This is, for a system of the form:

    .. math::
        \\left\\{\\begin{array}{l}
        \\frac{\\text{d}u}{\\text{d}t}=f(u,v)\\\\
        \\frac{\\text{d}v}{\\text{d}t}=g(u,v)
        \\end{array}\\right.

    The nullclines are obtained by equalizing the derivatives to zero
    and isolating :math:`v` on each equation. The argument *f* is
    therefore expected to return the pair of values :math:`v` in terms
    of :math:`u` for each nullcline. See below for an example.

    * Arguments:
        - *f* [function]: Defines the two nullclines of the system.
        - *x* [tuple]: Optional, ``(0, 1)`` by default. The range of
          values to span the x-axis.
        - *y* [tuple]: Optional, ``(0, 1)`` by default. The range of
          values to span the y-axis.
        - *ics* [tuple]: Optional, ``None`` by default. Set of initial
          conditions to plot a trajectory. Must have three elements: the
          initial values of each component and the amount of time the
          trajectory is simulated.
        - *dt* [float]: Optional ``0.1`` by default. The time-step to
          simulate the trajectory (given *ics* is provided).
        - *title* [str]: Optional, ``None`` by default. Defines the plot
          title.
        - *filename* [str]: Optional, ``None`` by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - *figsize* [tuple]: Optional, ``None`` by default (default
          matplotlib size). Any iterable containing two values denoting
          the figure size (in inches) as [width, height].

    * Returns:
        - [matplotlib.figure.Figure]: The figure object containing the
          phase portrait of the system, unless *filename* is provided.

    * Example:
        Let's assume we want the phase portait of the following system:

        .. math::
            \\left\\{\\begin{array}{l}
            \\frac{\\text{d}u}{\\text{d}t}=u-5(u-2)^3+4-v\\\\
            \\frac{\\text{d}v}{\\text{d}t}=3u-v
            \\end{array}\\right.

        Then:

        >>> def f(u):
        ...     return [u - 5 * (u - 2) ** 3 + 4,
        ...             3 * u]
        >>> phase_portrait(f, x=(1, 3), y=(4, 8), ics=[1.25, 5.5, 100])

        .. image:: ../figures/phase_portrait_example.png
           :align: center
           :scale: 80
    '''

    fig, ax = plt.subplots(figsize=figsize)

    # Plotting nullclines
    xs = np.linspace(x[0], x[1], 101)
    ys = np.linspace(y[0], y[1], 101)
    y1, y2 = f(xs)

    ax.plot(xs, y1)
    ax.plot(xs, y2)

    # Plotting quiver
    mshx, mshy = np.meshgrid(xs[::5], ys[::5])
    u, v = f(mshx)

    plt.quiver(mshx, mshy, u - mshy, v - mshy, alpha=0.5)

    # Plotting trajectory
    if ics:
        x0, y0, t = ics

        xi = [x0]
        yi = [y0]

        for _ in np.arange(0, t, dt):
            auxx, auxy = f(xi[-1])
            xn = xi[-1] + dt * (auxx - yi[-1])
            yn = yi[-1] + dt * (auxy - yi[-1])

            xi.append(xn)
            yi.append(yn)

        ax.plot(xi, yi)
        ax.scatter(x0, y0, c='C2')

    ax.set_ylim(*y)
    ax.set_xlim(*x)

    ax.set_title(title)

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
        - [matplotlib.figure.Figure]: The figure object containing a
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
    colors = list(map(cmap, np.linspace(1, 0, len(df.columns[2:]))))

    y = list(range(len(df)))[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    # Gene-set scores for each individual method
    for i, col in enumerate(df.columns[2:]):
        ax.scatter(df[col], y, c=colors[i], alpha=.5, label=col)

    # Box plot of the gene-sets
    if boxes:
        ax.boxplot(df.iloc[:, 2:], positions=y, widths=.75, vert=False,
                   zorder=1, medianprops={'linewidth': 0},
                   flierprops={'markersize': 7})

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

    ax.set_title(title)

    ax.legend(loc=0)
    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    else:
        return fig


def similarity_heatmap(groups, labels=None, mode='j', cmap='nipy_spectral',
                       title=None, filename=None, figsize=None):
    '''
    Given a group of sets, generates a heatmap with the similarity
    indices across each possible pair.

    * Arguments:
        - *groups* [list]: Or any iterable of [set] objects.
        - *labels* [list]: Optional, ``None`` by default. Labels for the
          sets following the same order as provided in *groups*.
        - *mode* [str]: Optional, ``'j'`` (Jaccard) by default.
          Indicates which type of similarity index/coefficient is to be
          computed. Available options are: ``'j'`` for Jaccard, ``'sd'``
          for Sorensen-Dice and ``'ss'`` for Szymkiewicz–Simpson. See
          :py:func:`data_tools.iterables.similarity` for more
          information.
        - *cmap* [str]: Optional, ``'nipy_spectral'`` by default. The
          colormap used for the plot (can also be a
          [matplotlib.colors.Colormap] object). See other [str] options
          available in `Matplotlib's reference manual`_.
        - *title* [str]: Optional, ``None`` by default. Defines the plot
          title.
        - *filename* [str]: Optional, ``None`` by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - *figsize* [tuple]: Optional, ``None`` by default (default
          matplotlib size). Any iterable containing two values denoting
          the figure size (in inches) as [width, height].

    .. _`Matplotlib's reference manual`:
        https://matplotlib.org/examples/color/colormaps_reference.html

    * Returns:
        - [matplotlib.figure.Figure]: The figure object containing a
          combination of box and scatter plots of the gene-set scores,
          unless *filename* is provided.
    '''

    sims = []

    for (a, b) in itertools.product(groups, repeat=2):
        sims.append(similarity(set(a), set(b), mode=mode))

    # Convert similarity indices to square matrix
    sims = np.array(sims).reshape(len(groups), len(groups))

    # Plotting heatmap for a given similarity index
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sims, cmap=cmap, interpolation='none')
    fig.colorbar(im)

    if labels:

        try:
            a, b = map(len, [groups, labels])
            assert a == b

        except AssertionError as e:
            raise e('Invalid length of labels %d != %d' % (a, b))

        rng = range(len(groups))

        ax.set_xticks(rng)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(rng)
        ax.set_yticklabels(labels)

    ax.set_title(title)

    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    else:
        return fig


def similarity_histogram(groups, mode='j', bins=10, title=None, filename=None,
                         figsize=None):
    '''
    Given a group of sets, generates a histogram of the similarity
    indices across each possible pair (same-element pairs excluded).

    * Arguments:
        - *groups* [list]: Or any iterable of [set] objects.
        - *mode* [str]: Optional, ``'j'`` (Jaccard) by default.
          Indicates which type of similarity index/coefficient is to be
          computed. Available options are: ``'j'`` for Jaccard, ``'sd'``
          for Sorensen-Dice and ``'ss'`` for Szymkiewicz–Simpson. See
          :py:func:`data_tools.iterables.similarity` for more
          information.
        - *bins* [int]: Optional, ``10`` by default. Number of bins to
          show in the histogram.
        - *title* [str]: Optional, ``None`` by default. Defines the plot
          title.
        - *filename* [str]: Optional, ``None`` by default. If passed,
          indicates the file name or path where to store the figure.
          Format must be specified (e.g.: .png, .pdf, etc)
        - *figsize* [tuple]: Optional, ``None`` by default (default
          matplotlib size). Any iterable containing two values denoting
          the figure size (in inches) as [width, height].

    * Returns:
        - [matplotlib.figure.Figure]: The figure object containing a
          combination of box and scatter plots of the gene-set scores,
          unless *filename* is provided.
    '''

    sims = [similarity(a, b, mode=mode) for (a, b)
            in itertools.combinations_with_replacement(groups, 2)]

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(sims, bins=bins)
    ax.set_xlabel('Similarity index')
    ax.set_ylabel('Frequency')

    ax.set_title(title)

    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    else:
        return fig


def upset_wrap(N, labels=None, drop_empty=False, **kwargs):
    '''
    Wrapper for UpSetPlot package. Mostly just generates the Boolean
    multi-indexed ``pandas.Series`` the ``upsetplot.plot`` function
    needs as input.

    * Arguments:
        - *N* [list]: Or any iterable type containing [set] objects.
        - *labels* [list]: Optional, ``None`` by default. Labels for the
          sets following the same order as provided in *N*. If none is
          passed they will be labelled ``'set0'``, ``'set1'`` and so on.
        - *drop_empty* [bool]: Optional, ``False`` by default. Whether
          to remove the empty set intersections from the plot or not.
        - *\*\*kwargs*: Optional. Additional keyword arguments passed to
          ``upsetplot.UpSet`` class.

    * Returns:
        [dict]: Contains the ``matplotlib.axes.Axes`` instances for the
        UpSetPlot figure.
    '''

    ss = subsets(N)
    ids = [tuple([bool(int(j)) for j in i]) for i in ss.keys()]
    counts = list(map(len, ss.values()))

    if labels is None:
        labels = ['set%d' % i for i in range(len(N))]

    idx = pd.MultiIndex.from_tuples(ids, names=labels)
    series = pd.Series(counts, index=idx)

    if drop_empty:
        series[series == 0] = np.nan
        series.dropna(inplace=True)

    return usp.plot(series, **kwargs)


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
        - [matplotlib.figure.Figure]: The figure object containing a
          combination of box and scatter plots of the gene-set scores,
          unless *filename* is provided.

    * Example:
        >>> N = [{0, 1}, {2, 3}, {1, 3, 4}] # Sets A, B, C
        >>> venn(N)

        .. image:: ../figures/venn_example.png
           :align: center
           :scale: 100
    '''

    def ellipse(ax, x, y, w, h, a, color, alpha=1, label=None):
        e = matplotlib.patches.Ellipse(xy=(x, y), width=w, height=h, angle=a,
                                       color=color, alpha=alpha, label=label)

        ax.add_patch(e)

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
                label='%s (%d)' % (labels[i], len(N[i])) if sizes
                else labels[i])

    for i in range(len(text)):
        ax.text(xt[i], yt[i], text[keys[i]], fontdict={'ha': 'center'})

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    ax.set_title(title)

    ax.legend(loc=0)

    ax.axis('off')
    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    else:
        return fig


def volcano(logfc, logpval, thr_pval=0.05, thr_fc=2., c=('C0', 'C1'),
            labels=None, maxlabels=25, legend=True, title=None, filename=None,
            figsize=None, adj_txt_kwargs={}):
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
        - *labels* [list]: Optional, ``None`` by default. List of labels
          of the points, only the significant ones will be labeled. Must
          be in the same order as *logfc* and *logpval*.
        - *maxlabels* [int]: Optional, ``25`` by default. Maximum number
          of labels to show to avoid overcrowding. If the number of
          labels to show is above, there will be only shown the top data
          points up to the threshold. This ranking is computed as the
          product of the *logpval* and absolute *logfc*.
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
        - *adj_txt_kwargs* [dict]: Optional, ``{}`` by default. The key,
          value pairs of keyword arguments to pass to adjust_text
          function (see `adjustText reference manual`_ for more
          information).

    .. _`adjustText reference manual`:
        https://adjusttext.readthedocs.io/en/latest/

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

    # Converting data into numpy arrays
    logfc = np.array(logfc)
    logpval = np.array(logpval)

    max_x, max_y = map(max, [logfc, logpval])
    min_x = min(logfc)

    fig, ax = plt.subplots(figsize=figsize)

    # Boolean vector indicating which measurements are significant
    sig = [p >= thr_logpval and f >= thr_logfc
           for (p, f) in zip(logpval, abs(logfc))]

    sig_df = pd.DataFrame([(labels[i], logpval[i], logfc[i])
                           for i, v in enumerate(sig) if v],
                          columns=['label', 'logpval', 'logfc'])
    non_sig_df = pd.DataFrame([(labels[i], logpval[i], logfc[i])
                               for i, v in enumerate(sig) if not v],
                              columns=['label', 'logpval', 'logfc'])

    # Plotting significant points
    ax.scatter(sig_df['logfc'], sig_df['logpval'], color=c[1], marker='.',
               alpha=0.3, label='Significant', zorder=2.5)

    # Plotting non-significant points
    ax.scatter(non_sig_df['logfc'], non_sig_df['logpval'], color=c[0],
               marker='.', alpha=0.3, label='Non-significant', zorder=2.5)

    if labels is not None:
        if sum(sig) > maxlabels:
            sig_df['rank'] = sig_df['logpval'] * abs(sig_df['logfc'])
            sig_df.sort_values('rank', ascending=False, inplace=True)
            sig_df = sig_df.iloc[:maxlabels, :]

        txts = [ax.text(r['logfc'], r['logpval'], r['label'], ha='center',
                         va='center', size=5, zorder=5.5)
                for i, r in sig_df.iterrows()]
        adjust_text(txts, ax=ax, **adj_txt_kwargs)

    # Dashed lines denoting thresholds
    ax.plot([min_x - 1, max_x + 1], [thr_logpval, thr_logpval],
            'k--', alpha=0.3, zorder=3.5)  # -log(p-val) threshold line
    ax.plot([-thr_logfc, -thr_logfc], [-1, max_y + 1],
            'k--', alpha=0.3, zorder=3.5)  # log(fc) threshold line (left)
    ax.plot([thr_logfc, thr_logfc], [-1, max_y + 1],
            'k--', alpha=0.3, zorder=3.5)  # log(fc) threshold line (right)

    ax.set_xlim(1.2 * min_x, 1.2 * max_x)
    ax.set_ylim(-0.25, 1.1 * max_y)

    ax.set_xlabel(r'$\log_2(FC)$')
    ax.set_ylabel(r'$-\log_{10}(p$-val$)$')

    ax.set_title(title)

    if legend:
        ax.legend(loc=0)

    fig.tight_layout()

    if filename:
        fig.savefig(filename)

    else:
        return fig


###############################################################################
#+---------------------------------------------------------------------------+#
#|                           SUPLEMENTARY FUNCTIONS                          |#
#+---------------------------------------------------------------------------+#
###############################################################################


def get_rel_pos_circ(pt, r=1):  # NOTE: Move to spatial module?
    '''
    Returns the x, y coordinates on a circle of radius r (centered at
    (0, 0)) given a percentage of the circumference (range [0, 1]).

    * Arguments:
        - *pt* [float]: Percentage of the circumference (centered at
          (0, 0) and starting from (x, y) = (r, 0)) to retrieve the
          (x, y) coordinates.
        - *r* [float]: Optional, ``1`` by default. The radius of the
          circumference.

    * Returns:
        - [tuple]: The (x, y) coordinates of the queried point [float].

    * Examples:
        >>> get_rel_pos_circ(0)
        (1, 0)
        >>> get_rel_pos_circ(0.25)
        (0, 1)
    '''

    a = pt * 2 * np.pi

    return r * np.cos(a), r * np.sin(a)


def bezier_quad(pa, pb, pc=[0, 0], res=100):  # NOTE: Move to top level?
    '''
    Creates a Bézier quadratic curve between two points.

    * Arguments:
        - *pa* [list]: Or any iterable of two [float] values. Point of
           origin of the curve in Cartesian coordinates.
        - *pb* [list]: Or any iterable of two [float] values. Ending
          point of the curve in Cartesian coordinates.
        - *pc* [list]: Optional ``[0, 0]`` by default. Can be any
          iterable of two [float] values. Control point for the curve in
          Cartesian coordinates.
        - *res* [float]: Optional, ``1e2`` by default. Resolution of the
          curve (e.g. number of points).

    * Returns:
        - [numpy.ndarray]: Array of size (2, *res* + 1). Contains the
          (x, y) coordinates of the curve points (defined by *res*).
    '''

    t = np.linspace(0, 1, int(res) + 1)

    pa = np.array(pa).reshape(2, 1)
    pb = np.array(pb).reshape(2, 1)
    pc = np.array(pc).reshape(2, 1)

    return ((1 - t) * ((1 - t) * pa + t * pc) + t * ((1 - t) * pc + t * pb))
