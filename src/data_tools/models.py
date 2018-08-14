# -*- coding: utf-8 -*-

'''
data_tools.models
=================

Model classes module.
'''

__all__ = ['DoseResponse', 'Lasso']

import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold as kf
from sklearn.model_selection import ShuffleSplit as ss
from sklearn.model_selection import StratifiedKFold as skf
from sklearn.model_selection import StratifiedShuffleSplit as sss


class DoseResponse(object):
    '''
    Wrapper class for ``scipy.optimize.least_squares`` to fit
    dose-response curves on a pre-defined Hill function with the
    following form:

    .. math::
       R=\\frac{mD^n}{k^n+D^n}

    Where :math:`D` is the dose, :math:`k`, :math:`m` and :math:`n` are
    the parameters to be fitted.

    * Arguments:
        - *d_data* [numpy.ndarray]: Or any iterable (1D). Contains the
          training data corresponding to the dose.
        - *r_data* [numpy.ndarray]: Or any iterable (1D). Contains the
          training data corresponding to the response.
        - *x0* [list]: Optional, ``[1, 1, 1]`` by default. Or any
          iterable of three elements. Contains the initial guess for the
          parameters. Parameters are considered to be in alphabetical
          order. This is, first element corresponds to :math:`k`, second
          is :math:`m` and last is :math:`n`.
        - *x_scale* [list]: Optional, ``[1, 1, 1]`` by default. Or any
          iterable of three elements. Scale of each parameter. May
          improve the fitting if the scaled parameters have similar
          effect on the cost function.
        - *bounds* [tuple]: Optional ``([0, 0, -inf], [inf, inf, inf])``
          by default. Two-element tuple containing the lower and upper
          boundaries for the parameters (elements of the tuple are
          iterables of three elements each).

    - Attributes:
        - *model* [scipy.optimize.OptimizeResult]: Contains the result
          of the optimized model. See `SciPy's reference <https://docs.\
          scipy.org/doc/scipy/reference/generated/scipy.optimize.Optimi\
          zeResult.html#scipy.optimize.OptimizeResult>`_ for more
          information.
        - *params* [numpy.ndarray]: Three-element array containing the
          fitted parameters :math:`k`, :math:`m` and :math:`n`.
    '''

    def __init__(self, d_data, r_data, x0=[1, 1, 1], x_scale=[1, 1, 1],
                 bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf])):

        def residuals(p, x, y):
            return self.__hill(x, *p) - y

        self.__xdata = d_data
        self.__ydata = r_data

        ftol = 1e-15
        max_nfev = 1e15
        diff_step = 1e-15

        self.model = least_squares(residuals, x0, x_scale=x_scale,
                                   args=(self.__xdata, self.__ydata),
                                   tr_solver='exact', bounds=bounds, ftol=ftol,
                                   diff_step=diff_step, max_nfev=max_nfev)

        self.params = self.model.x

    def __hill(self, x, k, m, n):

        return m * x ** n / (k ** n + x ** n)

    def ec(self, p=50):
        '''
        Computes the effective concentration for the specified
        percentage of maximal concentration (:math:`EC_{p}`).

        * Arguments:
            - *p* [int]: Optional, ``50`` by default (:math:`EC_{50}`).
              Defines the percentage of the maximal from which the
              effective concentration is to be computed.

        * Returns
            - [float]: Value of the :math:`EC_{p}` computed according
              to the model parameters.
        '''

        k, m, n = self.params

        return (p * k ** n / (p * m - p)) ** (1 / n)

    def plot(self, title=None, filename=None, figsize=None, legend=True):
        '''
        Plots the data points and the fitted function together.

        * Arguments:
            - *title* [str]: Optional, ``None`` by default. Defines the
              plot title.
            - *filename* [str]: Optional, ``None`` by default. If
              passed, indicates the file name or path where to store the
              figure. Format must be specified (e.g.: .png, .pdf, etc)
            - *figsize* [tuple]: Optional, ``None`` by default (default
              matplotlib size). Any iterable containing two values
              denoting the figure size (in inches) as [width, height].
            - *legend* [bool]: Optional, ``True`` by default. Indicates
              whether to show the plot legend or not.

        * Returns:
            - [matplotlib.figure.Figure]: Figure object showing the data
              points and the fitted model function.
        '''

        rng = np.linspace(min(self.__xdata), max(self.__xdata), 1000)

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(self.__xdata, self.__ydata, label='Data')
        ax.plot(rng, self.__hill(rng, *self.params), 'k', label='Fit')

        if title:
            ax.set_title(title)

        if legend:
            ax.legend(loc=0)

        fig.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig


class Lasso(LogisticRegressionCV):
    '''
    Wrapper class inheriting from
    ``sklearn.linear_model.LogisticRegressionCV`` with L1
    regularization.

    * Arguments:
        - *Cs* [int]: Optional, ``500`` by default. Integer or list of
          float values of regularization parameters to test. If an
          integer is passed, it will determine the number of values
          taken from a logarithmic scale between ``1e-4`` and ``1e4``.
          Note that the value of the parameter is defined as the inverse
          of the regularization strength.
        - *cv* [int]: Optional, ``10`` by default. Denotes the number of
          cross validation (CV) folds.
        - *sampler* [str]: Optional, ``'skf'`` by default. Determines
          which sampling method is used to generate the test and
          training sets for CV. Methods available are K-Fold (``'kf'``),
          Shuffle Split (``'ss'``) and their stratified variants
          (``'skf'`` and ``'sss'`` respectively).
        - *solver* [str]: Optional, ``'liblinear'`` by default.
          Determines which solver algorithm to use. Note that L1
          regularization can only be handled by ``'liblinear'`` and
          ``'saga'``. Additionally if the classification is multinomial,
          only the latter option is available.
        - *\*\*kwargs*: Optional. Any other keyword argument accepted by
          the ``sklearn.linear_model.LogisticRegressionCV`` class.

        Other keyword arguments and functions available from the parent
        class ``LogisticRegressionCV`` can be fount in `Scikit-Learn's
        reference <http://scikit-learn.org/stable/modules/generated/skl\
        earn.linear_model.LogisticRegressionCV.html>`_.
    '''

    def __init__(self, Cs=500, cv=10, sampler='skf', solver='liblinear',
                 **kwargs):

        super(self.__class__, self).__init__()

        self.penalty = 'l1'
        self.solver = solver
        self.Cs = Cs
        self.sampler = sampler
        self.cv_folds = cv

        if self.sampler == 'skf':
            self.cv = skf(n_splits=self.cv_folds)

        elif self.sampler == 'sss':
            self.cv = sss(n_splits=self.cv_folds)

        elif self.sampler == 'kf':
            self.cv = kf(n_splits=self.cv_folds)

        elif self.sampler == 'ss':
            self.cv = ss(n_splits=self.cv_folds)

        else:
            raise(Exception('Selected sampler is not a valid. Please choose '
                            '"skf" for stratified K-fold or "sss" for '
                            'stratified shuffle split. Also "sk" and "ss" for '
                            'the respective non-stratified methods.'))

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.x = None
        self.y = None

    def fit_data(self, x, y, silent=False):
        '''
        Fits the data to the logistic model.

        * Arguments:
            - *x* [pandas.DataFrame]: Contains the values/measurements
              [float] of the features (columns) for each
              sample/replicate (rows).
            - *y* [pandas.Series]: List or any iterable containing the
              observed class of each sample (must have the same order as
              in *x*).
            - *silent* [bool]: Optional, ``False`` by default.
              Determines whether messages are printed or not.
        '''

        self.x = x
        self.y = y

        start = time.time()

        if len(set(self.y)) > 2:
            self.multi_class = 'multinomial'

        # FIXME: Multinomial models may have problems here
        self.fit(self.x.values, self.y.values)
        self.key = self.scores_.keys()[0]

        self.accuracy = self.score(self.x, self.y)
        aux = pd.Series(self.coef_[0], index=self.x.columns)
        self.predictors = aux.iloc[aux.nonzero()].copy()
        self.predictors.sort_values(ascending=False, inplace=True)

        if not silent:
            print('Model trained, elapsed time %.3f s.' %(time.time() -
                                                          start))
            print('Number of predictors: %d.' %len(self.predictors))
            print('Accuracy = %.4f.' %self.accuracy)
            print('Optimum C = %.6f.\n' %self.C_[0])

    # TODO: add example figure
    def plot_score(self, filename=None, figsize=None):
        '''
        Plots the mean score across all folds obtained during CV.
        The optimum C parameter chosen and its score are highlighted.

        * Arguments:
            - *filename* [str]: Optional, ``None`` by default. If
              passed, indicates the file name or path where to store the
              figure. Format must be specified (e.g.: .png, .pdf, etc)
            - *figsize* [tuple]: Optional, ``None`` by default (default
              matplotlib size). Any iterable containing two values
              denoting the figure size (in inches) as [width, height].

        * Returns:
            - [matplotlib.figure.Figure]: Figure object containing the
              score plot.
        '''

        fig, ax = plt.subplots(figsize=figsize)

        mean = np.mean(self.scores_[self.key], axis=0)

        ax.plot(self.Cs_, mean, c='k')

        # Highlighting parameter choice
        arg = mean.argmax()
        x_, y_ = self.Cs_[arg], mean[arg]
        ax.scatter(x_, y_, c='r', s=75)
        ax.text(x_ * 1.01, y_ * -1.01, '(%.4f, %.4f)' %(x_, y_))

        ax.set_ylabel('Score'); ax.set_xlabel(r'$C$')
        ax.set_title(r'$C$ score (Mean over %dx CV)' %self.cv_folds)
        ax.set_xlim(self.Cs_[0], self.Cs_[-1])
        ax.set_xscale('log')

        fig.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig

    # TODO: add example figure
    def plot_coef(self, filename=None, figsize=None):
        '''
        Plots the non-zero coefficients for the fitted predictor
        features.

        * Arguments:
            - *filename* [str]: Optional, ``None`` by default. If
              passed, indicates the file name or path where to store the
              figure. Format must be specified (e.g.: .png, .pdf, etc)
            - *figsize* [tuple]: Optional, ``None`` by default (default
              matplotlib size). Any iterable containing two values
              denoting the figure size (in inches) as [width, height].

        * Returns:
            - [matplotlib.figure.Figure]: Figure object containing the
              bar plot of the non-zero coefficients.
        '''

        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(range(len(self.predictors)), self.predictors.values,
               color='k', align='center')

        ax.set_xticks(range(len(self.predictors)))
        ax.set_xticklabels(self.predictors.index, rotation='vertical',
                           va='top')
        ax.set_title('Non-zero coefficient values')
        ax.set_xlim(-1, len(self.predictors))

        fig.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig
