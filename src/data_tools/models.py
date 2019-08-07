# -*- coding: utf-8 -*-

'''
data_tools.models
=================

Model classes module.

Contents
--------
'''

from __future__ import print_function

__all__ = ['DoseResponse', 'Lasso', 'Linear', 'PowerLaw']

import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
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
        - *x0* [list]: Optional, ``None`` by default. Or any
          iterable of three elements. Contains the initial guess for the
          parameters. Parameters are considered to be in alphabetical
          order. This is, first element corresponds to :math:`k`, second
          is :math:`m` and last is :math:`n`. If ``None`` (default), the
          initial guess is inferred from *r_data*.
        - *x_scale* [list]: Optional, ``None`` by default. Or any
          iterable of three elements. Scale of each parameter. May
          improve the fitting if the scaled parameters have similar
          effect on the cost function. If ``None`` (default), the scale
          is inferred from *x0*.
        - *bounds* [tuple]: Optional ``([0, 0, -inf], [inf, inf, inf])``
          by default. Two-element tuple containing the lower and upper
          boundaries for the parameters (elements of the tuple are
          iterables of three elements each).

    - Attributes:
        - *x0* [list]: Contains the initial guess for the parameters.
          Parameters are considered to be in alphabetical order. This
          is, first element corresponds to :math:`k`, second is
          :math:`m` and last is :math:`n`.
        - *x_scale* [list]: Scale of each parameter.
        - *model* [scipy.optimize.OptimizeResult]: Contains the result
          of the optimized model. See `SciPy's reference <https://docs.\
          scipy.org/doc/scipy/reference/generated/scipy.optimize.Optimi\
          zeResult.html#scipy.optimize.OptimizeResult>`_ for more
          information.
        - *params* [numpy.ndarray]: Three-element array containing the
          fitted parameters :math:`k`, :math:`m` and :math:`n`.
    '''

    def __init__(self, d_data, r_data, x0=None, x_scale=None,
                 bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf])):

        def residuals(p, x, y):
            return self.__hill(x, *p) - y

        self.__xdata = d_data
        self.__ydata = r_data

        if not x0:
            half_y = ((max(self.__ydata) - min(self.__ydata)) / 2)
            k_inf = self.__xdata[np.argmin(abs(self.__ydata - half_y))]
            self.x0 = [k_inf, max(self.__ydata),
                       np.sign(self.__ydata[np.argmax(self.__xdata)]
                               - self.__ydata[np.argmin(self.__xdata)])]

        else:
            self.x0 = x0

        if not x_scale:
            self.x_scale = [10 ** int(np.log10(abs(i))) for i in self.x0]

        else:
            self.x_scale = x_scale

        ftol = 1e-15
        max_nfev = 1e15
        diff_step = 1e-15

        self.model = least_squares(residuals, self.x0, x_scale=self.x_scale,
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

        return ((p * k ** n) / (m * 100 - p)) ** (1 / n)

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


class Linear(object):
    '''
    Linear regression model using least squares. We define the model as
    :math:`y=mx+b`. The slope :math:`m` is computed by dividing the
    covariance of :math:`x` and :math:`y` over the variance of
    :math:`x`:

    .. math::
       m=\\frac{S_{xy}}{S_{xx}}

    Which are defined as follows:

    .. math::
       S_{xx}=\\sum x^2-\\frac{(\\sum x)^2}{n}\\\\
       S_{xy}=\\sum xy-\\frac{(\\sum x)(\\sum y)}{n}

    Where :math:`n` is the number of :math:`(x,y)` data points. The
    intercept is then obtained from:

    .. math::
       b=\\frac{\\sum y+a\\sum x}{n}

    * Arguments:
        - *x* [np.ndarray]: The independent variable to fit the linear
          model.
        - *y* [np.ndarray]: The dependent variable to fit the linear
          model.

    * Attributes:
        - *n* [int]: Number of data points provided.
        - *var* [float]: Variance of the independent variable.
        - *covar* [float]: Covariance between dependent and independent
          variables.
        - *slope* [float]: The slope of the linear model fitted to the
          provided data.
        - *intercept* [float]: The intercept of the fitted model.
    '''

    def __init__(self, x, y):
        assert len(x) == len(y), 'x and y must have the same length!'

        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = np.array(val)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = np.array(val)

    @property
    def n(self):
        return len(self.x)

    @n.setter
    def n(self, val):
        self.n = val

    @property
    def var(self):
        return np.square(self.x).sum() - np.square(self.x.sum()) / self.n

    @var.setter
    def var(self, val):
        self.var = val

    @property
    def covar(self):
        return (np.multiply(self.x, self.y).sum()
                - np.multiply(self.x.sum(), self.y.sum()) / self.n)

    @covar.setter
    def covar(self, val):
        self.covar = val

    @property
    def slope(self):
        return self.covar / self.var

    @slope.setter
    def slope(self, val):
        self.slope = val

    @property
    def intercept(self):
        return (self.y.sum() - self.slope * self.x.sum()) / self.n

    @intercept.setter
    def intercept(self, val):
        self.intercept = val


class PowerLaw(object):
    '''
    Fits a power law model to the provided data. Given :math:`y=ax^k`,
    the data is log-transformed and fitted to a linear model using
    least squares, since applaying log to both sides of the model:

    .. math::
       \\log(y)=k\\log(x)+\\log(a)

    This can be interpreted as a linear model of slope :math:`k` and
    intercept :math:`\\log(a)`.

    * Arguments:
        - *x* [np.ndarray]: The independent variable to fit the model.
        - *y* [np.ndarray]: The dependent variable to fit the model.

    * Attributes:
        - *lm* [data_tools.models.Linear]: The linear model of the data
          in log space.
        - *a* [float]: The constant of the power law distribution.
        - *k* [float]: The exponent of the power law distribution.
    '''

    def __init__(self, x, y):
        assert len(x) == len(y), 'x and y must have the same length!'

        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, val):
        self._x = np.array(val)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, val):
        self._y = np.array(val)

    @property
    def lm(self):
        return Linear(np.log10(self.x), np.log10(self.y))

    @lm.setter
    def lm(self, value):
        self.lm = val

    @property
    def a(self):
        return 10 ** self.lm.intercept

    @a.setter
    def a(self, val):
        self.a = val

    @property
    def k(self):
        return self.lm.slope

    @k.setter
    def k(self, val):
        self.k = val

    def plot(self, filename=None, figsize=None, grid=False):
        '''
        Plots the data and the fitted model in a log-log scale.

        * Arguments:
            - *filename* [str]: Optional, ``None`` by default. If
              passed, indicates the file name or path where to store the
              figure. Format must be specified (e.g.: .png, .pdf, etc)
            - *figsize* [tuple]: Optional, ``None`` by default (default
              matplotlib size). Any iterable containing two values
              denoting the figure size (in inches) as [width, height].
            - *grid* [bool]: Optional, ``False`` by default. Whether to
              show the plot grid lines.

        * Returns:
            - [matplotlib.figure.Figure]: Figure object containing the
              score plot.
        '''

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(self.x, self.y)
        ax.plot(self.x, self.a * self.x ** self.k, 'k--')

        ax.set_xscale('log')
        ax.set_yscale('log')

        if grid:
            ax.grid(which='both')

        fig.tight_layout()

        if filename:
            fig.savefig(filename)

        return fig
