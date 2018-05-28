# -*- coding: utf-8 -*-

'''
data_tools.Lasso
================

Class for logistic regression models with L1 regularization.
'''

__all__ = ['Lasso']

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold as kf
from sklearn.model_selection import ShuffleSplit as ss
from sklearn.model_selection import StratifiedKFold as skf
from sklearn.model_selection import StratifiedShuffleSplit as sss

class Lasso(LogisticRegressionCV):
    '''
    Statistical model developer. Uses LASSO Logistic regression with cross
    validation (CV) for penalty parameter "C". Inherits from
    sklearn.linear_model.LogisticRegressionCV.
    '''

    def __init__(self, Cs=np.logspace(-2, 0, 500), cv=10, sampler='skf',
                 cores=1):
        '''
        Initializes the Model instance. Keyword argument Cs specifies the
        inverse penalty parameters to fit using CV. It can be a [list] or
        [np.array] with explicit values or an [int] as the number of points in
        the log range [1e-4, 1e+4]. CV folds are specified by the keyword
        argument cv (10x by default) [int] (NOTE: algorithm uses OvR
        (one-versus-rest) to fit predictors and stratified K-fold for CV).
        If the data is multinomial for N categories, N -1 models are computed
        for each category versus the control samples (healthy patients).
        '''

        super(self.__class__, self).__init__()

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

        self.penalty = 'l1'
        self.solver = 'saga'
        self.n_jobs = cores

    def fit_data(self, x, y, silent=False):
        '''
        Fits the passed data, assumed to follow pandas data structures
        [pandas.DataFrame] or [pandas.Series]. Where x is the normalized MS
        intensity [float] and y the disease status category [int].
        '''

        self.X = x
        self.Y = y

        start = time.time()

        if len(set(self.Y)) > 2:
            self.multi_class = 'multinomial'

#        else:
#            self.multi_class = 'ovr'

#        if len(np.unique(self.Y))>2:
            # Will perform individual models for each category vs. control
#            self.multiclass = True

#            control = 1 # Define the control group category
#            c_X, c_Y = self.X[self.Y==control], self.Y[self.Y==control]

#            self.models = dict()
#            self.keys = []

#            for k in np.unique(self.Y[self.Y!=control]):
#                if len(self.Y[self.Y==k])<self.cv_folds:
#                    if not silent:
#                        print 'Group %d discarded due to few samples.\n' %k

#                    continue

#                if not silent: print 'Computing model for group %d:' %k

#                x = pd.concat([c_X, self.X[self.Y==k]])
#                y = pd.concat([c_Y, self.Y[self.Y==k]])
#                self.keys.append(k)
#                self.models[k] = Model(Cs=self.Cs, cv=self.cv_folds,
#                                       sampler=self.sampler)
#                self.models[k].fit_data(x, y)

#            if not silent: print 'Total time %.3f s.' %(time.time() - start)

#        else:
#            self.multiclass = False

        self.fit(self.X, self.Y)
        self.key = self.scores_.keys()[0]

        self.accuracy = self.score(self.X, self.Y)
        aux = pd.Series(self.coef_[0], index=self.X.columns)
        self.predictors = aux.iloc[aux.nonzero()].copy()
        self.predictors.sort_values(ascending=False, inplace=True)

        if not silent:
            print 'Model trained, elapsed time %.3f s.' %(time.time() -
                                                          start)
            print 'Number of predictors: %d.' %len(self.predictors)
            print 'Accuracy = %.4f.' %self.accuracy
            print 'Optimum C = %.6f.\n' %self.C_[0]

    def compute_second_line(self):
        '''
        Method that computes second-line predictors. This is, initially
        selected predictors' data is dropped out, and another model is trained
        with the remaining data.
        '''

#        if self.multiclass:
#            for k in self.keys:
#                print 'Computing second-line predictors for group %d:' %k
#                self.models[k].compute_second_line()

#        else:
        dropX = self.X.drop(self.predictors.index, axis=1)
        self.second_line = Model(Cs=self.Cs, cv=self.cv_folds,
                                 sampler=self.sampler)
        self.second_line.fit_data(dropX, self.Y)
        print 'Second-line predictors computed.\n'

    def plot_score(self):
        '''
        Plots the mean score across all folds obtained during CV. Highlights
        the optimum C parameter chosen (NOTE: Since C represents the inverse
        regularization parameter, it's chosen so that maximizes the score).
        '''

#        if self.multiclass:
#            for k in self.keys:
#                print 'C scores of group %d:' %k
#                self.models[k].plot_score()
#        else:
        mean = np.mean(self.scores_[self.key], axis=0)

        fig = plt.figure(); ax = fig.gca()

        ax.plot(self.Cs_, mean, c='k')
        ax.set_ylabel('Score'); ax.set_xlabel(r'$C$')
        ax.set_title(r'$C$ score (Mean over %dx CV)' %self.cv_folds)
        ax.set_xlim(self.Cs_[0], self.Cs_[-1])
        ax.set_xscale('log')

        arg = mean.argmax()
        x_, y_ = self.Cs_[arg], mean[arg]
        ax.scatter(x_, y_, c='r', s=75)
        ax.text(x_ * 1.01, y_ * 1.01, '(%.4f, %.4f)' %(x_, y_))
        fig.tight_layout()

    def plot_coef(self):
        '''
        Plots the non-zero coefficients for the fitted predictors.
        '''

#        if self.multiclass:
#            for k in self.keys:
#                print 'Predictors for group %d:' %k
#                self.models[k].plot_coef()

#        else:
        fig = plt.figure(); ax = fig.gca()

        ax.bar(range(len(self.predictors)), self.predictors.values,
               color='k', align='center')
        ax.set_xticks(range(len(self.predictors)))
        ax.set_xticklabels(self.predictors.index, rotation='vertical',
                           va='top')
        ax.set_title('Non-zero coefficient values')
        ax.set_xlim(-1, len(self.predictors))
        fig.tight_layout()

    def plot_sample_probs(self, x=None, y=None):
        '''

        '''

        if x is None: x = self.X

        if y is None: y = self.Y

        probs = pd.DataFrame(self.predict_proba(x), index=x.index)
        pred = pd.Series(self.predict(x), index=x.index, name='Pred.')
        res = pd.concat([probs[1], pred, y], axis=1)
        res.sort_values(by=1, inplace=True)

        fig = plt.figure(figsize=(18, 5)); ax = fig.gca()

        ax.scatter(range(len(res)), res.loc[:, 1], label='Pred. prob.',
                   marker='.')
        ax.scatter(range(len(res)), res.loc[:, 'Pred.'], label='Pred. stage',
                   marker='s', s=18)
        ax.scatter(range(len(res)), res.loc[:, 'CKD stage'], label='CKD stage',
                   marker='x', s=15)
        ax.plot(range(len(res)), len(res) * [.5], '--k', lw=.5)

        plt.xticks(range(len(res)), res.index, rotation=90, fontsize=8)
        ax.set_title('Prediction probabilities')
        ax.set_xlabel('Patient ID')
        ax.set_xlim(-1, len(res))
        ax.legend(loc=0)

        fig.tight_layout()
