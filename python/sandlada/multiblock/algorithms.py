# -*- coding: utf-8 -*-
"""
The :mod:`sandlada.multiblock.Algorithms` module includes several projection
based latent variable algorithms.

These algorithms all have in common that they maximise a criteria on the form

    f(w_1, ..., w_n) = \sum_{i,j=1}^n c_{i,j} g(cov(X_iw_i, X_jw_j)),

with possibly very different constraints put on the weights w_i or on the
scores t_i = X_iw_i (e.g. unit 2-norm of weights, unit variance of scores,
L1/LASSO constraint on the weights etc.).

This includes methods such as PCA (f(p) = cov(Xp, Xp)),
PLS-R (f(w, c) = cov(Xw, Yc)), PLS-PM (the criteria above), RGCCA (the
criteria above), etc.

Created on Fri Feb  8 17:24:11 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

import abc
import warnings
import numpy as np
import prox_op
from util import *

__all__= ['NIPALS_PCA']

class BaseAlgorithm(object):
    """Baseclass for all multiblock (and single-block) algorithms.
    
    All algorithmsm, even the single-block algorithms, must return a list
    of 
    """

    __metaclass__ = abc.ABCMeta

    def __init__(prox_op = prox_op.ProxOp(),
                 max_iter = MAX_ITER, tolerance = TOLERANCE, **kwargs):

        self.prox_op   = prox_op
        self.max_iter  = max_iter
        self.tolerance = tolerance

    @abc.abstractmethod
    def run(self, X, *args, **kwargs):
        raise NotImplementedError('Abstract method "run" must be specialised!')


class NIPALS_PCA(BaseAlgorithm):

    def __init__(**kwargs):
        BaseAlgorithm.__init__(self, **kwargs)

    def run(self, X, *args, **kwargs):

        if isinstance(X, (tuple, list)):
            X = X[0]

        p = _start_vector(X, largest = True)
        XX = dot(X.T, X)

        iterations = 0
        while True:

            p_ = dot(XX, p)
            p = self.prox_op.prox(p)
            p_ = p_ / norm(p_)

            diff = p - p_
            p = p_
            if dot(diff.T, diff) < tolerance:
                break

            iterations += 1
            if iterations >= max_iter:
                warnings.warn('Maximum number of iterations reached '
                              'before convergence')
                break

        return [p]
