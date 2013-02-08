# -*- coding: utf-8 -*-
"""
The :mod:`sklearn.NIPALS` module includes several different projection based
latent variable methods that all are computed using the NIPALS algorithm.
"""

# Author: Tommy Löfstedt <tommy.loefstedt@cea.fr>
# License: BSD Style.

__all__ = ['PCAfast','PCA', 'SVD', 'PLSR', 'PLSC', 'center', 'scale', 'direct']

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_arrays

import abc
import warnings
import numpy as np
#from scipy import linalg
from numpy.linalg import norm
from numpy import dot

# Settings
_MAXITER    = 500
_TOLERANCE  = 5e-7

# NIPALS mode
_NEWA       = "NewA"
_A          = "A"
_B          = "B"

# Inner weighting schemes
_HORST      = "Horst"
_CENTROID   = "Centroid"
_FACTORIAL  = "Factorial"

# Available algorithms
_RGCCA      = "RGCCA"
_NIPALS     = "NIPALS"


# TODO: Make these algorithms nicely OO!
def _NIPALS(X, C, mode, scheme, not_normed = [], soft_threshold = 0,
            max_iter = _MAXITER, tolerance = _TOLERANCE, **kwargs):
    """Inner loop of the NIPALS algorithm.

    Performs the NIPALS algorithm on the supplied tuple or list of numpy arrays
    in X. This method applies for 1, 2 or more blocks.

    One block would result in e.g. PCA; two blocks would result in e.g. SVD or
    CCA; and multiblock (n > 2) would be for instance GCCA, PLS-PM or MAXDIFF.

    This function uses Wold's procedure (based on Gauss-Siedel iteration) for
    fast convergence.

    Parameters
    ----------
    X          : A tuple or list with n numpy arrays of shape [M, N_i],
                 i=1,...,n. These are the training set.

    C          : Adjacency matrix that is a numpy array of shape [n, n]. If an
                 element in position C[i,j] is 1, then block i and j are
                 connected, and 0 otherwise. If C is None, then all matrices
                 are assumed to be connected.

    mode       : A tuple or list with n elements with the mode to use for a
                 matrix. The mode is represented by a string: "A", "B" or
                 "NewA". If mode = None, then "NewA" is used.

    scheme     : The inner weighting scheme to use in the algorithm. The scheme
                 may be "Horst", "Centroid" or "Factorial". If scheme = None,
                 then Horst's schme is used (the inner weighting scheme is the
                 identity).

    max_iter   : The number of iteration before the algorithm is forced to
                 stop. The default number of iterations is 500.

    tolerance  : The level below which we treat numbers as zero. This is used
                 as stop criterion in the algorithm. Smaller value will give
                 more acurate results, but will take longer time to compute.
                 The default tolerance is 5E-07.

    not_normed : In some algorithms, e.g. PLS regression, the weights or
                 loadings of some matrices (e.g. Y) are not normalised. This
                 tuple or list contains the indices in X of those matrices
                 that should not be normalised. Thus, for PLS regression, this
                 argument would be not_normed = (2,). If not_normed = None,
                 then all matrices are subject to normalisation of either
                 weights or scores depending on the modes used.

    soft_threshold : A tuple or list of the soft threshold level to use for
                     each block.

    Returns
    -------
    W          : A list with n numpy arrays of weights of shape [N_i, 1].
    """

    n = len(X)

    mode           = _make_list(mode, n, _NEWA)
    scheme         = _make_list(scheme, n, _HORST)
    soft_threshold = _make_list(soft_threshold, n, 0)

    W = []
    for Xi in X:
        w = _start_vector(Xi, largest = True)
#        w = np.random.rand(Xi.shape[1],1)
#        w /= norm(w)
        W.append(w)

    # Main NIPALS loop
    iterations = 0
    while True:
        converged = True
        for i in range(n):
            Xi = X[i]
            ti = dot(Xi, W[i])
            ui = np.zeros(ti.shape)
            for j in range(n):
                Xj = X[j]
                wj = W[j]
                tj = dot(Xj, wj)

                # Determine weighting scheme and compute weight
                if scheme[i] == _HORST:
                    eij = 1
                elif scheme[i] == _CENTROID:
                    eij = _sign(_corr(ti, tj))
                elif scheme[i] == _FACTORIAL:
                    eij = _corr(ti, tj)

                # Internal estimation usin connected matrices' score vectors
                if C[i,j] != 0 or C[j,i] != 0:
                    ui += eij*tj

            # External estimation
            if mode[i] == _NEWA or mode[i] == _A:
                # TODO: Ok with division here?
                wi = dot(Xi.T, ui) / dot(ui.T, ui)
            elif mode[i] == _B:
                wi = dot(np.pinv(Xi), ui) # TODO: Precompute to speed up!

            # Apply soft thresholding if greater-than-zero value supplied
            if soft_threshold[i] > 0:
                wi = _soft_threshold(wi, soft_threshold[i], copy = False)

            # Normalise weight vectors according to their weighting scheme
            if mode[i] == _NEWA and not i in not_normed:
                # Normalise weight vector wi to unit variance
                wi /= norm(wi)
            elif (mode[i] == _A or mode[i] == _B) and not i in not_normed:
                # Normalise score vector ti to unit variance
                wi /= norm(dot(Xi, wi))
                wi *= np.sqrt(wi.shape[0])

            # Check convergence for each weight vector. They all have to leave
            # converged = True in order for the algorithm to stop.
            diff = wi - W[i]
            if dot(diff.T, diff) > tolerance:
                converged = False

            # Save updated weight vector
            W[i] = wi

        if converged:
            break

        if iterations >= max_iter:
            warnings.warn('Maximum number of iterations reached '
                          'before convergence')
            break

        iterations += 1

    return W


# TODO: Make a private method of BasePLS!
def _RGCCA(X, C, tau, scheme, not_normed = [],
           max_iter = _MAXITER, tolerance = _TOLERANCE):
    """Inner loop of the RGCCA algorithm.

    Performs the RGCCA algorithm on the supplied tuple or list of numpy arrays
    in X. This method applies for 1, 2 or more blocks.

    One block would result in e.g. PCA; two blocks would result in e.g. SVD or
    CCA; and multiblock (n > 2) would be for instance SUMCOR, SSQCOR or SUMCOV.

    Parameters
    ----------
    X          : A tuple or list with n numpy arrays of shape [M, N_i],
                 i=1,...,n. These are the training set.

    C          : Adjacency matrix that is a numpy array of shape [n, n]. If an
                 element in position C[i,j] is 1, then block i and j are
                 connected, and 0 otherwise. If C is None, then all matrices
                 are assumed to be connected such that C has ones everywhere
                 except for on the diagonal.:

    tau        : A tuple or list with n shrinkage constants tau[i]. If tau is a
                 single real, all matrices will use this value.

    scheme     : The inner weighting scheme to use in the algorithm. The scheme
                 may be "Horst", "Centroid" or "Factorial". If scheme = None,
                 then Horst's scheme is used (where the inner weighting scheme
                 is the identity).

    max_iter   : The number of iteration before the algorithm is forced to
                 stop. The default number of iterations is 500.

    tolerance  : The level below which we treat numbers as zero. This is used
                 as stop criterion in the algorithm. Smaller value will give
                 more acurate results, but will take longer time to compute.
                 The default tolerance is 5E-07.

    not_normed : In some algorithms, e.g. PLS regression, the weights or
                 loadings of some matrices (e.g. Y) are not normalised. This
                 tuple or list contains the indices in X of those matrices
                 that should not be normalised. Thus, for PLS regression, this
                 argument would be not_normed = (2,). If not_normed = None,
                 then all matrices are subject to normalisation of either
                 weights or scores depending on the modes used.

    Returns
    -------
    W          : A list with n numpy arrays of weights of shape [N_i, 1].
    """

    n = len(X)

    invIXX = []
    W      = []
    for i in range(n):
        Xi = X[i]
        XX = dot(Xi.T, Xi)
        I  = np.eye(XX.shape[0])

        w  = _start_vector(Xi, largest = True)
#        w  = np.random.rand(Xi.shape[1],1)
#        w /= norm(w)

        invIXX.append(np.linalg.pinv(tau[i]*I + ((1-tau[i])/w.shape[0])*XX))
        invIXXw  = dot(invIXX[i], w)
        winvIXXw = dot(w.T, invIXXw)
        w        = invIXXw/np.sqrt(winvIXXw)

        W.append(w)

    # Main RGCCA loop
    iterations = 0
#    h = []
    while True:

#        h_ = 0
#        for i in range(n):
#            Xi = X[i]
#            ti = dot(Xi, W[i])
#            for j in range(n):
#                tj = dot(X[j], W[j])
#
#                c = _cov(ti, tj)
#
#                # Determine weighting scheme and compute weight
#                if scheme == _HORST:
#                    pass
#                elif scheme == _CENTROID:
#                    c = np.abs(c)
#                elif scheme == _FACTORIAL:
#                    c = c*c
#
#                h_ += C[i,j]*c
#        h.append(h_)

        converged = True
        for i in range(n):
            Xi = X[i]
            ti = dot(Xi, W[i])
            ui = np.zeros(ti.shape)
            for j in range(n):
                tj = dot(X[j], W[j])

                # Determine weighting scheme and compute weight
                if scheme == _HORST:
                    eij = 1
                elif scheme == _CENTROID:
                    eij = _sign(_cov(ti, tj))
                elif scheme == _FACTORIAL:
                    eij = _cov(ti, tj)

                # Internal estimation using connected matrices' score vectors
                if C[i,j] != 0 or C[j,i] != 0:
                    ui += eij*tj

            # Outer estimation for block i
            wi = dot(Xi.T, ui)
            invIXXw  = dot(invIXX[i], wi)
            # Should we normalise?
            if not i in not_normed:
                winvIXXw = dot(wi.T, invIXXw)
                wi        = invIXXw / np.sqrt(winvIXXw)
            else:
                wi        = invIXXw

            # Check convergence for each weight vector. They all have to leave
            # converged = True in order for the algorithm to stop.
            diff = wi - W[i]
            if dot(diff.T, diff) > tolerance:
                converged = False

            # Save updated weight vector
            W[i] = wi

        if converged:
            break

        if iterations >= max_iter:
            warnings.warn('Maximum number of iterations reached '
                          'before convergence')
            break

        iterations += 1

    return W


def _soft_threshold(w, l, copy = True):
    worig = w.copy()
    lorig = l

    warn = False
    while True:
        sign = np.sign(worig)
        if copy:
            w = np.absolute(worig) - l
        else:
            np.absolute(worig, w)
            w -= l
            w[w < 0] = 0
        w = np.multiply(sign,w)

        if np.linalg.norm(w) > _TOLERANCE:
            break
        else:
            warn = True
            # TODO: Can this be improved?
            l *= 0.9 # Reduce by 10 % until at least one variable is significant

    if warn:
        warnings.warn('Soft threshold was too large (all variables purged).'\
                ' Threshold reset to %f (was %f)' % (l, lorig))
    return w


def _make_list(a, n, default = None):
    # If a list, but empty
    if isinstance(a, (tuple, list)) and len(a) == 0:
        a = None
    # If only one value supplied, create a list with that value
    if a != None:
        if not isinstance(a, (tuple, list)):
            a = [a for i in xrange(n)]
    else: # None or empty list supplied, create a list with the default value
        a = [default for i in xrange(n)]
    return a


def _start_vector(X, random = True, ones = False, largest = False):
    if largest: # Using row with largest sum of squares
        idx = np.argmax(np.sum(X**2, axis=1))
        w = X[[idx],:].T
    elif ones:
        w = np.ones((X.shape[1],1))
    else: # random start vector
        w = np.random.rand(X.shape[1],1)

    w /= norm(w)
    return w


def _sign(v):
    if v < 0:
        return -1
    else:
        return 1


def _corr(a,b):
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    norma = norm(a_)
    normb = norm(b_)

    if norma < _TOLERANCE or normb < _TOLERANCE:
        return 0

    ip = dot(a_.T, b_)
    return ip / (norma * normb)


def _cov(a,b):
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    ip = dot(a_.T, b_)

    return ip[0,0] / (a_.shape[0] - 1)


def center(X, return_means = False, copy = True):
    """ Centers the numpy array(s) in X

    Arguments
    ---------
    X            : The matrices to center
    return_means : Whether or not the computed means are to be computed as well
    copy         : Whether or not to return a copy, or center in-place

    Returns
    -------
        Centered X, means
    """

    is_list = True
    if not isinstance(X, (tuple, list)):
        X       = [X]
        is_list = False

    means = []
    for i in xrange(len(X)):
        X[i] = np.asarray(X[i])
        mean = X[i].mean(axis = 0)
        if copy:
            X[i] = X[i] - mean
        else:
            X[i] -= mean
        means.append(mean)

    if not is_list:
        X     = X[0]
        means = means[0]

    if return_means:
        return X, means
    else:
        return X


def scale(X, centered = True, return_stds = False, copy = True):
    """ Scales the numpy arrays in arrays to standard deviation 1
    Returns
    -------
        Scaled arrays, stds
    """

    is_list = True
    if not isinstance(X, (tuple, list)):
        X       = [X]
        is_list = False

    stds = []
    for i in xrange(len(X)):
        if centered == True:
            ddof = 1
        else:
            ddof = 0
        std = X[i].std(axis = 0, ddof = ddof)
        std[std < _TOLERANCE] = 1.0
        if copy:
            X[i] = X[i] / std
        else:
            X[i] /= std
        stds.append(std)

    if not is_list:
        X    = X[0]
        stds = stds[0]

    if return_stds:
        return X, stds
    else:
        return X


def direct(W, T = None, P = None, compare = False):
    if compare and T == None:
        raise ValueError("In order to compare you need to supply two arrays")

    for j in xrange(W.shape[1]):
        w = W[:,[j]]
        if compare:
            t = T[:,[j]]
            cov = dot(w.T, t)
            if P != None:
                p = P[:,[j]]
                cov2 = dot(w.T, p)
        else:
            cov = dot(w.T, np.ones(w.shape))
        if cov < 0:
            if not compare:
                w *= -1
                if T != None:
                    t = T[:,[j]]
                    t *= -1
                    T[:,j] = t.ravel()
                if P != None:
                    p = P[:,[j]]
                    p *= -1
                    P[:,j] = p.ravel()
            else:
                t = T[:,[j]]
                t *= -1
                T[:,j] = t.ravel()

            W[:,j] = w.ravel()

        if compare and P != None and cov2 < 0:
            p = P[:,[j]]
            p *= -1
            P[:,j] = p.ravel()

    if T != None and P != None:
        return W, T, P
    elif T != None and P == None:
        return W, T
    elif T == None and P != None:
        return W, P
    else:
        return W


class BasePLS(BaseEstimator, TransformerMixin):
    __metaclass__ = abc.ABCMeta

    def __init__(self, C = None, num_comp = 2, tau = None,
                 center = True, scale = True, mode = None, scheme = None,
                 not_normed = None, copy = True, normalise_directions = False,
                 max_iter = _MAXITER, tolerance = _TOLERANCE,
                 soft_threshold = 0):

        # Supplied by the user
        self.C              = C
        self.num_comp       = num_comp
        self.tau            = tau
        self.center         = center
        self.scale          = scale
        self.mode           = mode
        self.scheme         = scheme
        self.not_normed     = not_normed
        self.copy           = copy
        self.max_iter       = max_iter
        self.tolerance      = tolerance
        self.normal_dir     = normalise_directions
        self.soft_threshold = soft_threshold


    @abc.abstractmethod
    def _get_transform(self, index = 0):
        raise NotImplementedError('Abstract method "_get_transform" must be specialised!')


    def _algorithm(self, *args, **kwargs):
        return _NIPALS(*args, **kwargs)


    def _deflate(self, X, w, t, p, index = None):
        return X - dot(t, p.T) # Default is deflation with loadings p


    def _check_inputs(self, X):

        if self.n < 1:
            raise ValueError('At least one matrix must be given')

        err_msg = 'The mode must be either "%s", "%s" or "%s"' \
                    % (_NEWA, _A, _B)
        if self.mode == None:
            self.mode = [_NEWA]*self.n # Default mode is New A
        if not isinstance(self.mode, (tuple, list)):
            if not self.mode in (_NEWA, _A, _B):
                raise ValueError(err_msg)
            self.mode = [self.mode]*self.n # If only one mode is given, all matrices gets this mode
        for m in self.mode:
            if not m in (_NEWA, _A, _B):
                raise ValueError(err_msg)

        err_msg = 'The scheme must be either "%s", "%s" or "%s"' \
                    % (_HORST, _CENTROID, _FACTORIAL)
        if self.scheme == None:
            self.scheme = [_HORST]*self.n # Default scheme is Horst
        if not isinstance(self.scheme, (tuple, list)):
            if not self.scheme in (_HORST, _CENTROID, _FACTORIAL):
                raise ValueError(err_msg)
            self.scheme = [self.scheme]*self.n
        for s in self.scheme:
            if not s in (_HORST, _CENTROID, _FACTORIAL):
                raise ValueError(err_msg)

        # TODO: Add Schäfer and Strimmer's method here if tau == None!
        err_msg = 'The shrinking factor tau must be of type float or a ' \
                  'list or tuple with floats'
        if self.tau == None:
            self.tau = [0.5]*self.n
        if not isinstance(self.tau, (tuple, list)):
            if not isinstance(self.tau, (float, int)):
                raise ValueError(err_msg)
            self.tau = [float(self.tau)]*self.n
        for t in self.tau:
            if not isinstance(t, (float,int)):
                raise ValueError(err_msg)

        err_msg = 'The argument center must be of type bool or a ' \
                  'list or tuple with bools'
        if self.center == None:
            self.center = [True]*self.n
        if not isinstance(self.center, (tuple, list)):
            if not isinstance(self.center, (bool,)):
                raise ValueError(err_msg)
            self.center = [self.center]*self.n
        for c in self.center:
            if not isinstance(c, (bool,)):
                raise ValueError(err_msg)

        err_msg = 'The argument scale must be of type bool or a ' \
                  'list or tuple with bools'
        if self.scale == None:
            self.scale = [True]*self.n
        if not isinstance(self.scale, (tuple, list)):
            if not isinstance(self.scale, (bool,)):
                raise ValueError(err_msg)
            self.scale = [self.scale]*self.n
        for s in self.scale:
            if not isinstance(s, (bool,)):
                raise ValueError(err_msg)

        if self.not_normed == None:
            self.not_normed = ()

        # Number of rows
        try:
            M = X[0].shape[0]
        except:
            print "Here!"
            print X
        minN = float('Inf')

        for i in xrange(self.n):
            if X[i].ndim == 1:
                X[i] = X[i].reshape((X[i].size, 1))
            if X[i].ndim != 2:
                raise ValueError('The matrices in X must be 1- or 2D arrays')

            if X[i].shape[0] != M:
                raise ValueError('Incompatible shapes: X[%d] has %d samples, '
                                 'while X[%d] has %d' % (0,M, i,X[i].shape[0]))

            minN = min(minN, X[i].shape[1])

        if self.num_comp < 1:
            raise ValueError('Invalid number of components')
        if self.num_comp > minN:
            warnings.warn('Too many components! No more than %d can be '
                          'computed' % (minN,))
            self.num_comp = minN

        if self.C == None and self.n == 1:
            self.C = np.ones((1,1))
        elif self.C == None and self.n > 1:
            self.C = np.ones((self.n,self.n)) - np.eye(self.n)


    def _preprocess(self, X):
        self.means = []
        self.stds  = []
        for i in xrange(self.n):
            if self.center[i]:
                X[i], means = center(X[i], return_means = True, copy = self.copy)
            else:
                means = np.zeros((1, X[i].shape[1]))
            self.means.append(means)

            if self.scale[i]:
                X[i], stds = scale(X[i], centered=self.center[i], return_stds = True, copy = self.copy)
            else:
                stds = np.ones((1, X[i].shape[1]))
            self.stds.append(stds)

        return X


    def fit(self, *X):
        # Copy since this will contain the residual (deflated) matrices
        X = check_arrays(*X, dtype = np.float, copy = self.copy,
                         sparse_format = 'dense')
        # Number of matrices
        self.n = len(X)

        self._check_inputs(X)
        X = self._preprocess(X)

        # Results matrices
        self.W  = []
        self.T  = []
        self.P  = []
        self.Ws = []
        for i in xrange(self.n):
            M, N = X[i].shape
            w  = np.zeros((N, self.num_comp))
            t  = np.zeros((M, self.num_comp))
            p  = np.zeros((N, self.num_comp))
            ws = np.zeros((N, self.num_comp))
            self.W.append(w)
            self.T.append(t)
            self.P.append(p)
            self.Ws.append(ws)

        # Outer loop, over components
        for a in xrange(self.num_comp):
            # Inner loop, weight estimation
            w = self._algorithm(X = X,
                                C = self.C,
                                tau = self.tau,
                                mode = self.mode,
                                scheme = self.scheme,
                                max_iter = self.max_iter,
                                tolerance = self.tolerance,
                                not_normed = self.not_normed,
                                soft_threshold = self.soft_threshold)

            # Compute scores and loadings
            for i in xrange(self.n):

                # If we should make all weights correlate with np.ones((N,1))
                if self.normal_dir:
                    w[i] = direct(w[i])

                # Score vector
                t  = dot(X[i], w[i]) / dot(w[i].T, w[i])

                # Test for null variance
                if dot(t.T, t) < self.tolerance:
                    warnings.warn('Scores of block X[%d] are too small at '
                                  'iteration %d' % (i, a))

                # Loading vector
                p = dot(X[i].T, t) / dot(t.T, t)

                self.W[i][:,a] = w[i].ravel()
                self.T[i][:,a] = t.ravel()
                self.P[i][:,a] = p.ravel()

                # Generic deflation method. Overload for specific deflation!
                X[i] = self._deflate(X[i], w[i], t, p, i)

        # Compute W*, the rotation from input space X to transformed space T
        # such that T = XW(P'W)^-1 = XW*
        for i in xrange(self.n):
            self.Ws[i] = dot(self.W[i], np.linalg.inv(dot(self.P[i].T,self.W[i])))

        return self


    def transform(self, *X, **kwargs):

        copy = kwargs.get('copy', True)

        n = len(X)
        if n > self.n:
            raise ValueError('Model was trained for %d matrices', self.n)

        T  = []
        for i in xrange(n):
            X_ = np.asarray(X[i])
            # TODO: Use center() and scale() instead! (Everywhere!)
            # Center and scale
            if copy:
                X_ = (X_ - self.means[i]) / self.stds[i]
            else:
                X_ -= self.means[i]
                X_ /= self.stds[i]

            # Apply rotation
            t = dot(X_, self._get_transform(i))

            T.append(t)

        return T


    def fit_transform(self, *X, **fit_params):
        return self.fit(*X, **fit_params).transform(*X)


class PCA(BasePLS):

    def __init__(self, **kwargs):
        BasePLS.__init__(self, C = np.ones((1,1)), **kwargs)

    def _get_transform(self, index = 0):
        return self.P

    def fit(self, *X, **kwargs):
        BasePLS.fit(self, X[0])
        self.T = self.T[0]
        self.P = self.W[0]
        del self.W

        return self

    def transform(self, *X, **kwargs):
        T = BasePLS.transform(self, X[0], **kwargs)
        return T[0]

    def fit_transform(self, *X, **fit_params):
        return self.fit(X[0], **fit_params).transform(X[0])


class PCAfast(PCA):

    def __init__(self, **kwargs):
        PCA.__init__(self, **kwargs)

    def _algorithm(self, *args, **kwargs):
        return _NIPALS_PCA(*args, **kwargs)


class SVD(PCA):

    def __init__(self, **kwargs):
        PCA.__init__(self, **kwargs)

    def _get_transform(self, index = 0):
        return self.V

    def fit(self, *X, **kwargs):
        PCA.fit(self, X[0])
        self.U = self.T
        # Move norms of U to the diagonal matrix S
        norms = np.sum(self.U**2,axis=0)**(0.5)
        self.U /= norms
        self.S = np.diag(norms)
        self.V = self.P
        del self.T
        del self.P

        return self


class PLSR(BasePLS, RegressorMixin):

    def __init__(self, **kwargs):
        BasePLS.__init__(self, mode = _NEWA, scheme = _HORST, not_normed = [1],
                         **kwargs)

    def _get_transform(self, index = 0):
        if index < 0 or index > 1:
            raise ValueError("Index must be 0 or 1")
        if index == 0:
            return self.Ws
        else:
            return self.C

    def _deflate(self, X, w, t, p, index = None):
        if index == 0:
            return X - dot(t, p.T) # Deflate X using its loadings
        else:
            return X # Do not deflate Y

    def fit(self, X, Y = None, **kwargs):
        Y = kwargs.get('y', Y)
        if Y == None:
            raise ValueError('Y is not supplied')
        BasePLS.fit(self, X, Y)
        self.C  = self.W[1]
        self.U  = self.T[1]
        self.Q  = self.P[1]
        self.W  = self.W[0]
        self.T  = self.T[0]
        self.P  = self.P[0]
        self.Ws = self.Ws[0]

        self.B = dot(self.Ws, self.C.T)

        return self

    def predict(self, X, copy = True):
        X = np.asarray(X)
        if copy:
            X = (X - self.means[0]) / self.stds[0]
        else:
            X -= self.means[0]
            X /= self.stds[0]

        if hasattr(self, 'B'):
            Ypred = dot(X, self.B)
        else:
            Ypred = dot(X, self.Bx)

        return (Ypred*self.stds[1]) + self.means[1]

    def transform(self, X, Y = None, **kwargs):
        Y = kwargs.get('y', Y)
        if Y != None:
            T = BasePLS.transform(self, X, Y, **kwargs)
        else:
            T = BasePLS.transform(self, X, **kwargs)
            T = T[0]
        return T

    def fit_transform(self, X, Y = None, **kwargs):
        Y = kwargs.get('y', Y)
        return self.fit(X, Y, **kwargs).transform(X, Y)


class PLSC(PLSR):

    def __init__(self, **kwargs):
        BasePLS.__init__(self, mode = _NEWA, scheme = _HORST, **kwargs)

    def _get_transform(self, index = 0):
        if index < 0 or index > 1:
            raise ValueError("Index must be 0 or 1")
        if index == 0:
            return self.Ws
        else: # index == 1
            return self.Cs

    def _deflate(self, X, w, t, p, index = None):
        return X - dot(t, p.T) # Deflate X using its loadings

    def fit(self, X, Y = None, **kwargs):
        Y = kwargs.get('y', Y)
        PLSR.fit(self, X, Y)
        self.Cs = dot(self.C, np.linalg.inv(dot(self.Q.T,self.C)))

        self.Bx = self.B
        self.By = dot(self.Cs, self.W.T)
        del self.B

        return self

    def predict(self, X, Y = None, copy = True):

        Ypred = PLSR.predict(self, X, copy = copy)

        if Y != None:
            Y = np.asarray(Y)
            if copy:
                Y = (Y - self.means[1]) / self.stds[1]
            else:
                Y -= self.means[1]
                Y /= self.stds[1]
    
            Xpred = (dot(Y, self.By)*self.stds[0]) + self.means[0]

            return Ypred, Xpred

        return Ypred


#class CCA(BasePLS):
#
    #def __init__(self, **kwargs):
        #BasePLS.__init__(self, mode=(_B, _B), scheme=(_FACTORIAL,_FACTORIAL),
                         #tau = (0, 0), **kwargs)
#
#
    #def _get_transform(self, index = 0):
        #if index == 0:
            #return self.Ws
        #elif index == 1:
            #return self.Cs
        #else:
            #raise ValueError("Index must be 0 or 1")
#
#
    #def _algorithm(self, **kwargs):
        #return _RGCCA(**kwargs)
#
#
    #def fit(self, *X, **kwargs):
        #BasePLS.fit(self, *X)
        #self.C  = self.W[1]
        #self.U  = self.T[1]
        #self.Q  = self.P[1]
        #self.Cs = self.Ws[1]
        #self.W  = self.W[0]
        #self.T  = self.T[0]
        #self.P  = self.P[0]
        #self.Ws = self.Ws[0]
#
        #return self
#
#
    #def transform(self, X, Y = None, **kwargs):
        #if Y != None:
            #T = BasePLS.transform(self, X, Y, **kwargs)
        #else:
            #T = BasePLS.transform(self, X, **kwargs)
            #T = T[0]
        #return T


#class Enum(object):
#    def __init__(self, *sequential, **named):
#        enums = dict(zip(sequential, range(len(sequential))), **named)
#        for k, v in enums.items():
#            setattr(self, k, v)
#
#    def __setattr__(self, name, value): # Read-only
#        raise TypeError("Enum attributes are read-only.")
#
#    def __str__(self):
#        return "Enum: "+str(self.__dict__)