# -*- coding: utf-8 -*-
"""
The :mod:`sandlada.multiblock.ProxOp` module includes several proximal
operators (or approximate operators).

Created on Thu Feb 7 11:50:00 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

#import abc
import warnings
import numpy as np
from sandlada.utils import *

class ProxOp(object):
    """Baseclass for proximal operators.

    The baseclass also works as an identity operator,
    i.e. one for which x == ProxOp.prox(x).
    """
#    __metaclass__ = abc.ABCMeta

#    @abc.abstractmethod
    def prox(self, x, *args, **kwargs):
#        raise NotImplementedError('Abstract method "prox" must be specialised!')
        return x


class L1(ProxOp):

    def __init__(self, *l):
        self.parameter = list(l)

    def prox(self, x, index, allow_empty = False,
             normalise = norm):

        xorig = x.copy()
        lorig = self.parameter[index]
        l     = lorig

        warn = False
        while True:
            x = xorig / normalise(xorig)

            sign = np.sign(x)
            np.absolute(x, x)
            x -= l
            x[x < 0] = 0
            x = np.multiply(sign,x)

            if norm(x) > TOLERANCE or allow_empty:
                break
            else:
                warn = True
                # TODO: Improved this!
                l *= 0.9 # Reduce by 10 % until at least one variable is significant

        if warn:
            warnings.warn('Soft threshold was too large (all variables purged).'\
                    ' Threshold reset to %f (was %f)' % (l, lorig))
        return x


class L1_binsearch(ProxOp):
    """L1 regularisation.

    Chooses lambda such that

        lambda = 0 if |x| <= s or else
        lambda chosen by binary search such that |x| = s,

    where |.| is the L1-norm.
    """

    def __init__(self, *s):
        self.parameter = s

    def prox(self, x, index, normalise = norm):

        s = self.parameter[index]

        tol = 2**-10

        if norm1(x) > s:
            x = x / normalise(x)
            minl = 0
            maxl = np.absolute(x).max()
#            print maxl
#            aaa = L1(minl)
#            minv = norm1(aaa.prox(x, 0, allow_empty = True))
#            aaa = L1(maxl)
#            maxv = norm1(aaa.prox(x, 0, allow_empty = True))
            midv = 0

            op = L1(1)
            it = 0
#            while maxl - minl > tol and it <= MAX_ITER:
            while abs(s - midv) > tol and it <= MAX_ITER:
                midl = (maxl + minl) / 2.0
                op.parameter[0] = midl
                midv = norm1(op.prox(x, 0, allow_empty = True))

#                print minl, "-", midl, "-", maxl, "    ", minv, "-", midv, "-", maxv

                if midv < s:
                    maxl = midl
                elif midv > s:
                    minl = midl

                it += 1

            op.parameter[0] = (maxl + minl) / 2.0
            x = op.prox(x, 0)

        return x


class L0_binsearch(ProxOp):
    """L1 regularisation.

    Lambda is defined as

        lambda = 0 if |x| <= n or else
        lambda chosen by binary search such that |x| = n,

    where |.| is the L0-norm.
    """

    def __init__(self, *n):
        self.parameter = n

    def prox(self, x, index, normalise = norm):

        n = self.parameter[index]
        tol = 2**-10

        if norm0(x) > n:
            x = x / normalise(x)

            minl = 0
            maxl = np.absolute(x).max()
#            print maxl
#            aaa = L1(minl)
#            minv = norm0(aaa.prox(x, 0, allow_empty = True))
#            aaa = L1(maxl)
#            maxv = norm0(aaa.prox(x, 0, allow_empty = True))

            midl = (maxl + minl) / 2.0
            op = L1(midl)
            midv = norm0(op.prox(x, 0, allow_empty = True))

            it = 0
            while abs(n - midv) > tol and it <= MAX_ITER:
                midl = (maxl + minl) / 2.0
                op.parameter[0] = midl
                midv = norm0(op.prox(x, 0, allow_empty = True))

#                print minl, "-", midl, "-", maxl, "    ", minv, "-", midv, "-", maxv

                if midv < n:
                    maxl = midl
                elif midv > n:
                    minl = midl

                it += 1

            op.parameter[0] = (maxl + minl) / 2.0
            x = op.prox(x, 0)

        return x


class L0_by_count(ProxOp):

    def __init__(self, *num):
        self.parameter = num

    def prox(self, x, index, normalise = norm):

        target_num = self.parameter[index]
        minf = float("-Inf")

        x = x / normalise(x)

        cp  = np.absolute(x)
        ind = np.zeros(target_num, int)
        for i in xrange(target_num):
            idx     = np.argmax(cp)
            ind[i]  = idx
            cp[idx] = minf

        l = x[ind[-1]]

        x[np.absolute(x) < l] = 0

        return x
