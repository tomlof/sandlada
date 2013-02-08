# -*- coding: utf-8 -*-
"""
The :mod:`sandlada.util` module includes common functions and constants.

Please add anything you need throughout the whole package to this module.
(As opposed to having several commong definitions scattered all over the source)

Created on Thu Feb 8 09:22:00 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

import numpy as _np
from numpy.linalg import norm

__all__ = ['norm', 'norm1', 'norm0', 'TOLERANCE', 'MAX_ITER']

# Settings
TOLERANCE = 5e-7
MAX_ITER  = 500

def norm1(x):
    return norm(x, ord = 1)

def norm0(x):
    return _np.count_nonzero(_np.absolute(x))
