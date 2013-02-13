"""
The :mod:`sandlada.multiblock` module includes several different projection
based latent variable methods for one or more blocks
"""

from .multiblock import PCA
from .multiblock import SVD
from .multiblock import EIGSym
from .multiblock import PLSR
from .multiblock import PLSC
from .multiblock import O2PLS
from .multiblock import center
from .multiblock import scale
from .multiblock import direct

import algorithms
import prox_op

__all__ = ['PCA', 'SVD', 'EIGSym', 'PLSR', 'PLSC', 'O2PLS',
           'center', 'scale', 'direct', 'prox_op', 'algorithms']
