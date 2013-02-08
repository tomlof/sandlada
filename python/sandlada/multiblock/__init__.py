"""
The :mod:`sandlada.multiblock` module includes several different projection
based latent variable methods for one or more blocks
"""

from .multiblock import PCA
from .multiblock import SVD
from .multiblock import PLSR
from .multiblock import PLSC
from .multiblock import center
from .multiblock import scale
from .multiblock import direct

from .multiblock import ProxOp

__all__ = ['PCA', 'SVD', 'PLSR', 'PLSC', 'center', 'scale', 'direct',
           'ProxOp']
