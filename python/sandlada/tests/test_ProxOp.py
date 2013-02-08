# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:46:35 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

import numpy as np
from util import *

def test_ProxOp():
    import ProxOp

    op = ProxOp.L1(0.9) # Keep ~10 %

    op = ProxOp.L1_binsearch(1) # Enforces |x|_1 <= s = 1, by finding lambda

    op = ProxOp.L0_binsearch(10) # Keep 10 variables, by finding lambda

    op = ProxOp.L0_by_count(10) # Keep the 10 absolute largest variables

#    a = np.random.rand(10,1)
#    b = op.prox(a, 0)

#    print a
#    print b
#    print norm(a)
#    print norm1(b)
#    print norm0(b)

if __name__ == "__main__":
    test_ProxOp()