# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:31:38 2020

@author: pjohn
"""

import sys
sys.path.append('../')
import projSplit as ps 
import numpy as np
import pytest 
from matplotlib import pyplot as plt
from utils import runCVX_lasso
from utils import getLSdata
from scipy.sparse.linalg import aslinearoperator

def test_linear_op_l1():
    
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    lam = 0.01
    step = 1.0
    regObj = ps.L1(lam,step)
    
    p = 15
    H = np.random.normal(0,1,[p,d])
    lam = 0.01
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H))
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    
    
if __name__ == '__main__':
    test_linear_op_l1()
    
    
    
    
