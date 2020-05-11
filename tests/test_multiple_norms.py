# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:25:50 2020

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



def test_l1_multi_lasso():
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
    fac = 5 # add the same regularizer twice, same as using 
            # it once with twice the parameter
    for _ in range(fac):
        projSplit.addRegularizer(regObj)

    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    opt,xopt = runCVX_lasso(A,y,fac*lam)
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-3
    
    #xps = projSplit.getSolution()
    #plt.plot(xps)
    #plt.plot(xopt)
    #plt.show()
    
    # test with intercept 
    projSplit.addData(A,y,2,processor,normalize=False,intercept=True)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    opt,xopt = runCVX_lasso(AwithIntercept,y,fac*lam,True)
    
    #print('cvx opt val = {}'.format(opt))
    #print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-3
    
    # test multi-data-blocks
    
    for bblocks in range(2,11):
        projSplit.run(maxIterations=2000,keepHistory = True, nblocks = bblocks)
        ps_val = projSplit.getObjective()
        print('cvx opt val = {}'.format(opt))
        print('ps opt val = {}'.format(ps_val))
        assert abs(ps_val-opt)<1e-3
    
    

if __name__ == '__main__':
    test_l1_multi_lasso()
    
    
    