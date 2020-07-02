# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:25:50 2020

@author: pjohn
"""

import sys
sys.path.append('../')
import projSplit as ps 
from regularizers import L1
import lossProcessors as lp

import numpy as np
import pytest 
from matplotlib import pyplot as plt
from utils import runCVX_lasso
from utils import getLSdata

stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)        
f2bt = lp.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()

@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt)]) 
def test_embedded(processor):
    m = 40
    d = 10
    A,y = getLSdata(m,d)        
    projSplit = ps.ProjSplitFit()    
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj,embed=True)
    
    for nblocks in range(1,11,3):
        projSplit.run(maxIterations=1000,keepHistory = True, nblocks = nblocks)
        ps_val = projSplit.getObjective()
        
        opt,xopt = runCVX_lasso(A,y,lam)
        
        print('cvx opt val = {}'.format(opt))
        print('ps opt val = {}'.format(ps_val))
        assert abs(ps_val-opt)<1e-3
        
    
    projSplit.addRegularizer(regObj,embed=True)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 5)
    ps_val = projSplit.getObjective()
    
    opt,xopt = runCVX_lasso(A,y,2*lam)
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-3
    
    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=True,intercept=True)
    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj,embed=True)    
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj,embed=True)            
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 5)
    ps_val = projSplit.getObjective()
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    
    opt,xopt = runCVX_lasso(AwithIntercept,y,2*lam,True,True)
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))    
    assert abs(ps_val-opt)<1e-3
    
    
    
    
stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)        
f2bt = lp.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()

@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt),(back_exact)]) 
def test_l1_multi_lasso(processor):
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()    
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)
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
    test_embedded()
    
    
    