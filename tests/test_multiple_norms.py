# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:25:50 2020

"""
getNewOptVals = False 

import sys
sys.path.append('../')
import projSplit as ps 
from regularizers import L1
import lossProcessors as lp

import numpy as np
import pickle
import pytest 
from matplotlib import pyplot as plt
if getNewOptVals:
    from utils import runCVX_lasso
    from utils import getLSdata
    cache = {}
else:
    with open('results/cache_multiple_norms','rb') as file:
        cache = pickle.load(file)


stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)        
f2bt = lp.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()

@pytest.mark.parametrize("processor,testNumber",[(f2fixed,0),(f2bt,1),(f2affine,2),(f1fixed,3),(f1bt,4)]) 
def test_embedded(processor,testNumber):
    m = 40
    d = 10
    if getNewOptVals and (testNumber==0):
        A,y = getLSdata(m,d)        
        cache['A_embed']=A
        cache['y_embed']=y
    else:
        A = cache['A_embed']
        y = cache['y_embed']
        
    projSplit = ps.ProjSplitFit()    
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)
    
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False,embed=regObj)
        
    if getNewOptVals and (testNumber == 0):
        opt,_ = runCVX_lasso(A,y,lam)
        cache['embed_opt1'] = opt
    else:
        opt = cache['embed_opt1']
    
    for nblocks in range(1,11,3):
        projSplit.run(maxIterations=1000,keepHistory = True, nblocks = nblocks)
        ps_val = projSplit.getObjective()        
        print('cvx opt val = {}'.format(opt))
        print('ps opt val = {}'.format(ps_val))
        assert abs(ps_val-opt)<1e-2
        
    
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 5)
    ps_val = projSplit.getObjective()
    
    if getNewOptVals and (testNumber == 0):
        opt2,_ = runCVX_lasso(A,y,2*lam)
        cache['embed_opt2'] = opt2
    else:
        opt2 = cache['embed_opt2']
        
    
    
    print('cvx opt val = {}'.format(opt2))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt2)<1e-2
    
    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)    
    
    projSplit.addData(A,y,2,processor,normalize=True,intercept=True,embed=regObj)
    
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj)            
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 5)
    ps_val = projSplit.getObjective()
    
    if getNewOptVals and (testNumber == 0):
        AwithIntercept = np.zeros((m,d+1))
        AwithIntercept[:,0] = np.ones(m)
        AwithIntercept[:,1:(d+1)] = A
        
        opt3,_ = runCVX_lasso(AwithIntercept,y,2*lam,True,True)
        cache['embed_opt3']=opt3
    else:
        opt3 = cache['embed_opt3']
        
    print('cvx opt val = {}'.format(opt3))
    print('ps opt val = {}'.format(ps_val))    
    assert abs(ps_val-opt3)<1e-2
    
    
    
    
stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)        
f2bt = lp.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
backLB = lp.BackwardLBFGS()

@pytest.mark.parametrize("processor,testNumber",[(backLB,0),(f2fixed,1),(f2bt,2),(f2affine,3),(f1fixed,4),(f1bt,5),(back_exact,6)]) 
def test_l1_multi_lasso(processor,testNumber):
    m = 40
    d = 10
    if getNewOptVals and (testNumber==0):
        A,y = getLSdata(m,d)    
        cache['Amulti']=A
        cache['ymulti']=y
    else:
        A=cache['Amulti']
        y=cache['ymulti']
    
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
    
    if getNewOptVals and (testNumber==0):
        opt,_ = runCVX_lasso(A,y,fac*lam)
        cache['opt_multi']=opt
    else:
        opt = cache['opt_multi'] 
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-2
    
    
    # test with intercept 
    projSplit.addData(A,y,2,processor,normalize=False,intercept=True)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    if getNewOptVals and (testNumber==0):
        AwithIntercept = np.zeros((m,d+1))
        AwithIntercept[:,0] = np.ones(m)
        AwithIntercept[:,1:(d+1)] = A
        opt_multi_inter,_ = runCVX_lasso(AwithIntercept,y,fac*lam,True)
        cache['opt_multi_inter']=opt_multi_inter
    else:
        opt_multi_inter = cache['opt_multi_inter']
    
    #print('cvx opt val = {}'.format(opt))
    #print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt_multi_inter)<1e-2
    
    # test multi-data-blocks
    
    for bblocks in range(2,11):
        projSplit.run(maxIterations=2000,keepHistory = True, nblocks = bblocks)
        ps_val = projSplit.getObjective()
        print('cvx opt val = {}'.format(opt_multi_inter))
        print('ps opt val = {}'.format(ps_val))
        assert abs(ps_val-opt_multi_inter)<1e-2
    
    
def test_writeCache2Disk():
    if getNewOptVals:
        with open('results/cache_multiple_norms','wb') as file:
            pickle.dump(cache,file)
            


    
    