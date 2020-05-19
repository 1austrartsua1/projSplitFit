# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:16:24 2020

@author: pjohn
"""
import sys
sys.path.append('../')
import projSplit as ps 
import numpy as np
import pytest 
from matplotlib import pyplot as plt
from utils import runCVX_LR
from utils import getLSdata

stepsize = 1.0
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed])
def test_ls_fixed(processor):
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    projSplit.addData(A,y,2,processor)
    projSplit.run(maxIterations = 100,keepHistory = True)

stepsize = 5e-1
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed])
def test_cyclic(processor):
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    
    result = np.linalg.lstsq(A,y,rcond=None)
    xhat = result[0]
    LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
    gamma = 1e-1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)

    #projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=2000,keepHistory = True, nblocks = 5,blockActivation="cyclic")     
    
    ps_opt = projSplit.getObjective()
    
    assert abs(ps_opt - LSval)<1e-2
    
    for blks in range(2,7):   
        projSplit.run(maxIterations=2000,keepHistory = True, nblocks = 5,
                  blockActivation="cyclic",resetIterate=True, blocksPerIteration=blks)

        ps_opt = projSplit.getObjective()
        assert abs(ps_opt - LSval)<1e-2
    
    

stepsize = 5e-1
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed])
def test_ls_noIntercept_noNormalize(processor):
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    gamma = 1e-1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)    
    projSplit.run(maxIterations = 1000,keepHistory = True,nblocks = 10)
    ps_sol = projSplit.getSolution()
    
    result = np.linalg.lstsq(A,y,rcond=None)
    xhat = result[0]
    
    
    assert(np.linalg.norm(xhat-ps_sol)<1e-5)
    
    psSolVal = projSplit.getObjective()
    LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
    assert(abs(psSolVal - LSval) < 1e-5)
    
stepsize = 1e0
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed])
def test_ls_noIntercept_Normalize(processor):
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=True,intercept=False)    
    projSplit.run(maxIterations = 3000,keepHistory = True,nblocks = 10)
    #ps_sol = projSplit.getSolution()
    
    
    result = np.linalg.lstsq(A,y,rcond=None)
    xhat = result[0]
        
    psSolVal = projSplit.getObjective()
    LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
    
    print('LS val = {}'.format(LSval))
    print('PS val = {}'.format(psSolVal))
    
    #psvals = projSplit.historyArray[0]
    #plt.plot(psvals)
    #plt.show()
    
    #assert np.linalg.norm(xhat - ps_sol)<1e-2
    assert(abs(psSolVal - LSval) < 1e-2)
 
stepsize = 1e-1
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed])
def test_Intercept_noNormalize(processor):
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=True)  
    projSplit.run(maxIterations = 2000,keepHistory = True,nblocks = 1)
    ps_sol = projSplit.getSolution()
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
    xhat = result[0]  
    
    ps_val = projSplit.getObjective()
    LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
    print('LS val = {}'.format(LSval))
    print('PS val = {}'.format(ps_val))
    
    
    assert(abs(ps_val - LSval) < 1e-2)
    assert np.linalg.norm(xhat-ps_sol)<1e-1
    
    
stepsize = 5e-1
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed]) 
def test_ls_blocks(processor):
    
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    gamma = 1e-1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False)    
    projSplit.run(maxIterations = 1000,keepHistory = True,nblocks = 10)
    assert projSplit.getObjective() >= 0, "objective is not >= 0"
    sol = projSplit.getSolution()
    assert sol.shape == (d+1,)
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
    xhat = result[0]    
    
    LSresid = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
    PSresid = projSplit.getObjective()
    PSresid2 = 0.5*np.linalg.norm(AwithIntercept.dot(sol)-y,2)**2/m
    assert abs(LSresid - PSresid) <1e-5
    assert abs(PSresid - PSresid2)<1e-5
    
    
    projSplit.run(maxIterations = 1000,keepHistory = True,resetIterate=True,blockActivation="random",nblocks = 10)
    PSrandom = projSplit.getObjective()
    assert abs(PSresid - PSrandom)<1e-5
    
    projSplit.run(maxIterations = 1000,keepHistory = True,resetIterate=True,blockActivation="cyclic",nblocks = 10)    
    PScyclic = projSplit.getObjective()
    assert abs(PSresid  - PScyclic)<1e-5

stepsize = 1e-1
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed]) 
def test_lr(processor):
    projSplit = ps.ProjSplitFit()
    m = 50
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = 2.0*(np.random.normal(0,1,m)>0)-1.0
    lam = 0.0
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,'logistic',processor,normalize=False,intercept=False)    
    projSplit.run(maxIterations = 2000,keepHistory = True,nblocks = 10)    
    ps_opt_val = projSplit.getObjective()
    
    [opt, xopt] = runCVX_LR(A,y,lam)
    print("ps opt is {}".format(ps_opt_val))
    print("cvx opt is {}".format(opt))
        
    assert abs(opt - ps_opt_val)<1e-2
    
stepsize = 1e0
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed]) 
def test_lr_normalize(processor):
    projSplit = ps.ProjSplitFit()
    m = 50
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = 2.0*(np.random.normal(0,1,m)>0)-1.0
    lam = 0.0
    gamma = 1e-4
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,'logistic',processor,normalize=True,intercept=False)    
    projSplit.run(maxIterations = 2000,keepHistory = True,nblocks = 10)    
    
    ps_opt_val = projSplit.getObjective()
    
    [opt, xopt] = runCVX_LR(A,y,lam)
    print("ps opt is {}".format(ps_opt_val))
    print("cvx opt is {}".format(opt))
    
    #func_val = projSplit.historyArray[0]
    #plt.plot(func_val)
    #plt.show()
        
    assert abs(opt - ps_opt_val)<1e-2
    
stepsize = 1e0
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed]) 
def test_lr_norm_intercept(processor):
    projSplit = ps.ProjSplitFit()
    m = 50
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = 2.0*(np.random.normal(0,1,m)>0)-1.0
    lam = 0.0
    gamma = 1e-4
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,'logistic',processor,normalize=True,intercept=True)    
    projSplit.run(maxIterations = 2000,keepHistory = True,nblocks = 10)    
    
    ps_opt_val = projSplit.getObjective()
    
    xps = projSplit.getSolution()
    
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    
    [opt, xopt] = runCVX_LR(AwithIntercept,y,lam)
    print("ps opt is {}".format(ps_opt_val))
    print("cvx opt is {}".format(opt))
    
    assert abs(opt - ps_opt_val)<1e-2
    assert np.linalg.norm(xps - xopt)<1e-1
    
    
    projSplit.addData(A,y,'logistic',processor,normalize=False,intercept=True)    
    projSplit.run(maxIterations = 2000,keepHistory = True,nblocks = 10)    
    
    ps_opt_val = projSplit.getObjective()
    assert abs(opt - ps_opt_val)<1e-2
    

stepsize = 1e-1
f2fixed = ps.Forward2Fixed(stepsize)
@pytest.mark.parametrize("processor",[f2fixed]) 
def test_blockIs1bug(processor):
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    projSplit = ps.ProjSplitFit()
    
    gamma = 1e1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1,blockActivation="random")     
    
    ps_opt = projSplit.getObjective()
    print('ps func opt = {}'.format(ps_opt))
    
    result = np.linalg.lstsq(A,y,rcond=None)
    xhat = result[0]
    LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
    print('LSval = {}'.format(LSval))
    
    #xps = projSplit.getSolution()
    #plt.plot(xhat - xps)
    #plt.show()
    
    #ps_func_vals = projSplit.getHistory()[0]
    #plt.plot(ps_func_vals)
    #plt.show()
    

if __name__ == "__main__":    
    test_lr_normalize()
    
    
    

    