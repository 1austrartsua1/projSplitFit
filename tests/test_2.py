# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:16:24 2020

@author: pjohn
"""
import sys
sys.path.append('/home/pj222/gitFolders/projSplitFit')
sys.path.append('/home/pj222/gitFolders/proj_split_v2.0/group_logistic_regression')
import projSplit as ps 
import numpy as np
import pytest 
from matplotlib import pyplot as plt
from utils import runCVX_LR



def test_ls_fixed():
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    stepsize = 1.0
    processor = ps.Forward2Fixed(stepsize)
    projSplit.addData(A,y,2,processor)
    projSplit.run(maxIterations = 100,keepHistory = True)
    

def test_ls_noIntercept_noNormalize():
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    stepsize = 5e-1
    processor = ps.Forward2Fixed(stepsize)
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
    assert(psSolVal - LSval < 1e-5)
    
def test_ls_noIntercept_Normalize():
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    stepsize = 1e0
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=True,intercept=False)    
    projSplit.run(maxIterations = 3000,keepHistory = True,nblocks = 10)
    ps_sol = projSplit.getSolution()
    
    
    result = np.linalg.lstsq(A,y,rcond=None)
    xhat = result[0]
        
    psSolVal = projSplit.getObjective()
    LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
    
    print('LS val = {}'.format(LSval))
    print('PS val = {}'.format(psSolVal))
    
    #psvals = projSplit.historyArray[0]
    #plt.plot(psvals)
    #plt.show()
    
    #assert np.linalg.norm(xhat - ps_sol)<1e-3
    assert(abs(psSolVal - LSval) < 1e-3)
 
def test_Intercept_noNormalize():
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    stepsize = 5e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=True)  
    projSplit.run(maxIterations = 1000,keepHistory = True,nblocks = 10)
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
    
    
    assert(abs(ps_val - LSval) < 1e-3)
    assert np.linalg.norm(xhat-ps_sol)<1e-1
    
    
    
def test_ls_blocks():
    
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    stepsize = 5e-1
    processor = ps.Forward2Fixed(stepsize)
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

def test_lr():
    projSplit = ps.ProjSplitFit()
    m = 50
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = 2.0*(np.random.normal(0,1,m)>0)-1.0
    lam = 0.0
    
    stepsize = 1e-1 
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,'logistic',processor,normalize=False,intercept=False)    
    projSplit.run(maxIterations = 2000,keepHistory = True,nblocks = 10)    
    ps_opt_val = projSplit.getObjective()
    
    [opt, xopt] = runCVX_LR(A,y,lam)
    print("ps opt is {}".format(ps_opt_val))
    print("cvx opt is {}".format(opt))
        
    assert abs(opt - ps_opt_val)<1e-2
    
def test_lr_normalize():
    projSplit = ps.ProjSplitFit()
    m = 50
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = 2.0*(np.random.normal(0,1,m)>0)-1.0
    lam = 0.0
    
    stepsize = 1e0 
    processor = ps.Forward2Fixed(stepsize)
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
        
    assert abs(opt - ps_opt_val)<1e-3
    
def test_lr_norm_intercept():
    projSplit = ps.ProjSplitFit()
    m = 50
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = 2.0*(np.random.normal(0,1,m)>0)-1.0
    lam = 0.0
    
    stepsize = 1e0 
    processor = ps.Forward2Fixed(stepsize)
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
    
    assert abs(opt - ps_opt_val)<1e-3
    assert np.linalg.norm(xps - xopt)<1e-1
    
    
    projSplit.addData(A,y,'logistic',processor,normalize=False,intercept=True)    
    projSplit.run(maxIterations = 2000,keepHistory = True,nblocks = 10)    
    
    ps_opt_val = projSplit.getObjective()
    assert abs(opt - ps_opt_val)<1e-3
    

def test_lasso():
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    lam = 2.0
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    stepsize = 5e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e-1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False)
    stepL1 = stepsize
    regObj = ps.L1(lam,stepL1)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)    
    

if __name__ == "__main__":    
    test_lr()
    
    
    

    