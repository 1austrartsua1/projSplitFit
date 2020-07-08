# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:11:50 2020

@author: pjohn
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
    from utils import runCVX_LR
    from utils import getLRdata
    cache = {}
else:
    with open('results/cache_L1LR','rb') as file:
        cache = pickle.load(file)


stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)
f2bt = lp.Forward2Backtrack()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
backLBFGS = lp.BackwardLBFGS()

processors = [f2fixed,f2bt,f1fixed,f1bt,backLBFGS]

toDo = []
testNumber = 0
for processor in processors:
    for inter in [False,True]:
        for norm in [False,True]:
            toDo.append((processor,norm,inter,testNumber))
            testNumber += 1

@pytest.mark.parametrize("processor,nrm,inter,testNumber",toDo) 
def test_L1LR(processor,nrm,inter,testNumber):

    m = 40
    d = 10
    if getNewOptVals and (testNumber == 0):            
        A,y = getLRdata(m,d)
        cache['A']=A
        cache['y']=y
    else:
        A = cache['A']
        y = cache['y']
        
    
    projSplit = ps.ProjSplitFit()
    
    
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,'logistic',processor,normalize=nrm,intercept=inter)
    lam = 5e-2
    step = 1.0
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    if getNewOptVals:
        opt = cache.get((nrm,inter,'opt'))
        if opt is None:        
            if nrm:
                Anorm = A            
                scaling = np.linalg.norm(Anorm,axis=0)
                scaling += 1.0*(scaling < 1e-10)
                A = Anorm/scaling
                
            if inter:
                 AwithIntercept = np.zeros((m,d+1))
                 AwithIntercept[:,0] = np.ones(m)
                 AwithIntercept[:,1:(d+1)] = A
                 A = AwithIntercept 
                 
                 
            opt, _ = runCVX_LR(A,y,lam,inter)
            cache[(nrm,inter,'opt')]=opt
    else:
        opt = cache[(nrm,inter,'opt')]
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-2
    
  
    
def test_writeCache2Disk():
    if getNewOptVals:
        with open('results/cache_L1LR','wb') as file:
            pickle.dump(cache,file)
    
    
    
    
    
    
    


if __name__ == '__main__':
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    nrm = False
    inter = False
    test_L1LR(processor,nrm,inter)
    
    
    
    
    
    

