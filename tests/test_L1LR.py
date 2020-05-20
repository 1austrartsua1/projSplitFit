# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:11:50 2020

@author: pjohn
"""

import sys
sys.path.append('../')
import projSplit as ps 
import numpy as np
import pytest 
from matplotlib import pyplot as plt
from utils import runCVX_LR
from utils import getLRdata


stepsize = 1e-1
f2fixed = ps.Forward2Fixed(stepsize)
f2bt = ps.Forward2Backtrack()
toDo = [(f2fixed,False,False),(f2fixed,True,False),
        (f2fixed,False,True),(f2fixed,True,True)]
toDo.extend(
        [(f2bt,False,False),(f2bt,True,False),
        (f2bt,False,True),(f2bt,True,True)]        
        )

@pytest.mark.parametrize("processor,nrm,inter",toDo) 
def test_L1LR(processor,nrm,inter):
    
    m = 40
    d = 10
    A,y = getLRdata(m,d)
    
    projSplit = ps.ProjSplitFit()
    
    
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,'logistic',processor,normalize=nrm,intercept=inter)
    lam = 5e-2
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
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
         
         
    opt, xopt = runCVX_LR(A,y,lam,inter)
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-3
    
  
    
    
    
    
    
    
    
    
    


if __name__ == '__main__':
    test_L1LR()
    
    

