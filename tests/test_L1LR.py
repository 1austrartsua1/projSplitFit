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


def test_L1LR():
    
    m = 40
    d = 10
    A,y = getLRdata(m,d)
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,'logistic',processor,normalize=False,intercept=False)
    lam = 5e-2
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    opt, xopt = runCVX_LR(A,y,lam)
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-3
    
    projSplit.addData(A,y,'logistic',processor,normalize=True,intercept=False)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    Anorm = np.copy(A)            
    scaling = np.linalg.norm(Anorm,axis=0)
    scaling += 1.0*(scaling < 1e-10)
    Anorm = Anorm/scaling
    opt, xopt = runCVX_LR(Anorm,y,lam)
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    #ps_vals = projSplit.getHistory()[0]
    #plt.plot(ps_vals)
    #plt.show()
    assert abs(ps_val-opt)<1e-3
    
    projSplit.addData(A,y,'logistic',processor,normalize=False,intercept=True)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    opt, xopt = runCVX_LR(AwithIntercept,y,lam,True)
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-3
    
    
    
    
    
    
    
    
    


if __name__ == '__main__':
    test_L1LR()
    
    

