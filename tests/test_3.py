# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:24:06 2020

@author: pjohn
"""
import sys
sys.path.append('/home/pj222/gitFolders/projSplitFit')
import projSplit as ps 
import numpy as np
import pytest 
from matplotlib import pyplot as plt
from utils import runCVX_lasso
from utils import getLSdata


    
    
def test_l1_lasso():
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    lam = 1.0
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    opt,xopt = runCVX_lasso(A,y,lam)
    #print('cvx opt val = {}'.format(opt))
    #print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-3
    
    #ps_vals = projSplit.getHistory()[0]
    #plt.plot(ps_vals)
    #plt.show()
    
    for numBlocks in range(2,10):
        projSplit.run(maxIterations=2000,keepHistory = True, nblocks = numBlocks)
        ps_val = projSplit.getObjective()
        #print('cvx opt val = {}'.format(opt))
        #print('ps opt val = {}'.format(ps_val))
        assert abs(ps_val-opt)<1e-2
        
    
    
    
    
    
    
    
if __name__ == '__main__':
    pass
    
    
    