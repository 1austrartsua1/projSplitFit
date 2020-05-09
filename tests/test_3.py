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


def test_lasso():
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    lam = 0.0
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    stepsize = 5e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    stepL1 = stepsize
    regObj = ps.L1(lam,stepL1)
    #projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 5,blockActivation="cyclic")     
    
    opt,xopt = runCVX_lasso(A,y,lam)
    print('cvx func opt = {}'.format(opt))
    ps_opt = projSplit.getObjective()
    print('ps func opt = {}'.format(ps_opt))
    
    result = np.linalg.lstsq(A,y,rcond=None)
    xhat = result[0]
    LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
    print('LSval = {}'.format(LSval))
    
    ps_func_vals = projSplit.getHistory()[0]
    plt.plot(ps_func_vals)
    plt.show()
    
    
    
    
    
if __name__ == '__main__':
    test_lasso()
    
    
    