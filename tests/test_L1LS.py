# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:24:06 2020

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
import cvxpy as cvx

def test_user_defined():
    
    def val1(x,nu):
        return 0.5*nu*np.linalg.norm(x,2)**2
    
    def prox1(x,nu,step):
        return(1+step*nu)**(-1)*x
        
    def val2(x,nu):
        return nu*np.linalg.norm(x,2)
    
    def prox2(x,nu,step):
        normx = np.linalg.norm(x,2)
        if normx <= step*nu:
            return 0*x
        else:
            return (normx - step*nu)*x/normx 

    tau = 0.2
    def val3(x,nu):        
        if((x<=tau)&(x>=-tau)).all():            
            return 0
        else:
            return float('inf')
        
    def prox3(x,nu,step):
        ones = np.ones(x.shape)        
        return tau*(x>=tau)*ones - tau*(x<=-tau)*ones + ((x<=tau)&(x>=-tau))*x 
    
    funcList = [(val3,prox3),(val1,prox1),(val2,prox2)]
    
    i = 0
    for (val,prox) in funcList:
        m = 40
        d = 10
        A,y = getLSdata(m,d)    
        
        projSplit = ps.ProjSplitFit()
        stepsize = 1e-1
        processor = ps.Forward2Fixed(stepsize)        

        gamma = 1e0        
        projSplit.setDualScaling(gamma)
        projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
        nu = 5.5
        step = 1e0
        regObj = ps.Regularizer(val,prox,nu = nu,step=step)
        projSplit.addRegularizer(regObj)        
        projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1,
                      resetIterate=True)
        ps_val = projSplit.getObjective()
        
        (m,d) = A.shape
        x_cvx = cvx.Variable(d)
        f = (1/(2*m))*cvx.sum_squares(A@x_cvx - y)
        
        if i == 0:                        
            constraints = [-tau <= x_cvx, x_cvx <= tau]
        elif i ==1:
            f += 0.5*nu*cvx.norm(x_cvx,2)**2
            constraints = []
        elif i == 2:
            f += nu*cvx.norm(x_cvx,2)
            constraints = []
        
            
            
        obj =  cvx.Minimize(f)
        prob = cvx.Problem(obj,constraints)
        prob.solve(verbose=True)
        opt = prob.value
        xopt = x_cvx.value
        xopt = np.squeeze(np.array(xopt))
        
        if i == 0:
            assert(np.linalg.norm(xopt-projSplit.getSolution(),2)<1e-2)    
        else:
            print('cvx opt val = {}'.format(opt))
            print('ps opt val = {}'.format(ps_val))        
            assert abs(ps_val-opt)<1e-2
        i += 1
    
    # test combined 
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)        

    gamma = 1e0        
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    nu1 = 0.01
    step = 1e0
    regObj = ps.Regularizer(val1,prox1,nu = nu1,step=step)
    projSplit.addRegularizer(regObj)        
    nu2 = 0.05
    step = 1e0
    regObj = ps.Regularizer(val2,prox2,nu = nu2,step=step)
    projSplit.addRegularizer(regObj)            
    step = 1e0
    regObj = ps.Regularizer(val3,prox3,step=step)
    projSplit.addRegularizer(regObj)        
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1,
                  resetIterate=True)
    ps_val = projSplit.getObjective()
    
    x_cvx = cvx.Variable(d)
    f = (1/(2*m))*cvx.sum_squares(A@x_cvx - y)
    
    
    constraints = [-tau <= x_cvx, x_cvx <= tau]
    
    f += 0.5*nu1*cvx.norm(x_cvx,2)**2            
    f += nu2*cvx.norm(x_cvx,2)
    
    obj =  cvx.Minimize(f)
    prob = cvx.Problem(obj,constraints)
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))    
    assert(np.linalg.norm(xopt-projSplit.getSolution(),2)<1e-2)  
    
    
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
    lam = 0.01
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    opt,xopt = runCVX_lasso(A,y,lam)
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-3
    
    xps = projSplit.getSolution()
    #plt.plot(xps)
    #plt.plot(xopt)
    #plt.show()
    
    ps_vals = projSplit.getHistory()[0]
    #plt.plot(ps_vals)
    #plt.show()
    
    for numBlocks in range(2,10):
        projSplit.run(maxIterations=2000,keepHistory = True, nblocks = numBlocks)
        ps_val = projSplit.getObjective()
        #print('cvx opt val = {}'.format(opt))
        #print('ps opt val = {}'.format(ps_val))
        assert abs(ps_val-opt)<1e-2
        
def test_l1_normalized():
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=True,intercept=False)
    lam = 0.03
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    Anorm = np.copy(A)            
    scaling = np.linalg.norm(Anorm,axis=0)
    scaling += 1.0*(scaling < 1e-10)
    Anorm = Anorm/scaling
    opt,xopt = runCVX_lasso(Anorm,y,lam)
    #print('cvx opt val = {}'.format(opt))
    #print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-3
    
def test_l1_intercept():
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=True)
    lam = 1e-3
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    opt,xopt = runCVX_lasso(AwithIntercept,y,lam,True)
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    
    #xps = projSplit.getSolution()
    #plt.plot(xps-xopt)
    #plt.show()
    
    #ps_all_vals = projSplit.getHistory()[0]
    #plt.plot(ps_all_vals)
    #plt.show()
    assert abs(ps_val-opt)<1e-3
    

def test_l1_intercept_and_normalize():
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 5e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=True,intercept=True)
    lam = 2e-4
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 10)
    ps_val = projSplit.getObjective()
    
    
    Anorm = np.copy(A)            
    scaling = np.linalg.norm(Anorm,axis=0)
    scaling += 1.0*(scaling < 1e-10)
    Anorm = Anorm/scaling
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = Anorm
    opt,xopt = runCVX_lasso(AwithIntercept,y,lam,True)
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    
    #xps = projSplit.getSolution()
    #plt.plot(xps-xopt)
    #plt.show()
    
    #ps_all_vals = projSplit.getHistory()[0]
    #plt.plot(ps_all_vals)
    #plt.show()
    
    assert abs(ps_val-opt)<1e-3
    
    
if __name__ == '__main__':
    test_user_defined()
    
    
    