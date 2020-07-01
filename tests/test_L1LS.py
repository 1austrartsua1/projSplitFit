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


stepsize = 1e-1
f2fixed = ps.Forward2Fixed(stepsize)        
f1fixed = ps.Forward1Fixed(stepsize)
f2bt = ps.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = ps.Forward2Affine()
f1bt = ps.Forward1Backtrack()
@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt)]) 
def test_user_defined_embedded(processor):
    
    def val1(x):
        return 0.5*np.linalg.norm(x,2)**2
    
    def prox1(x,scale):
        return(1+scale)**(-1)*x
        
    def val2(x):
        return np.linalg.norm(x,2)
    
    def prox2(x,scale):
        normx = np.linalg.norm(x,2)
        if normx <= scale:
            return 0*x
        else:
            return (normx - scale)*x/normx 

    tau = 0.2
    def val3(x):        
        if((x<=tau)&(x>=-tau)).all():            
            return 0
        else:
            return float('inf')
        
    def prox3(x,scale):
        ones = np.ones(x.shape)        
        return tau*(x>=tau)*ones - tau*(x<=-tau)*ones + ((x<=tau)&(x>=-tau))*x 
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
        
    projSplit = ps.ProjSplitFit()
    

    gamma = 1e0        
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=True)
        
    regObj = []
    nu = [0.01,0.03,0.1]
    step = [1.0,1.0,1.0]
    
    regObj.append(ps.Regularizer(val1,prox1,nu[0],step[0]))
    regObj.append(ps.Regularizer(val2,prox2,nu[1],step[1]))
    regObj.append(ps.Regularizer(val3,prox3,nu[2],step[2]))
    
    
    projSplit.addRegularizer(regObj[0])
    projSplit.addRegularizer(regObj[1],embed=True)
    projSplit.addRegularizer(regObj[2],embed=True)
    
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 5,
                      resetIterate=True)
    
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    
    (m,d) = AwithIntercept.shape
    x_cvx = cvx.Variable(d)
    f = (1/(2*m))*cvx.sum_squares(AwithIntercept@x_cvx - y)
    
    constraints = [-tau <= x_cvx[1:d], x_cvx[1:d] <= tau]
    
    f += 0.5*nu[0]*cvx.norm(x_cvx[1:d],2)**2            
    f += nu[1]*cvx.norm(x_cvx[1:d],2)
    
       
    obj =  cvx.Minimize(f)
    prob = cvx.Problem(obj,constraints)
    prob.solve(verbose=False)
    #opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))
    xps,_ = projSplit.getSolution()
    print("Norm error = {}".format(np.linalg.norm(xopt-xps,2)))
    assert(np.linalg.norm(xopt-xps,2)<1e-2)
    
    
    
stepsize = 1e-1
f2fixed = ps.Forward2Fixed(stepsize)
f1fixed = ps.Forward1Fixed(stepsize)        
f2bt = ps.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = ps.Forward2Affine()
f1bt = ps.Forward1Backtrack()
@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt)]) 
def test_user_defined(processor):
    
    
    def val1(x):
        return 0.5*np.linalg.norm(x,2)**2
    
    def prox1(x,scale):
        return(1+scale)**(-1)*x
        
    def val2(x):
        return np.linalg.norm(x,2)
    
    def prox2(x,scale):
        normx = np.linalg.norm(x,2)
        if normx <= scale:
            return 0*x
        else:
            return (normx - scale)*x/normx 

    tau = 0.2
    def val3(x):        
        if((x<=tau)&(x>=-tau)).all():            
            return 0
        else:
            return float('inf')
        
    def prox3(x,scale):
        ones = np.ones(x.shape)        
        return tau*(x>=tau)*ones - tau*(x<=-tau)*ones + ((x<=tau)&(x>=-tau))*x 
    
    funcList = [(val3,prox3),(val1,prox1),(val2,prox2)]
    
    i = 0
    for (val,prox) in funcList:
        m = 40
        d = 10
        A,y = getLSdata(m,d)    
        
        projSplit = ps.ProjSplitFit()

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
            xps,_ = projSplit.getSolution()
            assert(np.linalg.norm(xopt-xps,2)<1e-2)    
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
    xps,_ = projSplit.getSolution()
    assert(np.linalg.norm(xopt-xps,2)<1e-2)  
    
stepsize = 1e-1
f2fixed = ps.Forward2Fixed(stepsize)        
f2bt = ps.Forward2Backtrack(growFactor=1.1,growFreq=10)
f1fixed = ps.Forward1Fixed(stepsize)
f2affine = ps.Forward2Affine()
f1bt = ps.Forward1Backtrack()
@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt)]) 
def test_l1_lasso(processor):
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()    
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
        
    
    for numBlocks in range(2,10):
        projSplit.run(maxIterations=2000,keepHistory = True, nblocks = numBlocks)
        ps_val = projSplit.getObjective()
        #print('cvx opt val = {}'.format(opt))
        #print('ps opt val = {}'.format(ps_val))
        assert abs(ps_val-opt)<1e-2
        
stepsize = 1e-1
f2fixed = ps.Forward2Fixed(stepsize)        
f2bt = ps.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = ps.Forward2Affine()
f1fixed = ps.Forward1Fixed(stepsize)
f1bt = ps.Forward1Backtrack()
@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt)]) 
def test_l1_normalized(processor):    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
        
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

stepsize = 1e-1
f2fixed = ps.Forward2Fixed(stepsize)        
f2bt = ps.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = ps.Forward2Affine()
f1fixed = ps.Forward1Fixed(stepsize)
f1bt = ps.Forward1Backtrack()
@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt)])     
def test_l1_intercept(processor):
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()    
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
    
    
    assert abs(ps_val-opt)<1e-3
    

stepsize = 5e-1
f2fixed = ps.Forward2Fixed(stepsize)        
f2bt = ps.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = ps.Forward2Affine()
f1fixed = ps.Forward1Fixed(stepsize)
f1bt = ps.Forward1Backtrack()
@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt)]) 
def test_l1_intercept_and_normalize(processor):
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()    
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=True,intercept=True)
    lam = 1e-3
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=5000,keepHistory = True, nblocks = 10,primalTol=1e-3,dualTol=1e-3)
    ps_val = projSplit.getObjective()
    
    primViol = projSplit.getPrimalViolation()
    dualViol = projSplit.getDualViolation()
    print("primal violation = {}".format(primViol))
    print("dual violation = {}".format(dualViol))
    
    
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
    
    
    
    assert abs(ps_val-opt)<1e-2
    
    
if __name__ == '__main__':
    test_l1_intercept_and_normalize()
    
    
    