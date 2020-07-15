# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:24:06 2020

@author: pjohn
"""
getNewOptVals = False


import sys
sys.path.append('../')
import projSplit as ps 
from regularizers import Regularizer  
from regularizers import L1
import lossProcessors as lp

import numpy as np
import pickle 
import pytest 
from matplotlib import pyplot as plt

if getNewOptVals:
    from utils import runCVX_lasso
    from utils import getLSdata
    import cvxpy as cvx
    cache = {}
else:
    with open('results/cache_L1LS','rb') as file:
        cache = pickle.load(file)

stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)        
f1fixed = lp.Forward1Fixed(stepsize)
f2bt = lp.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = lp.Forward2Affine()
f1bt = lp.Forward1Backtrack()
@pytest.mark.parametrize("processor,testNumber",[(f2fixed,0),(f2bt,1),(f2affine,2),(f1fixed,3),(f1bt,4)]) 
def test_user_defined_embedded(processor,testNumber):
    
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
    if getNewOptVals and (testNumber==0):
        A,y = getLSdata(m,d)    
        cache['Aembed']=A
        cache['yembed']=y
    else:
        A=cache['Aembed']
        y=cache['yembed']
        
    projSplit = ps.ProjSplitFit()
    

    gamma = 1e0        
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=True)
        
    regObj = []
    nu = [0.01,0.03,0.1]
    step = [1.0,1.0,1.0]
    
    regObj.append(Regularizer(prox1,val1,nu[0],step[0]))
    regObj.append(Regularizer(prox2,val2,nu[1],step[1]))
    regObj.append(Regularizer(prox3,val3,nu[2],step[2]))
    
    
    projSplit.addRegularizer(regObj[0])
    projSplit.addRegularizer(regObj[1],embed=True)
    projSplit.addRegularizer(regObj[2],embed=True)
    
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 5,
                      resetIterate=True)
    
    
    if getNewOptVals and (testNumber==0):
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
        cache['xoptembedded']=xopt
    else:
        xopt=cache['xoptembedded']
        
    xps,_ = projSplit.getSolution()
    print("Norm error = {}".format(np.linalg.norm(xopt-xps,2)))
    assert(np.linalg.norm(xopt-xps,2)<1e-2)
    
    
    
stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)
f1fixed = lp.Forward1Fixed(stepsize)        
f2bt = lp.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = lp.Forward2Affine()
f1bt = lp.Forward1Backtrack()
@pytest.mark.parametrize("processor,testNumber",[(f2fixed,0),(f2bt,1),(f2affine,2),(f1fixed,3),(f1bt,4)]) 
def test_user_defined(processor,testNumber):
    
    
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
    m = 40
    d = 10
    if getNewOptVals and (testNumber==0):
        A,y = getLSdata(m,d)    
        cache['Auser']=A
        cache['yuser']=y
    else:
        A=cache['Auser']
        y=cache['yuser']
        
    for (val,prox) in funcList:
        
        
        projSplit = ps.ProjSplitFit()

        gamma = 1e0 
        projSplit.setDualScaling(gamma)
        projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
        nu = 5.5
        step = 1e0
        regObj = Regularizer(prox,val,nu = nu,step=step)
        projSplit.addRegularizer(regObj)        
        projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1,
                      resetIterate=True,primalTol=1e-12,dualTol=1e-12)
        ps_val = projSplit.getObjective()
        
        (m,d) = A.shape
        if getNewOptVals and (testNumber==0):
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
            cache[(i,'optuser')]=opt
            cache[(i,'xuser')]=xopt
        else:
            opt=cache[(i,'optuser')]
            xopt=cache[(i,'xuser')]
            
            
            
        
        if i == 0:
            xps,_ = projSplit.getSolution()
            print(np.linalg.norm(xopt-xps,2))            
            assert(np.linalg.norm(xopt-xps,2)<1e-2)    
        else:
            print('cvx opt val = {}'.format(opt))
            print('ps opt val = {}'.format(ps_val))        
            assert abs(ps_val-opt)<1e-2
        i += 1
    
    # test combined 
    m = 40
    d = 10
    if getNewOptVals and (testNumber==0):
        A,y = getLSdata(m,d)    
        cache['Acombined']=A
        cache['ycombined']=y
    else:
        A=cache['Acombined']
        y=cache['ycombined']
        
    
    projSplit = ps.ProjSplitFit()
    
            
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    nu1 = 0.01
    step = 1e0
    regObj = Regularizer(prox1,val1,nu = nu1,step=step)
    projSplit.addRegularizer(regObj)        
    nu2 = 0.05
    step = 1e0
    regObj = Regularizer(prox2,val2,nu = nu2,step=step)
    projSplit.addRegularizer(regObj)            
    step = 1e0
    regObj = Regularizer(prox3,val3,step=step)
    projSplit.addRegularizer(regObj)        
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1,
                  resetIterate=True,primalTol=1e-12,dualTol = 1e-12)
    ps_val = projSplit.getObjective()
    xps,_ = projSplit.getSolution()
    
    if getNewOptVals and (testNumber==0):
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
        cache['optcombined']=opt
        cache['xcombined']=xopt
    else:
        opt=cache['optcombined']
        xopt=cache['xcombined']
        
        
    assert(np.linalg.norm(xopt-xps,2)<1e-2)  
    
stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)        
f2bt = lp.Forward2Backtrack(growFactor=1.1,growFreq=10)
f1fixed = lp.Forward1Fixed(stepsize)
f2affine = lp.Forward2Affine()
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
backCG = lp.BackwardCG()
backLBFGS = lp.BackwardLBFGS()

@pytest.mark.parametrize("processor,testNumber",[(backLBFGS,0),(f2fixed,1),(f2bt,2),(f2affine,3),(f1fixed,4),(f1bt,5),(back_exact,6),(backCG,7)]) 
def test_l1_lasso_blocks(processor,testNumber):
    m = 40
    d = 10
    if getNewOptVals and (testNumber==0):
        A,y = getLSdata(m,d)    
        cache['lassoA']=A
        cache['lassoy']=y
    else:
        A=cache['lassoA']
        y=cache['lassoy']
        
    
    projSplit = ps.ProjSplitFit()    
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    if getNewOptVals and (testNumber==0):
        opt,xopt = runCVX_lasso(A,y,lam)
        cache['optlasso']=opt
        cache['xlasso']=xopt
    else:
        opt=cache['optlasso']
        xopt=cache['xlasso']
        
        
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    assert abs(ps_val-opt)<1e-2
        
    
    for numBlocks in range(2,10):
        projSplit.run(maxIterations=2000,keepHistory = True, nblocks = numBlocks)
        ps_val = projSplit.getObjective()
        #print('cvx opt val = {}'.format(opt))
        #print('ps opt val = {}'.format(ps_val))
        assert abs(ps_val-opt)<1e-2
        

    

stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)        
f2bt = lp.Forward2Backtrack(growFactor=1.1,growFreq=10)
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
backCG = lp.BackwardCG()
backLBFGS = lp.BackwardLBFGS()
processors = [f2fixed,f2bt,f2affine,f1fixed,f1bt,back_exact,backCG,backLBFGS]
Tests = []
for inter in [False,True]:
    for norm in [False,True]:
        for processor in processors:
            Tests.append((processor,inter,norm))
            


@pytest.mark.parametrize("processor,inter,norm",Tests) 
def test_l1_intercept_and_normalize(processor,inter,norm):
    m = 40
    d = 10
    if getNewOptVals:
        A = cache.get('Al1intAndNorm')
        y = cache.get('yl1intAndNorm')
        if A is None:
            A,y = getLSdata(m,d)    
            cache['Al1intAndNorm']=A
            cache['yl1intAndNorm']=y
    else:
        A=cache['Al1intAndNorm']
        y=cache['yl1intAndNorm']
        
        
                
    projSplit = ps.ProjSplitFit() 
    if inter and norm:
        gamma = 1e-2
    elif (inter==False) and norm:
        gamma=1e-4
    else:
        gamma = 1e0
        
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=norm,intercept=inter)
    lam = 1e-3
    step = 1.0
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=5000,keepHistory = True, nblocks = 10,primalTol=1e-3,dualTol=1e-3)
    ps_val = projSplit.getObjective()
    
    primViol = projSplit.getPrimalViolation()
    dualViol = projSplit.getDualViolation()
    print("primal violation = {}".format(primViol))
    print("dual violation = {}".format(dualViol))
    
    if getNewOptVals:
        opt = cache.get((inter,norm,'l1opt'))
        if opt is None:
            if norm:
                Anorm = np.copy(A)            
                scaling = np.linalg.norm(Anorm,axis=0)
                scaling += 1.0*(scaling < 1e-10)
                Anorm = Anorm/scaling
            else:
                Anorm = A
            
            AwithIntercept = np.zeros((m,d+1))            
            if inter:                
                AwithIntercept[:,0] = np.ones(m)
            else:
                AwithIntercept[:,0] = np.zeros(m)
                
            AwithIntercept[:,1:(d+1)] = Anorm
                
            opt,_ = runCVX_lasso(AwithIntercept,y,lam,True)
            cache[(inter,norm,'l1opt')]=opt
    else:
            opt=cache[(inter,norm,'l1opt')]
        
            
    
    print('cvx opt val = {}'.format(opt))
    print('ps opt val = {}'.format(ps_val))
    
    
    
    assert abs(ps_val-opt)<1e-2
    
def test_writeCache2Disk():
    if getNewOptVals:
        with open('results/cache_L1LS','wb') as file:
            pickle.dump(cache,file)
            
    
    

    
    
    