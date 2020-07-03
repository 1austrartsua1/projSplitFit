# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:31:38 2020

@author: pjohn
"""

import sys
sys.path.append('../')
import projSplit as ps 
from regularizers import L1
import lossProcessors as lp

import numpy as np
import pytest 
import cvxpy as cvx
from utils import getLSdata
from scipy.sparse.linalg import aslinearoperator
from matplotlib import pyplot as plt


def test_linear_op_data_term_wrong():
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    p = 15
    d2 = 11
    H = np.random.normal(0,1,[d2,p])
    try:
        projSplit.addData(A,y,2,processor,normalize=False,intercept=False,
                      linearOp = aslinearoperator(H))
        notExcept = True
                          
    except:
        notExcept = False
    
    assert notExcept == False 
                             

f2fix = lp.Forward2Fixed()
back2exact = lp.BackwardExact()
backCG = lp.BackwardCG()
f1bt = lp.Forward1Backtrack()
TryAll = []
for i in [False,True]:
    for j in [False,True]:
        for k in [False,True]:
            for l in [False,True]:
                for p in [f2fix,back2exact,f1bt,backCG]:
                    TryAll.append((i,j,k,l,p))
                
                
@pytest.mark.parametrize("norm,inter,addL1,add2L1,processor",TryAll)
def test_linear_op_data_term(norm,inter,addL1,add2L1,processor):
    
                             
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()    
    
    processor.setStep(5e-1)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    p = 15
    d2 = 10
    H = np.random.normal(0,1,[d2,p])
    
    projSplit.addData(A,y,2,processor,normalize=norm,intercept=inter,
                      linearOp = aslinearoperator(H))
    
    
    
    lam = 0.01
    step = 1.0
    if addL1:
        regObj = L1(lam,step)
        projSplit.addRegularizer(regObj)
    
    if add2L1:
        regObj2 = L1(lam,step)
        projSplit.addRegularizer(regObj2)
        
    projSplit.run(maxIterations=10000,keepHistory = True, 
                  nblocks = 3,primalTol=1e-3,dualTol=1e-3)
    ps_val = projSplit.getObjective()
    
    primViol = projSplit.getPrimalViolation()
    dualViol = projSplit.getDualViolation()
    print("primal violation = {}".format(primViol))
    print("dual violation = {}".format(dualViol))
    
    
    
    if norm == True:         
        scaling = np.linalg.norm(A,axis=0)
        scaling += 1.0*(scaling < 1e-10)
        A = A/scaling
    if inter == True:
        AwithIntercept = np.zeros((m,d+1))
        AwithIntercept[:,0] = np.ones(m)
        AwithIntercept[:,1:(d+1)] = A
        A = AwithIntercept
        HwithIntercept = np.zeros((d2+1,p+1))
        HwithIntercept[:,0] = np.zeros(d2+1)
        HwithIntercept[0] = np.ones(p+1)
        HwithIntercept[0,0] = 1.0
        HwithIntercept[1:(d2+1),1:(p+1)] = H
        H = HwithIntercept
        
    
    (m,_) = A.shape
    if inter:
        x_cvx = cvx.Variable(p+1)
    else:
        x_cvx = cvx.Variable(p)
        
    f = (1/(2*m))*cvx.sum_squares(A@H@x_cvx - y)
    if addL1:
        f += lam*cvx.norm(x_cvx,1)
    
    if add2L1:
        f += lam*cvx.norm(x_cvx,1)
        
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))
    
    #if norm and inter and (not addL1) and (add2L1):
    #    ps_vals = projSplit.getHistory()[0]
    #    plt.plot(ps_vals)
    #    plt.show()
        
    
    print("ps opt = {}".format(ps_val))
    print("cvx opt = {}".format(opt))
    assert(ps_val-opt<1e-2)
    
@pytest.mark.parametrize("norm,inter",[(False,False)]) 
def test_linear_op_l1(norm,inter):
    
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
        
    p = 15
    H = np.random.normal(0,1,[p,d])
    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H))
    projSplit.run(maxIterations=5000,keepHistory = True, nblocks = 1,
                  primalTol=1e-3,dualTol=1e-3)
    ps_val = projSplit.getObjective()
    

        
        
    (m,d) = A.shape
    x_cvx = cvx.Variable(d)
    f = (1/(2*m))*cvx.sum_squares(A@x_cvx - y)
    f += lam*cvx.norm(H @ x_cvx,1)
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))
    
    primViol = projSplit.getPrimalViolation()
    dualViol = projSplit.getDualViolation()
    print("primal violation = {}".format(primViol))
    print("dual violation = {}".format(dualViol))
    
    print("ps val = {}".format(ps_val))
    print("cvx val = {}".format(opt))
    assert ps_val - opt < 1e-3
    

Todo = [(False,False),(True,False),
        (False,True),(True,True)]
@pytest.mark.parametrize("norm,inter",Todo) 
def test_multi_linear_op_l1(norm,inter):
    
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    gamma = 1e0
    if norm and inter:
        gamma = 1e2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
        
    numregs = 5
    H = []
    lam = []
    for i in range(numregs):
        p = np.random.randint(1,100)
        H.append(np.random.normal(0,1,[p,d]))
        lam.append(0.001*(i+1))
        step = 1.0
        regObj = L1(lam[-1],step)
        projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H[-1]))
        
    projSplit.run(maxIterations=5000,keepHistory = True, nblocks = 1,
                  primalTol=1e-4,dualTol=1e-4)
    ps_val = projSplit.getObjective()
    
    if norm:
        Anorm = A            
        scaling = np.linalg.norm(Anorm,axis=0)
        scaling += 1.0*(scaling < 1e-10)
        Anorm = Anorm/scaling
        A = Anorm
    
    if inter:
        AwithIntercept = np.zeros((m,d+1))
        AwithIntercept[:,0] = np.ones(m)
        AwithIntercept[:,1:(d+1)] = A
        A = AwithIntercept
        
        
    (m,d) = A.shape
    x_cvx = cvx.Variable(d)
    f = (1/(2*m))*cvx.sum_squares(A@x_cvx - y)
    for i in range(numregs):
        if inter:
            f += lam[i]*cvx.norm(H[i] @ x_cvx[1:d],1)
        else:
            f += lam[i]*cvx.norm(H[i] @ x_cvx,1)
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))
    
    print("ps val = {}".format(ps_val))
    print("cvx val = {}".format(opt))
    
    #if norm and inter:
    #    ps_val = projSplit.getHistory()[0]
    #    plt.plot(ps_val)
    #    plt.show()
    assert ps_val - opt < 1e-2
    

    
def test_multi_linear_op_l1_inter_multiblocks():
    
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=True)
        
    numregs = 5
    H = []
    lam = []
    for i in range(numregs):
        p = np.random.randint(1,100)
        H.append(np.random.normal(0,1,[p,d]))
        lam.append(0.001*(i+1))
        step = 1.0
        regObj = L1(lam[-1],step)
        projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H[-1]))
        
    projSplit.run(maxIterations=5000,keepHistory = True, nblocks = 9,primalTol=1e-3,dualTol=1e-3)
    ps_val = projSplit.getObjective()
    
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    
    (m,d) = AwithIntercept.shape
    x_cvx = cvx.Variable(d)
    f = (1/(2*m))*cvx.sum_squares(AwithIntercept@x_cvx - y)
    for i in range(numregs):
        p = H[i].shape[0]
        zerCol = np.zeros((p,1))
        Htild = np.concatenate((zerCol,H[i]),axis=1)
        f += lam[i]*cvx.norm(Htild @ x_cvx,1)
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))
    
    print("ps val = {}".format(ps_val))
    print("cvx val = {}".format(opt))
    assert ps_val - opt < 1e-2
    

    
if __name__ == '__main__':
    test_multi_linear_op_l1_inter_multiblocks()
    
    
    
    
