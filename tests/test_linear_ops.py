# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:31:38 2020

@author: pjohn
"""

import sys
sys.path.append('../')
import projSplit as ps 
import numpy as np
import pytest 
import cvxpy as cvx
from utils import getLSdata
from scipy.sparse.linalg import aslinearoperator

def test_linear_op_l1():
    
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
        
    p = 15
    H = np.random.normal(0,1,[p,d])
    lam = 0.01
    step = 1.0
    regObj = ps.L1(lam,step)
    projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H))
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
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
    
    print("ps val = {}".format(ps_val))
    print("cvx val = {}".format(opt))
    assert abs(ps_val - opt) < 1e-3
    

def test_multi_linear_op_l1():
    
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e0
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
        regObj = ps.L1(lam[-1],step)
        projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H[-1]))
        
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    
    (m,d) = A.shape
    x_cvx = cvx.Variable(d)
    f = (1/(2*m))*cvx.sum_squares(A@x_cvx - y)
    for i in range(numregs):
        f += lam[i]*cvx.norm(H[i] @ x_cvx,1)
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))
    
    print("ps val = {}".format(ps_val))
    print("cvx val = {}".format(opt))
    assert abs(ps_val - opt) < 1e-2
    
def test_multi_linear_op_l1_normalize():
    
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=True,intercept=False)
        
    numregs = 5
    H = []
    lam = []
    for i in range(numregs):
        p = np.random.randint(1,100)
        H.append(np.random.normal(0,1,[p,d]))
        lam.append(0.001*(i+1))
        step = 1.0
        regObj = ps.L1(lam[-1],step)
        projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H[-1]))
        
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1)
    ps_val = projSplit.getObjective()
    
    
    (m,d) = A.shape
    Anorm = np.copy(A)            
    scaling = np.linalg.norm(Anorm,axis=0)
    scaling += 1.0*(scaling < 1e-10)
    Anorm = Anorm/scaling
    
    x_cvx = cvx.Variable(d)
    f = (1/(2*m))*cvx.sum_squares(Anorm@x_cvx - y)
    for i in range(numregs):
        f += lam[i]*cvx.norm(H[i] @ x_cvx,1)
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))
    
    print("ps val = {}".format(ps_val))
    print("cvx val = {}".format(opt))
    assert abs(ps_val - opt) < 1e-2
        
def test_multi_linear_op_l1_inter():
    
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
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
        regObj = ps.L1(lam[-1],step)
        projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H[-1]))
        
    projSplit.run(maxIterations=2000,keepHistory = True, nblocks = 1)
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
    assert abs(ps_val - opt) < 1e-2
    
def test_multi_linear_op_l1_inter_multiblocks():
    
    
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    
    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = ps.Forward2Fixed(stepsize)
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
        regObj = ps.L1(lam[-1],step)
        projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H[-1]))
        
    projSplit.run(maxIterations=2000,keepHistory = True, nblocks = 9)
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
    assert abs(ps_val - opt) < 1e-2
    

    
if __name__ == '__main__':
    test_multi_linear_op_l1_inter_multiblocks()
    
    
    
    
