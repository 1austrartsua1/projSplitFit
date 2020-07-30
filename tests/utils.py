# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:59:48 2020

@author: pjohn
"""

# utilities for testing

import cvxpy as cvx
import numpy as np


def runCVX_LR(A,y,lam,intercept=False):
    (m,d) = A.shape
    x_cvx = cvx.Variable(d)
    f = (1/m)*cvx.sum(cvx.logistic(-cvx.multiply(y,A @ x_cvx)))
    
    if intercept:
        # the intercept term is assumed to be the first coefficient
        f += lam * cvx.norm(x_cvx[1:d], 1)
    else:
        f += lam * cvx.norm(x_cvx, 1)
    
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))

    return opt, xopt

def runCVX_lasso(Ain,y,lam,intercept = False,normalize = False):
    if normalize:        
        A = np.copy(Ain)            
        n = A.shape[0]
        scaling = np.linalg.norm(A,axis=0)
        scaling += 1.0*(scaling < 1e-10)
        A = np.sqrt(n)*A/scaling
    else:
        A = Ain
        
    
    (m,d) = A.shape
    x_cvx = cvx.Variable(d)
    f = (1/(2*m))*cvx.sum_squares(A@x_cvx - y)
    if intercept:
        # the intercept term is assumed to be the first coefficient
        f += lam * cvx.norm(x_cvx[1:d], 1)
    else:
        f += lam * cvx.norm(x_cvx, 1)
        
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))

    return [opt, xopt]

def getLSdata(m,d):    
    
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    return A,y

def getLRdata(m,d):
    A = np.random.normal(0,1,[m,d])
    y = 2.0*(np.random.normal(0,1,m)>0)-1.0
    return A,y



    
    
    
    
    
    