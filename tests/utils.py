# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:59:48 2020

@author: pjohn
"""

# utilities for testing

import cvxpy as cvx
import numpy as np


def runCVX_LR(A,y,lam):
    (m,d) = A.shape
    x_cvx = cvx.Variable(d)
    f = (1/m)*cvx.sum(cvx.logistic(-cvx.multiply(y,A @ x_cvx)))
    f += lam * cvx.norm(x_cvx, 1)
    
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))

    return [opt, xopt]

def runCVX_lasso(A,y,lam):
    (m,d) = A.shape
    x_cvx = cvx.Variable(d)
    f = (1/(2*m))*cvx.sum_squares(A@x_cvx - y)
    f += lam * cvx.norm(x_cvx, 1)
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value
    xopt = np.squeeze(np.array(xopt))

    return [opt, xopt]
    
    