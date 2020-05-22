# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:51:55 2020

@author: pjohn
"""
import sys
sys.path.append('../')
import projSplit as ps 
import numpy as np
import pytest 
from matplotlib import pyplot as plt
import cvxpy as cvx

ToDo = []
f2bt = ps.Forward2Backtrack()

for p in np.linspace(1.01,4,20):
    ToDo.append((p,f2bt))

@pytest.mark.parametrize("p,process",ToDo)
def test_other_p(p,process):
    process.setStep(1.0)
    print(p)
    gamma = 1e0
    projSplit = ps.ProjSplitFit(gamma)
    m = 40
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    
    projSplit.addData(A,y,p,process,normalize=False,intercept=False)
    
    lam = 0.01
    regObj = ps.L1(lam,1.0)
    projSplit.addRegularizer(regObj)
    
    projSplit.run(primalTol=1e-3,dualTol=1e-3,keepHistory = True,nblocks=2,
                  maxIterations=1000,historyFreq=1)
    
    ps_val = projSplit.getObjective()
    ps_vals = projSplit.getHistory()[0]
    
    #plt.plot(ps_vals)
    #plt.show()
    
    x_cvx = cvx.Variable(d)
    f = (1/(m*p))*cvx.pnorm(A @ x_cvx - y,p)**p
    f += lam * cvx.norm(x_cvx, 1)
    prob = cvx.Problem(cvx.Minimize(f))
    prob.solve(verbose=True)
    opt = prob.value
    
    print("ps val  = {}".format(ps_val))
    print("cvx val  = {}".format(opt))
    
    assert abs(ps_val - opt) < 1e-2
    
    