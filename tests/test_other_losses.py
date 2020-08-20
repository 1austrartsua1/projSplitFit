# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:51:55 2020

"""

getNewOptVals = False

import sys
sys.path.append('../')
import projSplitFit as ps 
from regularizers import L1
import lossProcessors as lp

import numpy as np
import pickle

import pytest
from matplotlib import pyplot as plt

if getNewOptVals:
    import cvxpy as cvx
    cache_otherLosses = {}
else:
    np.random.seed(1)
    with open('results/cache_otherLosses','rb') as file:
        cache_otherLosses = pickle.load(file)

ToDo = []
f2bt = lp.Forward2Backtrack()

testNumber = 0
for p in np.linspace(1.1,4,20):
    ToDo.append((p,f2bt,testNumber))
    testNumber += 1


@pytest.mark.parametrize("p,process,testNumber",ToDo)
def test_other_p(p,process,testNumber):
    process.setStep(1.0)
    gamma = 1e0
    projSplit = ps.ProjSplitFit(gamma)
    m = 40
    d = 20
    if getNewOptVals:
        A = np.random.normal(0,1,[m,d])
        y = np.random.normal(0,1,m)
        cache_otherLosses[(testNumber,'A')] = A
        cache_otherLosses[(testNumber,'y')] = y
    else:
        A = cache_otherLosses[(testNumber,'A')]
        y = cache_otherLosses[(testNumber,'y')]

    projSplit.addData(A,y,p,process,normalize=False,intercept=False)

    lam = 0.01
    regObj = L1(lam,1.0)
    projSplit.addRegularizer(regObj)

    projSplit.run(primalTol=1e-3,dualTol=1e-3,keepHistory = True,nblocks=2,
                  maxIterations=1000,historyFreq=1)


    ps_val = projSplit.getObjective()

    #ps_vals = projSplit.getHistory()[0]
    #plt.plot(ps_vals)
    #plt.show()

    if getNewOptVals:
        x_cvx = cvx.Variable(d)
        f = (1/(m*p))*cvx.pnorm(A @ x_cvx - y,p)**p
        f += lam * cvx.norm(x_cvx, 1)
        prob = cvx.Problem(cvx.Minimize(f))
        prob.solve(verbose=True)
        opt = prob.value
        cache_otherLosses[(testNumber,'opt')] = opt
    else:
        opt = cache_otherLosses[(testNumber,'opt')]


    print("ps val  = {}".format(ps_val))
    print("cvx val  = {}".format(opt))

    assert abs(ps_val - opt) < 1e-2


def test_writeCache2Disk():
    if getNewOptVals:
        with open('results/cache_otherLosses','wb') as file:
            pickle.dump(cache_otherLosses,file)
