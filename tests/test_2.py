# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:16:24 2020

@author: pjohn
"""
import sys
sys.path.append('/home/pj222/gitFolders/projSplitFit')
import projSplit as ps 
import numpy as np
import pytest 
from matplotlib import pyplot as plt


def test_ls_fixed():
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    stepsize = 1.0
    processor = ps.Forward2Fixed(stepsize)
    projSplit.addData(A,y,2,processor)
    projSplit.run(maxIterations = 100,keepHistory = True)

def test_ls_blocks():
    
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    stepsize = 5e-1
    processor = ps.Forward2Fixed(stepsize)
    gamma = 1e-1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=True)    
    projSplit.run(maxIterations = 1000,keepHistory = True,nblocks = 10)
    assert projSplit.getObjective() >= 0, "objective is not >= 0"
    sol = projSplit.getSolution()
    assert sol[1].shape == (d,)
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.zeros(m)
    AwithIntercept[:,1:(d+1)] = A
    result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
    xhat = result[0]    
    print("LS Soln error = {}".format(np.linalg.norm(AwithIntercept.dot(xhat)-y)**2/m))
    print("PS with greedy block selection final objective = {}".format(projSplit.getObjective()))
    funcVals = projSplit.getHistory()[0]
    #plt.semilogy(funcVals)
    #plt.show()
    
    projSplit.run(maxIterations = 1000,keepHistory = True,resetIterate=True,blockActivation="random",nblocks = 10)
    print("PS with random block selection final objective = {}".format(projSplit.getObjective()))
    funcVals = projSplit.getHistory()[0]
    #plt.semilogy(funcVals)
    #plt.show()

    projSplit.run(maxIterations = 1000,keepHistory = True,resetIterate=True,blockActivation="cyclic",nblocks = 10)
    print("PS with cyclic block selection final objective = {}".format(projSplit.getObjective()))
    funcVals = projSplit.getHistory()[0]
    #plt.semilogy(funcVals)
    #plt.show()

if __name__ == "__main__":
    test_ls_blocks()
    