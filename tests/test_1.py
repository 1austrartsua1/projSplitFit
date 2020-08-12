# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:51:55 2020

@author: pjohn
"""

import sys
sys.path.append('../')
import projSplitFit as ps
from regularizers import L1
import lossProcessors as lp
import numpy as np
import pytest
from scipy.sparse.linalg import aslinearoperator

class ProcessDummy(lp.LossProcessor):
        def __init__(self):
            self.embedOK = True


# createApartition test

#print(ps.createApartition(100,10))
#print(ps.createApartition(5,10))

# addData test


# getParams test
def test_getParams():
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    processDummy = ProcessDummy()
    projSplit.addData(A,y,2,processDummy)

    nvar = projSplit.numPrimalVars()
    nobs = projSplit.numObservations()
    assert (nvar==d+1) ,"test failed, nvar!=d+1"
    assert (nobs == m), "test failed, nobs != m"


def test_L1():
    #projSplit = ps.ProjSplitFit()
    scale = 15.0
    regObj = L1(scale)
    assert regObj.getScaling()==scale
    scale = -1.0
    regObj = L1(scale)
    assert (regObj.getScaling(),regObj.getStep())==(1.0,1.0)

    regObj = L1(scale)
    assert (regObj.getScaling(),regObj.getStep())==(1.0,1.0)

    scale = 11.5
    rho = 3.0
    regObj = L1(scale,rho)
    lenx = 10
    x = np.ones(lenx)
    assert regObj.evaluate(x) == lenx*scale


    toTest = regObj.getProx(x)
    assert toTest.shape == (lenx,)
    diff = toTest - np.zeros(lenx)
    assert (diff == 0.0).all()

def test_add_regularizer():
    projSplit = ps.ProjSplitFit()
    scale = 11.5
    regObj = L1(scale)
    projSplit.addRegularizer(regObj)
    scale2 = 15.7
    regObj.setScaling(scale2)
    assert (projSplit.allRegularizers[0].getScaling()==scale2)

# outdated test since we changed embed to be an argument of addData
#def test_add_regularizer2():
#    projSplit = ps.ProjSplitFit()
#    scale = 11.5
#    regObj = L1(scale)
#    projSplit.addRegularizer(regObj,embed = True)
#    scale2 = 15.7
#    regObj.setScaling(scale2)
#    assert (projSplit.embedded.getScaling()==scale2)


def test_add_linear_ops():
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    processDummy = ProcessDummy()
    projSplit.addData(A,y,2,processDummy)

    p = 11
    H = np.random.normal(0,1,[p,d])
    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)

    projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H))

    d2 = 9
    H = np.random.normal(0,1,[p,d2])
    try:
        projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H)) == - 1
        noExcept = True
    except:
        noExcept = False

    assert noExcept == False


def test_add_linear_ops_v2():
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    processDummy = ProcessDummy()
    d2 = 9
    p = 15
    H = np.random.normal(0,1,[p,d2])
    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H))
    try:
        projSplit.addData(A,y,2,processDummy)==-1
        noExcept = True
    except:
        noExcept = False
    assert noExcept == False


def test_good_embed():
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    processDummy = ProcessDummy()
    regObj = L1()
    projSplit.addData(A,y,2,processDummy,embed=regObj)
    assert projSplit.numRegs == 0


def test_bad_embed():
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    processDummy = ProcessDummy()
    processDummy.embedOK = False
    regObj = L1()
    projSplit.addData(A,y,2,processDummy,embed=regObj)
    assert (projSplit.numRegs == 1)
