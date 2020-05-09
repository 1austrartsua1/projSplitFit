# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:51:55 2020

@author: pjohn
"""

import sys
sys.path.append('/home/pj222/gitFolders/projSplitFit')
import projSplit as ps 
import numpy as np
import pytest 

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
    processDummy = None
    projSplit.addData(A,y,2,processDummy)

    nvar,nobs = projSplit.getParams()
    assert (nvar==d+1) ,"test failed, nvar!=d+1"
    assert (nobs == m), "test failed, nobs != m"    
        

def test_L1():
    #projSplit = ps.ProjSplitFit()
    scale = 15.0
    regObj = ps.L1(scale)
    assert regObj.getScalingAndStepsize()==(scale,1.0)
    scale = -1.0
    regObj = ps.L1(scale)
    assert regObj.getScalingAndStepsize()==(1.0,1.0)
    
    regObj = ps.L1(scale)
    assert regObj.getScalingAndStepsize()==(1.0,1.0)
    
    scale = 11.5
    rho = 3.0
    regObj = ps.L1(scale,rho)
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
    regObj = ps.L1(scale)
    projSplit.addRegularizer(regObj)
    scale2 = 15.7
    regObj.setScaling(scale2)
    assert (projSplit.allRegularizers[0].getScalingAndStepsize()==(scale2,1.0))

def test_add_regularizer2():
    projSplit = ps.ProjSplitFit()    
    scale = 11.5
    regObj = ps.L1(scale)
    projSplit.addRegularizer(regObj,embed = True)
    scale2 = 15.7
    regObj.setScaling(scale2)
    assert (projSplit.embedded.getScalingAndStepsize()==(scale2,1.0))













