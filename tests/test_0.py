# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:12:34 2020

@author: pjohn
"""

# testing projSplit
# test-0
import sys
sys.path.append('../')
import projSplitFit as ps

import pytest
import numpy as np
#--------------------------------------------------------------------
# Create a ProjSplitFit object
def test_getGamma1():
    projSplit = ps.ProjSplitFit()
    gamma = projSplit.getDualScaling()
    assert gamma == 1.0, "failed, gamma != 1"

#--------------------------------------------------------------------
# Create a ProjSplitFit object with nondefault good gamma
gamma2test = [1,10,100,1000,1e4,1e6,1e10,1e15,1e-5,1e-11]
@pytest.mark.parametrize("gammain",gamma2test)
def test_getGamma2(gammain):
    projSplit = ps.ProjSplitFit(gammain)
    gamma = projSplit.getDualScaling()
    assert gamma == gammain, "failed, gamma != gammain"


#--------------------------------------------------------------------
# bad Gammas
gamma2test = [0.0,-10,"hello world",[1,2]]
@pytest.mark.parametrize("gammain",gamma2test)
def test_bad_gamma(gammain):
    projSplit = ps.ProjSplitFit(gammain)
    gamma = projSplit.getDualScaling()
    assert gamma == 1.0,"failed, gamma != 1.0"



#--------------------------------------------------------------------
# bad get things before run/initialization etc
def test_bad_getParams():
    projSplit = ps.ProjSplitFit()
    try:
        testing = projSplit.numPrimalVars()
        testing = projSplit.numObservations()


        testing = projSplit.getObjective()


        testing = projSplit.getSolution()


        testing = projSplit.getDualViolation()


        testing = projSplit.getHistory()


        testing = projSplit.getPrimalViolation()

        testing = projSplit.getScale()
        noExcept = True
    except:
        noExcept = False

    assert noExcept == False

ToDo = [([1]),([1,1,1]),([1,1,1,1,1,1,1]),([])]
@pytest.mark.parametrize("y",ToDo)
def test_bad_dims(y):
    psObj = ps.ProjSplitFit()
    obs = np.array([[1,2,3],[4,5,6]])
    with pytest.raises(Exception):
        psObj.addData(obs,y,2)
