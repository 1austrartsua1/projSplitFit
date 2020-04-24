# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:12:34 2020

@author: pjohn
"""

# testing projSplit
# test-0
import sys
sys.path.append('/home/pj222/gitFolders/projSplitFit')
import projSplit as ps 
import pytest 
#--------------------------------------------------------------------
# Create a ProjSplitFit object
def test_getGamma1():    
    projSplit = ps.ProjSplitFit()
    gamma = projSplit.getDualScaling()
    assert gamma == 1.0, "failed, gamma != 1"
    
#--------------------------------------------------------------------
# Create a ProjSplitFit object with nondefault good gamma 
def test_getGamma2():
    gammain = 11.5    
    projSplit = ps.ProjSplitFit(gammain)
    gamma = projSplit.getDualScaling()
    assert gamma == gammain, "failed, gamma != gammain"
    
    
#--------------------------------------------------------------------    
# bad Gammas
def test_bad_gamma():
    gammain = 0.0
    projSplit = ps.ProjSplitFit(gammain)
    gamma = projSplit.getDualScaling()
    assert gamma == 1.0,"failed, gamma != 1.0"
    
    gammain = -10
    projSplit = ps.ProjSplitFit(gammain)
    gamma = projSplit.getDualScaling()
    assert gamma == 1.0,"failed, gamma != 1.0"

    
    gammain = "hello world"
    projSplit = ps.ProjSplitFit(gammain)
    gamma = projSplit.getDualScaling()
    assert gamma == 1.0,"failed, gamma != 1.0"

    gammain = [1,2]
    projSplit = ps.ProjSplitFit(gammain)
    gamma = projSplit.getDualScaling()
    assert gamma == 1.0,"failed, gamma != 1.0"


#--------------------------------------------------------------------
# bad get things before run/initialization etc
def test_bad_getParams():
    projSplit = ps.ProjSplitFit()
    testing = projSplit.getParams()
    assert testing == None, "output of getParams is not None"

    testing = projSplit.getObjective()
    assert testing == None, "output of getObjective is not None"

    testing = projSplit.getSolution()
    assert testing == None, "output of getSolution is not None"

    testing = projSplit.getDualViolation()
    assert testing == None, "output of getDualViolation is not None"

    testing = projSplit.getHistory()
    assert testing == None, "output of getHistory is not None"

    testing = projSplit.getPrimalViolation()
    assert testing == None, "output of getPrimalViolation is not None"
    
    


    
    
    