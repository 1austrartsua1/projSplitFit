# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:31:38 2020
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
from scipy.sparse.linalg import aslinearoperator
from matplotlib import pyplot as plt

if getNewOptVals:
    import cvxpy as cvx
    from utils import getLSdata
    cache = {}
else:
    np.random.seed(1)
    with open('results/cache_linear_ops','rb') as file:
        cache = pickle.load(file)

toDo = []
for norm in [False,True]:
    for inter in [False,True]:
        toDo.append((norm,inter))


@pytest.mark.parametrize("norm,inter",toDo)
def test_linear_op_l1(norm,inter):


    m = 40
    d = 10
    p = 15
    if getNewOptVals:
        A = cache.get('AlinL1')
        y = cache.get('ylinL1')
        H = cache.get('HlinL1')
        if A is None:
            A,y = getLSdata(m,d)
            H = np.random.normal(0,1,[p,d])
            cache['AlinL1']=A
            cache['ylinL1']=y
            cache['HlinL1']=H
    else:
        A=cache['AlinL1']
        y=cache['ylinL1']
        H=cache['HlinL1']


    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=norm,intercept=inter)


    lam = 0.01
    step = 1.0
    regObj = L1(lam,step)
    projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H))
    projSplit.run(maxIterations=5000,keepHistory = True, nblocks = 1,
                  primalTol=1e-3,dualTol=1e-3)
    ps_val = projSplit.getObjective()



    if getNewOptVals:
        opt = cache.get((norm,inter,'optlinL1'))
        if opt is None:
            (m,d) = A.shape
            if norm:
                Anorm = A
                scaling = np.linalg.norm(Anorm,axis=0)
                scaling += 1.0*(scaling < 1e-10)
                Anorm = np.sqrt(m)*Anorm/scaling
                A = Anorm
            if inter:
                AwithIntercept = np.zeros((m,d+1))
                AwithIntercept[:,0] = np.ones(m)
                AwithIntercept[:,1:(d+1)] = A
                A = AwithIntercept

                HwithIntercept = np.zeros((p,d+1))
                HwithIntercept[:,0] = np.zeros(p)
                HwithIntercept[:,1:(d+1)] = H
                H = HwithIntercept
                x_cvx = cvx.Variable(d+1)

            else:
                x_cvx = cvx.Variable(d)

            f = (1/(2*m))*cvx.sum_squares(A@x_cvx - y)
            f += lam*cvx.norm(H @ x_cvx,1)
            prob = cvx.Problem(cvx.Minimize(f))
            prob.solve(verbose=True)
            opt = prob.value
            cache[(norm,inter,'optlinL1')]=opt


    else:
        opt=cache[(norm,inter,'optlinL1')]


    primViol = projSplit.getPrimalViolation()
    dualViol = projSplit.getDualViolation()
    print("primal violation = {}".format(primViol))
    print("dual violation = {}".format(dualViol))

    print("ps val = {}".format(ps_val))
    print("cvx val = {}".format(opt))
    assert ps_val - opt < 1e-2



Todo = [(False,False,0,1),(True,False,1,1),
        (False,True,2,1),(True,True,3,1)]
Todo.extend([(False,False,4,9),(True,False,5,9),
        (False,True,6,9),(True,True,7,9)])
@pytest.mark.parametrize("norm,inter,testNumber,numblocks",Todo)
def test_multi_linear_op_l1(norm,inter,testNumber,numblocks):


    m = 40
    d = 10
    numregs = 5
    if getNewOptVals and (testNumber==0):
        A,y = getLSdata(m,d)
        cache['AmutliLinL1']=A
        cache['ymutliLinL1']=y
        H = []
        for i in range(numregs):
            p = np.random.randint(1,100)
            H.append(np.random.normal(0,1,[p,d]))

        cache['HmultiLinL1']=H
    else:
        H=cache['HmultiLinL1']
        A=cache['AmutliLinL1']
        y=cache['ymutliLinL1']


    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    gamma = 1e0
    if norm and inter:
        gamma = 1e2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=norm,intercept=inter)


    lam = []
    for i in range(numregs):
        lam.append(0.001*(i+1))
        step = 1.0
        regObj = L1(lam[-1],step)
        projSplit.addRegularizer(regObj,linearOp = aslinearoperator(H[i]))

    projSplit.run(maxIterations=5000,keepHistory = True, nblocks = numblocks,
                  primalTol=1e-6,dualTol=1e-6)
    ps_val = projSplit.getObjective()

    if getNewOptVals:
        if norm:
            Anorm = A
            m = Anorm.shape[0]
            scaling = np.linalg.norm(Anorm,axis=0)
            scaling += 1.0*(scaling < 1e-10)
            Anorm = np.sqrt(m)*Anorm/scaling
            A = Anorm

        if inter:
            AwithIntercept = np.zeros((m,d+1))
            AwithIntercept[:,0] = np.ones(m)
            AwithIntercept[:,1:(d+1)] = A
            A = AwithIntercept


        (m,d) = A.shape
        x_cvx = cvx.Variable(d)
        f = (1/(2*m))*cvx.sum_squares(A@x_cvx - y)
        for i in range(numregs):
            if inter:
                f += lam[i]*cvx.norm(H[i] @ x_cvx[1:d],1)
            else:
                f += lam[i]*cvx.norm(H[i] @ x_cvx,1)
        prob = cvx.Problem(cvx.Minimize(f))
        prob.solve(verbose=True)
        opt = prob.value
        cache[(norm,inter,'opt')]=opt
    else:
        opt=cache[(norm,inter,'opt')]


    print("ps val = {}".format(ps_val))
    print("cvx val = {}".format(opt))


    assert ps_val - opt < 1e-2




def test_linear_op_data_term_wrong():
    m = 40
    d = 10
    if getNewOptVals:
        A,y = getLSdata(m,d)
        cache['Awrongdata']=A
        cache['ywrongdata']=y
    else:
        A=cache['Awrongdata']
        y=cache['ywrongdata']



    projSplit = ps.ProjSplitFit()
    stepsize = 1e-1
    processor = lp.Forward2Fixed(stepsize)
    gamma = 1e0
    projSplit.setDualScaling(gamma)
    p = 15
    d2 = 11
    H = np.random.normal(0,1,[d2,p])
    try:
        projSplit.addData(A,y,2,processor,normalize=False,intercept=False,
                      linearOp = aslinearoperator(H))
        notExcept = True

    except:
        notExcept = False

    assert notExcept == False


f2fix = lp.Forward2Fixed()
back2exact = lp.BackwardExact()
backCG = lp.BackwardCG()
f1bt = lp.Forward1Backtrack()
backLB = lp.BackwardLBFGS()
TryAll = []
testNumber = 0
for i in [False,True]:
    for j in [False,True]:
        for k in [False,True]:
            for l in [False,True]:
                for p in [backLB,f2fix,back2exact,f1bt,backCG]:
                    TryAll.append((i,j,k,l,p,testNumber))
                    testNumber += 1

@pytest.mark.parametrize("norm,inter,addL1,add2L1,processor,testNumber",TryAll)
def test_linear_op_data_term(norm,inter,addL1,add2L1,processor,testNumber):


    m = 40
    d = 10
    p = 15
    d2 = 10

    if getNewOptVals and (testNumber==0):

        A,y = getLSdata(m,d)
        H = np.random.normal(0,1,[d2,p])
        cache['AdataTerm']=A
        cache['ydataTerm']=y
        cache['HdataTerm']=H
    else:
        A = cache['AdataTerm']
        y = cache['ydataTerm']
        H = cache['HdataTerm']


    projSplit = ps.ProjSplitFit()

    processor.setStep(5e-1)
    gamma = 1e0
    projSplit.setDualScaling(gamma)




    projSplit.addData(A,y,2,processor,normalize=norm,intercept=inter,
                      linearOp = aslinearoperator(H))

    lam = 0.01
    step = 1.0
    if addL1:
        regObj = L1(lam,step)
        projSplit.addRegularizer(regObj)

    if add2L1:
        regObj2 = L1(lam,step)
        projSplit.addRegularizer(regObj2)

    projSplit.run(maxIterations=10000,keepHistory = True,
                  nblocks = 3,primalTol=1e-3,dualTol=1e-3)
    ps_val = projSplit.getObjective()

    primViol = projSplit.getPrimalViolation()
    dualViol = projSplit.getDualViolation()
    print("primal violation = {}".format(primViol))
    print("dual violation = {}".format(dualViol))



    if getNewOptVals:

        opt = cache.get((addL1,add2L1,inter,norm,'optdata'))

        if opt == None:

            if norm == True:
                scaling = np.linalg.norm(A,axis=0)
                scaling += 1.0*(scaling < 1e-10)
                A = np.sqrt(A.shape[0])*A/scaling
            if inter == True:
                AwithIntercept = np.zeros((m,d+1))
                AwithIntercept[:,0] = np.ones(m)
                AwithIntercept[:,1:(d+1)] = A
                A = AwithIntercept
                HwithIntercept = np.zeros((d2+1,p+1))
                HwithIntercept[:,0] = np.zeros(d2+1)
                HwithIntercept[0] = np.ones(p+1)
                HwithIntercept[0,0] = 1.0
                HwithIntercept[1:(d2+1),1:(p+1)] = H
                H = HwithIntercept


            (m,_) = A.shape
            if inter:
                x_cvx = cvx.Variable(p+1)
            else:
                x_cvx = cvx.Variable(p)

            f = (1/(2*m))*cvx.sum_squares(A@H@x_cvx - y)
            if addL1:
                f += lam*cvx.norm(x_cvx,1)

            if add2L1:
                f += lam*cvx.norm(x_cvx,1)

            prob = cvx.Problem(cvx.Minimize(f))
            prob.solve(verbose=True)
            opt = prob.value
            cache[(addL1,add2L1,inter,norm,'optdata')]=opt

    else:
        opt=cache[(addL1,add2L1,inter,norm,'optdata')]


    print("ps opt = {}".format(ps_val))
    print("cvx opt = {}".format(opt))
    assert(ps_val-opt<1e-2)










def test_writeCache2Disk():
    if getNewOptVals:
        with open('results/cache_linear_ops','wb') as file:
            pickle.dump(cache,file)
