# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:16:24 2020

@author: pjohn
"""
import sys
sys.path.append('../')
import projSplit as ps 
import lossProcessors as lp

import numpy as np
import pytest 
from matplotlib import pyplot as plt
from utils import runCVX_LR
from utils import getLSdata

#np.random.seed(1987)


    
    
    
@pytest.mark.parametrize("gf",[(1.0),(1.1),(1.2),(1.5)])
def test_f1backtrack(gf):
    
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    processor = lp.Forward1Backtrack(growFactor=gf,growFreq=10)
    
    projSplit.setDualScaling(1e-1)
    projSplit.addData(A,y,2,processor,intercept=True,normalize=True)
    projSplit.run(maxIterations = None,keepHistory = True,
                  primalTol=1e-3,dualTol=1e-3,nblocks=5)
    ps_val = projSplit.getObjective()
    
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
    xhat = result[0]  
    LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
    
    assert ps_val - LSval < 1e-2
    

@pytest.mark.parametrize("gf",[(1.0),(1.1),(1.2),(1.5)])
def test_f2backtrack(gf):
    
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    processor = lp.Forward2Backtrack(growFactor=gf,growFreq=10)
    
    projSplit.setDualScaling(1e-1)
    projSplit.addData(A,y,2,processor,intercept=True,normalize=True)
    projSplit.run(maxIterations = None,keepHistory = True,
                  primalTol=1e-3,dualTol=1e-3,nblocks=5)
    ps_val = projSplit.getObjective()
    
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
    xhat = result[0]  
    LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
    
    assert ps_val - LSval < 1e-2
    

stepsize = 1e-1
f2fixed = lp.Forward2Fixed(stepsize)
f2backtrack = lp.Forward2Backtrack()
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
backCG = lp.BackwardCG()
backLBFGS = lp.BackwardLBFGS()
ToDo = []
for i in [False,True]:
    for j in [False,True]:
        for blk in range(1,5):
            for process in [backLBFGS,f2fixed,f2backtrack,f2affine,f1fixed,f1bt,back_exact,backCG]:
                ToDo.append((process,i,j,blk))
        
@pytest.mark.parametrize("processor,inter,norm,nblk",ToDo)
def test_ls_PrimDual(processor,inter,norm,nblk):
    processor.setStep(1e-1)
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    projSplit.setDualScaling(1e-1)
    projSplit.addData(A,y,2,processor,intercept=inter,normalize=norm)
    projSplit.run(maxIterations = None,keepHistory = True,
                  primalTol=1e-3,dualTol=1e-3,nblocks=nblk,historyFreq=1)
    
    print("Primal violation = {}".format(projSplit.getPrimalViolation()))
    print("Dual violation = {}".format(projSplit.getDualViolation()))    
    
    assert projSplit.getPrimalViolation()<1e-3
    assert projSplit.getDualViolation()<1e-3

stepsize = 5e-1
f2fixed = lp.Forward2Fixed(stepsize)
f2bt = lp.Forward2Backtrack()
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
backCG = lp.BackwardCG(maxIter=100)
backLBFGS = lp.BackwardLBFGS()

@pytest.mark.parametrize("processor",[(backLBFGS),(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt),(back_exact),(backCG)])
def test_cyclic(processor):
    processor.setStep(5e-1)
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    
    result = np.linalg.lstsq(A,y,rcond=None)
    xhat = result[0]
    LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
    gamma = 1e-1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)

    #projSplit.addRegularizer(regObj)
    projSplit.run(maxIterations=2000,keepHistory = True, nblocks = 5,blockActivation="cyclic")     
    #fps = projSplit.getHistory()[0]
    #plt.plot(fps)
    #plt.show()
    
    ps_opt = projSplit.getObjective()
    print("PS opt = {}".format(ps_opt))
    print("LS opt = {}".format(LSval))
    assert abs(ps_opt - LSval)<1e-2
    
    
    
    for blks in range(2,7):   
        projSplit.run(maxIterations=2000,keepHistory = True, nblocks = 5,
                  blockActivation="cyclic",resetIterate=True, blocksPerIteration=blks)

        ps_opt = projSplit.getObjective()
        assert abs(ps_opt - LSval)<1e-2
    
    

stepsize = 5e-1
f2fixed = lp.Forward2Fixed(stepsize)
f2bt = lp.Forward2Backtrack()
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
backCG = lp.BackwardCG(maxIter=100)
backLBFGS = lp.BackwardLBFGS()
toDo = [(f2fixed,False,False),(f2fixed,True,False),
        (f2fixed,False,True),(f2fixed,True,True)]
toDo.extend([(backLBFGS,False,False),(backLBFGS,True,False),
        (backLBFGS,False,True),(backLBFGS,True,True)])
toDo.extend([(f2bt,False,False),(f2bt,True,False),
        (f2bt,False,True),(f2bt,True,True)])
toDo.extend([(f2affine,False,False),(f2affine,True,False),
        (f2affine,False,True),(f2affine,True,True)])
toDo.extend([(f1fixed,False,False),(f1fixed,True,False),
        (f1fixed,False,True),(f1fixed,True,True)])
toDo.extend([(f1bt,False,False),(f1bt,True,False),
        (f1bt,False,True),(f1bt,True,True)])
toDo.extend([(back_exact,False,False),(back_exact,True,False),
        (back_exact,False,True),(back_exact,True,True)])
toDo.extend([(backCG,False,False),(backCG,True,False),
        (backCG,False,True),(backCG,True,True)])


@pytest.mark.parametrize("processor,norm,inter",toDo)
def test_ls_Int_Norm(processor,norm,inter):
    processor.setStep(5e-1)
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=norm,intercept=inter)    
    projSplit.run(maxIterations = 5000,keepHistory = True,nblocks = 10,primalTol=0.0,
                  dualTol=0.0)
    ps_sol,_ = projSplit.getSolution()
    
    if inter:
        AwithIntercept = np.zeros((m,d+1))
        AwithIntercept[:,0] = np.ones(m)
        AwithIntercept[:,1:(d+1)] = A
        result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
        xhat = result[0]  
    else:
        result = np.linalg.lstsq(A,y,rcond=None)
        xhat = result[0]
    
    if norm == False:
        assert(np.linalg.norm(xhat-ps_sol)<1e-2)
    
    psSolVal = projSplit.getObjective()
    if inter:
        LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
    else:
        LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
        
    assert(abs(psSolVal - LSval) < 1e-2)
    
    
stepsize = 5e-1
f2fixed = lp.Forward2Fixed(stepsize)
f2bt = lp.Forward2Backtrack()
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
backCG = lp.BackwardCG()
backLBFGS = lp.BackwardLBFGS()
@pytest.mark.parametrize("processor",[(backLBFGS),(f2fixed),(f2bt),(f2affine),(f1fixed),(f1bt),(back_exact),(backCG)]) 
def test_ls_blocks(processor):
    processor.setStep(5e-1)
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    gamma = 1e-1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False)    
    projSplit.run(maxIterations = 1000,keepHistory = True,nblocks = 10)
    assert projSplit.getObjective() >= 0, "objective is not >= 0"
    sol,_ = projSplit.getSolution()
    assert sol.shape == (d+1,)
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
    xhat = result[0]    
    
    LSresid = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
    PSresid = projSplit.getObjective()
    PSresid2 = 0.5*np.linalg.norm(AwithIntercept.dot(sol)-y,2)**2/m
    assert abs(LSresid - PSresid) <1e-2
    assert abs(PSresid - PSresid2)<1e-2
    
    
    projSplit.run(maxIterations = 1000,keepHistory = True,resetIterate=True,blockActivation="random",nblocks = 10)
    PSrandom = projSplit.getObjective()
    assert abs(PSresid - PSrandom)<1e-2
    
    projSplit.run(maxIterations = 1000,keepHistory = True,resetIterate=True,blockActivation="cyclic",nblocks = 10)    
    PScyclic = projSplit.getObjective()
    assert abs(PSresid  - PScyclic)<1e-2

stepsize = 1e0
f2fixed = lp.Forward2Fixed(stepsize)
f2bt = lp.Forward2Backtrack()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
backLBFGS = lp.BackwardLBFGS()

toDo = [(backLBFGS,False,False),(backLBFGS,True,False),
        (backLBFGS,False,True),(backLBFGS,True,True)]
toDo.extend([(f2fixed,False,False),(f2fixed,True,False),
        (f2fixed,False,True),(f2fixed,True,True)])
toDo.extend(
        [(f2bt,False,False),(f2bt,True,False),
        (f2bt,False,True),(f2bt,True,True)]
        )
toDo.extend(
        [(f1fixed,False,False),(f1fixed,True,False),
        (f1fixed,False,True),(f1fixed,True,True)]
        )
toDo.extend(
        [(f1bt,False,False),(f1bt,True,False),
        (f1bt,False,True),(f1bt,True,True)]
        )


@pytest.mark.parametrize("processor,norm,inter",toDo) 
def test_lr(processor,norm,inter):
    processor.setStep(1e0)
    projSplit = ps.ProjSplitFit()
    m = 50
    d = 10
    A = np.random.normal(0,1,[m,d])
    y = 2.0*(np.random.normal(0,1,m)>0)-1.0
    lam = 0.0
    gamma = 1e-4
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,'logistic',processor,normalize=norm,intercept=inter)    
    projSplit.run(maxIterations = 3000,keepHistory = True,nblocks = 10)    
    ps_opt_val = projSplit.getObjective()
    
    if inter:
         AwithIntercept = np.zeros((m,d+1))
         AwithIntercept[:,0] = np.ones(m)
         AwithIntercept[:,1:(d+1)] = A
         A = AwithIntercept 
    
        
    [opt, xopt] = runCVX_LR(A,y,lam,intercept=inter)
    print("ps opt is {}".format(ps_opt_val))
    print("cvx opt is {}".format(opt))
        
    assert abs(opt - ps_opt_val)<1e-2
    
stepsize = 1e-1
f2bt = lp.Forward2Backtrack()
f2fixed = lp.Forward2Fixed(stepsize)
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f1fixed),(f1bt),(back_exact)]) 
def test_blockIs1bug(processor):
    m = 40
    d = 10
    A,y = getLSdata(m,d)    
    projSplit = ps.ProjSplitFit()
    
    gamma = 1e1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1,blockActivation="random")     
    
    ps_opt = projSplit.getObjective()
    print('ps func opt = {}'.format(ps_opt))
    
    result = np.linalg.lstsq(A,y,rcond=None)
    xhat = result[0]
    LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
    print('LSval = {}'.format(LSval))
    assert abs(LSval - ps_opt)<1e-2


back_exact = lp.BackwardExact()
backCG = lp.BackwardCG(maxIter=100)
backLBFGS = lp.BackwardLBFGS()
ToDo = [(1,False,False,backLBFGS),(1,True,False,backLBFGS),(1,False,True,backLBFGS),(1,True,True,backLBFGS)]
ToDo.extend([(2,False,False,backLBFGS),(2,True,False,backLBFGS),(2,False,True,backLBFGS),(2,True,True,backLBFGS)])
ToDo.extend([(4,False,False,backLBFGS),(4,True,False,backLBFGS),(4,False,True,backLBFGS),(4,True,True,backLBFGS)])
ToDo.extend([(8,False,False,backLBFGS),(8,True,False,backLBFGS),(8,False,True,backLBFGS),(8,True,True,backLBFGS)])
ToDo.extend([(16,False,False,backLBFGS),(16,True,False,backLBFGS),(16,False,True,backLBFGS),(16,True,True,backLBFGS)])

ToDo.extend([(1,False,False,back_exact),(1,True,False,back_exact),(1,False,True,back_exact),(1,True,True,back_exact)])
ToDo.extend([(2,False,False,back_exact),(2,True,False,back_exact),(2,False,True,back_exact),(2,True,True,back_exact)])
ToDo.extend([(4,False,False,back_exact),(4,True,False,back_exact),(4,False,True,back_exact),(4,True,True,back_exact)])
ToDo.extend([(8,False,False,back_exact),(8,True,False,back_exact),(8,False,True,back_exact),(8,True,True,back_exact)])
ToDo.extend([(16,False,False,back_exact),(16,True,False,back_exact),(16,False,True,back_exact),(16,True,True,back_exact)])

ToDo.extend([(1,False,False,backCG),(1,True,False,backCG),(1,False,True,backCG),(1,True,True,backCG)])
ToDo.extend([(2,False,False,backCG),(2,True,False,backCG),(2,False,True,backCG),(2,True,True,backCG)])
ToDo.extend([(4,False,False,backCG),(4,True,False,backCG),(4,False,True,backCG),(4,True,True,backCG)])
ToDo.extend([(8,False,False,backCG),(8,True,False,backCG),(8,False,True,backCG),(8,True,True,backCG)])
ToDo.extend([(16,False,False,backCG),(16,True,False,backCG),(16,False,True,backCG),(16,True,True,backCG)])

@pytest.mark.parametrize("nblk,inter,norm,processor",ToDo) 
def test_backward(nblk,inter,norm,processor):
    m = 80
    d = 20
    A,y = getLSdata(m,d)    
    projSplit = ps.ProjSplitFit()        
    gamma = 1e-3
    #gamma = 1e0
    projSplit.setDualScaling(gamma)    
    projSplit.addData(A,y,2,processor,normalize=norm,intercept=inter)
    
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = nblk,blockActivation="random",
                  primalTol = 0.0,dualTol = 0.0)     
    
    #psvals = projSplit.getHistory()[0]
    #plt.plot(psvals)
    #plt.show()
    ps_opt = projSplit.getObjective()
    print('ps func opt = {}'.format(ps_opt))
    
    result = np.linalg.lstsq(A,y,rcond=None)
    xhat = result[0]
    LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
    print('LSval = {}'.format(LSval))
    
    assert ps_opt - LSval <1e-2

if __name__ == "__main__":    
    
    processor = lp.BackwardLBFGS()
    nblk = 16
    inter = False
    norm = False
    test_backward(nblk,inter,norm,processor)
    
    
    
    

    