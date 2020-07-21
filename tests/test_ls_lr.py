# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:16:24 2020
"""

getNewOptVals = False
import sys
sys.path.append('../')
import projSplit as ps 
import lossProcessors as lp

import numpy as np
import pickle
import pytest 
from matplotlib import pyplot as plt



if getNewOptVals:
    from utils import runCVX_LR
    from utils import getLSdata
    cache = {}
else:
    with open('results/cache_lslr','rb') as file:
        cache = pickle.load(file)
        
    



    
    
    
@pytest.mark.parametrize("gf",[(1.0),(1.1),(1.2),(1.5)])
def test_f1backtrack(gf):
    
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    if getNewOptVals and (gf==1.0):
        A = np.random.normal(0,1,[m,d])
        y = np.random.normal(0,1,m)
        cache['Af1bt']=A
        cache['yf1bt']=y
    else:
        A=cache['Af1bt']
        y=cache['yf1bt']
        
        
    processor = lp.Forward1Backtrack(growFactor=gf,growFreq=10)
    
    projSplit.setDualScaling(1e-1)
    projSplit.addData(A,y,2,processor,intercept=True,normalize=True)
    projSplit.run(maxIterations = None,keepHistory = True,
                  primalTol=1e-3,dualTol=1e-3,nblocks=5)
    ps_val = projSplit.getObjective()
    
    if getNewOptVals and (gf==1.0):
        AwithIntercept = np.zeros((m,d+1))
        AwithIntercept[:,0] = np.ones(m)
        AwithIntercept[:,1:(d+1)] = A
        result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
        xhat = result[0]  
        LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
        cache['optf1bt']=LSval
    else:
        LSval=cache['optf1bt']
        
        
        
    
    assert ps_val - LSval < 1e-2
    

@pytest.mark.parametrize("gf",[(1.0),(1.1),(1.2),(1.5)])
def test_f2backtrack(gf):
    
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    if getNewOptVals and (gf==1.0):
        A = np.random.normal(0,1,[m,d])
        y = np.random.normal(0,1,m)
        cache['Af2bt']=A
        cache['yf2bt']=y
    else:
        A=cache['Af2bt']
        y=cache['yf2bt']
    
    processor = lp.Forward2Backtrack(growFactor=gf,growFreq=10)
    
    projSplit.setDualScaling(1e-1)
    projSplit.addData(A,y,2,processor,intercept=True,normalize=True)
    projSplit.run(maxIterations = None,keepHistory = True,
                  primalTol=1e-3,dualTol=1e-3,nblocks=5)
    ps_val = projSplit.getObjective()
    
    
    if getNewOptVals and (gf==1.0):
        AwithIntercept = np.zeros((m,d+1))
        AwithIntercept[:,0] = np.ones(m)
        AwithIntercept[:,1:(d+1)] = A
        result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
        xhat = result[0]  
        LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
        cache['optf2bt']=LSval
    else:
        LSval=cache['optf2bt']
    
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
firsttest = True
for i in [False,True]:
    for j in [False,True]:
        for blk in range(1,5):
            for process in [backLBFGS,f2fixed,f2backtrack,f2affine,f1fixed,f1bt,back_exact,backCG]:
                ToDo.append((process,i,j,blk,firsttest))
                firsttest=False 
        
@pytest.mark.parametrize("processor,inter,norm,nblk,firsttest",ToDo)
def test_ls_PrimDual(processor,inter,norm,nblk,firsttest):
    processor.setStep(1e-1)
    projSplit = ps.ProjSplitFit()
    m = 10
    d = 20
    if getNewOptVals and firsttest:
        A = np.random.normal(0,1,[m,d])
        y = np.random.normal(0,1,m)
        cache['AprimDual']=A
        cache['yprimDual']=y
    else:
        A=cache['AprimDual']
        y=cache['yprimDual']
        
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

@pytest.mark.parametrize("processor,firsttest",[(backLBFGS,True),(f2fixed,False),(f2bt,False),
                                      (f2affine,False),(f1fixed,False),(f1bt,False),
                                      (back_exact,False),(backCG,False)])
def test_cyclic(processor,firsttest):
    processor.setStep(5e-1)
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    if getNewOptVals and firsttest:
        A = np.random.normal(0,1,[m,d])
        y = np.random.normal(0,1,m)
        cache['Acyclic']=A
        cache['ycyclic']=y
    else:
        A=cache['Acyclic']
        y=cache['ycyclic']
        
    
    if getNewOptVals and firsttest:
        result = np.linalg.lstsq(A,y,rcond=None)
        xhat = result[0]
        LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
        cache['optCyclic']=LSval
    else:
        LSval=cache['optCyclic']
        
        
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
    
    


f2fixed = lp.Forward2Fixed()
f2bt = lp.Forward2Backtrack()
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed(stepsize)
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
backCG = lp.BackwardCG(maxIter=100)
backLBFGS = lp.BackwardLBFGS()
processors = [f2fixed,f2bt,f2affine,f1fixed,f1bt,back_exact,backCG,backLBFGS]
toDo = []
firsttest = True
for inter in [False,True]:
    for norm in [False,True]:
        for process in processors:
            toDo.append((process,norm,inter,firsttest))
            firsttest = False
            

print(toDo)

@pytest.mark.parametrize("processor,norm,inter,firsttest",toDo)
def test_ls_Int_Norm(processor,norm,inter,firsttest):
    processor.setStep(5e-1)
    projSplit = ps.ProjSplitFit()
    m = 20
    d = 10
    print(f"firsttest = {firsttest}")
    print(f"getNewOptVals={getNewOptVals}")
    if getNewOptVals and firsttest:
        A = np.random.normal(0,1,[m,d])
        y = np.random.normal(0,1,m)
        cache['AlsintNorm']=A
        cache['ylsintNorm']=y
    else:
        A=cache['AlsintNorm']
        y=cache['ylsintNorm']
        
    gamma = 1e-2
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=norm,intercept=inter)    
    projSplit.run(maxIterations = 5000,keepHistory = True,nblocks = 10,primalTol=0.0,
                  dualTol=0.0)
    ps_sol = projSplit.getSolution()
    
    if getNewOptVals:
        LSval = cache.get((inter,norm,'lsIntNormOpt'))
        if LSval is None:
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
            
            
            if inter:
                LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
            else:
                LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
            cache[(inter,norm,'lsIntNormOpt')]=LSval
            
    else:
        LSval = cache.get((inter,norm,'lsIntNormOpt'))
    
    psSolVal = projSplit.getObjective()
            
    assert(abs(psSolVal - LSval) < 1e-2)
    
    

f2fixed = lp.Forward2Fixed()
f2bt = lp.Forward2Backtrack()
f2affine = lp.Forward2Affine()
f1fixed = lp.Forward1Fixed()
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
    if getNewOptVals:
        A = cache.get('AlsBlocks')
        y = cache.get('ylsBlocks')
        if A is None:
            A = np.random.normal(0,1,[m,d])
            y = np.random.normal(0,1,m)
        cache['AlsBlocks']=A
        cache['ylsBlocks']=y
    else:
        A = cache.get('AlsBlocks')
        y = cache.get('ylsBlocks')
        
        
            
    gamma = 1e-1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False)    
    projSplit.run(maxIterations = 1000,keepHistory = True,nblocks = 10)
    assert projSplit.getObjective() >= 0, "objective is not >= 0"
    sol = projSplit.getSolution()
    assert sol.shape == (d+1,)
    
    if getNewOptVals:
        LSresid = cache.get('optlsblocks')
        if LSresid is None:
            AwithIntercept = np.zeros((m,d+1))
            AwithIntercept[:,0] = np.ones(m)
            AwithIntercept[:,1:(d+1)] = A
            result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
            xhat = result[0]    
            
            LSresid = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
            cache['optlsblocks']=LSresid
    else:
        LSresid = cache.get('optlsblocks')
            
    
    PSresid = projSplit.getObjective()    
    assert abs(LSresid - PSresid) <1e-2
    
    
    
    projSplit.run(maxIterations = 1000,keepHistory = True,resetIterate=True,blockActivation="random",nblocks = 10)
    PSrandom = projSplit.getObjective()
    assert abs(PSresid - PSrandom)<1e-2
    
    projSplit.run(maxIterations = 1000,keepHistory = True,resetIterate=True,blockActivation="cyclic",nblocks = 10)    
    PScyclic = projSplit.getObjective()
    assert abs(PSresid  - PScyclic)<1e-2


f2fixed = lp.Forward2Fixed()
f2bt = lp.Forward2Backtrack()
f1fixed = lp.Forward1Fixed()
f1bt = lp.Forward1Backtrack()
backLBFGS = lp.BackwardLBFGS()
processors = [f2fixed,f2bt,f1fixed,f1bt,backLBFGS]

toDo = []
for norm in [False,True]:
    for inter in [False,True]:
        for process in processors:
            toDo.append((process,norm,inter))
            
@pytest.mark.parametrize("processor,norm,inter",toDo) 
def test_lr(processor,norm,inter):
    processor.setStep(1e0)
    projSplit = ps.ProjSplitFit()
    m = 50
    d = 10
    if getNewOptVals:
        A = cache.get('Alr')
        y = cache.get('ylr')
        if A is None:
            A = np.random.normal(0,1,[m,d])
            y = 2.0*(np.random.normal(0,1,m)>0)-1.0
            cache['Alr']=A
            cache['ylr']=y
    else:
        A = cache.get('Alr')
        y = cache.get('ylr')
        
    lam = 0.0
    gamma = 1e-4
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,'logistic',processor,normalize=norm,intercept=inter)    
    projSplit.run(maxIterations = 3000,keepHistory = True,nblocks = 10)    
    ps_opt_val = projSplit.getObjective()
    
    if getNewOptVals:
        opt = cache.get((inter,'optlr'))
        if opt is None:
            if inter:
                 AwithIntercept = np.zeros((m,d+1))
                 AwithIntercept[:,0] = np.ones(m)
                 AwithIntercept[:,1:(d+1)] = A
                 A = AwithIntercept 
            
                
            opt, _ = runCVX_LR(A,y,lam,intercept=inter)
            cache[(inter,'optlr')]=opt
    else:
        opt = cache.get((inter,'optlr'))
        
            
    print("ps opt is {}".format(ps_opt_val))
    print("cvx opt is {}".format(opt))
        
    assert abs(opt - ps_opt_val)<1e-2
    

f2bt = lp.Forward2Backtrack()
f2fixed = lp.Forward2Fixed(1e-1)
f1fixed = lp.Forward1Fixed()
f1bt = lp.Forward1Backtrack()
back_exact = lp.BackwardExact()
@pytest.mark.parametrize("processor",[(f2fixed),(f2bt),(f1fixed),(f1bt),(back_exact)]) 
def test_blockIs1bug(processor):
    m = 40
    d = 10
    if getNewOptVals:
        A = cache.get('AblockBug')
        y = cache.get('yblockBug')
        if A is None:
            
            A,y = getLSdata(m,d)    
            cache['AblockBug']=A
            cache['yblockBug']=y
    else:
        A = cache.get('AblockBug')
        y = cache.get('yblockBug')
        
    projSplit = ps.ProjSplitFit()
    
    gamma = 1e1
    projSplit.setDualScaling(gamma)
    projSplit.addData(A,y,2,processor,normalize=False,intercept=False)
    
    projSplit.run(maxIterations=1000,keepHistory = True, nblocks = 1,blockActivation="random")     
    
    ps_opt = projSplit.getObjective()
    print('ps func opt = {}'.format(ps_opt))
    
    if getNewOptVals:
        LSval = cache.get('optBug')
        if LSval is None:
            result = np.linalg.lstsq(A,y,rcond=None)
            xhat = result[0]
            LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
            cache['optBug']=LSval
    else:
        LSval = cache.get('optBug')
            
            
    print('LSval = {}'.format(LSval))
    assert abs(LSval - ps_opt)<1e-2


back_exact = lp.BackwardExact()
backCG = lp.BackwardCG()
backLBFGS = lp.BackwardLBFGS()

ToDo = []
for nblk in [1,2,10]:
    for inter in [False,True]:
        for norm in [False,True]:
            for processor in [back_exact,backCG,backLBFGS]:
                ToDo.append((nblk,inter,norm,processor))
                
@pytest.mark.parametrize("nblk,inter,norm,processor",ToDo) 
def test_backward(nblk,inter,norm,processor):
    m = 80
    d = 20
    if getNewOptVals:
        A = cache.get('Aback')
        y = cache.get('yback')
        if A is None:
            
            A,y = getLSdata(m,d)    
            cache['Aback']=A
            cache['yback']=y
    else:
        A = cache.get('Aback')
        y = cache.get('yback')
        
    projSplit = ps.ProjSplitFit()        
    gamma = 1e-3
    #if nblk==10:
    #    gamma = 1e3
    projSplit.setDualScaling(gamma)    
    projSplit.addData(A,y,2,processor,normalize=norm,intercept=inter)
    
    projSplit.run(maxIterations=5000,keepHistory = True, nblocks = nblk,blockActivation="random",
                  primalTol = 1e-6,dualTol = 1e-6)     
    
    #psvals = projSplit.getHistory()[0]
    #plt.plot(psvals)
    #plt.show()
    ps_opt = projSplit.getObjective()
    print('ps func opt = {}'.format(ps_opt))
    
    if getNewOptVals:
        LSval = cache.get((inter,'optback'))
        if LSval is None:
            if inter:
                AwithIntercept = np.zeros((m,d+1))
                AwithIntercept[:,0] = np.ones(m)
                AwithIntercept[:,1:(d+1)] = A
                result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
                xhat = result[0]  
                LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
            else:
                result = np.linalg.lstsq(A,y,rcond=None)
                xhat = result[0]
                LSval = 0.5*np.linalg.norm(A.dot(xhat)-y,2)**2/m
            cache[(inter,'optback')]=LSval
    else:
        LSval = cache.get((inter,'optback'))
                        
    print('LSval = {}'.format(LSval))
    
    assert ps_opt - LSval <1e-2


def test_writeCache2Disk():
    if getNewOptVals:
        with open('results/cache_lslr','wb') as file:
            pickle.dump(cache,file)
            


    
    
    
    

    
