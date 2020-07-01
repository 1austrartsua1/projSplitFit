
import sys
sys.path.append('../')
import projSplit as ps 
import lossProcessors as lp

import numpy as np
from matplotlib import pyplot as plt
from utils import runCVX_LR


doplots = True 
BA = "greedy"
processor = lp.Forward1Backtrack(1.0,growFactor=1.2,growFreq=10)
#processor = lp.Forward2Backtrack(1.0,Delta =0.0,growFactor=1.2,growFreq=10)
def testSGD():
    print("Least Squares with SGD and PS")
    gamma = 1e-12
    projSplit = ps.ProjSplitFit(gamma)
    m = 2000
    d = 2000
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    
    stepSGD = 100.0
    iterSGD = 10000
    iterPS = 1000
    blocksSGD = 200
    blocksPS = 20
    freqSGD = 1
    
    processor.setStep(stepSGD)
    projSplit.addData(A,y,2,processor,intercept=True,normalize=True)
    
    
    
    fsgd,timessgd = projSplit.runSGD(iterSGD,blocksSGD,freqSGD,stepSGD)
    
    projSplit.run(maxIterations = iterPS,keepHistory = True,blockActivation = BA,
                  primalTol=1e-9,dualTol=1e-9,nblocks=blocksPS,historyFreq=1)
    ps_val = projSplit.getObjective()
    fps = projSplit.getHistory()[0]
    tps = projSplit.getHistory()[1]
    
    print("Final stepsize is {}".format(processor.getStep()))
    
    getLS = False 
    if getLS:
        AwithIntercept = np.zeros((m,d+1))
        AwithIntercept[:,0] = np.ones(m)
        AwithIntercept[:,1:(d+1)] = A
        result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
        xhat = result[0]  
        LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
        print("ls sol = {}".format(LSval))
    print("fsgd opt = {}".format(fsgd[-1]))
    print("ps opt = {}".format(ps_val))
    
    
    if doplots:
        plt.plot(timessgd,fsgd)
        plt.plot(tps,fps)
        plt.legend(['sgd','ps'])
        plt.show()

def test_sgd_lr():
    print("Logistic regression with SGD and PS")
    gamma = 1e-12    
    projSplit = ps.ProjSplitFit(gamma)
    m = 2000
    d = 2000
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    
    stepSGD = 1000.0
    iterSGD = 10000
    iterPS = 1000
    blocksSGD = 200
    blocksPS = 20
    
    freqSGD = 1
    
    
    processor.setStep(stepSGD)
    projSplit.addData(A,y,'logistic',processor,intercept=True,normalize=True)
    
    
    
    fsgd,timessgd = projSplit.runSGD(iterSGD,blocksSGD,freqSGD,stepSGD)
    
    projSplit.run(maxIterations = iterPS,keepHistory = True,blockActivation = BA,
                  primalTol=1e-9,dualTol=1e-9,nblocks=blocksPS,historyFreq=1)
    ps_val = projSplit.getObjective()
    fps = projSplit.getHistory()[0]
    tps = projSplit.getHistory()[1]
    
    print("Final stepsize is {}".format(processor.getStep()))
    
    getCVX = False 
    if getCVX:
        AwithIntercept = np.zeros((m,d+1))
        AwithIntercept[:,0] = np.ones(m)
        AwithIntercept[:,1:(d+1)] = A
        A = AwithIntercept             
        [opt, xopt] = runCVX_LR(A,y,0.0,True)
        
        print("cvx sol = {}".format(opt))
        
    print("fsgd opt = {}".format(fsgd[-1]))
    print("ps opt = {}".format(ps_val))
    
    if doplots:
        plt.plot(timessgd,fsgd)
        plt.plot(tps,fps)
        plt.legend(['sgd','ps'])
        plt.show()
    

if __name__=="__main__":
    testSGD()
    test_sgd_lr()
    