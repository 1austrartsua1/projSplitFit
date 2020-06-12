
import sys
sys.path.append('../')
import projSplit as ps 
import numpy as np
from matplotlib import pyplot as plt
from utils import runCVX_LR


def testSGD():
    
    gamma = 1e-5
    projSplit = ps.ProjSplitFit(gamma)
    m = 200
    d = 100
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    
    stepSGD = 10.0
    iterSGD = 2000
    blocksSGD = 10
    freqSGD = 1
    
    processor = ps.Forward1Backtrack(stepSGD)
    projSplit.addData(A,y,2,processor,intercept=True,normalize=True)
    
    
    
    fsgd,timessgd = projSplit.runSGD(iterSGD,blocksSGD,freqSGD,stepSGD)
    
    projSplit.run(maxIterations = iterSGD,keepHistory = True,
                  primalTol=1e-9,dualTol=1e-9,nblocks=blocksSGD,historyFreq=1)
    ps_val = projSplit.getObjective()
    fps = projSplit.getHistory()[0]
    
    AwithIntercept = np.zeros((m,d+1))
    AwithIntercept[:,0] = np.ones(m)
    AwithIntercept[:,1:(d+1)] = A
    result = np.linalg.lstsq(AwithIntercept,y,rcond=None)
    xhat = result[0]  
    LSval = 0.5*np.linalg.norm(AwithIntercept.dot(xhat)-y,2)**2/m
    print("ls sol = {}".format(LSval))
    print("fsgd opt = {}".format(fsgd[-1]))
    print("ps opt = {}".format(ps_val))
    
    #plt.plot(fsgd)
    #plt.plot(fps)
    #plt.legend(['sgd','ps'])
    #plt.show()

def test_sgd_lr():
    gamma = 1e-8
    projSplit = ps.ProjSplitFit(gamma)
    m = 200
    d = 100
    A = np.random.normal(0,1,[m,d])
    y = np.random.normal(0,1,m)
    
    stepSGD = 100.0
    iterSGD = 2000
    blocksSGD = 10
    freqSGD = 1
    
    processor = ps.Forward1Backtrack(stepSGD,growFactor=1.2,growFreq=10)
    projSplit.addData(A,y,'logistic',processor,intercept=True,normalize=True)
    
    
    
    fsgd,timessgd = projSplit.runSGD(iterSGD,blocksSGD,freqSGD,stepSGD)
    
    projSplit.run(maxIterations = iterSGD,keepHistory = True,
                  primalTol=1e-9,dualTol=1e-9,nblocks=blocksSGD,historyFreq=1)
    ps_val = projSplit.getObjective()
    fps = projSplit.getHistory()[0]
    
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
    
    #plt.plot(fsgd)
    #plt.plot(fps)
    #plt.legend(['sgd','ps'])
    #plt.show()
    

if __name__=="__main__":
    testSGD()
    test_sgd_lr()
    