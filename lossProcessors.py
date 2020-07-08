# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:16:02 2020

@author: pjohn
"""

from numpy import zeros
from numpy import copy as npcopy 
from numpy import identity
from numpy.linalg import inv as npinv
from numpy.linalg import norm
import userInputVal as ui
#-----------------------------------------------------------------------------
# processor class and related objects
#-----------------------------------------------------------------------------
        
class ProjSplitLossProcessor(object):
    pMustBe2 = False # This flag to True for lossProcessors which can only be applied 
                     # to the case where p=2, i.e. quadratic loss. 
                     # Such as Forward2Affine, BackwardExact, and BackwardCG
    embedOK = False  # This flag is True if this lossProcessor can handle an embedded 
                     # regularizer. Examples which can are Forward1x and Forward2x
                     # but backward classes cannot. 
                     
    @staticmethod
    def getAGrad(psObj,point,thisSlice):        
        yhat = psObj.A[thisSlice].dot(point)
        gradL = psObj.loss.derivative(yhat,psObj.yresponse[thisSlice])        
        grad = (1.0/psObj.nrowsOfA)*psObj.A[thisSlice].T.dot(gradL)        
        return grad  
      
    def getStep(self):
        return self.step
    
    def setStep(self,step):
        self.step = step
    
    def initialize(self,psObj):
        '''
            can implement required initialization for the loss processor
        '''
        pass 
    
    def update(self,psObj,block):
        '''
            Update psObj.xdata[block] and psObj.ydata[block].
            This method must be implemented. 
        '''
        pass
        
        
#############
class Forward2Fixed(ProjSplitLossProcessor):
    def __init__(self,step=1.0):
        
        self.step = ui.checkUserInput(step,float,'float','stepsize',default=1.0,low=0.0)             
        self.embedOK = True
        
    def update(self,psObj,block):        
        thisSlice = psObj.partition[block]
        gradHz = self.getAGrad(psObj,psObj.Hz,thisSlice)
        t = psObj.Hz - self.step*(gradHz - psObj.wdata[block])        
        psObj.xdata[block][1:] = psObj.embedded.getProx(t[1:]) 
        psObj.xdata[block][0] = t[0]        
        a = self.step**(-1)*(t-psObj.xdata[block])
        gradx = self.getAGrad(psObj,psObj.xdata[block],thisSlice)        
        psObj.ydata[block] = a + gradx        
        
class Forward2Backtrack(ProjSplitLossProcessor):
    def __init__(self,initialStep=1.0,Delta=1.0,backtrackFactor=0.7,
                 growFactor=1.0,growFreq=None):
        self.embedOK = True
        self.step = ui.checkUserInput(initialStep,float,'float','stepsize',default=1.0,low=0.0)                     
        self.Delta = ui.checkUserInput(Delta,float,'float','Delta',default=1.0,low=0.0)                     
        self.decFactor = ui.checkUserInput(backtrackFactor,float,'float','backtrackFactor',default=0.7,low=0.0,high=1.0)                     
        self.growFactor = ui.checkUserInput(growFactor,float,'float','growFactor',default=1.1,high=1.0,highAllowed=True)
        if growFreq == None:
            self.growFreq = None
        else:
            self.growFreq = ui.checkUserInput(growFreq,int,'int','growFreq',default=10,low = 0)
        
    def update(self,psObj,block):
        thisSlice = psObj.partition[block]
        gradHz = self.getAGrad(psObj,psObj.Hz,thisSlice)
        if self.growFreq is not None:
            if psObj.k % self.growFreq == 0:
                # time to grow the stepsize
                self.step *= self.growFactor
                psObj.embedded.setStep(self.step)
                
        while True:
            t = psObj.Hz - self.step*(gradHz - psObj.wdata[block])        
            psObj.xdata[block][1:] = psObj.embedded.getProx(t[1:]) 
            psObj.xdata[block][0] = t[0]        
            a = self.step**(-1)*(t-psObj.xdata[block])
            gradx = self.getAGrad(psObj,psObj.xdata[block],thisSlice)        
            psObj.ydata[block] = a + gradx  
            lhs = psObj.Hz - psObj.xdata[block]
            rhs = psObj.ydata[block] - psObj.wdata[block]
            if lhs.T.dot(rhs)>=self.Delta*norm(lhs,2)**2:
                break
            else:
                self.step *= self.decFactor
                psObj.embedded.setStep(self.step)
            
        
    
class Forward2Affine(ProjSplitLossProcessor):
    def __init__(self,Delta=1.0):
        self.embedOK = False
        self.Delta = ui.checkUserInput(Delta,float,'float','Delta',default=1.0,low=0.0)                     
        self.pMustBe2 = True 
        
    def update(self,psObj,block):
        thisSlice = psObj.partition[block]
        gradHz = self.getAGrad(psObj,psObj.Hz,thisSlice)
        lhs = gradHz - psObj.wdata[block]
        
        yhat = psObj.A[thisSlice].dot(lhs)        
        affinePart = (1.0/psObj.nrowsOfA)*psObj.A[thisSlice].T.dot(yhat)        
        normLHS = norm(lhs,2)**2
        step = normLHS/(self.Delta*normLHS + lhs.T.dot(affinePart))
        psObj.xdata[block] = psObj.Hz - step*lhs
        psObj.ydata[block] = gradHz - step*affinePart
        
        
    
class  Forward1Fixed(ProjSplitLossProcessor):
    def __init__(self,stepsize, blendFactor=0.1):
        self.step = ui.checkUserInput(stepsize,float,'float','stepsize',default=1.0,low=0.0)             
        self.alpha = ui.checkUserInput(blendFactor,float,'float','blendFactor',default=0.1,low=0.0,high=1.0)
        self.embedOK = True 
        
    def initialize(self,psObj):    
        '''
           this routine is used by Forward1Fixed 
           to initialize the gradients of xdata
        '''
        self.gradxdata = zeros(psObj.xdata.shape)
        # gradxdata will store the gradient of the loss for each xdata[block]
        
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            self.gradxdata[block] = self.getAGrad(psObj,psObj.xdata[block],thisSlice)     
        
    def update(self,psObj,block):                                
        thisSlice = psObj.partition[block]
        t = (1-self.alpha)*psObj.xdata[block] +self.alpha*psObj.Hz \
            - self.step*(self.gradxdata[block] - psObj.wdata[block])
        psObj.xdata[block][1:] = psObj.embedded.getProx(t[1:]) 
        psObj.xdata[block][0] = t[0]   
        self.gradxdata[block] = self.getAGrad(psObj,psObj.xdata[block],thisSlice) 
        psObj.ydata[block] = self.step**(-1)*(t-psObj.xdata[block])+self.gradxdata[block]
        
  
    
class Forward1Backtrack(ProjSplitLossProcessor):
    def __init__(self,initialStep=1.0, blendFactor=0.1,backTrackFactor = 0.7, 
                 growFactor = 1.0, growFreq = None):
        self.embedOK = True 
        self.step = ui.checkUserInput(initialStep,float,'float','initialStep',default=1.0,low=0.0)             
        self.alpha = ui.checkUserInput(blendFactor,float,'float','blendFactor',default=0.1,low=0.0,high=1.0)
        
        self.delta = backTrackFactor
        self.delta = ui.checkUserInput(backTrackFactor,float,'float','backTrackFactor',default=0.7,low=0.0,high=1.0)
        self.growFac = ui.checkUserInput(growFactor,float,'float','growFactor',default=1.0,low=1.0,lowAllowed=True)
        
        if growFreq == None:
            self.growFreq = None
        else:
            self.growFreq = ui.checkUserInput(growFreq,int,'int','growFreq',default=10,low = 0)
            
        self.eta = float('inf')
        
    def initialize(self,psObj):
        '''
           this routine is used by Foward1Backtrack
           to initialize the gradients of xdata, \hat{theta}, \hat{w}, xdata, and ydata
        '''
        
        self.thetahat = zeros(psObj.xdata.shape)
        self.what = zeros(psObj.xdata.shape)
        self.gradxdata = zeros(psObj.xdata.shape)
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            self.thetahat[block][1:] = psObj.embedded.getProx(self.thetahat[block][1:])
            self.thetahat[block][0] = 0.0
            self.what[block] = -psObj.embedded.getStepsize()**(-1)*self.thetahat[block]
            self.gradxdata[block] = self.getAGrad(psObj,self.thetahat[block],thisSlice)        
            self.what[block] += self.gradxdata[block]
        
        psObj.xdata = self.thetahat
        psObj.ydata = self.what
        
    def update(self,psObj,block):        
        
        if self.growFreq is not None:
            if psObj.k % self.growFreq == 0:
                # time to grow the stepsize
                upper_bound = (1+self.alpha*self.eta)*self.step 
                desired_step = self.growFac*self.step
                self.step = min([upper_bound,desired_step])                     
                psObj.embedded.setStep(self.step)
                
        
        thisSlice = psObj.partition[block]
        
        phi = (psObj.Hz - psObj.xdata[block]).T.dot(psObj.ydata[block] - psObj.wdata[block])
        
        xold = npcopy(psObj.xdata[block])
        yold = npcopy(psObj.ydata[block])
        
        t1 = (1-self.alpha)*xold +self.alpha*psObj.Hz
        t2 = npcopy(self.gradxdata[block]) 
        t2 -= psObj.wdata[block]
        while True:
            t = t1 - self.step*t2
            psObj.xdata[block][1:] = psObj.embedded.getProx(t[1:]) 
            psObj.xdata[block][0] = t[0]   
            
            self.gradxdata[block] = self.getAGrad(psObj,psObj.xdata[block],thisSlice) 
            psObj.ydata[block] = self.step**(-1)*(t-psObj.xdata[block])+self.gradxdata[block]
            
            yhat = self.step**(-1)*( (1-self.alpha)*xold +self.alpha*psObj.Hz - psObj.xdata[block] )\
                    + psObj.wdata[block]
            phiPlus = (psObj.Hz - psObj.xdata[block]).T.dot(psObj.ydata[block] - psObj.wdata[block])
            
            lhs1 = norm(psObj.xdata[block] - self.thetahat[block],2)
            rhs1 = (1-self.alpha)*norm(xold -self.thetahat[block] ,2) \
                    + self.alpha*norm(psObj.Hz-self.thetahat[block],2) \
                    + self.step*norm(psObj.wdata[block] - self.what[block],2)
            if lhs1 <= rhs1:                
                numer = norm(yhat-psObj.wdata[block],2)**2
                denom = norm(psObj.ydata[block]-psObj.wdata[block],2)**2
                rhs2_1 = 0.5*(self.step/self.alpha)*(denom + self.alpha*numer)
                                                
                rhs2_2 = (1-self.alpha)*(phi - 0.5*(self.step/self.alpha)*norm(yold-psObj.wdata[block],2)**2)
                
                if phiPlus >= rhs2_1 + rhs2_2:
                    #backtracking termination criteria satisfied
                    self.eta = numer/denom
                    break
            
            self.step *= self.delta
            psObj.embedded.setStep(self.step)
                
                
    
#############
class BackwardExact(ProjSplitLossProcessor):
    def __init__(self,stepsize=1.0):
        self.embedOK = False
        self.pMustBe2 = True
        
        self.step = ui.checkUserInput(stepsize,float,'float','stepsize',default=1.0,low=0.0)                     
        
        self.stepChanged = False # This flag is set to True whenever the stepsize is changed via
                                 # the settep method below. This is used by the backwardExact class
                                 # which needs to update precomputed inverses whenever
                                 # the stepsize is changed. 
        
    
    def initialize(self,psObj):
        block_len = len(psObj.partition[0])
        # block length is the number of observations in each block
        # we only check the len of the first block because our createApartition()
        # function guarantees that all blocks are within 1 of the same block_len
        if block_len < psObj.ncolsOfA//2:
            # wide matrices, use the matrix inversion lemma 
            self.matInvLemma = True
            
        else:
            self.matInvLemma = False
        
        self.Aty = []        
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            self.Aty.append(psObj.A[thisSlice].T.dot(psObj.yresponse[thisSlice]))
        
        if self.matInvLemma == False:
            self.matInv = []        
            for block in range(psObj.nDataBlocks):
                thisSlice = psObj.partition[block]                
                mat2inv = (self.step/psObj.nrowsOfA)*psObj.A[thisSlice].T.dot(psObj.A[thisSlice])
                (d,_) = mat2inv.shape
                mat2inv += identity(d)
                self.matInv.append(npinv(mat2inv))
        else:
            self.matInv = []        
            for block in range(psObj.nDataBlocks):
                thisSlice = psObj.partition[block]
                mat2inv = (self.step/psObj.nrowsOfA)*psObj.A[thisSlice].dot(psObj.A[thisSlice].T)
                (n,_) = mat2inv.shape
                mat2inv += identity(n)
                self.matInv.append(npinv(mat2inv))
                
            
    def update(self,psObj,block):
        
        if self.stepChanged: 
            # if the stepsize is changed,
            # we need to re-initialize the cached matrix inverses.                         
            self.initialize(psObj)
            self.stepChanged = False
                                
        
        thisSlice = psObj.partition[block]
        t = psObj.Hz + self.step*psObj.wdata[block]
        
        input2inv = t + (self.step/psObj.nrowsOfA)*self.Aty[block]
        
            
        if self.matInvLemma == True:
            #using the matrix inversion lemma 
            temp = self.matInv[block].dot(psObj.A[thisSlice].dot(input2inv))            
            psObj.xdata[block] = input2inv - (self.step/psObj.nrowsOfA)*psObj.A[thisSlice].T.dot(temp)                                        
        else:
            #not using the matrix inversion lemma
        
            psObj.xdata[block] = self.matInv[block].dot(input2inv)        
                                                
        psObj.ydata[block] = (self.step)**(-1)*(t - psObj.xdata[block])
            
            
    def setStep(self,step):
        self.step = step
        self.stepChanged = True 
        
        
class BackwardCG(ProjSplitLossProcessor):
    def __init__(self,relativeErrorFactor=0.9,stepsize=1.0,maxIter=100):
        self.embedOK = False
        self.pMustBe2 = True
        
        self.step = ui.checkUserInput(stepsize,float,'float','stepsize',default=1.0,low=0.0)
        self.sigma = ui.checkUserInput(relativeErrorFactor,float,'float',
                                       'relativeErrorFactor',default=0.9,low=0.0,high=1.0,lowAllowed=True)     
        self.maxIter = ui.checkUserInput(maxIter,int,'int','maxIter',default=100,low=0)
        
    
    def initialize(self,psObj):
                

        self.Aty = []        
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            self.Aty.append(psObj.A[thisSlice].T.dot(psObj.yresponse[thisSlice]))
            
    
    def update(self,psObj,block):
                            
        thisSlice = psObj.partition[block]
        def Acg(x):
            # helper function returns the matrix multiply for the "conjugate
            # gradient" matrix, i.e. the lhs of the linear equation we are trying
            # to solve which defines the backward step.             
            temp = psObj.A[thisSlice].dot(x)
            temp = psObj.A[thisSlice].T.dot(temp)
            return x + (self.step/psObj.nrowsOfA)*temp
            
            
        
        t = psObj.Hz + self.step*psObj.wdata[block]
        b = t + (self.step/psObj.nrowsOfA)*self.Aty[block] # b is the input to the inverse
        x = psObj.xdata[block]        
        Hz = psObj.Hz
        w = psObj.wdata[block]
        
        
        # run conjugate gradient method
        
        Acgx = Acg(x)
        r = b - Acgx
        p = r
        i = 0
        while True:
            rTr = r.T.dot(r)            
            Ap = Acg(p)
            denom = p.T.dot(Ap)
            alpha = rTr/denom
            
            x = x + alpha*p
        
            Acgx = Acgx + alpha*Ap
            #gradfx is gradient w.r.t. the least squares slice. 
            gradfx = (1.0/self.step)*(Acgx - x) - (1/psObj.nrowsOfA)*self.Aty[block]
            
            i+=1
            if i>= self.maxIter:
                break
            
            e = x+self.step*gradfx - t
            err1 = e.T.dot(Hz - x) + self.sigma*norm(Hz - x)**2
            if err1 >= 0:
                err2 = e.T.dot(gradfx - w) \
                       - self.step*norm(gradfx - w)
                if err2<=0:
                    break
                        
            rplus = r - alpha*Ap
            beta = rplus.T.dot(rplus)/rTr
            p = rplus + beta*p
            r = rplus
                   
        psObj.xdata[block] = x
        psObj.ydata[block] = gradfx
        
        
class BackwardLBFGS(ProjSplitLossProcessor):
    def __init__(self,step=1.0,relativeErrorFactor = 0.9,memory = 10,c1 = 1e-4,
                 c2 = 0.9,shrinkFactor = 0.7, growFactor = 1.1,
                 maxiter=100,lineSearchIter = 20):
        self.embedOK = False
        self.step = ui.checkUserInput(step,float,'float','stepsize',default=1.0,low=0.0)                     
        self.sigma = ui.checkUserInput(relativeErrorFactor,float,'float',
                                       'relativeErrorFactor',default=0.9,low=0.0,high=1.0,lowAllowed=True)     
        
        self.m = ui.checkUserInput(memory,int,'int','memory',default=10,low=1)
        self.c1 = ui.checkUserInput(c1,float,'float','c1',default=1e-4,low=0.0,high=1.0)
        self.c2 = ui.checkUserInput(c2,float,'float','c2',default=1e-4,low=0.0,high=1.0)
        if self.c1 >= self.c2:
            print("Warning: c1 must be less than c2. Setting to default c1=1e-4,c2=0.9")
            self.c1=1e-4
            self.c2=0.9
        
        self.shrinkFactor = ui.checkUserInput(shrinkFactor,float,'float','shrinkFactor',
                                              default=0.7,low=0.0,high=1.0)

        self.growFactor = ui.checkUserInput(growFactor,float,'float','growFactor',
                                         default=1.0,low=1.0,lowAllowed=True)
        
        self.maxiter = ui.checkUserInput(maxiter,int,'int','maxiter',default=100,low=0)
        self.lineSearchIter = ui.checkUserInput(lineSearchIter,int,'int','maxiter',default=20,low=0)
        
        
    def Fprox(self,psObj,x,thisSlice,t):
        Ax = psObj.A[thisSlice].dot(x)        
        f = (self.step/psObj.nrowsOfA)\
            *sum(psObj.loss.value(Ax,psObj.yresponse[thisSlice])) 
        f += 0.5*norm(t - x,2)**2
        return f
    
    def gradprox(self,psObj,x,thisSlice,t):
        return self.step*self.getAGrad(psObj,x,thisSlice) + x - t
        
    def update(self,psObj,block):
        thisSlice = psObj.partition[block]
        t = psObj.Hz + self.step*psObj.wdata[block]
        x = psObj.xdata[block]
        d = len(x)
        Y = zeros([self.m,d])
        S = zeros([self.m,d])
        rho = zeros(self.m)
        alpha = zeros(self.m)
                
        
                
        grad = self.gradprox(psObj,x,thisSlice,t)
        f = self.Fprox(psObj,x,thisSlice,t)
        z = grad
        
        k = 0
        while k < self.maxiter:        
            p = -z
            
            xnew,gradnew,fnew = self.wolfeLineSearch(psObj,x,p,grad,f,t,thisSlice)
            gradfx = (gradnew - (xnew - t))/self.step 
            k += 1
            if self.passesErrCheck(psObj,xnew,t,block,gradfx) or (k>=self.maxiter):
                x = xnew
                break                                      
            
            snew = xnew - x                        
            x = xnew            
            ynew = gradnew - grad
            grad = gradnew
            f = fnew
            
            self.shift(Y,ynew)
            self.shift(S,snew)
            
            rhonew = 1.0/ynew.T.dot(snew)
            self.shift(rho,rhonew)
            
            q = grad
            for i in range(self.m-1,-1,-1):
                alpha[i] = rho[i]*S[i].T.dot(q)
                q = q - alpha[i]*Y[i]
           
            gamma = snew.T.dot(ynew)/(ynew.T.dot(ynew))
            z = gamma*q
            
            for i in range(self.m):
                beta = rho[i]*Y[i].T.dot(z)
                z = z + (alpha[i] - beta)*S[i]
        
        psObj.xdata[block] = x
        psObj.ydata[block] = gradfx 
                                
    @staticmethod
    def shift(vec,newel):
        vec[0:-1] = vec[1:]
        vec[-1] = newel
        
    def wolfeLineSearch(self,psObj,x,p,grad,f,t,thisSlice):
        
        direcDeriv = grad.T.dot(p)
        step = 1.0
        stepNotFound = True
        niter = 0
        gradNotComputed = True
        while stepNotFound:
            xTrial = x + step*p
            fTrial = self.Fprox(psObj,xTrial,thisSlice,t)
            
            cond1 = fTrial - f - self.c1*step*direcDeriv
            if cond1 <= 0:
                gradNotComputed = False
                gradTrial = self.gradprox(psObj,xTrial,thisSlice,t)
                cond2 = gradTrial.T.dot(p) - self.c2*direcDeriv
                if cond2 >= 0:
                    stepNotFound = False
                else:
                    step = self.growFactor*step
                    
            else:
                step = self.shrinkFactor*step
                
            niter += 1
            if (niter >= self.lineSearchIter):
                stepNotFound = False
                
        if gradNotComputed:
            gradTrial = self.gradprox(psObj,xTrial,thisSlice,t)
        return xTrial,gradTrial,fTrial
    
    def passesErrCheck(self,psObj,x,t,block,gradfx):
        w = psObj.wdata[block]        
        e = x+self.step*gradfx - t
        err1 = e.T.dot(psObj.Hz - x) + self.sigma*norm(psObj.Hz - x)**2
        if err1 >= 0:
            err2 = e.T.dot(gradfx - w) \
                   - self.step*norm(gradfx - w)
            if err2<=0:
                return True
            else:
                return False
        else:
            return False 
        
    
        

class StochTwoForwardStep(ProjSplitLossProcessor):
    def __init__(self):
        pass

class StochOneForwardStep(ProjSplitLossProcessor):
    def __init__(self):
        pass
    
    