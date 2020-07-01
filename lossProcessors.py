# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:16:02 2020

@author: pjohn
"""

from numpy import array
from numpy import concatenate
from numpy import zeros
from numpy import copy as npcopy 


from numpy.linalg import norm
#-----------------------------------------------------------------------------
# processor class and related objects
#-----------------------------------------------------------------------------
        
class ProjSplitLossProcessor(object):
    pMustBe2 = False 
    @staticmethod
    def getAGrad(psObj,point,thisSlice):
        #point[0] is the intercept term
        #point[1:len(point)] are the coefficients and 
        #len(point) must equal the num cols of A. 
        yhat = point[0]+psObj.A[thisSlice].dot(point[1:])
        gradL = psObj.loss.derivative(yhat,psObj.yresponse[thisSlice])        
        grad = (1.0/psObj.nobs)*psObj.A[thisSlice].T.dot(gradL)
        if psObj.intercept:            
            grad0 = array([(1.0/psObj.nobs)*sum(gradL)])
        else:
            grad0 = array([0.0])
        grad = concatenate((grad0,grad))
        return grad  
      
    def getStep(self):
        return self.step
    
    def setStep(self,step):
        self.step = step
        
#############
class Forward2Fixed(ProjSplitLossProcessor):
    def __init__(self,step=1.0):
        self.step = step
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
        self.step = initialStep
        self.Delta = Delta
        self.decFactor = backtrackFactor
        self.growFactor = growFactor
        self.growFreq = growFreq
        
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
        self.Delta=Delta 
        self.pMustBe2 = True 
        
    def update(self,psObj,block):
        thisSlice = psObj.partition[block]
        gradHz = self.getAGrad(psObj,psObj.Hz,thisSlice)
        lhs = gradHz - psObj.wdata[block]
        
        yhat = lhs[0]+psObj.A[thisSlice].dot(lhs[1:])        
        affinePart = (1.0/psObj.nobs)*psObj.A[thisSlice].T.dot(yhat)
        if psObj.intercept:            
            affine0 = array([(1.0/psObj.nobs)*sum(affinePart)])
        else:
            affine0 = array([0.0])
        affinePart = concatenate((affine0,affinePart))
        normLHS = norm(lhs,2)**2
        step = normLHS/(self.Delta*normLHS + lhs.T.dot(affinePart))
        psObj.xdata[block] = psObj.Hz - step*lhs
        psObj.ydata[block] = gradHz - step*affinePart
        
        
    
class  Forward1Fixed(ProjSplitLossProcessor):
    def __init__(self,stepsize, blendFactor=0.1):
        self.step = stepsize
        self.alpha = blendFactor
        self.embedOK = True 
    
    def __initializeGradXdata(self,psObj):
        '''
           this routine is used by Forward1Fixed 
           to initialize the gradients of xdata
        '''
        psObj.gradxdata = zeros(psObj.xdata.shape)
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            psObj.gradxdata[block] = self.getAGrad(psObj,psObj.xdata[block],thisSlice)     
        
    def update(self,psObj,block):
        if psObj.gradxdata is None:
            self.__initializeGradXdata(psObj)
                        
        thisSlice = psObj.partition[block]
        t = (1-self.alpha)*psObj.xdata[block] +self.alpha*psObj.Hz \
            - self.step*(psObj.gradxdata[block] - psObj.wdata[block])
        psObj.xdata[block][1:] = psObj.embedded.getProx(t[1:]) 
        psObj.xdata[block][0] = t[0]   
        psObj.gradxdata[block] = self.getAGrad(psObj,psObj.xdata[block],thisSlice) 
        psObj.ydata[block] = self.step**(-1)*(t-psObj.xdata[block])+psObj.gradxdata[block]
        
  
    
class Forward1Backtrack(ProjSplitLossProcessor):
    def __init__(self,initialStep=1.0, blendFactor=0.1,backTrackFactor = 0.7, 
                 growFactor = 1.0, growFreq = None):
        self.embedOK = True 
        self.step = initialStep
        self.alpha = blendFactor
        self.delta = backTrackFactor
        self.growFac = growFactor
        self.growFreq = growFreq
        self.eta = float('inf')
        
    def __initialize1fBacktrack(self,psObj):
        '''
           this routine is used by Foward1Backtrack
           to initialize the gradients of xdata, \hat{theta}, \hat{w}, xdata, and ydata
        '''
        # initalize theta_hat
        psObj.thetahat = zeros(psObj.xdata.shape)
        psObj.what = zeros(psObj.xdata.shape)
        psObj.gradxdata = zeros(psObj.xdata.shape)
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            psObj.thetahat[block][1:] = psObj.embedded.getProx(psObj.thetahat[block][1:])
            psObj.thetahat[block][0] = 0.0
            psObj.what[block] = -psObj.embedded.getScalingAndStepsize()[1]**(-1)*psObj.thetahat[block]
            psObj.gradxdata[block] = self.getAGrad(psObj,psObj.thetahat[block],thisSlice)        
            psObj.what[block] += psObj.gradxdata[block]
        
        psObj.xdata = psObj.thetahat
        psObj.ydata = psObj.what
        
    def update(self,psObj,block):
        if psObj.gradxdata is None:
            self.__initialize1fBacktrack(psObj)
        
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
        t2 = npcopy(psObj.gradxdata[block]) 
        t2 -= psObj.wdata[block]
        while True:
            t = t1 - self.step*t2
            psObj.xdata[block][1:] = psObj.embedded.getProx(t[1:]) 
            psObj.xdata[block][0] = t[0]   
            
            psObj.gradxdata[block] = self.getAGrad(psObj,psObj.xdata[block],thisSlice) 
            psObj.ydata[block] = self.step**(-1)*(t-psObj.xdata[block])+psObj.gradxdata[block]
            
            yhat = self.step**(-1)*( (1-self.alpha)*xold +self.alpha*psObj.Hz - psObj.xdata[block] )\
                    + psObj.wdata[block]
            phiPlus = (psObj.Hz - psObj.xdata[block]).T.dot(psObj.ydata[block] - psObj.wdata[block])
            
            lhs1 = norm(psObj.xdata[block] - psObj.thetahat[block],2)
            rhs1 = (1-self.alpha)*norm(xold -psObj.thetahat[block] ,2) \
                    + self.alpha*norm(psObj.Hz-psObj.thetahat[block],2) \
                    + self.step*norm(psObj.wdata[block] - psObj.what[block],2)
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
class BackwardCG(ProjSplitLossProcessor):
    def __init__(self,relativeErrorFactor):
        pass
    def update(self,psObj,thisSlice):
        pass

class BackwardLBFGS(ProjSplitLossProcessor):
    def __init__(self,relativeErrorFactor = 0.9,memory = 10,c1 = 1e-4,
                 c2 = 0.9,shrinkFactor = 0.7, growFactor = 1.1):
        pass
    def update(self,psObj,thisSlice):
        pass
        
class BackwardExact(ProjSplitLossProcessor):
    def __init__(self,stepsize):
        pass
    def update(self,psObj,thisSlice):
        pass
    