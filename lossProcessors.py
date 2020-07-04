# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:16:02 2020

@author: pjohn
"""

from numpy import array
from numpy import concatenate
from numpy import zeros
from numpy import copy as npcopy 
from numpy import ones
from numpy import identity
from numpy.linalg import inv as npinv


from numpy.linalg import norm
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
    
    def initialize(self,psObj):
        pass 
        
        
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
        self.step = initialStep
        self.alpha = blendFactor
        self.delta = backTrackFactor
        self.growFac = growFactor
        self.growFreq = growFreq
        self.eta = float('inf')
        
    def initialize(self,psObj):
        '''
           this routine is used by Foward1Backtrack
           to initialize the gradients of xdata, \hat{theta}, \hat{w}, xdata, and ydata
        '''
        # initalize theta_hat
        self.thetahat = zeros(psObj.xdata.shape)
        self.what = zeros(psObj.xdata.shape)
        self.gradxdata = zeros(psObj.xdata.shape)
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            self.thetahat[block][1:] = psObj.embedded.getProx(self.thetahat[block][1:])
            self.thetahat[block][0] = 0.0
            self.what[block] = -psObj.embedded.getScalingAndStepsize()[1]**(-1)*self.thetahat[block]
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
        self.step = stepsize 
        self.stepChanged = False # This flag is set to True whenever the stepsize is changed via
                                 # the settep method below. This is used by the backwardExact class
                                 # which needs to update precomputed inverses whenever
                                 # the stepsize is changed. 
        
    
    def initialize(self,psObj):
        block_len = len(psObj.partition[0])
        # block length is the number of observations in each block
        # we only check the len of the first block because our createApartition()
        # function guarantees that all blocks are within 1 of the same block_len
        if block_len < psObj.ncol//2:
            # wide matrices, use the matrix inversion lemma 
            self.matInvLemma = True
            
        else:
            self.matInvLemma = False
        
                 
        
        if psObj.intercept == True:
            # need to add a ones col to deal with intercept
            onesCol = ones((psObj.nobs,1))
            # make a copy of the data matrix to deal with the intercept called Atilde
            self.Atilde = concatenate((onesCol,psObj.A), axis = 1)
        else:
            self.Atilde = psObj.A 

        self.Aty = []        
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            self.Aty.append(self.Atilde[thisSlice].T.dot(psObj.yresponse[thisSlice]))
        
        if self.matInvLemma == False:
            self.matInv = []        
            for block in range(psObj.nDataBlocks):
                thisSlice = psObj.partition[block]                
                mat2inv = (self.step/psObj.nobs)*self.Atilde[thisSlice].T.dot(self.Atilde[thisSlice])
                (d,_) = mat2inv.shape
                mat2inv += identity(d)
                self.matInv.append(npinv(mat2inv))
        else:
            self.matInv = []        
            for block in range(psObj.nDataBlocks):
                thisSlice = psObj.partition[block]
                mat2inv = (self.step/psObj.nobs)*self.Atilde[thisSlice].dot(self.Atilde[thisSlice].T)
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
        if psObj.intercept:
            input2inv = t + (self.step/psObj.nobs)*self.Aty[block]
        else:
            input2inv = t[1:] + (self.step/psObj.nobs)*self.Aty[block]
            
        if self.matInvLemma == True:
            #using the matrix inversion lemma 
            temp = self.matInv[block].dot(self.Atilde[thisSlice].dot(input2inv))
            if psObj.intercept:                
                psObj.xdata[block] = input2inv - (self.step/psObj.nobs)*self.Atilde[thisSlice].T.dot(temp)            
            else:                
                psObj.xdata[block][1:] = input2inv - (self.step/psObj.nobs)*self.Atilde[thisSlice].T.dot(temp)            
                psObj.xdata[block][0] = 0.0
                
        else:
            #not using the matrix inversion lemma
            if psObj.intercept:
                psObj.xdata[block] = self.matInv[block].dot(input2inv)
            else:
                psObj.xdata[block][1:] = self.matInv[block].dot(input2inv)
                psObj.xdata[block][0] = 0.0
                                    
            
        psObj.ydata[block] = (self.step)**(-1)*(t - psObj.xdata[block])
            
            
    def setStep(self,step):
        self.step = step
        self.stepChanged = True 
        
        
class BackwardCG(ProjSplitLossProcessor):
    def __init__(self,relativeErrorFactor=0.9,stepsize=1.0,maxIter=100):
        self.embedOK = False
        self.pMustBe2 = True
        self.step = stepsize
        self.sigma = relativeErrorFactor
        self.maxIter = maxIter
    
    def initialize(self,psObj):
        
        if psObj.intercept == True:
            # need to add a ones col to deal with intercept
            onesCol = ones((psObj.nobs,1))
            # make a copy of the data matrix to deal with the intercept called Atilde
            self.Atilde = concatenate((onesCol,psObj.A), axis = 1)
        else:
            self.Atilde = psObj.A 

        self.Aty = []        
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            self.Aty.append(self.Atilde[thisSlice].T.dot(psObj.yresponse[thisSlice]))
            
    
    def update(self,psObj,block):
                            
        thisSlice = psObj.partition[block]
        def Acg(x):
            # helper function returns the matrix multiply for the "conjugate
            # gradient" matrix, i.e. the lhs of the linear equation we are trying
            # to solve which defines the backward step.             
            temp = self.Atilde[thisSlice].dot(x)
            temp = self.Atilde[thisSlice].T.dot(temp)
            return x + (self.step/psObj.nobs)*temp
            
            
        if psObj.intercept == True:
            t = psObj.Hz + self.step*psObj.wdata[block]
            b = t + (self.step/psObj.nobs)*self.Aty[block] # b is the input to the inverse
            x = psObj.xdata[block]        
            Hz = psObj.Hz
            w = psObj.wdata[block]
        else:
            t = psObj.Hz[1:] + self.step*psObj.wdata[block][1:]
            b = t + (self.step/psObj.nobs)*self.Aty[block] # b is the input to the inverse
            x = psObj.xdata[block][1:]        
            Hz = psObj.Hz[1:]
            w = psObj.wdata[block][1:]
        
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
            gradfx = (1.0/self.step)*(Acgx - x) - (1/psObj.nobs)*self.Aty[block]
            
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
            
        if psObj.intercept == True:
            psObj.xdata[block] = x
            psObj.ydata[block] = gradfx
        else:
            psObj.xdata[block][1:] = x
            psObj.ydata[block][1:] = gradfx
            psObj.xdata[block][0] = 0.0
        
class BackwardLBFGS(ProjSplitLossProcessor):
    def __init__(self,relativeErrorFactor = 0.9,memory = 10,c1 = 1e-4,
                 c2 = 0.9,shrinkFactor = 0.7, growFactor = 1.1):
        self.embedOK = False
        
    def update(self,psObj,block):
        
        pass
        

class StochasticTwoForwardStep(ProjSplitLossProcessor):
    def __init__(self):
        pass

class StochasticOneForwardStep(ProjSplitLossProcessor):
    def __init__(self):
        pass
    
    