'''
projSplit module.

Classes implemented here:
    - projSplitFit
    - Regularizer 
    - LossPlugIn
Functions implemented here:
    - totalVariation1d()
    - dropFirst()
    - L1()
    - groupL2()
'''
import scipy.sparse.linalg as sp
import numpy as np
import time 

class ProjSplitFit(object):
    def __init__(self,dualScaling=1.0):
        
        self.setDualScaling(dualScaling)                
        self.allRegularizers = []
        self.embedded = None         
        self.process = None 
        self.dataAdded = False
        self.numRegs = 0
        self.z = None
        self.nDataBlocks = None
    
    def setDualScaling(self,dualScaling):
        try:    
            self.gamma = float(dualScaling)
            if (dualScaling <= 0):
                print("dualScaling cannot be <= zero")
                print("setting dualScaling to default value of 1.0")
                self.gamma = 1.0
        except:
            print("Warning: dualScaling argument must be a float")
            print("setting dualScaling to default value of 1.0")
            self.gamma = 1.0
    
    def getDualScaling(self):
        return self.gamma 
                            
    def addData(self,observations,responses,loss,process,intercept=True,
                normalize=True): 
        '''
        XX insert comments here 
        '''
        
        try:
            (self.nobs,self.nvar) = observations.shape
            ny = len(responses)
        except:
            print("Error: observations and responses should be arrays, i.e. ")
            print("NumPy arrays. Aborting, did not add data")            
            return -1

        if (self.nobs!=ny):
            print("Error: len(responses) != num observations")
            print("aborting. Data not added")
            return -1
        
        if (self.nobs == 0) | (self.nvar == 0):
            print("Error! number of observations or variables is 0! Aborting addData")
            print("Please call with valid data, i.e. numpy arrays")
            self.A = None
            self.yresponse = None
            return -1 
        
        self.yresponse = responses
        
        if normalize == True:
            print("Normalizing columns of A to unit norm")
            self.normalize = True
            self.A = np.copy(observations)            
            self.scaling = np.linalg.norm(self.A,axis=0)
            self.scaling += 1.0*(self.scaling < 1e-10)
            self.A = self.A/self.scaling
        else:
            print("Not normalizing columns of A")            
            self.A = observations
            self.normalize = False 
        
        self.loss = Loss(loss)
               
        self.process = process
        
        if(self.embedded != None):
            # check that process is the right type
            pass
        
                                                                
        
        self.intercept = intercept        
                
        self.dataAdded = True
        self.resetIterate = True
        
        
        
    
    def getParams(self):
        '''
        After the addData method has been called, one may call the getDataParams method to
        retrieve some of the relevant parameters in the model.
        nvar,nobs,nblocks = psfObj.getParams()
        Here
        nvar: Number of variables, V as defined in (2) immediately above
        nobs: Number of observations
        nblocks: Number of blocks
        If the addData method has not been called yet, getParams returns None.
        '''
        if self.dataAdded == False:
            return None 
        else:        
            return (self.nvar + int(self.intercept),self.nobs)
    
    def addRegularizer(self,regObj, linearOp=None, embed = False):
        #self.allRegularizers
        #reg.linearOp        
        if embed == True:
            if  (linearOp is not None):            
                print("embedded regularizer cannot use linearOp != None")                
                print("Aborting without adding regularizer")
                return -1
            
            
            if self.process is not None:
                # test that processOp is of the right type
                pass
            
            self.embedded = regObj            
        else:            
            self.allRegularizers.append(regObj)
            
        regObj.addLinear(linearOp)
        self.numRegs += 1
        self.resetIterate = True # Ensures we reset the variables if we add another regularizer 
    
    def getObjective(self):
        #self.A
        
        #self.z
        #self.intercept
        #self.onesIntercept
        #self.nobs
        #self.y
        #self.loss 
        #self.allRegularizers
        #reg.linearOp
        #reg.intercept
        if self.z is None :
            print("Method not run yet, no objective to return. Call run() first.")
            return None
        
        Az = self.A.dot(self.z[1:len(self.z)])
        if self.intercept:
            Az += self.z[0]
        currentLoss = (1.0/self.nobs)*sum(self.loss.value(Az,self.yresponse))        
        for reg in self.allRegularizers:
            if reg.linearOp is not None:
                Hiz = reg.linearOp(self.z[1:len(self.z)])  
            else:
                Hiz = self.z[1:len(self.z)] 
            currentLoss += reg.evaluate(Hiz)
                
        if self.embedded != None:
            reg = self.embedded 
            if reg.linearOp is not None:
                Hiz = reg.linearOp(self.z[1:len(self.z)])
            else:
                Hiz = self.z[1:len(self.z)] 
            currentLoss += reg.evaluate(Hiz)
        
        return currentLoss
        
    
    def getSolution(self):
        #z is always of length d+1 as even if there is no intercept
        # I just don't update the intercept
        
        if self.z is None:
            print("Method not run yet, no solution to return. Call run() first.")
            return None
                
        out = np.zeros(self.z.shape)
        
        if self.normalize:
            out[1:len(self.z)] = self.z[1:len(self.z)]/self.scaling
        else:
            out[1:len(self.z)] = self.z[1:len(self.z)]

        if self.intercept:
            out[0] = self.z[0]
            return out
        else:
            return out[1:len(self.z)]
        
        
    
    def getPrimalViolation(self):
        #self.primalErr
        '''
        After at least one call to run, the function call
        objVal = psfObj.getPrimalViolation()
        returns a float containing max_i{||G_i z^k - x_i^k||_\infty}. 
        If run has not been called yet, it returns None.
        '''
        if self.z is None:
            print("Method not run yet, no primal violation to return. Call run() first.")
            return None
        else:
            return self.primalErr
    
    def getDualViolation(self):
        #self.dualErr
        '''
        After at least one call to run, the function call
        objVal = psfObj.getDualViolation()
        returns a float containing max_i{||y_i^k - w_i^k||_\infty}.  
        If run has not been called yet, it returns None
        '''
        if self.z is None:
            print("Method not run yet, no dual violation to return. Call run() first.")
            return None
        else:
            return self.dualErr
    
    def getHistory(self):
        #self.historyArray
        '''
        After at least one call to run with keepHistory set to True, the function call
        historyArray = psfObj.getHistory()
        returns a two-dimensional four-column NumPy array with each row corresponding to an iteration. 
        In each row of this array, the columns have the following interpretation:
        Column Number Interpretation
        0             Objective value
        1             Primal violation
        2             Dual violation
        3             Cumulative run time
        '''
        if self.z is None:
            print("Method not run yet, no history to return. Call run() first.")
            return None
        if self.historyArray is None:
            print("run() was called without the keepHistory option, no history")
            print("call run with the keepHistory argument set to True")
            return None
        return self.historyArray
    
    def run(self,primalTol = 1e-6, dualTol=1e-6,maxIterations=None,keepHistory = False, 
            nblocks = 1, blockActivation="greedy", blocksPerIteration=1, resetIterate=False):
        
        
        
        if type(nblocks) == int:
            if nblocks >= 1:
                if nblocks > self.nobs:
                    print("more blocks than num rows. Setting nblocks equal to nrows")
                    numBlocks = self.nobs 
                else:
                    numBlocks = nblocks
            else:
                print("Error: nblocks must be greater than 1, setting nblocks to 1")
                numBlocks = 1
        else:
            print("Error: nblocks must INT be greater than 1, setting nblocks to 1")
            numBlocks = 1
                    
            
        if self.nDataBlocks is not None:
            if(self.nDataBlocks != numBlocks):
                print("change of the number of blocks, resetting iterates automatically")
                self.resetIterate = True
        
        self.nDataBlocks = numBlocks
                                        
        self.partition = createApartition(self.nobs,self.nDataBlocks)
        
        if self.dataAdded == False:
            print("Must add data before calling run(). Aborting...")
            return -1
                
        if blocksPerIteration >= self.nDataBlocks:
            blocksPerIteration = self.nDataBlocks
            
        self.cyclicPoint = 0
        
        if self.numRegs == 0:
            self.numPSblocks = self.nDataBlocks
        else:
            # if all nonembedded regularizers have a linear op
            # then we add an additional dummy variable to projective splitting
            # corresponding to objective function
            allRegsHaveLinOps = True 
            i = 0
            for reg in self.allRegularizers:
                if reg.linearOp is None:                                        
                    allRegsHaveLinOps = False
                    lastReg = self.allRegularizers[-1]
                    if not (lastReg.linearOp is None):      
                        #swap the two regularizers to ensure 
                        #the last block corresponds to no linear op
                        self.allRegularizers[i] = lastReg
                        self.allRegularizers[-1] = reg
                    break
                i += 1
                    
            if allRegsHaveLinOps == True:
                self.numPSblocks = self.nDataBlocks + self.numRegs + 1
            else:
                self.numPSblocks = self.nDataBlocks + self.numRegs

        
            
        numPSvars = self.nvar + 1
            
        if (resetIterate == True) | (self.resetIterate == True):
                                       
            self.x = np.zeros((self.numPSblocks,numPSvars))            
            self.y = np.zeros((self.numPSblocks,numPSvars))
            self.w = np.zeros((self.numPSblocks,numPSvars))
            self.z = np.zeros(numPSvars)
            
            self.u = np.zeros((self.numPSblocks-1,numPSvars))
            self.resetIterate = False 
        
        if maxIterations is None:
            maxIterations = float('Inf')
        
        
        k = 0
        objective = []
        times = [0]
        primalErrs = []
        dualErrs = []
        ################################
        # BEGIN MAIN ALGORITHM LOOP
        ################################
        while(k < maxIterations):  
            #print('iteration = {}'.format(k))
            t0 = time.time()
            self.updateLossBlocks(blockActivation,blocksPerIteration)        # update all blocks (xi,yi) from (xi,yi,z,wi)
            self.updateRegularizerBlocks()            
            self.projectToHyperplane() # update (z,w1...wn) from (x1..xn,y1..yn,z,w1..wn)
            t1 = time.time()
            self.primalErr = 1.0
            self.dualErr = 1.0
            if keepHistory == True:
                objective.append(self.getObjective())
                times.append(times[-1]+t1-t0)
                primalErrs.append(self.primalErr)
                dualErrs.append(self.dualErr)
                
            if self.primalErr < primalTol:
                print("Less than primal tol, finishing run")
                break
            if self.dualErr < dualTol:
                print("Less than dual tol, finishing run")
                break
            k += 1
            
        
        if keepHistory == True:
            self.historyArray = [objective]
            self.historyArray.append(times)
            self.historyArray.append(primalErrs)
            self.historyArray.append(dualErrs)
            self.historyArray = np.array(self.historyArray)        
            
    def updateLossBlocks(self,blockActivation,blocksPerIteration):        
        
        if blockActivation == "greedy":            
            phis = np.sum((self.z - self.x)*(self.y - self.w),axis=1)
            phis = phis[0:self.nDataBlocks]
            
            if phis.min() >= 0:
                activeBlocks = np.random.choice(range(self.nDataBlocks),blocksPerIteration,replace=False)
            else:
                activeBlocks = phis.argsort()[0:blocksPerIteration]
        elif blockActivation == "random":
            activeBlocks = np.random.choice(range(self.nDataBlocks),blocksPerIteration,replace=False)
        elif blockActivation == "cyclic":
            
            activeBlocks = []
            i = 0
            currentPoint = self.cyclicPoint
            while(i<blocksPerIteration):
                activeBlocks.append(currentPoint)
                currentPoint += 1
                i += 1
                if currentPoint == self.nDataBlocks:
                    currentPoint = 0
            self.cyclicPoint = currentPoint     
            
            
        
        
        
        for i in activeBlocks:
            # update this block
            self.process.update(self,self.partition[i],i)
                
    def updateRegularizerBlocks(self):
        i = 0
        for reg in self.allRegularizers: 
            blockInd = self.nDataBlocks + i            
            if (reg.linearOp is None):
                Giz = self.z[1:len(self.z)]
            else:
                Giz = reg.linearOp.matvec(self.z[1:len(self.z)])
            t = Giz + reg.step*self.w[blockInd][1:len(self.z)]
            self.x[blockInd][1:len(self.z)] = reg.getProx(t)
            self.y[blockInd][1:len(self.z)] = reg.step**(-1)*(t - self.x[blockInd][1:len(self.z)])
            i += 1
            
    
    def projectToHyperplane(self):
        # compute u and v for data blocks
        if self.nDataBlocks == self.numPSblocks:
            self.u[0:(self.nDataBlocks-1)] = self.x[0:(self.nDataBlocks-1)]-self.x[-1]
        else:
            self.u[0:self.nDataBlocks] = self.x[0:self.nDataBlocks]-self.x[-1]
        v = sum(self.y[0:self.nDataBlocks])
        
        # compute u and v for regularizer blocks except the final regularizer 
        for i in range(len(self.allRegularizers)-1):
            if  (self.allRegularizers[i].linearOp is None):
                self.u[i+self.nDataBlocks] = self.x[i+self.nDataBlocks] - self.x[-1]                
                v += self.y[i+self.nDataBlocks]
            else:
                Gxn = self.allRegularizers[i].linearOp.matvec(self.x[-1])
                self.u[i+self.nDataBlocks] = self.x[i+self.nDataBlocks] - Gxn
                v += self.allRegularizers[i].linearOp.rmatvec(self.y[i+self.nDataBlocks])
        
        # compute v for final regularizer block
        i = len(self.allRegularizers) - 1  
        if i >= 0:
            if  (self.allRegularizers[i].linearOp is None):                
                v += self.y[i+self.nDataBlocks]
            else:
                v += self.allRegularizers[i].linearOp.rmatvec(self.y[i+self.nDataBlocks])
                        
        # compute pi
        pi = np.linalg.norm(self.u,'fro')**2 + self.gamma**(-1)*np.linalg.norm(v,2)**2
        
        # compute phi 
        if pi > 0:
            phi = self.z.dot(v)            
            
            if len(self.w)>1:               
                phi += np.sum(self.u*self.w[0:(self.numPSblocks-1)])
            
            phi -= np.sum(self.x*self.y)            
            # compute tau 
            tau = phi/pi
            # update z and w         
        
            self.z = self.z - self.gamma**(-1)*tau*v
            if len(self.w) > 1:              
                self.w[0:(len(self.w)-1)] = self.w[0:(len(self.w)-1)] - tau*self.u
                self.w[-1] = -np.sum(self.w[0:(len(self.w)-1)],axis=0)
            
        else:
            print("Gradient of the hyperplane is 0, converged")
        
        
        
        
        
        
        
            
        
    
#-----------------------------------------------------------------------------
class Regularizer(object):
    def __init__(self,value,prox,nu=1.0,step=1.0):
        self.value = value 
        self.prox = prox 
        if type(nu)==int:
            nu = float(nu)
        if type(step)==int:
            step = float(step)
            
        if type(nu)==float:
            if nu>=0:
                self.nu = nu    
            else:
                print("Error: scaling must be >=0, keeping it set to 1.0")
                self.nu = 1.0 
        else:
            print("Error: scaling must be float>=0, setting it to 1.0")
        if type(step)==float:
            if step>=0:
                self.step = step    
            else:
                print("Error: scaling must be >=0, keeping it set to 1.0")
        else:
            print("Error: scaling must be float>=0, setting it to 1.0")
                
    
    def addLinear(self,linearOp):        
        self.linearOp = linearOp        
        
    def setScaling(self,nu):
        if type(nu)==float:
            if nu>=0:
                self.nu = nu    
            else:
                print("Error: scaling must be >=0, keeping it set to 1.0")
                self.nu = 1.0 
        else:
            print("Error: scaling must be float>=0, setting it to 1.0")
            self.nu = 1.0 

    def setStep(self,step):
        if type(step)==float:
            if step>=0:
                self.step = step    
            else:
                print("Error: stepsize must be >=0, keeping it set to 1.0")
                self.step = 1.0 
        else:
            print("Error: stepsize must be float>=0, setting it to 1.0")
            self.step = 1.0 
            
    def getScalingAndStepsize(self):
        return self.nu,self.step  
    
    def evaluate(self,x):
        return self.value(x,self.nu)
    
    def getProx(self,x):
        return self.prox(x,self.nu,self.step)
    
def L1val(x,nu):
    return nu*np.linalg.norm(x,1)

def L1prox(x,nu,rho):
    rhonu = rho * nu
    out = (x> rhonu)*(x-rhonu)
    out+= (x<-rhonu)*(x+rhonu)
    return out

def L1(scaling=1.0,step=1.0):
    out = Regularizer(L1val,L1prox,scaling,step)    
    return out 

def partialL1(dimension,groups,scaling = 1.0):
    pass

def groupL2(dimension,groups,scaling = 1.0):
    pass
    
#-----------------------------------------------------------------------------
class Loss(object):
    def __init__(self,p):
        if (type(p) == int) | (type(p) == float): 
            if (p>=1):
                self.value = lambda yhat,y: (1.0/p)*abs(yhat-y)**p         
                if(p>1):
                    self.derivative = lambda yhat,y:  (2.0*(yhat>=y)-1.0)*abs(yhat-y)**(p-1)
                else:
                    self.derivative = None 
                    
            else:
                print("Error, lossFunction is not >= 1")
        elif(p == 'logistic'):  
            self.value = lambda yhat,y: LR_loss(yhat,y)
            self.derivative = lambda yhat,y: LR_derivative(yhat,y)
            
        elif(type(p) == LossPlugIn):
            self.value = p.value 
            self.derivative = p.derivative 
        else:
            print("lossFunction input error")


def LR_loss(yhat,y):
    score = -yhat*y
    
    return LR_loss_from_score(score)

def LR_loss_from_score(score):
    pos = np.log(1 + np.exp(score))
    pos2 = (~np.isinf(pos))*np.nan_to_num(pos)
    neg = score + np.log(1+np.exp(-score))
    neg2 = (~np.isinf(neg)) * np.nan_to_num(neg)
    coef = 0.5*np.ones(len(pos))
    coef = coef+0.5*np.isinf(pos)+0.5*np.isinf(neg)
    return coef*(pos2+neg2) 

def LR_derivative(yhat,y):
    score = -yhat*y    
    return -np.exp(score - LR_loss_from_score(score))*y

    
class LossPlugIn(object):
    def __init__(self,value,derivative):
        self.value = value
        self.derivative = derivative


#-----------------------------------------------------------------------------
class ProjSplitLossProcessor(object):
    @staticmethod
    def getAGrad(psObj,point,thisSlice,block):
        yhat = point[0]+psObj.A[thisSlice].dot(point[1:len(point)])
        gradL = psObj.loss.derivative(yhat,psObj.yresponse[thisSlice])        
        grad = (1.0/psObj.nobs)*psObj.A[thisSlice].T.dot(gradL)
        if psObj.intercept:            
            grad0 = np.array([(1.0/psObj.nobs)*sum(gradL)])
        else:
            grad0 = np.array([0.0])
        grad = np.concatenate((grad0,grad))
        return grad        
    

#############
class Forward2Fixed(ProjSplitLossProcessor):
    def __init__(self,step=1.0):
        self.step = step
        
    def update(self,psObj,thisSlice,block):        
        gradz = ProjSplitLossProcessor.getAGrad(psObj,psObj.z,thisSlice,block)
        
        psObj.x[block] = psObj.z - self.step*(gradz - psObj.w[block])        
        gradx = ProjSplitLossProcessor.getAGrad(psObj,psObj.x[block],thisSlice,block)        
        psObj.y[block] = gradx        
        
class Forward2Backtrack(ProjSplitLossProcessor):
    def __init__(self,initialStep,acceptThreshold,backtrackFactor,
                 growFactor=1.0,growFreq=None):
        pass 
    def update(self,psObj,thisSlice):
        pass
    
class Forward2Affine(ProjSplitLossProcessor):
    def __init__(self,acceptThreshold):
        pass
    def update(self,psObj,thisSlice):
        pass
    
class  Forward1Fixed(ProjSplitLossProcessor):
    def __init__(self,stepsize, blendFactor=0.1,includeInterceptTerm = False):
        pass
    def update(self,psObj,thisSlice):
        pass

class Forward1Backtrack(Forward1Fixed):
    def __init__(self,initialStep, blendFactor=0.1,includeInterceptTerm = False, 
                      backTrackFactor = 0.7, growFactor = 1.0, growFreq = None):
        pass
    def update(self,psObj,thisSlice):
        pass
    
    
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
        
#-----------------------------------------------------------------------------

def totalVariation1d(n):
    pass

def dropFirst(n):
    pass

#-----------------------------------------------------------------------------
    
def createApartition(nrows,n_partitions):
    
    if nrows%n_partitions == 0:
        partition_size = nrows // n_partitions
        partition_list = [range(i*partition_size,(i+1)*partition_size) for i in range(0,n_partitions)]
    else:
        n_with_ceil = nrows%n_partitions
        flr = nrows//n_partitions
        ceil = flr+1
        partition_list = [range(i*ceil,(i+1)*ceil) for i in range(0,n_with_ceil)]            
        endFirstPart = n_with_ceil*ceil
        partition_list.extend([range(endFirstPart + i*flr, endFirstPart + (i+1)*flr) 
                                for i in range(0,n_partitions - n_with_ceil)])

    return partition_list




