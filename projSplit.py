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
        self.nvar = None 
        self.z = None
        self.allRegularizers = []
        self.embedded = None         
        self.process = None 
        self.A = None
        self.numRegs = 0
    
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
                            
    def addData(self,observations,responses,lossFunction,process,interceptTerm=True,
                normalize=True,nblocks=10): 
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
            self.y = None
            return -1 
        
        self.y = responses
        
        if normalize == True:
            print("Normalizing columns of A to unit norm")
            self.A = np.copy(observations)            
            self.scaling = np.linalg.norm(self.A,axis=0)
            self.scaling += 1.0*(self.scaling < 1e-10)
            self.A = self.A/self.scaling
        else:
            print("Not normalizing columns of A")            
            self.A = observations
        
        self.loss = Loss(lossFunction)
               
        self.process = process
        
        if(self.embedded != None):
            # check that process is the right type
            pass
        
                                                                
        if interceptTerm == True:
            self.intercept = 0.0 
        else:
            self.intercept = None
        
        if type(nblocks) == int:
            if nblocks >= 1:
                if nblocks > self.nobs:
                    print("more blocks than num rows. Setting nblocks equal to nrows")
                    self.nDataBlocks = self.nobs 
                else:
                    self.nDataBlocks = nblocks
            else:
                print("Error: nblocks must be greater than 1, setting nblocks to 1")
                self.nDataBlocks = 1
        else:
            print("Error: nblocks must INT be greater than 1, setting nblocks to 1")
            self.nDataBlocks = 1
                                        
        self.partition = createApartition(self.nobs,self.nDataBlocks)
        
        
        
        
    
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
        if self.nvar == None:
            return None 
        else:        
            return (self.nvar + int(self.intercept!=None),self.nobs,self.nDataBlocks)
    
    def addRegularizer(self,regObj, linearOp=None, embed = False):
        #self.allRegularizers
        #reg.linearOp        
        if embed == True:
            if (self.linearOp != None) & (linearOp != self.linearOp):            
                print("embedded regularizer must use the same")
                print("linear op as the loss.")
                print("Aborting without adding regularizer")
                return -1
            
            
            if self.process != None:
                # test that processOp is of the right type
                pass
            self.embedded = regObj            
        else:            
            self.allRegularizers.append(regObj)
            
        regObj.addLinear(linearOp)
        self.numRegs += 1
        self.z = None # Ensures we reset the variables if we add another regularizer 
    
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
        #reg.interceptTerm
        if self.z == None:
            return None
        
        Az = self.A.dot(self.z)
        if self.intercept != None:
            Az += self.intercept*self.onesIntercept    
        currentLoss = (1.0/self.nobs)*self.loss.value(Az,self.y)
        for reg in self.allRegularizers:
            Hiz = reg.linearOp(self.z)            
            currentLoss += reg.evaluate(Hiz)
        if self.embedded != None:
            reg = self.embedded 
            Hiz = reg.linearOp(self.z)
            currentLoss += reg.evaluate(Hiz)
            
        return currentLoss
        
    
    def getSolution(self):
        #self.normalize
        #self.scale 
        if self.z == None:
            return None
        
        if self.normalize:
            out = self.z*self.scaling             
        else:
            out =  self.z  

        return [self.intercept,out]
        
    
    def getPrimalViolation(self):
        #self.primalErr
        '''
        After at least one call to run, the function call
        objVal = psfObj.getPrimalViolation()
        returns a float containing max_i{||G_i z^k - x_i^k||_\infty}. 
        If run has not been called yet, it returns None.
        '''
        if self.z == None:
            pass
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
        if self.z == None:
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
        if(self.z == None):
            return None
        
        if (self.keepHistory == False):
            return None
        else:
            return self.historyArray
    
    def run(self,primalTol = 1e-6, dualTol=1e-6,maxIterations=None,keepHistory = False, 
            blockActivation="greedy", blocksPerIteration=1, resetIterate=False):
        
        if self.A == None:
            print("Must add data before calling run(). Aborting...")
            return -1
        
        if self.numRegs == 0:
            self.numPSblocks = self.nDataBlocks + 1
        else:
            # if all nonembedded regularizers have a linear op
            # then we add an additional dummy variable to projective splitting
            # corresponding to objective function
            allRegsHaveLinOps = True 
            for reg in self.allRegularizers:
                if reg.linearOp == None:
                    allRegsHaveLinOps = False
                    break
                    
            if allRegsHaveLinOps == True:
                self.numPSblocks = self.nDataBlocks + self.numRegs + 1
            else:
                self.numPSblocks = self.nDataBlocks + self.numRegs
        
            
        numPSvars = self.nvar + int(self.intercept!=None)
            
        if (resetIterate == True) | (self.z == None):
            
                
            
            self.x = np.zeros((self.numPSblocks,numPSvars))            
            self.y = np.zeros((self.numPSblocks,numPSvars))
            self.w = np.zeros((self.numPSblocks,numPSvars))
            self.z = np.zeros(self.numPSvars)
            
            self.u = np.zeros((self.numPSblocks,numPSvars))
            
        
        if maxIterations == None:
            maxIter = float('Inf')
        
        k = 0
        objective = []
        times = [0]
        primalErrs = []
        dualErrs = []
        ################################
        # BEGIN MAIN ALGORITHM LOOP
        ################################
        while(k < maxIter):
            t0 = time.time()
            self.updateBlock(blockActivation,blocksPerIteration)        # update all blocks (xi,yi) from (xi,yi,z,wi)
            
            self.projectToHyperplane() # update (z,w1...wn) from (x1..xn,y1..yn,z,w1..wn)
            t1 = time.time()
            self.primalErr = None
            self.dualErr = None
            if keepHistory == True:
                objective.append(None)
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
            
    def updateBlock(self,blockActivation,blocksPerIteration):
        pass
    
    def projectToHyperplane(self):
        # compute u and v
        self.u[0:self.nDataBlocks] = self.x[0:self.nDataBlocks]-self.x[-1]
        v = sum(self.y[0:self.nDataBlocks])
        
        for i in range(self.numPSblocks - self.nDataBlocks - 1):
            if self.allRegularizers[i].linearOp == None:
                self.u[i+self.nDataBlocks] = self.x[i+self.nDataBlocks] - self.x[-1]
                v += self.y[i+self.nDataBlocks]
            else:
                Gxn = self.allRegularizers[i].linearOp.matvec(self.x[-1])
                self.u[i+self.nDataBlocks] = self.x[i+self.nDataBlocks] - Gxn
                v += self.allRegularizers[i].linearOp.rmatvec(self.y[i+self.nDataBlocks])
                        
        # compute pi
        pi = np.linalg.norm(self.u,'fro')**2 + self.gamma**(-1)*np.linalg.norm(v,2)**2
        
        # compute phi 
        if pi > 0:
            phi = self.z.dot(v)
            phi += sum(self.u*self.w[0:(self.numPSblocks-1)])
            phi -= sum(self.x*self.y)
        
        # compute tau 
        
        # update z
        
        #update w
        
            
        
    
#-----------------------------------------------------------------------------
class Regularizer(object):
    def __init__(self,value,prox):
        self.value = value 
        self.prox = prox
        self.nu = 1.0
    
    def addLinear(self,linearOp):        
        self.linearOp = linearOp        
        
    def setScaling(self,nu):
        if type(nu)==float:
            if nu>=0:
                self.nu = nu    
            else:
                print("Error: scaling must be >=0, keeping it set to 1.0")
        else:
            print("Error: scaling must be float>=0, setting it to 1.0")

    def getScaling(self):
        return self.nu 
    
    def evaluate(self,x):
        return self.value(x,self.nu)
    
    def getProx(self,x,rho):
        return self.prox(x,self.nu,rho)
    
def L1val(x,nu):
    return nu*np.linalg.norm(x,1)

def L1prox(x,nu,rho):
    rhonu = rho * nu
    out = (x> rhonu)*(x-rhonu)
    out+= (x<-rhonu)*(x+rhonu)
    return out

def L1(scaling=1.0):
    out = Regularizer(L1val,L1prox)
    out.setScaling(scaling)
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
    pass

#############
class Forward2Backtrack(ProjSplitLossProcessor):
    def __init__(self,initialStep,acceptThreshold,backtrackFactor,
                 growFactor=1.0,growFreq=None):
        pass
class Forward2Affine(ProjSplitLossProcessor):
    def __init__(self,acceptThreshold):
        pass
    
class  Forward1Fixed(ProjSplitLossProcessor):
    def __init__(self,stepsize, blendFactor=0.1,includeInterceptTerm = False):
        pass

class Forward1Backtrack(Forward1Fixed):
    def __init__(self,initialStep, blendFactor=0.1,includeInterceptTerm = False, 
                      backTrackFactor = 0.7, growFactor = 1.0, growFreq = None):
        pass 
    
    
#############
class BackwardCG(ProjSplitLossProcessor):
    def __init__(self,relativeErrorFactor):
        pass

class BackwardLBFGS(ProjSplitLossProcessor):
    def __init__(self,relativeErrorFactor = 0.9,memory = 10,c1 = 1e-4,
                 c2 = 0.9,shrinkFactor = 0.7, growFactor = 1.1):
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




