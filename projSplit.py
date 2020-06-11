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


import numpy as np
from numpy.linalg import norm
import time 

class ProjSplitFit(object):
    '''
    ProjSplitFit is the class used for creating a data-fitting problem and solving
    it with projective splitting. Please refer to 
    [1] arxiv.org/abs/1803.07043 (algorithm definition page 9)
    [2] arxiv.org/abs/1902.09025 (algorithm definiteion pages 10-11)
    - To create an object, call
        psobj = ProjSplitFit(dualScaling)
        dualScaling (defaults to 1.0) is gamma in the above algorithm definitions. 
    
    The general optimization objective this can solve is
    
    (1) min_(z,z_int){ (1.0/n)*sum_{i=1}^n loss(z_int + a_i^T (G_0 z),y_i)
                        + sum_{k = 1}^{numReg} h_i(G_i z) }
    
    where
        - a_1...a_n are feature vectors forming the rows of a data matrix A
        - y_1...y_n are the response values
        - loss is the loss function dealt with via the Loss class defined in this 
            module (see the addData method)
        - G_0...G_{numReg} are linear operators (possibly the identity)
        - h_i are generic functions dealt with via the Regularizer class defined 
            in this module (see addRegularizer)
        - z_int is the intercept variable to be fit
        - z is a vector of parameters to be fit
            
    The data (A,y), loss, and linear operator G_0 are added via the addData method.
    
    Each regularizer must be added individually via a call to the addRegularizer
    method.                 
    '''
    def __init__(self,dualScaling=1.0):
        '''
        ----------
        parameters
        ----------
            dualScaling (defaults to 1.0) is gamma in 
            [1] arxiv.org/abs/1803.07043 (algorithm definition page 9)
            [2] arxiv.org/abs/1902.09025 (algorithm definiteion pages 10-11).
            dualScaling must be an int or float > 0
        '''        
        self.setDualScaling(dualScaling)                
        self.allRegularizers = []
        self.embeddedFlag = False
        self.process = None 
        self.dataAdded = False
        self.numRegs = 0
        self.z = None
        self.nDataBlocks = None        
        
    
    def setDualScaling(self,dualScaling):
        '''
        Changes the dual scaling parameter (gamma), i.e.
        psobj.setDualScaling(newGamma) where newGamma is a positive int or float
        '''
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
        '''
        Returns the current setting of dualScaling
        '''
        return self.gamma 
                            
    def addData(self,observations,responses,loss,process,intercept=True,
                normalize=True, linearOp = None): 
        '''        
        Adds data for the data fitting model
        -----------
        Parameters
        ----------
        -observations: n x d numpy array/matrix with each row being a_i 
        from Eq. (1).
        -responses: n x 1 vector/list/numpy array with each element equal to 
        y_i from Eq. (1)
        -process: Selects a process object which sets how projective 
        splitting will handle the loss. Process is an object of a class derived
        from ProjSplitLossProcessor. Here are the possible classes:
        Forward2Fixed, Forward2Backtrack, Forward2Affine, Forward1Fixed, 
        Forward1Backtrack, BackwardCG, BackwardLBFGS, BackwardExact       
        -intercept: (bool) whether to include an intercept/constant term in the 
            linear model
        -normalize: (bool) whether to normalize columns of the data matrix to have 
        unit norm (if true, data matrix will be copied)
        -linearOp: adds matrix G_0 in Eq. (1)
        '''
        
        try:
            (self.nobs,self.ncol) = observations.shape
            ny = len(responses)
        except:
            print("Error: observations and responses should be arrays, i.e. ")
            print("NumPy arrays. They must have a shape attribute. Aborting, did not add data")            
            raise Exception

        if (self.nobs!=ny):
            print("Error: len(responses) != num observations")
            print("aborting. Data not added")
            raise Exception
        
        if (self.nobs == 0) | (self.ncol == 0):
            print("Error! number of observations or variables is 0! Aborting addData")
            print("Please call with valid data, i.e. numpy arrays")
            self.A = None
            self.yresponse = None
            raise Exception 
        
        if process.pMustBe2 and (loss != 2):
            print("Warning: this process object only works for the squared loss")
            print("Using Forward2Backtrack() as the process object")
            process = Forward2Backtrack()
            
        
        if linearOp is None:
            self.dataLinOp = MyLinearOperator(matvec=lambda x:x,rmatvec=lambda x:x)
            self.nvar = self.ncol
            self.dataLinOpFlag = False
        elif linearOp.shape[0] != self.ncol:
            print("Error! number of columns of the data matrix is {}".format(self.ncol))
            print("while number of rows of the composed linear operator is {}".format(linearOp.shape[0]))
            print("These must be equal! Aborting addData call")
            self.A = None
            self.yresponse = None
            self.nvar = None
            raise Exception
        else:
            self.dataLinOp = linearOp
            self.nvar = linearOp.shape[1]
            self.dataLinOpFlag = True 
            
        
        # check that all of the regularizers added so far have linear ops 
        # which are consistent with the added data
        for reg in self.allRegularizers:
            if reg.linearOpFlag == True:
                if reg.linearOp.shape[1] != self.nvar:
                    print("ERROR: linear operator added with a regularizer")
                    print("has number of columns which is inconsistent with the added data")
                    print("Added data has {} columns".format(self.nvar))
                    print("A linear operator has {} columns".format(reg.linearOp.shape[1]))
                    print("These must be equal, aborting add data")
                    self.A = None
                    self.yresponse = None
                    self.nvar = None
                    raise Exception
        
        self.yresponse = responses
        
        if normalize == True:
            print("Normalizing columns of A to unit norm")
            self.normalize = True
            self.A = np.copy(observations)            
            self.scaling = norm(self.A,axis=0)
            self.scaling += 1.0*(self.scaling < 1e-10)
            self.A = self.A/self.scaling
        else:
            print("Not normalizing columns of A")            
            self.A = observations
            self.normalize = False 
        
        self.loss = Loss(loss)
               
        self.process = process
        
        if (self.embeddedFlag == True) & (self.process.embedOK == False):
            print("WARNING: A regularizer was added with embedded = True")
            print("But embedding is not possible with this process object")
            print("Moving embedded regularizer to be an ordinary regularizer")
            regObj = self.embedded 
            self.allRegularizers.append(regObj)
            self.embedded = None
            self.embeddedFlag = False
            self.numRegs += 1
            
                
        self.intercept = intercept                        
        self.dataAdded = True
        self.resetIterate = True
        
                    
    def getParams(self):
        '''
        After the addData method has been called, one may call the getDataParams method to
        retrieve some of the relevant parameters in the model.
        nvar,nobs = psfObj.getParams()
        Here
        nvar: Number of variables
        nobs: Number of observations

        If the addData method has not been called yet, getParams raises an exception.
        '''
        if self.dataAdded == False:
            raise Exception
        else:        
            return (self.nvar + int(self.intercept),self.nobs)
    
    def addRegularizer(self,regObj, linearOp=None, embed = False):
        '''        
        -adds a regularizer to the optimization problem (each h_i and G_i in 
          Eq. (1)))
        ----------
        parameters
        ----------
        -regObj: an object of class Regularizer (defined in this module)
        -linearOp: adds matrix G_i in Eq. (1)
        -embed: internal option in projective splitting (see Eq. (8)-(9) in
          [1]. For forward-type loss process updates, perform the "prox" of 
          this regularizer in a forward-backward style update)
        '''
        
        if (linearOp is not None) & (self.dataAdded):
            #check the dimensions make sense            
            if linearOp.shape[1] != self.nvar:
                print("ERROR: linear operator added with this regularizer")
                print("has number of columns which is inconsistent with the added data")
                print("Added data has {} columns".format(self.nvar))
                print("This linear operator has {} columns".format(linearOp.shape[1]))
                print("These must be equal, aborting addRegularizer")
                raise Exception 
            
        if embed == True:
            OK2Embed = True
            if (self.process is not None): 
                if(self.process.embedOK == False):                
                    print("WARNING: This regularizer was added with embedded = True")
                    print("But embedding is not possible with the process object used in addData()")
                    print("Moving embedded regularizer to be an ordinary regularizer")
                    self.allRegularizers.append(regObj)
                    self.numRegs += 1
                    OK2Embed = False                
                    
            if OK2Embed & (linearOp is not None):                                
                print("WARNING: embedded regularizer cannot use linearOp != None")                
                print("Moving embedded regularizer to be an ordinary regularizer")
                self.allRegularizers.append(regObj)
                self.numRegs += 1  
                OK2Embed = False
            
            if OK2Embed:
                if self.embeddedFlag == True:
                    print("Warning: Regularizer embed option set to True")
                    print("But an earlier regularizer is already embedded")
                    print("Can only embed one regularizer")
                    print("Moving old regularizer to be non-embedded, the new regularizer will be embedded")
                    self.allRegularizers.append(self.embedded)
                    self.numRegs += 1
                        
                self.embedded = regObj 
                self.embeddedFlag = True                 
                
        else:            
            self.allRegularizers.append(regObj)
            self.numRegs += 1
        
        if linearOp is not None:     
            regObj.addLinear(linearOp,True)
        else:
            regObj.addLinear(
                MyLinearOperator(matvec=lambda x:x,rmatvec=lambda x:x), False)
        
        self.resetIterate = True # Ensures we reset the variables if we add another regularizer 
    
    def getObjective(self):
        '''
        Returns the current objective value, as in (1), evaluated at the current 
        primal iterate z^k. If the method has not been run yet, raises an exception
        '''        
        if self.z is None :
            print("Method not run yet, no objective to return. Call run() first.")
            raise Exception
                
        currentLoss,Hz = self.__getLoss(self.z)
        
        for reg in self.allRegularizers:
            Hiz = reg.linearOp.matvec(self.z[1:len(self.z)])              
            currentLoss += reg.evaluate(Hiz)
                
        if self.embeddedFlag == True:
            reg = self.embedded         
            currentLoss += self.embeddedScaling*reg.evaluate(Hz)/reg.getScalingAndStepsize()[0]
        
        return currentLoss
        
    
    def getSolution(self):
        '''
        Returns the current primal Solution z^k. If the run() has not been called yet,
        raises an exception. 
        '''
        
        if self.z is None:
            print("Method not run yet, no solution to return. Call run() first.")
            raise Exception
        
        
        Hz = self.dataLinOp.matvec(self.z[1:len(self.z)])
        out = np.zeros(self.ncol+1)
        
        
        
        if self.normalize:
            out[1:(self.ncol+1)] = Hz/self.scaling
        else:
            out[1:(self.ncol+1)] = Hz 

        if self.intercept:
            out[0] = self.z[0]
            return out,self.z 
        else:
            return out[1:(self.ncol+1)],self.z 
        
        
    
    def getPrimalViolation(self):        
        '''
        After at least one call to run, the function call
        objVal = psfObj.getPrimalViolation()
        returns a float containing max_i{||G_i z^k - x_i^k||_2}. 
        If run has not been called yet, raises an exception.
        '''
        if self.z is None:
            print("Method not run yet, no primal violation to return. Call run() first.")
            raise Exception
        else:
            return self.primalErr
    
    def getDualViolation(self):
        '''
        After at least one call to run, the function call
        objVal = psfObj.getDualViolation()
        returns a float containing max_i{||y_i^k - w_i^k||_2}.  
        If run has not been called yet, it raises an exception
        '''
        if self.z is None:
            print("Method not run yet, no dual violation to return. Call run() first.")
            raise Exception
        else:
            return self.dualErr
    
    def getHistory(self):
        '''
        After at least one call to run with keepHistory set to True, the function call
        historyArray = psfObj.getHistory()
        returns a two-dimensional four-column NumPy array with each row 
        corresponding to an iteration for which the history statistics were 
        recorded. The total number of rows is num iterations divided by the 
        historyFreq parameter, which can be set in run() and defaults to 10. 
        In each column of this array, the rows have the following interpretation:
        Row Number Interpretation
        0             Objective value
        1             Cumulative run time
        2             Primal violation
        3             Dual violation
        4             Value of Phi(p^k) used in hyperplane construction 
        
        If run() has not yet been called with keepHistory set to True, 
        this function will raise an Exception when called. 
        '''
        if self.z is None:
            print("Method not run yet, no history to return. Call run() first.")
            raise Exception
        if self.historyArray is None:
            print("run() was called without the keepHistory option, no history")
            print("call run with the keepHistory argument set to True")
            raise Exception
        return self.historyArray
    
    
    
    
    def runSGD(self,maxIterations,nblocks=1,historyFreq=10,step0 = 1.0,
               stepStat="fixed",exponent = 0.75):
        '''
            returns the SGD function value history 
        '''
        if self.dataAdded == False:
            print("Must add data before calling runSGD(). Aborting...")
            raise Exception        
            
        # check that there are no regularizers as regularizers are not supported 
        # by this implementation of SGD. 
        if (self.numRegs > 0) or (self.embeddedFlag == True):
            print("Error: Tried to run SGD but there are regularizers present")
            print("Aborting")
            raise Exception 
        
        
        numBlocks = self.__setBlocks(nblocks)
        SGDpartition = createApartition(self.nobs,numBlocks)
        
        zsgd = np.zeros(self.nvar+1)
        Hzsgd = np.zeros(self.ncol+1)
        Fsgd = []
        sgdtimes = [0]
        k = 0
        step = step0
        
        while(k<maxIterations):            
            t0 = time.time()
            Hzsgd[1:] = self.dataLinOp.matvec(zsgd[1:])
            Hzsgd[0] = zsgd[0]
            activeBlock = np.random.randint(0,numBlocks)
            thisSlice = SGDpartition[activeBlock]
            gradHz = ProjSplitLossProcessor.getAGrad(self,Hzsgd,thisSlice)
            Gtgrad = self.dataLinOp.rmatvec(gradHz[1:])
            Gtgrad = np.concatenate((np.array([gradHz[0]]),Gtgrad))
            
            zsgd = zsgd - step*Gtgrad
            t1 = time.time()
            if step == "decay":
                step = step0*(k+2)**(exponent)
                
            if k%historyFreq == 0:
                fsgd,_ = self.__getLoss(zsgd)
                Fsgd.append(fsgd)                
                sgdtimes.append(sgdtimes[-1]+t1-t0)
            k += 1
        return Fsgd,sgdtimes[1:]
            
    
    def run(self,primalTol = 1e-6, dualTol=1e-6,maxIterations=None,keepHistory = False, 
            historyFreq = 10, nblocks = 1, blockActivation="greedy", blocksPerIteration=1, 
            resetIterate=False):
        '''
        Run projective splitting to minimize the objective in (1).
        ----------
        parameters
        ----------
        primalTol: (float>0) Continue running algorithm if primal error, 
            max_i ||G_i z^k - x_i^k||_2, is greater than primalTol. 
            Note that to terminate the method, both primal error AND dual error 
            must be smaller than their respective tolerances.
        dualTol: (float>0) Continue running algorithm if dual error, 
            max_i ||y_i^k - w_i^k||_2, is greater than dualTol. 
            Note that to terminate the method, both primal error AND dual error 
            must be smaller than their respective tolerances.
        maxIterations: (int>0) terminate algorithm if ran for more than 
            maxIterations iterations
        keepHistory: (bool) if True, record the history (see getHistory() method)
        historyFreq: (int>0) frequency to keep history, defaults to every 10 
            iterations. Note that to keepHistory requires computing the objective
            which may be slow for large problems. Hence we allow historyFreq
            to be set larger than 1. 
        nBlocks: (int>0) number of blocks in the projective splitting decomposition 
            of the loss
        blockActivation: (string) strategy for selecting blocks of the loss in
        projective splitting. Defaults to "greedy". Other valid choices are
            "random" and "cyclic". 
        blocksPerIteration: (int>0) Number of blocks to update in each iteration
        resetIterate: (bool) if True, the current values of all variables in
            projective splitting (eg: z^k, w_i^k etc) are erased and initialized 
            to 0. 
        '''
        
        if self.dataAdded == False:
            print("Must add data before calling run(). Aborting...")
            raise Exception        
        
        if (blockActivation != "greedy") and (blockActivation != "cyclic") \
            and (blockActivation != "greedy"):
                print("Warning: chosen blockActivation is not recognised")
                print("Using greedy instead")
                blockActivation = "greedy"
                                
        numBlocks = self.__setBlocks(nblocks)
                                
        if self.nDataBlocks is not None:
            if(self.nDataBlocks != numBlocks):
                print("change of the number of blocks, resetting iterates automatically")
                self.resetIterate = True
        
        self.nDataBlocks = numBlocks
                                        
        self.partition = createApartition(self.nobs,self.nDataBlocks)
                
                
        if blocksPerIteration >= self.nDataBlocks:
            blocksPerIteration = self.nDataBlocks
            
        self.cyclicPoint = 0
        
        if self.embeddedFlag == False:
            # if no embedded reg added, create an artificial embedded reg
            # with a "pass-through prox
            self.embedded = Regularizer(None,(lambda x,nu,step:x))
        else:
            if self.embedded.getScalingAndStepsize()[1] != self.process.getStep():
                print("WARNING: embedded regularizer must use the same stepsize as the Loss update process")
                print("Setting the embedded regularizer stepsize to be the process stepsize")
                self.embedded.setStep(self.process.getStep())
            
            # the scaling used must be divided down by the number of blocks because
            # this term is distributed equally over all loss blocks
            self.embeddedScaling = self.embedded.getScalingAndStepsize()[0]
            self.embedded.setScaling(self.embeddedScaling/self.nDataBlocks)
        
        if self.numRegs == 0:
            if self.dataLinOpFlag == False:
                self.numPSblocks = self.nDataBlocks
            else:
                # if there are no regularizers and the data term is composed 
                # with a linear operator, we must add a dummy regularizer
                # which has a pass-through prox and 0 value
                self.addRegularizer(Regularizer(lambda x,nu: 0, lambda x,nu,step: x))                 
        
        if self.numRegs != 0:                
            # if all nonembedded regularizers have a linear op
            # then we add an additional dummy variable to projective splitting
            # corresponding to objective function
            allRegsHaveLinOps = True 
            i = 0
            for reg in self.allRegularizers:
                if reg.linearOpFlag == False:                                        
                    allRegsHaveLinOps = False
                    lastReg = self.allRegularizers[-1]
                    if (lastReg.linearOpFlag == True):      
                        #swap the two regularizers to ensure 
                        #the last block corresponds to no linear op
                        self.allRegularizers[i] = lastReg
                        self.allRegularizers[-1] = reg
                    break
                i += 1
                    
            if allRegsHaveLinOps == True:
                self.addRegularizer(Regularizer(lambda x,nu: 0, lambda x,nu,step: x))                
                            
            self.numPSblocks = self.nDataBlocks + self.numRegs
            
                                
        self.nDataVars = self.ncol + 1
        
        self.gradxdata = None # this is a variable used in Forward1Fixed and Forward1Backtrack processors
        
        if (resetIterate == True) | (self.resetIterate == True):
            
            self.z = np.zeros(self.nvar+1)
            self.v = np.zeros(self.nvar+1)
            self.Hz = np.zeros(self.nDataVars)
            self.Hx = np.zeros(self.nDataVars)
            self.xdata = np.zeros((self.nDataBlocks,self.nDataVars))            
            self.ydata = np.zeros((self.nDataBlocks,self.nDataVars))
            self.wdata = np.zeros((self.nDataBlocks,self.nDataVars))

            if self.numRegs > 0:                        
                self.udata = np.zeros((self.nDataBlocks,self.nDataVars))
            else:
                self.udata = np.zeros((self.nDataBlocks - 1,self.nDataVars))
            
            self.xreg = []
            self.yreg = []
            self.wreg = []
            self.ureg = []
            
            i = 0
            for reg in self.allRegularizers:
                if i == self.numRegs - 1:
                        regVars = self.nvar + 1
                elif reg.linearOpFlag == True:
                        regVars = reg.linearOp.shape[0]                                                            
                else:
                    regVars = self.nvar 
                    
                self.xreg.append(np.zeros(regVars))    
                self.yreg.append(np.zeros(regVars))    
                self.wreg.append(np.zeros(regVars))    
                i += 1
                if i != self.numRegs:
                    self.ureg.append(np.zeros(regVars))    
                

            self.resetIterate = False 
        
        if maxIterations is None:
            maxIterations = float('Inf')
        
        
        self.k = 0
        objective = []
        times = [0]
        primalErrs = []
        dualErrs = []
        phis = []
        ################################
        # BEGIN MAIN ALGORITHM LOOP
        ################################
        while(self.k < maxIterations):  
            #print('iteration = {}'.format(self.k))
            t0 = time.time()
            # update all blocks (xi,yi) from (xi,yi,z,wi)
            self.__updateLossBlocks(blockActivation,blocksPerIteration)        
            self.__updateRegularizerBlocks()            
            
            if (self.primalErr < primalTol) & (self.dualErr < dualTol):
                print("primal and dual tolerance reached, finishing run")
                break            
            
            
            phi = self.__projectToHyperplane() # update (z,w1...wn) from (x1..xn,y1..yn,z,w1..wn)
            t1 = time.time()

            if (keepHistory == True) and (self.k % historyFreq == 0):
                objective.append(self.getObjective())
                times.append(times[-1]+t1-t0)
                primalErrs.append(self.primalErr)
                dualErrs.append(self.dualErr)
                phis.append(phi)
                
            
            self.k += 1
            
        
        if keepHistory == True:
            self.historyArray = [objective]
            self.historyArray.append(times[1:])
            self.historyArray.append(primalErrs)
            self.historyArray.append(dualErrs)
            self.historyArray.append(phis)
            self.historyArray = np.array(self.historyArray) 
        
        if self.embeddedFlag == True:
            # we modified the embedded scaling to deal with multiple num blocks
            # now set it back to the previous value
            self.embedded.setScaling(self.embeddedScaling)
            
    def __setBlocks(self,nblocks):
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
            print("Error: nblocks must be of type INT greater than 1, setting nblocks to 1")
            numBlocks = 1
        return numBlocks
            
            
    def __updateLossBlocks(self,blockActivation,blocksPerIteration):        
        
        self.Hz[1:(self.ncol+1)] = self.dataLinOp.matvec(self.z[1:len(self.z)])
        self.Hz[0] = self.z[0]
        if blockActivation == "greedy":            
            phis = np.sum((self.Hz - self.xdata)*(self.ydata - self.wdata),axis=1)            
            
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
            self.process.update(self,i)
        
        self.primalErr = norm(self.Hz - self.xdata,ord=2,axis=1).max()
        self.dualErr =   norm(self.ydata - self.wdata,ord=2,axis=1).max()
                
    def __updateRegularizerBlocks(self):
        i = 0        
        for i in range(self.numRegs-1):             
            reg = self.allRegularizers[i]
            Giz = reg.linearOp.matvec(self.z[1:len(self.z)])                        
            t = Giz + reg.step*self.wreg[i]
            self.xreg[i] = reg.getProx(t)
            self.yreg[i] = reg.step**(-1)*(t - self.xreg[i])
            primal_err_i = norm(Giz - self.xreg[i],2)
            if self.primalErr<primal_err_i:
                self.primalErr = primal_err_i
            dual_err_i = norm(self.wreg[i] - self.yreg[i],2)
            if self.dualErr<dual_err_i:
                self.dualErr = dual_err_i
                
            i += 1
            
        # update coefficients corresponding to the last block
        # including the intercept term
        if self.numRegs > 0:
            reg = self.allRegularizers[-1]
            t = self.z[1:len(self.z)] + reg.step*self.wreg[-1][1:len(self.z)]
            self.xreg[-1][1:len(self.z)] = reg.getProx(t)
            self.yreg[-1][1:len(self.z)] = reg.step**(-1)*(t - self.xreg[-1][1:len(self.z)])
            
            if self.intercept:
                t_intercept = self.z[0] + reg.step*self.wreg[-1][0]
                self.xreg[-1][0] = t_intercept                
            else:
                self.xreg[-1][0] = 0.0
                
            self.yreg[-1][0] = 0.0
            
            primal_err_i = norm(self.xreg[-1]-self.z,2)
            if self.primalErr<primal_err_i:
                self.primalErr = primal_err_i
                
            dual_err_i = norm(self.yreg[-1]-self.wreg[-1],2)
            if self.dualErr<dual_err_i:
                self.dualErr = dual_err_i
                
                                    
    def __projectToHyperplane(self):
                            
        # compute u and v for data blocks
        if self.numRegs > 0:
            self.Hx[1:(self.ncol+1)] = self.dataLinOp.matvec(self.xreg[-1][1:len(self.z)])
            self.Hx[0] = self.xreg[-1][0]
            self.udata = self.xdata - self.Hx            
        else:
            # if there are no regularizers, the last block corresponds to the 
            # last data block. Further, dataLinOp must be the identity
            self.udata = self.xdata[0:(self.nDataBlocks-1)] - self.xdata[-1]
            
        vin = sum(self.ydata)
        v = self.dataLinOp.rmatvec(vin[1:(self.ncol+1)])
        v = np.concatenate((np.array([vin[0]]),v))
        
        # compute u and v for regularizer blocks except the final regularizer 
        for i in range(self.numRegs - 1):
            Gxn = self.allRegularizers[i].linearOp.matvec(self.xreg[-1][1:len(self.z)])
            self.ureg[i] = self.xreg[i] - Gxn
            Gstary = self.allRegularizers[i].linearOp.rmatvec(self.yreg[i])
            v += np.concatenate((np.array([0.0]),Gstary))
                        
        # compute v for final regularizer block        
        if self.numRegs>0:            
            v += self.yreg[-1]
                                    
        # compute pi
        pi = norm(self.udata,'fro')**2 + self.gamma**(-1)*norm(v,2)**2
        for i in range(self.numRegs - 1):
            pi += norm(self.ureg[i],2)**2
                
        # compute phi 
        if pi > 0:
            phi = self.__getPhi(v)
                                    
            if phi > 0:               
                # compute tau 
                tau = phi/pi
                # update z and w         
            
                self.z = self.z - self.gamma**(-1)*tau*v
                if (len(self.wdata) > 1) | (self.numRegs > 0):              
                    if self.numRegs == 0:               
                        # if no regularizers, the linearOp corresponding to the 
                        # data block must be the identity
                        self.wdata[0:(self.nDataBlocks-1)] = self.wdata[0:(self.nDataBlocks-1)] - tau*self.udata
                        self.wdata[-1] = -np.sum(self.wdata[0:(self.nDataBlocks-1)],axis=0)
                    else:
                        self.wdata = self.wdata - tau*self.udata                        
                        negsumw = -np.sum(self.wdata,axis=0)
                        GstarNegSumw = self.dataLinOp.rmatvec(negsumw[1:len(negsumw)])
                        GstarNegSumw = np.concatenate((np.array([negsumw[0]]),GstarNegSumw))
                        for i in range(self.numRegs - 1):
                            self.wreg[i] = self.wreg[i] - tau*self.ureg[i]
                            Gstarw = self.allRegularizers[i].linearOp.rmatvec(self.wreg[i])
                            GstarNegSumw -= np.concatenate((np.array([0.0]),Gstarw))
                        
                        self.wreg[-1] = GstarNegSumw
            
        else:
            print("Gradient of the hyperplane is 0, converged")
            phi = None
        
        return phi
    
    def __getPhi(self,v):
        phi = self.z.dot(v)            
            
        if (len(self.wdata) > 1) | (self.numRegs > 0):       
            if self.numRegs == 0:
                phi += np.sum(self.udata*self.wdata[0:(self.numPSblocks-1)])
            else:
                phi += np.sum(self.udata*self.wdata)
            
            for i in range(self.numRegs - 1):
                phi += self.ureg[i].dot(self.wreg[i])
        
        phi -= np.sum(self.xdata*self.ydata)
        
        for i in range(self.numRegs):     
            phi -= self.xreg[i].dot(self.yreg[i])
            
        return phi
    
    def __getLoss(self,z):
        Hz = self.dataLinOp.matvec(z[1:])
        AHz = self.A.dot(Hz)
        if self.intercept:
            #XX do I need this if statement or does z[0] just stay at 0?
            AHz += z[0]
        currentLoss = (1.0/self.nobs)*sum(self.loss.value(AHz,self.yresponse))     
        return currentLoss,Hz
        
                
#-----------------------------------------------------------------------------
class Regularizer(object):
    '''
      Objects of this class are used as inputs to the addRegularizer method
      of class ProjSplitFit to define regularizers in the objective function. 
      Recall the objective function:
      (1) min_(z,z_int){ (1.0/n)*sum_{i=1}^n(z_int + loss(a_i^T (G_0 z),y_i)) 
                        + sum_{k = 1}^{numReg} h_i(G_i z) }
      The regularizer class essentially defines each h_i(G_i z) term via
      methods for evaluating the function h_i, its prox, and the matrix G_i. 
      The regularizer object defines these features for a single function h().
      Note the matrix G is added in the addRegularizer method of projSplitFit. 
    '''
    def __init__(self,value,prox,nu=1.0,step=1.0):
        '''
        ----------
        parameters
        ----------
        value: (function) must be a function of one parameter:  a numpy-style 
            array x. Value returns a float which is the value of h(x)
        '''
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
                
    
    def addLinear(self,linearOp,linearOpFlag):
        self.linearOp = linearOp        
        self.linearOpFlag = linearOpFlag        
        
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
        try:        
            if step>=0:
                self.step = float(step)
            else:
                print("Error: stepsize must be >=0, keeping it set to 1.0")
                self.step = 1.0 
        except:
            print("Error: stepsize must be float>=0, setting it to 1.0")
            self.step = 1.0 
            
    def getScalingAndStepsize(self):
        return self.nu,self.step  
    
    def evaluate(self,x):
        return self.value(x,self.nu)
    
    def getProx(self,x):
        return self.prox(x,self.nu,self.step)
    
def L1val(x,nu):
    return nu*norm(x,1)

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
        
        if(p == 'logistic'):  
            self.value = lambda yhat,y: LR_loss(yhat,y)
            self.derivative = lambda yhat,y: LR_derivative(yhat,y)            
        elif(type(p) == LossPlugIn):
            self.value = p.value 
            self.derivative = p.derivative 
        else:
            
            try:
                if (p>=1):
                    self.value = lambda yhat,y: (1.0/p)*abs(yhat-y)**p         
                    if(p>1):
                        self.derivative = lambda yhat,y:  (2.0*(yhat>=y)-1.0)*abs(yhat-y)**(p-1)
                    else:
                        self.derivative = None                     
                elif(p<1):
                    print("Error, lossFunction is not >= 1")
                    raise Exception                                                                      
            except:                              
                print("lossFunction input error")
                print("for loss, input either an int or double >= 1, 'logistic', or an object derived from class LossPlugIn")
                raise Exception
            
                
            


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
    pMustBe2 = False 
    @staticmethod
    def getAGrad(psObj,point,thisSlice):
        #point[0] is the intercept term
        #point[1:len(point)] are the coefficients and 
        #len(point) must equal the num cols of A. 
        yhat = point[0]+psObj.A[thisSlice].dot(point[1:len(point)])
        gradL = psObj.loss.derivative(yhat,psObj.yresponse[thisSlice])        
        grad = (1.0/psObj.nobs)*psObj.A[thisSlice].T.dot(gradL)
        if psObj.intercept:            
            grad0 = np.array([(1.0/psObj.nobs)*sum(gradL)])
        else:
            grad0 = np.array([0.0])
        grad = np.concatenate((grad0,grad))
        return grad  
      
    def getStep(self):
        return self.step
    
    def setStep(self,step):
        self.step = step
        
    def initializeGradXdata(self,psObj):
        '''
           this routine is used by Forward1Fixed 
           to initialize the gradients of xdata
        '''
        psObj.gradxdata = np.zeros(psObj.xdata.shape)
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            psObj.gradxdata[block] = self.getAGrad(psObj,psObj.xdata[block],thisSlice)        
        
    def initialize1fBacktrack(self,psObj):
        '''
           this routine is used by Foward1Backtrack
           to initialize the gradients of xdata, \hat{theta}, \hat{w}, xdata, and ydata
        '''
        # initalize theta_hat
        psObj.thetahat = np.zeros(psObj.xdata.shape)
        psObj.what = np.zeros(psObj.xdata.shape)
        psObj.gradxdata = np.zeros(psObj.xdata.shape)
        for block in range(psObj.nDataBlocks):
            thisSlice = psObj.partition[block]
            psObj.thetahat[block][1:psObj.nDataVars] = psObj.embedded.getProx(psObj.thetahat[block][1:psObj.nDataVars])
            psObj.thetahat[block][0] = 0.0
            psObj.what[block] = -psObj.embedded.getScalingAndStepsize()[1]**(-1)*psObj.thetahat[block]
            psObj.gradxdata[block] = self.getAGrad(psObj,psObj.thetahat[block],thisSlice)        
            psObj.what[block] += psObj.gradxdata[block]
        
        psObj.xdata = psObj.thetahat
        psObj.ydata = psObj.what
        

#############
class Forward2Fixed(ProjSplitLossProcessor):
    def __init__(self,step=1.0):
        self.step = step
        self.embedOK = True
        
    def update(self,psObj,block):        
        thisSlice = psObj.partition[block]
        gradHz = ProjSplitLossProcessor.getAGrad(psObj,psObj.Hz,thisSlice)
        t = psObj.Hz - self.step*(gradHz - psObj.wdata[block])        
        psObj.xdata[block][1:psObj.nDataVars] = psObj.embedded.getProx(t[1:psObj.nDataVars]) 
        psObj.xdata[block][0] = t[0]        
        a = self.step**(-1)*(t-psObj.xdata[block])
        gradx = ProjSplitLossProcessor.getAGrad(psObj,psObj.xdata[block],thisSlice)        
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
        gradHz = ProjSplitLossProcessor.getAGrad(psObj,psObj.Hz,thisSlice)
        if self.growFreq is not None:
            if psObj.k % self.growFreq == 0:
                # time to grow the stepsize
                self.step *= self.growFactor
                psObj.embedded.setStep(self.step)
                
        while True:
            t = psObj.Hz - self.step*(gradHz - psObj.wdata[block])        
            psObj.xdata[block][1:psObj.nDataVars] = psObj.embedded.getProx(t[1:psObj.nDataVars]) 
            psObj.xdata[block][0] = t[0]        
            a = self.step**(-1)*(t-psObj.xdata[block])
            gradx = ProjSplitLossProcessor.getAGrad(psObj,psObj.xdata[block],thisSlice)        
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
        gradHz = ProjSplitLossProcessor.getAGrad(psObj,psObj.Hz,thisSlice)
        lhs = gradHz - psObj.wdata[block]
        
        yhat = lhs[0]+psObj.A[thisSlice].dot(lhs[1:len(lhs)])        
        affinePart = (1.0/psObj.nobs)*psObj.A[thisSlice].T.dot(yhat)
        if psObj.intercept:            
            affine0 = np.array([(1.0/psObj.nobs)*sum(affinePart)])
        else:
            affine0 = np.array([0.0])
        affinePart = np.concatenate((affine0,affinePart))
        normLHS = norm(lhs,2)**2
        step = normLHS/(self.Delta*normLHS + lhs.T.dot(affinePart))
        psObj.xdata[block] = psObj.Hz - step*lhs
        psObj.ydata[block] = gradHz - step*affinePart
        
        
    
class  Forward1Fixed(ProjSplitLossProcessor):
    def __init__(self,stepsize, blendFactor=0.1):
        self.step = stepsize
        self.alpha = blendFactor
        self.embedOK = True 
        
    def update(self,psObj,block):
        if psObj.gradxdata is None:
            ProjSplitLossProcessor.initializeGradXdata(ProjSplitLossProcessor,psObj)
                        
        thisSlice = psObj.partition[block]
        t = (1-self.alpha)*psObj.xdata[block] +self.alpha*psObj.Hz \
            - self.step*(psObj.gradxdata[block] - psObj.wdata[block])
        psObj.xdata[block][1:psObj.nDataVars] = psObj.embedded.getProx(t[1:psObj.nDataVars]) 
        psObj.xdata[block][0] = t[0]   
        psObj.gradxdata[block] = ProjSplitLossProcessor.getAGrad(psObj,psObj.xdata[block],thisSlice) 
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
        
    def update(self,psObj,block):
        if psObj.gradxdata is None:
            ProjSplitLossProcessor.initialize1fBacktrack(ProjSplitLossProcessor,psObj)
        
        if self.growFreq is not None:
            if psObj.k % self.growFreq == 0:
                # time to grow the stepsize
                upper_bound = (1+self.alpha*self.eta)*self.step 
                desired_step = self.growFac*self.step
                self.step = min([upper_bound,desired_step])                     
                psObj.embedded.setStep(self.step)
                
        
        thisSlice = psObj.partition[block]
        
        phi = (psObj.Hz - psObj.xdata[block]).T.dot(psObj.ydata[block] - psObj.wdata[block])
        
        xold = np.copy(psObj.xdata[block])
        yold = np.copy(psObj.ydata[block])
        
        t1 = (1-self.alpha)*xold +self.alpha*psObj.Hz
        t2 = np.copy(psObj.gradxdata[block]) 
        t2 -= psObj.wdata[block]
        while True:
            t = t1 - self.step*t2
            psObj.xdata[block][1:psObj.nDataVars] = psObj.embedded.getProx(t[1:psObj.nDataVars]) 
            psObj.xdata[block][0] = t[0]   
            
            psObj.gradxdata[block] = ProjSplitLossProcessor.getAGrad(psObj,psObj.xdata[block],thisSlice) 
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
    
#-----------------------------------------------------------------------------

def totalVariation1d(n):
    pass

def dropFirst(n):
    pass

class MyLinearOperator():
    '''
    Because scipy's linear operator requires passing in the shape
    of a linear operator, we had to create my own "dumb" linear operator class.
    This is used because a regularizer may be created with no composed linear operator,
    and to deal with this we create an identity linear operator which just passes
    through the input to output. But we don't necessarily know the dimensions at
    that point, because addData() may not yet have been called. 
    '''
    def __init__(self,matvec,rmatvec):
        self.matvec=matvec
        self.rmatvec=rmatvec
        
        
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




