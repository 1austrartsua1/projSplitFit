'''
projSplit module.

'''
from numpy import sum as npsum
from numpy.linalg import norm
from numpy import copy as npcopy
from numpy import zeros
from numpy import ones
from numpy import concatenate
from numpy import array
from numpy.random import choice

from scipy.sparse.linalg import aslinearoperator

from time import time 
from random import sample
from random import uniform

from regularizers import Regularizer 
from losses import Loss
import lossProcessors as lp
import projSplitUtils as ut
import userInputVal as ui



class ProjSplitFit(object):
    '''
    ProjSplitFit is the class used for creating a data-fitting problem and solving
    it with projective splitting. 
    
    Please refer to 

    * arxiv.org/abs/1803.07043 (algorithm definition page 9)
    * arxiv.org/abs/1902.09025 (algorithm definiteion pages 10-11)
    - To create an object, call
        psobj = ProjSplitFit(dualScaling)
        dualScaling (defaults to 1.0) is gamma in the above algorithm definitions. 
    
    The general optimization objective this can solve is
    
    min_(z,z_int){ (1.0/n)*sum_{i=1}^n loss(z_int + a_i^T (H z),y_i)
                        + sum_{i = 1}^{numReg} h_i(G_i z) }
    
    
    where
        - a_1...a_n are feature vectors forming the rows of a data matrix A
        - y_1...y_n are the response values
        - loss is the loss function dealt with via the Loss class defined in losses.py 
         (see the addData method)
        - H,G_1...G_{numReg} are linear operators (possibly the identity)
        - h_i are generic functions dealt with via the Regularizer class defined 
            in Regularizer.py (see also addRegularizer)
        - z_int is the intercept variable to be fit
        - z is a vector of parameters to be fit
            
    The data (A,y), loss, and linear operator H are added via the addData method.
    
    Each regularizer must be added individually via a call to the addRegularizer
    method, along with the linear operators G_i. 
        
    '''
    def __init__(self,dualScaling=1.0):
        '''        
        parameters
        ----------
        dualScaling : float, optional
            the primal-dual scaling parameter which is gamma in 
            [1] arxiv.org/abs/1803.07043 (algorithm definition page 9)
            [2] arxiv.org/abs/1902.09025 (algorithm definiteion pages 10-11).
            dualScaling must be > 0 and defaults to 1.0.
        '''        
        self.setDualScaling(dualScaling)                
        
        self.allRegularizers = []
        self.numRegs = 0
        self.embeddedRegInUse = False
        self.dataAdded = False
        self.runCalled = False
        
        
    
    def setDualScaling(self,dualScaling):
        '''
        Changes the dual scaling parameter (gamma)
        
        
        parameters
        ---------
        dualScaling : float, optional
            the primal-dual scaling parameter which is gamma in 
            [1] arxiv.org/abs/1803.07043 (algorithm definition page 9)
            [2] arxiv.org/abs/1902.09025 (algorithm definiteion pages 10-11).
            dualScaling must be > 0 and defaults to 1.0.
        
        '''
        self.gamma = ui.checkUserInput(dualScaling,float,'float','dualScaling',
                                       default=1.0,low=0.0)
       
    
    def getDualScaling(self):
        '''
        Returns the current setting of dualScaling
        
        Returns
        -------
        float
            the dualScaling parameter
        '''
        return self.gamma 
                            
    def addData(self,observations,responses,loss,process=lp.Forward2Backtrack(),
                intercept=True,normalize=True,linearOp = None): 
        '''        
        Adds data for the data fitting model.
        
        The optimization problem is 
        (1) min_(z,z_int){ (1.0/n)*sum_{i=1}^n loss(z_int + a_i^T (H z),y_i)
                        + sum_{i = 1}^{numReg} h_i(G_i z) }
                
        Parameters
        ----------
            observations : ndnumpy array/matrix 
                each row being a_i from Eq. (1)
                
            responses : 1d array vector/list/numpy array 
                each element equal to y_i from Eq. (1)
                
            loss : int>=1 or string or losses.LossPlugIn
                May be an int>=1, 'logistic' or an object of class LossPlugIn
                defined in losses.py
                
            process : lossProcessors.ProjSplitLossProcessor, optional
                An object of a class derived from ProjSplitLossProcessor. 
                Default is Forward2Backtrack()      
                
            intercept : bool,optional 
                whether to include an intercept/constant term in the linear model. 
                Default is True.
                
            normalize : bool,optional
                whether to normalize columns of the data matrix to have unit norm.
                If true, data matrix will be copied. Default is True. 
                
            linearOp : scipy.sparse.linalg.LinearOperator,optional
                adds matrix H in Eq. (1). Defaults to the identity. 
        '''
        
        try:
            (self.nrowsOfA,self.ncolsOfA) = observations.shape            
        except:
            print("Error: observations and responses should be 2D arrays, i.e. ")
            print("NumPy arrays. They must have a shape attribute. Aborting, did not add data")            
            raise Exception("Observations and responses should be 2D numpy-like arrays")

        try:
            if (self.nrowsOfA!=len(responses)):
                raise Exception("Error: len(responses) != num observations. Aborting. Data not added")
            self.yresponse = array(responses)
            
            if len(self.yresponse.shape) > 2:
                raise Exception("responses must be a list or a 1D array")
            elif (len(self.yresponse.shape)==2) and (self.yresponse.shape[1] != 1):
                raise Exception("responses must be a list or a 1D array")
                
        except:
            raise Exception("responses must be a list or a 1D array")
        
        if (self.nrowsOfA == 0) | (self.ncolsOfA == 0):
            self.A = None
            self.yresponse = None
            raise Exception("Error. A dimension of the observation matrix is 0. Must be 2D.") 
            
        
        if isinstance(process,lp.ProjSplitLossProcessor) == False:
            raise Exception("process must be an object of a class derived from ProjSplitLossProcessor")
        else:
            self.process = process
        
        if self.process.pMustBe2 and (loss != 2):
            print("Warning: this process object only works for the squared loss")
            print("Using Forward2Backtrack() as the process object")
            self.process = lp.Forward2Backtrack()
            
        
        if linearOp is None:
            self.dataLinOp = ut.MyLinearOperator(matvec=lambda x:x,rmatvec=lambda x:x)
            self.nPrimalVars = self.ncolsOfA
            self.linOpUsedWithLoss = False
        
        else:
            try:
        
                if linearOp.shape[0] != self.ncolsOfA:
                    print("Error! number of columns of the data matrix is {}".format(self.ncolsOfA))
                    print("while number of rows of the composed linear operator is {}".format(linearOp.shape[0]))
                    print("These must be equal! Aborting addData call")
                    self.A = None
                    self.yresponse = None
                    self.nPrimalVars = None
                    raise Exception("Error! number of columns of the data matrix must equal number rows of composed linear operator")
                else:
                    # expandOperator to deal with the intercept term
                    # the first entry of the input is the intercept which is 
                    # just passed through
                    linearOp = aslinearoperator(linearOp)
                    matvec,rmatvec = ut.expandOperator(linearOp)
                    self.dataLinOp = ut.MyLinearOperator(matvec,rmatvec)
                    self.nPrimalVars = linearOp.shape[1]
                    self.linOpUsedWithLoss = True 
            except:
                print("Error: linearOp must be a linear operator and must have ")
                print("a shape member and support matvec and rmatvec methods")
                print("Aborting add data")
                self.A = None
                self.yresponse = None
                self.nPrimalVars = None
                raise Exception("Invalid linear op")
                      
                    
        # check that all of the regularizers added so far have linear ops 
        # which are consistent with the added data
        for reg in self.allRegularizers:
            if reg.linearOpUsed:
                if reg.linearOp.shape[1] != self.nPrimalVars:
                    print("ERROR: linear operator added with a regularizer")
                    print("has number of columns which is inconsistent with the added data")
                    print("Added data has {} columns".format(self.nPrimalVars))
                    print("A linear operator has {} columns".format(reg.linearOp.shape[1]))
                    print("These must be equal, aborting add data")
                    self.A = None
                    self.yresponse = None
                    self.nPrimalVars = None
                    raise Exception("Col number mismatch in linear operator")
        
        if normalize:
            print("Normalizing columns of A to unit norm")
            self.normalize = True
            self.A = npcopy(observations)            
            self.scaling = norm(self.A,axis=0)
            self.scaling += 1.0*(self.scaling < 1e-10)
            self.A = self.A/self.scaling
        else:
            print("Not normalizing columns of A")            
            self.A = observations
            self.normalize = False 
        
        self.loss = Loss(loss)
           
        
        
        if self.embeddedRegInUse & (self.process.embedOK == False):
            print("WARNING: A regularizer was added with embedded = True")
            print("But embedding is not possible with this process object")
            print("Moving embedded regularizer to be an ordinary regularizer")
            regObj = self.embedded 
            self.allRegularizers.append(regObj)
            self.embedded = None
            self.embeddedRegInUse = False
            self.numRegs += 1
             
        if (intercept not in [False,True]):
            print("Warning: intercept should be a bool")
            print("Setting to False, no intercept")
            intercept = 0
        else:
            intercept = int(intercept)
        
        col2Add = intercept*ones((self.nrowsOfA,1))
        self.A = concatenate((col2Add,self.A),axis=1)
                       
        self.intercept = intercept

        # completed a successful call to addData()                
        self.dataAdded = True 
        # since data have been added must reset the variables z^k, x_i^k etc.
        self.internalResetIterate = True 
         
    
    def numPrimalVars(self):
        '''
        Retrieve the number of primal variables.
        
        After the addData method has been called, one may call this method. 
        
        If the addData method has not been called yet, getParams raises an exception.
        
        Returns
        ------        
            nPrimalVars: int
                Number of primal variables not including the intercept            
        
        '''
        if self.dataAdded == False:
            raise Exception("Cannot get params until data is added")
        else:        
            return self.nPrimalVars + int(self.intercept)
        
    def numObservations(self):
        '''
        Retrieve the number of observations. 
        
        After the addData method has been called, one may call this method. 
        
        If the addData method has not been called yet, this method raises an exception.
        
        Returns
        ------                                    
            nrowsOfA: int
                Number of observations

        
        '''
        if self.dataAdded == False:
            raise Exception("Cannot get params until data is added")
        else:        
            return self.nrowsOfA
    
    def addRegularizer(self,regObj, linearOp=None, embed = False):
        '''        
        adds a regularizer to the optimization problem.
        
        Recall the optimization problem
        
        (1) min_(z,z_int){ (1.0/n)*sum_{i=1}^n loss(z_int + a_i^T (H z),y_i)
                        + sum_{i = 1}^{numReg} h_i(G_i z) }
        
        Adds each h_i and G_i in Eq. (1)
        
        Parameters
        ----------
            regObj : regularizers.Regularizer 
                an object of class Regularizer (defined in regularizer.py)
                
            linearOp : scipy.sparse.linalg.LinearOperator,optional
                adds matrix G_i in Eq. (1)
                
            embed : bool,optional
                internal option in projective splitting (see Eq. (8)-(9) in
                [1]. For forward-type loss process updates, perform the "prox" of 
                this regularizer in a forward-backward style update. Defaults to 
                False
            
        '''
        if isinstance(regObj,Regularizer) == False:
            raise Exception("regObj must be an object of class Regularizer")
        
        if (linearOp is not None) & self.dataAdded:
            try:
                #check the dimensions make sense            
                if linearOp.shape[1] != self.nPrimalVars:
                    print("ERROR: linear operator added with this regularizer")
                    print("has number of columns which is inconsistent with the added data")
                    print("Added data has {} columns".format(self.nPrimalVars))
                    print("This linear operator has {} columns".format(linearOp.shape[1]))
                    print("These must be equal, aborting addRegularizer")
                    raise Exception("Invalid col number in added linear op")
            except:
                raise Exception("Invalid linearOp does not support shape")
            
        if embed:
            OK2Embed = True
            if self.dataAdded: 
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
                if self.embeddedRegInUse:
                    print("Warning: Regularizer embed option set to True")
                    print("But an earlier regularizer is already embedded")
                    print("Can only embed one regularizer")
                    print("Moving old regularizer to be non-embedded, the new regularizer will be embedded")
                    self.allRegularizers.append(self.embedded)
                    self.numRegs += 1
                        
                self.embedded = regObj 
                self.embeddedRegInUse = True                 
                
        else:            
            self.allRegularizers.append(regObj)
            self.numRegs += 1
        
        
        self.__addLinear(regObj,linearOp)
        
        
        self.internalResetIterate = True # Ensures we reset the variables if we add another regularizer 
    
    def getObjective(self):
        '''
        Returns the current objective value 
        
        Evaluated at the current primal iterate z^k. 
        If the method has not been run yet, raises an exception.
        
        If a loss or regularizer was added without defining a value method,
        calling getObjective raises an Exception. 
        
        Returns
        -------
            currentLoss : float
                the current objective value evaluated at the current iterate
        '''        
        if self.runCalled == False:
            raise Exception("Method not run yet, no objective to return. Call run() first.")
                
        currentLoss,Hz = self.__getLoss(self.z)
        
        for reg in self.allRegularizers:
            Hiz = reg.linearOp.matvec(self.z[1:]) 
            getVal = reg.evaluate(Hiz)
            if getVal is None:
                raise Exception("Regularizer added without defining its value method")
            else:                    
                currentLoss += getVal
                
        if self.embeddedRegInUse:
            reg = self.embedded      
            getVal = reg.evaluate(Hz)
            if getVal is None:
                raise Exception("Regularizer added without defining its value method")
            else:
                currentLoss += self.embeddedScaling*getVal/reg.getScaling()
        
        return currentLoss
        
    
    def getSolution(self):
        '''
        Returns the current solution vector. 
        
        Returns (Hz^k, z^k) where z^k is the current primal Solution 
        and Hz^k where H is the linear operator added with the data. 
        
        If the normalize option is set to True in addData, the scaling which 
        was applied to each feature is applied to the entries of Hz^k, so the 
        same results can be obtained with the original non-normalized data matrix. 
        Note that the scaling is not applied to z^k, the second output. 
        
        If the run() method has not been called yet, raises an exception. 
        
        If intercept was set to True in addData, the intercept coefficient is the 
        first entry of both z^k and Hz^k. 
        
        Returns
        -------
            out : 1D numpy array 
                Hz^k
                
            z : 1d numy array
                z^k
        '''
        
        if self.runCalled == False:
            raise Exception("Method not run yet, no solution to return. Call run() first.")
        
        
        Hz = self.dataLinOp.matvec(self.z)
        out = zeros(self.ncolsOfA+1)
        
        if self.normalize:
            out[1:] = Hz[1:]/self.scaling
        else:
            out[1:] = Hz[1:] 

        if self.intercept:
            out[0] = self.z[0]
            return out,self.z 
        else:
            return out[1:],self.z 
        
        
    def getPrimalViolation(self):        
        '''
        Returns the current primal violation. 
        
        After at least one call to the method run(), returns a float 
        equal to the primal violation. 
        If run has not been called yet, raises an exception.
        
        Returns
        -------
            primalErr : float
                Primal Violation. 
        '''
        if self.runCalled == False:
            raise Exception("Method not run yet, no primal violation to return. Call run() first.")
        else:
            return self.primalErr
    
    def getDualViolation(self):
        '''
        Returns the current dual violation. 
        
        After at least one call to the method run(), returns a float 
        equal to the dual violation. 
        If run has not been called yet, raises an exception.
        
        Returns
        -------
            dualErr : float
                Dual Violation. 
        '''
        if self.runCalled == False:
            raise Exception("Method not run yet, no dual violation to return. Call run() first.")
        else:
            return self.dualErr
    
    def getHistory(self):
        '''
        Returns array of history data on most recent run().
        
        After at least one call to run with keepHistory set to True, the function call
        historyArray = psfObj.getHistory()
        returns a two-dimensional five-row NumPy array with each row 
        corresponding to an iteration for which the history statistics were 
        recorded. The total number of columns is num iterations divided by the 
        historyFreq parameter, which can be set in run() and defaults to 10. 
        In each row of this array, the rows have the following interpretation:
        Row Number Interpretation
        0             Objective value
        1             Cumulative run time
        2             Primal violation
        3             Dual violation
        4             Value of Phi(p^k) used in hyperplane construction 
        
        If run() has not yet been called with keepHistory set to True, 
        this function will raise an Exception when called.
        
        If keepHistory is set to True and a regularizer or the loss is added without 
        implementing its value method, an Exception will be raised. 
        
        Returns
        -------
            historyArray : ndarray 
                ndarray with 5 rows. 
        '''
        if self.runCalled == False:
            raise Exception("Method not run yet, no history to return. Call run() first.")
        if self.historyArray is None:
            print("run() was called without the keepHistory option, no history")
            print("call run with the keepHistory argument set to True")
            raise Exception("No history to return as keepHistory option was False in call to run()")
        return self.historyArray
    
    def run(self,primalTol = 1e-6, dualTol=1e-6,maxIterations=None,keepHistory = False, 
            historyFreq = 10, nblocks = 1, blockActivation="greedy", blocksPerIteration=1, 
            resetIterate=False,verbose=False):
        '''
        Run projective splitting.
                
        Parameters
        ----------
            primalTol : float>=0.0,optional 
                Continue running algorithm if primal error is greater than primalTol. 
                Note that to terminate the method, both primal error AND dual error 
                must be smaller than their respective tolerances. Default 1e-6.
                
            dualTol : float>=0.0,optional 
                Continue running algorithm if dual error is greater than dualTol. 
                Note that to terminate the method, both primal error AND dual error 
                must be smaller than their respective tolerances. Default 1e-6.
                
            maxIterations : int>0,optional 
                Terminate algorithm if ran for more than maxIterations iterations.
                Default is None meaning never terminate. 
                
            keepHistory : bool,optional 
                If True, record the history (see getHistory() method). Default False.
                
            historyFreq : int>0,optional 
                Frequency to keep history, defaults to every 10 iterations. 
                Note that to keepHistory requires computing the objective
                which may be slow for large problems. 
                
            nBlocks : int>0,optional 
                number of blocks in the projective splitting decomposition 
                of the loss. Defaults to 1.
                
            blockActivation : string,optional 
                Strategy for selecting blocks of the loss to process at each iteration. 
                Defaults to "greedy". Other valid choices are "random" and "cyclic". 
                
            blocksPerIteration : int>0,optional 
                Number of blocks to update in each iteration. Defaults to 1.
                
            resetIterate : bool,optional 
                If True, the current values of all variables in projective splitting 
                (eg: z^k, w_i^k etc) are erased and initialized to 0. Defaults to False. 
                
            verbose : bool,optional
                Verbose as in printing iteration counts etc. Defaults to False.
                
        '''
        
        if self.dataAdded == False:
            raise Exception("Must add data before calling run(). Aborting...")        
        
        if (blockActivation != "greedy") and (blockActivation != "cyclic") \
            and (blockActivation != "random"):
                print("Warning: chosen blockActivation is not recognised")
                print("Using greedy instead")
                blockActivation = "greedy"

                                
        numBlocks = self.__setBlocks(nblocks)
                                
        if self.runCalled:
            if(self.nDataBlocks != numBlocks):
                print("change of the number of blocks, resetting iterates automatically")
                self.internalResetIterate = True
        
        self.nDataBlocks = numBlocks
                                                                                
        if blocksPerIteration >= self.nDataBlocks:
            blocksPerIteration = self.nDataBlocks
            
        self.partition = ut.createApartition(self.nrowsOfA,self.nDataBlocks)
                    
        self.__setUpRegularizers()
                                            
        self.nDataBlockVars = self.ncolsOfA + 1 # extra 1 for the intercept term
        
        if resetIterate or self.internalResetIterate:           
            self.internalResetIterate = False
            self.__initializeVariables()
                                                                        
        if maxIterations is None:
            maxIterations = float('Inf')
                
        self.k = 0
        objective = []
        times = [0]
        primalErrs = []
        dualErrs = []
        phis = []
        self.runCalled = True 
        
        ################################
        # BEGIN MAIN ALGORITHM LOOP
        ################################
        while(self.k < maxIterations):  
            if verbose and (self.k%historyFreq == 0):
                print('iteration = {}'.format(self.k))
            t0 = time()            
            self.__updateLossBlocks(blockActivation,blocksPerIteration)        
            self.__updateRegularizerBlocks()            
            
            if (self.primalErr < primalTol) & (self.dualErr < dualTol):
                print("primal and dual tolerance reached, finishing run")                
                break            
                        
            phi = self.__projectToHyperplane() # update (z,w1...wn) from (x1..xn,y1..yn,z,w1..wn)
            
            if phi == "converged":
                print("Gradient of the hyperplane is 0, converged, finishing run")                
                break
            t1 = time()

            if keepHistory and (self.k % historyFreq == 0):
                objective.append(self.getObjective())
                times.append(times[-1]+t1-t0)
                primalErrs.append(self.primalErr)
                dualErrs.append(self.dualErr)
                phis.append(phi)
                
            
            self.k += 1
            
        
        if keepHistory:
            self.historyArray = [objective]
            self.historyArray.append(times[1:])
            self.historyArray.append(primalErrs)
            self.historyArray.append(dualErrs)
            self.historyArray.append(phis)
            self.historyArray = array(self.historyArray) 
        
    
        
        if self.embeddedRegInUse:
            # we modified the embedded scaling to deal with multiple num blocks
            # now set it back to the previous value
            self.embedded.setScaling(self.embeddedScaling)
        
    
    @staticmethod
    def __addLinear(regObj,linearOp=None):        
        if linearOp is None:
            regObj.linearOp = ut.MyLinearOperator(matvec=lambda x:x,rmatvec=lambda x:x)
            regObj.linearOpUsed = False
        else:
            try:
                regObj.linearOp = aslinearoperator(linearOp)        
                regObj.linearOpUsed = True        
            except:
                raise Exception("linearOp invalid. Use scipy.sparse.linalg.aslinearoperator or similar")
                
    def __initializeVariables(self):
        self.z = zeros(self.nPrimalVars+1)            
        self.Hz = zeros(self.nDataBlockVars)            
        self.xdata = zeros((self.nDataBlocks,self.nDataBlockVars))            
        self.ydata = zeros((self.nDataBlocks,self.nDataBlockVars))
        self.wdata = zeros((self.nDataBlocks,self.nDataBlockVars))
        
        # initialize the loss processor auxiliary data structures 
        # if it has any 
        self.process.initialize(self) 

        if self.numRegs > 0:                        
            self.udata = zeros((self.nDataBlocks,self.nDataBlockVars))
        else:
            self.udata = zeros((self.nDataBlocks - 1,self.nDataBlockVars))
        
        self.xreg = []
        self.yreg = []
        self.wreg = []
        self.ureg = []
        
        i = 0
        for reg in self.allRegularizers:
            if i == self.numRegs - 1:
                    nRegularizerVars = self.nPrimalVars + 1 # extra 1 for intercept ONLY for last block
            elif reg.linearOpUsed:
                    nRegularizerVars = reg.linearOp.shape[0]                                                            
            else:
                nRegularizerVars = self.nPrimalVars 
                
            self.xreg.append(zeros(nRegularizerVars))    
            self.yreg.append(zeros(nRegularizerVars))    
            self.wreg.append(zeros(nRegularizerVars))    
            i += 1
            if i != self.numRegs:
                self.ureg.append(zeros(nRegularizerVars))    
            
    def __setBlocks(self,nblocks):
        try:        
            if nblocks >= 1:
                if nblocks > self.nrowsOfA:
                    print("more blocks than num rows. Setting nblocks equal to nrows")
                    numBlocks = self.nrowsOfA 
                else:
                    numBlocks = nblocks
            else:
                print("Error: nblocks must be greater than 1, setting nblocks to 1")
                numBlocks = 1
        except:
            print("Error: nblocks must be of type INT greater than 1, setting nblocks to 1")
            numBlocks = 1
        return numBlocks
    
    def __setUpRegularizers(self):
        
        if self.embeddedRegInUse == False:
            # if no embedded reg added, create an artificial embedded reg
            # with a "pass-through" prox
            self.embedded = Regularizer(lambda x,scale:x,lambda x:0)
        else:
            if self.embedded.getStepsize() != self.process.getStep():
                print("WARNING: embedded regularizer must use the same stepsize as the Loss update process")
                print("Setting the embedded regularizer stepsize to be the process stepsize")
                self.embedded.setStep(self.process.getStep())
            
            # the scaling used must be divided down by the number of blocks because
            # this term is distributed equally over all loss blocks
            self.embeddedScaling = self.embedded.getScaling()
            self.embedded.setScaling(self.embeddedScaling/self.nDataBlocks)
        
        if self.numRegs == 0:
            if self.linOpUsedWithLoss == False:
                self.numPSblocks = self.nDataBlocks
            else:
                # if there are no regularizers and the data term is composed 
                # with a linear operator, we must add a dummy regularizer
                # which has a pass-through prox and 0 value
                self.addRegularizer(Regularizer(lambda x,scale: x, lambda x: 0))                 
        
        if self.numRegs != 0:                
            # if all nonembedded regularizers have a linear op
            # then we add an additional dummy variable to projective splitting
            # corresponding to 0 objective function
            allRegsHaveLinOps = True 
            i = 0
            for reg in self.allRegularizers:
                if reg.linearOpUsed == False:                                        
                    allRegsHaveLinOps = False
                    lastReg = self.allRegularizers[-1]
                    if lastReg.linearOpUsed:      
                        #swap the two regularizers to ensure 
                        #the last block corresponds to no linear op
                        self.allRegularizers[i] = lastReg
                        self.allRegularizers[-1] = reg
                    break
                i += 1
                    
            if allRegsHaveLinOps:
                self.addRegularizer(Regularizer(lambda x,scale: x, lambda x: 0))                
                            
            self.numPSblocks = self.nDataBlocks + self.numRegs
            
            
    def __updateLossBlocks(self,blockActivation,blocksPerIteration):        
        
        self.Hz = self.dataLinOp.matvec(self.z)
        
        if blockActivation == "greedy":            
            phis = npsum((self.Hz - self.xdata)*(self.ydata - self.wdata),axis=1)            
            
            if phis.min() >= 0:
                activeBlocks = choice(range(self.nDataBlocks),blocksPerIteration,replace=False)
            else:
                activeBlocks = phis.argsort()[0:blocksPerIteration]
        elif blockActivation == "random":
            activeBlocks = choice(range(self.nDataBlocks),blocksPerIteration,replace=False)
        elif blockActivation == "cyclic":
            if self.k == 0:
                self.cyclicPoint = 0
            
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
            self.process.update(self,i)
        
        self.primalErr = norm(self.Hz - self.xdata,ord=2,axis=1).max()
        self.dualErr =   norm(self.ydata - self.wdata,ord=2,axis=1).max()
                
    def __updateRegularizerBlocks(self):
                
        for i in range(self.numRegs-1):             
            reg = self.allRegularizers[i]
            Giz = reg.linearOp.matvec(self.z[1:])                        
            t = Giz + reg.step*self.wreg[i]
            self.xreg[i] = reg.getProx(t)
            self.yreg[i] = reg.step**(-1)*(t - self.xreg[i])
            primal_err_i = norm(Giz - self.xreg[i],2)
            if self.primalErr<primal_err_i:
                self.primalErr = primal_err_i
            dual_err_i = norm(self.wreg[i] - self.yreg[i],2)
            if self.dualErr<dual_err_i:
                self.dualErr = dual_err_i
                
                        
        # update coefficients corresponding to the last block
        # including the intercept term
        if self.numRegs > 0:
            reg = self.allRegularizers[-1]
            t = self.z + reg.step*self.wreg[-1]
            self.xreg[-1][1:] = reg.getProx(t[1:])
            self.xreg[-1][0] = t[0]
            self.yreg[-1] = reg.step**(-1)*(t - self.xreg[-1])
                                    
            primal_err_i = norm(self.xreg[-1]-self.z,2)
            if self.primalErr<primal_err_i:
                self.primalErr = primal_err_i
                
            dual_err_i = norm(self.yreg[-1]-self.wreg[-1],2)
            if self.dualErr<dual_err_i:
                self.dualErr = dual_err_i
                    
                
                                    
    def __projectToHyperplane(self):
                            
        # compute u and v for data blocks
        if self.numRegs > 0:            
            self.udata = self.xdata - self.dataLinOp.matvec(self.xreg[-1])            
        else:
            # if there are no regularizers, the last block corresponds to the 
            # last data block. Further, dataLinOp must be the identity
            self.udata = self.xdata[:-1] - self.xdata[-1]
            
        vin = sum(self.ydata)
        v = self.dataLinOp.rmatvec(vin)        
        
        # compute u and v for regularizer blocks except the final regularizer 
        for i in range(self.numRegs - 1):
            Gxn = self.allRegularizers[i].linearOp.matvec(self.xreg[-1][1:])
            self.ureg[i] = self.xreg[i] - Gxn
            Gstary = self.allRegularizers[i].linearOp.rmatvec(self.yreg[i])
            v += concatenate((array([0.0]),Gstary))
                        
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
                
                if len(self.wdata) + len(self.wreg) > 1:
                    # if there is more than one w block, update w. Otherwise
                    # if there is just a single block, it will just stay at 0.
                    self.__updatew(tau)
                                
        else:            
            phi = "converged"
        
        return phi
    
    
    
    def __getPhi(self,v):
        phi = self.z.dot(v)            
            
        if len(self.wdata) + len(self.wreg) > 1:       
            if len(self.wreg) == 0:
                phi += npsum(self.udata*self.wdata[0:(self.numPSblocks-1)])
            else:
                phi += npsum(self.udata*self.wdata)
            
            for i in range(self.numRegs - 1):
                phi += self.ureg[i].dot(self.wreg[i])
        
        phi -= npsum(self.xdata*self.ydata)
        
        for i in range(self.numRegs):     
            phi -= self.xreg[i].dot(self.yreg[i])
            
        return phi
    
    def __getLoss(self,z):
        Hz = self.dataLinOp.matvec(z)
        AHz = self.A.dot(Hz)
        getVal = self.loss.value(AHz,self.yresponse)
        if getVal is None:
            print("ERROR: If you don't implement a losses value func, set getHistory to")
            print("False and do not compute objective values")            
            raise Exception("Losses value function is not implemented. Cannot compute objective values.")
        currentLoss = (1.0/self.nrowsOfA)*sum(self.loss.value(AHz,self.yresponse))     
        return currentLoss,Hz
    
    def __updatew(self,tau):        
            if len(self.wreg) == 0:               
                # if no regularizers, the linearOp corresponding to the 
                # data block must be the identity
                self.wdata[0:(self.nDataBlocks-1)] = self.wdata[0:(self.nDataBlocks-1)] - tau*self.udata
                self.wdata[-1] = -npsum(self.wdata[0:(self.nDataBlocks-1)],axis=0)
            else:
                self.wdata = self.wdata - tau*self.udata                        
                negsumw = -npsum(self.wdata,axis=0)
                GstarNegSumw = self.dataLinOp.rmatvec(negsumw)                        
                for i in range(self.numRegs - 1):
                    self.wreg[i] = self.wreg[i] - tau*self.ureg[i]
                    Gstarw = self.allRegularizers[i].linearOp.rmatvec(self.wreg[i])
                    GstarNegSumw -= concatenate((array([0.0]),Gstarw))
                
                self.wreg[-1] = GstarNegSumw
        

