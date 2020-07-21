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
    r'''
    ProjSplitFit is the class used for creating a data-fitting problem and solving
    it with projective splitting.

    Please refer to

    * arxiv.org/abs/1803.07043 (algorithm definition page 9)
    * arxiv.org/abs/1902.09025 (algorithm definiteion pages 10-11)

    To create an object, call::

        psobj = ProjSplitFit(dualScaling)

    dualScaling (defaults to 1.0) is gamma in the algorithm definitions from the above papers.

    The general optimization objective this can solve is

    .. math::

      \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}
                \frac{1}{n}\sum_{i=1}^n\ell (z_0 + a_i^\top H z,y_i)
                   + \sum_{j=1}^{n_r}h_j(G_j z)


    where

    * :math:`z_0\in\mathbb{R}` is the intercept variable
    * :math:`z\in\mathbb{R}^d` is the parameter vector
    * :math:`\ell:\mathbb{R}\times\mathbb{R}\to\mathbb{R}_+` is the loss
    * :math:`y_i` for :math:`i=1,\ldots,n` are the labels
    * :math:`H\in\mathbb{R}^{d' \times d}` is a matrix (typically the identity)
    * :math:`a_i\in\mathbb{R}^{d'}` are the observations, forming the rows of the :math:`n\times d'` observation/data matrix :math:`A`
    * :math:`h_j` for :math:`j=1,\ldots,n_r` are convex functions which are *regularizers*, typically nonsmooth
    * :math:`G_j` for :math:`j=1,\ldots,n_r` are matrices, typically the identity.

    The data :math:`A` and :math:`y` are added via the ``addData`` method.

    regularizers are added via the ``addRegularizer`` method.

    The algorithm is run via the ``run`` method.

    '''
    def __init__(self,dualScaling=1.0):
        '''
        parameters
        ----------
        dualScaling : :obj:`float`, optional
            the primal-dual scaling parameter which is gamma in
            arxiv.org/abs/1803.07043 (algorithm definition page 9) and
            arxiv.org/abs/1902.09025 (algorithm definiteion pages 10-11).
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
        dualScaling : :obj:`float`, optional
            the primal-dual scaling parameter which is gamma in
            arxiv.org/abs/1803.07043 (algorithm definition page 9).
            dualScaling must be > 0 and defaults to 1.0.

        '''
        self.gamma = ui.checkUserInput(dualScaling,float,'float','dualScaling',
                                       default=1.0,low=0.0)


    def getDualScaling(self):
        '''
        Returns the current setting of dualScaling

        Returns
        -------
        :obj:`float`
            the dualScaling parameter
        '''
        return self.gamma

    def addData(self,observations,responses,loss,process=lp.Forward2Backtrack(),
                intercept=True,normalize=True,linearOp = None):
        r'''
        Adds data for the data fitting model.

        Recall that the general optimization objective solved by this package is

        .. math::

            \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}
              \frac{1}{n}\sum_{i=1}^n \ell (z_0 + a_i^\top H z,y_i) 
                + \sum_{j=1}^{n_r}h_j(G_j z)

        Parameters
        ----------
        observations : 2d ndarray or matrix
            each row of the observations matrix being :math:`a_i` above

        responses : 1d ndarray or :obj:`list` or numpy array
            each element equal to :math:`y_i` above

        loss : :obj:`float` or :obj:`string` or :obj:`losses.LossPlugIn`
            May be a :obj:`float` greater than 1, the :obj:`string` 'logistic', or an object of class :obj:`losses.LossPlugIn`

        process : :obj:`lossProcessors.LossProcessor`, optional
            An object of a class derived from :obj:`lossProcessors.LossProcessor`.
            Default is :obj:`Forward2Backtrack`

        intercept : :obj:`bool`,optional
            whether to include an intercept/constant term in the linear model.
            Default is True.

        normalize : :obj:`bool`,optional
            whether to normalize columns of the data matrix to have unit norm.
            If True, data matrix will be copied. Default is True.

        linearOp : :obj:`scipy.sparse.linalg.LinearOperator` or similar, optional
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


        if isinstance(process,lp.LossProcessor) == False:
            raise Exception("process must be an object of a class derived from LossProcessor")
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
        Retrieve the number of primal variables (possibly including the intercept).

        After the ``addData`` method has been called, one may call this method.

        If the ``addData`` method has not been called yet, getParams raises an exception.

        Returns
        ------
            nPrimalVars: :obj:`int`
                Number of primal variables including the intercept if that option is taken

        '''
        if self.dataAdded == False:
            raise Exception("Cannot get params until data is added")
        else:
            return self.nPrimalVars + int(self.intercept)

    def numObservations(self):
        '''
        Retrieve the number of observations.

        After the ``addData`` method has been called, one may call this method.

        If the ``addData`` method has not been called yet, this method raises an exception.

        Returns
        ------
            nrowsOfA: :obj:`int`
                Number of observations


        '''
        if self.dataAdded == False:
            raise Exception("Cannot get params until data is added")
        else:
            return self.nrowsOfA

    def addRegularizer(self,regObj, linearOp=None, embed = False):
        r'''
        adds a regularizer to the optimization problem.

        Recall the optimization problem

        .. math::

            \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}\frac{1}{n}\sum_{i=1}^n \ell (z_0 + a_i^\top H z,y_i) + \sum_{j=1}^{n_r}h_j(G_j z)

        This method adds each :math:`h_j` and :math:`G_j` above

        Parameters
        ----------
            regObj : :obj:`regularizers.Regularizer`
                object of class :obj:`regularizers.Regularizer`
                
            linearOp : :obj:`scipy.sparse.linalg.LinearOperator` or similar,optional
                adds matrix :math:`G_j` in above

            embed : :obj:`bool`,optional
                internal option in projective splitting. For forward-type loss process updates,
                perform the "prox" of this regularizer in a forward-backward style update.
                Defaults to False

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
        Returns the current objective value evaluated at the current primal iterate :math:`z^k`.
        If the method has not been run yet, raises an exception.

        If a loss or regularizer was added without defining a value method,
        calling ``getObjective`` raises an Exception.

        Returns
        -------
        currentLoss : :obj:`float`
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

    
    def getScaling(self):
        r'''
        Returns the scaling vector. For the :math:`n\times d'` data matrix
        :math:`A`, the scaling vector is :math:`d'\times 1` vector containing the 
        scaling factors used for each feature. This scaling vector can be used with 
        new test data to normalize the features. 
        If the ``normalize`` argument to ``addData`` was set to False,
        then an exception will be raised.
        
        If no data have been added yet, raises an exception. 
        
        Returns         
        --------
          scaling : 1D NumPy array
            scaling vector or None if ``normalize`` set to False in 
        '''
        if self.dataAdded==False:
            raise Exception("No data added yet so cannot return scale vector")
        
        if self.normalize == False:
            raise Exception("No normalization applied so cannot return a scale vector")
        
        return self.scaling 
        

    def getSolution(self,descale=False):
        r'''
        Returns the current primal solution vector. Recall the objective function
        
        .. math::

            \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}
                \frac{1}{n}\sum_{i=1}^n\ell (z_0 + a_i^\top H z,y_i)
                   + \sum_{j=1}^{n_r}h_j(G_j z)
        
        Returns the current primal solution :math:`z^k`.
        
        If the ``intercept`` argument was True in ``addData``, the intercept coefficient is the
        first entry of :math:`z^k`.
        
        If the ``run`` method has not been called yet, raises an exception.

        Parameters
        ----------
        
            descale : :obj:`bool`,optional
                    Defaults to False. 
                    If the ``normalize`` argument to ``addData`` was set to True
                    and the ``descale`` argument here is True, the normalization
                    that was applied to the columns of the data matrix is applied 
                    to the entries of :math:`z^k`, meaning that the user may 
                    use the original unnormalized data matrix with this new feature,
                    and also may use it on new data. However, if a linear operator
                    was added with ``addData`` via argument ``linOp``, then a warning
                    message will be printed and the solution vector will not be descaled. 
            
        Returns
        -------
            z : 1D numpy array
                :math:`z^k` 

        '''

        if self.runCalled == False:
            raise Exception("Method not run yet, no solution to return. Call run() first.")
        
            
        if descale:
            if self.normalize:
                if self.linOpUsedWithLoss:
                    print("Warning: Cannot descale because of the presence of a linear operator")
                    print("composed with the data. Just returning the unnormalized solution vector")
                    out = self.z 
                else:
                    out = self.z[1:]/self.scaling[1:]
                    out = concatenate((array(self.z[0]),out))
            else:
                out  = self.z 
        else:
            out  = self.z 
            
        if (self.intercept==False):
            out = out[1:]
            
        
        return out
        


    def getPrimalViolation(self):
        '''
        Returns the current primal violation.

        After at least one call to the method ``run``, returns a :obj:`float`
        equal to the primal violation.
        
        The primal violation is 
        
        .. math::
            \max_i \|G_i z^k - x_i^k\|_2
        
        where, with some abuse of notation, :math:`G_i` is the linear operator
        associated with the ith block. 

        If ``run`` has not been called yet, raises an exception.

        Returns
        -------
            primalErr : :obj:`float`
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
        
        The dual violation is
        
        .. math::
            \max_i \|y_i^k - w_i^k\|_2
        
        If run has not been called yet, raises an exception.

        Returns
        -------
            dualErr : :obj:`float`
                Dual Violation.
        '''
        if self.runCalled == False:
            raise Exception("Method not run yet, no dual violation to return. Call run() first.")
        else:
            return self.dualErr

    def getHistory(self):
        '''
        Returns array of history data on most recent run().

        After at least one call to run with keepHistory set to True, the function call::

            historyArray = psfObj.getHistory()

        returns a two-dimensional five-row NumPy array with each column
        corresponding to an iteration for which the history statistics were
        recorded. The total number of columns is num iterations divided by the
        ``historyFreq`` parameter, which can be set as an argument to ``run`` and defaults to 10.
        In each row of this array, the rows have the following interpretation:        

        0. Objective value
        1. Cumulative run time
        2. Primal violation
        3. Dual violation
        4. Value of :math:`\phi(p^k)` used in hyperplane construction

        If ``run`` has not yet been called with ``keepHistory`` set to True,
        this function will raise an Exception when called.

        If ``keepHistory`` is set to True and a regularizer or the loss is added without
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
        r'''
        Run projective splitting.

        Parameters
        ----------
            primalTol : :obj:`float`,optional
                Continue running algorithm if primal error is greater than ``primalTol``.
                The primal error is

                .. math::
                    \max_i \|G_i z^k - x_i^k\|_2

                where, with some abuse of notation, :math:`G_i` is the linear operator
                associated with the ith block. Note that to terminate the method, both primal error AND dual error
                must be smaller than their respective tolerances. Or the number of iterations
                exceeds the maximum number. Default 1e-6.

            dualTol : :obj:`float`,optional
                Continue running algorithm if dual error is greater than dualTol.
                The dual error is

                .. math::
                    \max_i \|y_i^k - w_i^k\|_2

                Note that to terminate the method, both primal error AND dual error
                must be smaller than their respective tolerances. Or the number of iterations
                exceeds the maximum number. Default 1e-6.

            maxIterations : :obj:`int`,optional
                Terminate algorithm if ran for more than ``maxIterations`` iterations.
                Default is None meaning do not terminate until primalTol and dualTol are reached.

            keepHistory : :obj:`bool`,optional
                If True, record the history (see ``getHistory`` method). Default False.

            historyFreq : :obj:`int`,optional
                Frequency to keep history, defaults to every 10 iterations.
                Note that to keep history requires computing the objective
                which may be slow for large problems.

            nBlocks : :obj:`int`,optional
                Number of blocks in the projective splitting decomposition
                of the loss. Defaults to 1. Blocks are contiguous indices and the 
                number of indices in each block varies by at-most one. 
                
                For example if number of observations is 100 and nblocks is set to 10
                then the blocks would be 
                
                    [ 
                    [0,1,...,9],
                    [10,11,...,19],
                    ...
                    [90,91,...,99]
                    ]
                
                If the number of observations was 105 and nblocks is set to 10, then 
                the blocks would be 5 blocks of 11 and 5 blocks of 10, i.e.
                
                    [
                    [0,1,...,10],
                    [11,12,..22],
                    ...
                    [44,45,...,54],
                    [55,56,...,64],
                    ...
                    [95,96,...,104]
                    ]
                
                This uses the formula
                
                .. math::
                    n = \lceil n/n_b \rceil n\%n_b
                         + \lfloor n/n_b \rfloor(n_b - n\%n_b).
                
            blockActivation : :obj:`string`,optional
                Strategy for selecting blocks of the loss to process at each iteration.
                Defaults to "greedy". Other valid choices are "random" and "cyclic".

            blocksPerIteration : :obj:`int`,optional
                Number of blocks to update in each iteration. Defaults to 1.

            resetIterate : :obj:`bool`,optional
                If True, the current values of all variables (if ``run`` has been called before)
                in projective splitting (eg: :math:`z^k, w_i^k` etc) are erased and initialized to 0. Defaults to False.

            verbose : :obj:`bool`,optional
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
        
        blocksPerIteration = ui.checkUserInput(blocksPerIteration,int,'int','blocksPerIteration',default=1,low=1,lowAllowed=True)

        try:            
            if blocksPerIteration >= self.nDataBlocks:
                blocksPerIteration = self.nDataBlocks
        except:
            print("Warning: blocksPerIteration should be a positive int")
            print("Setting blocksPerIteartion to 1")
            blocksPerIteration =1

        self.partition = ut.createApartition(self.nrowsOfA,self.nDataBlocks)

        self.__setUpRegularizers()

        self.nDataBlockVars = self.ncolsOfA + 1 # extra 1 for the intercept term
        
        
        resetIterate = ui.checkUserBool(resetIterate,"resetIterate")
                    
        if resetIterate or self.internalResetIterate:
            self.internalResetIterate = False
            self.__initializeVariables()
            
        keepHistory = ui.checkUserBool(keepHistory,"keepHistory")
        verbose = ui.checkUserBool(verbose,"verbose")                
            
        if maxIterations != None:            
            maxIterations = ui.checkUserInput(maxIterations,int,'int','maxIterations',default=1000,low=1,lowAllowed=True)
             
        if maxIterations is None:
            maxIterations = float('Inf')
            
        historyFreq = ui.checkUserInput(historyFreq,int,'int','historyFreq',default=10,low=1,lowAllowed=True)
        primalTol = ui.checkUserInput(primalTol,float,'float','primalTol',default=1e-6,low=0.0,lowAllowed=True)
        dualTol = ui.checkUserInput(dualTol,float,'float','dualTol',default=1e-6,low=0.0,lowAllowed=True)

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
        else:
            self.historyArray = None



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
            print("Error: nblocks must be of type int greater than 1, setting nblocks to 1")
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
