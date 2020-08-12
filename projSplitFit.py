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
from numpy import ndarray
from numpy import sqrt

from scipy.sparse.linalg import aslinearoperator
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse import hstack

from time import time


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

    * :cite:`for1`, arxiv.org/abs/1803.07043 (algorithm definition page 9)
    * :cite:`coco`, arxiv.org/abs/1902.09025 (algorithm definition pages 10-11)

    To create an object, call::

        psobj = ProjSplitFit(dualScaling)

    ``dualScaling`` (which defaults to 1.0) is :math:`\gamma` in the algorithm
    definitions from the above papers.

    The general optimization objective this can solve is

    .. math::

      \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}
                \frac{1}{n}\sum_{i=1}^n\ell (z_0 + a_i^\top H z,r_i)
                   + \sum_{j=1}^{n_r}\nu_j h_j(G_j z)


    where

    * :math:`z_0\in\mathbb{R}` is the intercept variable
    * :math:`z\in\mathbb{R}^d` is the parameter vector
    * :math:`\ell:\mathbb{R}\times\mathbb{R}\to\mathbb{R}_+` is the loss
    * :math:`r_i` for :math:`i=1,\ldots,n` are the responses (or labels)
    * :math:`H\in\mathbb{R}^{d' \times d}` is a matrix (typically the identity)
    * :math:`a_i\in\mathbb{R}^{d'}` are the observations, forming the rows of the :math:`n\times d'` observation/data matrix :math:`A`
    * :math:`h_j` for :math:`j=1,\ldots,n_r` are convex functions which are *regularizers*, typically nonsmooth
    * :math:`G_j` for :math:`j=1,\ldots,n_r` are matrices, typically the identity.
    * :math:`\nu_j` are positive scalar penalty parameters that multiply the regularizer functions.

    The data :math:`A` and :math:`y` are introduced via the ``addData`` method.

    Regularizers are introduced through the ``addRegularizer`` method.

    The ``run`` method solves the problem.

    '''
    def __init__(self,dualScaling=1.0):
        '''
        parameters
        ----------
        dualScaling : :obj:`float`, optional
            the primal-dual scaling parameter which is :math:`\gamma` in
            :cite:`for1` (algorithm definition on page 9) and
            :cite:`coco` (algorithm definition on pages 10-11).
            ``dualScaling`` must be positive, and defaults to 1.0.
        '''
        self.setDualScaling(dualScaling)

        self.allRegularizers = []
        self.numRegs = 0
        self.dataAdded = False
        self.runCalled = False



    def setDualScaling(self,dualScaling):
        '''
        Changes the dual scaling parameter (gamma)


        parameters
        ---------
        dualScaling : :obj:`float`, optional
            the primal-dual scaling parameter, which is gamma in
            :cite:`for1` (algorithm definition on page 9).
            Must be positive and defaults to 1.0.

        '''
        self.gamma = ui.checkUserInput(dualScaling,float,'float','dualScaling',
                                       default=1.0,low=0.0)


    def getDualScaling(self):
        '''
        Returns the current setting of ``dualScaling``

        Returns
        -------
        :obj:`float`
            the ``dualScaling`` parameter
        '''
        return self.gamma


    def addData(self,observations,responses,loss,process=lp.Forward2Backtrack(),
                intercept=True,normalize=True,linearOp = None,embed = None):
        r'''
        Introduces the data for the fitting model, and configures the loss function.

        Recall that the general optimization objective solved by this package is

        .. math::

            \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}
              \frac{1}{n}\sum_{i=1}^n \ell (z_0 + a_i^\top H z,r_i)
                + \sum_{j=1}^{n_r}\nu_j h_j(G_j z)

        Parameters
        ----------
        observations : 2d :obj:`numpy.ndarray` or :obj:`scipy.sparse.spmatrix`
            A 2D numpy array or scipy sparse matrix. The rows of this matrix
            are the vectors :math:`a_i` above. All
            :obj:`scipy.sparse.spmatrix` subclasses are supported. Internally,
            the matrix is converted to :obj:`scipy.sparse.csr_matrix` format,
            since this format is the most convenient for the row slicing and
            arithmetic operations required by the solution algorithm.

        responses : 1d :obj:`numpy.ndarray` or :obj:`list`
            the elements within this object comprise the response values
            :math:`r_i` above.  The number of elements should equal the number
            of rows in ``observations``.

        loss : :obj:`float` or :obj:`string` or :obj:`losses.LossPlugIn`
            Specifies the loss function :math:`\ell`.
            May be a :obj:`float` :math:`p > 1` to indicate the :math:`\ell_p^p`
            loss, the :obj:`string` 'logistic' to specify the logistic loss,
            function, or an object of class :obj:`losses.LossPlugIn`.

        process : :obj:`lossProcessors.LossProcessor`, optional
            An object of a class derived from :obj:`lossProcessors.LossProcessor`.
            Default is :obj:`Forward2Backtrack()`

        intercept : :obj:`bool`,optional
            whether to include an intercept/constant term in the linear model.
            The default value is ``True``.

        normalize : :obj:`bool`,optional
            whether to normalize columns of the data matrix to have square norm equal to num rows.
            If True, data matrix will be copied. Default is True.

        linearOp : :obj:`scipy.sparse.linalg.LinearOperator` or 2D :obj:`numpy.ndarray` or 2D :obj:`scipy.sparse.spmatrix`, optional
            Introduces the matrix :math:`H` in the above problem
            formulation. Defaults to the identity. If this argument is a
            sparse matrix, it will be converted to
            :obj:`scipy.sparse.csr_matrix` format, as this format is
            the most convenient for the arithmetic operations required in
            the solution algorithm.

        embed : :obj:`regularizers.Regularizer`,optional
            Embeds a regularizer into the loss, meaning that the proximal operator
            is evaluated in-line with the loss processing update. Only available for
            the following forward-type loss processors: ``Forward1Fixed``,
            ``Forward1Backtrack``, ``Forward2Fixed``, ``Forward2Backtrack``. If
            embed is used with any other loss processor, a warning is
            printed and the regularizer is added as an ordinary regularizer instead.

        '''

        try:
            (self.nrowsOfA,self.ncolsOfA) = observations.shape
        except:
            print("Error: observations and responses should be 2D arrays, i.e. ")
            print("NumPy arrays. They must have a shape attribute. Aborting, did not add data")
            raise Exception("Observations and responses should be 2D numpy-like arrays")

        if issparse(observations):
            #sparse matrix format
            observations = csr_matrix(observations)
            self.sparseObservationMtx = True
        elif isinstance(observations,ndarray) == False:
            raise Exception("Observations must be either a numpy ndarray or a scipy.sparse matrix")
        else:
            self.sparseObservationMtx = False

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
                    self.yresponse = None
                    self.nPrimalVars = None
                    raise Exception("Error! number of columns of the data matrix must equal number rows of composed linear operator")
                else:
                    # expandOperator to deal with the intercept term
                    # the first entry of the input is the intercept which is
                    # just passed through
                    matvec,rmatvec = ut.expandOperator(linearOp)
                    self.dataLinOp = ut.MyLinearOperator(matvec,rmatvec)
                    self.nPrimalVars = linearOp.shape[1]
                    self.linOpUsedWithLoss = True
            except:
                print("Error: linearOp must be a linear operator and must have ")
                print("a shape member and support matvec and rmatvec methods")
                print("Aborting add data")
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
                    self.yresponse = None
                    self.nPrimalVars = None
                    raise Exception("Col number mismatch in linear operator")

        if embed is None:
            self.embeddedRegInUse = False
        else:
            if isinstance(embed,Regularizer) == False:
                raise Exception("embed must be an object of class Regularizer")

            if(self.process.embedOK == False):
                print("WARNING: addData was called with a regularizer embedded.")
                print("But embedding is not possible with this process object.")
                print("Moving embedded regularizer to be an ordinary regularizer.")
                self.embeddedRegInUse = False
                self.addRegularizer(embed)
            else:
                self.embedded = embed
                self.embeddedRegInUse = True



        if normalize:
            print("Normalizing columns of A to have square norm equal to num rows")
            self.normalize = True
            if self.sparseObservationMtx == False:
                self.A = npcopy(observations)
                self.scaling = norm(self.A,axis=0)
                self.scaling += 1.0*(self.scaling < 1e-10)
                self.A = sqrt(self.nrowsOfA)*self.A/self.scaling
            else:
                self.A = csr_matrix(observations,copy=True)
                self.scaling = sparse_norm(self.A,axis=0)
                self.scaling += 1.0 * (self.scaling < 1e-10)
                self.scaling = 1.0 / self.scaling
                self.A = self.A.multiply(sqrt(self.nrowsOfA)*self.scaling)
                self.A = csr_matrix(self.A)
        else:
            #print("Not normalizing columns of A")
            self.A = observations
            self.normalize = False

        self.loss = Loss(loss)

        if (intercept not in [False,True]):
            print("Warning: intercept should be a bool")
            print("Setting to False, no intercept")
            intercept = 0
        else:
            intercept = int(intercept)

        col2Add = intercept * ones((self.nrowsOfA, 1))
        if self.sparseObservationMtx == False:
            self.A = concatenate((col2Add,self.A),axis=1)
        else:
            self.A = hstack((col2Add, self.A))
            self.A = csr_matrix(self.A)

        self.intercept = intercept

        # completed a successful call to addData()
        self.dataAdded = True
        # since data have been added must reset the variables z^k, x_i^k etc.
        self.internalResetIterate = True


    def numPrimalVars(self):
        '''
        Retrieve the number of primal variables (possibly including the intercept).

        Should only be invoked after calling the ``addData`` method; otherwise,
        calling this method raises an exception.

        Returns
        ------
            nPrimalVars: :obj:`int`
                Number of primal variables, including the intercept if present

        '''
        if self.dataAdded == False:
            raise Exception("Cannot get params until data is added")
        else:
            return self.nPrimalVars + int(self.intercept)


    def numObservations(self):
        '''
        Retrieve the number of observations.

        Should only be invoked after calling the ``addData`` method; otherwise,
        calling this method raises an exception.

        Returns
        ------
            nrowsOfA: :obj:`int`
                Number of observations


        '''
        if self.dataAdded == False:
            raise Exception("Cannot get params until data is added")
        else:
            return self.nrowsOfA


    def addRegularizer(self,regObj, linearOp=None):
        r'''
        Introduces a regularizer term into the optimization problem.

        Recall the optimization problem

        .. math::

            \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}\frac{1}{n}\sum_{i=1}^n
            \ell (z_0 + a_i^\top H z,r_i) + \sum_{j=1}^{n_r}\nu_j h_j(G_j z)

        This method adds each :math:`h_j`, :math:`\nu_j`, and :math:`G_j` above

        Parameters
        ----------
            regObj : :obj:`regularizers.Regularizer`
                object of class :obj:`regularizers.Regularizer`

            linearOp : :obj:`scipy.sparse.linalg.LinearOperator` or 2D :obj:`numpy.ndarray` or 2D :obj:`scipy.sparse.spmatrix`, optional
                Introduces the matrix :math:`G_j` above, which otherwise defaults to
                an identity matrix.  If a sparse matrix is supplied, it is
                internally converted to the :obj:`scipy.sparse.csr_matrix` format.

        '''
        if isinstance(regObj,Regularizer) == False:
            raise Exception("regObj must be an object of class Regularizer")

        if (linearOp is not None) & self.dataAdded:
            try:
                linopCols = linearOp.shape[1]
            except:
                raise Exception("Invalid linearOp does not support shape")

            #check the dimensions make sense
            if linopCols != self.nPrimalVars:
                print("ERROR: linear operator added with this regularizer")
                print("has number of columns which is inconsistent with the added data")
                print("Added data has {} columns".format(self.nPrimalVars))
                print("This linear operator has {} columns".format(linopCols))
                print("These must be equal, aborting addRegularizer")
                raise Exception("Invalid col number in added linear op")

        self.allRegularizers.append(regObj)
        self.numRegs += 1
        self.__addLinear(regObj,linearOp)

        self.internalResetIterate = True # Ensures we reset the variables if we add another regularizer


    def getObjective(self,ergodic=False):
        r'''
        Returns the current objective value evaluated at the current primal iterate
        :math:`z^k`.  If the method has not been run yet, raises an exception.

        If a loss or regularizer was added without defining its value method,
        calling ``getObjective`` raises an exception.

        Parameters
        -----------
        ergodic : :obj:`bool` or :obj:`string`, optional
           Whether to compute objective at the primal iterate :math:`z^k`,
           or one of its two averaged versions. If ``False`` (the default),
           uses the primal iterate. If "simple", evaluate at
           :math:`\frac{1}{k}\sum_{t=1}^k z^t`; if "weighted", evaluate at

           .. math::
              \frac{\sum_{t=1}^k\tau_t z^t}{\sum_{t=1}^k\tau_t}

           where the :math:`\tau_t` are the stepsizes used in the hyperplane
           projections.

        Returns
        ---------
        currentLoss : :obj:`float`
            the current objective value evaluated at the current iterate
        '''
        if self.runCalled == False:
            raise Exception("Method not run yet, no objective to return. Call run() first.")

        if ergodic == "simple":
            z2use = self.zbar
        elif ergodic == "weighted":
            z2use = self.zbarWeighted
        else:
            z2use = self.z

        currentLoss,Hz = self.__getLoss(z2use)



        for reg in self.allRegularizers:
            Hiz = reg.linearOp.matvec(z2use[1:])
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
        :math:`A`, the scaling vector is :math:`d'\times 1` vector containing
        the scaling factors used to normalize new test data. If the
        ``normalize`` argument to ``addData`` was ``False``, then
        the method simply returns a vector of ones.

        If no data have been added yet, raises an exception.

        Returns
        --------
          scaling : 1D NumPy array
            scaling vector
        '''
        if self.dataAdded==False:
            raise Exception("No data added yet so cannot return scale vector")

        if self.normalize == False:
            return ones(self.ncolsOfA)


        return self.scaling


    def getSolution(self,descale=False,ergodic=False):
        r'''
        Returns the current primal solution :math:`z^k`.

        If the ``intercept`` argument was True in ``addData``, the intercept coefficient
        is returned as the first entry of :math:`z^k`.

        If the ``run`` method has not been called yet, raises an exception.

        Parameters
        ----------

            descale : :obj:`bool`,optional
                    Defaults to False.
                    If the ``normalize`` argument to ``addData`` was set to
                    True and ``descale`` is True, the normalization that was
                    applied to the columns of the data matrix is applied to
                    the entries of :math:`z^k`, meaning that one may use it to
                    make predictions using unnormalized data. However, if a
                    linear operator was added with ``addData`` via argument
                    ``linOp``, then a warning message will be printed and the
                    solution vector will not be descaled.

            ergodic : :obj:`bool` or :obj:`string`,optional
                    Whether to return
                    the primal iterate :math:`z^k`, or one of its two averaged
                    versions. If ``False``, return the primal iterate. If "simple",
                    return :math:`\frac{1}{k}\sum_{t=1}^k z^k`; if "weighted", return

                    .. math::
                      \frac{\sum_{t=1}^k \tau_t z^t}{\sum_{t=1}^k\tau_t}

                    where :math:`\tau_t` are the stepsizes used in the hyperplane projections.

        Returns
        -------
            z : 1D numpy array
                :math:`z^k`

        '''

        if self.runCalled == False:
            raise Exception("Method not run yet, no solution to return. Call run() first.")

        if ergodic == "simple":
            z2use = self.zbar
        elif ergodic == "weighted":
            z2use = self.zbarWeighted
        else:
            z2use = self.z


        if descale:
            if self.normalize:
                if self.linOpUsedWithLoss:
                    print("Warning: Cannot descale because of the presence of a linear operator")
                    print("composed with the data. Just returning the unnormalized solution vector")
                    out = z2use
                else:
                    out = z2use[1:]/self.scaling[1:]
                    out = concatenate((array(z2use[0]),out))
            else:
                out  = z2use
        else:
            out  = z2use

        if (self.intercept==False):
            out = out[1:]

        return out




    def getPrimalViolation(self):
        r'''
        Returns the current primal violation.  A solution is exactly optimal
        if both its primal and dual violation are zero.

        After at least one call to the method ``run``, this method returns a
        :obj:`float` equal to the primal violation.

        Recall the objective

        .. math::

          \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}
                    \frac{1}{n}\sum_{i=1}^n\ell (z_0 + a_i^\top H z,r_i)
                       + \sum_{j=1}^{n_r}\nu_j h_j(G_j z)

        In the notation of :cite:`for1`, the primal violation is

        .. math::
            \max\{\max_{i=1,..,n_b} \|H z^k - x_i^k\|_2 , \max_{j=1,..,n_r}\|G_jz^k - x_{j+n_b}^k\|_2\}

        where, :math:`n_b` is the number of blocks in the loss (controlled by ``nblocks``
        argument to ``run``).


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
        r'''
        Returns the current dual violation.  A solution is exactly optimal
        if both its primal and dual violation are zero.

        After at least one call to the method run(), returns a float
        equal to the dual violation.

        Recall the objective

        .. math::
            \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}
                    \frac{1}{n}\sum_{i=1}^n\ell (z_0 + a_i^\top H z,r_i)
                       + \sum_{j=1}^{n_r}\nu_j h_j(G_j z)

        In the notation of :cite:`for1`, dual violation is

        .. math::
            \max\{ \max_{i=1,..,n_b} \|y_i^k - w_i^k\|_2 , \max_{j=1,..,n_r} \|y_{j+n_b}-w_j^k\|_2\}

        where, :math:`n_b` is the number of blocks in the loss (controlled by ``nblocks``
        argument to ``run``).

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
        Returns array of history data from most recent invocation of ``run`` for which
        the ``keepHistory`` was set to ``True``.

        After at least one call to run with keepHistory set to ``True``, the function
        call::

            historyArray = psfObj.getHistory()

        returns a two-dimensional, five-row NumPy array with each column
        corresponding to an iteration for which the history statistics were
        recorded. The total number of columns is the number of iterations
        divided by the ``historyFreq`` parameter, which can be set as an
        argument to ``run`` and defaults to 10. In each row of this array, the
        rows have the following interpretation:

        0. Objective value
        1. Cumulative run time
        2. Primal violation
        3. Dual violation
        4. Value of the :math:`\phi(p^k)` quantity used in hyperplane construction

        If ``run`` has not yet been called with ``keepHistory`` set to True,
        this function will raise an Exception when called.

        If ``keepHistory`` is set to True and a regularizer or the loss is added without
        implementing its value method, an exception will be raised.

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
            resetIterate=False,verbose=False,ergodic=None,equalizeStepsizes=False):
        r'''
        Run projective splitting.

        Parameters
        ----------
            primalTol : :obj:`float`,optional
                Continue running algorithm if primal error is greater than ``primalTol``.
                In the notation of :cite:`for1`, the primal violation is

                .. math::
                    \max\{\max_{i=1,..,n_b} \|H z^k - x_i^k\|_2 , \max_{j=1,..,n_r}\|G_{j}z^k - x_{j+n_b}^k\|_2\}

                where, :math:`n_b` is the number of blocks in the loss (controlled by ``nblocks``
                argument to ``run``) and :math:`n_r` is the number of regularizers.
                To terminate the method, both primal error and dual error
                must be smaller than their respective tolerances, or the
                number of iterations must exceed ``maxIteration``. Default 1e-6.

            dualTol : :obj:`float`,optional
                Continue running algorithm if dual error is greater than dualTol.
                The dual error is

                .. math::
                    \max\{ \max_{i=1,..,n_b} \|y_i^k - w_i^k\|_2 , \max_{j=1,..,n_r} \|y_{j+n_b}-w_j^k\|_2\}

                where, :math:`n_b` is the number of blocks in the loss (controlled by ``nblocks``
                argument to ``run``) and :math:`n_r` is the number of regularizers.
                To terminate the method, both primal error and dual error
                must be smaller than their respective tolerances, or the
                number of iterations must exceed ``maxIteration``. Default 1e-6.

            maxIterations : :obj:`int`,optional
                Terminate algorithm as soon as it has run for
                more than ``maxIterations`` iterations. Default is ``None``,
                which means not to terminate until the ``primalTol`` and ``dualTol``
                conditions are reached.

            keepHistory : :obj:`bool`,optional
                If ``True``, record the algorithm history (see the ``getHistory``
                method). Default is ``False``. Note that to keep history requires
                computing the objective value, which may be slow for large
                problems.

            historyFreq : :obj:`int`,optional
                If ``keepHistory`` is ``True``, history information is recorded
                every ``historyFreq`` iterations.  Defaults to 10.

            nblocks : :obj:`int`,optional
                Number of blocks in the projective splitting decomposition
                of the loss. Defaults to 1. Blocks are contiguous indices and the
                number of indices in each block varies by at most one.

                ``nblocks`` must be an integer in the range 1 to :math:`n`, where
                :math:`n` is the number of observations.

                In conjunction with the greedy activation method (see below), choosing
                ``nblocks`` larger than 1 has been shown to greatly improve algorithm
                performance for some problem classes.

                Suppose ``nblocks`` is set to :math:`b` and the number of
                observations is :math:`n`.  Then the first :math:`n\!\! \mod b`
                blocks have :math:`\lceil n/b \rceil` observations and
                the remainder have :math:`\lfloor n/b \rfloor` observations.
                If :math:`b` divides :math:`n`, this means that all blocks
                have :math:`n/b` observations.

                For example, if number of observations is 100 and nblocks is set to 10
                then the blocks would be

                    [
                    [0,1,...,9],
                    [10,11,...,19],
                    ...
                    [90,91,...,99]
                    ]

                If the number of observations is 105 and nblocks is set to 10, then
                the blocks would be 5 blocks of size 11 and 5 blocks of 10, that is,

                    [
                    [0,1,...,10],
                    [11,12,..22],
                    ...
                    [44,45,...,54],
                    [55,56,...,64],
                    ...
                    [95,96,...,104]
                    ]

            blockActivation : :obj:`string`,optional
                Strategy for selecting blocks of the loss to process at each iteration.
                Defaults to "greedy". Other valid choices are "random" and "cyclic".
                If there is only one block, all these choices are equivalent.

            blocksPerIteration : :obj:`int`,optional
                Number of blocks to update in each iteration. Defaults to 1.  Must
                be a positive integer in the range 1 to ``nblocks``.

            resetIterate : :obj:`bool`,optional
                If ``True``, the current
                values of all working variables (if ``run`` has been called before) in
                the projective splitting algorithm (eg: :math:`z^k, w_i^k` etc) are
                overwritten with zero vectors before starting the run. Defaults
                to ``False``, meaning that the algorithm starts from its previous state.

            verbose : :obj:`bool`,optional
                If ``True``, will print iteration counts every 100 iterations.
                Defaults to ``False``.

            ergodic : :obj:`bool` or :obj:`string`, optional

               If ``keepHistory=True``, whether to compute the objective at the primal
               iterate :math:`z^k`, or one of its two averaged versions. If
               ``False``, use the primal iterate. If "simple", evaluate at
               :math:`\frac{1}{k}\sum_{t=1}^k z^t`; if "weighted", evaluate at

               .. math::
                  \frac{\sum_{t=1}^k\tau_t z^t}{\sum_{t=1}^k\tau_t}

               where :math:`\tau_t` are the stepsizes used in the hyperplane projections.

            equalizeStepsizes : :obj:`bool`, optional
                Applies only when using backtracking loss processors
                (``Forward2Backtrack`` and ``Forward1Backtrack``).  If
                ``True``, set the regularizer stepsizes according to the
                stepsizes returned by backtracking. Defaults to ``False``.

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

        blocksPerIteration = ui.checkUserInput(blocksPerIteration,int,'int','blocksPerIteration',default=1,low=1,
                                               lowAllowed=True)

        try:
            if blocksPerIteration >= self.nDataBlocks:
                blocksPerIteration = self.nDataBlocks
        except:
            print("Warning: blocksPerIteration should be a positive int")
            print("Setting blocksPerIteartion to 1")
            blocksPerIteration =1

        self.partition = ut.createApartition(self.nrowsOfA,self.nDataBlocks,self.sparseObservationMtx)

        self.__createListOfSparseMatrices()

        self.__setUpRegularizers()

        self.nDataBlockVars = self.ncolsOfA + 1 # extra 1 for the intercept term


        resetIterate = ui.checkUserBool(resetIterate,"resetIterate")

        if resetIterate or self.internalResetIterate:
            self.internalResetIterate = False
            self.__initializeVariables()

        keepHistory = ui.checkUserBool(keepHistory,"keepHistory")
        verbose = ui.checkUserBool(verbose,"verbose")

        if maxIterations != None:
            maxIterations = ui.checkUserInput(maxIterations,int,'int','maxIterations',
                                              default=1000,low=1,lowAllowed=True)

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
        sumTau = 0.0
        interTime = 0.0

        ################################
        # BEGIN MAIN ALGORITHM LOOP
        ################################

        while(self.k < maxIterations):

            if verbose and (self.k%100 == 0):
                print('iteration = {}'.format(self.k))
            t0 = time()
            self.__updateLossBlocks(blockActivation,blocksPerIteration)
            self.__equalizeStepsizes(equalizeStepsizes)
            self.__updateRegularizerBlocks()

            if (self.primalErr < primalTol) & (self.dualErr < dualTol):
                print("primal and dual tolerance reached, finishing run")
                break

            phi,tau = self.__projectToHyperplane() # update (z,w1...wn) from (x1..xn,y1..yn,z,w1..wn)

            if phi == "converged":
                print("Gradient of the hyperplane is 0, converged, finishing run")
                break

            self.zbar = (self.k/(self.k+1.0))*self.zbar + (1.0/(self.k+1))*self.z

            if tau > 0:
                self.zbarWeighted = (sumTau/(sumTau+tau))*self.zbarWeighted + (tau/(sumTau+tau))*self.z
                sumTau += tau

            t1 = time()
            interTime += t1-t0

            if keepHistory and (self.k % historyFreq == 0):
                objective.append(self.getObjective(ergodic=ergodic))
                times.append(times[-1]+interTime)
                interTime = 0.0
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


    def __equalizeStepsizes(self,equalizeStepsizes):
        if equalizeStepsizes:
            steps = getattr(self.process,"steps",None)
            if steps is not None :
                averageStep = sum(steps)/len(steps)
                # set all regularizers new stepsize equal to averageStep, except
                # the embedded regularizer (if any).
                for reg in self.allRegularizers:
                    reg.setStep(averageStep)


    def __createListOfSparseMatrices(self):
        # for sparse matrices, it is much more efficient (faster) to preslice the
        # matrices and store a list of pre-sliced matrices.
        # To make this backwards compatible, we need to replace partition with just range(nblocks)
        # so that calls like thisSlice = partition[block] just return the block.
        if self.sparseObservationMtx:
            self.Afull = self.A
            self.yresponseFull = self.yresponse
            self.A = []
            self.yresponse = []
            for part in self.partition:
                self.A.append(self.Afull[part])
                self.yresponse.append(self.yresponseFull[part])
            self.partition = range(len(self.partition))


    @staticmethod
    def __addLinear(regObj,linearOp=None):
        if linearOp is None:
            regObj.linearOp = ut.MyLinearOperator(matvec=lambda x:x,rmatvec=lambda x:x)
            regObj.linearOpUsed = False
        else:
            try:
                if not issparse(linearOp):
                    regObj.linearOp = aslinearoperator(linearOp)
                else:
                    regObj.linearOp = ut.MySparseLinearOperator(linearOp)
                regObj.linearOpUsed = True
            except:
                raise Exception("linearOp invalid. Use scipy.sparse.linalg.aslinearoperator or a scipy sparse matrix format")

    def __initializeVariables(self):
        self.z = zeros(self.nPrimalVars+1)
        self.zbar = zeros(self.nPrimalVars+1)
        self.zbarWeighted = zeros(self.nPrimalVars+1)
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
            if self.embedded.getStep() != self.process.getStep():
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
                if len(self.allRegularizers)>0:
                    step = self.allRegularizers[0].getStep()
                else:
                    step = 1.0
                self.addRegularizer(Regularizer(lambda x,scale: x, lambda x: 0,step=step))

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
        tau = 0.0

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

        return phi,tau



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
        if self.sparseObservationMtx:
            AHz = self.Afull.dot(Hz)

            getVal = self.loss.value(AHz,self.yresponseFull)
        else:
            AHz = self.A.dot(Hz)
            getVal = self.loss.value(AHz,self.yresponse)
        if getVal is None:
            print("ERROR: If you don't implement a losses value func, set getHistory to")
            print("False and do not compute objective values")
            raise Exception("Losses value function is not implemented. Cannot compute objective values.")
        currentLoss = (1.0/self.nrowsOfA)*sum(getVal)
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
