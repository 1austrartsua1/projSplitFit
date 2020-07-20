# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:47:47 2020

@author: pjohn
"""
from numpy.linalg import norm
import userInputVal as ui
from numpy import ones


#-----------------------------------------------------------------------------
# Regularizer class and related objects
#-----------------------------------------------------------------------------

class Regularizer(object):
    r'''
      Regularizer class to use as an input to the ``ProjSplitFit.addRegularizer`` method.

      Recall the objective function

      .. math::

        \min_{z\in\mathbb{R}^d,z_0\in \mathbb{R}}
               \frac{1}{n}\sum_{i=1}^n \ell (z_0 + a_i^\top H z,y_i)
                  + \sum_{j=1}^{n_r}h_j(G_j z)

      The regularizer class essentially defines each :math:`h_i(G_i z)`` term via
      methods for evaluating the prox of :math:`h_i` and the function itself.
      Note the matrix :math:`G_i` is added in the addRegularizer method of projSplitFit.
      
      The user may use objects of this class to define regularizers, or may use one 
      of the built-in regularizers. 
      
      To use this class, one must define a function 
      for computing the prox of the regularizer and can then use that as an input 
      to the constructor to create a ``Regularizer`` object. 
      
    '''
    def __init__(self,prox,value=None,scaling=1.0,step=1.0):
        r'''
            Only define *value* if you wish to compute objective function values
            within ``ProjSplitFit`` to monitor progress, as its not necessary for the
            actual operation of ``ProjSplitFit``. However, if the value function is
            set to None, but then the ``ProjSplit.getObjective()`` method is called,
            then it will raise an Exception.

            parameters
            ----------
            prox : function
                must be a function of two parameters: a numpy-style array
                and a float which is the multiple applied to the function.
                That is, this function must return :math:`\text{prox}_{\eta h}(x)``
                for arbitrary inputs :math:`x` and :math:`\eta`.

            value : function,optional
                must be a function of one parameter:  a numpy-style
                array. Must returns a float which is the value of :math:`h(x)``.
                Default is None, meaning not defined. Note that this is the value
                of the *unscaled* function. In other words, with a scaling of 1.

            scaling : :obj:`float`,optional
                Scaling to use with this regularizer in the objective.
                The function will appear in the objective as

                .. math::
                  \nu h(x)

                for a scaling :math:`\nu`. Defaults to 1.0

            step : :obj:`float`,optional
                Stepsize to use in the proximal steps of projective splitting
                with this regularizer. Defaults to 1.0
        '''
        try:
            test = ones(100)

            if value is not None:
                output = value(test)
                output = float(output)

            output = prox(test,1.1)
            if len(output) != 100:
                print("Error: make sure prox outputs an array of same length as first input")
                raise Exception("Error: prox method passed into Regularizer invalid")
        except:
            print("value (if not None) must be a function of one numpy style array and return a float")
            print("prox must be a function with two arguments, the first being a numpy style array")
            print("and the second being a float. Must return an array same size as input")
            raise Exception("Error: value or prox is invalid")


        self.value = value
        self.prox = prox

        self.nu = ui.checkUserInput(scaling,float,'float','scaling',default=1.0,low=0.0,lowAllowed=True)
        self.step = ui.checkUserInput(step,float,'float','step',default=1.0,low=0.0)

    def setScaling(self,scaling):
        r'''
        Set the scaling. That is, in the objective, the regularizer will
        be scaled by scaling=nu. It will appear as

        .. math::
          \nu h(x)

        Parameters
        ----------
        scaling : :obj:`float`
            scaling
        '''
        self.nu = ui.checkUserInput(scaling,float,'float','scaling',default=1.0,low=0.0,lowAllowed=True)

    def setStep(self,step):
        '''
        Set the stepsize being used in the proximal steps for this regularizer by projective
        splitting.

        Parameters
        ------------
        step : :obj:`float`
            stepsize
        '''
        self.step = ui.checkUserInput(step,float,'float','step',default=1.0,low=0.0)

    def getScaling(self):
        '''
        Get the scaling being used for this  regularizer in the objective.

        Returns
        -------
        scaling : :obj:`float`

        '''
        return self.nu

    def getStepsize(self):
        '''
        get the stepsize being used in the proximal steps for this regularizer
        by projective splitting.

        Returns
        ------
        stepsize : :obj:`float`

        '''
        return self.step

    def evaluate(self,x):
        if self.value is None:
            return None
        else:
            return self.nu*self.value(x)

    def getProx(self,x):
        return self.prox(x,self.nu*self.step)



def L1(scaling=1.0,step=1.0):
    r'''
    Create the L1 regularizer. The output is an object of class ``regularizers.Regularizer``
    which may be input to ``ProjSplitFit.addRegularizer``.

    Scaling is the coefficient :math:`\nu` that will
    be applied to the function in the objective. That is, it will appear as

    .. math::
      \nu\|z\|_1

    step is the stepsize that projective splitting will use for the proximal steps
    w.r.t. this regularizer.

    Parameters
    -----------

    Scaling : :obj:`float`,optional
        Defaults to 1.0
    Stepsize : :obj:`float`,optional
        Defaluts to 1.0

    Returns
    --------
    regObj : :obj:`regularizers.Regularizer` object
    '''
    def L1val(x):
        return norm(x,1)

    def L1prox(x,scale):
        out = (x> scale)*(x-scale)
        out+= (x<-scale)*(x+scale)
        return out

    out = Regularizer(L1prox,L1val,scaling,step)
    return out

def L2sq(scaling=1.0,step=1.0):
    r'''
    Create the L2 squared regularizer. The output is an object of class ``regularizers.Regularizer``
    which may be input to ``ProjSplitFit.addRegularizer``.

    Scaling is the coefficient :math:`\nu` that will
    be applied to the function in the objective. That is, it will appear as

    .. math::
      \frac{\nu}{2}\|z\|_2^2.

    Note the factor of 0.5.
    
    step is the stepsize that projective splitting will use for the proximal steps
    w.r.t. this regularizer.

    Parameters
    -----------

    Scaling : :obj:`float`,optional
        Defaults to 1.0
    Stepsize : :obj:`float`,optional
        Defaluts to 1.0

    Returns
    --------
    regObj : :obj:`regularizers.Regularizer` object
    '''
    def val(x):
        return 0.5*norm(x,2)**2

    def prox(x,scale):
        return(1+scale)**(-1)*x

    out = Regularizer(prox,val,scaling,step)
    return out

def L2(scaling=1.0,step=1.0):
    r'''
    Create the L2 norm regularizer. Not to be confused with the L2sq regularizer,
    which is this function *squared*.

    The output is an object of class ``regularizers.Regularizer``
    which may be input to ``ProjSplitFit.addRegularizer``.

    Scaling is the coefficient :math:`\nu` that will
    be applied to the function in the objective. That is, it will appear as

    .. math::
      \nu\|z\|_2

    step is the stepsize that projective splitting will use for the proximal steps
    w.r.t. this regularizer.

    Parameters
    -----------

    Scaling : :obj:`float`,optional
        Defaults to 1.0
    Stepsize : :obj:`float`,optional
        Defaluts to 1.0

    Returns
    --------
    regObj : :obj:`regularizers.Regularizer` object
    '''
    def val(x):
        return norm(x,2)

    def prox(x,scale):
        normx = norm(x,2)
        if normx <= scale:
            return 0*x
        else:
            return (normx - scale)*x/normx

    out = Regularizer(prox,val,scaling,step)
    return out
