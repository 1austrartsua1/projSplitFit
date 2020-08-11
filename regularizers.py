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
               \frac{1}{n}\sum_{i=1}^n \ell (z_0 + a_i^\top H z,r_i)
                  + \sum_{j=1}^{n_r}\nu_j h_j(G_j z)

      The regularizer class is used to define each :math:`\nu_j h_j(G_j z)`` term,
      with the exception of the optional linear operator :math:`G_j`, which is supplied
      when calling ``addRegularizer`` to introduce the regularizer to the formulation.
            
      You may use standard built-in regularizers, or create objects of this
      class to define new regularizers.  When defining your own regularizers,
      you must provide a function implementing the regularizer's proximal
      operator ("prox").   If you wish to compute objective values, you must
      also supply a function to compute the regularizer value.
    '''

    def __init__(self,prox,value=None,scaling=1.0,step=1.0):
        r''' It is only necessary to define *value* if you wish to compute
            objective function values, either by calling ``getObjective`` or
            by using the ``keepHistory`` option of the ``run`` method of
            ``projSplitFit``. 

            parameters
            ----------
            prox : function 
                must be a function of two parameters: a
                ``numpy``-style array :math:`s` and a positive ``float``
                :math:`\eta`.  This function must return the vector
                :math:`\text{prox}_{\eta h_j}(s) = \arg\min_{x}\left\{\eta h_j(x) +
                (1/2)\| x - s \|^2\right\}`` for arbitrary inputs
                :math:`s` and :math:`0<\eta<\infty`.

            value : function,optional
                must be a function of one parameter:  a numpy-style
                array. Must returns a float which is the value of :math:`h(x)``.
                The default is ``None``, meaning undefined. The returned value
                should not include the scaling factor :math:`\nu_j`.

            scaling : :obj:`float`,optional
                the objective scaling factor :math:`\nu_j` to use with this regularizer.
                The function will appear in the objective as :math:`\nu_j h_j(G_j z)`.
                Must be positive and defaults to 1.0.

            step : :obj:`float`,optional 
                the stepsize :math:`\eta` to use in the proximal steps of
                projective splitting with this regularizer. Must be positive
                and defaults to 1.0.  Will be overridden on an
                interation-by-iteration basis if the ``equalizeStepsizes``
                option is enabled in the ``run`` method of ``projSplitFit``.
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
        Set the scaling factor :math:`\nu_j`.

        Parameters
        ----------
        scaling : :obj:`float`
            scaling factor; must be a positive and finite
        '''
        self.nu = ui.checkUserInput(scaling,float,'float','scaling',default=1.0,low=0.0,lowAllowed=True)


    def setStep(self,step):
        r'''
        Set the stepsize :math:`\eta` for proximal steps for this regularizer.

        Parameters
        ------------
        step : :obj:`float`
            stepsize; must be finite and positive
        '''
        self.step = ui.checkUserInput(step,float,'float','step',default=1.0,low=0.0)


    def getScaling(self):
        r'''
        Get the scaling :math:`\nu_j` being used for this regularizer.

        Returns
        -------
        scaling : :obj:`float`

        '''
        return self.nu


    def getStep(self):
        r'''
        get the stepsize :math:`\eta` being used in the proximal steps for 
        this regularizer.

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
    Returns an L1 regularizer. The output is an object of class 
    ``regularizers.Regularizer`` suitable as input to ``ProjSplitFit.addRegularizer``.

    *scaling* is the coefficient :math:`\nu_j` that will
    be applied to the function. That is, the regularizer will appear as
    :math:`\nu_j\|z\|_1` or :math:`\nu_j\|G_j z\|_1` in the objective
    formulation, depending on whether a linear operator :math:`G_j` is
    is supplied when it is introduced into the formulation with
    ``addRegularizer``.

    *step* is the stepsize :math:`\eta` that projective splitting will use
    for proximal steps using this regularizer, unless overridden by the
    ``equalizeStepsizes`` option of the ``run`` method of ``projSplitFit``.

    Parameters
    -----------

    Scaling : :obj:`float`,optional
        Defaults to 1.0.  Must be positive and finite
    Stepsize : :obj:`float`,optional
        Defaults to 1.0.  Must be positive and finite

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
    Create an L2 squared regularizer. The output is an object of class
    ``regularizers.Regularizer`` which may be passed to
    ``ProjSplitFit.addRegularizer``.

    *scaling* is the coefficient :math:`\nu_j` that will be applied to the
    regularizer in the objective. That is, ithe regularizer will appear as
    :math:`(\nu_j/2)\|\,\cdot\,\|_2^2`.  Note the factor of 0.5.

    *step* is the stepsize that projective splitting will use for the proximal steps
    performed on this regularizer.

    Parameters
    -----------

    Scaling : :obj:`float`,optional
        Defaults to 1.0.  Must be positive and finite.
    Stepsize : :obj:`float`,optional
        Defaluts to 1.0.  Must be positive and finite.

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
    Create an L2-norm regularizer. Not to be confused with the ``L2sq`` regularizer,
    which is the same function squared and divided by 2.

    The output is an object of class ``regularizers.Regularizer``,
    which may be passed to ``ProjSplitFit.addRegularizer``.

    *scaling* is the coefficient :math:`\nu_j` that will
    be applied to the function in the objective. That is, the regularizer
    will appear as :math:`\nu_j\|\,\cdot\,\|_2`.

    *step* is the stepsize :math:`\eta` that projective splitting will use in
    proximal steps with respect to this regularizer.

    Parameters
    -----------

    Scaling : :obj:`float`,optional
        Defaults to 1.0.  Must be positive and finite.
    Stepsize : :obj:`float`,optional
        Defaluts to 1.0.  Must be positive and finite.

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
