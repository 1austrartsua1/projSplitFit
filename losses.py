# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:11:49 2020

@author: pjohn
"""
from numpy import log
from numpy import exp
from numpy import isinf
from numpy import nan_to_num
from numpy import ones


#-----------------------------------------------------------------------------
# loss class and related objects
#-----------------------------------------------------------------------------
    
class Loss(object):
    '''
    Loss class for defining the loss in ProjSplitFit.addRegularizer method.
    
    Used internally within the addRegularizer method. 
        
    '''
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
                    raise Exception("Error, lossFunction p is not >= 1")                                                                      
            except:                              
                print("for loss, input either an int or float >= 1, 'logistic', or an object derived from class LossPlugIn")
                raise Exception("lossFunction input error")
            
                
def LR_loss(yhat,y):
    score = -yhat*y
    
    return LR_loss_from_score(score)

def LR_loss_from_score(score):
    pos = log(1 + exp(score))
    pos2 = (~isinf(pos))*nan_to_num(pos)
    neg = score + log(1+exp(-score))
    neg2 = (~isinf(neg)) * nan_to_num(neg)
    coef = 0.5*ones(len(pos))
    coef = coef+0.5*isinf(pos)+0.5*isinf(neg)
    return coef*(pos2+neg2) 

def LR_derivative(yhat,y):
    score = -yhat*y    
    return -exp(score - LR_loss_from_score(score))*y

    
class LossPlugIn(object):
    '''
    Objects of this class may be used as the input ``loss`` to the 
    ``ProjSplitFit.addData`` method to define custom losses. 
    
    The user may set the argument ``loss`` to ``ProjSplitFit.addData`` to an
    integer :math:`p\geq 1` to use the :math:`\ell_p^p` loss, or they may set it to
    "logistic" to use the logistic loss. 
    
    However, if the user would like to define their own loss, then they must 
    write a function for computing the derivative of the loss and pass it into 
    the constructor to get an object of this class. This can then be used as 
    the input ``loss`` to ``ProjSplitFit.addData``.
    
    '''
    def __init__(self,derivative,value=None):
        '''
        Only implement value if you wish to compute objective function values 
        of the outputs of ``ProjSplitFit`` to monitor progress. It is not necessary for the operation
        of ``ProjSplitFit``. However, if the value function is 
        set to None, but then the ``ProjSplitFit.getObjective`` method is called, 
        then it will raise an Exception. Similarly if ``ProjSplitFit.run`` is called 
        with the ``keepHistory`` argument set to True.
        
        Parameters
        ----------
        derivative : function
            Function of two 1D NumPy arrays of the same length. 
            Must output an array of the same length
            as the two inputs which is the derivative wrt the first argument of 
            the loss evaluated at each pair of elements in the input arrays. 
            That is, for inputs::
                
                [x_1,x_2,...,x_n], [y_1,y_2,...,y_n]
                        
            output:: 
                
                [z_1,z_2,...,z_n]
                
            where 
            
            ..  math::
                
                z_i = \\frac{\\partial}{\\partial x}\\ell(x_i,y_i)
            
            and the partial derivative is w.r.t. the first argument to :math:`\ell`.
                                  
        value : function,optional
            Must handle two float inputs and output a float. Defaults to None, not supported. 
            Outputs

            .. math::            
                
                \ell(x,y)
            
            for inputs x and y. 
            
    
        '''
        try:
            test = ones(100)
            if value is not None:
                output = value(3.7,4.5)
                output = float(output)
            output = derivative(test,test)                        
            if len(output)!= 100:
                raise Exception
        except:
            print("Value should be a function of one array which outputs a float")
            print("derivative is a function of two arrays of the same shape which outputs an array")
            print("of the same shape")
            raise Exception("Value or derivative incorrect")
        
        if value is None:
            self.value = lambda x,y:None
        else:    
            self.value = value
        
        self.derivative = derivative
