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
    '''
      Regularizer class for use within the ProjSplitFit.addRegularizer method. 
      
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
        parameters
        ----------
        
        value : function 
            must be a function of one parameter:  a numpy-style 
            array x. Must returns a float which is the value of h(x)
            
        prox : function 
            must be a function of two parameters: a numpy-style array
            x and a float which is the scaling eta applied to the function. 
            That is, this function must return prox_{eta*h}(x) for inputs x and eta>=0. 
        
        nu : float,optional
            Scaling to use with this regularizer. Defaults to 1.0
            
        step : float,optional
            Stepsize to use in the proximal steps with this regularizer. 
            Defaults to 1.0
        '''
        try:
            test = ones(100)            
            output = value(test)
            output = float(output)
                
            output = prox(test,1.1)
            if len(output) != 100:
                print("Error: make sure prox outputs an array of same length as first input")
                raise Exception("Error: prox method passed into Regularizer invalid")
        except:
            print("value must be a function of one numpy style array and return a float")
            print("prox must be a function with two arguments, the first being a numpy style array")
            print("and the second being a float. Must return an array same size as input")
            raise Exception("Error: value or prox is invalid")
            
        self.value = value 
        self.prox = prox 
        
        self.nu = ui.checkUserInput(nu,float,'float','nu',default=1.0,low=0.0,lowAllowed=True)
        self.step = ui.checkUserInput(step,float,'float','step',default=1.0,low=0.0)
                       
    def setScaling(self,nu):
        '''
        Set scaling
        
        Parameters
        ----------
        nu : float
            scaling
        '''
        self.nu = ui.checkUserInput(nu,float,'float','nu',default=1.0,low=0.0,lowAllowed=True)

    def setStep(self,step):
        '''
        Set stepsize
        
        Parameters
        -------
        step : float
            stepsize
        '''
        self.step = ui.checkUserInput(step,float,'float','step',default=1.0,low=0.0)
            
    def getScaling(self):
        '''
        Get scaling
        
        Returns
        -------
        float : scaling
        '''
        return self.nu
    
    def getStepsize(self):
        '''
        get the stepsize
        
        Returns
        ------
        float : stepsize
        '''
        return self.step 
    
    def evaluate(self,x):        
        return self.nu*self.value(x)
    
    def getProx(self,x):                        
        return self.prox(x,self.nu*self.step)        
    
def L1val(x):
    return norm(x,1)

def L1prox(x,scale):    
    out = (x> scale)*(x-scale)
    out+= (x<-scale)*(x+scale)
    return out

def L1(scaling=1.0,step=1.0):
    '''
    Create the L1 regularizer.
    
    Parameters
    Scaling : float,optional
        Defaults to 1.0
    Stepsize : float,optional
        Defaluts to 1.0
    '''
    out = Regularizer(L1val,L1prox,scaling,step)    
    return out 

def partialL1(dimension,groups,scaling = 1.0):
    pass

def groupL2(dimension,groups,scaling = 1.0):
    pass
    