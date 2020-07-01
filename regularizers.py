# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 15:47:47 2020

@author: pjohn
"""
from numpy.linalg import norm

#-----------------------------------------------------------------------------
# Regularizer class and related objects
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
        prox: (function) must be a function of two parameters: a numpy-style array
            x and a scaling eta applied to the function. That is, this function must 
            return prox_{eta*h}(x) for inputs x and eta>=0. 
        '''
        self.value = value 
        self.prox = prox 
        
        try:            
            nu = float(nu)                                                    
            if nu>=0:
                self.nu = nu    
            else:
                print("Error: scaling must be >=0, keeping it set to 1.0")
                self.nu = 1.0 
                                                    
        except:            
            print("Error: scaling must be float>=0, setting it to 1.0")
            self.nu = 1.0 
            
        try:
            step = float(step)
            if step>=0:
                    self.step = step    
            else:
                print("Error: scaling must be >=0, keeping it set to 1.0")
                self.step = 1.0
            
        except:
            print("Error: scaling must be >=0, keeping it set to 1.0")
            self.step = 1.0
    
    def addLinear(self,linearOp,linearOpFlag):
        self.linearOp = linearOp        
        self.linearOpFlag = linearOpFlag        
        
    def setScaling(self,nu):
        try:        
            if nu>=0:
                self.nu = nu    
            else:
                print("Error: scaling must be >=0, keeping it set to 1.0")
                self.nu = 1.0 
        except:
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
    out = Regularizer(L1val,L1prox,scaling,step)    
    return out 

def partialL1(dimension,groups,scaling = 1.0):
    pass

def groupL2(dimension,groups,scaling = 1.0):
    pass
    