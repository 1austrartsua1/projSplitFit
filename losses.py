# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:11:49 2020

@author: pjohn
"""
import numpy as np

#-----------------------------------------------------------------------------
# loss class and related objects
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
