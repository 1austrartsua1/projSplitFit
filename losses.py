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
    def __init__(self,value,derivative):
        try:
            test = ones(100)
            output = value(test)
            output = float(output)
            output = derivative(output,output)
            if len(output)!= 100:
                raise Exception
        except:
            print("Value should be a function of one array which outputs a float")
            print("derivative is a function of two arrays of the same shape which outputs an array")
            print("of the same shape")
            raise Exception("Value or derivative incorrect")
            
        self.value = value
        self.derivative = derivative
