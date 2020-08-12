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
    r'''
    Objects of this class may be used as the ``loss`` argument
    of the ``ProjSplitFit.addData`` method, to define customized loss functions.
    That argument also accepts ``float`` or ``int`` values :math:`p > 1`, which
    are interpreted as specifying the :math:`\ell_p^p` loss, or the string
    "logistic" to specify the logistic loss function.

    Other choices require creating a ``LossPlugIn`` object.  This in turn
    requires supplying a function to compute the derivative of the loss function.
    If you plan to compute objective function values, you must also supply a
    function to compute the loss function value.
    '''

    def __init__(self,derivative,value=None):
        r'''
        You need only supply a *value* function if you wish to compute
        objective function values (either with ``ProjSplitFit.getObjective``
        or by enabling history collection in ``ProjSplitFit.run``).

        Parameters
        ----------
        derivative : :obj:`function`
            Function of two 1D ``numpy`` arrays of the same length, the first
            containing predicted values and the second containing actual
            response values.  Must output an array of the same length as the
            two inputs, whose elements consists of partial derivatives with
            respect to the predicted values. Specifically, supposing that the
            two input arrays are :math:`q = [q_0 \; q_1 \;
            \cdots q_k]` and :math:`q = [r_0 \; r_1 \; \cdots r_k]`, the
            returned array should be contain elements of the form
            :math:`\frac{\partial}{\partial q_i}\ell(q_i,r_i)` for each
            input index :math:`i`.

        value : :obj:`function`, optional
            Must accept two ``float`` arguments and return a single ``float``.
            If supplied the arguments :math:`q_i` (for the prediction) and :math:`r_i`
            (for the response), the function should return :math:`\ell(q_i,r_i)`.
            Defaults to ``None``.  If the default is used, however, attempting to
            compute the objective value will raise an exception.
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
