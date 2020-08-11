

import sys
sys.path.append('../')
import projSplitFit as ps 
import losses as ls
from regularizers import L1

from numpy.random import normal




def test_one_sided():
    projSplit = ps.ProjSplitFit()

    def deriv(x,y):
        return (x>=y)*(x-y)

    def val(x,y):
        return (x>=y)*(x-y)**2

    loss = ls.LossPlugIn(deriv)
    loss = ls.LossPlugIn(deriv,val)
    m = 20
    d = 50
    A = normal(0,1,[m,d])
    y = normal(0,1,m)

    projSplit.addData(A,y,loss=loss,intercept=False,normalize=False)
    projSplit.addRegularizer(L1())
    projSplit.run(keepHistory=False,nblocks=10)


    primTol = projSplit.getPrimalViolation()
    dualTol = projSplit.getDualViolation()

    assert primTol <1e-6
    assert dualTol <1e-6
