import numpy as np
import sys
sys.path.append('../')
import projSplitFit as ps

import losses as ls
def deriv(x,y):
    return (x>=y)*(x-y)
def val(x,y):
    return (x>=y)*(x-y)**2

projSplit = ps.ProjSplitFit()
m = 500
d = 200
np.random.seed(1)
A = np.random.normal(0,1,[m,d])
r = np.random.normal(0,1,m)
loss = ls.LossPlugIn(derivative=deriv, value=val)
projSplit.addData(A,r,loss=loss)
projSplit.setDualScaling(10.0)
projSplit.run(verbose=True)
print(f"Objective value = {projSplit.getObjective()}")


