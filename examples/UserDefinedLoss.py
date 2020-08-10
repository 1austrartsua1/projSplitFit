import numpy as np
import sys
sys.path.append('../')
import projSplit as ps

import losses as ls
def deriv(x,y):
    return (x>=y)*(x-y)
def val(x,y):
    return (x>=y)*(x-y)**2

projSplit = ps.ProjSplitFit()
m = 500
d = 200
A = np.random.normal(0,1,[m,d])
y = np.random.normal(0,1,m)
loss = ls.LossPlugIn(derivative=deriv, value=val)
projSplit.addData(A,y,loss=loss)
projSplit.run()
print(f"optimal val = {projSplit.getObjective()}")

