import sys
sys.path.append('../')
import numpy as np
import projSplit as ps
from regularizers import L1
### Basic Setup with a Quadratic Loss
m = 500
d = 1000
A = np.random.normal(0,1,[m,d])
y = np.random.normal(0,1,m)

### User-defined and Multiple regularizers
from regularizers import Regularizer
def prox_g(z,sigma):
    return (z>=0)*z
def value_g(x):
    if any(x < -1e-7):
        return float('Inf')
    return 0.0

regObjNonneg = Regularizer(prox=prox_g, value=value_g)
gamma = 1e1
projSplit = ps.ProjSplitFit(gamma)
projSplit.addRegularizer(regObjNonneg)
lam1 = 0.1
projSplit = ps.ProjSplitFit()
projSplit.addData(A,y,loss=2,intercept=False,normalize=False)
regObj = L1(scaling=lam1)
projSplit.addRegularizer(regObj)
regObjNonneg = Regularizer(prox=prox_g, value=value_g)
projSplit.addRegularizer(regObjNonneg)
projSplit.run()
optimalVal = projSplit.getObjective()
z = projSplit.getSolution()
print(f"optimal value = {optimalVal}")


