import sys
sys.path.append('../')
import numpy as np
import projSplitFit as ps
### Basic Setup with a Quadratic Loss

### Test on random data
m = 500
d = 1000
np.random.seed(1)
A = np.random.normal(0,1,[m,d])
r = np.random.normal(0,1,m)

projSplit = ps.ProjSplitFit()
projSplit.addData(A,r,loss=2,intercept=False)
projSplit.run()
optimalVal = projSplit.getObjective()
z = projSplit.getSolution()
print(f"Objective value LS prob = {optimalVal}")

### changing the dual scaling
gamma = 1e2
projSplit.setDualScaling(gamma)
projSplit.run()

### adding a regularizer
from regularizers import L1
lam1 = 0.1
regObj = L1(scaling=lam1)
projSplit.addRegularizer(regObj)
projSplit.run()
optimalVal = projSplit.getObjective()
print(f"Objective value L1LS prob = {optimalVal}")