import numpy as np
import scipy
import sys
sys.path.append('../')

# Define the 1-D variation linear variation operator
def applyOperator(x):
    return x[:(len(x)-1)] - x[1:]
def applyAdjoint(u):
    return np.pad(u,(0,1)) - np.pad(u,(1,0))
def varop1d(n):
    return scipy.sparse.linalg.LinearOperator(shape=(n-1,n),matvec=applyOperator,
                                              rmatvec=applyAdjoint)

from regularizers import L1
import projSplitFit as ps
m = 500
d = 1000
np.random.seed(1)
A = np.random.normal(0,1,[m,d])
r = np.random.normal(0,1,m)

lam1 = 1e-2
regObj = L1(scaling=lam1)
gamma = 0.1
projSplit = ps.ProjSplitFit(gamma)
projSplit.addData(A,r,loss=2)
G = varop1d(d)
projSplit.addRegularizer(regObj,linearOp=G)
projSplit.run(verbose=True)
print(f"optimal val = {projSplit.getObjective()}")
