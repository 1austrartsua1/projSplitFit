import sys
sys.path.append('../')
import projSplitFit as ps
from scipy.sparse.linalg import LinearOperator
import numpy as np
import regularizers
import lossProcessors

### get the trip-advisor reviews data
print("Getting data...")
import scipy.sparse as sp
X = sp.load_npz('data/S_train.npz') # training matrix (scipy.sparse format)
H = sp.load_npz('data/S_A.npz')     # this matrix also in scipy.sparse format
r = np.load('data/y_train.npy')     # training labels

### create projective splitting object
projSplit = ps.ProjSplitFit()

### add data composed with linear operator
projSplit.addData(X,r,loss=2,linearOp=H,normalize=False)

### first regularizer
mu=0.5
lam=1e-5
def applyG(x):
    return x[:-1]
def applyGtranspose(v):
    return np.concatenate((v,np.array([0])))
(_,nv) = H.shape
shape = (nv-1,nv)
G = LinearOperator(shape,matvec=applyG,rmatvec = applyGtranspose)
projSplit.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G)

### second regularizer
regObj2 = regularizers.L1(scaling=lam*(1-mu))
projSplit.addRegularizer(regObj2,linearOp=H)

### In some cases there is a random block selection even though we use greedy
### This occurs when no blocks have negative \phi_i
### Fix the random number seed so results are reproducible
np.random.seed(1)

# Set dual scaling parameter
projSplit.setDualScaling(1e-2)

# Problem is run to low accuracy since it is very difficult and ill-conditioned
projSplit.run(nblocks=10,maxIterations=None,verbose=True,
              primalTol=1e-2,dualTol=1e-2)

# Get results
objVal = projSplit.getObjective()
solVector = projSplit.getSolution()

print(f"Objective value = {objVal}")
