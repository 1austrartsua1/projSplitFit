

import projSplit as ps
from scipy.sparse.linalg import LinearOperator
import numpy as np
import regularizers

### get the trip-advisor reviews data
print("getting data...")
import scipy.sparse as sp
X = sp.load_npz('data/S_train.npz') # training matrix (scipy.sparse format)
H = sp.load_npz('data/S_A.npz')     # this matrix also in scipy.sparse format
y = np.load('data/y_train.npy')     # training labels

### create projective splitting object
projSplit = ps.ProjSplitFit()

### add data composed with linear operator
projSplit.addData(X,y,loss=2,linearOp=H,normalize=False)

### first regularizer
mu=0.5
lam=1e-4

def applyG(x):
    return x[:-1]
def applyGtranspose(v):
    return np.concatenate((v,np.array([0])))
(_,ngamma) = H.shape
shape = (ngamma-1,ngamma)
G = LinearOperator(shape,matvec=applyG,rmatvec = applyGtranspose)
projSplit.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G)

### second regularizer
regObj2 = regularizers.L1(scaling=lam*(1-mu))
projSplit.addRegularizer(regObj2,linearOp=H)

print("running projective splitting...")
projSplit.run(nblocks=10,maxIterations=1000)
print(f"optimal val = {projSplit.getObjective()}")

print("training error")