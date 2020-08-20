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
X = sp.load_npz('data/trip_advisor/S_train.npz') # training matrix (scipy.sparse format)
H = sp.load_npz('data/trip_advisor/S_A.npz')     # this matrix also in scipy.sparse format
r = np.load('data/trip_advisor/y_train.npy')     # training labels

# Convert responses of 5 to +1, and all other responses to -1
r = 1 - 2*(r < 5)

### create projective splitting object
projSplit = ps.ProjSplitFit()

### add data composed with linear operator
projSplit.addData(X,r,loss='logistic',linearOp=H,normalize=False)

### first regularizer
mu=0.5
lam=1e-4
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
projSplit.setDualScaling(1e-4)

# Problem is for a limited number of iterations since it is very difficult
projSplit.run(nblocks=10,maxIterations=20000,verbose=True,
              primalTol=1e-2,dualTol=1e-2,keepHistory=True)

# Get results
objVal = projSplit.getObjective()
solVector = projSplit.getSolution()

print(f"Objective value = {objVal}")

try:
    import matplotlib.pyplot as plt
except:
    print("matplotlib appears not to be available, exiting without showing chart")
    sys.exit(0)

history = projSplit.getHistory()
# history[1,:] is time, history[0,:] is the objective value
plt.plot(history[1,:], history[0,:])
plt.xlabel("Time")
plt.ylabel("Objective Value")
plt.show()


