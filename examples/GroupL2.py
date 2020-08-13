import numpy as np
import sys
sys.path.append('../')
import projSplitFit as ps
import regularizers as rg

m = 500
d = 200
groupSize = 10

np.random.seed(1)
A = np.random.normal(0,1,[m,d])
trueCoefs = 0.01*np.array(range(d))
r = A @ trueCoefs + np.random.normal(0,1,m)

ngroups = d // groupSize
groups = [range(i*groupSize,(i+1)*groupSize) for i in range(ngroups)]
print(groups)

projSplit = ps.ProjSplitFit()
projSplit.addData(A,r,loss=2,intercept=False)
projSplit.addRegularizer(rg.L1(scaling=1.0))
projSplit.addRegularizer(rg.groupL2(d,groups,scaling=1.0))

projSplit.run(verbose=True)

print(f"Objective value = {projSplit.getObjective()}")
print()

try:
    import matplotlib.pyplot as plt
    sol = projSplit.getSolution()
    plt.style.use('ggplot')
    plt.bar(range(d), sol)
    plt.show()
except:
    print("matplotlib appears not to be available, exiting without showing chart")