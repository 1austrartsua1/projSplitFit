
getNewOptVals = False      


import sys
sys.path.append('../')
import projSplit as ps 
from regularizers import L2
from regularizers import L2sq
import numpy as np 
import pickle 
import pytest 
import cvxpy as cvx

if getNewOptVals:        
    cache = {}
else:    
    with open('results/cache_BuiltInRegs','rb') as file:
        cache = pickle.load(file)


L2sqReg = L2sq(scaling=5.5)
L2reg = L2(scaling = 3.7)
m = 10
d = 50

xcvx = cvx.Variable(d)
fL2sq = 5.5*0.5*cvx.norm(xcvx,2)**2
fL2 = 3.7*cvx.norm(xcvx,2)

@pytest.mark.parametrize("builtInReg,cvxf,testNum",[(L2sqReg,fL2sq,0),(L2reg,fL2,1)])
def test_a_reg(builtInReg,cvxf,testNum):
    projSplit = ps.ProjSplitFit()
        
    if getNewOptVals:
        A = cache.get('Areg')
        y = cache.get('yreg')
        if A is None:
            A = np.random.normal(0,1,[m,d])
            y = np.random.normal(0,1,m)
            cache['Areg']=A
            cache['yreg']=y
    else:
        A = cache['Areg']
        y = cache['yreg']
        
    projSplit.addData(A,y,intercept=False,normalize=False,loss=2)
    projSplit.addRegularizer(builtInReg)
    projSplit.run(nblocks=5)
    psval = projSplit.getObjective()
    
    if getNewOptVals:
        opt = cache.get(('optreg',testNum))
        if opt is None:        
            cvxf += (1/(2*m))*cvx.sum_squares(A@xcvx - y) 
            prob = cvx.Problem(cvx.Minimize(cvxf))
            prob.solve(verbose=False)
            opt = prob.value
            cache[('optreg',testNum)]=opt
    else:
        opt=cache[('optreg',testNum)]
        
    print(f"psval = {psval}")
    print(f"CVX opt = {opt}")
    
    assert psval - opt <1e-5
    

    
def test_writeResult():
    if getNewOptVals:
        with open('results/cache_BuiltInRegs','wb') as file:
            pickle.dump(cache,file)
    
    
    
    
        
    

    
        
    
