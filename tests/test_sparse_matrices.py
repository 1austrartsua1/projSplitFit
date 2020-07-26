getNewOptVals = False

import sys
sys.path.append('../')

import projSplit as ps
import regularizers

import pytest
import numpy as np
import scipy.sparse as sp

import pickle

if getNewOptVals:
    from utils import runCVX_lasso
    import cvxpy as cvx
    cache = {}
else:
    with open('results/cache_sparse','rb') as file:
        cache = pickle.load(file)


sparse_types = [sp.csr_matrix,sp.csc_matrix,sp.bsr_matrix,sp.coo_matrix,sp.dia_matrix,sp.lil_matrix]
ToDo = []
for mat in sparse_types:
    for INT in False,True:
        for Norm in False,True:
            for reg in False,True:
                ToDo.append((mat,INT,Norm,reg))

@pytest.mark.parametrize("sparse_type,INT,Norm,reg",ToDo)
def test_ls_with_lin_op(sparse_type,INT,Norm,reg):
    if Norm:
        gamma = 1e-3
    else:
        gamma = 1e0
    psObj = ps.ProjSplitFit(gamma)
    m = 25
    d = 15
    d2 = 11
    p = 7
    nu = 1e-4
    if getNewOptVals:
        A = cache.get('Als')
        y = cache.get('yls')
        H = cache.get('Hls')
        Hreg = cache.get('Hregls')
        if A is None:
            A = np.random.normal(0,1,[m,d])
            y = np.random.normal(0,1,m)
            H = np.random.normal(0,1,[d,d2])
            Hreg = np.random.normal(0,1,[p,d2])
            cache['Als']=A
            cache['yls']=y
            cache['Hls'] = H
            cache['Hregls'] = Hreg
    else:
        A = cache.get('Als')
        y = cache.get('yls')
        H = cache.get('Hls')
        Hreg = cache.get('Hregls')

    H = sparse_type(H)
    Hreg = sparse_type(Hreg)
    psObj.addData(A,y,2,linearOp=H,intercept=INT,normalize=Norm)
    if reg:
        psObj.addRegularizer(regularizers.L1(scaling=nu),linearOp=Hreg)

    psObj.run(nblocks=10,maxIterations=1000)
    psOpt = psObj.getObjective()


    if getNewOptVals:
        opt = cache.get(('lsOpt',INT,Norm,reg))
        if opt is None:
            xcvx = cvx.Variable(d2+1)
            H = H.toarray()
            if Norm:
                scaling = np.linalg.norm(A, axis=0)
                scaling += 1.0 * (scaling < 1e-10)
                A = A / scaling
            col2Add = int(INT) * np.ones((m, 1))
            A = np.concatenate((col2Add, A), axis=1)


            zeros2add = np.zeros((d,1))
            H = np.concatenate((zeros2add, H), axis=1)
            zeros2add = np.zeros((1,d2+1))
            H = np.concatenate((zeros2add, H), axis=0)
            H[0,0] = 1

            f = (1/(2*m))*cvx.sum_squares(A@H@xcvx - y)
            if reg:
                Hreg = Hreg.toarray()
                zeros2add = np.zeros((p, 1))
                Hreg = np.concatenate((zeros2add, Hreg), axis=1)

                f+= nu*cvx.norm(Hreg @ xcvx,1)

            prob = cvx.Problem(cvx.Minimize(f))
            prob.solve(verbose=False)
            opt = prob.value
            cache[('lsOpt',INT,Norm)] = opt
    else:
        opt = cache.get(('lsOpt',INT,Norm))

    print(f"psOpt = {psOpt}")
    print(f"cvxOpt = {opt}")
    assert psOpt - opt < 1e-2

def test_wrong_obs_matrix():
    psObj = ps.ProjSplitFit()
    obs = [[1,2,3],[4,5,6]]
    y = [1,1]
    with pytest.raises(Exception):
        psObj.addData(obs,y,2)
    obs = "hello world"
    with pytest.raises(Exception):
        psObj.addData(obs,y,2)

toDo = [(sp.csr_matrix),(sp.csc_matrix),(sp.bsr_matrix),(sp.coo_matrix),
        (sp.dia_matrix),(sp.lil_matrix),(sp.dok_matrix)]
@pytest.mark.parametrize("sparse_type",toDo)
def test_add_data_works_with_sparse_obs(sparse_type):
    psObj = ps.ProjSplitFit()
    obs = [[1, 2, 3], [4, 5, 6]]
    y = [1, 1]
    obs = sparse_type(obs)
    psObj.addData(obs,y,2)
toDo = [(sp.csr_matrix),(sp.csc_matrix),(sp.bsr_matrix),(sp.coo_matrix),
        (sp.dia_matrix),(sp.lil_matrix),(sp.dok_matrix)]
@pytest.mark.parametrize("sparse_type",toDo)
def test_add_data_works_with_sparse_linearOp(sparse_type):
    psObj = ps.ProjSplitFit()
    obs = np.array([[1, 2, 3], [4, 5, 6]])
    y = [1,1]
    H = sparse_type([[1,1,7,8],[7,9,7,8],[4,4,3,4]])
    psObj.addData(obs,y,2,linearOp=H)

toDo = [(sp.csr_matrix),(sp.csc_matrix),(sp.bsr_matrix),(sp.coo_matrix),
        (sp.dia_matrix),(sp.lil_matrix),(sp.dok_matrix)]
@pytest.mark.parametrize("sparse_type",toDo)
def test_add_regularizer_works_with_sparse_linearOp(sparse_type):
    psObj = ps.ProjSplitFit()
    regObj = regularizers.L1()
    H = sparse_type([[1, 1, 7, 8], [7, 9, 7, 8], [4, 4, 3, 4]])
    psObj.addRegularizer(regObj,linearOp=H)

toDo = [(sp.csr_matrix),(sp.csc_matrix),(sp.bsr_matrix),(sp.coo_matrix),
        (sp.dia_matrix),(sp.lil_matrix),(sp.dok_matrix)]
@pytest.mark.parametrize("sparse_type",toDo)
def test_all_adds_together(sparse_type):
    psObj = ps.ProjSplitFit()
    obs = sparse_type([[1, 2, 3], [4, 5, 6]])
    y = [1, 1]
    H = sparse_type([[1, 1, 7, 8], [7, 9, 7, 8], [4, 4, 3, 4]])
    psObj.addData(obs, y, 2, linearOp=H)
    regObj = regularizers.L1()
    G = sparse_type([[1,1,1,11], [7,7,11,42]])
    psObj.addRegularizer(regObj, linearOp=G)


def test_writeResult():
    if getNewOptVals:
        with open('results/cache_sparse','wb') as file:
            pickle.dump(cache,file)













