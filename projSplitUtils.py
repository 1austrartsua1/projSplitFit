# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:24:16 2020

Utilities Methods
"""

#-----------------------------------------------------------------------------
#Miscelaneous utilities

from numpy import concatenate
from numpy import array

from scipy.sparse.linalg import aslinearoperator
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
        
def totalVariation1d(n):
    pass

def dropFirst(n):
    pass

class MyLinearOperator():
    # MyLinearOperator allows us to define "pass through" identity operators
    # for when there really is no operator.
    # I did not use scipy's linear operator because this requires you to know the
    # shape, but if this is being used with a regularizer one might not know
    # the shape yet.
    def __init__(self,matvec,rmatvec,shape=None):
        self.matvec=matvec
        self.rmatvec=rmatvec
        self.shape = shape
        
def expandOperator(linearOp):
    if not issparse(linearOp):
        linearOp = aslinearoperator(linearOp)
        expandMatVec = lambda x: concatenate((array([x[0]]),linearOp.matvec(x[1:])))
        expandrMatVec = lambda x: concatenate((array([x[0]]),linearOp.rmatvec(x[1:])))
    else:
        linearOp = csr_matrix(linearOp)
        expandMatVec = lambda x: concatenate((array([x[0]]), linearOp.dot(x[1:])))
        expandrMatVec = lambda x: concatenate((array([x[0]]), linearOp.T.dot(x[1:])))

    return expandMatVec,expandrMatVec

def MySparseLinearOperator(linearOp):
    linearOp = csr_matrix(linearOp)
    matvec = lambda x : linearOp.dot(x)
    rmatvec = lambda x : linearOp.T.dot(x)
    shape = linearOp.shape
    return MyLinearOperator(matvec,rmatvec,shape)

def createApartition(nrows,n_partitions,sparseMtx):
    
    if nrows%n_partitions == 0:
        partition_size = nrows // n_partitions
        if sparseMtx:
            partition_list = [list(range(i*partition_size,(i+1)*partition_size)) for i in range(0,n_partitions)]
        else:
            partition_list = [range(i*partition_size,(i+1)*partition_size) for i in range(0,n_partitions)]
    else:
        n_with_ceil = nrows%n_partitions
        flr = nrows//n_partitions
        ceil = flr+1
        if sparseMtx:
            partition_list = [list(range(i*ceil,(i+1)*ceil)) for i in range(0,n_with_ceil)]            
        else:
            partition_list = [range(i*ceil,(i+1)*ceil) for i in range(0,n_with_ceil)]            
            
        endFirstPart = n_with_ceil*ceil
        if sparseMtx:
            partition_list.extend([list(range(endFirstPart + i*flr, endFirstPart + (i+1)*flr)) 
                                for i in range(0,n_partitions - n_with_ceil)])
        else:
            partition_list.extend([range(endFirstPart + i*flr, endFirstPart + (i+1)*flr) 
                                for i in range(0,n_partitions - n_with_ceil)])
    

    return partition_list








