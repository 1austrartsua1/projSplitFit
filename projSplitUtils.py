# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 11:24:16 2020

Utilities Methods
"""

#-----------------------------------------------------------------------------
#Miscelaneous utilities

from numpy import concatenate
from numpy import array

        
def totalVariation1d(n):
    pass

def dropFirst(n):
    pass

class MyLinearOperator():
    # MyLinearOperator allows us to define "pass through" identity operators
    # for when there really is no operator.
    def __init__(self,matvec,rmatvec):
        self.matvec=matvec
        self.rmatvec=rmatvec
        
def expandOperator(linearOp):
    expandMatVec = lambda x: concatenate((array([x[0]]),linearOp.matvec(x[1:])))
    expandrMatVec = lambda x: concatenate((array([x[0]]),linearOp.rmatvec(x[1:])))
    return expandMatVec,expandrMatVec
    
def createApartition(nrows,n_partitions):
    
    if nrows%n_partitions == 0:
        partition_size = nrows // n_partitions
        partition_list = [range(i*partition_size,(i+1)*partition_size) for i in range(0,n_partitions)]
    else:
        n_with_ceil = nrows%n_partitions
        flr = nrows//n_partitions
        ceil = flr+1
        partition_list = [range(i*ceil,(i+1)*ceil) for i in range(0,n_with_ceil)]            
        endFirstPart = n_with_ceil*ceil
        partition_list.extend([range(endFirstPart + i*flr, endFirstPart + (i+1)*flr) 
                                for i in range(0,n_partitions - n_with_ceil)])

    return partition_list








