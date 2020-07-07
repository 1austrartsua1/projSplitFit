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
    '''
    Because scipy's linear operator requires passing in the shape
    of a linear operator, we had to create my own "dumb" linear operator class.
    This is used because a regularizer may be created with no composed linear operator,
    and to deal with this we create an identity linear operator which just passes
    through the input to output. But we don't necessarily know the dimensions at
    that point, because addData() may not yet have been called. 
    '''
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








