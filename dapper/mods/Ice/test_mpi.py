#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 17:31:40 2022

@author: ivo
"""

from mpi4py import MPI as mpi
import numpy as np

comm=mpi.COMM_WORLD
rank = comm.rank

if rank==0:
    A=np.ones((3,10))
    for n in range(0,np.size(A,0)):
        A[n,:]=np.inf
    
    A=np.array(A,dtype=int)
    print('origin ',A[:,0])
else:
    A=None
    
if rank==0:
    A1=-np.ones((2,10))
else:
    A1=-np.ones((2,10))
  
A1=np.array(A1,dtype=int)
comm.Scatter(A,A1,root=0)

print('rank=value',rank,A1)    