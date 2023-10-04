#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of localisation. 
"""

import numpy as np
from dapper.mods import Liu2002 as modelling
from dapper.tools.randvars import RV_from_function, GaussRV
from dapper.mods.Liu2002 import tools
from matplotlib import pyplot as plt
from dapper.da_methods import EnKF
import dill, os
from copy import copy, deepcopy 
from multiprocessing import Process, Pool
from mpi4py import MPI
from matplotlib import rcParams

rcParams['lines.linewidth'] = 2


#Directory to store data and figures. 
fig_dir = "/home/ivo/dpr_data/synthetic/localisation/"
#Domain length
L = 8e6 #m
#Number of draws
T = 1
#Number of ensemble members 
Nens = 4000
#Degree of freedom 
K = 200
#Observational/background error standard deviation
sigma_obs = 1.0
ncells = 79
Lo = L/ncells
Nobs_dx = Lo
order = 4


#MPI stuff
def create_comm():
    mpi_info = {}
    comm = MPI.COMM_WORLD
    mpi_info['comm'] = comm
    mpi_info['rank'] = comm.Get_rank()
    mpi_info['size'] = comm.Get_size()
    
    return mpi_info

mpi_info = create_comm()

def create_mean_model(rv_obs):
    """ Background mean. """
    
    model = modelling.SpectralModel(L) #m
    model.signal_factory(K=K, rho=modelling.rho_true(L=L/(6*np.pi), b=8*np.pi/L), 
                         sig=10., seed=1000) 
    
    model.dyn_coord_factory('uniform', 2*K+1, 0)
    model.obs_coord_factory('uniform', rv_obs.M)
    
    #Time steps
    tseq = modelling.Chronology(dt=1, dto=1., T=T, BurnIn=0)   
    
    #Initial conditions (irrelevant)
    x0 = np.zeros((2*K+1,))
    X0 = GaussRV(mu=x0, C=0)
    
    #State generator
    Dyn = {'M' : 2*K+1, 'model' : model.step, 'noise' : 0, 'linear' : None}
    
    #Observations
    model.interpolator = model.state2fourier
    Obs = {'M' : rv_obs.M, 'model' : model.obs_factory(model.interpolator, 0), 
           'noise' : rv_obs , 'linear' : None}
    
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.model = model
    
    return HMM

def create_truth_model(mean_model, rv_obs):
    """ Create truth from which observations are taken. """
    model = deepcopy(mean_model.model)
    
    noise = modelling.SpectralModel(L)
    noise.dyn_coords = model.dyn_coords
    noise.obs_coords = model.obs_coords
    for n in range(Nens+1):
        noise.signal_factory(K=K, rho=modelling.rho_true(L=L/(6*np.pi), b=8*np.pi/L),
                             sig=1., seed=2000 + n*100)
    #modelling.rho_back(L/(12*np.pi)),
    
    #Time steps
    tseq = mean_model.tseq  
    
    #Initial conditions (irrelevant)
    x0 = np.zeros((2*K+1,))
    X0 = GaussRV(mu=x0, C=0)
    
    #State generator
    RV = RV_from_function(noise.sample_coords)
    Dyn = {'M' : 2*K+1, 'model' : model.step, 'noise' : RV, 
           'linear' : None}
    
    #Observations
    model.interpolator = model.state2fourier
    Obs = {'M' : rv_obs.M, 'model' : model.obs_factory(model.interpolator, 0), 
           'noise' : rv_obs , 'linear' : None}
    
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.model, HMM.noise = model, noise 
    
    return HMM

def create_dg_model(truth, order, ncells, rv_obs):  
    """ Create projection truth on DG. """
    N = (order+1) * ncells
    
    model = deepcopy(truth.model)
    model.dyn_coord_factory('dg', ncells, order)
    model.apply_legendre()
    
    noise = deepcopy(truth.noise)
    noise.dyn_coord_factory('dg', ncells, order)
    noise.apply_legendre()
    
    #Time steps
    tseq = truth.tseq
    
    #Initial conditions (irrelevant)
    x0 = np.zeros((N,))
    X0 = GaussRV(mu=x0, C=0)
    
    #State generator
    RV = RV_from_function(noise.sample_legendre)
    Dyn = {'M' : N, 'model' : model.step_legendre, 'noise' : RV, 
           'linear' : None}
    
    #Observations
    model.interpolator = model.state2legendre
    Obs = {'M' : rv_obs.M, 'model' : model.obs_factory(model.interpolator, 0),
           'noise' : rv_obs, 'linear' : None}
    
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.model, HMM.noise = model, noise
    
    return HMM

def simulate(order, ncells, Lo):
    """ Run single instance of model."""
    
    Nobs = int( (L / Nobs_dx) / (Lo / Nobs_dx))
    rv_mean = GaussRV(C=np.ones((Nobs,)) * 1e-8, mu=0)
    rv_truth = GaussRV(C=np.ones((Nobs,)) * sigma_obs**2, mu=0 )
    
    HMM = {}
    HMM['mean'] = create_mean_model(rv_mean)
    HMM['truth'] = create_truth_model(HMM['mean'], rv_truth)
    HMM['dg'] = create_dg_model(HMM['truth'], order, ncells, rv_truth)
    
    xx, yy = {}, {}
    for key, HMM1 in HMM.items():
        if hasattr(HMM1,'noise'):
            HMM1.noise.member = Nens
        xx[key], yy[key] = HMM1.simulate()
        
    return HMM, xx, yy

def create_ensemble():
    HMM, xx, yy = simulate(order, ncells, Lo)
    return HMM, xx, yy

if __name__ == "__main__":
    rv = GaussRV(C=np.ones((1,)) * 1e-8, mu=0)
    HMM = {}
    HMM['mean'] = create_mean_model(rv)
    HMM['truth'] = create_truth_model(HMM['mean'], rv)
    HMM['dg'] = create_dg_model(HMM['truth'], order, ncells, rv)
    
    dx = L / ncells
    model = HMM['dg'].noise
    coords = np.linspace(0,L,ncells*(order+1)+1,endpoint=True)
    vals, coefs = [], []
    for n in range(Nens):
        if np.mod(n,50)==0:
            print('n ',n,'/',Nens)
        vals.append(HMM['truth'].noise.functions[n](coords,1))
        coefs.append(HMM['dg'].noise.functions[n].vals2coef(1))
        
    data = {'HMM':HMM,'coords':coords,'vals':np.array(vals),'coefs':np.array(coefs)}
    fname = "dg1_ensemble_{:02d}.pkl".format(int(order))
    with open(os.path.join(fig_dir,fname), 'bw') as stream:
        dill.dump(data, stream)