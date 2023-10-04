#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculations for paper. 
Experiment generating massive ensemble used to study localization and 
covariance errors. 
"""

import numpy as np
from dapper.tools.chronos import Chronology
from dapper.mods.Liu2002 import exp_setup as exp
from dapper.da_methods import E3DVar
import dapper.tools.localization as loc
import matplotlib.pyplot as plt
import os
from datetime import datetime
from scipy.stats import randint

# Directory
fig_dir = os.path.join(exp.fig_dir,'rmse_Nobs_order_noloc_smoothing')
io = exp.TreeIO(fig_dir, try_loading=False)

# Iterator that creates subensembles.
class SubEnsemble:

    def __init__(self, E, N, size):
        self.E = E
        self.N = N
        self.size = size
        self.seed = 5000

    def __iter__(self):
        self.N0 = 0
        return self

    def __next__(self):
        if self.size <= 0:
            raise StopIteration
        else:
            np.random.seed(self.seed)
            ind = randint.rvs(0, np.size(self.E, 0), size=(self.N,))
            self.size -= 1
            self.seed += 100
            return self.E[ind]

def exp_spectral_Nobs(T=50, Nens=16, sig_obs=1.0):
    tseq = Chronology(dt=1, dto=1, T=T)
    # Background for ncells=79 and poly order 10
    tree = exp.MeanTreeNode(None, tseq, io=io, K=829, sig=10.0)

    # Truth
    for slope in np.array([-1.5,-2.,-2.5,-3.]): 
        tree.add_truth(slope=slope, name='slope{:02d}'.format(int(-10*slope)))
    
    # Observation
    for sig in np.array([0.5,1.0,1.5]):
        Nobs_list  = [int(exp.Ncells * o) for o in [.1,.5,1,1.5,3,5,9,13]]
        for Nobs in Nobs_list:
            Lo = float(exp.L/Nobs)
            tree.add_obs(Nobs, sig, obs_type='uniform', Lo=Lo,
                          name='Sigo{:.1f}_No{:03d}'.format(sig, Nobs))
    
    # Linear model
    def localization(model):
        batcher = loc.LegendreBatcher(model.dyn_coords, 0)
        taperer = loc.OptimalTaperer(0, period=exp.L)
        coorder = loc.FunctionCoorder(model.obs_coords)
        return loc.Localizer(batcher, taperer, coorder)
    
    tree.add_model('lin', exp.Ncells, name="Nodal")
    
    # DG Model
    for order in np.array([0,1,2,4,6,8,10]):  
        
        #DG model
        def localization(model):
            order = model.order
            batcher = loc.LegendreBatcher(model.dyn_coords, order)
            taperer = loc.OptimalTaperer(order, period=exp.L)
            coorder = loc.FunctionCoorder(model.obs_coords)
            return loc.Localizer(batcher, taperer, coorder)

        tree.add_model('dg', exp.Ncells, order=order,
                        name='DG{:02d}'.format(order))

    # Experiment
    tree.add_xp(E3DVar, Nens, loc_rad=exp.L/exp.Ncells)

    return tree

def run():
    t_start = datetime.now()

    tree = exp_spectral_Nobs()
    comm = exp.MpiComm()
    tree.execute(comm)

    t_end = datetime.now()
    if comm.rank == 0:
        print("Clock time", (t_end-t_start))

    return tree

if __name__ == '__main__':
    tree = run()

    truths = exp.collect(tree, 1)
    models = exp.collect(tree, 3)
    xps = exp.collect(tree, 4)
    
    #plot_DA(truths, models, xps)
    
    

