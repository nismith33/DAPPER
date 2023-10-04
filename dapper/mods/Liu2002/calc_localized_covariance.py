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
from dapper.da_methods import LETKF, EnKF, SL_EAKF, E3DVar
import dapper.tools.localization as loc
import matplotlib.pyplot as plt
import os
import dill
from datetime import timedelta, datetime
from scipy.stats import randint

# Directory
fig_dir = os.path.join(exp.fig_dir, 'localized_covariance_new')
io = exp.TreeIO(fig_dir, try_loading=True)

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


def exp_spectral_Nobs(T=1, Nens=10000, sig_obs=1.0):
    tseq = Chronology(dt=1, dto=1, T=T)
    # Background
    tree = exp.MeanTreeNode(None, tseq, io=io, K=395, sig=1.)

    # Truth
    for slope in np.array([-1.]): #, -1.-1.5, -.5):  # -4.5
        tree.add_truth(slope=slope, name='slope{:02d}'.format(int(-10*slope)))
    # Observation
    for sig in np.array([1.]):
        for Nobs in np.arange(1, 2, 2):  # 8
            tree.add_obs(Nobs, sig, obs_type='uniform',
                         name='Sigo{:.1f}_No{:03d}'.format(sig_obs, Nobs))
    
    # DG Model
    for order in np.array([4]):  # 8
        # Linear model
        d = exp.Ncells*(order+1)
        def localization(model):
            batcher = loc.LegendreBatcher(model.dyn_coords, 0)
            taperer = loc.OptimalTaperer(0, period=exp.L)
            coorder = loc.FunctionCoorder(model.obs_coords)
            return loc.Localizer(batcher, taperer, coorder)
        
        tree.add_model('lin', d, name="lin{:03d}".format(d),    
                       localization=localization)
    
        #DG model
        def localization(model):
            batcher = loc.LegendreBatcher(model.dyn_coords, order)
            taperer = loc.OptimalTaperer(order, period=exp.L)
            coorder = loc.FunctionCoorder(model.obs_coords)
            return loc.Localizer(batcher, taperer, coorder)

        tree.add_model('dg', exp.Ncells, order=order,
                       name='DG{:02d}'.format(order),
                       localization=localization)

    # Experiment
    tree.add_xp(E3DVar, Nens, loc_rad=exp.L/exp.Ncells)

    return tree


def run():
    t_start = datetime.now()

    tree = exp_spectral_Nobs()
    comm = exp.SingleComm()
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
