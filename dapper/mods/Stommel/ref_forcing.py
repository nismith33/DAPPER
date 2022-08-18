# -*- coding: utf-8 -*-

""" Test run with Stommel model. """
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from copy import copy 
import os

def exp_ref_forcing(N=100,seed=1000):
    # Timestepping. Timesteps of 1 day, running for 10 year.
    tseq = modelling.Chronology(stommel.year, kko=np.array([], dtype=int), 
                                T=200*stommel.year, BurnIn=0)  # 1 observation/year
    # Create default Stommel model
    model = stommel.StommelModel()
    # Add initial perturbation to surface temperature
    temp_air_default = model.temp_air[0]
    sigs = np.array([2.0, 2.0]) 
    model.temp_air = [stommel.add_init_noise(temp_air_default, 10*n+100+seed, sigs) for n in range(N+1)]
    # Add initial perturbation to surface salinity
    sigs = np.array([0.2, 0.2]) 
    salt_air_default = model.salt_air[0]
    model.salt_air = [stommel.add_init_noise(salt_air_default, 10*n+150+seed, sigs) for n in range(N+1)]
    # Initial conditions
    x0 = model.x0
    #Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
    #salt diffusion parameter, transport parameter
    B = np.append(np.array([0.5, 0.5, 0.05, 0.05]), 0.3*x0[4:])**2 
    X0 = modelling.GaussRV(C=B, mu=x0)
    # Dynamisch model. All model error is assumed to be in forcing.
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': 0
           }
    # Observational variances
    R = B[:4]  # C2,C2,ppt2,ppt2
    # Observation
    Obs = model.obs_ocean()
    Obs['noise'] = modelling.GaussRV(C=R, mu=np.zeros_like(R))
    # Create model.
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    # Create DA
    xp = EnKF('Sqrt',N)
    
    return xp, HMM, model

if __name__=='__main__':
    xp, HMM, model = exp_forcing_noise()
    
    #Run
    xx, yy = HMM.simulate()
    Efor, Eana = xp.assimilate(HMM, xx, yy)
    
    #Plot
    fig, ax = stommel.time_figure(HMM.tseq)    
    for n in range(np.size(Efor,1)):
        stommel.plot_truth(ax, Efor[:,n,:], yy)
    stommel.plot_eq(ax, HMM.tseq, stommel.StommelModel(), stommel.prob_change(Efor) * 100.)
    
    #Save figure 
    fig.savefig(os.path.join(stommel.fig_dir,'forcing_noise.png'),
                format='png',dpi=500)
    

    
    
    

