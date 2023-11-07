# -*- coding: utf-8 -*-

""" 
As ref_forcing, but now temperature and salinity are assimilated for t<=Tda.
"""
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from copy import copy 
import os
import pickle as pkl
from dapper.mods.Stommel import hadley



def exp_ref_forcing_da(N=100, seed=1000):
    # Timestepping. Timesteps of 1 day, running for 200 year.
    Tda = 20 * stommel.year #time period over which DA takes place. 
    kko = np.arange(1, len(hadley['yy'])+1)
    tseq = modelling.Chronology(stommel.year/12, kko=kko, 
                                T=2*stommel.year, BurnIn=0)  # 1 observation/month
    # Create default Stommel model
    model = stommel.StommelModel()
    #Switch on heat exchange with atmosphere. 
    #Start with default stationary surface temperature and salinity. 
    default_temps = stommel.hadley_air_temp(N)
    default_salts = stommel.hadley_air_salt(N)
    #Add additional periodic forcing 
    temp_forcings, salt_forcings = stommel.budd_forcing(model, model.init_state, 86., 0.0, 
                                                        stommel.Bhat(0.0,0.0), 0.00)
    temp_forcings = [stommel.add_functions(f0,f1) for f0,f1 in zip(default_temps,temp_forcings)]
    salt_forcings = [stommel.add_functions(f0,f1) for f0,f1 in zip(default_salts,salt_forcings)]
    model.fluxes.append(stommel.TempAirFlux(temp_forcings))
    model.fluxes.append(stommel.SaltAirFlux(salt_forcings))
    # Initial conditions
    x0 = model.x0
    #Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
    #salt diffusion parameter, transport parameter
    B = stommel.State().zero()
    B.temp += np.mean(hadley['R'][:2]) #C2
    B.salt += np.mean(hadley['R'][2:]) #ppt2
    B.temp_diff += (0.3*model.init_state.temp_diff)**2
    B.salt_diff += (0.3*model.init_state.salt_diff)**2
    B.gamma += (0.3*model.init_state.gamma)**2
    X0 = modelling.GaussRV(C=B.to_vector(), mu=x0)
    # Dynamisch model. All model error is assumed to be in forcing.
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': 0
           }
    # Observation
    Obs = model.obs_hadley()
    # Create model.
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    # Create DA
    xp = EnKF('Sqrt',N)
    
    return xp, HMM, model 

if __name__=='__main__':
    xp, HMM, model = exp_ref_forcing_da()
    
    #Run
    xx, yy = HMM.simulate()
    yy = hadley['yy']
    Efor, Eana = xp.assimilate(HMM, xx, yy)
    
    #Plot
    fig, ax = stommel.time_figure_with_phase(HMM.tseq)    
    for n in range(np.size(Efor,1)):
        stommel.plot_truth_with_phase(ax, model, Efor[:,n,:], yy)
        
    #Add equilibrium based on unperturbed initial conditions. 
    model.ens_member=0
    stommel.plot_eq(ax, HMM.tseq, model, stommel.array2states(np.mean(Efor,axis=1),HMM.tseq.times))

    

    
    
    

