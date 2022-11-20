# -*- coding: utf-8 -*-

""" Stommel model with perturbations in initial conditions, forcing and 
unperturbed climate change.
"""
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy 
import os

def exp_clima_forcing(N=100, seed=1000):
    #Time period over which climate change takes place
    T_warming=100*stommel.year
    #Data assimilation period in this or other experiments.
    Tda = 20 * stommel.year
    # Timestepping. Timesteps of 1 day, running for 200 year.
    tseq = modelling.Chronology(stommel.year, 
                                kko=np.array([],dtype=int), 
                                T=200*stommel.year, BurnIn=0)  # 1 observation/year
    #Create model
    model = stommel.StommelModel()
    #Heat air fluxes 
    #Start with default stationary atm. temperature. 
    functions = stommel.default_air_temp(N)
    #Add linear warming with 6C/T_warming over the pole and 3C/T_warming over the equatior. 
    trend = interp1d(np.array([0.,T_warming]), np.array([[0.,6.],[0.,3.]]), 
                     fill_value='extrapolate', axis=1)
    trended = [stommel.add_functions(func, trend) for func in functions]
    #Add random temperature perturbations with std dev. of 2C
    noised = [stommel.add_noise(func, seed=seed+n*20+1, sig=np.array([2.,2.])) 
              for n,func in enumerate(trended)]
    #For time<Tda all ensemble member n uses noised[0] after that noised[n]
    functions = [stommel.merge_functions(Tda, noised[0], func) 
                 for func in noised]
    #Activate surface heat flux. 
    model.fluxes.append(stommel.TempAirFlux(functions))
    #Salinity air fluxes 
    functions = stommel.default_air_salt(N)
    #Add random salinity perturbations with std dev. of 0.2ppt
    noised = [stommel.add_noise(func, seed=seed+n*20+2, sig=np.array([.2,.2])) 
              for n,func in enumerate(functions)]
    #For time<Tda all ensemble member n uses noised[0] after that noised[n]
    functions = [stommel.merge_functions(Tda, noised[0], func) 
                 for func in noised]
    #Activate surface salinity flux. 
    model.fluxes.append(stommel.SaltAirFlux(functions))
    #Melt flux 
    melt_rate = -stommel.V_ice * np.array([1.0/(model.dx[0,0]*model.dy[0,0]), 0.0]) / T_warming #ms-1
    #Default evaporation-percipitation flux (=0)
    functions = stommel.default_air_ep(N)
    #Add effect Greenland melt with annual rate melt_rate
    functions = [stommel.merge_functions(T_warming, lambda t:func(t)+melt_rate, func)
                 for func in functions]
    #Activate EP flux. 
    model.fluxes.append(stommel.EPFlux(functions))
    #Default nitial conditions
    x0 = model.x0
    #Variance in initial conditions and parameters.
    B = stommel.State().zero()
    B.temp += 0.5**2 #C2
    B.salt += 0.05**2 #ppt2
    B.temp_diff += (0.3*model.init_state.temp_diff)**2
    B.salt_diff += (0.3*model.init_state.salt_diff)**2
    B.gamma += (0.3*model.init_state.gamma)**2
    X0 = modelling.GaussRV(C=B.to_vector(), mu=x0)
    # Dynamisch model. All model error is assumed to be in forcing.
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': 0
           }
    # Default observations. 
    Obs = model.obs_ocean()
    # Create model.
    HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    # Create DA
    xp = EnKF('Sqrt',N)
    
    return xp, HMM, model


if __name__=='__main__':
    xp, HMM, model = exp_clima_forcing()
    
    #Run
    xx, yy = HMM.simulate()
    Efor, Eana = xp.assimilate(HMM, xx, yy)
    
    #Plot
    fig, ax = stommel.time_figure(HMM.tseq)    
    for n in range(np.size(Efor,1)):
        stommel.plot_truth(ax, Efor[:,n,:], yy)
        
    #Add equilibrium based on unperturbed initial conditions. 
    model.ens_member=0
    stommel.plot_eq(ax, HMM.tseq, model, stommel.prob_change(Efor) * 100.)
    
    #Save figure 
    fig.savefig(os.path.join(stommel.fig_dir, 'clima_forcing.png'),
                format='png', dpi=500)
    
    

    
    
    

