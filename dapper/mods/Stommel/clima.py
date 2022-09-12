# -*- coding: utf-8 -*-

""" 
Run Stommel model without perturbation but with global warming and ice melt.
"""
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from copy import copy 
import os

# Number of ensemble members
N = 0
#Time over which DA will occur in this or other experiment
Tda = 20 * stommel.year
#Time over which climate change will take place
T_warming = 100*stommel.year
# Timestepping. Timesteps of 1 day, running for 10 year.
kko = np.array([])
tseq = modelling.Chronology(stommel.year, 
                            kko=np.array([],dtype=int), T=200*stommel.year, 
                            BurnIn=0)  # 1 observation/year
# Create default Stommel model
model = stommel.StommelModel()
#Heat air fluxes 
functions = stommel.default_air_temp(N)
trend = interp1d(np.array([0.,T_warming]), np.array([[0.,6.],[0.,3.]]), 
                 fill_value='extrapolate', axis=1)
trended = [stommel.add_functions(func, trend) for func in functions]
model.fluxes.append(stommel.TempAirFlux(trended))
#Salinity air fluxes 
functions = stommel.default_air_salt(N)
model.fluxes.append(stommel.SaltAirFlux(functions))
#Melt flux 
melt_rate = -stommel.V_ice * np.array([1.0/(model.dx[0,0]*model.dy[0,0]), 0.0]) / T_warming #ms-1
functions = stommel.default_air_ep(N)
functions = [stommel.merge_functions(T_warming, lambda t:func(t)+melt_rate, func)
             for func in functions]
model.fluxes.append(stommel.EPFlux(functions))
# Initial conditions
x0 = model.x0
#Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
#salt diffusion parameter, transport parameter
B = stommel.State().zero()
X0 = modelling.GaussRV(C=B.to_vector(), mu=x0)
# Dynamisch model. All model error is assumed to be in forcing.
Dyn = {'M': model.M,
       'model': model.step,
       'noise': 0
       }
# Observational variances
R = np.array([1.0, 1.0, 0.1, 0.1])**2  # C2,C2,ppt2,ppt2
# Observation
Obs = model.obs_ocean()
# Create model.
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
# Create DA
xp = EnKF('Sqrt',N)

#Run
xx, yy = HMM.simulate()

#Plot
fig, ax = stommel.time_figure(tseq)    
stommel.plot_truth(ax, xx, yy)

model.ens_member=0
stommel.plot_eq(ax, tseq, model)

#Save figure 
fig.savefig(os.path.join(stommel.fig_dir, 'clima.png'),
            format='png', dpi=500)
    
    

    
    
    

