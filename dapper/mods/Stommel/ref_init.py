# -*- coding: utf-8 -*-

""" 
Run Stommel model using perturbed initial conditions only.
"""
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from copy import copy 
import os



# Number of ensemble members
N = 100
# Timestepping. Timesteps of 1 day, running for 200 year.
kko = np.array([])
tseq = modelling.Chronology(stommel.year, kko=kko, 
                            T=200*stommel.year, BurnIn=0)  # 1 observation/year
# Create default Stommel model
model = stommel.StommelModel()
#Use default initial conditions.
default_init = model.init_state

#Switch on heat exchange with atmosphere. Assume stationary air temperatures. 
model.fluxes.append(stommel.TempAirFlux(stommel.default_air_temp(N)))
#Switch on salinity exchange with atmosphere. Assume stationary air salinity. 
model.fluxes.append(stommel.SaltAirFlux(stommel.default_air_salt(N)))
#Add additional periodic forcing 
temp_forcings, salt_forcings = stommel.budd_forcing(model, default_init, 100., .1, .1)
model.fluxes.append(stommel.FunctionTempFlux(temp_forcings))
model.fluxes.append(stommel.FunctionSaltFlux(salt_forcings))

# Initial conditions
x0 = model.init_state.to_vector()
#Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
#salt diffusion parameter, transport parameter
B = stommel.State().zero()
B.temp += 0.5**2 #C2
B.salt += 0.05**2 #ppt2
X0 = modelling.GaussRV(C=B.to_vector(), mu=x0)
# Dynamisch model. All model error is assumed to be in forcing.
Dyn = {'M': model.M,
       'model': model.step,
       'noise': 0
       }
#Default observations. 
Obs = model.obs_ocean()
# Create model.
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
 
#Plot
fig, ax = stommel.time_figure(tseq)    
for n in range(N):
    xx,yy=HMM.simulate()
    stommel.plot_truth(ax, xx, yy)
    
#Plot equilibria values based on unperturbed initial conditions.
model.ens_member=0
stommel.plot_eq(ax, tseq, model)
    
#Save figure 
fig.savefig(os.path.join(stommel.fig_dir,'ocean_noise.png'),format='png',dpi=500)

print('End of example')
    
    
    

