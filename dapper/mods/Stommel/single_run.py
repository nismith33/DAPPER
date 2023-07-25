""" 
Run Stommel model without perturbations for a single run that 
starts off from non-equilibrium conditions and should convergence to 
an equilibrium.
"""
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
import os

# Number of ensemble members
N = 0
# Timestepping. Timesteps of 1 day, running for 400 year.
tseq = modelling.Chronology(stommel.year*10, kko=np.array([],dtype=int), 
                            T=400*stommel.year, BurnIn=0)  # 1 observation/year
# Create default Stommel model
model = stommel.StommelModel()
#Switch on heat exchange with atmosphere. Assume stationary air temperatures. 
model.fluxes.append(stommel.TempAirFlux(stommel.default_air_temp(N)))
#Switch on salinity exchange with atmosphere. Assume stationary air salinity. 
model.fluxes.append(stommel.SaltAirFlux(stommel.default_air_salt(N)))
#Add additional periodic forcing 
Omega = 2 * np.pi / (100 * stommel.year) #angular period 
temp_forcings = [lambda time : 1e-5 * np.array([-.5,.5]) * np.sin(Omega * time)]
model.fluxes.append(stommel.FunctionTempFlux(temp_forcings))
salt_forcings = [lambda time : 1e-6 * np.array([-.5,.5]) * np.sin(Omega * time)]
model.fluxes.append(stommel.FunctionSaltFlux(salt_forcings))
# Adjust default initial conditions.
x0 = model.init_state
x0.temp += np.array([[-1.,1.]])
x0.salt += np.array([[0.,.4]]) #Move initial state away from equilibrium
#Set initial conditions. 
x0 = x0.to_vector()
X0 = modelling.GaussRV(C=np.zeros_like(x0), mu=x0)
#Print model information. 
stommel.display(model)

# Dynamisch model. All model error is assumed to be in forcing.
Dyn = {'M': model.M,
       'model': model.step,
       'noise': 0.0
       }
# Observation
Obs = model.obs_ocean()
# Create model.
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
# Generate truth
xx, yy = HMM.simulate()

#Create figure
fig,ax=stommel.time_figure(tseq)
stommel.plot_truth(ax, xx, yy)
stommel.plot_eq(ax, tseq, model)

#Save figure 
#fig.savefig(os.path.join(stommel.fig_dir, 'single_run.png'),
#            format='png', dpi=500)

print('End of example')
