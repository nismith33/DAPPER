# -*- coding: utf-8 -*-

""" Test run with Stommel model. """
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from copy import copy 
import os


# Number of ensemble members
N = 100
# Timestepping. Timesteps of 1 day, running for 10 year.
kko = np.array([])
tseq = modelling.Chronology(stommel.year, kko=kko, 
                            T=200*stommel.year, BurnIn=0)  # 1 observation/year
# Create default Stommel model
model = stommel.StommelModel()
default_init = model.init_state
# Initial conditions
x0 = model.init_state.to_vector()
#Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
#salt diffusion parameter, transport parameter
B = np.array([.5, .5, 0.05, 0.05, 0.0, 0.0, 0.0])**2  
X0 = modelling.GaussRV(C=B, mu=x0)
# Dynamisch model. All model error is assumed to be in forcing.
Dyn = {'M': model.M,
       'model': model.step,
       'noise': 0
       }
# Observational variances
R = np.array([0.5, 0.5, 0.1, 0.1])**2  # C2,C2,ppt2,ppt2
# Observation
Obs = model.obs_ocean()
Obs['noise'] = modelling.GaussRV(C=R, mu=np.zeros_like(R))
# Create model.
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
 
#Plot
fig, ax = stommel.time_figure(tseq)    
for n in range(N):
    xx,yy=HMM.simulate()
    stommel.plot_truth(ax, xx, yy)
stommel.plot_eq(ax, tseq, stommel.StommelModel())
    

#Save figure 
fig.savefig(os.path.join(stommel.fig_dir,'ocean_noise.png'),format='png',dpi=500)

print('End of example')
    
    
    

