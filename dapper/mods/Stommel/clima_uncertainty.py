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
# Add melt
default_ep_flux = model.ep_flux[0]
model.ep_flux = [stommel.add_melt(default_ep_flux, model, sig_T=10) for n in range(N)]
default_temp_air = model.temp_air[0]
model.temp_air = [stommel.add_warming(default_temp_air, sigs=[.5, 1.]) for n in range(N)]
# Initial conditions
x0 = model.x0
#Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
#salt diffusion parameter, transport parameter
B = np.zeros_like(x0)
X0 = modelling.GaussRV(C=B, mu=x0)
# Dynamisch model. All model error is assumed to be in forcing.
Dyn = {'M': model.M,
       'model': model.step,
       'noise': 0
       }
# Observational variances
R = np.array([1.0, 1.0, 0.1, 0.1])**2  # C2,C2,ppt2,ppt2
# Observation
Obs = model.obs_ocean()
Obs['noise'] = modelling.GaussRV(C=R, mu=np.zeros_like(R))
# Create model.
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
# Create DA
xp = EnKF('Sqrt',N)

#Run
xx, yy = HMM.simulate()
Efor, Eana = xp.assimilate(HMM, xx, yy)

#Count SA
drho = np.diff(model.eos(Efor[-1,:,0:2], Efor[-1,:,2:4]), axis=1)
SA = np.sum(drho>0.)
drho = np.diff(model.eos(Efor[0,:,0:2], Efor[0,:,2:4]), axis=1)
SA0 = np.sum(drho>0.)

#Plot
fig, ax = stommel.time_figure(tseq)    
for n in range(np.size(Efor,1)):
    stommel.plot_truth(ax, Efor[:,n,:], yy)
stommel.plot_eq(ax, tseq, stommel.StommelModel(), stommel.prob_change(Efor) * 100.)

#Save figure 
fig.savefig(os.path.join(stommel.fig_dir, 'clima_uncertainty.png'),
            format='png', dpi=500)
    
    

    
    
    

