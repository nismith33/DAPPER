# -*- coding: utf-8 -*-

""" 
Run Stommel with perturbations in initial conditions, parameters, forcing 
and climate change. Climate change after t>=Tda is different for each ensemble member.
"""
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from copy import copy 
from scipy.interpolate import interp1d
import os

# Number of ensemble members
N = 100
#DA time 
Tda = 20 * stommel.year
#Time over which climate change will occur
T_warming = 100 * stommel.year
# Timestepping. Timesteps of 1 day, running for 10 year.
tseq = modelling.Chronology(stommel.year, 
                            kko=np.array([],dtype=int), 
                            T=200*stommel.year, BurnIn=0)  # 1 observation/year
#Create model
model = stommel.StommelModel()
#Heat air fluxes 
functions = stommel.default_air_temp(N)

for n, func in enumerate(functions):
    #Same heating rate for all members for t<=Tda, after that differences.
    trend = interp1d(np.array([0.,Tda,T_warming]), 
                 np.array([[0.,1.2,np.random.normal(loc=6,scale=1.5)],
                           [0.,1.2,np.random.normal(loc=3,scale=0.5)]]),
                 fill_value='extrapolate', axis=1)
    functions[n]=stommel.add_functions(functions[n], trend)
    
noised = [stommel.add_noise(func, seed=n*20+1, sig=np.array([2.,2.])) 
          for n,func in enumerate(functions)]
functions = [stommel.merge_functions(Tda, noised[0], func) 
             for func in noised]
model.fluxes.append(stommel.TempAirFlux(functions))
#Salinity air fluxes 
functions = stommel.default_air_salt(N)
noised = [stommel.add_noise(func, seed=n*20+2, sig=np.array([.2,.2])) 
          for n,func in enumerate(functions)]
functions = [stommel.merge_functions(Tda, noised[0], func) 
             for func in noised]
model.fluxes.append(stommel.SaltAirFlux(functions))
#Melt flux 
A = np.array([1,0])/(model.dx[0]*model.dy[0]) #area scaling for flux
    
functions = []
for func in range(N+1):
    #Period over which ice melts.
    T1 = np.random.normal() * 10 * stommel.year + T_warming
    #Same melting rate for t<=Tda (rate0), after that differences (rate1)
    rate0 = -stommel.V_ice / T_warming * A
    rate1 = (-stommel.V_ice - rate0 * Tda) / (T1-Tda) * A
    
    functions.append(interp1d(np.array([0,Tda,T1]), np.array([rate0, rate1, 0*A]), 
                              kind='previous', fill_value='extrapolate', axis=0))
    
#model.fluxes.append(stommel.EPFlux(functions))
# Initial conditions
x0 = model.x0
#Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
#salt diffusion parameter, transport parameter
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
    
model.ens_member=0
stommel.plot_eq(ax, tseq, model, stommel.prob_change(Efor) * 100.)

#Save figure 
fig.savefig(os.path.join(stommel.fig_dir, 'clima_uncertainty.png'),
            format='png', dpi=500)
    
    

    
    
    

