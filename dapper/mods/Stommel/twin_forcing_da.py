# -*- coding: utf-8 -*-

""" 
As ref_forcing, but now temperature and salinity are assimilated.
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
import shutil

fig_dir = stommel.fig_dir
shutil.copy(__file__, fig_dir) 

def exp_ref_forcing_da(N=100, seed=1000):
    np.random.seed(seed)
    # Timestepping. Timesteps of 1 day, running for 200 year.
    kko = np.arange(1, len(hadley['yy'][1:]))
    tseq = modelling.Chronology(stommel.year/12, kko=kko,
                                T=400*stommel.year)  # 1 observation/month
    # Create default Stommel model
    model = stommel.StommelModel()
    # Switch on heat exchange with atmosphere.
    # Start with default stationary surface temperature and salinity.
    default_temps = stommel.hadley_air_temp(N)
    default_salts = stommel.hadley_air_salt(N)
    ## Add additional periodic forcing
    #temp_forcings, salt_forcings = stommel.budd_forcing(model, model.init_state, 86., 0.0,
    #                                                    stommel.Bhat(0.0, 0.0), 0.00)
    #temp_forcings = [stommel.add_functions(
    #    f0, f1) for f0, f1 in zip(default_temps, temp_forcings)]
    #salt_forcings = [stommel.add_functions(
    #    f0, f1) for f0, f1 in zip(default_salts, salt_forcings)]
    
    
    def clima_T(t):
        """ Warming surface due climate change in K. """
        T0 = np.max(tseq.tto)
        if t<T0:
            #No warming over DA period
            return np.array([0,0])
        else:
            #.06/.03 C warming per year over pole/equator
            return np.array([.06,.03]) * (t-T0) / stommel.year
        
    def clima_S(t):
        """ Freshening due to melt Greenland. """
        T0 = np.max(tseq.tto)
        volume = 2.9e15 #m3
        melt_period = 10000 * stommel.year 
        A = model.dx[0,:] * model.dy[0,:] #area stommel boxes 

        if t<T0:
            return np.array([0,0])
        else:
            return np.array([-volume/melt_period,0]) / A
    
    temp_forcings = [stommel.add_functions(f, clima_T) for f in default_temps]
    model.fluxes.append(stommel.TempAirFlux(temp_forcings))
    model.fluxes.append(stommel.SaltAirFlux(default_salts))
    
    melt_rates = [clima_S for _ in range(N)]
    model.fluxes.append(stommel.EPFlux(melt_rates))
    
    # Initial conditions
    x0 = model.x0
    # Variance Ocean temp[2], ocean salinity[2], temp diffusion parameter,
    # salt diffusion parameter, transport parameter
    B = stommel.State().zero()
    B.temp += hadley['R'][:2]  # C2
    B.salt += hadley['R'][2:]  # ppt2
    B.temp_diff += np.log(1.3)**2  # (0.0*model.init_state.temp_diff)**2
    B.salt_diff += np.log(1.3)**2  # (0.0*model.init_state.salt_diff)**2
    B.gamma += np.log(1.3)**2  # (0.0*model.init_state.gamma)**

    # Transform modus value in x0 to mean value.
    x0[4] += B.temp_diff
    x0[5] += B.salt_diff
    x0[6] += B.gamma

    #Create sampler for initial conditions. 
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
    xp = EnKF('Sqrt', N, infl=1.0)
    return xp, HMM, model,temp_forcings,salt_forcings

if __name__ == '__main__':
    xp, HMM, model, temp_forcings, salt_forcings = exp_ref_forcing_da()

    # Run
    xx, yy = HMM.simulate()
    #yy = hadley['yy'][HMM.tseq.kko]
    Efor, Eana = xp.assimilate(HMM, xx, yy)

    # Calculate etas
    states = stommel.array2states(stommel.ens_modus(Efor), HMM.tseq.times)
    model.ens_member = 0
    etas = np.array([(model.eta1(s), model.eta2(s), model.eta3(s))
                    for s in states])
    trans = np.array([model.fluxes[0].transport(
        s)*np.mean(model.dx*model.dz) for s in states]).flatten()

    # Plot
    fig, ax = stommel.time_figure_with_phase(HMM.tseq)
    for n in range(np.size(Efor, 1)):
        stommel.plot_truth_with_phase(ax, HMM, model, Efor[:, n, :], yy)
    fig.savefig(os.path.join(fig_dir,'truth_with_phase.png'),format='png',dpi=300)


    # Plot spread
    fig, ax = stommel.time_figure(HMM.tseq)
    stommel.plot_relative_spread(ax, HMM.tseq, Efor, yy)
    fig.savefig(os.path.join(fig_dir,'relative_spread.png'),format='png',dpi=300)

    fig = plt.figure()
    ax = fig.subplots(1, 1)
    for n, eta in enumerate(etas.T):
        ax.plot(HMM.tseq.times/stommel.year, eta, label='eta'+str(n+1))
    ax.grid()
    ax.set_xlabel('Time [year]')
    plt.legend()
    fig.savefig(os.path.join(fig_dir,'etas.png'),format='png',dpi=300)

    #Plot transport
    fig = plt.figure()
    ax = fig.subplots(1, 1)
    ax.plot(HMM.tseq.times/stommel.year, trans/1e6)
    ax.plot(HMM.tseq.times/stommel.year, np.ones_like(HMM.tseq.times)
            * stommel.Q_overturning/1e6, 'k--')
    ax.grid()
    ax.set_xlabel('Time [year]')
    ax.set_ylabel('Transport [Sv]')
    fig.savefig(os.path.join(fig_dir,'transport.png'),format='png',dpi=300)
    
    #Save 1st and last analysis
    with open(os.path.join(fig_dir,'ana_stats.pkl'),'wb') as stream:
        stats = {'mean0':np.mean(Eana[0],axis=0),
                 'mean1':np.mean(Eana[-1],axis=0),
                 'var0':np.var(Eana[0],axis=0,ddof=1),
                 'var1':np.var(Eana[-1],axis=0,ddof=1),
                 'times':[HMM.tseq.otimes[0],HMM.tseq.otimes[-1]],
                 }
        pkl.dump(stats,stream)
        
#%% Print best estimates and 95 confidence interval
from scipy.stats import norm 

mu, sig = np.mean(Eana[-1],0)[4:], np.std(Eana[-1],0,ddof=1)[4:]
best = np.exp(stommel.ens_modus(Eana)[-1,4:])

for b1, mu1, sig1 in zip(best, mu, sig):
    norm1 = norm(loc=mu1, scale=sig1)
    print('(best, confidence 90)',b1, np.exp(norm1.ppf(.05)), np.exp(norm1.ppf(.95)))
    
A_Tair = np.sqrt(np.sum(np.diff(hadley['surface_temperature_harmonic'],axis=1)[1:]**2))
A_Sair = np.sqrt(np.sum(np.diff(hadley['surface_salinity_harmonic'],axis=1)[1:]**2))
dys = [hadley['geo_eq']['dy'],hadley['geo_pole']['dy']]
dybar = np.prod(dys)/np.sum(dys)
print('Omega dimensionaless ',2*np.pi / stommel.year * (hadley['geo_eq']['dz']/best[0]))
print('Tair amplitude', A_Tair, A_Tair * (model.eos.alpha_T * best[-1] / best[0]) * hadley['geo_eq']['dz']/ dybar)
print('Sair amplitude', A_Sair, A_Sair * (best[1]/best[0]) * (model.eos.alpha_S * best[-1] / best[0]) * hadley['geo_eq']['dz'] / dybar)