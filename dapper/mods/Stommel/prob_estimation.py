# -*- coding: utf-8 -*-

""" Test run with Stommel model. """
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from copy import copy 
import os

from dapper.mods.Stommel.forcing_noise    import exp_forcing_noise
from dapper.mods.Stommel.forcing_noise_da import exp_forcing_noise_da
from dapper.mods.Stommel.clima_forcing import exp_clima_forcing 
from dapper.mods.Stommel.clima_forcing_da import exp_clima_forcing_da

def sample_stat(samples):
    from scipy.stats import t
    mu = np.mean(samples)
    std = np.std(samples)
    N = len(samples)
    #r = np.array([t.ppf(.025,N), t.ppf(.975,N)]) * (std / np.sqrt(N))
    r=np.quantile(samples,[.025,.975])
    r=np.abs(r-mu)
    return mu, std, r

def prob_figure():
    plt.close('all')
    fig = plt.figure(figsize=(8,4))
    ax = fig.subplots(1,1)
    return fig, [ax]

def plot_prob(ax, pos, probs):
    ax[0].plot(np.ones_like(pref) * pos, probs, 'bo', alpha=.5)
    mu, std, r = sample_stat(probs)
    ax[0].errorbar(np.array([pos]),np.array([mu]),color='k',
                   marker='o',yerr=np.reshape(r,(2,-1)))
    
    ax[0].set_ylabel('Probability SA [%]')
    
    for ax1 in ax:
        ax1.grid()
        ax1.set_xlim(0,5)
        ax1.set_ylim(0,100)
        ax1.xaxis.set_ticks(np.array([1,2,3,4]))
        ax1.xaxis.set_ticklabels(['Ref','Ref DA','Climate','Climate DA'])


if __name__=='__main__':
    NG = 50  #number of instances
    N  = 50 #ensemble members per instant
    
    pref, pref_da, pclima, pclima_da = [], [], [], []
    
    seed=1000
    for ng in range(NG):
        print('Running instant ', ng)
        
        seed+=20
        xp, HMM, model = exp_forcing_noise(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        pref.append(stommel.prob_change(Efor) * 100.)
        
        seed+=20
        xp, HMM, model = exp_forcing_noise_da(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        pref_da.append(stommel.prob_change(Efor) * 100.)
        
        seed+=20
        xp, HMM, model = exp_clima_forcing(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        pclima.append(stommel.prob_change(Efor) * 100.)
        
        seed+=20
        xp, HMM, model = exp_clima_forcing_da(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        pclima_da.append(stommel.prob_change(Efor) * 100.)
        
    fig,ax = prob_figure() 
    plot_prob(ax, 1, pref)
    plot_prob(ax, 2, pref_da)
    plot_prob(ax, 3, pclima)
    plot_prob(ax, 4, pclima_da)
    
    #Save figure 
    fig_dir='/home/ivo/dpr_data/stommel'
    fig.savefig(os.path.join(fig_dir, 'probabilities.png'),format='png',dpi=500)
    
    
   

