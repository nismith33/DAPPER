# -*- coding: utf-8 -*-

""" Test run with Stommel model. """
import numpy as np
import dapper.mods as modelling
import dapper.mods.Stommel as stommel
from dapper.da_methods.ensemble import EnKF
import matplotlib.pyplot as plt
from copy import copy 
import os

from dapper.mods.Stommel.ref_forcing    import exp_ref_forcing
from dapper.mods.Stommel.ref_forcing_da import exp_ref_forcing_da
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
    ax[0].plot(np.ones_like(probs) * pos, probs, 'bo', alpha=.5)
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


def error(x, E):
    return x - np.mean(E, axis=0)

def rmse(errors):
    return np.sqrt(np.mean(errors**2))

def bootstrap(func, data, N, confidence=.9):
    r = []
    for _ in range(N):
        data1 = np.random.choice(data, size=len(data))
        r.append(func(data1))
        
    f0 = func(data)
    q = (1.-confidence) * np.array([.5, -.5]) + np.array([0., 1.])
    return (f0, (np.abs(np.quantile(r,q[0])-f0),
                 np.abs(np.quantile(r,q[1])-f0)))

def error_figure():
    plt.close('all')
    fig = plt.figure(figsize=(6,11))
    ax = fig.subplots(3,1)
    
    for ax1 in ax:
        ax1.grid()
        ax1.xaxis.set_ticks=np.arange(0,6)
        ax1.xaxis.set_ticklabels(['','ref','ref_da','clima','clima_da',''])

def plot_error(ax, errors, i, name):
    ax.set_ylabel("RMSE " + name)
    x,y,dy = [], [], []
    
    for n,key in enumerate(errors):
        x.append(n+1)
        y1 = bootstrap(rmse, errors[key][:,i], 400)
        y.append(y1[0])
        dy.append(y1[1])
        
    x = np.array(x)
    y = np.array(y)
    dy = np.array(dy)
    
    ax.errorbar(x,y,yerr=dy)

if __name__=='__main__':
    NG = 50  #number of instances
    N  = 50 #ensemble members per instant
    
    prob_flip, errors={}, {}
    for key in ['ref','ref_da','clima','clima_da']:
        prob_flip[key] = []    
        errors[key] = []
    
    seed=1000
    for ng in range(NG):
        print('Running instant ', ng)
        
        seed+=20
        xp, HMM, model = exp_ref_forcing(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        prob_flip['ref'].append(stommel.prob_change(Efor) * 100.)
        errors['ref'].append(error(xx[-1], Efor[-1]))
        
        seed+=20
        xp, HMM, model = exp_ref_forcing_da(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        prob_flip['ref_da'].append(stommel.prob_change(Efor) * 100.)
        errors['ref_da'].append(error(xx[-1], Efor[-1]))
        
        seed+=20
        xp, HMM, model = exp_clima_forcing(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        prob_flip['clima'].append(stommel.prob_change(Efor) * 100.)
        errors['clima'].append(error(xx[-1], Efor[-1]))
        
        seed+=20
        xp, HMM, model = exp_clima_forcing_da(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        prob_flip['clima_da'].append(stommel.prob_change(Efor) * 100.)
        errors['clima_da'].append(error(xx[-1], Efor[-1]))
        
    fig,ax = prob_figure() 
    for n, key in enumerate(prob_flip):
        plot_prob(ax, n+1, prob_flip[key])
    
    #Save figure 
    fig.savefig(os.path.join(stommel.fig_dir, 'probabilities.png'),
                format='png', dpi=500)
    
    #Plot errors
    fig, ax = error_figure()
    for ax1, i, name in zip(ax, (4,5,6), ['temp_diff', 'salt_diff','trans']):
        plot_error(ax1, errors, i, name)
    
    
    

        
    
   

