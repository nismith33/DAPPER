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
    fig = plt.figure(figsize=(11,6))
    ax = fig.subplots(2,3)
    fig.subplots_adjust(left=.1,right=.98,wspace=.3)
    
    for ax1 in np.reshape(ax,(-1)):
        ax1.grid()
        
    return fig, ax

def plot_error(ax, errors):
    draws = []
    ax = np.reshape(ax,(-1))
    for key in errors:
        errors1 = np.array(errors[key])
        
        temp = errors1[:,:,1]-errors1[:,:,0]
        salt = errors1[:,:,3]-errors1[:,:,2]
        
        x=np.arange(0,np.size(temp,1))
        ax[0].plot(x, np.sqrt(np.mean(temp**2,axis=0)), label=key)
        
        draw, = ax[1].plot(x, np.sqrt(np.mean(salt**2,axis=0)), label=key)
        draws.append(draw)
        
        for n,ax1 in enumerate(ax[3:]):
            ax1.plot(x, np.sqrt(np.mean(errors1[:,:,n+4]**2,axis=0)))
        
    ax[0].set_ylabel('RMSE temp diff. [C]')
    ax[1].set_ylabel('RMSE salt diff. [ppt]')
    ax[3].set_ylabel('RMSE temp_diff [mms-1]')
    ax[4].set_ylabel('RMSE salt_diff [mms-1]')
    ax[5].set_ylabel('RMSE gamma [ms-1]')
    for ax1 in ax:
        ax1.set_xlabel('Time [year]')
        
    ax[1].legend(handles=draws, loc='upper left', framealpha=1.)
        
    return draws

def plot_perror(ax, errors):
    
    ax.set_ylabel('RMSE probability')
    ax.set_ylim(0,1)
    
    for key in errors:
        x=np.arange(0, np.size(errors[key],1))
        vals = np.mean(np.abs(np.array(errors[key])), axis=0)
        ax.plot(x, vals, label=key)

def plot_spreads(ax, spreads):
    draws = []
    ax = np.reshape(ax,(-1))
    for key in errors:
        spread1 = np.array(spreads[key])

        x=np.arange(0,np.size(spread1,1))
        ax[0].plot(x, np.sqrt(np.mean(spread1[:,:,0]+spread1[:,:,1],axis=0)), 
                   label=key)
        
        draw, = ax[1].plot(x, np.sqrt(np.mean(spread1[:,:,3]+spread1[:,:,2],axis=0)), 
                   label=key)
        draws.append(draw)
        
        for n,ax1 in enumerate(ax[3:]):
            ax1.plot(x, np.sqrt(np.mean(spread1[:,:,n+4],axis=0)))
        
    ax[0].set_ylabel('Spread temp diff. [C]')
    ax[1].set_ylabel('Spread salt diff. [ppt]')
    ax[3].set_ylabel('Spread temp_diff [mms-1]')
    ax[4].set_ylabel('Spread salt_diff [mms-1]')
    ax[5].set_ylabel('Spread gamma [ms-1]')
    for ax1 in ax:
        ax1.set_xlabel('Time [year]')
        
    ax[1].legend(handles=draws, loc='upper left', framealpha=1.)
        
    return draws

def plot_pspread(ax, spreads):
    
    ax.set_ylabel('Spread probability')
    ax.set_ylim(0,1)
    
    for key in errors:
        x=np.arange(0, np.size(errors[key],1))
        vals = np.sqrt(np.mean(np.array(spreads[key]), axis=0))
        ax.plot(x, vals, label=key)

    
def plot_ensemble(filename, xx, yy, Efor):
    plt.close('all')
    fig=plt.figure(figsize=(8,4))
    ax = fig.subplots(1,2)
    drawings = []
    
    for ax1 in ax:
        ax1.grid()
        ax1.set_xlabel('Time [year]')
        ax1.set_xlim(0,200)
    ax[0].set_ylabel("Temperature difference [C]")
    ax[0].set_ylim(-4,18)
    ax[1].set_ylabel("Salinity difference [ppt]")
    ax[1].set_ylim(0,4)
    
    t=np.arange(0,np.size(Efor,0))
    for x in np.transpose(Efor, (1,0,2)):
        drawing, = ax[0].plot(t, x[:,1]-x[:,0], 'k-', c='0.9', label='member')
        ax[1].plot(t, x[:,3]-x[:,2], 'k-', c='0.9', label='member')
    drawings.append(drawing)
    
    drawing, = ax[0].plot(t, np.mean(Efor[:,:,1]-Efor[:,:,0],axis=1), 'g-', label='forecast')
    ax[1].plot(t, np.mean(Efor[:,:,3]-Efor[:,:,2],axis=1), 'g-', label='forecast')
    drawings.append(drawing)
    
    drawing,=ax[0].plot(t, xx[:,1]-xx[:,0], 'k-', label='truth')
    ax[1].plot(t, xx[:,3]-xx[:,2], 'k-', label='truth')
    drawings.append(drawing)
    
    if len(yy)>0:
        t = np.arange(1,len(yy)+1)
        yy = np.array(yy)
        drawing,=ax[0].plot(t, yy[:,1]-yy[:,0], 'bo', label='obs', markersize=.8)
        ax[1].plot(t, yy[:,3]-yy[:,2], 'bo', label='obs', markersize=.8)
        drawings.append(drawing)
        
    ax[0].legend(handles=drawings,loc='lower right')
        
    fig.savefig(os.path.join(stommel.fig_dir,'ens',filename), format='png', dpi=300)
    
def confusion_matrix(matrix, xx, Efor):
    xfor = np.mean(Efor[-1], axis=0)
    states=stommel.array2states(np.array([xx[-1], xfor]))
    
    if states[0].regime=='TH' and states[1].regime=='TH':
        matrix[0,0] += 1
    elif states[0].regime!='TH' and states[1].regime=='TH':
        matrix[1,0] += 1
    elif states[0].regime=='TH' and states[1].regime!='TH':
        matrix[0,1] += 1
    elif states[0].regime!='TH' and states[1].regime!='TH':
        matrix[1,1] += 1
    
    return matrix

def plot_confusion_matrix(matrix):
    plt.close('all')
    fig = plt.figure(figsize=(8,8))
    ax = np.reshape(fig.subplots(2,2),(-1))
    
    for ax1, key in zip(ax, matrix):
        drawing=ax1.pcolor(matrix[key]/np.sum(matrix[key]), vmin=0, vmax=1)
        ax1.set_ylabel('Truth')
        ax1.set_xticks([.5,1.5])
        ax1.xaxis.set_ticklabels(['TH','SA'])
        ax1.set_xlabel('Forecast')
        ax1.set_yticks([.5,1.5])
        ax1.yaxis.set_ticklabels(['TH','SA'])
        
        plt.colorbar(drawing,ax=ax1)
        ax1.set_title(key)
        
    return fig, ax

def plot_entropy(entropy):
    plt.close('all')
    fig = plt.figure(figsize=(6,4))
    ax = fig.subplots(1,1)
    
    ax.set_xlabel('Time [year]')
    ax.set_ylabel('Cross entropy')
    ax.grid()
    ax.set_xlim(0,200)
    
    for key in entropy:
        x=np.arange(0, np.size(entropy[key],1))
        ax.plot(x, np.mean(np.array(entropy[key]), axis=0), label=key)
        
    ax.legend(framealpha=1.)
        

if __name__=='__main__':
    NG = 50  #number of instances
    N  = 50 #ensemble members per instant
    
    prob_flip, errors, spread, cmatrix, perrors, pspread, pentropy={}, {}, {}, {}, {}, {}, {}
    for key in ['ref','ref_da','clima','clima_da']:
        prob_flip[key] = []    
        errors[key] = []
        spread[key] = []
        cmatrix[key] = np.zeros((2,2))
        perrors[key] = []
        pspread[key] = []
        pentropy[key] = []
    
    seed=1000
    for ng in range(NG):
        print('Running instant ', ng)
        
        np.random.seed(seed)
        xp, HMM, model = exp_ref_forcing(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        perrors['ref'].append(stommel.error_prob(xx, Efor))
        pspread['ref'].append(stommel.spread_prob(Efor))
        pentropy['ref'].append(stommel.cross_entropy(xx,Efor))
        errors['ref'].append(np.mean(Efor,axis=1)-xx)
        spread['ref'].append(np.var(Efor,axis=1,ddof=1))
        cmatrix['ref']=confusion_matrix(cmatrix['ref'], xx, Efor)
        plot_ensemble("ens_ref_{:02d}".format(ng), xx, yy, Efor)
        
        np.random.seed(seed)
        xp, HMM, model = exp_ref_forcing_da(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        perrors['ref_da'].append(stommel.error_prob(xx, Efor))
        pspread['ref_da'].append(stommel.spread_prob(Efor))
        pentropy['ref_da'].append(stommel.cross_entropy(xx,Efor))
        errors['ref_da'].append(np.mean(Efor,axis=1)-xx)
        spread['ref_da'].append(np.var(Efor,axis=1,ddof=1))
        cmatrix['ref_da']=confusion_matrix(cmatrix['ref_da'], xx, Efor)
        plot_ensemble("ens_ref_da_{:02d}".format(ng), xx, yy, Efor)
        
        np.random.seed(seed)
        xp, HMM, model = exp_clima_forcing_da(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        perrors['clima_da'].append(stommel.error_prob(xx, Efor))
        pspread['clima_da'].append(stommel.spread_prob(Efor))
        pentropy['clima_da'].append(stommel.cross_entropy(xx,Efor))
        errors['clima_da'].append(np.mean(Efor,axis=1)-xx)
        spread['clima_da'].append(np.var(Efor,axis=1,ddof=1))
        cmatrix['clima_da']=confusion_matrix(cmatrix['clima_da'], xx, Efor)
        plot_ensemble("ens_clima_da_{:02d}".format(ng), xx, yy, Efor)
        
        np.random.seed(seed)
        xp, HMM, model = exp_clima_forcing(N, seed)
        xx, yy = HMM.simulate()
        Efor, Eana = xp.assimilate(HMM, xx, yy)
        perrors['clima'].append(stommel.error_prob(xx, Efor))
        pspread['clima'].append(stommel.spread_prob(Efor))
        pentropy['clima'].append(stommel.cross_entropy(xx,Efor))
        errors['clima'].append(np.mean(Efor,axis=1)-xx)
        spread['clima'].append(np.var(Efor,axis=1,ddof=1))
        cmatrix['clima']=confusion_matrix(cmatrix['clima'], xx, Efor)
        plot_ensemble("ens_clima_{:02d}".format(ng), xx, yy, Efor)
        
        seed+=20
        
    #fig,ax = prob_figure() 
    #for n, key in enumerate(prob_flip):
    #    plot_prob(ax, n+1, prob_flip[key])
    #fig.savefig(os.path.join(stommel.fig_dir, 'probabilities.png'),
    #            format='png', dpi=500)
    
    #Plot errors
    fig, ax = error_figure()
    plot_error(ax, errors)
    plot_perror(ax[0,2], perrors)
    #Save figure 
    fig.savefig(os.path.join(stommel.fig_dir, 'rmse_time.png'),
                format='png', dpi=500)
    
    #Plot spread
    fig, ax = error_figure()
    plot_spreads(ax, spread)
    plot_pspread(ax[0,2], pspread)
    #Save figure 
    fig.savefig(os.path.join(stommel.fig_dir, 'spread_time.png'),
                format='png', dpi=500)
    
    fig,ax = plot_confusion_matrix(cmatrix)
    fig.savefig(os.path.join(stommel.fig_dir, 'confusion_matrix.png'),
                format='png', dpi=500)
    
    plot_entropy(pentropy)
    plt.savefig(os.path.join(stommel.fig_dir, 'entropy.png'),
                format='png', dpi=500)

    
    
    

        
    
   

