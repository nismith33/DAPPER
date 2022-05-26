#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:35:11 2022

@author: ivo
"""

from dapper.mods.Ice.demo import *
from dapper.tools.datafiles import NetcdfIO
import netCDF4 as netcdf
from dapper.stats import Stats
from dapper.stats import ens_mean, ens_spread, ens_rmse, ens_ranks
from dapper.stats import window
import os

from dapper.mods.Ice.forcings import AR1

HMM = aev_pnormal()


fig_dir='/home/ivo/dpr_data/mpi_test/test'
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)

xps = xpList()
xps = xp_wind(xps,'test')

nc=xps[0].save_nc
xx,yy=nc.read_truth()

with nc.ostream as stream:
    times = np.array(stream['time'][:])
    timeos = np.array(stream['timeo'][:])
    Efor = np.array(stream['forecast'][:,:,:])
    Eana = np.array(stream['analysis'][:,:,:])
 
stat=Stats(xps[0], HMM, xx, yy, False, store_u=True)
for k,e in enumerate(Efor):
    stat.assess(k=k,faus='u',E=e)
    if times[k] in timeos:
        ko = np.where(times[k]==timeos)[0][0]
        stat.assess(k=k,ko=ko,faus='f',E=e)
        stat.assess(k=k,ko=ko,faus='a',E=Eana[ko])   
    
def window(to,valso,t,vals):
    """
    Combine analysis with forecast into DA windows. 

    Parameters
    ----------
    to : array of floats
        Analysis time. 
    valso : array of floats
        Array with analysis values. 
    t : array of floats
        Array with forecast times.
    vals : array of floats
        Array with forecast values. 

    Returns
    -------
    windows : list
        List with (time,value)-pairs for each DA window.

    """
    windows = [(t[t<min(to)],vals[t<min(to)])]
    for tl,tu in zip(to[:-1],to[1:]):
        in_for = [tl<t1 and t1<=tu for t1 in t] 
        in_ana = [tl<=t1 and t1<tu for t1 in to]
        t_win = np.append(to[in_ana],t[in_for])
        val_win = np.concatenate((valso[in_ana],vals[in_for]), axis=0)
        windows.append((t_win,val_win))
    return windows
    
#%% Plot RMSE

names = [('velocity_ice','ms-1'),('thickness_ice','m'),('stress','Pa')]

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

plt.close('all')
fig=plt.figure(figsize=(11,8))

def plot_rmse(ax, name, unit):
    
    #Time axis
    time_vals = times/60/24
    xlim = (min(time_vals),max(time_vals))
    locator = ticker.FixedLocator(np.arange(xlim[0],xlim[1]))
    ax.xaxis.set_minor_locator(locator)
    ax.set_xlim(xlim)
    ax.set_xlabel('time [days]')
    ax.grid()
    
    #Forecast RMSE
    rms_for = ens_rmse(xx ,Efor, sector=HMM.sectors[name])
    spr_for = ens_spread(Efor, sector=HMM.sectors[name])
    
    #Analysis RMSE
    in_ana = [t1 in timeos for t1 in times]
    rms_ana = ens_rmse(xx[in_ana], Eana, sector=HMM.sectors[name])
    spr_ana = ens_spread(Eana, sector=HMM.sectors[name])
    
    #Combine forecasts and analysis into windows.
    rms_win = window(timeos,rms_ana,times,rms_for)
    spr_win = window(timeos,spr_ana,times,spr_for)
    
    #Plot spread
    for spr1 in spr_win:  
        ax.plot(spr1[0]/60/24, spr1[1],'k-',label='spread')
    #Plot RMSE
    for rms1 in rms_win:
        ax.plot(rms1[0]/60/24, rms1[1],'b-',label='RMSE')
    
    ax.set_ylabel(name+' ['+unit+']')
    return ax

for figno,(name,unit) in enumerate(names):
    ax=fig.add_subplot(1,3,figno+1)
    ax=plot_rmse(ax,name,unit)
 
#Set legend
fig.subplots_adjust(wspace=.35, hspace=None)
fig.savefig(os.path.join(fig_dir,'rms_spread.png'),dpi=600,format='png')
    
#%%

names = [('velocity_ice','ms-1'),('thickness_ice','m'),('stress','Pa')]

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

def time_str(time):
    from datetime import timedelta 
    
    days = int(np.floor(time/24/60))
    time = time-days*24*60
    
    hours = int(time/60)
    time = time-hours*60
    
    return "{:d} days {:02d}:{:02d}".format(days,int(hours),int(time))
    
def plot_ens_values(ax, truth, E, name):
    sector = HMM.sectors[name]
    x = HMM.coordinates[sector]
    
    #Plot ensemble. 
    for e in E:
        ax.plot(x*1e1, e[sector], 'k-', c="0.65")
    e_mean = np.mean(E[:,sector], axis=0)
    e_std = np.std(E[:,sector], axis=0)
    ax.plot(x*1e1,e_mean,'k-',label=r'ensemble mean')
    ax.plot(x*1e1,e_mean+e_std,'k--',label=r'mean+/-std')
    ax.plot(x*1e1,e_mean-e_std,'k--')
    ax.plot(x*1e1,truth[sector],'b-',label='truth')
    
    xlim = (np.floor(min(x)*1e1), np.ceil(max(x)*1e1))
    ax.set_xlim(xlim)
    ax.grid()
    
    return ax
    
def plot_for_values(t, axes=None, fig_name=None):
    if axes is None:
        plt.close('all')
        fig=plt.figure(figsize=(11,8))
        fig.subplots_adjust(wspace=.35, hspace=None)
        axes = fig.subplots(1,len(names))
    else:
        for ax in axes:
            ax.cla()
    
    it = np.where(t==times)[0][0]
    for ax, name in zip(axes, names):
        ax = plot_ens_values(ax, xx[it,:], Efor[it,:,:], name[0])
        ax.set_ylabel('Forecast '+name[0]+' ['+name[1]+']')
        ax.set_xlabel('Position [km]')
        ax.set_title(time_str(times[it]))

    #Set legend
    plt.legend(loc='upper right')
    
    if fig_name is not None:                
        plt.savefig(os.path.join(fig_dir,fig_name),dpi=150,format='png')
    
    return axes

def plot_forana_values(t, axes=None, fig_name=None):
    if axes is None:
        plt.close('all')
        fig=plt.figure(figsize=(11,8))
        fig.subplots_adjust(wspace=.35, hspace=None)
        axes = fig.subplots(2,len(names))
    else:
        for ax in np.reshape(axes,(-1)):
            ax.cla()
    
    it = np.where(t==times)[0][0]
    ito = np.where(t==timeos)[0][0]
    for ax_for, ax_ana, name in zip(axes[0], axes[1], names):
        ax_for = plot_ens_values(ax_for, xx[it,:], Efor[it,:,:], name[0])
        ax_for.set_ylabel('Forecast '+name[0]+' ['+name[1]+']')
        
        ax_ana = plot_ens_values(ax_ana, xx[it,:], Eana[ito,:,:], name[0])
        ax_ana.set_ylabel('Analysis '+name[0]+' ['+name[1]+']')
        
        ax_ana.set_xlabel('Position [km]')
        ax_for.set_title(time_str(times[it]))
        
        if name[0]=='velocity_ice':
            sector = HMM.sectors['velocity_ice']
            x_obs = HMM.coordinates[sector][2::5]
            ax_for.errorbar(x_obs*1e1, yy[ito], yerr=2e-3,
                            fmt='o', color='r')
        
        ylims=np.array([ax_for.get_ylim(),ax_ana.get_ylim()])
        ax_for.set_ylim(min(ylims[:,0]),max(ylims[:,1]))
        ax_ana.set_ylim(min(ylims[:,0]),max(ylims[:,1]))
        
        

    #Set legend
    plt.legend(loc='upper right')
    
    if fig_name is not None:                
        plt.savefig(os.path.join(fig_dir,fig_name),dpi=150,format='png')
    
    return axes

axes = None
if len(timeos)==1:
    for it,t in enumerate(times[::6]):
        axes=plot_for_values(t, axes, 
                             fig_name='for_values_{:03d}.png'.format(it))
else:
    for it,t in enumerate(timeos):
        axes=plot_forana_values(t, axes, 
                             fig_name='forana_values_{:03d}.png'.format(it))



#%% Rank histogram

names = [('velocity_ice','ms-1'),('thickness_ice','m'),('stress','Pa')]

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

def plot_rank(ax, truth, E, name):
    sector=HMM.sectors[name]
    r = ens_ranks(truth, E, sector)
    r = np.sum(r,axis=0)
    r = r/sum(r)
    
    bins = np.arange(0,np.size(E,1)+1)
    ax.bar(bins,r)
    
    ax.plot(bins,np.ones_like(bins)*(1./(len(bins)-1)),'k-')
    ax.grid()
    
    return ax
    
def plot_for_ranks(fig_name=None):
    plt.close('all')
    fig=plt.figure(figsize=(11,8))
    fig.subplots_adjust(wspace=.35, hspace=None)
    axes = fig.subplots(1,len(names))
    
    for ax,name in zip(axes,names):
        ax=plot_rank(ax, xx, Efor, name[0])
        ax.set_xlabel("Rank")
        ax.set_title(name[0])
        
    axes[0].set_ylabel("Probability")
    
    if fig_name is not None:                
        fig.savefig(os.path.join(fig_dir,fig_name),dpi=400,format='png')
    
    return axes  

def plot_forana_ranks(fig_name=None):
    plt.close('all')
    fig=plt.figure(figsize=(11,8))
    fig.subplots_adjust(wspace=.35, hspace=None)
    axes = fig.subplots(2,len(names))
    
    in_ana = [t1 in timeos for t1 in times]
    for ax_for, ax_ana, name in zip(axes[0],axes[1],names):
        ax_for=plot_rank(ax_for, xx[in_ana], Efor[in_ana], name[0])
        ax_ana=plot_rank(ax_ana, xx[in_ana], Eana, name[0])
        ax_ana.set_xlabel("Rank")
        ax_for.set_title(name[0])
        
    axes[0][0].set_ylabel("Probability forecast")
    axes[1][0].set_ylabel("Probability analysis")
    
    if fig_name is not None:                
        fig.savefig(os.path.join(fig_dir,fig_name),dpi=400,format='png')
    
    return axes  

fig_name = 'rank.png'
if len(timeos)==1:
    plot_for_ranks(fig_name)
else:
    plot_forana_ranks(fig_name)