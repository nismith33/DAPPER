#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:35:11 2022

@author: ivo
"""

import matplotlib.pyplot as plt
from dapper.tools.inflation import ObsInflator, AnaInflator, AdaptiveRTPP
import matplotlib.ticker as ticker

import os
import shutil
import sys
import numpy as np
import netCDF4 as netcdf
from dapper.tools.datafiles import NetcdfIO
from dapper.stats import Stats
from dapper.stats import ens_mean, ens_spread, ens_rmse, ens_ranks
import matplotlib
#from dapper.stats import window

font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 14}
matplotlib.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['text.usetex'] = False

basename = 'p4_refinement_OBSthickness_ice_NO016'
#basename = 'p4_obsres_OBSthickness_ice_NO016'
data_dir = '/home/ivo/dpr_data/obsdensity/p4_Q'
mod_dir = '/home/ivo/Code/DAPPER/dapper/mods/Ice'

# Copy file wit experiment setup
sys.path.append(data_dir)
shutil.copyfile(os.path.join(data_dir, basename+'.py'),
                os.path.join(mod_dir, 'exp_setup.py'))

# Create directory to store figures.
fig_dir = os.path.join(data_dir, basename)
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)

# This run
nc = NetcdfIO(os.path.join(data_dir, basename+'.nc'))
xx, yy = nc.read_truth()
with nc.ostream as stream:
    times = np.array(stream['time'][:])
    timeos = np.array(stream['timeo'][:])
    Efor = np.array(stream['forecast'][:, :, :])
    Eana = np.array(stream['analysis'][:, :, :])
    
    in_win = times<=24*16*60
    times = times[in_win]; Efor=Efor[in_win]; xx=xx[in_win]
    in_win = timeos<=24*16*60
    timeos = timeos[in_win]; Eana=Eana[in_win]; #yy=yy[in_win]

# Import the experiment.
from dapper.mods.Ice.exp_setup import create_exp_refinement, aev_pnormal
HMM, xps = create_exp_refinement()
xps = xps[0:1]

for xp in xps:
    if xp.save_nc.file_path == nc.file_path:
        HMM = aev_pnormal(xp)
        break

# #Run without DA
Efor0, Eana0 = None, None
# nc0=nc=NetcdfIO(os.path.join(data_dir,'wind_noda3.nc'))
# with nc0.ostream as stream:
#      times0 = np.array(stream['time'][:])
#      timeos0 = np.array(stream['timeo'][:])
#      Efor0 = np.array(stream['forecast'][:,:,:])
#      Eana0 = np.array(stream['analysis'][:,:,:])

#      in0 = times0<=max(times)
#      Efor0=Efor0[in0,:,:]
#      in0 = timeos0<=max(timeos)
#      Eana0=Eana0[in0,:,:]

# Dapper statistics
stat = Stats(xps[0], HMM, xx, yy, False, store_u=True)
for k, e in enumerate(Efor):
    stat.assess(k=k, faus='u', E=e)
    if times[k] in timeos:
        ko = np.where(times[k] == timeos)[0][0]
        stat.assess(k=k, ko=ko, faus='f', E=e)
        stat.assess(k=k, ko=ko, faus='a', E=Eana[ko])

def window(to, valso, t, vals):
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
    windows = [(t[t < min(to)], vals[t < min(to)])]
    for tl, tu in zip(to[:-1], to[1:]):
        in_for = [tl < t1 and t1 <= tu for t1 in t]
        in_ana = [tl <= t1 and t1 < tu for t1 in to]
        t_win = np.append(to[in_ana], t[in_for])
        val_win = np.concatenate((valso[in_ana], vals[in_for]), axis=0)
        windows.append((t_win, val_win))
    return windows


def plot_time(t): return t/60/24

# %% Plot RMSE


names = [('velocity_ice', 'ms-1'), ('thickness_ice', 'm'), ('stress', '')]


def create_time_axis(ax, with_label=True):
    time_vals = times/60/24
    xlim = (min(time_vals), max(time_vals))
    locator = ticker.FixedLocator(np.arange(xlim[0], xlim[1]))
    ax.xaxis.set_minor_locator(locator)
    ax.set_xlim(xlim)
    ax.grid()

    if with_label:
        ax.set_xlabel('time [days]')

    return ax

def plot_obs_sig(ax, obs_db, name):
    selection = obs_db['field_name'] == name
    db = obs_db[selection]

    times = np.sort(np.unique(db['time']))
    sigs = []
    for time in times:
        selection = db['time'] == time
        if not any(selection):
            sigs.append(np.nan)
        else:
            sigs.append(np.mean(db['var'][selection]))

    times = plot_time(times)
    sigs = np.sqrt(np.array(sigs))

    handle, = ax.plot(times, sigs, 'm-',
                      label='obs. std.', linewidth=2.)

    return handle

def plot_rmse_time(ax, name, unit, Efor0=None):
    """ Plot RMS error together with ensemble spread as function of time."""

    # Time axis
    ax = create_time_axis(ax)

    if Efor0 is not None:
        in_ana = [t1 in times for t1 in timeos]
        spr_for0 = ens_spread(Efor0[in_ana], sector=HMM.sectors[name])
        rms_for0 = ens_rmse(xx[in_ana], Efor0[in_ana],
                            sector=HMM.sectors[name])
        plt_spr_for0 = ax.plot(plot_time(timeos), spr_for0, 'k:',
                               label='spread no DA', linewidth=0.8)
        plt_rms_for0 = ax.plot(plot_time(timeos), rms_for0, 'c:',
                               label='RMSE no DA', linewidth=0.8)
    

    # Analysis RMSE
    #Eana=Efor[0::3]
    #timeos = times[0::3]
    in_ana = [t1 in timeos for t1 in times]
    rms_ana = ens_rmse(xx[in_ana], Eana, sector=HMM.sectors[name])
    spr_ana = ens_spread(Eana, sector=HMM.sectors[name])

    # Forecast RMSE    
    rms_for = ens_rmse(xx[in_ana], Efor[in_ana], sector=HMM.sectors[name])
    spr_for = ens_spread(Efor[in_ana], sector=HMM.sectors[name])

    # Plot
    plt_spr_for = ax.plot(plot_time(timeos), spr_for, 'k-',
                          alpha=0.3, label='spread forecast', linewidth=3.5)
    plt_rms_for = ax.plot(plot_time(timeos), rms_for, 'b-',
                          alpha=0.3, label='RMSE forecast', linewidth=3.5)
    plt_spr_ana = ax.plot(plot_time(timeos), spr_ana, 'k-',
                          label='spread analysis', linewidth=1)
    plt_rms_ana = ax.plot(plot_time(timeos), rms_ana, 'b-',
                          linewidth=1, label='RMSE analysis')

    # Plot observations
    #plt_sig_obs = plot_obs_sig(ax, xps[0].obs_db, name)
    if name=='thickness_ice' and True:
        sig=np.sqrt(np.mean(xps[0].obs_db['var'][xps[0].obs_db['field_name']=='thickness_ice']))
        plt.plot(times,np.ones_like(times)*sig,'m--',linewidth=2.)
    
    # Calculate total RMSE
    for sector, isector in HMM.sectors.items():
        in_ana = [no for no,t in enumerate(times) if t in timeos]
        
        E1=Efor[in_ana,:,:]; E1=E1[:,:,isector]
        xx1=xx[in_ana,:]; xx1=xx1[:,isector]
        error = np.mean(E1, axis=1)-xx1
        rmse_for = np.sqrt(np.mean(error**2))
        
        error = xx[in_ana, :]
        error = np.mean(Eana[:, :, isector], axis=1)-error[:, isector]
        rmse_ana = np.sqrt(np.mean(error**2))

        if sector==name:
            print('RMSE for/ana ', sector, rmse_for, rmse_ana)
        
    if name=='velocity_ice':
        ax.set_ylim(0.,.012)
    elif name=='stress':
        ax.set_ylim(0.,200.)
    #elif name=='thickness_ice':
    #    ax.set_ylim(0.,5e-3)

    ax.legend(loc='upper left',framealpha=1.)
    ax.set_ylabel(name+' ['+unit+']',fontsize=font['size'])

    return ax


def plot_rmse_window(ax, name, unit):
    """ Plot RMS error together with ensemble spread as function of time. 
    Each line represents a window."""

    # Time axis
    ax = create_time_axis(ax)

    # Forecast RMSE
    rms_for = ens_rmse(xx, Efor, sector=HMM.sectors[name])
    spr_for = ens_spread(Efor, sector=HMM.sectors[name])

    # Analysis RMSE
    in_ana = [t1 in timeos for t1 in times]
    rms_ana = ens_rmse(xx[in_ana], Eana, sector=HMM.sectors[name])
    spr_ana = ens_spread(Eana, sector=HMM.sectors[name])

    # Combine forecasts and analysis into windows.
    rms_win = window(timeos, rms_ana, times, rms_for)
    spr_win = window(timeos, spr_ana, times, spr_for)

    # Plot spread
    for spr1 in spr_win:
        plt_spr, = ax.plot(plot_time(spr1[0]), spr1[1], 'k-', label='spread')
    # Plot RMSE
    for rms1 in rms_win:
        plt_rms, = ax.plot(plot_time(rms1[0]), rms1[1], 'b-', label='RMSE')
    plt_handles = [plt_spr, plt_rms]

    # Plot observations
    #plt_sig_obs = plot_obs_sig(ax, xps[0].obs_db, name)
    #plt_handles + [plt_sig_obs]

    ax.legend(handles=plt_handles, loc='upper right')
    ax.set_ylabel(name+' ['+unit+']')

    return ax


plt.close('all')
fig = plt.figure(figsize=(8, 11))
for figno, (name, unit) in enumerate(names):
    ax = fig.add_subplot(3, 1, figno+1)
    ax = plot_rmse_window(ax, name, unit)
fig.subplots_adjust(hspace=.25)
fig.savefig(os.path.join(fig_dir, 'rms_spread_window.png'),
            dpi=600, format='png')

plt.close('all')
fig = plt.figure(figsize=(21, 6))
for figno, (name, unit) in enumerate(names):
    ax = fig.add_subplot(1,3, figno+1)
    ax = plot_rmse_time(ax, name, unit)
fig.subplots_adjust(wspace=0.2,left=.06,right=.98,top=.98)
fig.savefig(os.path.join(fig_dir, 'rms_spread_time.png'),
            dpi=600, format='png')

if Efor0 is not None:
    plt.close('all')
    fig = plt.figure(figsize=(8, 11))
    for figno, (name, unit) in enumerate(names):
        ax = fig.add_subplot(3, 1, figno+1)
        ax = plot_rmse_time(ax, name, unit, Efor0)
        fig.subplots_adjust(hspace=.25)
        fig.savefig(os.path.join(
            fig_dir, 'rms_spread_time_noda.png'), dpi=600, format='png')

# %%

names = [('velocity_ice', 'ms-1'), ('thickness_ice', 'm'), ('stress', '')]


def time_str(time):
    """ Convert time in minutes to time string."""
    from datetime import timedelta

    days = int(np.floor(time/24/60))
    time = time-days*24*60

    hours = int(time/60)
    time = time-hours*60

    return "{:d} days {:02d}:{:02d}".format(days, int(hours), int(time))


def plot_obs_space(ax, yy, obs_db, name, time):
    selection = obs_db['time'] == time
    db = obs_db[selection]

    selection = db['field_name'] == name
    if any(selection):
        yy_field = yy[selection]
        error_field = np.sqrt(db['var'][selection])
        x_field = np.reshape(db['coordinates'][selection], (-1))

        ax.errorbar(x_field*1e1, yy_field, yerr=error_field,
                    fmt='o', color='r')

    return ax


def plot_ens_values(ax, truth, E, name):
    """ Plot solution as function of space for truth and each ensemble member."""
    sector = HMM.sectors[name]
    x = HMM.coordinates[sector]

    # Plot ensemble.

    e_mean = np.mean(E[:, sector], axis=0)
    e_std = np.std(E[:, sector], axis=0)
    for e in E:
        ax.plot(x*1e1, e[sector], 'k-', c="0.65")
    ax.plot(x*1e1, e_mean, 'k-', label=r'ensemble mean')
    ax.plot(x*1e1, e_mean+e_std, 'k--', label=r'mean+/-std')
    ax.plot(x*1e1, e_mean-e_std, 'k--')
    ax.plot(x*1e1, truth[sector], 'b-', label='truth')

    xlim = (np.floor(min(x)*1e1), np.ceil(max(x)*1e1))
    ax.set_xlim(xlim)
    ax.grid()

    return ax


def plot_for_values(t, axes=None, fig_name=None):
    """ Plot forecast as function of space for truth and each ensemble member."""
    if axes is None:
        plt.close('all')
        fig = plt.figure(figsize=(11, 8))
        fig.subplots_adjust(wspace=.35, hspace=None)
        axes = fig.subplots(1, len(names))
    else:
        for ax in axes:
            ax.cla()

    it = np.where(t == times)[0][0]
    for ax, name in zip(axes, names):
        ax = plot_ens_values(ax, xx[it, :], Efor[it, :, :], name[0])
        ax.set_ylabel('Forecast '+name[0]+' ['+name[1]+']')
        ax.set_xlabel('Position [km]')
        ax.set_title(time_str(times[it]))

    # Set legend
    plt.legend(loc='upper right')

    if fig_name is not None:
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=150, format='png')

    return axes


def plot_forana_values(t, axes=None, fig_name=None):
    """ Plot forecast and matching analysis as function of space for truth and each ensemble member."""
    if axes is None:
        plt.close('all')
        fig = plt.figure(figsize=(11, 8))
        fig.subplots_adjust(wspace=.35, hspace=None)
        axes = fig.subplots(2, len(names))
    else:
        for ax in np.reshape(axes, (-1)):
            ax.cla()

    it = np.where(t == times)[0][0]
    ito = np.where(t == timeos)[0][0]
    for ax_for, ax_ana, name in zip(axes[0], axes[1], names):
        ax_for = plot_ens_values(ax_for, xx[it, :], Efor[it, :, :], name[0])
        ax_for.set_ylabel('Forecast '+name[0]+' ['+name[1]+']')

        ax_ana = plot_ens_values(ax_ana, xx[it, :], Eana[ito, :, :], name[0])
        ax_ana.set_ylabel('Analysis '+name[0]+' ['+name[1]+']')

        ax_ana.set_xlabel('Position [km]')
        ax_for.set_title(time_str(times[it]))

        ax_for = plot_obs_space(ax_for, yy[ito],
                                xps[0].obs_db, name[0], times[it])

        ylims = np.array([ax_for.get_ylim(), ax_ana.get_ylim()])
        ax_for.set_ylim(min(ylims[:, 0]), max(ylims[:, 1]))
        ax_ana.set_ylim(min(ylims[:, 0]), max(ylims[:, 1]))

    # Set legend
    plt.legend(loc='upper right')

    if fig_name is not None:
        plt.savefig(os.path.join(fig_dir, fig_name), dpi=150, format='png')

    return axes


axes = None
if len(timeos) == 1:
    for it, t in enumerate(times[::6]):
        axes = plot_for_values(t, axes,
                               fig_name='for_values_{:03d}.png'.format(it))
else:
    for it, t in enumerate(timeos):
        axes = plot_forana_values(t, axes,
                                  fig_name='forana_values_{:03d}.png'.format(it))


# %% Rank histogram

names = [('velocity_ice', 'ms-1'), ('thickness_ice', 'm'), ('stress', 'Pa')]


def plot_rank(ax, truth, E, name):
    """ Plot histogram with ranks truth."""
    sector = HMM.sectors[name]
    r = ens_ranks(truth, E, sector)
    r = np.sum(r, axis=0)
    r = r/sum(r)

    bins = np.arange(0, np.size(E, 1)+1)
    ax.bar(bins, r)

    ax.plot(bins, np.ones_like(bins)*(1./(len(bins)-1)), 'k-')
    ax.grid()

    return ax


def plot_for_ranks(fig_name=None):
    """ Plot histogram with ranks truth in forecast ensembles."""
    plt.close('all')
    fig = plt.figure(figsize=(11, 8))
    fig.subplots_adjust(wspace=.35, hspace=None)
    axes = fig.subplots(1, len(names))

    for ax, name in zip(axes, names):
        ax = plot_rank(ax, xx, Efor, name[0])
        ax.set_xlabel("Rank")
        ax.set_title(name[0])

    axes[0].set_ylabel("Probability")

    if fig_name is not None:
        fig.savefig(os.path.join(fig_dir, fig_name), dpi=400, format='png')

    return axes


def plot_forana_ranks(fig_name=None):
    """ Plot histogram with ranks in forecast and analysis ensembles."""
    plt.close('all')
    fig = plt.figure(figsize=(11, 8))
    fig.subplots_adjust(wspace=.35, hspace=None)
    axes = fig.subplots(2, len(names))

    in_ana = [t1 in timeos for t1 in times]
    for ax_for, ax_ana, name in zip(axes[0], axes[1], names):
        ax_for = plot_rank(ax_for, xx[in_ana], Efor[in_ana], name[0])
        ax_ana = plot_rank(ax_ana, xx[in_ana], Eana, name[0])
        ax_ana.set_xlabel("Rank")
        ax_for.set_title(name[0])

    axes[0][0].set_ylabel("Probability forecast")
    axes[1][0].set_ylabel("Probability analysis")

    if fig_name is not None:
        fig.savefig(os.path.join(fig_dir, fig_name), dpi=400, format='png')

    return axes


fig_name = 'rank.png'
if len(timeos) == 1:
    plot_for_ranks(fig_name)
else:
    plot_forana_ranks(fig_name)

# %%

def calculate_inflation():    
    in_ana = [t in timeos for t in times]
    Efor1 = Efor[in_ana, :, :]

    inflators = [ObsInflator(), AnaInflator(), AdaptiveRTPP()]

    for it, t in enumerate(timeos):
        for inflator in inflators:
            inflator.update_for(HMM.Obs, Efor1[it], t, yy[it])
            inflator.update_ana(HMM.Obs, Eana[it], t, yy[it])

    return inflators

def plot_inflation(inflators, names):
    plt.close('all')
    fig = plt.figure(figsize=(8, 4))
    axes = fig.add_subplot(1, 1, 1)
    ax = axes


    for n, (inflator, name) in enumerate(zip(inflators, names)):
        n=int(n/len(names)*255)
        smoothed = np.convolve(inflator.filter.factors, np.ones((8,))/8., 
                               mode='same')
        plt.plot(plot_time(inflator.filter.times), inflator.filter.factors, '.',
                 color=plt.cm.jet(n))
        plt.plot(plot_time(inflator.filter.times), smoothed, '-',
                 label=name, color=plt.cm.jet(n))
    ax.grid()

    ax.set_xlabel('Time [days]')
    ax.set_ylabel('Inflation factor')
    ax.set_ylim(0.,10.)
    plt.legend(loc='upper right')

    return axes

# Calculate inflation factor
inflators = calculate_inflation()
plot_inflation(inflators, ['omb','amb','artpp'])