#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:59:19 2022

File to create plot of RMSE as function of observation density. 

@author: ivo
"""

import os
import shutil
import sys
import numpy as np
import netCDF4 as netcdf
from dapper.tools.datafiles import NetcdfIO
from dapper.stats import Stats
from dapper.stats import ens_mean, ens_spread, ens_rmse, ens_ranks
import matplotlib.pyplot as plt
import matplotlib
#from dapper.stats import window

font = {'family': 'normal',
        'weight': 'bold',
        'size': 12}
matplotlib.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['text.usetex'] = False

basename = 'p4_obssig_OBSstress_SIG1.00e+02'
data_dir = '/home/ivo/dpr_data/obssig_stress'
mod_dir = '/home/ivo/Code/DAPPER/dapper/mods/Ice'

# Copy file wit experiment setup
sys.path.append(data_dir)
shutil.copyfile(os.path.join(data_dir, basename+'.py'),
                os.path.join(mod_dir, 'exp_setup.py'))

# Create directory to store figures.
fig_dir = os.path.join(data_dir, basename)
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)

# Import the experiment.
from dapper.mods.Ice.exp_setup import create_exp_obssig, aev_pnormal
HMM, xps = create_exp_obssig()

def inner_l2(name, x1, x2):
    """Calculate the L2 inner product averaged over space."""
    isector = HMM.sectors[name]
    x1, x2 = x1[isector], x2[isector]

    one = np.ones_like(x1)
    denominator = HMM.model.inner_l2(name, one, one)
    nominator = HMM.model.inner_l2(name, x1, x2)
    return nominator/denominator

# Function to calculate RMSE


def calc_rmse(name, ens, truth):
    """Calculate RMSE over the whole period for field for 1 time."""
    error = np.mean(ens, axis=0) - truth
    return np.sqrt(inner_l2(name, error, error))


def calc_rmse_time(xp):
    """Calculate RMSE over the whole period."""

    # Read file.
    xx, _ = xp.save_nc.read_truth()
    with xp.save_nc.ostream as stream:
        times = np.array(stream['time'][:])
        timeos = np.array(stream['timeo'][:])
        Efor = np.array(stream['forecast'][:, :, :])
        Eana = np.array(stream['analysis'][:, :, :])

    # Generate dictionaries to store RMSEs
    rmse_for, rmse_ana = {}, {}
    for sector in HMM.sectors:
        rmse_for['time'] = timeos
        rmse_ana['time'] = timeos
        rmse_for[sector] = []
        rmse_ana[sector] = []
        
    in_ana = [t in timeos for t in times]

    # Calculate RMSE forecast
    for E, x in zip(Efor[in_ana], xx[in_ana]):
        for sector in HMM.sectors:
            rmse_for[sector].append(calc_rmse(sector, E, x))

    # Calculate RMSE analysis
    for E, x in zip(Eana, xx[in_ana]):
        for sector in HMM.sectors:
           rmse_ana[sector].append(calc_rmse(sector, E, x))

    # Convert
    for sector in HMM.sectors:
        rmse_for[sector] = np.array(rmse_for[sector])
        rmse_ana[sector] = np.array(rmse_ana[sector])

    return rmse_for, rmse_ana


def plt_rmse_Nobs_field(xps, axes, names):
    """Calculate time-averaged RMSE."""
    n_cells = 16.

    # Calcalute obs density
    n_obs, rmse_for, rmse_ana = [], {}, {}
    for xp in xps:
        db = xp.obs_db
        n_obs.append(np.sum(db['time'] == min(db['time']))/n_cells)
        print('calculate ', n_obs[-1])

        rmse_for1, rmse_ana1 = calc_rmse_time(xp)
        for name in names:
            rmse_for['time'] = rmse_for1['time']
            rmse_ana['time'] = rmse_ana1['time']

            if name not in rmse_for:
                rmse_for[name] = []
                rmse_ana[name] = []
            else:
                rmse_for[name].append(np.sqrt(np.mean(rmse_for1[name]**2)))
                rmse_ana[name].append(np.sqrt(np.mean(rmse_ana1[name]**2)))

    # Convert to numpy arrays
    for name in names:
        rmse_for[name] = np.array(rmse_for[name])
        rmse_ana[name] = np.array(rmse_ana[name])

    # Calculate rmse.
    for ax, name in zip(axes, names):
        unit = HMM.model.metas[name].unit
        unit = unit.as_si.as_str

        ax.plot(n_obs, rmse_for[name], 'k-',
                alpha=0.3, linewidth=4.,
                label='Forecast')
        ax.plot(n_obs, rmse_ana[name], 'k-',
                alpha=1.0, linewidth=2.,
                label='Analysis')

        ax.legend(loc='upper right')
        ax.set_xlim(0., 6.)
        ax.set_title('RMSE '+name+' ['+unit+']', fontsize=font['size'])
        ax.grid()

        ax.set_xlabel('Observation density', fontsize=font['size'])


def plt_rmse_Nobs(save_fig=False):
    """Plot dependence RMSE on observation density."""
    plt.close('all')
    fig = plt.figure(figsize=(12, 6))
    fig.subplots_adjust(wspace=.4, hspace=None, left=.12, right=.98)
    axes = fig.subplots(1, 3)
    names = ['velocity_ice', 'thickness_ice', 'stress']
    plt_rmse_Nobs_field(xps, axes, names)

    if save_fig:
        fig_path = os.path.join(fig_dir, 'rmse_Nobs.png')
        fig.savefig(fig_path, dpi=400, format='png')
        print('saved ', fig_path)
