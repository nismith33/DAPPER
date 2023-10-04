#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:15:52 2022

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

font = {'family': 'normal',
        'weight': 'bold',
        'size': 12}
matplotlib.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['text.usetex'] = False

basename = 'porder_noda_p{:02d}'
data_dir = '/home/ivo/dpr_data/P12'
orders=np.arange(4,12,2)

# Copy file wit experiment setup
sys.path.append(data_dir)
shutil.copyfile(os.path.join(data_dir, basename.format(orders[0])+'.py'),
                os.path.join(data_dir, 'exp_setup.py'))

# Create directory to store figures.
fig_dir = os.path.join(data_dir, basename.format(orders[0]))
if not os.path.isdir(fig_dir):
    os.mkdir(fig_dir)

from exp_setup import create_exp_polyorder, aev_pnormal
HMM, xps = create_exp_polyorder()

def time_str(time):
    """ Convert time in minutes to time string."""
    from datetime import timedelta

    days = int(np.floor(time/24/60))
    time = time-days*24*60

    hours = int(time/60)
    time = time-hours*60

    return "{:d} days {:02d}:{:02d}".format(days, int(hours), int(time))

HMMs=[]
def plot_space(ax, time, name):
    
    for ip,order in enumerate(orders):
        if len(HMMs)<=ip:
            HMMs.append(aev_pnormal(order=order))
        isector = HMMs[ip].sectors[name[0]]
        x = HMMs[ip].coordinates[isector]
        
        nc = NetcdfIO(os.path.join(data_dir, (basename+'.nc').format(order)))
        with nc.ostream as stream:
            times = np.array(stream['time'][:])
            it = np.where(times==time)[0][0]
            Efor = np.array(stream['forecast'][it, :, isector])
        Efor = np.mean(Efor,axis=0)    
        
        ax.plot(x*10, Efor, label='P{:d}'.format(order)) 
        
        if name[0]=='velocity_ice':
            ax.set_ylim(-.12,.12)
        elif name[0]=='thickness_ice':
            ax.set_ylim(1.9,2.1)
        elif name[0]=='stress':
            ax.set_ylim(-1600.,1600.)
        
    ax.set_xlim(0.,100.)
    ax.set_ylabel(name[0]+' ['+name[1]+']',fontsize=font['size'])
    ax.grid()
    ax.legend(loc='upper right',framealpha=.6,ncol=2)
    return ax
    

def plot_backgrounds():
    names = [('velocity_ice', 'ms-1'), ('thickness_ice', 'm'), ('stress', 'Pa')]
    fig = plt.figure(figsize=(10.5,6))
    axes = fig.subplots(1,3)
    fig.subplots_adjust(wspace=.4, hspace=.02, top=.94, right=.98,left=.12)
    times = np.arange(0,24*60,3)*60.

    for it,time in enumerate(times):
        for ax,name in zip(axes,names):
            ax.cla()
            ax = plot_space(ax, time, name)
            ax.set_xlabel('Position [km]',fontsize=font['size'])
            ax.set_title(time_str(time))   
        
        
        file_path = os.path.join(fig_dir,'poly_values_{:03d}.png'.format(it))
        print(file_path)
        fig.savefig(file_path,format='png',dpi=400)
