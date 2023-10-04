#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 08:38:07 2022

@author: ivo
"""

import numpy as np 
import matplotlib.pyplot as plt
import os, sys, shutil
from datetime import datetime, timedelta
from dapper.tools.datafiles import NetcdfIO

data_dir = '/home/ivo/dpr_data/obsdensity/p4'
fig_dir = data_dir
mod_dir = '/home/ivo/Code/DAPPER/dapper/mods/Ice'

basename = 'p4_refinement_OBSthickness_ice_NO016'

# Copy file wit experiment setup
sys.path.append(data_dir)
shutil.copyfile(os.path.join(data_dir, basename+'.py'),
                os.path.join(mod_dir, 'exp_setup.py'))

from dapper.mods.Ice.exp_setup import create_exp_refinement, aev_pnormal, obs_uniform
HMM, xps = create_exp_refinement()

def plt_truth(save_name=None):
    """ Plot velocity and thickness truth.""" 
    
    
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.subplots(1,2)
    fig.subplots_adjust(left=.2,right=.96, top=.9, bottom=.1, wspace=.4)
    
    savenc = NetcdfIO(os.path.join(data_dir,'truth.nc'))
    with savenc.ostream as nc:
        x = nc['xx'][-1,:]
        time = float(nc['time'][-1])
        
    
    time = datetime(2000,1,1) + timedelta(minutes=time)
    
    #velocity
    ind = HMM.sectors['velocity_ice']
    coord = HMM.coordinates[ind]
    
    ax1 = ax[0]
    ax1.set_title(time.strftime('%Y-%m-%d %H:%M'))
    ax1.plot(coord*10, x[ind],'k-')
    ax1.set_xlabel('Position [km]'); ax1.set_ylabel('velocity [ms-1]')
    ax1.set_xlim(0,100)
    ax1.grid()
    
    #thickness
    ind = HMM.sectors['thickness_ice']
    coord = HMM.coordinates[ind]
    
    ax1 = ax[1]
    ax1.set_title(time.strftime('%Y-%m-%d %H:%M'))
    ax1.plot(coord*10, x[ind],'k-')
    ax1.set_xlabel('Position [km]'); ax1.set_ylabel('thickness [m]')
    ax1.set_xlim(0,100)
    ax1.grid()
    
    if save_name is not None:
        fig_path = os.path.join(fig_dir,save_name)
        fig.savefig(fig_path, dpi=300, format='png')
        
    
def plt_truth_ens(save_name=None):
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.subplots(1,1)
    fig.subplots_adjust(left=.2,right=.96, top=.9, bottom=.1, wspace=.4)
    
    savenc = NetcdfIO(os.path.join(data_dir,'p4_refinement_OBSthickness_ice_NO016.nc'))
    with savenc.ostream as nc:
        Efor = nc['forecast'][-1,:,:]
        
        time = float(nc['time'][-1])
        timeos = nc['timeo'][:]
        ito=np.where(timeos==time)[0]
        
        yy = nc['yy'][ito,:]
        yy = yy[0]
         
    savenc = NetcdfIO(os.path.join(data_dir,'truth.nc'))
    with savenc.ostream as nc:
        x0 = nc['xx'][-1,:]
         
    model = aev_pnormal()
    obs = xps[0].obs_db
    obs = obs[obs['time']==time]
        
    time = datetime(2000,1,1) + timedelta(minutes=time)
    
    #thickness ensemble
    ind = model.sectors['thickness_ice']
    coord = model.coordinates[ind]
    
    for n in range(np.size(Efor,0)):
        h0, = plt.plot(coord*10., Efor[n,ind],'-',color=(.8,.8,.8), 
                       label='ensemble', linewidth=.5)
    
    #thickness truth
    ind = HMM.sectors['thickness_ice']
    coord = HMM.coordinates[ind]
    
    ax1 = ax
    ax1.set_title(time.strftime('%Y-%m-%d %H:%M'))
    h1, = ax1.plot(coord*10, x0[ind],'b-',label='truth')
    ax1.set_xlabel('Position [km]'); ax1.set_ylabel('thickness [m]')
    ax1.set_xlim(0,100)
    ax1.grid()
    
    h2=ax1.errorbar(np.reshape(obs['coordinates'],(-1))*10., yy, 
                 yerr=np.sqrt(obs['var']), linewidth=0.0, elinewidth=2., 
                 ecolor='r', label='obs')
    
    plt.legend([h0,h1,h2],['ensemble','truth','obs'], framealpha=1.)
    
    if save_name is not None:
        fig_path = os.path.join(fig_dir,save_name)
        fig.savefig(fig_path, dpi=300, format='png')
        print(fig_path)
    
    

       
    