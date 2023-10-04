#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment testing localization width. 
"""

import numpy as np 
from dapper.tools.chronos import Chronology
from dapper.mods.Liu2002 import exp_setup as exp
from dapper.da_methods import LETKF, EnKF
import dapper.tools.localization as loc
import matplotlib.pyplot as plt
import os, dill
from datetime import timedelta, datetime

#Directory
fig_dir = os.path.join(exp.fig_dir, 'spectral_Nobs')
io = exp.TreeIO(fig_dir)

def exp_spectral_Nobs(T=10, Nens=40, sig_obs=1.0):
    tseq = Chronology(dt=1, dto=1, T=T)
    #Background
    tree = exp.MeanTreeNode(None, tseq, io=io)
    #Truth 
    for slope in np.arange(-.5,-4.5,-.5): #-4.5
        tree.add_truth(slope=slope, name='slope{:02d}'.format(int(-10*slope)))
    #Observation
    for sig in np.array([1.]):
        for Nobs in np.arange(1,8,1) * exp.Ncells: #8
            tree.add_obs(Nobs, sig, obs_type='uniform',
                         name='No{:03d}'.format(Nobs))
    #Model    
    tree.add_model('lin', exp.Ncells, name='lin')
    for order in np.array([1,2,4,6,8]): #8
        tree.add_model('dg', exp.Ncells, order=order, 
                       name='DG{:02d}'.format(order))
    #Experiment
    dx = exp.L/exp.Ncells
    tree.add_xp(EnKF,Nens,upd_a='Sqrt')
    
    #Save this file
    def save():
        from shutil import copy
        import pathlib
        this_file = pathlib.Path(__file__)
        this_base = os.path.basename(this_file)
        copied_file = os.path.join(fig_dir, this_base)
        copy(this_file, copied_file)
        
        this_base = os.path.splitext(this_base)[0]
        save_file = os.path.join(fig_dir,this_base+'.pkl')
        with open(save_file,'wb') as stream:
            dill.dump(tree, stream)    
    
    #save()
    return tree
    
def quick_plot(xps):
    plt.close('all')
    fig = plt.figure(figsize=(10,6))
    ax = fig.subplots(1,1)
    fig.subplots_adjust(right=.75)
    
    lax = fig.add_axes((.8,.12,.18,.8))
    lax.axis('off')
    
    r = np.reshape(np.linspace(0, exp.L, 1000, endpoint=False), (-1,1))
    drawings = []
    
    ax.grid()
    ax.set_xlabel('Position [km]')
    ax.set_ylabel('Signal')
    ax.set_xlim(0, exp.L*1e-3)
    
    for nxp,xp in enumerate(xps):
        import matplotlib.colors as mcolors                        
        cmap=list(mcolors.TABLEAU_COLORS.values())
        color = cmap[np.mod(nxp, len(cmap))]
        
        print('xp {:d}: {}'.format(nxp, xp.name))
        
        #Plot truth 
        interp=xp.true_HMM.model.interpolator(xps[0].xx[-1:])
        truth = interp(r)
        drawing, = ax.plot(r[:,0]*1e-3, truth[0], '-', color=color,
                           label='truth')
        drawings.append(drawing)
        
        #Plot observation
        ry=xp.HMM.model.obs_coords(1)
        drawing, = ax.plot(ry*1e-3, xp.yy[-1],'o', markersize=3, 
                           color=color, label='obs')
        drawings.append(drawing)
    
        #Plot analysis
        e = np.mean(xp.E['ana'][-1], axis=0, keepdims=True)
        interp=xp.HMM.model.interpolator(e)
        model = interp(r)
        drawing, = ax.plot(r[:,0]*1e-3, model[0], '--', 
                           color=color, label='analysis')
        drawings.append(drawing)
        
    lax.legend(handles=drawings[:3], loc='lower left', ncol=1)
    
    return fig, ax

def run():
    t_start = datetime.now()
    
    tree = exp_spectral_Nobs()
    comm = exp.MpiComm()
    tree.execute(comm)
    
    t_end = datetime.now()
    if comm.rank==0:
        print("Clock time",(t_end-t_start))
        
    return tree
    
   
if __name__ == '__main__': 
    tree=run()
        
    truths = exp.collect(tree,1)
    models = exp.collect(tree,3)
    xps = exp.collect(tree,4)
    #fig, ax = quick_plot(xps)
    #fig.savefig(os.path.join(fig_dir,'spectral_Nobs.png'), format='png', dpi=400)
    
    
    