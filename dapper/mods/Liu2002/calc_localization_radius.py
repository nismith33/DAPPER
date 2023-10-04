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

#Number of ensemble members 
Nens = 8
#DG order
order = 4
#Directory
fig_dir = os.path.join(exp.fig_dir, 'loc_trial')


def lin_localizer(model):
    batcher = loc.SingleBatcher(model.dyn_coords)
    coorder = loc.FunctionCoorder(model.obs_coords)
    taperer = loc.DistanceTaperer()
    return loc.Localizer(batcher, taperer, coorder)

def dg_localizer(model):
    batcher = loc.LegendreBatcher(model.dyn_coords, order)
    coorder = loc.FunctionCoorder(model.obs_coords)
    taperer = loc.LegendreTaperer(order)
    return loc.Localizer(batcher, taperer, coorder)

def exp_localization_radius(T=1, Nens=8):
    tseq = Chronology(dt=1, dto=1, T=T)
    #Background
    tree = exp.MeanTreeNode(None, tseq)
    #Truth 
    tree.add_truth()
    #Observation
    tree.add_obs(exp.Ncells*5, .01)
    #Model    
    tree.add_model('dg', exp.Ncells, order=order, name='dg', 
                   localization=dg_localizer)
    #Experiment
    dx = exp.L/exp.Ncells
    taper = ['Gauss','Gauss','Step','Step','Step']
    for R in np.array([4,3,2,1,.5]):
        name = 'r_loc={:.1f}'.format(R)
        
        rads = np.array([4,max(1,R),R,R,R])*dx
        rads = rads[:,None]
        
        tree.add_xp(LETKF(Nens, loc_rad=rads, taper=taper), name=name)
    
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
    
    save()
    return tree
    
tree = exp_localization_radius()
tree.execute()
models = exp.collect_nodes(tree,3)
xps = exp.collect_nodes(tree,4)

def quick_plot(xps):
    plt.close('all')
    fig = plt.figure(figsize=(10,6))
    ax = fig.subplots(1,1)
    fig.subplots_adjust(right=.75)
    
    lax = fig.add_axes((.8,.12,.18,.8))
    lax.axis('off')
    
    r = np.reshape(np.linspace(0, exp.L, 1000, endpoint=False), (-1,1))
    drawings = []
    
    #Plot truth 
    interp=xps[0].true_HMM.model.interpolator(xps[0].xx[-1:])
    truth = interp(r)
    drawing, = ax.plot(r[:,0]*1e-3, truth[0], 'k-',label='truth')
    drawings.append(drawing)
    
    ry=xps[0].HMM.model.obs_coords(1)
    drawing, = ax.plot(ry*1e-3, xps[0].yy[-1],'bo',markersize=4,label='obs')
    drawings.append(drawing)
    
    ax.grid()
    ax.set_xlabel('Position [km]')
    ax.set_ylabel('Signal')
    ax.set_xlim(0,exp.L*1e-3)
    
    for xp in xps:
    
        #Plot analysis
        e = np.mean(xp.E[1][-1], axis=0, keepdims=True)
        interp=xp.HMM.model.interpolator(e)
        model = interp(r)
        drawing, = ax.plot(r[:,0]*1e-3, model[0], label=xp.name)
        drawings.append(drawing)
        
    lax.legend(handles=drawings, loc='lower left',ncol=1)
    
    return fig,ax
    
fig, ax = quick_plot(xps)
fig.savefig(os.path.join(fig_dir,'dg_loc_sig0_closeup.png'), format='png', dpi=400)
    