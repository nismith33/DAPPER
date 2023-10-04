""" Run with pink background error noise and no superobing."""
import numpy as np
from dapper.tools.randvars import GaussRV
from dapper.tools.chronos import Chronology
from dapper.da_methods import LETKF, EnKF
import dill
import os
import pathlib
from dapper.mods.Liu2002 import exp_setup as exp
from dapper.tools.localization import nd_Id_localization, no_localization
import dapper.tools.localization as loc

Nens = 8
Ncells = exp.Ncells
fig_dir = os.path.join(exp.fig_dir,'loc_trial')
tseq = Chronology(dt=1, dto=1., T=1, BurnIn=0)

mpi_info = exp.create_comm()

def simulate(order, ncells, Nobs, sigma_obs):
    """ Run single instance of model."""
    
    rv_mean = GaussRV(C=np.ones((Nobs,)) * 1e-8, mu=0)
    rv_truth = GaussRV(C=np.ones((Nobs,)) * sigma_obs**2, mu=0 )
    #rv_smooth = GaussRV(C=np.ones((Nobs,)) *  sigma_obs**2 / ( Lo / Nobs_dx ), mu=0 )
    rv_smooth = GaussRV(C=np.ones((Nobs,)) *  sigma_obs**2, mu=0 )
    
    HMM = {}
    HMM['mean'] = exp.create_mean_model(rv_mean, tseq, K=200, 
                                        obs_type='single')
    HMM['truth'] = exp.create_truth_model(HMM['mean'], rv_truth, exp.Nens)
    #HMM['smooth'] = create_smooth_model(HMM['truth'], Lo, rv_smooth)
    #HMM['lin'] = exp.create_lin_model(HMM['truth'], 0, ncells, rv_smooth)
    HMM['dg'] = exp.create_dg_model(HMM['truth'], order, ncells, rv_smooth)
    
    xx, yy = {}, {}
    for key, HMM1 in HMM.items():
            
        if hasattr(HMM1,'noise'):
            HMM1.noise.member = Nens
            
        print('simulate ',key)
        xx[key], yy[key] = HMM1.simulate()
        
    return HMM, xx, yy

def assimilate(xp, HMM, xx, yy, radius, tag):
    """ Assimilate truth in other models. """
    
    batcher = loc.LegendreBatcher(HMM.model.dyn_coords, order)
    taperer = loc.LegendreTaperer(order)
    localizer = loc.Localizer(batcher, taperer, HMM.model.obs_coords)
    HMM.Obs.localization = localizer
    
    #Assimilate 
    Efor, Eana = {}, {}
    for key,(radius1, tag1) in enumerate(radius, tag):
        print('assimilate ', key)
        xp.loc_rad, xp.tag = radius1, tag1
        Efor[key], Eana[key] = xp.assimilate(HMM, xx, yy)
                
    return Efor, Eana
    
def exp_loc(order, Nobs, sigma_obs, radius, tag):
    print('Nobs, order, radius', Nobs, order, radius)
    
    HMM, xx, yy = simulate(order, Ncells, Nobs, 1)
    
    xp = LETKF(N=Nens, loc_rad=0.)
    Efor, Eana = assimilate(xp, HMM['dg'], xx, yy['truth'], 
                            radius, tag)
    
    rms = exp.stats_rmse(HMM, xx, Efor, Eana)
    spread = exp.stats_spread(HMM, Efor, Eana)
    
    return HMM, xx, yy, rms, spread, Efor, Eana

def copy_this_file(fname):
    from shutil import copy
    this_file = pathlib.Path(__file__)
    this_base = os.path.basename(fname)
    copied_file = os.path.join(fig_dir, os.path.splitext(this_base)[0] + '.py')
    copy(this_file, copied_file)

if __name__ == "__main__":
    
    for i in range(1):
        if np.mod(i, mpi_info['size'])==mpi_info['rank'] and True:
            order, Nob = 4, 1
            radius = np.array([4,2,1,1,1])*exp.L/Ncells 
            radius = radius[...,None]
            HMM, xx, yy, rms, spread, Efor, Eana = exp_loc(order, Nob, 1.,
                                                           radius)
            
            data = {'HMM':HMM, 'xx':xx, 'yy':yy, 'rms':rms, 'spread':spread,
                    'Efor':Efor, 'Eana':Eana}
            fname = "obs1_{:02d}_{:03d}.pkl".format(int(order), int(Nob))
            copy_this_file(fname)
            with open(os.path.join(fig_dir,fname), 'bw') as stream:
                dill.dump(data, stream)
    

    
    
    
    
    
