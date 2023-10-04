""" Run with pink background error noise and no superobing."""
import numpy as np
from dapper.tools.randvars import GaussRV
from dapper.tools.chronos import Chronology
from dapper.da_methods import EnKF
import dill
import os
import pathlib
from dapper.mods.Liu2002 import exp_setup as exp

Nens = exp.Nens
Ncells = exp.Ncells
fig_dir = os.path.join(exp.fig_dir,'pink_noise_land2')
tseq = Chronology(dt=1, dto=1., T=20, BurnIn=0)

mpi_info = exp.create_comm()

def simulate(order, ncells, Nobs, sigma_obs):
    """ Run single instance of model."""
    
    rv_mean = GaussRV(C=np.ones((Nobs,)) * 1e-8, mu=0)
    rv_truth = GaussRV(C=np.ones((Nobs,)) * sigma_obs**2, mu=0 )
    #rv_smooth = GaussRV(C=np.ones((Nobs,)) *  sigma_obs**2 / ( Lo / Nobs_dx ), mu=0 )
    rv_smooth = GaussRV(C=np.ones((Nobs,)) *  sigma_obs**2, mu=0 )
    
    HMM = {}
    HMM['mean'] = exp.create_mean_model(rv_mean, exp.tseq, K=200, 
                                        obs_type='land_mask')
    HMM['truth'] = exp.create_truth_model(HMM['mean'], rv_truth, exp.Nens)
    #HMM['smooth'] = create_smooth_model(HMM['truth'], Lo, rv_smooth)
    HMM['lin'] = exp.create_lin_model(HMM['truth'], 0, ncells, rv_smooth)
    HMM['dg'] = exp.create_dg_model(HMM['truth'], order, ncells, rv_smooth)
    
    xx, yy = {}, {}
    for key, HMM1 in HMM.items():
        if hasattr(HMM1,'noise'):
            HMM1.noise.member = Nens
        xx[key], yy[key] = HMM1.simulate()
        
    return HMM, xx, yy

def assimilate(xp, HMM, xx, yy):
    """ Assimilate truth in other models. """
    
    #Assimilate 
    Efor, Eana, xfor, xana = {}, {}, {}, {}
    for key in HMM:
        Efor[key], Eana[key] = xp.assimilate(HMM[key], xx[key], yy['truth'])
                
    return Efor, Eana

def add_model_error(HMM, xx, name):
    """Add model error to observation operator."""
    y_smooth = np.array(grid(HMM['smooth'], xx['smooth'])[0])
    y_model = np.array(grid(HMM[name], xx[name])[0])
    var_model = np.mean( (y_model - y_smooth)**2 )
    
    C = HMM[name].Obs.noise.C.diag + var_model
    HMM[name].Obs.noise = GaussRV(C=C)
    
    return HMM
    
def exp_mask(order, Nobs, sigma_obs):
    #ncells = int((2*39+1)/(order+1))
    print('Nobs, order, cell ', Nobs, order, Ncells)
    
    HMM, xx, yy = simulate(order, Ncells, Nobs, 1)
    #HMM = add_model_error(HMM, xx, 'lin')
    #HMM = add_model_error(HMM, xx, 'dg')
    
    xp = EnKF('Sqrt', Nens)
    Efor, Eana = assimilate(xp, HMM, xx, yy)
    
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
    orders, Nobs = np.meshgrid(np.array([0,1,2,4,6,8]),
                               np.array([1,2,3,4,5,6,7,8])*Ncells)
    orders, Nobs = np.reshape(orders,(-1)), np.reshape(Nobs,(-1))
    
    for i in range(len(orders)):
        if np.mod(i, mpi_info['size'])==mpi_info['rank'] and True:
            order, Nob = orders[i], Nobs[i]
            HMM, xx, yy, rms, spread, Efor, Eana = exp_mask(order, Nob, 1.)
            
            data = {'HMM':HMM, 'xx':xx, 'yy':yy, 'rms':rms, 'spread':spread,
                    'Efor':Efor, 'Eana':Eana}
            fname = "land_red15_{:02d}_{:03d}.pkl".format(int(order), int(Nob))
            copy_this_file(fname)
            with open(os.path.join(fig_dir,fname), 'bw') as stream:
                dill.dump(data, stream)
    

    
    
    
    
    
