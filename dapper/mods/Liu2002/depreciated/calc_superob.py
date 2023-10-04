"""
Use superob in the assimilation.
"""
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
Nobs_dx = exp.Nobs_dx
fig_dir = os.path.join(exp.fig_dir,'superob_red15')
tseq = Chronology(dt=1, dto=1., T=20, BurnIn=0)
sigma_obs = 1.
    
mpi_info = exp.create_comm()

def simulate(order, Lo, spectrum):
    """ Run single instance of model."""
    
    #Number of observations per superob
    Navg = Lo / Nobs_dx
    #Number of observations after superobing
    Nobs = int( (exp.L / Nobs_dx) / Navg )
    rv_mean = GaussRV(C=np.ones((Nobs,)) * 1e-8, mu=0)
    rv_truth = GaussRV(C=np.ones((Nobs,)) * sigma_obs**2 , mu=0)
    rv_smooth = GaussRV(C=np.ones((Nobs,)) *  (sigma_obs**2 / Navg), mu=0 )
    
    HMM = {}
    HMM['mean'] = exp.create_mean_model(rv_mean, tseq, obs_type='uniform')
    HMM['truth'] = exp.create_truth_model(HMM['mean'], rv_truth, exp.Nens, 
                                          spectrum)
    HMM['smooth'] = exp.create_smooth_model(HMM['truth'], Lo, rv_smooth)
    HMM['lin'] = exp.create_lin_model(HMM['truth'], 0, Ncells, rv_smooth)
    HMM['dg'] = exp.create_dg_model(HMM['truth'], order, Ncells, rv_smooth)
    
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
        Efor[key], Eana[key] = xp.assimilate(HMM[key], xx[key], yy['smooth'])
                
    return Efor, Eana
            
def exp_superob(order, Lo, spectrum):
    print('order, Lo, spectrum', order, Lo, spectrum)
    
    HMM, xx, yy = simulate(order, Lo, spectrum)
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
    dx = exp.L / 79
    orders, Los, slopes = np.meshgrid(np.array([0,1,2,4,6,8]),
                              np.array([.2,.4,.6,.8,1.,1.2,1.5,2.])*dx,
                              np.array([-1.5,-2.,-2.5,-3.,-3.5]))
    orders, Los, slopes = np.reshape(orders,(-1)), np.reshape(Los,(-1)), np.reshape(slopes,(-1))
    
    for i in range(len(orders)):
        if np.mod(i, mpi_info['size'])==mpi_info['rank'] and True:
            order, Lo, slope = orders[i], Los[i], slopes[i]
            HMM, xx, yy, rms, spread, Efor, Eana = exp_superob(order, Lo, slope)
            
            data = {'HMM':HMM, 'xx':xx, 'yy':yy, 'rms':rms, 'spread':spread,
                    'Efor':Efor, 'Eana':Eana}
            fname = "exp_{:02d}_{:03d}_{:02d}.pkl".format(int(order), int(Lo*1e-3), int(-10*slope))
            copy_this_file(fname)
            with open(os.path.join(fig_dir,fname), 'bw') as stream:
                dill.dump(data, stream)
    

    
    
    
    
    
