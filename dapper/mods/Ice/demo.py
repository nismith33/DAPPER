"""Reproduce results from `bib.sakov2008deterministic`."""

import numpy as np

from dapper import xpList
import dapper
import dapper.mods as modelling
import dapper.da_methods as da
from dapper.mods.Ice import AdvElastoViscousModel
from dapper.tools.localization import nd_Id_localization
import dapper.tools.liveplotting as LP
import datetime
import matplotlib.pyplot as plt
from pyIce.units import Quantity, SiUnit
from pyIce.function_algebras import FunctionField
from pyIce.fields import MetaField
import dapper.mods.Ice.forcings as forcings
from copy import deepcopy 
import os
from dapper.tools import viz
#from mpi4py import MPI as mpi
from multiprocessing import Queue, Process, Barrier
from dapper.tools.datafiles import SaveXP

#%% mpi settings

#comm = mpi.COMM_WORLD
#rank = comm.Get_rank()
#mp = comm.Get_size()
#mp = None

#%% Setup model

def default_wind():
    return {'base_velocity':1./86400.*13/11.,
            'base_amplification':20.,
            'T_ramp':datetime.timedelta(hours=0.5)
            }

def default_noise_wind(seed=4000):
    return {'drawer':forcings.TimeDrawer(seed, size_storage=200),
            'velocity':0.1,'amplification':0.5,'signal':0.0}

default_noise_init = {'velocity_ice': Quantity(.0, 'ms-1'),
                      'thickness_ice': Quantity(0.0, 'm')}
default_seed = 2000

def aev_pnormal(xp=None):
    #Set the wind forcing noise parameters. 
    if hasattr(xp, 'noise_wind'):
        noise_wind = xp.noise_wind 
    else:
        noise_wind = []
        for _ in range(0,1):
            noise_wind.append(default_noise_wind())
     
    #Set initial conditions. 
    noise_init = deepcopy(default_noise_init)
    if hasattr(xp, 'noise_init'):
        for key,value in xp.noise_init.items():
            noise_init[key]=value
    
    #Build ice model.
    model = AdvElastoViscousModel(datetime.timedelta(seconds=30),
                                  4, 16, 5.)

    #Set magnitude wind forcing in m/s. Default magnitude is 0.
    velocity_unit=SiUnit('ms-1')
    T_ramp=datetime.timedelta(hours=0.5)
    
    wind_forcing = []
    for noise in noise_wind:
        wind=forcings.MovingWave(model, forcings.bellcurve_wind,
                                 **default_wind())
        wind.set_noise(**noise)
        wind_forcing.append(wind)
    
    model.forcing=forcings.EnsembleForcing(wind_forcing)
    model.velocity_air=MetaField((1,), model.forcing, name='velocity_air', 
                                 unit=SiUnit('ms-1')) 
    
    
    #Build model fields, timesteppers and runner.
    mpi = model.build_model()
    
    #Create unperturbed initial conditions.
    x0 = model.build_initial(datetime.datetime(2000, 1, 1))
    x0 = model.meta2si_units(x0)    
    
    #Set magnitude initial perturbations.
    var_init = np.zeros((model.M,))
    for key, value in noise_init.items():
        meta = model.metas[key]        
        std = float(value/meta.unit.as_si)
        var_init[model.indices[meta]] = std**2

    #Link DAPPER dynamical model to ice model. 
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': 0.,
           'linear': model.step
           }
    
    #part of model
    sectors={}
    coordinates = np.array([],dtype=float)
    for name,meta in model.metas.items():
        sectors[name]=model.indices[meta]
        coordinates=np.append(coordinates, model.coordinates[meta])

    #Create observation operator
    jj = np.arange(min(sectors['thickness_ice'])+2,
                   max(sectors['thickness_ice']),5)
    Obs = modelling.partial_Id_Obs(model.M, jj)
    Obs['noise'] = .01**2 #m2
    # modelling.GaussRV(C=CovMat(2*eye(Nx)))

    #Create time sequence. 
    T = 2*60 #minutes
    Burn =  0*60 #minutes
    tseq = modelling.Chronology(30, dto=T, T=T, Tplot=T, BurnIn=Burn)

    #Create operator for generating perturbed initial conditions.  
    X0 = modelling.GaussRV(C=var_init, mu=x0)
    
    #Create DAPPER model. 
    hmm = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, sectors=sectors)

    #Add plotters. 
    LPs = [(1, LP.sliding_diagnostics),
           (1, LP.sliding_marginals(Obs, zoomy=0.8, dims=[37,277])),
           (1, LP.spatial1d(obs_inds=jj, dims=sectors['thickness_ice'])),
           (1, LP.spatial1d(obs_inds=jj, dims=sectors['velocity_ice']))]
    hmm.liveplotters = LPs
    
    hmm.model = model
    hmm.coordinates = coordinates
    hmm.mpi = mpi
    return hmm

#%%

save_dir = '/home/ivo/dpr_data/mpi_test'

def run_truth():
    HMM = aev_pnormal()
    dapper.set_seed(default_seed)
    xx, yy = HMM.simulate()
    return HMM, xx, yy

def xp_wind(xps, xp_name):    
    xps += da.EnKF('Sqrt', N=2, infl=1.00, rot=True)
    
    xps[-1].save_xp=SaveXP(os.path.join(save_dir, xp_name+'.pkl'))
    xps[-1].noise_init = default_noise_init
    xps[-1].noise_wind = [default_noise_wind(5000+n*100) for n in range(0,xps[-1].N)]
    xps[-1].seed = default_seed + 1000

    return xps

def xps_run(xps, xx, yy):
    
    for xp in xps:
        HMM = aev_pnormal(xp)
        xp.mpi = HMM.mpi
        xp.assimilate(HMM, xx, yy, liveplots=False)
        
        if xp.mpi.is_root:
            xp.stats.average_in_time()
            
    return xps

if __name__ == '__main__' and True:
    xps = xpList()
    xps = xp_wind(xps,'wind_noda')
    
    HMM, xx, yy = run_truth()
    xps = xps_run(xps, xx, yy)
    

#%% Plot functions 


def viz_plot(HMM, xp):
    print(xp.avrgs)
    
    viz.plot_rank_histogram(xp.stats)
    viz.plot_err_components(xp.stats, sectors=HMM.sectors['thickness_ice'])
    viz.plot_err_components(xp.stats, sectors=HMM.sectors['velocity_ice'])
    #viz.plot_hovmoller(xx, sectors=HMM.sectors['thickness_ice'])

def plot_wind(xp, save_fig=False):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        
    HMM = aev_pnormal(xp)
    
    t=[datetime.datetime(2000,1,1)+n*datetime.timedelta(hours=1) for n in np.arange(0,3*24+1)]
    x=HMM.coordinates[HMM.sectors['velocity_ice']]
    
    forces = []
    velocity_unit=SiUnit('ms-1')
    T_ramp=datetime.timedelta(hours=0.5)
    for noise in [default_noise_wind(5000+n*100) for n in range(0,100)]:
        wind=forcings.MovingWave(HMM.model, forcings.bellcurve_wind,
                                  base_velocity=2.5/86400, base_amplification=20.,
                                  T_ramp=T_ramp)
        wind.set_noise(**noise)
        forces.append(wind)
    
    wind=forcings.EnsembleForcing(forces)
    
    N=12
    pmax=np.zeros((N,2))
    for it,t1 in enumerate(t[:3]):
        fig_path = os.path.join(save_dir,'wind_{:03}.png'.format(it))
        plt.close('all')
        plt.figure()
        plt.title('{}'.format(t1.strftime('%Y-%m-%d %H:%M:%S')))
        plt.ylabel('wind [ms-1]'); plt.xlabel('position [km]')
        plt.xlim((0,50)); plt.ylim((0,24))
        plt.grid(axis='both')
        
        ymax1=0.
        ymax2=0
        cmax=0.
        
        for member in range(0,N):
            wind.n = member
            y=wind(t1,x)
            
            ymax1+=max(y)/N
            ymax2+=max(y)**2/N
            cmax+=pmax[member,0]*(max(y)-10.)/N
            pmax[member,1]=pmax[member,0]
            pmax[member,0]=max(y)-10.
            
            plt.plot(x*10.,y)
            
        if save_fig:
            plt.savefig(fig_path, dpi=150, format='png')
            
            
    


