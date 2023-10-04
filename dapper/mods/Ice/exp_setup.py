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
import pyIce.units
from pyIce.function_algebras import FunctionField
from pyIce.fields import MetaField
import dapper.mods.Ice.forcings as forcings
from copy import deepcopy
import os
from dapper.tools import viz
from multiprocessing import Queue, Process, Barrier
from dapper.tools.datafiles import SaveXP, NetcdfIO
from dapper.mods.Ice.forcings import AR0, AR1, SpectralNoise
from datetime import datetime, timedelta
import dapper.tools.randvars as randvars

# %% Setup model

length_domain = 10.0 #x10km
default_seed = 2000

# Create time sequenc
T = 16*24*60  # minutes
Burn = 0*60  # minutes
tseq = modelling.Chronology(60., dto=60., T=T, Tplot=T, BurnIn=Burn)

def default_wind():
    return {'base_velocity': 1.1/86400.,
            'base_amplification': 15.,
            'T_ramp': timedelta(hours=0.5)
            }

def default_noise_wind(seed=default_seed):
    def func(dt):
        sig2, T = 5e-2, timedelta(minutes=10) #was 5e-3
        amplification = AR1(dt, 0., sig2, T, seed+100)

        sig2, T = 1e-4, timedelta(hours=12)
        center = AR1(dt, 0., sig2, T, seed+200)
        
        return {'amplification': amplification, 'center': center}

    return func

def spectral_noise_wind(amp_spectral=0.2, seed=default_seed):
    def func(dt):    
        #Amplitude storm
        sig2, T = 5e-2, timedelta(minutes=10) #was 5e-3
        amplification = AR1(dt, 0., sig2, T, seed+100)
        
        #Centre storm
        sig2, T = 1e-4, timedelta(hours=12)
        center = AR1(dt, 0., sig2, T, seed+200)
        
        #Small noise
        wavelengths, amplitudes = red_noise(-1.5, length_domain/16.)

        c_var= np.pi**2 / dt.total_seconds()
        ar = [AR1(dt, 0., c_var, T, seed+10*int(no)) for no,_ in enumerate(amplitudes)]
    
        spectral=SpectralNoise(ar, wavelengths, amp_spectral * amplitudes)

        return {'amplification': amplification, 'center': center,
                'spectral':spectral}

    return func

def red_noise(slope, max_wavelength, cutoff=0.01):
    wavelengths = length_domain / np.arange(1,1000)
    wavelengths = wavelengths[wavelengths<=max_wavelength]
    
    amplitudes  = wavelengths ** (-slope)
    wavelengths = wavelengths[amplitudes >= cutoff * np.max(amplitudes)]
    amplitudes  = amplitudes[amplitudes >= cutoff * np.max(amplitudes)]
    
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    
    return wavelengths, amplitudes


default_noise_init = {'velocity_ice': Quantity(0.0, 'ms-1'), #0.01 ms-1
                      'thickness_ice': Quantity(0.05, 'm')} #0.1 m


def default_obs():
    db_type = randvars.point_obs_type(1) 
    
    x_obs = np.arange(0.5,16.5)/16*5.0
    t_obs = np.reshape(tseq.tto,(1,-1))
    x_obs, t_obs = np.meshgrid(x_obs, t_obs)
    
    db_h = np.zeros((np.size(x_obs),), dtype=db_type)
    db_h['coordinates'] = np.reshape(x_obs, (-1,1))
    db_h['time'][:] = np.reshape(t_obs, (-1))
    db_h['field_name'][:] = 'thickness_ice'
    db_h['var'] = 3e-3**2
    
    db_v = np.zeros((np.size(x_obs),), dtype=db_type)
    db_v['coordinates'] = np.reshape(x_obs, (-1,1))
    db_v['time'][:] = np.reshape(t_obs, (-1))
    db_v['field_name'][:] = 'velocity_ice'
    db_v['var'] = 5.0e-3**2
    
    return db_h

def aev_pnormal(xp=None, obs_db=None, order=4):
    if hasattr(xp,'poly_order'):
        order = xp.poly_order 
    
    # Build ice model.
    model = AdvElastoViscousModel(timedelta(seconds=30),
                                  order, 16, length_domain)

    # Set the wind forcing noise parameters.
    if hasattr(xp, 'noise_wind'):
        noise_wind = [noise_wind(model.dt) for noise_wind in xp.noise_wind]
    else:
        noise_wind = []
        for _ in range(0, 1):
            noise_wind.append(default_noise_wind()(model.dt))

    # Set initial conditions.
    noise_init = deepcopy(default_noise_init)
    if hasattr(xp, 'noise_init'):
        for key, value in xp.noise_init.items():
            noise_init[key] = value

    # Set magnitude wind forcing in m/s. Default magnitude is 0.
    velocity_unit = SiUnit('ms-1')
    T_ramp = timedelta(hours=0.5)

    wind_forcing = []
    for noise in noise_wind:
        wind = forcings.MovingWave(model, forcings.bellcurve_wind,
                                   **default_wind())
        wind.set_noise(**noise)
        wind_forcing.append(wind)

    model.forcings = [forcings.EnsembleForcing(wind_forcing)]
    model.velocity_air = MetaField((1,), model.forcings[0], name='velocity_air',
                                   unit=SiUnit('ms-1'))

    # Build model fields, timesteppers and runner.
    mpi = model.build_model()

    # Create unperturbed initial conditions.
    x0 = model.build_initial(datetime(2000, 1, 1))
    x0 = model.meta2si_units(x0)

    # Set magnitude initial perturbations.
    var_init = np.zeros((model.M,))
    for key, value in noise_init.items():
        meta = model.metas[key]
        std = float(value/meta.unit.as_si)
        var_init[model.indices[meta]] = std**2

    

    # part of model
    sectors = {}
    coordinates = np.array([], dtype=float)
    for name, meta in model.metas.items():
        sectors[name] = model.indices[meta]
        coordinates = np.append(coordinates, model.coordinates[meta])

    # Create observation operator
    if hasattr(xp,'obs_db'):
        obs_db = xp.obs_db 
    elif obs_db is None:
        obs_db = default_obs()
    
    Obs=modelling.model_Obs(model.point_observer, obs_db)
    Obs['noise'] = randvars.TimeGaussRV(obs_db, default_seed) 

    # Create operator for generating perturbed initial conditions.
    #depreciated X0 = modelling.GaussRV(C=var_init, mu=x0)
    X0 = randvars.CoordinateRV(N_max=100, M=model.M, mu=x0)
    N0 = randvars.CoordinateRV(N_max=100, M=model.M, mu=0)
    wavelengths, amplitudes = red_noise(-2., length_domain/16.0)
    for key, amp in noise_init.items():
        X0.add_sector(key, sectors[key],coordinates[sectors[key]])
        phase = AR1(model.dt, mean=0., var=(np.pi*3/model.dt.total_seconds())**2, 
                     T=timedelta(minutes=1), base_seed=default_seed)
        amp1 = float(amp / model.metas[key].unit.as_si)
        X0.add_spectral(key, phase, wavelengths, amp1 * amplitudes)
        X0.ref_time = model.ref_time
        
        N0.add_sector(key, sectors[key],coordinates[sectors[key]])
        phase = AR1(model.dt, mean=0., var=(np.pi*3/model.dt.total_seconds())**2, 
                     T=timedelta(minutes=1), base_seed=default_seed+7)
        amp1 =  0.0 * float(amp / model.metas[key].unit.as_si) / np.sqrt(tseq.dt)
        N0.add_spectral(key, phase, wavelengths, amp1 * amplitudes)
        N0.ref_time = model.ref_time
        
    # Link DAPPER dynamical model to ice model.
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': N0,
           'linear': model.step
           }
        
    # Create DAPPER model.
    hmm = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, sectors=sectors)

    # Add plotters.
    LPs = [(1, LP.sliding_diagnostics),
           (1, LP.sliding_marginals(Obs, zoomy=0.8, dims=[37, 277])),
           #(1, LP.spatial1d(obs_inds=jj, dims=sectors['thickness_ice'])),
           #(1, LP.spatial1d(obs_inds=jj, dims=sectors['velocity_ice']))
           ]
    hmm.liveplotters = []

    hmm.model = model
    hmm.coordinates = coordinates
    hmm.mpi = mpi
    return hmm

#%%

def obs_uniform(number):
    """ Grid database with observation uniformly spread in space for
    different observation time."""
    #Point 1st observation.
    step=length_domain/number
    #All observations
    x_obs = np.arange(0.5,number)*step
    #All times.
    t_obs = np.reshape(tseq.tto,(1,-1))
    #Grid grid with all combinations. 
    x_obs, t_obs = np.meshgrid(x_obs, t_obs)
    
    db_type = randvars.point_obs_type(1) 
    db = np.zeros((np.size(x_obs),), dtype=db_type)
    db['coordinates'] = np.reshape(x_obs, (-1,1))
    db['time'][:] = np.reshape(t_obs, (-1))
    
    return db

# %%
from dapper.tools.inflation import ObsInflator, AdaptiveRTPP, ArithmeticMeanFilter
mean_w = np.ones((8,))
save_dir = '/home/ivo/dpr_data/obsdensity/p4_Q'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

def xp_wind(xps, nc_name, N=4):
    xps += da.EnKF_N(N, rot=False)
    xps[-1].save_nc = NetcdfIO(os.path.join(save_dir, nc_name+'.nc'))
    xps[-1].noise_init = default_noise_init
    xps[-1].noise_wind = [default_noise_wind(4000+n*1000)
                          for n in range(0, xps[-1].N)]
    xps[-1].seed = default_seed + 1000

    return xps

def xp_spectral(xps, nc_name, A, N=4):
    xps += da.EnKF_N(N, rot=False)
    xps[-1].save_nc = NetcdfIO(os.path.join(save_dir, nc_name+'.nc'))
    xps[-1].noise_init = default_noise_init
    xps[-1].noise_wind = [spectral_noise_wind(A, 4000+n*1000)
                          for n in range(0, xps[-1].N)]
    xps[-1].seed = default_seed + 1000
    xps[-1].amplitude_spectral = A
    return xps

def xp_refinement(xps, nc_name, number, field_name, obs_sig):
    """ Generate experiments with different numbers of observations. """
    file_name = nc_name + "_OBS{:s}_NO{:03d}.nc".format(field_name, number)
    xps[-1].save_nc = NetcdfIO(os.path.join(save_dir, file_name))
        
    db = obs_uniform(number)
    db['field_name'] = field_name
    db['var'] = obs_sig**2
    xps[-1].obs_db = db
        
    return xps

def xp_sigs(xps, nc_name, number, field_name, obs_sig):
    """ Generate experiments with different obs. error std. deviations"""
    file_name = nc_name + "_OBS{:s}_SIG{:8.2e}.nc".format(field_name, obs_sig)
    xps[-1].save_nc = NetcdfIO(os.path.join(save_dir, file_name))
        
    db = obs_uniform(number)
    db['field_name'] = field_name
    db['var'] = obs_sig**2
    xps[-1].obs_db = db
        
    return xps

def xp_rsigs(xps, nc_name, number, field_name, obs_rsig):
    """ Generate experiments with different obs. error std. deviations"""
    file_name = nc_name + "_OBS{:s}_RSIG{:8.2e}.nc".format(field_name, obs_rsig)
    xps[-1].save_nc = NetcdfIO(os.path.join(save_dir, file_name))
        
    db = obs_uniform(number)
    db['field_name'] = field_name
    db['rvar'] = obs_rsig**2
    xps[-1].obs_db = db
        
    return xps

def run_truth(HMM):
    write_time()
    dapper.set_seed(default_seed)
    xx, yy = HMM.simulate()
    write_time()
    return xx, yy

def write_time(nc=None):
    from datetime import datetime
    file_path = os.path.join(save_dir,'timing.txt')
    t = str(datetime.now())
    with open(file_path,'a') as stream:
        if nc is None:
            stream.write(t+' '+'HMM'+'\n') 
        else:
            stream.write(t+' '+nc.file_path+'\n')

def run_xps(xps, Truth):
    import dapper.tools.progressbar as pb
    yy, xx, xx_true = None, None, None

    for xp in xps:
        
        #Run the truth 
        if xx_true is None: 
            xx_true, _ = run_truth(Truth)
            
            if Truth.mpi.is_root:
                save_nc = NetcdfIO(os.path.join(save_dir, 'truth.nc'))
                save_nc.create_file()
                save_nc.create_dims(Truth, 0)
                save_nc.write_truth(xx_true, [])
            
        #Start time keeping. 
        write_time(xp.save_nc)
        
        #Resample states
        HMM = aev_pnormal(xp) 
        xx = np.array([Truth.model.project(x, HMM) for x in xx_true])
        
        #Update observation operator
        if hasattr(xp,'obs_db'):
            Obs = modelling.model_Obs(Truth.model.point_observer, xp.obs_db)
            Obs['noise'] = randvars.TimeGaussRV(xp.obs_db, default_seed) 
            Truth.Obs = modelling.Operator(**Obs)
        
        #Resample observations truth
        yy_true=[]
        #np.random.seed(100)
        for k, ko, t, dt in pb.progbar(HMM.tseq.ticker, 'Resampling', disable=True):
             if ko is not None:
                 yy1 = Truth.Obs(xx_true[k],t)
                 yy1 = yy1 + Truth.Obs.noise.sample(1)
                 yy_true.append(np.reshape(yy1,(-1)))
        
        #Resample observations based on on interpolated states. Don't use these. 
        yy=[]
        #np.random.seed(100)
        for k, ko, t, dt in pb.progbar(HMM.tseq.ticker, 'Resampling', disable=True):
             if ko is not None:
                 yy1 = HMM.Obs(xx[k],t)
                 yy1 = yy1 + HMM.Obs.noise.sample(1)
                 yy.append(np.reshape(yy1,(-1)))
        
        #Assimilate using the new model. 
        xp.mpi = HMM.mpi
        
        xp.assimilate(HMM, xx, yy_true, liveplots=False)
        
        #Output statistics.
        if xp.mpi.is_root:
            xp.stats.average_in_time()
            
        write_time(xp.save_nc)

    return xps

def copy_this_script(file_name):
    import os,shutil
    this_file = os.path.abspath(__file__)
    copy_file = os.path.join(save_dir,file_name)
    shutil.copyfile(this_file, copy_file)

def create_exp_forcing():
    #Name of experiment
    name = 'forcing'
    #Model for truth
    HMM = aev_pnormal()
    #Subexperiments
    xps = xpList()    
    amplitudes = np.arange(0,7)*.1
    
    for no,amp in enumerate(amplitudes[:4]):
        xps = xp_spectral(xps, name, amp, N=32)
        xps = xp_refinement(xps, name+"_SPECTRAL{:.1f}".format(amp), int(16), 
                            "thickness_ice", 3e-3)
        xps[-1].amp_spectral = amp
        xps[-1].univariate=False
        
    return HMM, xps

#Create list 
def create_exp_obssig():
    #Name of experiment
    name = 'p4_obssig'
    #Model for truth
    HMM = aev_pnormal(order)
    #Subexperiments
    xps = xpList()    
    #sigs = np.array([3e-3, 6e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 6e-2, 1e-1, 2e-1])
    sigs = np.array([400.,200.,150,125,100.,75.,50.,25.])
    
    for sig in sigs:
        xps = xp_wind(xps, name, N=32)
        xps = xp_sigs(xps, name, 16, 'stress', sig)
        
    return HMM, xps

#Create list 
def create_exp_inflation():
    from dapper.tools.inflation import FixedInflator
    
    #Model for truth
    HMM = aev_pnormal()
    #Subexperiments
    xps = xpList()    
    inflation = np.array([1.0,1.02,1.05,1.1]) #,1.02,1.05,1.1,1.2,1.4])
    for infl in inflation:
        name = 'inflation{:03d}'.format(int(infl*100))
        xps = xp_wind(xps, name, N=32)
        xps = xp_refinement(xps, name, int(16), 
                            "thickness_ice", 3e-3)
        xps[-1].infl = FixedInflator(lambda t:infl)
        
    return HMM, xps
    

#Create list 
def create_exp_polyorder():
    #Name of experiment
    name = 'porder_noda'
    #Model for truth
    HMM = aev_pnormal()
    #Subexperiments
    xps = xpList()    
    numbers = np.array([2,4,6,8,10], dtype=int)
    for number in numbers:
        xps = xp_wind(xps, name, N=32)
        xps = xp_refinement(xps, name, int(16), 
                            "thickness_ice", 3e3)
        xps[-1].poly_order = number
        
        file_name = (name + '_p{:02}.nc').format(number)
        xps[-1].save_nc = NetcdfIO(os.path.join(save_dir, file_name))
        
    return HMM, xps

#Create list 
def create_exp_refinement():
    #Name of experiment
    name = 'p4_refinement'
    #Model for truth
    HMM = aev_pnormal(order=12)
    #Subexperiments
    xps = xpList()    
    numbers = np.arange(1,6)*16
    spectral_amplitude = 0.2 #ms-1
    
    for number in numbers:
        xps = xp_wind(xps, name, N=32)
        xps = xp_refinement(xps, name, int(number), 
                            "thickness_ice",5e-2) #32e-3)
        xps[-1].univariate=False
        
    return HMM, xps
    
if __name__ == '__main__' and True:
    HMM, xps = create_exp_refinement()
    for xp in xps:
        copy_this_script(xp.save_nc.file_name+'.py')
    
    #xx,yy=run_truth(HMM)
    xps = run_xps(xps, HMM)


# %% Plot functions

import matplotlib
font = {'family': 'DejaVu Sans',
        'weight': 'bold',
        'size': 14}
matplotlib.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['text.usetex'] = False


def viz_plot(HMM, xp):
    print(xp.avrgs)

    viz.plot_rank_histogram(xp.stats)
    viz.plot_err_components(xp.stats, sectors=HMM.sectors['thickness_ice'])
    viz.plot_err_components(xp.stats, sectors=HMM.sectors['velocity_ice'])
    #viz.plot_hovmoller(xx, sectors=HMM.sectors['thickness_ice'])


def plot_wind(xp, save_fig=False):
    save_dir_w = os.path.join(save_dir, 'wind')
    if not os.path.isdir(save_dir_w) and save_fig:
        os.mkdir(save_dir_w)

    HMM = aev_pnormal(xp)

    t = [datetime(2000, 1, 1)+n*timedelta(hours=1) for n in np.arange(0, 24*24)]
    x = HMM.coordinates[HMM.sectors['velocity_ice']]

    noise_wind = [noise_wind(HMM.model.dt) for noise_wind in xp.noise_wind]

    wind_forcing = []
    for noise in noise_wind:
        wind = forcings.MovingWave(HMM.model, forcings.bellcurve_wind,
                                   **default_wind())
        wind.set_noise(**noise)
        wind_forcing.append(wind)

    HMM.model.forcings = [forcings.EnsembleForcing(wind_forcing)]
    wind = HMM.model.forcings[0]
    
    ranks = np.zeros_like(x)
    
    for it, t1 in enumerate(t):
        fig_path = os.path.join(save_dir_w, 'wind_{:03}.png'.format(it))
        plt.close('all')
        plt.figure(figsize=(8,6))
        t2=(t1-t[0]).total_seconds()/3600.
        plt.title('{:2d} days {:2d} hours'.format(int(t2/24),int(np.mod(t2,24))))
        plt.ylabel('wind [ms-1]',fontsize=font['size'])
        plt.xlabel('position [km]',fontsize=font['size'])
        plt.xlim((0, 100))
        plt.ylim((-20, 20))
        plt.grid(axis='both')
    
        for member in range(0, xp.N):
            wind.n = member
            y = wind(t1, x)
            if member == 0:
                print('min/max wind ', min(y), max(y))

            plt.plot(x*10.,y)
            
            if member==0:
                y0 = y+0.
            else:
                ranks += np.array(np.array(y0)>=np.array(y),dtype=int)
                

        if save_fig:
            plt.savefig(fig_path, dpi=400, format='png')

    return HMM.model.forcings, wind, ranks
