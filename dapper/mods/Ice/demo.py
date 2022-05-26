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
from dapper.tools.datafiles import SaveXP, NetcdfIO
from dapper.mods.Ice.forcings import AR0, AR1
from datetime import datetime, timedelta
import dapper.tools.randvars as randvars

# %% mpi settings

#comm = mpi.COMM_WORLD
#rank = comm.Get_rank()
#mp = comm.Get_size()
#mp = None

# %% Setup model

default_seed = 2000

def default_wind():
    return {'base_velocity': 1.1/86400.,
            'base_amplification': 15.,
            'T_ramp': timedelta(hours=0.5)
            }

def default_noise_wind(seed=default_seed):
    def func(dt):
        sig2, T = 5e-2, timedelta(minutes=10)
        amplification = AR1(dt, 0., sig2, T, seed+10)

        sig2, T = 1e-4, timedelta(days=1)
        center = AR1(dt, 0., sig2, T, seed+20)

        return {'amplification': amplification, 'center': center}

    return func


default_noise_init = {'velocity_ice': Quantity(.0, 'ms-1'),
                      'thickness_ice': Quantity(0.0, 'm')}


def default_obs(tseq):
    db_type = randvars.point_obs_type(1) 
    
    x_obs = np.reshape(np.linspace(.5, 4.5, 5),(-1,1))
    t_obs = np.reshape(tseq.tto,(1,-1))
    x_obs, t_obs = np.meshgrid(x_obs, t_obs)
    
    db = np.zeros((np.size(x_obs),), dtype=db_type)
    db['coordinates'] = np.reshape(x_obs, (-1,1))
    db['time'][:] = np.reshape(t_obs, (-1))
    db['field_name'][:] = 'thickness_ice'
    db['var'] = 1e-3
    
    return db
    

def aev_pnormal(xp=None):
    # Build ice model.
    model = AdvElastoViscousModel(timedelta(seconds=30),
                                  4, 16, 5.)

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
    
    # Create time sequence.
    T = 24*60  # minutes
    Burn = 0*60  # minutes
    tseq = modelling.Chronology(30, dto=30., T=T, Tplot=T, BurnIn=Burn)

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

    # Link DAPPER dynamical model to ice model.
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': 0.,
           'linear': model.step
           }

    # part of model
    sectors = {}
    coordinates = np.array([], dtype=float)
    for name, meta in model.metas.items():
        sectors[name] = model.indices[meta]
        coordinates = np.append(coordinates, model.coordinates[meta])

    Obs=modelling.model_Obs(model.point_observer, default_obs(tseq))

    # Create observation operator
    #jj = np.arange(min(sectors['thickness_ice'])+2,
    #               max(sectors['thickness_ice']), 5)
    #Obs = modelling.partial_Id_Obs(model.M, jj)
    Obs['noise'] = randvars.TimeGaussRV(default_obs(tseq)) # m2 (.01 for thickness)
    # modelling.GaussRV(C=CovMat(2*eye(Nx)))

    # Create operator for generating perturbed initial conditions.
    X0 = modelling.GaussRV(C=var_init, mu=x0)

    # Create DAPPER model.
    hmm = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0, sectors=sectors)

    # Add plotters.
    LPs = [(1, LP.sliding_diagnostics),
           (1, LP.sliding_marginals(Obs, zoomy=0.8, dims=[37, 277])),
           #(1, LP.spatial1d(obs_inds=jj, dims=sectors['thickness_ice'])),
           #(1, LP.spatial1d(obs_inds=jj, dims=sectors['velocity_ice']))
           ]
    hmm.liveplotters = LPs

    hmm.model = model
    hmm.coordinates = coordinates
    hmm.mpi = mpi
    return hmm

# %%


save_dir = '/home/ivo/dpr_data/mpi_test'


def run_truth():
    HMM = aev_pnormal()
    dapper.set_seed(default_seed)
    xx, yy = HMM.simulate()
    return HMM, xx, yy


def xp_wind(xps, nc_name):
    #xps += da.EnKF('Sqrt', N=4, infl=1., rot=False)
    xps += da.EnKF_N(N=32, rot=False)
    xps[-1].save_nc = NetcdfIO(os.path.join(save_dir, nc_name+'.nc'))
    xps[-1].noise_init = default_noise_init
    xps[-1].noise_wind = [default_noise_wind(4000+n*100)
                          for n in range(0, xps[-1].N)]
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
    xps = xp_wind(xps, 'test_obs')

    HMM, xx, yy = run_truth()
    #xps = xps_run(xps, xx, yy)


# %% Plot functions


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

    t = [datetime(2000, 1, 1)+n*timedelta(hours=1)
         for n in np.arange(0, 16*24+1)]
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

    N = 16
    pmax = np.zeros((N, 2))
    for it, t1 in enumerate(t):
        fig_path = os.path.join(save_dir_w, 'wind_{:03}.png'.format(it))
        plt.close('all')
        plt.figure()
        plt.title('{}'.format(t1.strftime('%Y-%m-%d %H:%M:%S')))
        plt.ylabel('wind [ms-1]')
        plt.xlabel('position [km]')
        plt.xlim((0, 50))
        plt.ylim((-20, 20))
        plt.grid(axis='both')

        for member in range(0, N):
            wind.n = member
            y = wind(t1, x)
            if member == 0:
                print('min/max wind ', min(y), max(y))

            # plt.plot(x*10.,y)

        if save_fig:
            plt.savefig(fig_path, dpi=150, format='png')

    return HMM.model.forcings, wind
