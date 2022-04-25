"""Reproduce results from `bib.sakov2008deterministic`."""

import numpy as np

from dapper import xpList
import dapper
import dapper.mods as modelling
import dapper.da_methods as da
from dapper.mods.Ice import AdvElastoViscousModel, IdentityModel
from dapper.tools.localization import nd_Id_localization
import dapper.tools.liveplotting as LP
import datetime
import matplotlib.pyplot as plt

plt.close('all')

############################
# Time series, model, initial condition
############################

def mysetup(HMM, xp):
    model = AdvElastoViscousModel(datetime.timedelta(seconds=20),
                                  2, 16, 5.)
    model.mp = None
    
    x0 = model.build_initial(datetime.datetime(2000,1,1))
    var_velocity = np.zeros((model.M,))
    var_velocity[model.indices[model.metas["velocity_ice"]]] = 1.
    
    Dyn = {'M': model.M,
           'model': model.step,
           'noise': 0.,
           'linear':model.step
           }
    
    jj = np.array([24])  # obs_inds
    Obs = modelling.partial_Id_Obs(model.M, jj)
    Obs['noise'] = 1e-3  # modelling.GaussRV(C=CovMat(2*eye(Nx)))

    T=6*60
    tseq = modelling.Chronology(5, dto=T, T=T, Tplot=T)
    
    X0 = modelling.GaussRV(C=var_velocity * 0.e-4, mu=x0)
    hmm = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
    
    params = dict(dims=[24,30])
    hmm.liveplotters=[(1, LP.sliding_marginals(Obs, zoomy=0.8, **params))]
    return dapper.seed_and_simulate(hmm, xp)

HMM,xx,yy = mysetup(None, None)


#%%

w = np.ones((HMM.Dyn.M))
#w[48:] = 0. 

xps = xpList()
xps += da.EnKF('Sqrt', N=12, infl=1.00, rot=True)

for xp in xps:
    xp.seed = 3000
    
#xp.assimilate(HMM, xx, yy, liveplots=True, field_weight=w)
save_as = xps.launch(HMM, liveplots=True, save_as='spread_test', 
                     field_weight=w, setup=mysetup)

#%%

#xp.stats.average_in_time()
#print(xp.avrgs.tabulate(['rmse.f', 'rmse.a']))

