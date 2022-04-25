""" 
Implement DA in 1D ice-model. 
"""

import dapper.mods as modelling
from abc import ABC, abstractmethod 
from pyIce.time_steppers import AdamsMoulton
import datetime
import numpy as np

#%% Define model components.

class IceModel(ABC):
    
    def set_defaults(self):
        self.ref_time = datetime.datetime(2000,1,1)
        self.indices = {}
        self.coordinates = {}
        self.metas = {}
        self._M = None
        self.mp = False
        
    @property 
    def obs(self):
        observer = self.runner.steppers[-1]
        return observer.state_as_vector(-1)
    
    @property 
    def M(self):
        if len(self.indices)==0:
            self._get_meta()
        return sum([len(ind) for ind in self.indices.values()])
        
    @abstractmethod 
    def _get_meta(self):
        pass
    
    @abstractmethod 
    def build_runner(self):
        pass
    
    def build_mesh(self, length, no_cells):
        import firedrake as fem
        self.mesh = fem.PeriodicIntervalMesh(no_cells, length)
        
    def _get_meta(self):
        from pyIce.cell_operations import global_coordinates
        state = self.builder.construct_state()
        
        lb=0
        for field in state.fields:
            ub=lb+np.size(field.as_function.dat.data)
            self.indices[field.meta]=np.arange(lb,ub)
            lb=ub
            
            self.metas[field.meta.name] = field.meta
            
            self.coordinates[field.meta]=global_coordinates(field.as_function)    
    
    def step1(self, x0, t, dt):
        from pyIce.initialisers import ArrayInitialiser
        from pyIce.activators import SingleTime 
        
        #Convert minutes to seconds.
        t, dt = 60*t, 60*dt
        
        time0 = self.ref_time + datetime.timedelta(seconds=int(t))
        print('Runner model at ',time0)
        no_steps = int(dt/self.dt.total_seconds())
        self.runner.stepables[-1].activator=SingleTime(time0 + no_steps * self.dt)
        
        if self.initialiser is None:
            self.initialiser=ArrayInitialiser(time0, x0)
        else:
            self.initialiser.time = time0 
            self.initialiser.array = x0
        
        self.runner.run_forward(self.initialiser, no_steps + 1)
        return self.obs
    
    def step(self, E, t, dt):
        if E.ndim == 1:
            return self.step1(E, t, dt)
        if E.ndim == 2:
            if self.mp:
                import dapper.tools.multiproc as multiproc
                with multiproc.Pool(self.mp) as pool:
                    E = pool.map(lambda x: self.step1(x, t=t, dt=dt), E)
                E = np.array(E)
            else:
                for n, x in enumerate(E):
                    E[n] = self.step1(x, t, dt)
            return E
    

class ElastoViscousModel(IceModel):
    
    def __init__(self, dt, order, no_cells, length):
        self.set_defaults()
        self.dt = dt 
        self.no_cells = no_cells
        self.length = length
        self.order = order
        
        self.build_runner()
    
    def set_defaults(self):
        super().set_defaults()
        self._thickness_ice = None
        self._velocity_ice = None
        
    @property 
    def boundaries(self):
        return {}
    
    @property 
    def thickness_ice(self):
        from pyIce.fields import MetaField
        from pyIce.units import m
        
        if self._thickness_ice is None:
            func = lambda t,x: 4.
            self._thickness_ice = MetaField((), func, name="thickness_ice", 
                                            unit=m, lbound=lambda: 0.)
            
        return self._thickness_ice
    
    @property 
    def velocity_ice(self):
        from pyIce.fields import MetaField
        from pyIce.units import m,s
        
        if self._velocity_ice is None:
            func = lambda t,x : np.array([1]) * 1e-2 * np.sin(2*np.pi*x/self.length)
            self._velocity_ice = MetaField((1,), func, name="velocity_ice", 
                                           unit=m/s)
            
        return self._velocity_ice


    
    def build_runner(self):
        from pyIce.models import ModelDirector, SameSpaceFactory
        from pyIce.time_steppers import ModelRunner, AdamsMoulton, StepTimeForward
        from pyIce.observers import ArrayCollector
        from pyIce.ice_model import EvpBuilder 
        from pyIce.activators import Always
        
        self.build_mesh(self.length, self.no_cells)
        
        evp_builder = EvpBuilder(self.mesh, 1, self.boundaries, 
                                 thickness_ice=self.thickness_ice,
                                 velocity_ice=self.velocity_ice)
        space_factory = SameSpaceFactory(self.mesh, 'DG', self.order, variant="spectral")
        director = ModelDirector(self.mesh, space_factory)
        director.build(evp_builder)
        self.builder=evp_builder
    
        steppers=[AdamsMoulton(evp_builder, self.dt, self.order, unit_time=evp_builder.unit_time),
                  StepTimeForward(self.dt), ArrayCollector(Always())]
        
        self.runner = ModelRunner(steppers)
        self.initialiser = None
        
    def build_initial(self, time):
        self.ref_time = time
        builder = self.runner.stepables[0].builder
        state = builder.construct_state()
        state.update(self.ref_time)
        return np.reshape(state.as_function.dat.data,(-1))
    
class AdvElastoViscousModel(ElastoViscousModel):
    
    def set_defaults(self):
        super().set_defaults()
        self._velocity_air = None
        
    @property 
    def velocity_air(self):
        from pyIce.fields import MetaField
        from pyIce.units import m,s
        
        def gauss_wind(t, x):
            t = (t-self.ref_time).total_seconds()
            T = datetime.timedelta(days=4).total_seconds()
            T_ramp = datetime.timedelta(hours=1).total_seconds()
            W = 0.5

            phase_speed = 3 * self.length / T
            t_random = np.random.normal(loc=0, scale=60.)
            phase = np.mod(x - phase_speed, self.length) - 0.5 * self.length

            signal = np.array([10.0]) * np.exp(-0.5 * phase ** 2 / W ** 2)
            time_factor = min(1.0, t / T_ramp)

            signal = signal + np.random.normal(loc=0., scale=time_factor*.2) * np.array([10.])
            return 1.* signal * time_factor
            
        if self._velocity_air is None:
            self._velocity_air = MetaField((1,), gauss_wind, name="velocity_air", 
                                           unit=m/s)
                
        return self._velocity_air
    
    def build_runner(self):
        from pyIce.models import ModelDirector, SameSpaceFactory
        from pyIce.time_steppers import ModelRunner, AdamsMoulton, StepTimeForward
        from pyIce.observers import ArrayCollector
        from pyIce.ice_model import EvpBuilder, TransportBuilder
        from pyIce.activators import Always
        
        self.build_mesh(self.length, self.no_cells)
        
        evp_builder = EvpBuilder(self.mesh, 1, self.boundaries, 
                                 thickness_ice=self.thickness_ice,
                                 velocity_ice=self.velocity_ice,
                                 velocity_air=self.velocity_air)
        trans_builder = TransportBuilder(self.mesh, 1, self.boundaries,
                                     **evp_builder.args)
        space_factory = SameSpaceFactory(self.mesh, 'DG', self.order, variant="spectral")
        director = ModelDirector(self.mesh, space_factory)
        director.build(evp_builder)
        director.build(trans_builder)
        self.builder=evp_builder
    
        steppers=[AdamsMoulton(trans_builder, self.dt, self.order, unit_time=evp_builder.unit_time),
                  AdamsMoulton(evp_builder, self.dt, self.order, unit_time=evp_builder.unit_time),
                  StepTimeForward(self.dt), ArrayCollector(Always())]
        
        self.runner = ModelRunner(steppers)
        self.initialiser = None
        
  
class IdentityModel(ElastoViscousModel):
    
    def step1(self, x0, t, dt):        
        time0 = self.ref_time + datetime.timedelta(seconds=int(t))
        print('Runner model at ',time0)
        return x0


                 
    
        
    
        

        