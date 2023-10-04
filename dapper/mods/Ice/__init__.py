""" 
Implement DA in 1D ice-model. 
"""

import dapper.mods as modelling
from abc import ABC, abstractmethod
from pyIce.time_steppers import AdamsMoulton
import datetime
import numpy as np
import firedrake as fem

from pyIce.initialisers import ArrayInitialiser
from pyIce.activators import SingleTime
from dapper.tools.seeding import set_seed
from pip._vendor.colorama import initialise
from mpi4py import MPI as mpi
from dapper.tools.multiproc import NoneMPI, EnsembleMPI
from ipython_genutils.testing.decorators import _x11_skip_msg


# %% Define model components.

class IceModel(ABC):

    def set_defaults(self):
        """Set default values for different attributes."""
        self.ref_time = datetime.datetime(2000, 1, 1)
        self.indices = {}
        self.coordinates = {}
        self.metas = {}
        self._M = None
        self.do_time_seeding = True
        
        self.tmp_states = []

    @property
    def M(self):
        """Number of elements in the state."""
        if len(self.indices) == 0:
            self._get_meta()
        return sum([len(ind) for ind in self.indices.values()])

    def build_model(self):
        """Build model in pyIce."""
        self.build_fields()  
        mpi = self.build_mesh(self.length, self.no_cells)
        self.runner, self.initialiser = self.build_runner()
        self.build_runner()
        return mpi

    @abstractmethod
    def build_runner(self):
        """Build model runner in pyIce."""
        pass
    
    @abstractmethod
    def build_fields(self):
        """Build Firedrake objects representing fields."""
        pass

    def build_mesh(self, length, no_cells):
        """Create model mesh in Firedrake. """
        import firedrake as fem

        ensemble_mpi = fem.Ensemble(fem.COMM_WORLD, 1)
        self.mesh = fem.PeriodicIntervalMesh(
                    no_cells, length, comm=ensemble_mpi.comm)
        mpi = EnsembleMPI(ensemble_mpi)
            
        return mpi
    
    def _build_state(self):
        """ Create a Firedrake representation of state. """
        return self.builder.construct_state()

    def _get_meta(self):
        """ Get meta data of pyIce model."""
        from pyIce.cell_operations import global_coordinates
        state = self._build_state()

        lb = 0
        for field in state.fields:
            ub = lb+np.size(field.as_function.dat.data)
            self.indices[field.meta] = np.arange(lb, ub)
            lb = ub

            self.metas[field.meta.name] = field.meta

            self.coordinates[field.meta] = global_coordinates(
                field.as_function)
            
    def point_observer(self, database, x, t):
        """ Observe state at specific points listed in 
        database. """
        
        #Select observations taken at current time. 
        selection = np.array(database['time'])==t
        yy = np.zeros((np.sum(selection),), dtype=float)
        
        while len(self.tmp_states)<1:
            self.tmp_states.append(self._build_state())
        state = self.tmp_states[0]
        
        if any(selection):
            state.set_data(np.reshape(x,(-1)))
             
            #Create database with observations at this time. 
            db = database[selection]
            
            #Sample observations. 
            for field in state.fields:
                selection = db['field_name']==field.meta.name
                if any(selection):
                    coord = db['coordinates'][selection]
                    yy[selection] = np.reshape(field.sample(coord), (-1))

        return yy
    
    def lin_observer(self, database, x, t):
        """ Observe state at state at specific points using linear 
        interpolation. """
        
        #Select observations taken at current time. 
        selection = np.array(database['time'])==t
        yy = np.zeros((np.sum(selection),), dtype=float)
        
        while len(self.tmp_states)<1:
            self.tmp_states.append(self._build_state())
        state = self.tmp_states[0]
        
        if any(selection):             
            #Create database with observations at this time. 
            db = database[selection]
            
            #Sample observations. 
            for field in state.fields:
                selection = db['field_name']==field.meta.name
                if any(selection):
                    coord = np.reshape(db['coordinates'][selection],(-1))
                    state_x = np.reshape(self.coordinates[field.meta],(-1))
                    state_val = np.reshape(x[self.indices[field.meta]],(-1))
                                           
                    if not np.all(np.diff(state_x) > 0):
                        raise ValueError('Non -increasing coordinates.')
                    yy[selection] = np.interp(x=coord, xp=state_x, fp=state_val)
        return yy
    
    def inner_l2(self, field_name, x1, x2):
        """Calculate L2-inner product by integrating over domain."""
        while len(self.tmp_states)<2:
            self.tmp_states.append(self._build_state())
        
        for field in self.tmp_states[0].fields:
            if field.meta.name == field_name:
                field.set_data(x1)
                f1 = field.as_function
        for field in self.tmp_states[1].fields:
            if field.meta.name == field_name:
                field.set_data(x2)
                f2 = field.as_function
                
        return fem.assemble(fem.inner(f1,f2)*fem.dx)
    
    def step1(self, x0, t, dt, n):
        """ Step 1 ensemble member forward in time. """
        # Convert minutes to seconds.
        t, dt = 60*t, 60*dt

        time0 = self.ref_time + datetime.timedelta(seconds=int(t))
        no_steps = int(dt/self.dt.total_seconds())

        # Save output only at end time step.
        self.runner.stepables[-1].activator = SingleTime(time0 + no_steps * self.dt)

        if self.initialiser is None:
            self.initialiser = ArrayInitialiser(time0, 0.*x0)
        self.initialiser.time = time0
        self.initialiser.array = x0

        #Pass ensemble member to forcings
        for m,_ in enumerate(self.forcings):
            self.forcings[m].n=n
        self.runner.run_forward(self.initialiser, no_steps + 1)
        observer = self.runner.steppers[-1]
        
        return observer.data[-1]

    def step(self, E, t, dt, members=None): 
        """ Step ensemble forward in time. """       
        E = self.si2meta_units(E)

        if E.ndim == 1:
            E = self.step1(E, t, dt, 0)
        elif members is not None:
            for n, member in enumerate(members):
                E[n] = self.step1(E[n], t, dt, member)
        else:
            for n, x in enumerate(E):
                E[n] = self.step1(x, t, dt, n)

        E = self.meta2si_units(E)

        return E

    def meta2si_units(self, E):
        """ Convert matrix ensemble with quantities in model units
        in SI units."""
        if len(self.indices) == 0:
            self._get_meta()

        ndim = np.ndim(E)
        if ndim == 1:
            E = np.reshape(E, (1, -1))

        for name, meta in self.metas.items():
            ind = self.indices[meta]
            ratio = float(meta.unit/meta.unit.as_si)
            E[:, ind] *= ratio

        if ndim == 1:
            E = np.reshape(E, (-1))
        return E

    def si2meta_units(self, E):
        """ Convert matrix ensemble with quantities in SI units into units
        used by model."""
        if len(self.indices) == 0:
            self._get_meta()

        ndim = np.ndim(E)
        if ndim == 1:
            E = np.reshape(E, (1, -1))

        for name, meta in self.metas.items():
            ind = self.indices[meta]
            ratio = float(meta.unit.as_si/meta.unit)
            E[:, ind] *= ratio

        if ndim == 1:
            E = np.reshape(E, (-1))
        return E
    
    def project(self, x, HMM):
        while len(self.tmp_states)<1:
            self.tmp_states.append(self._build_state())
        
        state_in = self.tmp_states[-1]
        state_in.set_data(x)
        
        xo = np.zeros_like(HMM.coordinates)
        for field in state_in.fields:
            indices = HMM.sectors[field.meta.name]
            coordinates = HMM.coordinates[indices]
            xo[indices] = np.reshape(field.sample(coordinates), (-1))
                
        return xo

class ElastoViscousModel(IceModel):

    def __init__(self, dt, order, no_cells, length):
        self.set_defaults()
        self.dt = dt
        self.no_cells = no_cells
        self.length = length
        self.order = order

    @property
    def boundaries(self):
        return {}

    

    def build_fields(self):
        self.build_thickness_ice()
        self.build_velocity_ice()
        self.build_damage()
        
    def build_damage(self):
        from pyIce.fields import MetaField
        from pyIce.units import SiUnit

        def func(t, x): return 0.1
        self.damage = MetaField((), func, name="damage",
                                unit=SiUnit(), lbound=lambda: 0., 
                                ubound=lambda: 1.)

    def build_thickness_ice(self):
        from pyIce.fields import MetaField
        from pyIce.units import m

        def func(t, x): return 2.
        self.thickness_ice = MetaField((), func, name="thickness_ice",
                                       unit=m, lbound=lambda: 0.)

    def build_velocity_ice(self):
        from pyIce.fields import MetaField
        from pyIce.units import m, s

        def func(t, x): return 0. * \
            np.array([1]) * 1e-2 * np.sin(2*np.pi*x/self.length)
        self.velocity_ice = MetaField(
            (1,), func, name="velocity_ice", unit=m/s)

    def build_runner(self):
        from pyIce.models import ModelDirector, SameSpaceFactory
        from pyIce.time_steppers import ModelRunner, AdamsMoulton, StepTimeForward
        from pyIce.observers import ArrayCollector
        from pyIce.ice_model import EvpBuilder
        from pyIce.activators import Always

        evp_builder = EvpBuilder(self.mesh, 1, self.boundaries,
                                 thickness_ice=self.thickness_ice,
                                 velocity_ice=self.velocity_ice,
                                 damage=self.damage)
        space_factory = SameSpaceFactory(
            self.mesh, 'DG', self.order, variant="spectral")
        director = ModelDirector(self.mesh, space_factory)
        director.build(evp_builder)
        self.builder = evp_builder

        steppers = [AdamsMoulton(evp_builder, self.dt, self.order, unit_time=evp_builder.unit_time),
                    StepTimeForward(self.dt), ArrayCollector(Always())]

        initialiser = None
        runner = ModelRunner(steppers)
        return runner, initialiser

    def build_initial(self, time):
        self.ref_time = time
        builder = self.runner.stepables[0].builder
        state = builder.construct_state()
        state.update(self.ref_time)
        return np.reshape(state.as_function.dat.data, (-1))


class AdvElastoViscousModel(ElastoViscousModel):

    def build_runner(self):
        from pyIce.models import ModelDirector, SameSpaceFactory
        from pyIce.time_steppers import ModelRunner, StepTimeForward
        from pyIce.time_steppers import AdamsMoulton, LegendreFilter
        from pyIce.observers import ArrayCollector
        from pyIce.ice_model import EvpBuilder, TransportBuilder
        from pyIce.activators import Always, PeriodicIterations

        evp_builder = EvpBuilder(self.mesh, 1, self.boundaries,
                                 thickness_ice=self.thickness_ice,
                                 velocity_ice=self.velocity_ice,
                                 velocity_air=self.velocity_air)
        trans_builder = TransportBuilder(self.mesh, 1, self.boundaries,
                                         **evp_builder.args)
        
        lfilter = np.ones((self.order+1,))
        #lfilter[-3:]=np.array([.99, .8, .05])
        lfilter = np.exp(-4*(np.arange(0, self.order+1)/self.order)**8)

        space_factory = SameSpaceFactory(self.mesh, 'DG', self.order,
                                         variant="spectral")
        director = ModelDirector(self.mesh, space_factory)
        director.build(evp_builder)
        director.build(trans_builder)
        self.builder = evp_builder

        steppers = [AdamsMoulton(trans_builder, self.dt, 2, unit_time=evp_builder.unit_time),
                    AdamsMoulton(evp_builder, self.dt, 2,
                                 unit_time=evp_builder.unit_time),
                    LegendreFilter(lfilter),
                    StepTimeForward(self.dt), ArrayCollector(Always())]

        filter_activator = PeriodicIterations(5)
        steppers[2].activator = filter_activator

        runner = ModelRunner(steppers)
        initialiser = None

        return runner, initialiser