""" 
Implement DA in 1D ice-model. 
"""

import dapper.mods as modelling
from abc import ABC, abstractmethod
from pyIce.time_steppers import AdamsMoulton
import datetime
import numpy as np

from pyIce.initialisers import ArrayInitialiser
from pyIce.activators import SingleTime
from dapper.tools.seeding import set_seed
from pip._vendor.colorama import initialise
from mpi4py import MPI as mpi
from dapper.tools.multiproc import NoneMPI, EnsembleMPI

        
        

# %% Define model components.

class IceModel(ABC):

    def set_defaults(self):
        self.ref_time = datetime.datetime(2000, 1, 1)
        self.indices = {}
        self.coordinates = {}
        self.metas = {}
        self._M = None
        self.do_time_seeding = True

    @property
    def M(self):
        if len(self.indices) == 0:
            self._get_meta()
        return sum([len(ind) for ind in self.indices.values()])

    @abstractmethod
    def build_runner(self):
        pass

    def build_mesh(self, length, no_cells):
        import firedrake as fem

        ensemble_mpi = fem.Ensemble(fem.COMM_WORLD, 1)
        self.mesh = fem.PeriodicIntervalMesh(
                    no_cells, length, comm=ensemble_mpi.comm)
        mpi = EnsembleMPI(ensemble_mpi)
            
        return mpi

    def _get_meta(self):
        from pyIce.cell_operations import global_coordinates
        state = self.builder.construct_state()

        lb = 0
        for field in state.fields:
            ub = lb+np.size(field.as_function.dat.data)
            self.indices[field.meta] = np.arange(lb, ub)
            lb = ub

            self.metas[field.meta.name] = field.meta

            self.coordinates[field.meta] = global_coordinates(
                field.as_function)

    def step1(self, x0, t, dt, n):
        # Convert minutes to seconds.
        t, dt = 60*t, 60*dt

        time0 = self.ref_time + datetime.timedelta(seconds=int(t))
        no_steps = int(dt/self.dt.total_seconds())

        # Save output only at end time step.
        self.runner.stepables[-1].activator = SingleTime(time0 + no_steps * self.dt)

        if self.initialiser is None:
            self.initialiser = ArrayInitialiser(time0, x0)
        else:
            self.initialiser.time = time0
            self.initialiser.array = x0

        self.forcing.n = n
        self.runner.run_forward(self.initialiser, no_steps + 1)
        observer = self.runner.steppers[-1]

        return observer.state_as_vector(-1)

    def step(self, E, t, dt, members=None):        
        E = self.si2meta_units(E)

        if E.ndim == 1:
            E = self.step1(E, t, dt, 0)
        elif members is not None:
            for n, (x, member) in enumerate(zip(E, members)):
                E[n] = self.step1(x, t, dt, member)
        else:
            for n, x in enumerate(E):
                E[n] = self.step1(x, t, dt, n)

        E = self.meta2si_units(E)

        return E

    def meta2si_units(self, E):
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

    def build_model(self):
        self.build_fields()  
        mpi = self.build_mesh(self.length, self.no_cells)
        self.runner, self.initialiser = self.build_runner()
        self.build_runner()
        return mpi

    def build_fields(self):
        self.build_thickness_ice()
        self.build_velocity_ice()

    def build_thickness_ice(self):
        from pyIce.fields import MetaField
        from pyIce.units import m

        def func(t, x): return 4.
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
                                 velocity_ice=self.velocity_ice)
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
        lfilter = np.zeros((self.order+1,))
        lfilter[-2:] = np.array([.1, 1.])

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

        filter_activator = PeriodicIterations(10)
        steppers[2].activator = filter_activator

        runner = ModelRunner(steppers)
        initialiser = None

        return runner, initialiser