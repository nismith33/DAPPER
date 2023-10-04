# -*- coding: utf-8 -*-
""" Functions that can be patched together into experiment. """
import numpy as np
from dapper.mods import Liu2002 as modelling
from dapper.mods import HiddenMarkovModel, Operator
from dapper.tools.randvars import RV_from_function, GaussRV
from dapper.tools.localization import nd_Id_localization
from sys import getsizeof

import dill
import os
from copy import copy, deepcopy
from mpi4py import MPI
from abc import ABC, abstractmethod

# Directory to store data and figures.
fig_dir = "/home/ivo/dpr_data/synthetic/"
# Domain length
L = 8e6  # m
# Number of cells
Ncells = 79
# Number of ensemble members
Nens_max = 100
# Spacing of artificial satellite observations
Nobs_dx = 1e3


def nice_ticks(xmin, xmax, max_ticks=13):
    """
    Create nicely spaced ticks.

    Parameters
    ----------
    xmin : float
        Lowest value in dataset.
    xmax : float
        Highest value in dataset
    max_ticks : int>=2, optional
        Maximum number of ticks. The default is 13.

    Returns
    -------
    tuplet of floats
        Limits
    array of floats
        Position of ticks. 
    """
    # Range between min and max vlues.
    if np.isclose(xmax, xmin, 1e-10):
        r = 0.1 * np.abs(xmax)
    else:
        r = np.abs(xmax-xmin)

    # Scale range such that values ar in [1,10)
    magnitude = 10**np.floor(np.log10(r))
    r = r/magnitude

    # Find smallest step with less than max_ticks ticks.
    steps = np.array([.1, .2, .25, .5, 1., 2., 2.5, 5.])
    ixmin = np.floor(xmin / (magnitude * steps))
    ixmax = np.ceil(xmax / (magnitude * steps))
    n = np.where((ixmax - ixmin) < max_ticks)[0][0]

    ticks = np.arange(ixmin[n], ixmax[n]+1) * steps[n] * magnitude

    return (np.min(ticks), np.max(ticks)), ticks


def collect(tree, selection=None, nodes=None):
    """
    Collect subset of tree nodes based on their indices. 

    Parameters
    ----------
    tree : TreeNode object
        Node making up the top of the tree. 
    selection : int>=0 | array of int
        If integer all nodes from that level will be collected. 
        If array, the ith element gives the node to be collected at the ith 
        level. If the ith element < 0, all nodes at that level are collected. 
    nodes : list of nodes, optional
        Used to make function recursive

    Returns
    -------
    List of nodes satisfying selection criteria.

    """
    if nodes is None:
        nodes = []

    if isinstance(selection, (int, np.int16, np.int32)):
        selection = np.ones((selection,)) * -1
    else:
        selection = np.array(selection, dtype=int)

    if len(selection) == 0:
        nodes.append(tree)
    else:
        ind, selection = selection[0], selection[1:]
        for n, child in enumerate(tree.children):
            if ind < 0 or ind == n:
                nodes = collect(child, selection, nodes)

    return nodes


class Comm(ABC):
    """ Class that takes care of parallel execution of a tree. 

    Attributes 
    ----------
    size : int>0
        Number of processes in this communicator. 
    procs : array of int>=0
        Index of each process in this communicator. 
    rank : int 
        Index of currently active proc. 

    """

    @abstractmethod
    def is_root(self):
        """ Return whether current process is the root process. """
        pass

    @abstractmethod
    def execute(self, children):
        """
        Execute a list of objects in parallel. 

        Parameters
        ----------
        children : TreeNode object
            List of objects that need to be executed. 

        Returns
        -------
        None.

        """
        pass

    def split(self, procs, Ntasks):
        """
        Divide the available processes over different tasks. 

        Parameters
        ----------
        procs : int>=0
            List of processes in this communicator. 
        Ntasks : TYPE
            Number of tasks that have to be distributed over processes. 

        Returns
        -------
        batch : array of int arrays
            Array with batches of tasks. Each batch will be executed in serial.
        batch_procs : array
            Array with for each batch a list arrays. Each of the latter arrays
            contains the processes assigned to one of the tasks in the batch. 

        """
        Niter = int(np.ceil(Ntasks/len(procs)))
        batches = np.array_split(np.arange(Ntasks), Niter)
        batch_procs = [np.array_split(procs, len(batch)) for batch in batches]
        return batches, batch_procs


class SingleComm(Comm):
    """ Class to run without parallelisation. """

    def __init__(self):
        """ Class constructor. """
        self.size = 1
        self.procs = np.arange(1)
        self.rank = 0

    @property
    def is_root(self):
        return True

    def execute(self, children):
        for child in children:
            child.execute(copy(self))


class MpiComm(Comm):
    """ Class to run using MPI parallelisation. """

    def __init__(self):
        """ Class constructor. """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.procs = np.arange(self.size)

        print('MPI active ', self.rank, ' of ', self.size)

    @property
    def is_root(self):
        return self.rank == self.procs[0]

    def execute(self, children):
        if len(children) == 0:
            return

        children = np.array(children, dtype=object)
        groups, procs = self.split(self.procs, len(children))
        for group, gprocs in zip(groups, procs):
            for child, cprocs in zip(children[group], gprocs):
                if self.rank in cprocs:
                    child_comm = copy(self)
                    child_comm.procs = cprocs
                    child.execute(child_comm)


class TreeIO:
    """ Class responsible for saving and loading tree nodes. """

    def __init__(self, file_dir, try_loading=True):
        """
        Class constructor. 

        Parameters
        ----------
        file_dir : str
            Path to directory where experiment output will be saved. 
        try_loading : bool, optional
            Indicate whether experiment should attempt to load existing nodes
            from file. The default is True.

        Raises
        ------
        FileNotFoundError
            Filepath does not exist. 

        Returns
        -------
        None.

        """
        import shutil
        self.try_loading = try_loading
        self.dir = file_dir
        if not os.path.exists(file_dir):
            raise FileNotFoundError(
                "Directory {} does not exist.".format(file_dir))

        try:
            if not os.path.exists(self.dir):
                os.mkdir(self.dir)
        except FileExistsError:
            pass

    def filepath(self, filename):
        """ Return full filepath to file with name filename."""
        return os.path.join(self.dir, 'tree', filename+'.pkl')

    def exists(self, filename):
        """ Check if file with name filename exists. """
        return os.path.exists(self.filepath(filename))

    def save(self, filename, obj):
        """ Save data in tree node to a file. """
        with open(self.filepath(filename), 'wb') as stream:
            dill.dump(obj, stream)

    def load(self, filename):
        """ Load data in tree node from a file. """
        if not self.try_loading:
            return None
        elif not os.path.exists(self.filepath(filename)):
            return None
        else:
            with open(self.filepath(filename), 'rb') as stream:
                return dill.load(stream)


class TreeNode(ABC):
    """ Abstract class representing a node in a tree. 

    Experiment is structed of tree with each leaf an assimilation experiment
    that needs to be executed and each node being a setting for observation
    operator or model used. This class makes up a template for these nodes. 

    Attributes
    ----------
    operator 
        Model/DA/observation operator build by this node. 
    value 
        Result generated by self.operator and stored in this node. 
    index : int array 
        Index of node in tree. E.g. self.index=[0,2,1] indicates that this 
        node is at level 3 in the tree. That it is the 2nd child of its parent
        and that its parent is the 3rd child of its grandparent. 
    xx : array of float
        Array with true solution. 
    yy : list of float arrays
        Value observations at different times. 
    E : dict of float arrays 
        Ensemble output for different times. 
    HMM : HiddenMarkovModel
        DAPPER object that contains the model and observation operator used
        to generate truth. 
    true_HMM : HiddenMarkovModel 
        DAPPER object used to generate downscaled version of the truth. 
    name : str
        Name given to this node. 
    io : TreeIO object 
        Object through which data is stored/loaded from the harddisk. 

    Methods 
    -------
    add_truth 
        Add a TruthTreeNode to the tree. 
    add_obs 
        Add an ObsTreeNode to the tree. 
    add_model 
        Add a ModelTreeNode to the tree. 
    add_xp
        Add a XpTreeNode with an assimilation method to the tree. 
    build
        Build the model/observation/DA operator to be stored in this node. 
    value 
        Result generated by self.operator 
    index2str
        Convert the node index in self.index into a string. 

    """

    def add_truth(self, *args, **kwargs):
        for child in self.children:
            child.add_truth(*args, **kwargs)

    def add_obs(self, Nobs, sig, *args, **kwargs):
        for child in self.children:
            child.add_obs(Nobs, sig, *args, **kwargs)

    def add_model(self, model_type, Ncells, *args, **kwargs):
        for child in self.children:
            child.add_model(model_type, Ncells, *args, **kwargs)

    def add_xp(self, xp_type, Nens, *args, **kwargs):
        for child in self.children:
            child.add_xp(xp_type, Nens, *args, **kwargs)

    @property
    def xx(self):
        if self.parent is None:
            return None
        else:
            return self.parent.xx
        
    @property 
    def smooth_xx(self):
        return self.xx

    @property
    def yy(self):
        if self.parent is None:
            return None
        else:
            return self.parent.yy

    @property
    def E(self):
        if self.parent is None:
            return None
        else:
            return self.parent.E

    @property
    def HMM(self):
        if self.parent is None:
            return None
        else:
            return self.parent.HMM

    @property
    def xp(self):
        if self.parent is None:
            return None
        else:
            return self.parent.xp

    @property
    def true_HMM(self):
        if self.parent is None:
            return None
        else:
            return self.parent.true_HMM

    @property
    def index2str(self):
        s = '{:02d}'.format(self.index[0])
        for ind in self.index[1:]:
            s += '_{:02d}'.format(ind)
        return s

    @property
    def io(self):
        if hasattr(self, '_io'):
            return self._io
        elif self.parent is None:
            return None
        else:
            return self.parent.io

    @property
    def name(self):
        if self.parent is None:
            s0 = ''
        else:
            s0 = self.parent.name

        if hasattr(self, '_name'):
            s1 = self._name
        else:
            s1 = '{:02d}'.format(self.index[-1])

        if len(s0) > 0 and len(s1) > 0:
            return s0 + '_' + s1
        else:
            return s0 + s1

    @abstractmethod
    def execute(self, comm=SingleComm()):
        pass

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    def __str__(self):
        return self.name
        #print('str ',self.index2str)
        #return self.index2str

class MeanTreeNode(TreeNode):

    def __init__(self, parent, tseq, **kwargs):
        self.parent = parent
        self.value = None
        self.children = []
        self.tseq = tseq 
        
        #Default arguments
        args = {'K': 2000,
                'sig': 1.,
                'seed': 1000,
                **kwargs}
        for key, value in args.items():
            setattr(self, '_'+key, value)

        if self.parent is None:
            self.index = np.array([0], dtype=int)
        else:
            ind = len(self.parent.children)
            self.index = np.append(self.parent.index, ind)

    def build(self):
        
        # temporary observation operator
        rv_obs = GaussRV(C=np.ones((1,)) * 1e-8, mu=0)

        # Build model
        model = modelling.SpectralModel(L)  # m
        model.K = self._K
        rho = modelling.rho_true(L=L/(6*np.pi), b=8*np.pi/L)
        model.signal_factory(K=self._K, rho=rho, sig=self._sig,
                             seed=self._seed)

        # Coordinates in model.
        model.dyn_coord_factory('uniform', 2*self._K+1, 0)
        model.obs_coord_factory('single', rv_obs.M)

        # Initial conditions (irrelevant)
        x0 = np.zeros((2*self._K+1,))
        X0 = GaussRV(mu=x0, C=0)

        # State generator
        Dyn = {'M': 2*self._K+1, 'model': model.step, 'noise': 0,
               'linear': None}

        # Observations
        model.interpolator = model.state2fourier
        Obs = {'M': rv_obs.M,
               'model': model.obs_factory(model.interpolator, 0),
               'noise': rv_obs, 'linear': None}

        # Create DAPPER object.
        HMM = HiddenMarkovModel(Dyn, Obs, self.tseq, X0)
        HMM.model = model

        return HMM

    def execute(self, comm=SingleComm()):
        print(f'execute {self.name} on process {comm.rank}')
        
        #build it 
        self.operator = self.build()
        
        if hasattr(self.operator, 'noise'):
            self.operator.noise.member = Nens_max
            
        data = self.io.load(str(self))
        if data is None:
            self.value, _ = self.operator.simulate()
        else:
            self.value = data['value']
        if data is None and comm.rank==comm.procs[0]:
             self.io.save(str(self),
                          {'operator': self.operator, 'value': self.value})
             
        comm.execute(self.children)

    def add_truth(self, *args, **kwargs):            
        if 'factory' in kwargs and kwargs['factory']=='gauss':
            self.children.append(GaussTreeNode(self, *args, **kwargs))
        else:
            self.children.append(TruthTreeNode(self, *args, **kwargs))


class TruthTreeNode(TreeNode):

    def __init__(self, parent, **kwargs):
        self.parent = parent
        self.index = len(self.parent.children) + 1
        self.children = []
        self.value = None

        if self.parent is None:
            self.index = np.array([0], dtype=int)
        else:
            ind = len(self.parent.children)
            self.index = np.append(self.parent.index, ind)

        args = {'slope': -1.5,
                'sig': 1.,
                'seed': 2000,
                **kwargs}
        for key, value in args.items():
            setattr(self, '_'+key, value)

    def build(self):
        HMM = self.parent.operator
        model = deepcopy(HMM.model)
    
        noise = modelling.SpectralModel(L)
        noise.dyn_coords = model.dyn_coords
        noise.obs_coords = model.obs_coords
        for n in range(Nens_max):
            noise.red_noise_factory(K=model.K, slope=self._slope,
                                    sig=self._sig, seed=self._seed+n*100) 

        # State generator
        RV = RV_from_function(noise.sample_coords)
        Dyn = {'M': 2*model.K+1, 'model': model.step, 'noise': RV,
               'linear': None}

        HMM = HiddenMarkovModel(Dyn, HMM.Obs, HMM.tseq, HMM.X0)
        HMM.model, HMM.noise = model, noise

        return HMM

    def execute(self, comm=SingleComm()):
        print(f'execute {self.name} on process {comm.rank}')
        
        self.operator = self.build()
        
        data = self.io.load(str(self))
        if data is None:
            self.value, _ = self.operator.simulate()
        else:
            self.value = data['value']
        if data is None and comm.rank==comm.procs[0]:
            self.io.save(str(self),
                         {'operator': self.operator, 'value': self.value})

        comm.execute(self.children)

    @property
    def xx(self):
        return self.value

    @property
    def true_HMM(self):
        return self.operator

    def add_obs(self, Nobs, sig, *args, **kwargs):
        self.children.append(ObsTreeNode(self, Nobs, sig, *args, **kwargs))
        
class GaussTreeNode(TruthTreeNode):
    
    def __init__(self, parent, **kwargs):
        self.parent = parent
        self.index = len(self.parent.children) + 1
        self.children = []
        self.value = None

        if self.parent is None:
            self.index = np.array([0], dtype=int)
        else:
            ind = len(self.parent.children)
            self.index = np.append(self.parent.index, ind)

        args = {'length': 1.0e3,
                'sig': 1.,
                'seed': 2000,
                **kwargs}
        for key, value in args.items():
            setattr(self, '_'+key, value)
            
    def build(self):
        HMM = self.parent.operator
        model = deepcopy(HMM.model)
    
        noise = modelling.SpectralModel(L)
        noise.dyn_coords = model.dyn_coords
        noise.obs_coords = model.obs_coords
        for n in range(Nens_max):
            noise.gauss_noise_factory(K=model.K, length=self._length,
                                      sig=self._sig, seed=self._seed+n*100)

        # State generator
        RV = RV_from_function(noise.sample_coords)
        Dyn = {'M': 2*model.K+1, 'model': model.step, 'noise': RV,
               'linear': None}

        HMM = HiddenMarkovModel(Dyn, HMM.Obs, HMM.tseq, HMM.X0)
        HMM.model, HMM.noise = model, noise

        return HMM
    
class SmoothTreeNode(TreeNode):
    
    def __init__(self, parent, smoother=None, Lo=0, **kwargs):
        self.parent = parent 
        self.children = []
        self.value = None 
        
        args = {'smoother':smoother,
                'Lo':Lo,
                **kwargs}
        for key, value in args.items():
            setattr(self, '_'+key, value)
        
    def build(self):
        HMM = deepcopy(self.parent.HMM)
        
        if self._smoother=='uniform':
            HMM.model.apply_uniform_weighting(self._Lo)
            HMM.noise.model.apply_uniform_weighting(self._Lo)
        elif self._smoother=='gaussian':
            HMM.model.apply_gaussian_weighting(self._Lo)
            HMM.noise.model.apply_gaussian_weighting(self._Lo)
            
        return HMM
    
    def execute(self, comm=SingleComm()):
        print(f'execute {self.name} on process {comm.rank}')
        
        self.operator = self.build()
        data = self.io.load(str(self))
        if data is None:
            self.value, _ = self.operator.simulate()
        else:
            self.value = data['value']
        if data is None and comm.rank==comm.procs[0]:
            self.io.save(str(self),
                         {'operator': self.operator, 'value': self.value})

        comm.execute(self.children)

    @property 
    def smooth_xx(self):
        return self.value

class ObsTreeNode(TreeNode):

    def __init__(self, parent, Nobs, sig, **kwargs):
        self.parent = parent
        self.children = []
        self.value = None

        args = {'mu': 0,
                'obs_type': 'uniform',
                'localization': None,
                'Nobs': Nobs,
                'sig': sig,
                'sigo_inflation':1,
                **kwargs}
        for key, value in args.items():
            setattr(self, '_'+key, value)

        if self.parent is None:
            self.index = np.array([0], dtype=int)
        else:
            ind = len(self.parent.children)
            self.index = np.append(self.parent.index, ind)

    def build(self):
        rv_obs = GaussRV(C=np.ones((self._Nobs,)) * self._sig**2 * self._sigo_inflation**2, mu=self._mu)

        HMM = deepcopy(self.parent.operator)
        model = HMM.model
        model.obs_coord_factory(self._obs_type, rv_obs.M)
        obs = {'M': rv_obs.M, 'model': model.obs_factory(model.interpolator, 0),
               'noise': rv_obs, 'linear': None,
               'localization': self._localization}
        HMM.Obs = Operator(**obs)

        return HMM
    
    def step_smooth(self, x, Lo):        
        n = np.size(x,1)
        k = np.fft.fftfreq(n, L/n) * 2 * np.pi
        #filter
        w = np.ones_like(k)
        w[1:] = np.sin(k[1:]*0.5*Lo) / (k[1:]*0.5*Lo)
        #Transform to fourier space 
        Fx = np.fft.fft(x, axis=1) * w.reshape(1,-1)
        #Transform back 
        return np.real(np.fft.ifft(Fx, axis=1))

    def execute(self, comm=SingleComm()):
        print(f'execute {self.name} on process {comm.rank}')
        
        self.operator = self.build()
        xx = self.parent.value
        Obs = self.operator.Obs

        if hasattr(self, '_Lo'):
            xxL = self.step_smooth(xx, self._Lo)
        else:
            xxL = xx

        data = self.io.load(str(self))
        if data is None:
            self.value = []
            for ko, to in zip(self.operator.tseq.kko, self.operator.tseq.tto):
                s = Obs.noise.sample(1)
                yy1 = s + Obs(xxL[ko], to) / self._sigo_inflation
                if len(yy1) == 0:
                    raise ValueError("empty yy")
                self.value.append(np.reshape(yy1, (-1)))
        else:
            self.value = data['value']
        if data is None and comm.rank==comm.procs[0]:
            self.io.save(str(self),
                         {'operator': self.operator, 'value': self.value})
       
        comm.execute(self.children)

    @property
    def yy(self):
        return self.value

    def add_model(self, model_type, Ncells, *args, **kwargs):
        self.children.append(ModelTreeNode(self, model_type, Ncells,
                                           *args, **kwargs))


class ModelTreeNode(TreeNode):

    def __init__(self, parent, model_type, Ncells, *args, **kwargs):
        self.parent = parent
        self.children = []
        self.value = None

        args = {'localization': lambda model: None,
                'model_type':model_type,
                'Ncells':Ncells,
                'interpolator':'state2lin',
                'limiter':None,
                **kwargs}
        for key, value in args.items():
            setattr(self, '_'+key, value)

        if self.parent is None:
            self.index = np.array([0], dtype=int)
        else:
            ind = len(self.parent.children)
            self.index = np.append(self.parent.index, ind)

    def build(self):
        if self._model_type == 'lin':
            return self.build_lin(self._interpolator)
        elif self._model_type == 'dg':
            return self.build_dg()
        else:
            raise ValueError

    def build_lin(self, interpolator):
        N = self._Ncells
        if np.mod(N, 2) == 0:
            N += 1

        model = deepcopy(self.parent.operator.model)
        model.dyn_coord_factory('uniform', N, 0)
        model.apply_cutoff(N / (2*model.K+1)) 

        noise = deepcopy(self.parent.operator.noise)
        noise.dyn_coord_factory('uniform', N, 0)
        noise.apply_cutoff(N / (2*model.K+1)) 

        # Time steps
        tseq = self.parent.operator.tseq
        Obs = self.parent.operator.Obs

        # Initial conditions (irrelevant)
        x0 = np.zeros((N,))
        X0 = GaussRV(mu=x0, C=0)

        # State generator
        RV = RV_from_function(noise.sample_coords)
        Dyn = {'M': N, 'model': model.step, 'noise': RV,
               'linear': None}

        # Observations
        rv_obs = GaussRV(C=Obs.noise.C, mu=Obs.noise.mu)        
        model.interpolator = getattr(model, interpolator)
        Obs = {'M': rv_obs.M, 'model':  model.obs_factory(model.interpolator, 0),
               'noise': rv_obs, 'linear': None,
               'localization': self._localization(model)}

        HMM = HiddenMarkovModel(Dyn, Obs, tseq, X0)
        HMM.model, HMM.noise = model, noise

        return HMM

    def build_dg(self):
        N = (self._order+1) * self._Ncells

        model = deepcopy(self.parent.operator.model)
        model.dyn_coord_factory('legendre', self._Ncells, self._order)        
        model.apply_legendre()

        noise = deepcopy(self.parent.operator.noise)
        noise.dyn_coord_factory('legendre', self._Ncells, self._order)
        noise.apply_legendre()

        # Time steps
        tseq = self.parent.operator.tseq
        Obs = self.parent.operator.Obs

        # Initial conditions (irrelevant)
        x0 = np.zeros((N,))
        X0 = GaussRV(mu=x0, C=0)
        
        if self._limiter=='muscle':
            limiter = modelling.MuscleLimiter(model)
        else:
            limiter = modelling.VoidLimiter()

        # State generator
        rv_obs = GaussRV(C=Obs.noise.C, mu=Obs.noise.mu)
        RV = RV_from_function(noise.sample_legendre)
        Dyn = {'M': N, 'model': model.step_legendre, 'noise': RV,
               'linear': None, 'limiter':limiter}

        # Observations
        model.interpolator = model.state2legendre
        Obs = {'M': rv_obs.M, 'model': model.obs_factory(model.interpolator, 0),
               'noise': rv_obs, 'linear': None,
               'localization': self._localization(model)}

        HMM = HiddenMarkovModel(Dyn, Obs, tseq, X0)
        HMM.model, HMM.noise = model, noise

        return HMM

    def execute(self, comm=SingleComm()):
        print(f'execute {self.name} on process {comm.rank}')
        self.operator = self.build()
        
        data = self.io.load(str(self))
        if data is not None:
            self.value = data['value']
        if data is None and comm.rank==comm.procs[0]:
            self.io.save(str(self),
                         {'operator': self.operator, 'value': self.value})
          
        comm.execute(self.children)

    @property
    def HMM(self):
        return self.operator

    def add_xp(self, xp_type, Nens, *args, **kwargs):
        self.children.append(XpTreeNode(self, xp_type, Nens, *args, **kwargs))


class XpTreeNode(TreeNode):

    def __init__(self, parent, xp_type, Nens, *args, **kwargs):
        self.parent = parent
        self.children = []
        self.value = None
        
        if Nens>Nens_max:
            raise ValueError("Number of ensemble members larger than maximum.")

        args = {'xp_type':xp_type,
                'Nens':Nens,
                **kwargs}
        for key, value in args.items():
            setattr(self, '_'+key, value)
        self.kwargs = kwargs 

        if self.parent is None:
            self.index = np.array([0], dtype=int)
        else:
            ind = len(self.parent.children)
            self.index = np.append(self.parent.index, ind)

    def build(self):
        return self._xp_type(N=self._Nens, **self.kwargs)

    def execute(self, comm):
        print(f'execute {self.name} on process {comm.rank}')
        self.operator = self.build()
        M = self.parent.operator.Dyn.M
        
        #Dummy 'truth'
        xx = np.zeros((np.size(self.xx,0),M))
        
        data = self.io.load(str(self))
        if data is None:
            self.value = self.operator.assimilate(self.parent.operator, xx, 
                                                  self.yy)
        else:
            self.value = data['value']
            
        if data is None and comm.rank==comm.procs[0]:
            self.io.save(str(self),
                         {'operator': self.operator, 'value': self.value})
       
        comm.execute(self.children)

        print('Completed ', self.name)

    @property
    def xp(self):
        return self.operator

    @property
    def E(self):
        return {'for': self.value[0], 'ana': self.value[1]}


def calc_rmse(xp):
    r = np.linspace(0, L, 1000, endpoint=False)
    ndiff = 3

    interp_true = xp.true_HMM.model.interpolator
    interp = xp.HMM.model.interpolator

    xfor = np.mean(xp.E['for'], axis=1)
    xana = np.mean(xp.E['ana'], axis=1)

    rmse = {'for': np.zeros((ndiff, np.size(xana, 0))),
            'ana': np.zeros((ndiff, np.size(xana, 0)))
            }

    for n in range(ndiff):
        ftrue = interp_true(xp.xx[1:], ndiff=n)(r)
        fana = interp(xana, ndiff=n)(r)
        ffor = interp(xfor[1:], ndiff=n)(r)

        rmse['for'][n] = np.mean((ftrue-ffor)**2, axis=1)
        rmse['ana'][n] = np.mean((ftrue-fana)**2, axis=1)

    rmse['for'], rmse['ana'] = np.sqrt(rmse['for']), np.sqrt(rmse['ana'])

    return rmse


# Depreciated part
def create_comm():
    """Create MPI communicator."""
    mpi_info = {}
    comm = MPI.COMM_WORLD
    mpi_info['comm'] = comm
    mpi_info['rank'] = comm.Get_rank()
    mpi_info['size'] = comm.Get_size()

    return mpi_info


def create_mean_model(rv_obs, tseq, K=200, obs_type='uniform'):
    """ Create HMM object for background mean. """

    # Create DAPPER object generating background mean.
    model = modelling.SpectralModel(L)  # m
    model.K = K
    model.signal_factory(K=K, rho=modelling.rho_true(L=L/(6*np.pi), b=8*np.pi/L),
                         sig=10., seed=1000)

    # Coordinates in model.
    model.dyn_coord_factory('uniform', 2*K+1, 0)
    model.obs_coord_factory(obs_type, rv_obs.M)

    # Initial conditions (irrelevant)
    x0 = np.zeros((2*K+1,))
    X0 = GaussRV(mu=x0, C=0)

    # State generator
    Dyn = {'M': 2*K+1, 'model': model.step, 'noise': 0, 'linear': None}

    # Observations
    model.interpolator = model.state2fourier
    Obs = {'M': rv_obs.M, 'model': model.obs_factory(model.interpolator, 0),
           'noise': rv_obs, 'linear': None,
           'localization': nd_Id_localization([len(model.dyn_coords)])}

    # Create DAPPER object.
    HMM = HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.model = model

    return HMM


def create_truth_model(mean_model, rv_obs, Nens, slope=-1.5):
    """ Create HMM object for truth from which observations are taken. """
    model = deepcopy(mean_model.model)

    noise = modelling.SpectralModel(L)
    noise.dyn_coords = model.dyn_coords
    noise.obs_coords = model.obs_coords
    for n in range(Nens+1):
        noise.red_noise_factory(K=model.K, slope=slope,
                                sig=1., seed=2000 + n*100)

    # Time steps
    tseq = mean_model.tseq

    # Initial conditions (irrelevant)
    x0 = np.zeros((2*model.K+1,))
    X0 = GaussRV(mu=x0, C=0)

    # State generator
    RV = RV_from_function(noise.sample_coords)
    Dyn = {'M': 2*model.K+1, 'model': model.step, 'noise': RV,
           'linear': None}

    # Observations
    model.interpolator = model.state2fourier
    Obs = {'M': rv_obs.M, 'model': model.obs_factory(model.interpolator, 0),
           'noise': rv_obs, 'linear': None,
           'localization': None}

    HMM = HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.model, HMM.noise = model, noise

    return HMM


def create_smooth_model(truth, Lo, rv_obs):
    """ Create HMM object smoothed truth for superob observations. """
    model = deepcopy(truth.model)
    model.apply_uniform_weighting(Lo)
    noise = deepcopy(truth.noise)
    noise.apply_uniform_weighting(Lo)

    # Time steps
    tseq = truth.tseq

    # Initial conditions (irrelevant)
    x0 = np.zeros((2*model.K+1,))
    X0 = GaussRV(mu=x0, C=0)

    # State generator
    RV = RV_from_function(noise.sample_coords)
    Dyn = {'M': 2*model.K+1, 'model': model.step, 'noise': RV,
           'linear': None}

    # Observations
    model.interpolator = model.state2fourier
    Obs = {'M': rv_obs.M, 'model': model.obs_factory(model.interpolator, 0),
           'noise': rv_obs, 'linear': None,
           'localization': None}

    HMM = HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.model, HMM.noise, HMM.Lo = model, noise, Lo

    return HMM


def create_lin_model(truth, order, ncells, rv_obs):
    """ Create projection truth on lower dimensional Fourier space. """
    N = (order + 1) * ncells
    if np.mod(N, 2) == 0:
        N += 1

    model = deepcopy(truth.model)
    model.Ncells, model.order = ncells, order
    model.dyn_coord_factory('uniform', N, 0)
    model.apply_cutoff(N / (2*model.K+1))

    noise = deepcopy(truth.noise)
    noise.dyn_coord_factory('uniform', N, 0)
    noise.apply_cutoff(N / (2*model.K+1))

    # Time steps
    tseq = truth.tseq

    # Initial conditions (irrelevant)
    x0 = np.zeros((N,))
    X0 = GaussRV(mu=x0, C=0)

    # State generator
    RV = RV_from_function(noise.sample_coords)
    Dyn = {'M': N, 'model': model.step, 'noise': RV,
           'linear': None}

    # Observations
    model.interpolator = model.state2lin
    Obs = {'M': rv_obs.M, 'model':  model.obs_factory(model.interpolator, 0),
           'noise': rv_obs, 'linear': None, 'localization': None}

    HMM = HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.model, HMM.noise = model, noise

    return HMM


def create_dg_model(truth, order, ncells, rv_obs, slope_limiter=None):
    """ Create projection truth on DG. """
    N = (order+1) * ncells

    model = deepcopy(truth.model)
    model.Ncells, model.order = ncells, order
    model.dyn_coord_factory('legendre', ncells, order)
    model.apply_legendre()

    noise = deepcopy(truth.noise)
    noise.dyn_coord_factory('legendre', ncells, order)
    noise.apply_legendre()

    # Time steps
    tseq = truth.tseq

    # Initial conditions (irrelevant)
    x0 = np.zeros((N,))
    X0 = GaussRV(mu=x0, C=0)

    # State generator
    RV = RV_from_function(noise.sample_legendre)    
    Dyn = {'M': N, 'model': model.step_legendre, 'noise': RV,
           'linear': None}

    # Observations
    model.interpolator = model.state2legendre
    Obs = {'M': rv_obs.M, 'model': model.obs_factory(model.interpolator, 0),
           'noise': rv_obs, 'linear': None, 'localization': None}

    HMM = HiddenMarkovModel(Dyn, Obs, tseq, X0)
    HMM.model, HMM.noise = model, noise

    return HMM


def stats_rmse(HMM, xx, Efor, Eana):
    """ Calculate RMSE. """
    from scipy.integrate import quadrature
    xp = np.linspace(0, L, 1000, endpoint=False)

    rms = {}
    for key in HMM:
        rms[key] = {'for': np.zeros((3, np.size(Efor[key], 0))),
                    'ana': np.zeros((3, np.size(Eana[key], 0)))}

    for ito, it in enumerate(HMM[key].tseq.kko):
        for ndiff in range(3):

            true = HMM['truth'].model.interpolator(
                xx['truth'][it], ndiff=ndiff)
            for key in HMM:
                if hasattr(HMM[key], 'Ncells'):
                    points = np.linspace(0, L, HMM[key].Ncells, endpoint=False)
                else:
                    points = None
                points = None

                ana1 = HMM[key].model.interpolator(
                    np.mean(Eana[key][ito, :, :], axis=0), ndiff=ndiff)

                def error2(x): return (ana1(x)-true(x))**2
                rms[key]['ana'][ndiff, ito] = np.sqrt(np.mean(error2(xp)))

                for1 = HMM[key].model.interpolator(
                    np.mean(Efor[key][it, :, :], axis=0), ndiff=ndiff)

                def error2(x): return (for1(x)-true(x))**2
                rms[key]['for'][ndiff, ito] = np.sqrt(np.mean(error2(xp)))

    return rms


def stats_spread(HMM, Efor, Eana):
    """ Calculate RMSE. """
    from scipy.integrate import quadrature
    xp = np.linspace(0, L, 1000, endpoint=False)

    spread = {}
    for key in HMM:
        spread[key] = {'for': np.zeros((np.size(Efor[key], 0),)),
                       'ana': np.zeros((np.size(Eana[key], 0),))}

    for ito, it in enumerate(HMM[key].tseq.kko):
        for key in HMM:
            if hasattr(HMM[key], 'Ncells'):
                points = np.linspace(0, L, HMM[key].Ncells, endpoint=False)
            else:
                points = None
            points = None

            ana1 = HMM[key].model.interpolator(Eana[key][ito, :, :], ndiff=0)
            def error2(x): return np.var(ana1(x), ddof=1)
            spread[key]['ana'][ito] = np.sqrt(np.mean(error2(xp)))

            for1 = HMM[key].model.interpolator(Efor[key][it, :, :], ndiff=0)
            def error2(x): return np.var(for1(x), ddof=1)
            spread[key]['for'][ito] = np.sqrt(np.mean(error2(xp)))

    return spread


def filepath(dir_path, order, Nobs):
    """Generate filepath."""

    fname = "red15_{:02d}_{:03d}.pkl".format(int(order), int(Nobs))
    return os.path.join(dir_path, fname)
