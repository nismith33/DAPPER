"""
Module containing model equations for dimensionless Stommel model. 
"""

import numpy as np
from scipy.optimize import fsolve
from dapper.mods.integration import rk4
import dapper.mods as modelling
import matplotlib.pyplot as plt
import dataclasses
import os
import pickle as pkl
from abc import ABC, abstractmethod

#Directory to store figures. 
fig_dir = "/home/ggorblin/dpr_data"
hadley_file = "/home/ggorblin/dpr_data/stommel/boxed_hadley.pkl"
if not os.path.exists(hadley_file):
    raise FileExistsError("Generate a file with Hadley EN4 output using tools/hadley_obs.")
    
#Import values based on Hadley EN4 observations. 
with open(hadley_file, 'rb') as stream:
    hadley = pkl.load(stream)
    ref = np.mean(hadley['yy'], axis=0)
    cross_area = 0.5*(hadley['geo_pole']['dx'] * hadley['geo_pole']['dz'] +
                      hadley['geo_eq']['dx'] * hadley['geo_eq']['dz'])

mm2m = 1e-3 #convert millimeter to meter
year = 86400 * 365.25 #convert year to seconds
func_type = type(lambda:0.0)

def harmonic2cos(Acos, Asin):
    """ Convert fit in form
        Acos cos(omega*t) + Asin sin(omega*t)
    into 
        A cos(omega*t - phase)
    """
    A = np.hypot(Asin, Acos)
    phase = np.arctan2(Asin, Acos)
    return A, phase

def Bhat(A,B):
    """
    Calculate Bhat from A and B as defined in Budd et al. 
    """
    return B - A

def budd_scale(model, state, Omega, B, Bhat, epsilon):
    """ 
    Transform dimensionless parameters in 
    
    Budd, C, Griffith, C & Kuske, R 2022, 'Dynamic tipping in the non-smooth Stommel-box model, with fast
    oscillatory forcing', Physica D: Nonlinear Phenomena, vol. 432, 132948.
    https://doi.org/10.1016/j.physd.2021.132948
    
    to parameters with dimensions. 
    
    Parameters
    ----------
    model : StommelModel object
        Object that contains parameters model like volume. 
    state : State object
        Object that contains state parameters like diffusion coefficients. 
    Omega : float 
        Nondimensional angular frequency periodic forcing. See Budd et al. 
    B : float 
        Nondimensional amplitude eta_2. See Budd et al. 
    Bhat : float 
        Nondimensional amplitude eta_1. See Budd et al. 
    epsilon : float 
        Nondimensional change rate eta_1. See Budd et al. 
        
    Returns
    -------
    period:
        Oscillation period is s. 
    amplitude_T:
        Amplitude of temperature oscillations in C. 
    amplitude_S:
        Amplitude of salinity oscillations in ppt. 
    epsilon:
        Salinity increase rate pole in ppt/second. 
        
    """
    
    #Dimensionless -> dimensional
    period = 2*np.pi/ Omega * model.time_scale(state)
    #amplitude temperature change due to periodic change eta1
    amplitude_T = 0.5 * B    * model.temp_scale(state) 
    #amplitude salinity change due to periodic change eta2
    amplitude_S = 0.5 * Bhat * model.salt_scale(state) / model.eta3(state)
    #rate salinity change due to  
    epsilon = epsilon * model.salt_scale(state) / model.eta3(state) #linear change eta2
    epsilon = epsilon / model.time_scale(state) #linear change eta2
    
    return period, amplitude_T, amplitude_S, epsilon

def budd_forcing(model, state, Omega, B, Bhat, epsilon=0.0):
    """Create forcing functions based on 
    
    Budd, C, Griffith, C & Kuske, R 2022, 'Dynamic tipping in the non-smooth Stommel-box model, with fast
    oscillatory forcing', Physica D: Nonlinear Phenomena, vol. 432, 132948.
    https://doi.org/10.1016/j.physd.2021.132948
    
    Parameters
    ----------
    model : StommelModel object
        Object that contains parameters model like volume. 
    state : State object
        Object that contains state parameters like diffusion coefficients. 
    Omega : float 
        Nondimensional angular frequency periodic forcing. See Budd et al. 
    B : float 
        Nondimensional amplitude eta_2. See Budd et al. 
    Bhat : float 
        Nondimensional amplitude eta_1. See Budd et al. 
    epsilon : float 
        Nondimensional change rate eta_1. See Budd et al. 
    
    """
   
    #Add dimensions. 
    period, amplitude_T, amplitude_S, epsilon = budd_scale(model, state, Omega, B, Bhat, epsilon) 
    
    #Angular frequency in Hz
    Omega = 2 * np.pi / period
    
    #Forcing surface temperature to achieve fluctuations eta1
    temp_forcings = [lambda time : np.array([-1.,1.]) * amplitude_T * np.sin(Omega * time)]
    #Forcing surface salinity to achieve fluctuations eta2
    salt_forcings = [lambda time : np.array([-1.,1.]) * amplitude_S * np.sin(Omega * time) 
                     - np.array([-1.0,0.0]) * epsilon * time]
    
    return temp_forcings, salt_forcings
    

def sample2linear(model):
    """Convert observation functional into matrix operator."""
    
    def linear(x, t):
        H = []
        for no in range(np.size(x)):
            x0 = np.zeros_like(x)
            x0[no] = 1.0
            H.append(model(x0,t))
        return np.array(H).T
    
    return linear
    

@dataclasses.dataclass
class LinearEquationState:
    """ Class representing linear equation of state for ocean water. """
    
    #Settings 
    rho_ref: float = 1027 #kg m-3
    T_ref: float = 10.0 #C
    S_ref: float = 35.0 #ppt
    alpha_T: float =  0.15 / rho_ref #kg m-3 K-1 / (kgm-3)
    alpha_S: float =  0.78 / rho_ref #kg m-3 ppt-1 /(kgm-3
    
    def __call__(self, T, S):
        """ Linear equation of state. 
        
        Parameters
        ----------
        T : float 
            Ocean temperature
        S : float 
            Ocean salinity
            
        """            
        return  self.rho_ref * (1 - self.alpha_T * (T-self.T_ref) +
                self.alpha_S * (S-self.S_ref))
                  
#Meriodional overturning 
Q_overturning = 18e6 #m3s-1
#Estimated depth mixed layer
mixing_depth = 10.0 #m (based on 2019 Hadley data)
#Estimated overturning
eos = LinearEquationState()
rho_pole = eos(ref[0], ref[2])
rho_eq = eos(ref[1], ref[3])
gamma_ref = Q_overturning * eos.rho_ref / (rho_pole - rho_eq) / cross_area
#Default diffusion 
eta = np.array([np.nan,np.nan,.3])
temp_diff = 1.e-5  
salt_diff = eta[2] * temp_diff
#Ice volume greenland ice sheet 
V_ice = 1710000 * 2 * 1e9 #m3


@dataclasses.dataclass 
class State:
    """Class containing all attributes that make up a physical state."""
    
    #temperature in ocean basin
    temp: np.ndarray = np.array([ref[0:2]]) #C 6,18
    #salinity in ocean basis
    salt: np.ndarray = np.array([ref[2:4]]) #ppt 35,36.5
    #surface heat flux coefficient
    temp_diff: np.ndarray = temp_diff / mm2m
    #surface salinity flux coefficient
    salt_diff: np.ndarray = salt_diff / mm2m
    #advective transport flux ceofficient 
    gamma: float = 0.0
    #time associated with state
    time: float = 0.0
    
    def to_vector(self):
        """Convert state into a 1D array."""
        v = np.array([], dtype=float)
        for key, value in dataclasses.asdict(self).items():
            if key!='time':
                v = np.append(v, value)
        return v
    
    def from_vector(self, v):
        """Read data from 1D into State object."""
        lb = 0
        for key, value in dataclasses.asdict(self).items():
            if key!='time':
                ub = lb + max(1,np.size(value))
                setattr(self, key, np.reshape(v[lb:ub], np.shape(value)))
                lb = ub
                
        return self
            
    def zero(self):
        """Set all values in State object to zero."""
        for key, value in dataclasses.asdict(self).items():
            if key!='time':
                setattr(self, key, value * 0.0)
                
        return self
          
    @property
    def regime(self):
        """~/eturn the regime for circulation."""
        
        rho = StommelModel.eos(self.temp[0], self.salt[0])
        if np.diff(rho)<=0:
            return 'TH' #thermohaline circulation as in present
        else:
            return 'SA' #contra solution
        
    @property 
    def N(self):
        """Return size of state vector."""
        return np.size(self.to_vector())
                
def array2states(array, times=np.nan):
    """Convert array with data to array of State objects."""
    shape = np.shape(array)
    array = np.reshape(array,(-1,shape[-1]))
    
    if np.ndim(times)==0:
        times *= np.ones((np.size(array,0)))
    
    states = np.array([State() for v in array], dtype=State)
    for n,v in enumerate(array):
        states[n].from_vector(v)
        states[n].time = times[n]
        
    states = np.reshape(states, shape[:-1])
        
    return states

def states2array(states):
    """Convert array of State objects into an array."""
    return np.array([s.to_vector() for s in states], dtype=float)

class Flux:
    """Class representing fluxes in model."""
    
    def __init__(self):
        self.ens_member = 0
    
    def left(self, state):
        """Return flux flowing into cell from left."""
        return State().zero()
    
    def right(self, state):
        """Return flux exiting cell to right."""
        return State().zero()
    
    def top(self, state):
        """Return flux exiting cell via top."""
        return State().zero()
    
    def bottom(self, state):
        """Return flux entering cell from bottom."""
        return State().zero()
        
class AdvectiveFlux(Flux):
    """ Class representing advective transport flux between pole and equator."""
    
    def __init__(self, eos):
        """Set equation-of-state."""
        self.eos = eos
    
    def transport(self, state):
        #Density 
        rho = self.eos(state.temp, state.salt)
        #Transport volume
        return np.abs(state.gamma * np.diff(rho, axis=1) / self.eos.rho_ref)
    
    def left(self, state):
        flux = State().zero()
        trans = self.transport(state)
        flux.temp[:,1:] += trans * (state.temp[:,:-1] - state.temp[:,1:])
        flux.salt[:,1:] += trans * (state.salt[:,:-1] - state.salt[:,1:])
        return flux
    
    def right(self, state):
        flux = State().zero()
        trans = self.transport(state)
        flux.temp[:,:-1] += trans * (state.temp[:,:-1] - state.temp[:,1:])
        flux.salt[:,:-1] += trans * (state.salt[:,:-1] - state.salt[:,1:])
        return flux

class FunctionTempFlux(Flux):
    """ Class representing a prescripted surface heat flux. """
    
    def __init__(self, functions):
        super().__init__()
        self.functions = functions 
        
    def top(self, state):
        n = np.mod(self.ens_member, len(self.functions))
        flux = State().zero()
        flux.temp[0] += self.functions[n](state.time)
        return flux
    
class FunctionSaltFlux(Flux):
    """ Class representing a prescripted surface salinity flux. """
    
    def __init__(self, functions):
        super().__init__()
        self.functions = functions 
        
    def top(self, state):
        n = np.mod(self.ens_member, len(self.functions))
        flux = State().zero()
        flux.salt[0] += self.functions[n](state.time)
        return flux
    
class TempAirFlux(Flux):
    """Class representing heat flux through top of ocean."""
    
    def __init__(self, functions):
        super().__init__()
        self.functions = functions 
        
    def top(self, state):
        n = np.mod(self.ens_member, len(self.functions))
        flux = State().zero()
        flux.temp[0] -= state.temp_diff * (self.functions[n](state.time) - state.temp[0]) * mm2m
        return flux
        
class SaltAirFlux(Flux):
    """Class representing salinity flux through top of ocean."""
    
    def __init__(self, functions):
        super().__init__()
        self.functions = functions 
        
    def top(self, state):
        n = np.mod(self.ens_member, len(self.functions))
        flux = State().zero()
        flux.salt[0] -= state.salt_diff * (self.functions[n](state.time) - state.salt[0]) * mm2m
        return flux
        
class EPFlux(Flux):
    """Class representing evaporation/percipitation flux top of ocean."""
    
    def __init__(self, functions=np.array([lambda t: np.array([0.0, 0.0])], dtype=func_type) ):
        super().__init__()
        self.functions = functions 
        
    def top(self, state):
        n = np.mod(self.ens_member, len(self.functions))
        flux = State().zero()
        flux.salt[0] -= state.salt[0] * self.functions[n](state.time)   
        return flux
        
@dataclasses.dataclass
class StommelModel:
    """Class containing all attributes and methods to represent the Stommel model."""
    
    #Geometry of basin 
    pole, eq = hadley['geo_pole'], hadley['geo_eq']
    dz: np.ndarray = np.array([[pole['dz'], eq['dz']]]) #m depth
    dy: np.ndarray = np.array([[pole['dy'], eq['dy']]]) #m latitude
    dx: np.ndarray = np.array([[pole['dx'], eq['dx']]]) #m longitude 
    V: np.ndarray = dx * dy * dz
    
    #Time
    time: float = 0.0
    #Initial state 
    init_state: State = State()
    #Memory for model state
    state: State = State()
    #Equation of state
    eos: LinearEquationState = LinearEquationState()
    #Ensemble member 
    _ens_member = 0
    #Fluxes 
    fluxes = []
    
    @property 
    def ens_member(self):
        """Get index currently active ensemble member."""
        return self._ens_member 
    
    @ens_member.setter 
    def ens_member(self, member):
        """Set index currently active ensemble member."""
        self._ens_member = member 
        for flux in self.fluxes:
            flux.ens_member = member 
    
    def __post_init__(self):
        """Part of object initialization to be carried out after __init__ provided by dataclass."""
        self.init_state.gamma = self.default_gamma(self.init_state)
        self.state.gamma = self.default_gamma(self.state)
        self.fluxes = self.default_fluxes()
        
    def default_gamma(self, state, Q=Q_overturning):
        """Reverse engineer advective flux coefficient gamma using temperature/salinity fields in state
        and meriodional overturning discharge Q."""
        
        rho  =  self.eos(state.temp, state.salt)
        drho =  np.diff(np.sum(self.V * rho, axis=0)/np.sum(self.V, axis=0))[0]
        rho0 = self.eos.rho_ref
        
        area = np.sum(np.mean(self.dx, axis=1) * np.mean(self.dz, axis=1))
        
        return -Q * (rho0 / drho) / area 
    
    def default_fluxes(self):
        """Set default fluxes."""
        return [AdvectiveFlux(self.eos)]
    
    def obs_hadley(self):
        #Size of observations.
        M = 2*np.size(self.dz, 1)
        
        #Function for taking a observation from single state. 
        def obs_TS1(x, t):
            self.state.time = t
            self.state.from_vector(x)
            return np.append(self.state.temp[0], self.state.salt[0])
        
        #Function for taking observation from ensemble of states. 
        def obs_model(x, t):            
            if np.ndim(x)==1:
                return obs_TS1(x, t)
            elif np.ndim(x)==2:
                return np.array([obs_TS1(x1,t) for x1 in x])
            else:
                msg = "x must be 1D or 2D array."
                raise TypeError(msg)
          
        
        #DAPPER Observation operator
        Obs = {'M':M, 'model': obs_model, 'linear': sample2linear(obs_model),
               'noise': modelling.GaussRV(C=hadley['R'], 
                                          mu=np.zeros_like(hadley['R']))}
        
        return Obs
    
    def obs_ocean(self, sig_temp=0.5, sig_salt=0.05):
        """ Sampling operator for ocean temperature and salinity. """
        
        #Size of observations.
        M = 2*np.size(self.dz, 1)
        
        #Function for taking a observation from single state. 
        def obs_TS1(x, t):
            self.state.time = t
            self.state.from_vector(x)
            return np.append(self.state.temp[0], self.state.salt[0])
        
        #Function for taking observation from ensemble of states. 
        def obs_model(x, t):            
            if np.ndim(x)==1:
                return obs_TS1(x, t)
            elif np.ndim(x)==2:
                return np.array([obs_TS1(x1,t) for x1 in x])
            else:
                msg = "x must be 1D or 2D array."
                raise TypeError(msg)
          
        #Observation error variance  
        R = np.append(np.ones_like(self.state.temp[0]) * sig_temp**2,
                      np.ones_like(self.state.salt[0]) * sig_salt**2) # C2,C2,ppt2,ppt2
        
        #DAPPER Observation operator
        Obs = {'M':M, 'model': obs_model, 'linear': sample2linear(obs_model),
               'noise': modelling.GaussRV(C=R, mu=np.zeros_like(R))}
        
        return Obs
    
    def tendency(self, state):
        """Calculate tendency (d/dt) for the model."""
        
        #Empty tendency
        tendency = State().zero()
        
        #Convergence vertical fluxes
        for flux in self.fluxes:
            top, bottom = flux.top(state), flux.bottom(state)
            tendency.temp -= (top.temp - bottom.temp) / self.dz 
            tendency.salt -= (top.salt - bottom.salt) / self.dz
    
        #Cross-section cells 
        Aleft = self.dz[:,:-1] * self.dx[:,:-1] 
        Aright = self.dz[:,1:] * self.dx[:,1:] 
        
        
        #Convergence horizontal fluxes. 
        for flux in self.fluxes:
            left, right = flux.left(state), flux.right(state)
            tendency.temp -= (right.temp * Aright - left.temp * Aleft ) / self.V  
            tendency.salt -= (right.salt * Aright - left.salt * Aleft ) / self.V 
            
        return tendency
    
    #Total tendency 
    def dxdt1(self, x, t):
        """Calculate tendency for 1 state."""
        self.state.time = t
        self.state.from_vector(x)
        
        return self.tendency(self.state).to_vector()
    
    #Forward model step for 1 ensemble member
    def step1(self, x0, t, dt):
        """ Step 1 model state forward in time using 4th order Runge-Kutta method. """ 
        return rk4(lambda x, t: self.dxdt1(x,t), x0, t, dt)
       
    #Forward model step an ensemble 
    def step(self, x, t, dt):
        """ Step all model states forward. """
        if np.ndim(x)==1:
            x = self.step1(x, t, dt)
        elif np.ndim(x)==2 and np.size(x,0)==1:
            self.ens_member = 0
            x[0] = self.step1(x[0], t, dt)
        elif np.ndim(x)==2:
            for no, x1 in enumerate(x):
                self.ens_member = no+1
                x[no] = self.step1(x1, t, dt)
        else:
            msg = "x must be 1D or 2D array."
            raise TypeError(msg)
        return x
    
    @property 
    def M(self):
        """Dimension of the model state."""
        return len(self.x0)
    
    def eta1(self, state):
        """Non-dimensional parameter eta1. See Dijkstra (2008)."""
        temp_air = state.temp[0] * 0.0
        for flux in self.fluxes:
            if isinstance(flux, TempAirFlux):
                temp_air = flux.functions[flux.ens_member](state.time)
        
        dtemp = np.diff(temp_air)[0]
        return dtemp / self.temp_scale(state)
    
    def eta2(self, state):
        """Non-dimensional parameter eta2. See Dijkstra (2008)."""
        salt_air = state.salt[0] * 0.0
        for flux in self.fluxes:
            if isinstance(flux, SaltAirFlux):
                salt_air = flux.functions[flux.ens_member](state.time)
        
        dsalt = np.diff(salt_air)[0]
        return (dsalt / self.salt_scale(state)) * self.eta3(state)
            
    def eta3(self, state):
        """Non-dimensional parameter eta3. See Dijkstra (2008)."""
        R_T = np.mean(state.temp_diff / self.dz[0]) * mm2m
        R_S = np.mean(state.salt_diff / self.dz[0]) * mm2m
        return R_S/ R_T
    
    def temp_scale(self, state):
        """Factor to transform nondimensional to dimensional temperature."""
        A = np.mean(self.dx[0]) * np.mean(self.dz[0])
        gamma = state.gamma * A
        return self.trans_scale(state) / (gamma * self.eos.alpha_T)
    
    def salt_scale(self, state):
        """Factor to transform nondimensional to dimensional salinity."""
        A = np.mean(self.dx[0]) * np.mean(self.dz[0])
        gamma = state.gamma * A
        return self.trans_scale(state) / (gamma * self.eos.alpha_S)  
        
    def trans_scale(self, state):
        """Factor to transform nondimensional to dimensional advective transport."""
        R_T = np.mean(state.temp_diff * mm2m / self.dz[0])
        V = np.prod(self.V[0]) / np.sum(self.V[0])
        return R_T * V
    
    def time_scale(self, state):
        """ Factor to transform nondimensional time to dimensional time. """
        return 1 / np.mean(state.temp_diff * mm2m / self.dz[0])

    @property        
    def x0(self):
        """Initial conditions."""
        return self.init_state.to_vector()
    
    def temp_eq(self, state, trans_eq):
        """Temperature difference equator-pole in equilibrium in C."""
        trans_eq = trans_eq / self.trans_scale(state)
        temp_eq  = self.eta1(state) / (1 +  np.abs(trans_eq))  
        return temp_eq * self.temp_scale(state)
        
    def salt_eq(self, state, trans_eq):
        """Salinity difference equator_pole in equilibrium in ppt."""
        trans_eq = trans_eq / self.trans_scale(state)
        salt_eq  = self.eta2(state) / (self.eta3(state) + np.abs(trans_eq))    
        return salt_eq * self.salt_scale(state)
    
    def trans_eq(self, state):
        """Meriodional transport pole->equator in equilibrium in m3."""  
        from scipy.optimize import root_scalar as find_roots
        import matplotlib.pyplot as plt
        
        #Function of transport that is equal to zero in equilibrium.
        f = lambda q: (q - self.eta1(state) / (1+np.abs(q)) + 
                       self.eta2(state) / (self.eta3(state)+np.abs(q)))
        
        roots = []
        q  = np.linspace(-3,3,30000) 
        while len(roots)==0: 
            #Evaluate values on grid. 
            fq = f(q)
            q=.5*q[1:]+.5*q[:-1]
            fq=fq[1:]*fq[:-1]
        
            #Find roots
            roots = [q1 for q1,f1 in zip(q,fq) if f1<0.]
            
            #Expand search area
            q *= 2
        
        return np.array(roots) * self.trans_scale(state) 
    
def default_air_temp(N):
    """ Unperturbed air temperature. """
    return np.array([lambda t: hadley['temperature'] for _ in range(N+1)], dtype=func_type)
    
def default_air_salt(N):
    """ Unperturbed air salinity."""
    return np.array([lambda t: hadley['salinity'] for _ in range(N+1)], dtype=func_type)

def default_air_ep(N, ep=np.array([0.,0.])):
    """ Unperturbed air salinity."""
    return np.array([lambda t: ep for _ in range(N+1)], dtype=func_type)

def hadley_air_temp(N):
    """ Perturbed air temperature. """
    def func(t):
        values  = hadley['surface_temperature_harmonic'][0] + 0. #0. to copy array!
        for omega, coef  in zip(hadley['harmonic_angular'], hadley['surface_temperature_harmonic'][1::2]):
            values += coef * np.cos(omega * t)
        for omega, coef  in zip(hadley['harmonic_angular'], hadley['surface_temperature_harmonic'][2::2]):
            values += coef * np.sin(omega * t)   
        return values
    
    return np.array([func for _ in range(N+1)], dtype=func_type)

def hadley_air_salt(N):
    """ Perturbed air salinity. """
    def func(t):
        values  = hadley['surface_salinity_harmonic'][0] + 0. #0. to copy array
        for omega, coef  in zip(hadley['harmonic_angular'], hadley['surface_salinity_harmonic'][1::2]):
            values += coef * np.cos(omega * t)
        for omega, coef  in zip(hadley['harmonic_angular'], hadley['surface_salinity_harmonic'][2::2]):
            values += coef * np.sin(omega * t)   
        return values
    
    return np.array([func for _ in range(N+1)], dtype=func_type)

def merge_functions(T, func1, func2):
    """ Merge two functions with different time domains into 1."""
    
    def merged_func(t):
        if t<=T:
            return func1(t)
        else:
            return func2(t)
        
    return merged_func

def add_noise(func, seed, sig):
    """ Add white noise to a forcing function."""
    
    def noised_func(t):
        np.random.seed(seed + int(t/86400))
        perturbation = np.random.normal(size=np.shape(sig)) * sig
        return func(t) + perturbation
        
    return noised_func    

def add_functions(func, func_add):
    """ Add two functions. """
    
    def trend_func(t):
        return func(t)+func_add(t)
    
    return trend_func

    
# Display model information
def display(model):
    state = model.init_state
    print('Model information:')
    print('eta1: {:.2f}'.format(model.eta1(state)))
    print('eta2: {:.2f}'.format(model.eta2(state)))
    print('eta3: {:.2f}'.format(model.eta3(state)))

    trans_eq = model.trans_eq(state)
    temp_eq = model.temp_eq(state, trans_eq)
    salt_eq = model.salt_eq(state, trans_eq)

    eq_str = '{:16s} {:16s} {:16s}'.format(
        'transport', 'temperature diff', 'salinity diff')
    print('\n' + eq_str)
    eq_str = '{:16s} {:16s} {:16s}'.format('[Sv]', '[C]', '[ppt]')
    print(eq_str)
    for trans, temp, salt in zip(trans_eq, temp_eq, salt_eq):
        print('{:16.4e} {:16.2f} {:16.3f}'.format(trans*1e-6, temp, salt))
    print()


def add_melt(func, model, sig_T=0.0):
    """ Add melt Greenland ice sheet to salinity. """
    V_ice = 1710000 * 2 * 1e9 * np.array([1.0, 0.0]) #m3
    T = np.random.normal(loc=100., scale=sig_T) * year
    
    rate = V_ice / (T * model.dx[0] * model.dy[0]) 

    def melted_func(t):
        if t<T:
            return func(t) - rate 
        else:
            return func(t)
    
    return melted_func

def add_warming(func, mu=np.array([3., 2.]), sigs=np.array([0., 0.])):
    T = 100 * year
    rate  = np.array([1., 0.])
    rate *= np.random.normal(loc=mu[1], scale=sigs[1])
    rate += np.random.normal(loc=mu[0], scale=sigs[0])
    rate /= T #C/s
    def warmed_func(t):
        return func(t) + min(t, T) * rate 
    return warmed_func   

def time_figure(tseq):
    plt.close('all')
    fig = plt.figure(figsize=(8, 4))
    ax = fig.subplots(1, 2)
    
    times = tseq.tt/year
        
    for ax1 in ax:
        ax1.grid()
        ax1.set_xlim((0, times[-1]))
        ax1.set_xlabel('Time [year]')
        
    ax[0].set_ylabel('Temperature diff. [C]')
    ax[1].set_ylabel('Salinity diff. [ppt]')
    return fig, ax

#modify time_figure, adding a third plot containing T-vs-S phase portrait
def time_figure_with_phase(tseq):
    plt.close('all')
    fig = plt.figure(figsize=(14, 4))
    ax = fig.subplots(1, 3)
    
    times = tseq.tt/year
        
    for ax1 in ax[0:2]:
        ax1.grid()
        ax1.set_xlim((0, times[-1]))
        ax1.set_xlabel('Time [year]')
    #ax[2].set_xlim((0,2.8))
    #ax[2].set_ylim((0,2.8))
    ax[2].set_aspect('equal', adjustable='box')
        
    ax[0].set_ylabel('Temperature diff. [C]')
    ax[1].set_ylabel('Salinity diff. [ppt]')
    ax[2].set_ylabel('Dimensionless Temperature')
    ax[2].set_xlabel('Dimensionless Salinity')
    return fig, ax

#Plot T vs t, S vs t, and T vs S phase portrait. The T and S in the phase portrait
#are rescaled to a dimensionless form.
def plot_truth_with_phase(ax,model,xx,yy):
    times = np.linspace(ax[0].get_xlim()[0], ax[0].get_xlim()[1], np.size(xx,0)) 
    
    states=array2states(xx,times)
    TH = np.array([s.regime=='TH' for s in states], dtype=bool)
    temp = np.reshape(np.diff([s.temp[0] for s in states],axis=1), (-1))
    salt = np.reshape(np.diff([s.salt[0] for s in states],axis=1), (-1))
    
    #Rescale the salt and heat
    temp_divisors = np.array([model.temp_scale(state) for state in states])
    salt_divisors = np.array([model.salt_scale(state) for state in states])
    scaled_temp = np.divide(temp ,temp_divisors)
    scaled_salt = np.divide(salt ,salt_divisors)
    #
    
    #TH
    mask = np.where(~TH, np.nan, 1.)
    ax[0].plot(times, temp * mask, 'b-', alpha=.7)
    ax[1].plot(times, salt * mask, 'b-', alpha=.7)
    ax[2].plot(scaled_salt * mask, scaled_temp * mask, 'b-', alpha=.7)
    
    #SA
    mask = np.where(TH, np.nan, 1.)
    ax[0].plot(times, temp * mask, 'r-', alpha=.7)
    ax[1].plot(times, salt * mask, 'r-', alpha=.7)
    ax[2].plot(scaled_salt * mask, scaled_temp * mask, 'r-', alpha=.7)
    
    #accentuate initial conditions in phase portrait
    ax[2].plot(scaled_salt[0],scaled_temp[0],'go')
    
    #Plot T=S line for reference
    axlim = max(scaled_temp.max(),scaled_salt.max())
    ax[2].plot(np.linspace(0,axlim),np.linspace(0,axlim), color="m", linestyle=':')
    
    
    if len(yy)>0:
        timeos = times[1:len(yy)+1]
        std_temp = np.sqrt(np.sum(hadley['R'][:2]))
        std_salt = np.sqrt(np.sum(hadley['R'][2:]))
        
        for to,y in zip(timeos,yy):
            ax[0].errorbar(to,np.diff(y[0:2]),std_temp,color='orange')
            ax[1].errorbar(to,np.diff(y[2:4]),std_salt,color='orange')
            #add dots at actual measurement
            ax[0].plot(to,np.diff(y[:2]), 'o', markersize=2, color='orange')
            ax[1].plot(to,np.diff(y[2:]), 'o', markersize=2, color='orange')

#plots all equilibrium solutions at every timestep.
def plot_all_eq(ax, tseq, model, xx, p=None):
    times = tseq.tt/year
    
    # Equilibrium values
    states=array2states(xx,times)
    for i in range(len(times)):
        state = states[i]
        trans_eq = model.trans_eq(state)
        temp_eq = model.temp_eq(state, trans_eq)
        salt_eq = model.salt_eq(state, trans_eq)

        for T, S in zip(temp_eq, salt_eq):
            ax[0].plot(times[i], T, 'go', markersize=1)
            ax[1].plot(times[i], S, 'go', markersize=1)
            
    if p is not None:
        msg = "{:.1f}% SA".format(p)
        ax[1].annotate(msg, xy=(0.05, .8), xycoords='axes fraction')

def plot_eq(ax, tseq, model, states=None, p=None):
    times, temp_eq, salt_eq = np.array([]), np.array([]), np.array([])
    if states is None:
        states = [model.init_state for _ in range(len(tseq.tt))]
        
    for time,state in zip(tseq.times/year, states):
        trans_eq1 = model.trans_eq(state)
        temp_eq1 = model.temp_eq(state, trans_eq1)
        salt_eq1 = model.salt_eq(state, trans_eq1)
        
        times = np.append(times, np.ones_like(trans_eq1) * time)
        temp_eq = np.append(temp_eq, temp_eq1)
        salt_eq = np.append(salt_eq, salt_eq1)
        
    for T, S in zip(temp_eq, salt_eq):
        ax[0].scatter(times, temp_eq, c='k', s=1)
        ax[1].scatter(times, salt_eq, c='k', s=1)
        
    if p is not None:
        msg = "{:.1f}% SA".format(p)
        ax[1].annotate(msg, xy=(0.05, .8), xycoords='axes fraction')

def plot_truth(ax,xx,yy):
    times = np.linspace(ax[0].get_xlim()[0], ax[0].get_xlim()[1], np.size(xx,0)) 
    
    states=array2states(xx,times)
    TH = np.array([s.regime=='TH' for s in states], dtype=bool)
    temp = np.reshape(np.diff([s.temp[0] for s in states],axis=1), (-1))
    salt = np.reshape(np.diff([s.salt[0] for s in states],axis=1), (-1))
      
    #TH
    mask = np.where(~TH, np.nan, 1.)
    ax[0].plot(times, temp * mask, 'b-', alpha=.7)
    ax[1].plot(times, salt * mask, 'b-', alpha=.7)
    
    #SA
    mask = np.where(TH, np.nan, 1.)
    ax[0].plot(times, temp * mask, 'r-', alpha=.7)
    ax[1].plot(times, salt * mask, 'r-', alpha=.7)
    
    if len(yy)>0:
        timeos = times[1:len(yy)+1]
        
        for to,y in zip(timeos,yy):
            ax[0].errorbar(to,np.diff(y[0:2]),.5,color='k')
            ax[1].errorbar(to,np.diff(y[2:4]),.05,color='k')
    

def prob_change(E):
    E = array2states(E)
    
    for it in range(21):
        selection = [s.regime=='TH' for s in E[it]]
        E=E[:,selection]
        
    TH0 = np.size(E,1)
    TH = np.sum([s.regime=='TH' for s in E[-1]] )
    
    return float((TH0-TH)/TH0)

def error_prob(xx, E):
    errors = []
    E = np.mean(E, axis=1)
    
    xx = array2states(xx)
    E = array2states(E)
    pxx = np.array([xx1.regime!='TH' for xx1 in xx], dtype=int)
    pE = np.array([E1.regime!='TH' for E1 in E], dtype=int)
    
    return pE-pxx

def spread_prob(E):
    spreads = []
    
    for E1 in E:
        x = np.mean(E1, axis=0)
        x = array2states(np.reshape(x, (1,-1)))
        states = array2states(E1)
        
        px = x[0].regime!='TH'
        pE = np.array([s.regime!='TH' for s in states], dtype=int)
        
        spreads.append(np.var(pE, ddof=1))
        
    return spreads

def cross_entropy(xx, E):       
    xx = array2states(xx)
    pSA = np.array([xx1.regime!='TH' for xx1 in xx], dtype=int)
   
    qSA = []
    for E1 in E:
        states = array2states(E1)
        qSA.append( np.mean([s.regime!='TH' for s in states]) )
    qSA = np.array(qSA, dtype=float)
    
    entropy = -pSA * np.log(qSA + 1e-14) -(1-pSA) * np.log((1-qSA) + 1e-14)
        
    return entropy
        
    
    
