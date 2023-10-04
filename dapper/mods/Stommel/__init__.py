"""
Module containing model equations for dimensionless Stommel model. 
"""

import numpy as np
from scipy.optimize import fsolve
from dapper.mods.integration import rk4
import matplotlib.pyplot as plt
import dataclasses

mm2m = 1e-6 #convert millimeter to meter
year = 86400 * 365 #s
func_type = type(lambda:0.0)

#Convert observation functional into matrix operator. 
def sample2linear(model):
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
mixing_depth = 50.0 #m
#Estimated overturning
gamma_ref = Q_overturning * LinearEquationState().rho_ref / (1027.5-1026.5) / (4.8e6 * 3.65e3) #m6/kg   

@dataclasses.dataclass 
class State:
    temp: np.ndarray = np.array([[ 7.0, 17.0]]) #C 6,18
    salt: np.ndarray = np.array([[35.0, 36.1]]) #ppt 35,36.5
    
    salt_diff: float = 3e-5 / mixing_depth / mm2m #m2/s/m = 1e3 mm/s
    temp_diff: float = 1e-4 / mixing_depth / mm2m #m2/s/m
    gamma: float = 0.0
    
    time: float = 0.0
    
    def to_vector(self):
        v = np.array([], dtype=float)
        for key, value in dataclasses.asdict(self).items():
            if key!='time':
                v = np.append(v, value)
        return v
    
    def from_vector(self, v):
        lb = 0
        for key, value in dataclasses.asdict(self).items():
            if key!='time':
                ub = lb + max(1,np.size(value))
                setattr(self, key, np.reshape(v[lb:ub], np.shape(value)))
                lb = ub
            
    def zero(self):
        for key, value in dataclasses.asdict(self).items():
            if key!='time':
                setattr(self, key, value * 0.0)
          
    @property
    def regime(self):
        """Return the regime for circulation."""
        
        rho = StommelModel.eos(self.temp[0], self.salt[0])
        if np.diff(rho)<=0:
            return 'TH' #thermohaline circulation as in present
        else:
            return 'SA' #contra solution
                
def array2states(array):
    shape = np.shape(array)
    array = np.reshape(array,(-1,shape[-1]))
    
    states = np.array([State() for v in array], dtype=State)
    for n,v in enumerate(array):
        states[n].from_vector(v)
        
    states = np.reshape(states, shape[:-1])
        
    return states

def states2array(states):
    return np.array([s.to_vector() for s in states], dtype=float)
        
@dataclasses.dataclass
class StommelModel:
    #Geometry of basin 
    dz: np.ndarray = np.array([[3.65e3, 3.65e3]]) #m depth
    dy: np.ndarray = np.array([[5.2e6, 5.2e6]]) #m latitude
    dx: np.ndarray = np.array([[4.8e6, 4.8e6]]) #m longitude 
    V: np.ndarray = dx * dy * dz
    
    #Time
    time: float = 0.0
    #Initial state 
    init_state: State = State()
    #Memory for model state
    state: State = State()
    #Memory for tendency
    tendency: State = State()
    
    #Atmospheric heat forcing
    temp_air: np.array = np.array([lambda t: np.array([8.5, 26.0 ])], dtype=func_type) #C
    #Atmospheric haline forcing
    salt_air: np.array = np.array([lambda t: np.array([32.8, 36.6])], dtype=func_type) #ppt
    #Evaporation/percipitation flux (evaportion is positive)
    ep_flux: np.array = np.array([lambda t: np.array([0.0, 0.0])], dtype=func_type) 
    #Equation of state
    eos: LinearEquationState = LinearEquationState()
    #Ensemble member 
    ens_member = 0
    
    
    def __post_init__(self):
        self.init_state.gamma = self.default_gamma(self.init_state)
        self.state.gamma = self.default_gamma(self.state)
        
    def default_gamma(self, state, Q=Q_overturning):
        rho  =  self.eos(state.temp, state.salt)
        drho =  np.diff(np.sum(self.V * rho, axis=0)/np.sum(self.V, axis=0))[0]
        rho0 = self.eos.rho_ref
        
        area = np.sum(np.mean(self.dx, axis=1) * np.mean(self.dz, axis=1))
        
        return -Q * (rho0 / drho) / area 
    
    def obs_ocean(self):
        """ Observe ocean temperature and salinity. """
        M = 2*np.size(self.dz, 1)
        
        def obs_TS1(x, t):
            self.state.time = t
            year = 365*86400
        
            if t<=30*year:
                self.state.from_vector(x)
                return np.append(self.state.temp, self.state.salt)
            else:
                return np.empty((4,))
        
        def obs_model(x, t):            
            if np.ndim(x)==1:
                return obs_TS1(x, t)
            elif np.ndim(x)==2:
                return np.array([obs_TS1(x1,t) for x1 in x])
            else:
                msg = "x must be 1D or 2D array."
                raise TypeError(msg)
        
        Obs = {'M':M, 'model': obs_model, 'linear': sample2linear(obs_model)}
        return Obs
    
    #Tendency from atmospheric forcing
    def tendency_surface(self, tendency, state):
        """Contribution surface fluxes to tendency."""
        n = np.mod(self.ens_member+1, len(self.temp_air)) #member 0 is the truth
        temp_air = self.temp_air[n](state.time)
        tendency.temp[0] += (temp_air-state.temp[0]) * state.temp_diff * mm2m / self.dz[0] 
        
        n = np.mod(self.ens_member+1, len(self.salt_air))
        salt_air = self.salt_air[n](state.time)
        tendency.salt[0] += (salt_air-state.salt[0]) * state.salt_diff * mm2m / self.dz[0] 
        
        
        n = np.mod(self.ens_member+1, len(self.ep_flux))
        ep_flux = self.ep_flux[n](state.time)
        tendency.salt[0] += ep_flux * state.salt[0] / self.dz[0]
        
        if state.time==86400:
            print('rate ', ep_flux)
        
        return tendency
    
    #Tendency from advective transport
    def tendency_trans(self, tendency, state):
        """Contribution advective transport to tendency."""
        A = 0.25 * (self.dz[:,1:] + self.dz[:,:-1]) * (self.dx[:,1:] + self.dx[:,:-1])
    
        rho = self.eos(state.temp, state.salt)
        trans = np.abs(state.gamma * A * np.diff(rho, axis=1) / self.eos.rho_ref)
    
        tendency.temp[:, 1:] += trans * (state.temp[:,:-1]-state.temp[:,1:]) / self.V[:, 1:]
        tendency.temp[:,:-1] -= trans * (state.temp[:,:-1]-state.temp[:,1:]) / self.V[:,:-1]
    
        tendency.salt[:, 1:] += trans * (state.salt[:,:-1]-state.salt[:,1:]) / self.V[:, 1:]
        tendency.salt[:,:-1] -= trans * (state.salt[:,:-1]-state.salt[:,1:]) / self.V[:,:-1]
        return tendency
    
    #Total tendency 
    def dxdt1(self, x, t):
        """Calculate tendency for 1 state."""
        self.state.time = t
        self.state.from_vector(x)
        
        self.tendency.zero()
        self.tendency = self.tendency_surface(self.tendency, self.state)
        self.tendency = self.tendency_trans(self.tendency, self.state)
        
        return self.tendency.to_vector()
    
    #Forward model step for 1 ensemble member
    def step1(self, x0, t, dt):
        """ Step 1 model state forward in time using 4th order Runge-Kutta method. """ 
        return rk4(lambda x, t: self.dxdt1(x,t), x0, t, dt)
       
    #Forward model step an ensemble 
    def step(self, x, t, dt):
        """ Step all model states forward. """
        if np.ndim(x)==1:
            x = self.step1(x, t, dt)
        elif np.ndim(x)==2:
            for no, x1 in enumerate(x):
                self.ens_member = no
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
        dtemp = np.diff(self.temp_air[self.ens_member](state.time))[0]
        return dtemp / self.temp_scale(state)
    
    def eta2(self, state):
        """Non-dimensional parameter eta2. See Dijkstra (2008)."""
        dsalt = np.diff(self.salt_air[self.ens_member](state.time))[0]
        return (dsalt / self.salt_scale(state)) * self.eta3(state)
            
    def eta3(self, state):
        """Non-dimensional parameter eta3. See Dijkstra (2008)."""
        R_T = np.mean(state.temp_diff / self.dz[0]) * mm2m
        R_S = np.mean(state.salt_diff / self.dz[0]) * mm2m
        return R_S/ R_T
    
    def temp_scale(self, state):
        A = np.mean(self.dx[0]) * np.mean(self.dz[0])
        gamma = state.gamma * A
        return self.trans_scale(state) / (gamma * self.eos.alpha_T)
    
    def salt_scale(self, state):
        A = np.mean(self.dx[0]) * np.mean(self.dz[0])
        gamma = state.gamma * A
        return self.trans_scale(state) / (gamma * self.eos.alpha_S)  
        
    def trans_scale(self, state):
        R_T = np.mean(state.temp_diff * mm2m / self.dz[0])
        V = np.prod(self.V[0]) / np.sum(self.V[0])
        return R_T * V

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
        
        f = lambda q: (q - self.eta1(state) / (1+np.abs(q)) + 
                       self.eta2(state) / (self.eta3(state)+np.abs(q)))
        
        q  = np.linspace(-3,3,30000)     
        fq = f(q)
        
        q=.5*q[1:]+.5*q[:-1]
        fq=fq[1:]*fq[:-1]
        
        roots = [q1 for q1,f1 in zip(q,fq) if f1<0.]
        
        return np.array(roots) * self.trans_scale(state)
    
    
        
    
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
    dz = 50.0 #m+
    
    rate = V_ice / (T * model.dx[0] * model.dy[0]) 

    def melted_func(t):
        if t<T:
            return func(t) - rate 
        else:
            return func(t)
    
    return melted_func

def add_warming(func, sigs=np.array([0.,0.])):
    T = 100 * year
    rate  = np.array([1., 1.]) * np.random.normal(loc=3.,scale=sigs[0])
    rate += np.array([1.,-1.]) * np.random.normal(loc=.5,scale=sigs[1])
    rate /= T #C/s
    def warmed_func(t):
        return func(t) + min(t, T) * rate 
    return warmed_func   

def add_init_noise(func, seed, sig):
    """ Add white noise to a forcing function."""
    np.random.seed(seed)
    perturbation = np.random.normal(size=np.shape(sig)) * sig
    def noised_func(t):
        return func(t) + perturbation
    return noised_func    

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


def plot_eq(ax, tseq, model, p=None):
    times = tseq.tt/year
    
    # Equilibrium values
    state = model.init_state
    trans_eq = model.trans_eq(state)
    temp_eq = model.temp_eq(state, trans_eq)
    salt_eq = model.salt_eq(state, trans_eq)

    for T, S in zip(temp_eq, salt_eq):
        ax[0].plot(times, np.ones_like(times) * T, 'k--')
        ax[1].plot(times, np.ones_like(times) * S, 'k--')
        
    if p is not None:
        msg = "{:.1f}% SA".format(p)
        ax[1].annotate(msg, xy=(0.05, .8), xycoords='axes fraction')

def plot_truth(ax,xx,yy):
    times = np.linspace(ax[0].get_xlim()[0], ax[0].get_xlim()[1], np.size(xx,0)) 
    
    states=array2states(xx)
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


        
        
    
    
