""" Different methods to calculate inflation factors """
import numpy as np
from abc import ABC, abstractmethod 
from copy import copy 
from numpy.polynomial import Polynomial

class InflationException(ValueError):
    """Class indicating that inflation factor is invalid."""
    
class FactorFilter(ABC):
    """Filters factors to smooth out rapid changes."""
    
    def default_init(self):
        self.times = np.array([], dtype=float)
        self.factors = np.array([], dtype=float)
        
    def push(self, time, factor):
        """Add the latest inflation factor to the 
        object. 
        
        Parameters
        ----------
        time : float 
            Current time. 
        factor : float 
            Unfiltered inflation factor. 
        
        """
        if not time in self.times:
            self.times = np.append(self.times, time)
            self.factors = np.append(self.factors, factor)
    
    @abstractmethod 
    def pop(self):
        """Returns the filtered inflation factor."""
        pass
    
class LastInFilter(FactorFilter):
    """No smoothing applied."""
    
    def __init__(self):
        self.default_init()
        
    def pop(self):
        return self.factors[-1]
    
class ArithmeticMeanFilter(FactorFilter):
    """Returns mean of most recent factors."""
    
    def __init__(self, weights, default):
        self.default_init()
        
        if hasattr(weights, '__iter__'):
            self.weights = weights
        else:
            self.weights = np.ones((int(weights),), dtype=float)
        self.total_w = np.sum(weights)
        
        self.default_factor = default 
        
    def pop(self):
        factors = np.ones_like(self.weights) * self.default_factor
        
        n = min(len(factors), len(self.factors))
        factors[:n] = self.factors[:-n-1:-1]
        
        return np.sum(factors * self.weights)/self.total_w
    
class HarmonicMeanFilter(FactorFilter):
    """Returns mean of most recent factors."""
    
    def __init__(self, weights, default):
        self.default_init()
        
        if hasattr(weights, '__iter__'):
            self.weights = weights
        else:
            self.weights = np.ones((int(weights),), dtype=float)
        self.total_w = np.sum(weights)
        
        self.default_factor = default
        
    def pop(self):
        factors = np.ones_like(self.weights) * self.default_factor
        
        n = min(len(factors), len(self.factors))
        factors[:n] = self.factors[:-n-1:-1]
        
        return np.total_w/np.sum(self.weights/factors)

class Inflator:
    """Trivial interface for objects carrying out
    ensemble inflation."""
        
    def __init__(self):
        """Class constructor."""
        
        #Filters that smooths factors in time.
        self.filter = LastInFilter()
    
    def factor(self):    
        """Returns factor."""
        return 1.0
    
    def check_time(self, time):
        """Check if input time matches that stored.
        
        Parameters
        ----------
        time : float 
            Current time. 
            
        """
        if time != self.time:
            msg = "Expecting analysis at time {}."
            raise ValueError(msg.format(self.time))
    
    def inflate_for(self, obs, E_for, time, yy):
        """Inflate the background ensemble.
        To be called before running analysis.
        
        Parameters
        ----------
        obs : operator object 
            Object representing observation operator.
        E_for : 2d Numpy array
            Array with in each row an background ensemble
            member. 
        time : float 
            Current time. 
        yy : 1D Numpy array 
            Array with observed values. 
        
        Returns
        -------
        Inflated background ensemble.
         
        """
        self.update_for(obs, E_for, time, yy)
        return E_for
        
    def inflate_ana(self, obs, E_ana, time, yy):
        """Inflate the analysis ensemble.
        To be called after running analysis. 
        
        Parameters
        ----------
        obs : operator object 
            Object representing observation operator.
        E_for : 2d Numpy array
            Array with in each row an analysis ensemble
            member.  
        time : float 
            Current time. 
        yy : 1D Numpy array 
            Array with observed values. 
        
        Returns
        -------
        Inflated analysis ensemble.
         
        """
        self.update_ana(obs, E_ana, time, yy)
        return E_ana  
    
    def update_for(self, obs, E_for, time, yy):
        self.time = time 
    
    def update_ana(self, obs, E_ana, time, yy):    
        self.check_time(time) 
        
    @staticmethod 
    def cross_cov(x,y):
        """Calculate ensemble cross-covariance between x and y."""
        N = np.size(x, 0)
        cov = np.mean((x-np.mean(x,axis=0,keepdims=True))
                      *(y-np.mean(y,axis=0,keepdims=True)),axis=0)
        #Bessel correction to account for uncertainty in mean.
        return (N/(N-1)) * cov
        
class EnsInflator(Inflator):
    """Abstract class that inflates the background 
    ensemble."""
    
    @property
    def factor(self):
        return 1.0
    
    def update_for(self, obs, E_for, time, yy):
        self.time = time 
        self.x_for = np.mean(E_for, axis=0, keepdims=True)
        self.filter.push(time, self.factor)
    
    def inflate_for(self, obs, E_for, time, yy):
        self.update_for(obs, E_for, time, yy)
        factor = self.filter.pop()
        return self.x_for + np.sqrt(factor) * (E_for - self.x_for)
        
class EnsRelaxor(Inflator):  
    """Abstract class that relaxes analysis to background."""
    
    @property 
    def factor(self):
        return 0.0
    
    def update_for(self, obs, E_for, time, yy):
        self.time = time 
        self.x_for = np.mean(E_for, axis=0, keepdims=True)
        self.E_for = copy(E_for)
        
    def update_ana(self, obs, E_ana, time, yy):
        self.check_time(time)
        self.x_ana = np.mean(E_ana, axis=0, keepdims=True)
        self.filter.push(time, self.factor)
    
    def inflate_ana(self, obs, E_ana, time, yy):
        self.update_ana(obs, E_ana, time, yy)
        factor = self.filter.pop()
        return (self.x_ana 
                + factor * (self.E_for - self.x_for)
                + (1.0-factor) * (E_ana - self.x_ana) 
                ) 
    
class FixedInflator(EnsInflator):
    """Background inflation using a time-dependent inflation
    factor."""
    
    def __init__(self, factor_function):
        """Class constructor.
        
        Parameters
        ----------
        factor_function : Python function
            Function providing inflation factor as function 
            of time.
        """
        #Filter to smooth factor in time. Trivial in this case.
        self.filter = LastInFilter()
        #Function to calculate factor. 
        self.factor_function = factor_function 
        
    @property 
    def factor(self):
        return self.factor_function(self.time)   
    
class FixedRelaxer(EnsRelaxor):
    """Relaxation factor using a time-dependent inflation
    factor."""
    
    def __init__(self, factor_function):
        """Class constructor.
        
        Parameters
        ----------
        factor_function : Python function
            Function providing relaxation factor as function 
            of time.
        """
        #Filter to smooth factor in time. Trivial in this case.
        self.fitler = LastInFilter()
        #Function to calculate factor. 
        self.factor_function = factor_function 
        
    @property 
    def factor(self):
        return self.factor_function(self.time)    

class ObsInflator(Inflator):
    """Inflation based on observation-background differences as
    described in Li, Kalnay and Miyoshi (2009) and Miyoshi (2011). 
    
    gamma = [tr(omb*omb^T)-N_{obs}]/tr(HPH^T) 
    with omb = y_{obs} - y_{back}
    H the sampling operator and P=\frac{1}{N_ens-1}X_{back}X^T_{back}    
    """
    
    def __init__(self, filter=LastInFilter()):
        """Class constructor.
        
        Parameters
        ----------
        filter : FactorFilter object 
            Time filter to smooth factors. 
        """
        self.filter = filter
        
    @property
    def factor(self):
        #Number of observations.
        No = np.size(self.R, 0)
        #Differences observation-background predictions. 
        omb = self.yy - self.Hx_for
        #Trace of background prediction covariance matrix.
        tr_for = Inflator.cross_cov(self.HE_for, self.HE_for)
        #Inflation factor.
        return (np.sum(omb**2/self.R)-No)/np.sum(tr_for/self.R)
        
    def update_for(self, obs, E_for, time, yy):
        self.time = time
        
        self.x_for = np.mean(E_for, axis=0, keepdims=True)
        self.Hx_for = np.reshape(obs(self.x_for, time), (1,-1))
        self.HE_for = obs(E_for, time)
        
        self.yy = yy
        self.R = obs.noise.C.diag 
        
        factor = max(1e-2, self.factor)
        self.filter.push(time, factor)
    
class AnaInflator(Inflator):
    """Inflation based on analysis-background differences 
    as described in Li, Kalnay and Miyoshi (2009) and Miyoshi (2011).
    
    gamma = [tr(omb*omb^T)-N_{obs}]/tr(HPH^T) 
    with omb = y_{obs} - y_{back}, amb = y_{ana}-y_{back}
    H the sampling operator and P=\frac{1}{N_ens-1}X_{back}X^T_{back}
    
    """
    
    def __init__(self, filter=LastInFilter()):
        """Class constructor.
        
        Parameters
        ----------
        filter : FactorFilter object 
            Time filter to smooth factors. 
        """
        self.filter = filter
        
    @property
    def factor(self):
        #Differences observations-background predictions.
        omb = self.yy - self.Hx_for 
        #Differences analysis-background predictions.
        amb = self.Hx_ana - self.Hx_for
        #Trace background prediction covariance matrix.
        tr_for = Inflator.cross_cov(self.HE_for, self.HE_for)
        #Inflation factor.
        return np.sum(omb*amb/self.R)/np.sum(tr_for/self.R)  
        
    def update_for(self, obs, E_for, time, yy):
        self.time = time
        
        self.x_for = np.mean(E_for, axis=0, keepdims=True)
        self.Hx_for = np.reshape(obs(self.x_for, time), (1,-1))
        self.HE_for = obs(E_for, time)
        
        self.yy = yy
        self.R = obs.noise.C.diag 
        
    def update_ana(self, obs, E_ana, time, yy):
        self.check_time(time)
        
        self.x_ana = np.mean(E_ana, axis=0, keepdims=True)
        self.Hx_ana = np.reshape(obs(self.x_ana, time), (1,-1))
        
        factor = max(1e-2, self.factor)
        self.filter.push(time, factor)
    
class AdaptiveRTPP(EnsRelaxor):
    """Adaptive relaxation-to-prior-perturbation scheme 
    from Kotsuki, Yoichiro and Miyoshi (2017)."""
    
    def __init__(self, filter=LastInFilter()):
        """Class constructor.
        
        Parameters
        ----------
        filter : FactorFilter object 
            Time filter to smooth factors. 
        """
        self.filter = filter
        
    def update_for(self, obs, E_for, time, yy):
        self.time = time
        
        self.E_for = copy(E_for)
        self.x_for = np.mean(E_for, axis=0, keepdims=True)
        self.Hx_for = np.reshape(obs(self.x_for, time), (1,-1))
        self.HE_for = obs(E_for, time)
        
        self.yy = yy
        self.R = obs.noise.C.diag 
        
    def update_ana(self, obs, E_ana, time, yy):
        self.check_time(time)
        
        self.x_ana = np.mean(E_ana, axis=0, keepdims=True)
        self.Hx_ana = np.reshape(obs(self.x_ana, time), (1,-1))
        self.HE_ana = obs(E_ana, time)

        factor = max(0.0, min(1.0, self.factor))
        self.filter.push(time, factor)
        
    @property
    def factor(self):
        #Trace background prediction covariance matrix. 
        tr_ff = Inflator.cross_cov(self.HE_for, self.HE_for)
        #Trace analysis-background prediction cross-covariance matrix. 
        tr_af = Inflator.cross_cov(self.HE_for, self.HE_ana)
        #Trace analysis prediction covariance matrix. 
        tr_aa = Inflator.cross_cov(self.HE_ana, self.HE_ana)
        #Differences observations-analysis predictions.
        oma = self.yy-self.Hx_ana
        #Differences analysis-background predictions. 
        amb = self.Hx_ana-self.Hx_for
        
        #See Kotsuki, Yoichiro and Miyoshi (2017) for
        #definitions.
        phi1 = np.sum(tr_ff/self.R)
        phi2 = np.sum(tr_af/self.R)
        phi3 = np.sum(tr_aa/self.R)
        phi4 = np.sum(oma*amb/self.R)
    
        l = np.zeros((3,))
        l[2] = phi1 - 2*phi2 + phi3
        l[1] = 2*phi2 - 3*phi3 
        l[0] = phi3 - phi4 
        poly = Polynomial(l)
        roots = np.real(poly.roots())
    
        #Relaxation factor. 
        if any(roots>0.) and any(roots<=0.):
            return np.max(roots)
        else:
            return 0.

    