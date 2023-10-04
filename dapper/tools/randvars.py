"""Classes of random variables."""

import numpy as np
import numpy.random as rnd
from numpy import sqrt
from struct_tools import NicePrint

from dapper.tools.matrices import CovMat

#Datatype to store point measurements.
def point_obs_type(dim):
    return np.dtype([('time',float), ('coordinates',float,(dim,)),
                    ('bias',float), ('var',float), ('rvar',float),
                    ('field_name',np.unicode_,32),
                    ('obs',float)])

class RV(NicePrint):
    """Class to represent random variables."""

    printopts = NicePrint.printopts.copy()
    printopts["ordering"] = "linenumber"
    printopts["reverse"] = True

    def __init__(self, M, **kwargs):
        """Initalization arguments:

        - M    <int>     : ndim
        - is0  <bool>    : if True, the random variable is identically 0
        - func <func(N)> : use this sampling function. Example:
                           `RV(M=4,func=lambda N: rand(N,4)`
        - file <str>     : draw from file. Example:
                           `RV(M=4,file=dpr.rc.dirs.data/'tmp.npz')`

        The following kwords (versions) are available,
        but should not be used for anything serious
        (use instead subclasses, like `GaussRV`).

        - icdf <func(x)> : marginal/independent  "inverse transform" sampling.
                           Example: `RV(M=4,icdf = scipy.stats.norm.ppf)`
        - cdf <func(x)>  : as icdf, but with approximate icdf, from interpolation.
                           Example: `RV(M=4,cdf = scipy.stats.norm.cdf)`
        - pdf  <func(x)> : "acceptance-rejection" sampling. Not implemented.
        """
        self.M = M
        for key, value in kwargs.items():
            setattr(self, key, value)

    def sample(self, N):
        if getattr(self, 'is0', False):
            # Identically 0
            E = np.zeros((N, self.M))
        elif hasattr(self, 'func'):
            # Provided by function
            E = self.func(N)
        elif hasattr(self, 'file'):
            # Provided by numpy file with sample
            data   = np.load(self.file)
            sample = data['sample']
            N0     = len(sample)
            if 'w' in data:
                w = data['w']
            else:
                w = np.ones(N0)/N0
            idx = rnd.choice(N0, N, replace=True, p=w)
            E   = sample[idx]
        elif hasattr(self, 'icdf'):
            # Independent "inverse transform" sampling
            icdf = np.vectorize(self.icdf)
            uu   = rnd.rand(N, self.M)
            E    = icdf(uu)
        elif hasattr(self, 'cdf'):
            # Like above, but with inv-cdf approximate, from interpolation
            if not hasattr(self, 'icdf_interp'):
                # Define inverse-cdf
                from scipy.interpolate import interp1d
                from scipy.optimize import fsolve
                cdf    = self.cdf
                Left,  = fsolve(lambda x: cdf(x) - 1e-9, 0.1)  # noqa
                Right, = fsolve(lambda x: cdf(x) - (1-1e-9), 0.1)  # noqa
                xx     = np.linspace(Left, Right, 1001)
                uu     = np.vectorize(cdf)(xx)
                icdf   = interp1d(uu, xx)
                self.icdf_interp = np.vectorize(icdf)
            uu = rnd.rand(N, self.M)
            E  = self.icdf_interp(uu)
        elif hasattr(self, 'pdf'):
            # "acceptance-rejection" sampling
            raise NotImplementedError
        else:
            raise KeyError
        assert self.M == E.shape[1]
        return E
    
class RV_from_function(RV):
    
    def __init__(self, function):
        self.function = function
        self.time = None
        
    def update(self, *args, **kwargs):
        self.time = args[1]
        
    def sample(self, N):
        return self.function(N, self.time)


# TODO 4: improve constructor (treatment of arg cases is too fragile).
class RV_with_mean_and_cov(RV):
    """Generic multivariate random variable characterized by mean and cov.

    This class must be subclassed to provide sample(),
    i.e. its main purpose is provide a common convenience constructor.
    """

    def __init__(self, mu=0, C=0, M=None):
        """Init allowing for shortcut notation."""
        self._M = None
         
        if isinstance(mu, CovMat):
            raise TypeError("Got a covariance paramter as mu. "
                            + "Use kword syntax (C=...) ?")
            
        self.args = {'M':M, 'mu':mu, 'C':C}
        self.rebuild(M)
        
    @property 
    def M(self):
        return self._M
        
    @M.setter
    def M(self, M):
        if self.M is not M:
            self.rebuild(M)
        
    def rebuild(self, M):
        self._set_M(self.args['mu'], self.args['C'], M)
        self._set_mu(self.args['mu'])
        self._set_C(self.args['C'])
             
    def _set_M(self, mu, C, M):
        if M is not None:
            self._M = int(M) 
        elif isinstance(mu, np.ndarray) and np.ndim(mu)==1:
            self._M = len(mu)
        elif isinstance(C, CovMat):
            self._M = int(C.M)
        elif isinstance(C, RV):
            self._M = int(C.M)
        elif isinstance(C, np.ndarray):
            self._M = np.size(C,0)
        else:
            raise TypeError("Could not deduce the value of M from mu/C.")
        
    def _set_mu(self, mu):
        if isinstance(mu, (int, float)):
            self.mu = np.ones((self.M,)) * mu
        elif isinstance(mu, np.ndarray):
            self.mu = mu 
        else:
            raise TypeError("Could not deduce the value of M from mu.")
        
        if len(self.mu) != self.M:
            raise TypeError("Inconsistent shapes of (M,mu,C)")
        
    def _set_C(self, C):
        if isinstance(C, CovMat):
            self.C = C
        elif isinstance(C, RV_with_mean_and_cov):
            self.C = C.C
        elif isinstance(C, (int,float)) and C==0:
            self.C = 0
        elif isinstance(C, (int,float)):
            self.C = CovMat(C*np.ones((self.M,)), 'diag')
        elif isinstance(C, np.ndarray) and np.ndim(C)==1:
            self.C = CovMat(C, 'diag')
        else:
            raise TypeError("Could not deduce the value of M from C.")
        
        if isinstance(C, CovMat) and self.M != self.C.M:
            raise TypeError("Inconsistent shapes of (M,mu,C)")    

    def sample(self, N):
        """Sample N realizations. Returns N-by-M (ndim) sample matrix.

        Example
        -------
        >>> plt.scatter(*(UniRV(C=randcov(2)).sample(10**4).T))  # doctest: +SKIP
        """
       
        if self.C == 0:
            D = np.zeros((N, self.M))
        else:
            D = self._sample(N)
        return self.mu + D

    def _sample(self, N):
        raise NotImplementedError("Must be implemented in subclass")


class GaussRV(RV_with_mean_and_cov):
    """Gaussian (Normal) multivariate random variable."""

    def _sample(self, N):           
        R = self.C.Right
        D = rnd.randn(N, len(R)) @ R
        return D
    
class TimeGaussRV(RV_with_mean_and_cov):
    """Gaussian (Normal) multivariate random variable."""
    
    def __init__(self, database, seed):
        """ Set database from which variances and means are retrieved."""
        self._time = None
        self.args = {}
        self.seed = seed
        
        self.database = database
        self.update(None, min(database['time']))
        
    @property
    def time(self):
        return self._time
            
    def update(self, *args, **kwargs):
        E, time = args[0], args[1]
        if self.time == time:
            return
        
        #Observation at this time. 
        selection = self.database['time'] == time 
        M = np.sum(selection)
        
        #Ensemble variance in observation space. 
        s = np.array(np.shape(E))
        if np.sum(s>1)>1:
            var = np.var(E, axis=0, ddof=1)
        else:
            var = np.zeros((M,))
          
        #Rebuild obs. covariance if necessary.  
        if M>0:
            self._time = time
            self.args['mu'] = self.database['bias'][selection]
            self.args['C']  = self.database['var'][selection]
            self.args['C'] += self.database['rvar'][selection] * var
            self.args['M'] = M
            self.rebuild(M)  
    
    def _sample(self, N):  
        np.random.seed(int(self.time) + self.seed)     
        R = self.C.Right
        D = rnd.randn(N, len(R)) @ R
        return D
    
class CoordinateRV(RV):
    
    def __init__(self, N_max, M, mu=0):
        from datetime import datetime, timedelta #ip:
        
        self._M = M
        self.mu = mu 
        self.N_max = N_max
        
        self.generators={}
        self.coordinates={}
        self.indices={}
        
        self.time = 0.
        self.ref_time = datetime(2000,1,1)
    
    @property 
    def M(self):
        return self._M 
    
    @M.setter
    def M(self, m):
        self._M = M 
        
        if np.all(self.mu==self.mu[0]):
            self.mu = self.mu[0]
        
        if isinstance(self.mu, (int,float)):
            self.mu = np.ones((self.M,)) * self.mu 
        elif np.size(mu) != self.M:
            raise ValueError('Cannot resize mu.')      
        
    def add_sector(self, sector, indices, coordinates):
        self.indices[sector] = indices 
        self.coordinates[sector] = coordinates
    
    def add_spectral(self, sector, phase_generator, wavelengths, amplitudes):
        from copy import deepcopy
        from dapper.mods.Ice.forcings import SpectralNoise
        
        self.generators[sector] = []
        for n in range(self.N_max):
            generators = [deepcopy(phase_generator) for A in amplitudes]
            for m,generator in enumerate(generators):
                generators[m].base_seed += m + 13*n
              
            self.generators[sector].append(SpectralNoise(generators, wavelengths, amplitudes))
            
    def update(self, *args, **kwargs):
        E, time = args[0], args[1]
        self.time = time
    
    def sample1(self, n, sector, E1):
        from datetime import datetime, timedelta #ip:
        generator = self.generators[sector][n]
        time = datetime(2000,1,1) + timedelta(minutes=self.time)
        
        for ind, coord in zip(self.indices[sector], self.coordinates[sector]):
            E1[ind] = generator(time, coord)
    
        return E1
    
    def sample(self, N):
        E = np.zeros((N, self.M))
        
        for n in range(N):
            for key in self.generators:
                E[n] = self.sample1(n, key, E[n])
                
        if N==1:
            return E[0] + self.mu
        else:
            return E + np.reshape(self.mu,(1,-1))
            

class LaplaceRV(RV_with_mean_and_cov):
    """Laplace (double exponential) multivariate random variable.

    This is an elliptical generalization. Ref:
    Eltoft (2006) "On the Multivariate Laplace Distribution".
    """

    def _sample(self, N):
        R = self.C.Right
        z = rnd.exponential(1, N)
        D = rnd.randn(N, len(R))
        D = z[:, None]*D
        return D @ R / sqrt(2)


class LaplaceParallelRV(RV_with_mean_and_cov):
    """A NON-elliptical multivariate version of Laplace (double exponential) RV."""

    def _sample(self, N):
        # R = self.C.Right   # contour: sheared rectangle
        R = self.C.sym_sqrt  # contour: rotated rectangle
        D = rnd.laplace(0, 1, (N, len(R)))
        return D @ R / sqrt(2)


class StudRV(RV_with_mean_and_cov):
    """Student-t multivariate random variable.

    Assumes the covariance exists,
    which requires degreee-of-freedom (dof) > 1+ndim.
    Also requires that dof be integer,
    since chi2 is sampled via Gaussians.
    """

    def __init__(self, dof, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dof = dof

    def _sample(self, N):
        R = self.C.Right
        nu = self.dof
        r = nu/np.sum(rnd.randn(N, nu)**2, axis=1)  # InvChi2
        D = sqrt(r)[:, None]*rnd.randn(N, len(R))
        return D @ R * sqrt((nu-2)/nu)


class UniRV(RV_with_mean_and_cov):
    """Uniform multivariate random variable.

    Has an elliptic-shape support.
    Ref: Voelker et al. (2017) "Efficiently sampling
    vectors and coordinates from the n-sphere and n-ball"
    """

    def _sample(self, N):
        R = self.C.Right
        D = rnd.randn(N, len(R))
        r = rnd.rand(N)**(1/len(R)) / np.sqrt(np.sum(D**2, axis=1))
        D = r[:, None]*D
        return D @ R * 2


class UniParallelRV(RV_with_mean_and_cov):
    """Uniform multivariate random variable.

    Has a parallelogram-shaped support, as determined by the cholesky factor
    applied to the (corners of) the hypercube.
    """

    def _sample(self, N):
        R = self.C.Right
        D = rnd.rand(N, len(R))-0.5
        return D @ R * sqrt(12)
