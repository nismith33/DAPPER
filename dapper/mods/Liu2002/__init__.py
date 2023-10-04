'''
This module contains the model as described in Liu & Rabier (2002). 
'''

import numpy as np
from scipy import fft
from numpy.fft import fftfreq
import math
from copy import copy, deepcopy

eps = 1e-7  # small value


def rho_white():
    """
    Return correlation function between point in white noise (i.e. delta function).
    """
    def rho(x1, x2):
        x1, x2 = np.asarray(x1), np.asarray(x2)
        return np.where(np.abs(x2-x1) < eps, 1., 0.)

    return rho


def rho_true(L, b):
    """
    Returns correlation function between points in truth. 

    Parameters 
    ----------
    x1 : float 
        One position. 
    x2 : float 
        Other position. 

    """
    def rho(x1, x2):
        r = np.abs(x2-x1)
        return (np.cos(b*r) + np.sin(b*r) / (L * b)) * np.exp(-r / L)

    return rho


def rho_back(L):
    """
    Returns correlation function between points in background. 

    Parameters 
    ----------
    x1 : float 
        One position. 
    x2 : float 
        Other position. 

    """
    def rho(x1, x2):
        r = np.abs(x2-x1)
        return (1 + r/L) * np.exp(-r / L)

    return rho


class SpectralModel:
    """
    Object responsible for dynamic and observing. 

    """

    _default_attributes = {'L': None, 'member': 0, 'functions': [],
                           'obs_coords': None, 'dyn_coords': None,
                          }

    def __init__(self, L, member=0):
        """
        Class constructor. 

        Parameters
        ----------
        L : float > 0 
            Domain length.
        member : int>=0
            Currently active ensemble member.

        """
        self.L = L
        self.member = member

        self.functions = []
        self.obs_coords = None
        self.dyn_coords = None
        

    def signal_factory(self, K, rho, sig=1, seed=1000):
        """
        Create new FourierFunction. 

        Parameters
        ----------
        K : int > 0 
            Dimension of model is 2K+1. 
        sig : float
            Standard deviation signal. 
        rho : Python function
            Correlation function to generate noise instances. 
        seed : int 
            Random seed used to sample phases of different Fourier components. 

        """
        from scipy.signal.windows import hann

        # spacing
        dx = self.L/(2*K+1)
        # Position
        x = np.arange(-K, K+1) * dx
        # Wavenumbers
        k = fft.fftfreq(2*K+1, dx) * 2 * np.pi
        # Fourier transform correlation
        rhos = rho(x, np.zeros_like(x))
        # Window
        window = hann(2*K+1, False)
        Frho = fft.fft(rhos)
        # Create fourier coefficient
        F = np.sqrt(np.abs(Frho)) 
        # Normalise coefficients
        F = F[k >= 0]
        k = k[k >= 0]
        F = F / np.linalg.norm(F) * sig
        # Output new function
        self.functions.append(FourierFunction(k, F, seed))

    def red_noise_factory(self, K, slope, sig=1, seed=1000):
        """
        Create members with spectrum red/pink noise.

        Parameters
        ----------
        K : int>0
            Number of Fourier coefficients 2*K+1
        slope : float
            Spectrum ~k**slope
        sig : float, optional
            Standard deviation members. The default is 1.
        seed : int, optional
            Seed used to randomly generate members. The default is 1000.

        """
        from scipy.signal.windows import hann

        # spacing
        dx = self.L/(2*K+1)
        # Wavenumbers
        k = fft.fftfreq(2*K+1, dx) * 2 * np.pi
        k = k[k >= 0]
        # Window
        Frho = np.abs(k[1:])**slope
        Frho = np.append([0.], Frho)
        # Create fourier coefficient
        F = np.sqrt(np.abs(Frho))
        # Normalise coefficients
        F = F / np.linalg.norm(F) * sig 
        # Output new function
        self.functions.append(FourierFunction(k, F, seed))

    def gauss_noise_factory(self, K, length, sig=1, seed=1000):
        """
        Create members with Gaussian noise. 

        Parameters
        ----------
        K : int>0
            Number of Fourier coefficients 2*K+1
        slope : float
            Length scale of Gaussian used. 
        sig : float, optional
            Standard deviation members. The default is 1.
        seed : int, optional
            Seed used to randomly generate members. The default is 1000.

        """
        from scipy.signal.windows import hann

        # spacing
        dx = self.L/(2*K+1)
        # Wavenumbers
        k = fft.fftfreq(2*K+1, dx) * 2 * np.pi
        k = k[k >= 0]
        # Window
        Frho = np.exp(-k[1:]**2 * length**2)
        Frho = np.append([0.], Frho)
        # Create fourier coefficient
        F = np.sqrt(np.abs(Frho))
        # Normalise coefficients
        F = F / np.linalg.norm(F) * sig
        # Output new function
        self.functions.append(FourierFunction(k, F, seed))

    def local2global(self, local_coords):
        """
        Convert local coordinates into grid of global coordinates. 

        Parameters
        ----------
        local_coords : array of float [0,1]
            Local coordinates

        Returns
        -------
        Vector with global coordinates. 

        """
        dx = self.L/self.ncells
        left = np.reshape(np.linspace(0, self.L, self.ncells, endpoint=False),
                          (-1, 1))
        cell_coord = np.reshape(local_coords, (1, -1))
        return np.reshape(cell_coord * dx + left, (-1))

    def global2local(self, global_coords):
        dx = self.L/self.ncells
        cell_indices = np.array(global_coords/dx, dtype=int)
        local_coords = np.mod(global_coords/dx, 1.)
        return local_coords, cell_indices

    def to_cell(self, x):
        """
        Reshape vector into 2D-array with cell index along axis 0.
        """
        return np.reshape(x, (self.ncells, self.order+1)).T

    def from_cell(self, x):
        """
        Reshape from 2D array with cell index along axis into vector. 
        """
        return np.reshape(x.T, (-1))

    def dyn_coord_factory(self, coord_type, ncells, order):
        """
        Add model coordinates to this object. 

        Parameters
        ----------
        coord_type : str
            Name of coordinate setting. 
        ncells : int
            Number of grid cells. 
        order : int
            Number of grid points per cell - 1

        Returns
        -------
        None.

        """
        from scipy.special import legendre
        self.ncells = ncells
        self.order = order

        if coord_type == 'uniform':
            N = (order+1) * ncells
            self.dyn_coords = np.linspace(0, self.L, N, endpoint=False)
        elif coord_type == 'dg':
            cell_coord, _ = np.polynomial.legendre.leggauss(self.order+1)
            self.dyn_coords = self.local2global(0.5 * cell_coord + 0.5)
        elif coord_type == 'legendre':
            self.dyn_coords = np.linspace(0, self.L, ncells+1, endpoint=True)
            self.dyn_coords = 0.5 * \
                self.dyn_coords[1:] + 0.5*self.dyn_coords[:-1]
            self.dyn_coords = self.dyn_coords[...,
                                              None] + np.zeros((1, self.order+1))
            self.dyn_coords = np.reshape(self.dyn_coords, (-1))
        else:
            raise ValueError("Unknown coordinate type {}".format(coord_type))

    def obs_coord_factory(self, coord_type, nobs):
        """
        Add locations observations to this object. 

        Parameters
        ----------
        coord_type : str
            Creating scheme to be used. 
        nobs : int
            Number of observations.

        Returns
        -------
        None

        """
        self.nobs = nobs

        if coord_type == 'uniform':
            dx = self.L/self.nobs
            coords = np.linspace(0.5*dx, self.L-0.5*dx, self.nobs,
                                 endpoint=True)

            def obs_coords(t):
                return coords

        elif coord_type == 'single':
            def obs_coords(t):
                return np.array([.5 * self.L])

        elif coord_type == 'random':
            def obs_coords(t):
                np.random.seed(int(t) + 500)
                return np.sort(np.random.uniform(size=self.nobs) * self.L)

        elif coord_type == 'land_mask':
            dx = 0.5*self.L/self.nobs
            coords = np.linspace(0.5*dx, 0.5*self.L-0.5*dx, self.nobs,
                                 endpoint=True)

            def obs_coords(t):
                return coords

        else:
            raise ValueError("Unknown coordinate type {}".format(coord_type))

        self.obs_coords = obs_coords

    def state2lin(self, x, ndiff=0):
        """Convert states to linear interpolation operator."""
        from scipy.interpolate import interp1d

        ndiff = int(ndiff)
        if np.ndim(x) == 1:
            x = np.reshape(x, (1, -1))

        # Coordinates
        coords = np.concatenate((self.dyn_coords[-1-ndiff:] - self.L,
                                 self.dyn_coords,
                                 self.dyn_coords[:1+ndiff] + self.L))
        coords = np.reshape(coords, (1, -1))

        # Values
        x = np.concatenate((x[:, -1-ndiff:], x, x[:, :1+ndiff]), axis=1)

        # Finite diffence differentation.
        for _ in range(ndiff):
            x = np.diff(x, n=1, axis=1) / np.diff(coords, n=1, axis=1)
            coords = 0.5 * coords[:, :-1] + 0.5 * coords[:, 1:]

        # Linear interpolation.
        coords = np.reshape(coords, (-1))
        interpolator = interp1d(coords, x, kind='linear', axis=1)

        return interpolator
    
    def state2spline(self, x, ndiff=0):
        """Convert states to spline interpolation operator."""
        from scipy.interpolate import CubicSpline

        if np.ndim(x) == 1:
            x = np.reshape(x, (1, -1))
        x = np.concatenate((x, x[:,:1]), axis=1)

        # Coordinates
        coords = np.reshape(self.dyn_coords, (1, -1))
        coords = np.concatenate((coords, coords[:,-1:]+coords[:,1:2]-coords[:,:1]), axis=1)

        # Spline interpolation.
        coords = np.reshape(coords, (-1))
        interpolator = CubicSpline(coords, x, bc_type='periodic', axis=1)
        
        # Derivative 
        interpolator = interpolator.derivative(int(ndiff))

        return interpolator
    
    def state2pchip(self, x, ndiff=0):
        """Convert states to spline interpolation operator."""
        from scipy.interpolate import PchipInterpolator

        if np.ndim(x) == 1:
            x = np.reshape(x, (1, -1))
        x = np.concatenate((x, x[:,:1]), axis=1)

        # Coordinates
        coords = np.reshape(self.dyn_coords, (1, -1))
        coords = np.concatenate((coords, coords[:,-1:]+coords[:,1:2]-coords[:,:1]), axis=1)

        # Spline interpolation.
        coords = np.reshape(coords, (-1))
        interpolator = PchipInterpolator(coords, x, axis=1)
        
        # Derivative 
        interpolator = interpolator.derivative(int(ndiff))

        return interpolator

    def state2fourier(self, x, ndiff=0):
        """Convert states into Fourier expansion."""

        ndiff = int(ndiff)
        if np.ndim(x) == 1:
            x = np.reshape(x, (1, -1))

        # projection matrix with fourier basis functions
        N = len(x[0])
        dx = self.L/N
        k = np.reshape(fftfreq(N, dx), (1, -1))
        coords = np.reshape(self.dyn_coords, (-1, 1))
        P = np.exp(2j * np.pi * coords * k)

        # Calculate fourier coefficients
        if np.all(np.diff(coords) == dx):
            # regular grid
            A = fft.fft(x.T, axis=0) / N
        else:
            # non-regular grid
            A, _, _, _ = np.linalg.lstsq(P, x.T, rcond=None)

        # Apply differentiation
        A = A * (2j * np.pi * k.T)**ndiff

        # interpolation function
        def interpolator(coords):
            coords = np.reshape(coords, (-1, 1))
            expk = np.exp(2j * np.pi * k * coords)
            return np.real(expk @ A).T

        return interpolator

    def state2poly(self, x, ndiff=0):
        """Convert states in polynomial interpolation."""
        ndiff = int(ndiff)
        if np.ndim(x) == 1:
            x = np.reshape(x, (1, -1))

        # Cell coordinates grouped by cell
        coords = np.reshape(self.dyn_coords, (self.ncells, self.order+1))
        local, cell = self.global2local(coords)

        # Values grouped by cell.
        x = np.reshape(x, (-1, self.ncells, self.order+1))

        # polynomial basis for each cell
        dx = self.L / self.ncells
        basis = lagrange_basis(local[0])
        basis = [p.deriv(ndiff) * (2/dx)**ndiff for p in basis]

        def interpolator(coords):
            coords = np.reshape(coords, (-1))

            local, cell = self.global2local(coords)
            P = np.array([p(local) for p in basis]).T

            y = []
            for x1 in x:
                y.append([np.dot(p, x1[n, :]) for p, n in zip(P, cell)])

            return y

        return interpolator

    def state2legendre(self, x, ndiff=0):
        """Convert states given as Legendre coefficients into polynomial interpolator."""
        from scipy.special import legendre

        ndiff = int(ndiff)
        if np.ndim(x) == 1:
            x = np.reshape(x, (1, -1))

        # Values grouped by cell.
        x = np.reshape(x, (-1, self.ncells, self.order+1))

        # polynomial basis for each cell
        dx = self.L / self.ncells
        #x /= np.reshape(dx**np.arange(0,self.order+1),(1,1,-1))
        #basis = [legendre(n) * (dx/2)**n for n in range(self.order+1)]
        #basis = [p.deriv(ndiff) * (2/dx)**ndiff for p in basis]
        basis = [legendre(n) for n in range(self.order+1)]
        basis = [p.deriv(ndiff)*(2/dx)**ndiff for p in basis]

        def interpolator(coords):
            coords = np.reshape(coords, (-1))

            local, cell = self.global2local(coords)
            P = np.array([p(2*local-1) for p in basis]).T

            y = np.zeros((np.size(x, 0), len(cell)))
            for n, celln in enumerate(cell):
                y[:, n] = x[:, celln, :]@P[n, :].T

            return y

        return interpolator

    def obs_factory(self, function, ndiff):
        """Create sampling operator."""
        def obs(x, t):
            interpolator = function(x, int(ndiff))

            y = interpolator(self.obs_coords(t))
            if np.ndim == 1:
                y = y[0]

            return y

        return obs

    def to_state(self, t):
        """
        Create a vector representation of current state. 

        Parameters
        ----------
        t : float
            Pseudo-time.

        Returns
        -------
        Array of float
            Vector with function values. 

        """
        return self.functions[self.member](self.dyn_coords, t)

    def step(self, x, t, dt):
        """
        Get values function on self.dyn_coords. 

        Parameters
        ----------
        x : 1D array of float | 2D array of float
            Array containing function values at dyn_coords along axis=1.
        t : float
            Pseudo-time.

        Returns
        -------
        x : array of float
            Array with function values at time t. 

        """
        if np.ndim(x) == 1:
            x[:] = self.functions[self.member](self.dyn_coords, t)
        elif np.ndim(x) == 2:
            for n in range(np.size(x, 0)):
                m = np.mod(n, len(self.functions))
                x[n, :] = self.functions[m](self.dyn_coords, t)
        else:
            raise TypeError("x must be 1D or 2D array.")

        return x

    def step_legendre(self, x, t, dt):
        """
        Get values function on self.dyn_coords. 

        Parameters
        ----------
        x : 1D array of float | 2D array of float
            Array containing function values at dyn_coords along axis=1.
        t : float
            Pseudo-time.

        Returns
        -------
        x : array of float
            Array with function values at time t. 

        """
        dx = self.L/self.ncells
        #scale = (2/dx)**np.arange(0, self.order+1)
        #scale = np.reshape(scale, (1,-1))
        scale = 1.

        if np.ndim(x) == 1:
            coef = self.functions[self.member].vals2coef(t)
            x[:] = np.reshape(coef.T * scale, (-1))
        elif np.ndim(x) == 2:
            for n in range(np.size(x, 0)):
                m = np.mod(n, len(self.functions))
                coef = self.functions[m].vals2coef(t)
                x[n, :] = np.reshape(coef.T * scale, (-1))
        else:
            raise TypeError("x must be 1D or 2D array.")

        return x

    def sample_coords(self, N, t):
        """ Sample functions with codomain in space at dyn_coords."""
        x = []
        for n in range(N):
            m = np.mod(n, len(self.functions))
            x.append(self.functions[m](self.dyn_coords, t))

        return np.array(x)

    def sample_legendre(self, N, t):
        """ Sample functions with codomain in Legendre space at dyn_coords."""
        x = []

        dx = self.L/self.ncells
        #scale = (2/dx)**np.arange(0, self.order+1)
        #scale = np.reshape(scale, (1,-1))
        scale = 1

        for n in range(N):
            m = np.mod(n, len(self.functions))
            coef = self.functions[m].vals2coef(t)
            x.append(np.reshape(coef.T * scale, (-1)))

        return np.array(x)

    def apply_uniform_weighting(self, Lo):
        """
        Smooth signal by convoluting it with a rectangular window of size Lo. 
        This function returns weighting for to be applied to Fourier coefficients. 

        If F[n] is the Fourier coefficient associated with wavenumber k[n] then
        w[n]F[n] is the associated Fourier coefficient of the smoothed function. 

        Parameters
        ----------
        Lo : float > 0 
            Width rectangular window. 

        Returns
        -------
        Fourier weights. 

        """

        for n, function in enumerate(self.functions):
            a = 0.5 * function.k * Lo
            w = np.array([np.sin(a1)/a1 if np.abs(a1)
                         > 1e-8 else 1. for a1 in a])
            self.functions[n].A = function.A * w

    def apply_gaussian_weighting(self, Lo):
        """
        Smooth signal by convoluting it with a Gaussian window of size Lo. 
        This function returns weighting for to be applied to Fourier coefficients. 

        If F[n] is the Fourier coefficient associated with wavenumber k[n] then
        w[n]F[n] is the associated Fourier coefficient of the smoothed function. 

        Parameters
        ----------
        Lo : float > 0 
            Width rectangular window. 

        """

        for n, function in enumerate(self.functions):
            w = np.exp(-(function.k * Lo)**2 / 4**2)
            self.functions[n].A *= w

    def apply_legendre(self):
        """
        Project signal onto Legendre polynomial basis. 
        """

        for n, function in enumerate(self.functions):
            leg = LegendreFunction(function, self.order, self.ncells)
            self.functions[n] = leg

    def apply_cutoff(self, fraction):
        """
        Remove small-scale part of signal. 

        Parameters
        ----------
        fraction : float [0,1]
            Retain only part of signal that consists of wavenumbers smaller 
            than fraction * max_wavenumber. 

        Returns
        -------
        None.

        """

        for n, function in enumerate(self.functions):
            k = function.k
            self.functions[n].A *= np.where(np.abs(k) <= np.max(k) * fraction,
                                            1, 0)


class LinearFunction:
    
    def __init__(self):
        dx = 1e-5
        self.poly = [np.polynomial.Polynomial([0, -dx]),
                     np.polynomial.Polynomial([0,  dx])]
    
    def __call__(self, x, t, ndiff=0):
        p0 = self.poly[0].deriv(ndiff)(x)
        p1 = self.poly[1].deriv(ndiff)(x)
        return np.where(x>=4e6, p1, p0)
        
    def draw(self,seed):
        pass 
    
    def integ(self):
        new = LinearFunction()
        new.poly = self.poly
        for n,p in new.poly:
            new.poly[n] = p.integ()
        return new    
    
    def deriv(self):
        new = LinearFunction()
        new.poly = self.poly
        for n,p in new.poly:
            new.poly[n] = p.deriv()
        return new    
        
class FourierFunction:
    """ 
    Class representing a function that is defined by its discrete Fourier transform. 
    This acts as a 'view' to a function created using the 'model' in noise factory. 

    """

    def __init__(self, k, A, seed=1000):
        """
        Class constructor.

        Parameters
        ----------
        k : array of float 
            Wavenumbers
        A : array of float 
            Fourier amplitudes. 

        """        
        self.k = k
        self.A = A     
        
        exp_ind = self.k > 2 * np.pi / (0.5 * (8e6/79)) 
        #self.A[~exp_ind] *= 0 #IP
           
        self.F = np.zeros_like(A)
        self.base_seed = seed
        self.seed = None
 
    @property
    def K(self):
        raise Exception('call to K depreciated')
        return int((len(self.k) - 1)/2)

    def __call__(self, x, t, ndiff=0):
        """
        Return value evaluated at points in space. 

        Parameters
        ----------
        x : float | array of floats 
            Coordinates to evaluate function. 
        ndiff : int>=0
            Order of differentation.

        Returns
        -------
        Function value at x. 

        """        
        self.draw(int(t) + self.base_seed)
        ndiff = int(ndiff)

        has_iter = hasattr(x, '__iter__')
        kx = np.reshape(self.k, (1, -1)) * np.reshape(x, (-1, 1))
        Dkx = np.reshape(self.k, (1, -1)) * np.ones_like(kx)
        amp = np.reshape(self.F, (1, -1))
        #amp[0,1:] *= np.sqrt(.5) #IP NEW

        value = np.sum(amp * np.exp(1j * kx) * (1j * Dkx)**ndiff,
                       axis=1)

        if has_iter: 
            return np.real(value) 
        else:
            return np.real(value[0])

    def draw(self, seed):
        """
        Draw new Fourier coefficients. 

        Parameters
        ----------
        seed : int
            Seed used to generate random phases and amplitudes. 

        Returns
        -------
        None.

        """    
        if self.seed != seed:
            np.random.seed(seed)
            self.seed = seed

            s = np.shape(self.A)
            self.F = self.A * (np.random.normal(size=s) +
                               1j * np.random.normal(size=s))

    def integ(self):
        """
        Integrand of current function. In only exact integrand if self.F[0] else
        polynomial must be added. 

        Returns
        -------
        Primative as FourierFunction

        """
        A = np.zeros_like(self.A, dtype=complex)
        A[1:] = np.array(self.A[1:], dtype=complex) / (1j * self.k[1:])
        return FourierFunction(self.k, A, self.base_seed)

    def deriv(self):
        """
        Derivative of current function.

        Returns
        -------
        Derivative as FourierFunction

        """
        return FourierFunction(self.k, self.A * (1j * self.k),
                               self.base_seed)


class PolyFourierFunction:

    def __init__(self, fouriers, polynomials):
        """
        Class constructor.

        """
        from scipy.interpolate import lagrange

        self.fouriers = np.array(fouriers, dtype=object, copy=False)
        self.polynomials = np.array(polynomials, dtype=object, copy=False)

        is_zero = self.is_zero(self.fouriers, self.polynomials)
        self.fouriers = self.fouriers[~is_zero]
        self.polynomials = self.polynomials[~is_zero]

        N_int = 12
        self.gauss_lobatto = np.polynomial.legendre.leggauss(N_int)[0]
        self.lagranges = np.array([
            np.polynomial.Polynomial(
                (lagrange(self.gauss_lobatto, w)).coef[::-1])
            for w in np.eye(N_int)
        ], dtype=object)

    def __call__(self, x, t, ndiff=0):
        def func(m, n): return (self.polynomials[m].deriv(n)(x)
                                * self.fouriers[m](x, t, ndiff-n)
                                * math.comb(ndiff, n))

        if len(self.polynomials) == 0:
            return np.zeros_like(x)
        else:
            return sum((func(m, n) for m in range(len(self.polynomials)) for n
                        in range(ndiff+1)))

    def difference(self, x1, x2, t, ndiff):
        s = np.shape(x1)
        x1 = np.reshape(x1, (1, -1))
        x2 = np.reshape(x2, (1, -1))

        print('s', np.shape(x1), x1)

        # Collect all wavenumbers and associated amplitudes
        kF = []
        for fourier, poly in zip(self.fouriers, self.polynomials):
            kF += list(zip(fourier.k, fourier.F * poly(x1)))
            kF += list(zip(fourier.k, fourier.F * poly(x2)))
        kF = np.array(kF).T
        print('kF', len(self.fouriers), np.shape(kF))

        # Calculate difference for each wavenumber
        k = np.unique(kF[0])
        vals = np.array([np.sum(kF[1][kF[0] == k1]) *
                        (np.exp(k1*x1)-np.exp(k1*x2)) for k1 in k])

        print('vals ', k[np.argmin(np.abs(vals))],
              k[np.argmax(np.abs(vals))], np.max(k))

        # Sum in ascending order
        vals = np.sum(vals, axis=0)
        vals = np.reshape(vals, s)

        return vals

    def is_zero(self, fouriers, polynomials):
        return np.array([np.all(np.isclose(p.coef, 0.)) or np.all(np.isclose(f.A, 0.))
                         for p, f in zip(polynomials, fouriers)], dtype=bool)

    def one(self):
        K = self.fouriers[0].K
        k = self.fouriers[0].k
        A = np.append(1, np.zeros((K-1,)))
        return FourierFunction(k, A)

    def deriv(self):
        polynomials = np.array([poly.deriv(1) for poly in self.polynomials],
                               dtype=object)
        fouriers = np.array([fourier.deriv() for fourier in self.fouriers],
                            dtype=object)

        is_zero = self.is_zero(fouriers, polynomials)
        polynomials = np.append(
            polynomials[~is_zero], self.polynomials[~is_zero])
        fouriers = np.append(self.fouriers[~is_zero], fouriers[~is_zero])
        return PolyFourierFunction(fouriers, polynomials)

    # Partial integration for fourier*poly
    def integ1(self, fourier, poly):
        fouriers, polys = [], []
        if not np.isclose(fourier.A[0], 0.):
            fouriers = [FourierFunction([0], fourier.A[:1], fourier.base_seed)]
            polys = [poly.integ()]

        if np.all(fourier.A[1:] == 0.):
            return fouriers, polys

        for n in np.arange(0, poly.degree()+1, 1):
            polys.append(poly.deriv(n) * (-1)**n)

            if n == 0:
                fouriers.append(fourier.integ())
            else:
                fouriers.append(fouriers[-1].integ())

        return fouriers, polys

    def integ(self):
        fouriers, polys = [], []
        for fourier, poly in zip(self.fouriers, self.polynomials):
            fourier_int, poly_int = self.integ1(fourier, poly)
            fouriers += fourier_int
            polys += poly_int

        return PolyFourierFunction(fouriers, polys)

    def integrate(self, time, lbound, ubound):
        from scipy import integrate
        value = 0

        for fourier, poly in zip(self.fouriers, self.polynomials):
            # Update Fourier coefficients to current time.
            _ = fourier(lbound, time)
            k = (1j * fourier.k)

            # Determine interpolation method
            int_poly = np.logical_and(fourier.k <= 0.2,
                                      ~np.isclose(np.abs(fourier.F), 0.))
            int_exp = np.logical_and(fourier.k > 0.2,
                                     ~np.isclose(np.abs(fourier.F), 0.))

            if any(int_exp):
                # Carry out intergration by parts
                lfourier = (fourier.F[int_exp] *
                            np.exp(k[int_exp]*lbound)).reshape((1, -1))
                ufourier = (fourier.F[int_exp] *
                            np.exp(k[int_exp]*ubound)).reshape((1, -1))
                # Polynomials
                lpoly = np.array([(-1)**n*(poly.deriv(n))(lbound) for n in
                                  range(poly.degree()+1)]).reshape((-1, 1))
                upoly = np.array([(-1)**n*(poly.deriv(n))(ubound) for n in
                                  range(poly.degree()+1)]).reshape((-1, 1))

                # Integration table
                table = upoly*ufourier - lpoly*lfourier
                for n in range(poly.degree()+1):
                    table[n] *= k[int_exp]**(-1-n)

                # Add integrand to output
                value += np.sum(np.real(table))

            if any(int_poly):
                gauss = self.gauss_lobatto.reshape(-1, 1)
                val1 = np.sum(fourier.F[int_poly].reshape((1, -1)) *
                              np.exp(gauss * k[int_poly].reshape((1, -1))),
                              axis=1)
                poly1 = sum([lagrange * v for lagrange,
                            v in zip(self.lagranges, val1)])
                primitive = (poly * poly1).integ()
                value += np.real(primitive(1))-np.real(primitive(-1))

        return value


class LegendreFunction:
    """
    Decorator creating new function by projecting FourierFunction onto 
    a Legendre basis. 

    """

    def __init__(self, fourier_function, order, ncells):
        from scipy.special import legendre

        self.fourier_function = fourier_function
        self.ncells = ncells
        self.time = None

        self.k = self.fourier_function.k
        self.L = 2*np.pi/np.min(self.k[self.k > 0])
        self.wavelength = 2*np.pi / np.max(self.k)
        self.dx = self.L / self.ncells

        self.order = order
        self.poly = [self.legendre(n) for n in range(order+1)]
        self.bipoly = [p * .5 * (2*n+1) for n, p in enumerate(self.poly)]

    def build4integrate(self):
        from scipy.interpolate import lagrange

        self.exp_ind = self.k > 2 * np.pi / (0.5 * self.dx) #IP old
        #self.exp_ind = self.k >= self.k[40] 
        
        if any(self.exp_ind):
            k = self.fourier_function.k[self.exp_ind].reshape(-1, 1)
            r = np.arange(0, self.ncells).reshape(1, -1) * self.dx
            self.expk = np.exp(1j * k * r)
            
        self.poly_ind = ~self.exp_ind
        if any(self.poly_ind):
            self.gauss_lobatto = np.polynomial.legendre.leggauss(8)[0]

            lagranges = np.array([
                np.polynomial.Polynomial(
                    (lagrange(self.gauss_lobatto, w)).coef[::-1])
                for w in np.eye(len(self.gauss_lobatto))], dtype=object)

            self.gaussw = np.zeros(
                (len(self.bipoly), 1, len(self.gauss_lobatto)))
            for n, poly in enumerate(self.bipoly):
                for m, lagrange in enumerate(lagranges):
                    self.gaussw[n, 0, m] = (
                        poly * lagrange).integ()(1) - (poly * lagrange).integ()(-1)

            self.gauss_lobatto = self.gauss_lobatto.reshape(1, -1) * 0.5 + 0.5
            self.gauss_lobatto = self.gauss_lobatto + \
                np.arange(0, self.ncells).reshape(-1, 1)
            self.gauss_lobatto = self.gauss_lobatto.reshape((-1,)) * self.dx
            
            self.fourier_function_LP = FourierFunction(self.fourier_function.k+0.,
                                                       self.fourier_function.A+0.,
                                                       self.fourier_function.base_seed+0)
            self.fourier_function_LP.A[~self.poly_ind] = 0.

    @staticmethod
    def legendre(order):
        c = np.zeros((order+1,))
        c[-1] = 1.
        p = np.polynomial.legendre.leg2poly(c)
        return np.polynomial.Polynomial(p)
            
    def __call__(self, x, t, ndiff=0):
        """
        Return value evaluated at points in space. 

        Parameters
        ----------
        x : float | array of floats 

        Returns
        -------
        Function value at x. 

        """
        ndiff = int(ndiff)
        y = self.fourier_function(x, t)

        coef = self.vals2coef(t)
        local, ind = self.global2local(x)
        P = np.array([p.deriv(ndiff)(local) * (2/self.dx) **
                     ndiff for n, p in enumerate(self.poly)]).T
                     
        return np.array([np.dot(p, coef[:, n]) for p, n in zip(P, ind)])

    def global2local(self, x):
        """ 
        Convert global coordinates into cell coordinates and cell indices. 

        Parameters
        ----------
        x : array of float 

        """
        ind = np.array(x/self.dx, dtype=int)
        local = 2. * np.mod(x / self.dx, 1.) - 1.
        return local, ind

    def local2global(self, local, ind):
        """
        Convert cell indices and cell coordinates into 
        """
        return self.dx * (ind + 0.5) + 0.5 * local

    def vals2coef_cell(self, t, cell):
        """
        Convert signal in a cell into coefficients for different Legendre
        polynomials. 

        Parameters
        ----------
        t : float
            Pseudo-time. 
        cell : int>=0
            Cell index. 

        Returns
        -------
        Array of float.
            Legendre coefficients. 

        """
        from scipy.integrate import quad
        def trans(ksi): return self.dx * (0.5 + cell) + 0.5 * ksi * self.dx
        def f(ksi): return self.fourier_function(trans(ksi), t)

        coef = []
        for n, p in enumerate(self.bipoly):
            def integrand(ksi): return f(ksi) * p(ksi)
            coef.append(quad(integrand, -1, 1)[0])

        return np.array(coef)

    def vals2coef_cell_integ(self, t, cell):
        """
        Convert signal in a cell into coefficients for different Legendre
        polynomials. 

        Parameters
        ----------
        t : float
            Pseudo-time. 
        cell : int>=0
            Cell index. 

        Returns
        -------
        Array of float.
            Legendre coefficients. 

        """
        # Shifted Fourier series
        k = self.fourier_function.k * self.dx * 0.5
        A = self.fourier_function.A * np.exp(1j * k * (2*cell+1.))
        fourier = FourierFunction(k, A, self.fourier_function.base_seed)

        coef = []
        for p in self.bipoly:
            c = PolyFourierFunction([fourier], [p]).integrate(t, -1, 1)
            coef.append(c)

        return np.array(coef)

    @staticmethod
    def concatenate_poly(p1, p2):
        return sum([c * p2**n for n, c in enumerate(p1.coef)])

    def vals2coef(self, t):
        """
        Convert signal into array of Legendre coefficients. 

        Parameters
        ----------
        t : float
            Pseudo-time. 

        Returns
        -------
        Array of float.
            Legendre coefficients for different grid cells. 


        """
        if t != self.time:
            self.time = t
            self._coef = np.zeros((self.order+1, self.ncells))

            #Create polynomial basis. 
            if not hasattr(self, 'poly_ind') or not hasattr(self, 'exp_ind'):
                self.build4integrate()

            if any(self.poly_ind): 
                #Evalualute the LP part of the Fourier series at Gauss-Legendre points. 
                vals = self.fourier_function_LP(self.gauss_lobatto, t).reshape((self.ncells, -1))
                    
                #Integrate to project onto polynomial. 
                for n, w in enumerate(self.gaussw):
                    self._coef[n] += np.sum(w * vals, axis=1)
                    
                #This ensures time is evaluated. 
                self.fourier_function(0,t)
            else:
                self.fourier_function(0, t)

            if any(self.exp_ind): 
                Fexpk = self.expk * self.fourier_function.F[self.exp_ind].reshape(-1, 1)
                k = 0.5 * 1j * self.k[self.exp_ind] * self.dx
                for n, poly in enumerate(self.bipoly):
                    for d in range(n+1):
                        self._coef[n] += np.sum(np.real(
                            (np.roll(Fexpk, (0, -1)) * poly.deriv(d)(1)
                             - Fexpk * poly.deriv(d)(-1))
                            * (k**(-d-1)).reshape((-1, 1)) * (-1)**d
                        ), axis=0)


        return self._coef
    
class VoidLimiter:
    
    def __init__(self):
        pass
    
    def __call__(self, E):
        return E
    
class MuscleLimiter(VoidLimiter):
    
    def __init__(self, model):
        self.dx = model.functions[0].dx 
        self.poly = model.functions[0].poly
        self.order = model.functions[0].order 
        self.s = None
        
    def __call__(self, E):
        Nens = np.size(E,0)
        E = E.reshape((Nens,-1,self.order+1))
        
        for n,e in enumerate(E):
            E[n] = self.apply_muscle(e)
            
        return E.reshape((Nens,-1))

    def apply_muscle(self, x):
        from matplotlib import pyplot as plt
        """ Apply MUSCL slope limiter."""
        
        if len(self.poly)<=1:
            return x
        if self.s is None:
            self.s = np.zeros((3, np.size(x,0)))
            
        #Slope based on Legendre coefficient
        derv = 2 * self.poly[1].coef[1]
        self.s[0] = derv * x[:,1] 
        #Slop based on finite difference to left 
        self.s[1] = (x[:,0] - np.roll(x[:,0], 1)) 
        #Slope base on finite difference to right
        self.s[2] = (np.roll(x[:,0],-1) - x[:,0]) 
                
        #Sign of slope after limiting
        sign = np.mean(np.sign(self.s), axis=0)
        sign = np.array(sign, dtype=int)
        
        #Index minimum slope 
        mins = np.min(np.abs(self.s), axis=0)
        argmins = np.argmin(np.abs(self.s), axis=0)
        
        #Slope limiting.
        x[:,1] = sign * mins / derv 
         
        #If limiting applied, remove higher orders. 
        mask = np.logical_or(argmins!=0, sign==0)
        x[mask,2:] = 0.0
        
        return x

def lagrange_basis(local):
    from scipy.interpolate import lagrange

    poly = []
    for n in range(len(local)):
        w = np.zeros_like(local)
        w[n] = 1.
        poly.append(lagrange(local, w))

    return poly
