#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:01:02 2022

This model contains different MetaFields that can be used as forcings in
the ice simulations.

@author: ivo
"""

import datetime
import numpy as np
from copy import copy 
from abc import ABC, abstractmethod
from pickle import NONE


class SpectrumDrawer(ABC):

    def build_fourier(self):
        self._time_seed = None
        self.rad = complex(0, 2*np.pi)
        self.freq = np.arange(0, self.rank)
        self.fourier0 = np.array(self.spectrum(self.freq), dtype=float)
        self.fourier0[0] = 0.
        self.fourier0 /= np.sqrt(0.5*np.sum(self.fourier0**2))

    @abstractmethod
    def _set_seed(self, time):
        pass

    def _reseed(self, time):
        self._set_seed(time)
        self.fourier = self.fourier0 * np.exp(self.rad *
                                              np.random.uniform(size=np.shape(self.freq)))

    def draw(self, time, coordinates):
        if self._time_seed != time:
            self._reseed(time)

        values = np.zeros_like(coordinates, dtype=complex)
        for freq, fourier in zip(self.freq, self.fourier):
            values += np.exp(self.rad*freq*coordinates)*fourier

        if hasattr(coordinates, '__iter__'):
            return np.real(values)
        else:
            return np.real(values)+0


def bellcurve_wind(phase):
    W = 0.1
    signal = np.array([1.]) * np.exp(-0.5 * phase**2 / W**2)
    return signal


class Drawer(ABC):

    @abstractmethod
    def draw(t, x):
        pass


class RandomDrawer(Drawer):

    def __init__(self, base_seed=1000):
        self.base_seed = base_seed

    def draw(self, t, x):
        return np.random.normal(size=np.shape(x))


class TimeDrawer(Drawer):

    def __init__(self, base_seed=1000, size_storage=1000):
        self.base_seed = base_seed
        self.previous_seed = None
        self.size_storage = size_storage

    def seed(self, t):
        t = int(t.timestamp())
        new_seed = t + self.base_seed

        if new_seed != self.previous_seed:
            self.previous_seed = new_seed
            np.random.seed(self.previous_seed)
            self.storage = np.random.normal(size=self.size_storage)

    def draw(self, t, x):
        self.seed(t)

        if hasattr(x, '__iter__'):
            hashes = [int(self.hash(x1) * self.size_storage) for x1 in x]
        else:
            hashes = int(self.hash(x) * self.size_storage)

        return self.storage[hashes]

    def decompose(self, number):
        from decimal import Decimal
        number = Decimal(number)

        sign, digits, exponent = number.as_tuple()
        sign = -2*sign+1
        exponent = len(digits) + exponent
        mantisse = float(number.scaleb(-1))
        return sign, mantisse, exponent

    def hash(self, x):
        sign, mantisse, exponent = self.decompose(x)

        mfac = [4559623, 8917763]
        afac = [.364987, .764987]
        hash1 = sign*exponent*mantisse
        for mfac1, afac1 in zip(mfac, afac):
            hash1 = np.mod(mfac1*hash1+afac1, 1.)

        return hash1


class Forcing(ABC):

    @abstractmethod
    def forcing(t, x):
        pass


class EnsembleForcing(Forcing):

    def __init__(self, forcings):
        self.forcings = forcings
        self._n = 0

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, no_member):
        if no_member > len(self.forcings):
            msg = "Cannot set member to {:d}. Only {:d} members in ensemble."
            raise ValueError(msg.format(no_member, len(self.forcings)))
        else:
            self._n = no_member

    def forcing(self, t, x):
        return self.forcings[self.n].forcing(t, x)

    def __call__(self, t, x):
        return self.forcing(t, x)

class AR(ABC):
    
    def default_init(self, dt, mean, var, r, base_seed):
        self.dt = dt 
        self.mean = np.reshape(mean,(-1))
        self.std = np.reshape(np.sqrt(var * dt.total_seconds()), (-1))
        self.base_seed = base_seed
        self.r = r
        
        self.shape = np.shape(mean)
        
        self.time = None 
        self.seed = None
        self.storage = {}
        
    def build_matrix(self):
        self.R = np.diag(np.ones_like(self.r),1)
        
        self.R[0] = 0.*self.R[0]
        self.R[0,0] = 1.
        
        r0 = np.sqrt(1.-np.sum(self.r**2))
        self.R[-1] = np.append(r0, self.r)
            
        self.invR=np.linalg.inv(self.R)
            
    def forward(self):
        self.time = self.time + self.dt
        self.set_seed(self.time)
        self.values[0] = np.random.normal(scale=self.std)
        self.values = self.R @ self.values
            
    def backward(self):
        self.time = self.time - self.dt 
        
        self.values = self.invR @ self.values 
        self.set_seed(self.time)
        self.values[0] = np.random.normal(scale=self.std)
          
    @abstractmethod 
    def build_values(self, time):
        pass
        
    def sample(self, time): 
        if self.time is None:
            self.build_matrix()
            self.build_values(time)
                      
        while self.time < time:
            self.forward()
        while self.time > time:
            self.backward()
            
        return np.reshape(self.values[-1] + self.mean, self.shape) 
    
    def set_seed(self, time):
        new_seed = int(time.timestamp()) + self.base_seed

        if new_seed != self.seed:
            self.seed = new_seed
            np.random.seed(self.seed)
    
class AR0(AR):
    
    def __init__(self, dt, mean=0., var=1., base_seed=1000):
        self.default_init(dt, mean, var, np.array([]), base_seed)
        
    def build_values(self, time):
        pass
        
    def build_matrix(self):
        pass
    
    def sample(self, time):
        if time!=self.time:
            self.time = time
            self.set_seed(self.time)
            self.values=np.random.normal(loc=self.mean, scale=self.std)
            
        return np.reshape(self.values+self.mean, self.shape)

class AR1(AR):

    def __init__(self, dt, mean=0., var=1., T=None, base_seed=1000):
        if T is None:
            r = np.exp(-1.)
        else:
            r = np.exp(-dt.total_seconds()/T.total_seconds())
        
        self.default_init(dt, mean, var, np.array([r]), base_seed)
       
    def build_values(self, time):        
        self.values = np.zeros((2,np.size(self.std)))

        self.time = time
        self.set_seed(self.time)
        self.values[0] = np.random.normal(scale=self.std)
        self.values[1] = np.random.normal(scale=self.std)
        
class SpectralNoise:
    """ 
    Function representing stochastic noise with a certain spectral profile.
    
    :type c_var: float 
    :param c_var: variance per unit time in phase speed. 
    :type wavelengths: Numpy array float>0.0
    :param wavelengths: spatial wavelengths in noise.
    :type amplitudes: Numpy array float>0.0
    :param amplitudes: amplitude of the different wavelengths in noise.
    
    """
    
    def __init__(self, phase_generator, wavelengths, amplitudes):
        """ Class constructor. """
        self.k = 2*np.pi/np.array(wavelengths, dtype=complex) * complex(0.0, 1.0)
        self.amplitudes = np.array(np.abs(amplitudes),dtype=complex)
        self.phase_generator = phase_generator
        self.time = None
        
    def _update_time(self, time):
        """
        Move noise forward in time. 
        
        :type time: datetime.datetime object 
        :param time: current time.
        
        """
        if self.time!=time:
            self.time = time
            dc = np.array([ar.sample(time) for ar in self.phase_generator]) * complex(0,1.)
            self.amplitudes = np.abs(self.amplitudes) * np.exp(dc)
        
    def __call__(self, time, coordinate):
        """
        Evaluate noise at time and position. 
        
        :type time: datetime.datetime 
        :param time: current time.
        :type coordinate: Numpy array of float 
        :param coordinate: position in grid. 
        
        """
        self._update_time(time)
        value = np.sqrt(2) * np.sum(self.amplitudes * np.exp(coordinate * self.k))
        return np.real( value )             
        
class MovingWave(Forcing):

    def __init__(self, model, base_function,
                 base_velocity=0., base_amplification=1.,
                 T_ramp=datetime.timedelta(seconds=0),
                 ):

        self.base_function = base_function
        self.base_velocity = base_velocity
        self.base_amplification = base_amplification
        self.spectral = None
        self.amplification = None
        self.T_ramp = T_ramp
        self.previous_time = None

        self.dt = model.dt
        self.ref_time = model.ref_time
        self.length = model.length
        self.center = None
        self.set_noise()

    def set_noise(self, amplification=lambda t:0., center=lambda t:0., spectral=None):
        self.noise_amplification = amplification 
        self.noise_center = center
        self.noise_spectral = spectral
            
    def update_amplification(self, time):
        self.amplification = self.base_amplification + self.noise_amplification.sample(time)
        
    def update_center(self, time):
        t = (time-self.ref_time).total_seconds()
        center  = t * self.base_velocity        
        self.center = center + self.noise_center.sample(time)

    def forcing(self, t, x):   
        self.update_amplification(t)
        self.update_center(t)
        
        if self.noise_spectral is not None:
            signal = self.noise_spectral(t,x)
        else:
            signal = 0.
            
        #Relative time.
        t=(t-self.ref_time)/datetime.timedelta(days=2)

        #Behaviour based on relative time.
        tmode = np.mod(int(t),4)
        if tmode==0:
            phase = np.mod(x+self.center, self.length)/self.length - 0.5
            ramp_factor = -1.
        elif tmode==2:
            phase = np.mod(x-self.center, self.length)/self.length - 0.5
            ramp_factor = 1.
        else:
            phase = (np.mod(x, self.length) - np.mod(self.center, self.length)) / self.length 
            ramp_factor = 0.
            
        #Smooth out jumps in time. 
        ramp_factor *= 1.-np.cos(np.pi*t)**4
        
        return ramp_factor * (signal + self.amplification * self.base_function(phase))
    
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def clone(self, N=1):
        clones = []
        for n in range(0, N):
            clones.append(self.copy())
            #clones[-1].seeder.base_seed += int(n*100)
        return clones

    def __call__(self, t, x):
        return self.forcing(t, x)
