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
from abc import ABC, abstractmethod


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
        
    def next(self, draw, *args):
        r0 = 1. 
        new  = self.mu
        for arg, r in zip(args, self.r):
            new += (arg - self.mu) * r
            r0  -= r**2
        new += np.sqrt(r0 * self.var) * draw
        return new
    
class AR0(AR):
    
    def __init__(self, dt, mean, variance):
        self.mu = mean
        self.r = []
        self.var = variance

class AR1(AR):

    def __init__(self, dt, mean, variance, correlation_time):
        self.mu = mean
        self.r = [np.exp(-dt / correlation_time)]
        self.var = variance
        
class MovingWave(Forcing):

    def __init__(self, model, base_function,
                 base_velocity=0., base_amplification=1.,
                 T_ramp=datetime.timedelta(seconds=0)):

        self.base_function = base_function
        self.base_velocity = base_velocity
        self.base_amplification = base_amplification
        self.amplification = None
        self.T_ramp = T_ramp
        self.previous_time = None

        self.dt = model.dt
        self.ref_time = model.ref_time
        self.length = model.length
        self.center = None
        self.set_noise()

    def set_noise(self, drawer=RandomDrawer(), signal=0., velocity=0.,
                  amplification=0.):
        self.drawer = drawer
        self.amplitude_noise = {}
        self.amplitude_noise['signal'] = signal
        self.amplitude_noise['velocity'] = velocity
        self.amplitude_noise['amplification'] = amplification

    def step_forward(self, time):
         if self.previous_time is None or time > self.previous_time :

            #self.velocity =
            #amp = self.amplitude_noise['velocity'] * \
            #    self.drawer.draw(time, -.33)
            #self.velocity += np.ones_like(self.velocity) * amp

            #amp = self.amplitude_noise['amplification'] * \
            #    self.drawer.draw(time, -.66)
            #self.amplification += np.ones_like(self.amplification) * amp
            
            #dt = (time - self.previous_time).total_seconds()
            #self.center = self.center + self.velocity * dt
            
            #Amplification factor total wind field. 
            dt = self.dt.total_seconds()
            var=self.amplitude_noise['amplification']**2
            draw = self.drawer.draw(time, -.66)
            
            T_corr=datetime.timedelta(minutes=10.)
            if self.amplification is None:
                ar=AR0(self.dt, self.base_amplification, var)
                self.amplification = ar.next(draw)
            else:
                ar=AR1(self.dt, self.base_amplification, var, T_corr)
                self.amplification = ar.next(draw, self.amplification)
                
            #Position center storm
            T_corr=datetime.timedelta(days=5.)
            var=(self.amplitude_noise['velocity'])**2 
            draw = self.drawer.draw(time, -.33)
            center  = (0.5 * self.length + (time-self.ref_time).total_seconds()
                       * self.base_velocity)
            
            if self.center is None:
                ar=AR0(self.dt, center, var)
                self.center = ar.next(draw)
            else:
                ar=AR1(self.dt, center, var, T_corr)
                self.center = ar.next(draw, center)

            self.previous_time = time

    def forcing(self, t, x):
        self.step_forward(t)
        phase = (np.mod(x, self.length) - np.mod(self.center, self.length)) / self.length 
        
        t=(t-self.ref_time)/datetime.timedelta(days=1)
        ramp_factor  = np.mod(int(t)+1,2)
        ramp_factor *= 1.-np.cos(np.pi*t)**4
        
        #t=(t-self.ref_time)/datetime.timedelta(hours=1)
        #ramp_factor = min(1.,t)
        
        signal = ramp_factor * self.amplification * self.base_function(phase)
        #signal += self.amplitude_noise['signal'] * self.drawer.draw(t,x)
        return signal

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
