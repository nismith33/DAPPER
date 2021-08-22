###############
# Description #
###############

"""Settings as in `bib.grudzien2020numerical`.

Set-up with three different `step` functions, using different SDE integrators.
The truth twin is generated by the order 2.0 Taylor scheme, for accuracy with
respect to convergence in the strong sense for generating the observation sequence.
The model simulation step sizes are varied in the settings below to demonstrate the
differences between the commonly uses Euler-Maruyama and the more statistically
robust Runge-Kutta method for SDE integration. See README below.
"""

###########
# imports #
###########

import numpy as np

import dapper.mods as modelling
from dapper.mods.Lorenz96 import Tplot, dstep_dx, x0
from dapper.mods.Lorenz96s import (em_ensemble_step, rk_ensemble_step,
                                   truth_step)

####################
# Main definitions #
####################

# -----------------#
# Exeriment sizes  #
# -----------------#

# Grudzien 2020 uses the below chronology with KObs=25000, BurnIn=5000,
# this is a shorter demonstration of the same model uncertainty / filter divergence
# relationships

# this configuration is shown to keep numerical discretization
# error bounded by approximately 10^-3 in expected value when using tmodel with the
# Runge-Kutta scheme but with expected error on order 10^-2 for Euler-Maruyama
ttruth_high_precision = modelling.Chronology(dt=0.001, dtObs=0.1,
                                             T=30, Tplot=Tplot, BurnIn=10)
tmodel_high_precision = modelling.Chronology(dt=0.001, dtObs=0.1,
                                             T=30, Tplot=Tplot, BurnIn=10)

# this configuration keeps the Euler-Maruyama discretization error on the order 10^-3
# and in this case the difference in performance of each scheme is not as noticeable
ttruth_low_precision = modelling.Chronology(dt=0.005, dtObs=0.1,
                                            T=30, Tplot=Tplot, BurnIn=10)
tmodel_low_precision = modelling.Chronology(dt=0.01, dtObs=0.1,
                                            T=30, Tplot=Tplot, BurnIn=10)

# we define the following parameter grids for diffusion and model error
Diffusions = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
SigmasR = np.array([0.1, 0.25, 0.5, 0.75, 1.0])

# --------------------------------------------- #
# Package the dynamical and observation models  #
# --------------------------------------------- #

# we define a low-dimensional Lorenz96s sytem for easy simulation times
# this model has 4 unstable / neutral exponents, exhibiting chaos
Nx = 10

# define the initial condition
x0 = x0(Nx)

# we define the model configurations for the two ensemble runs and the truth twin
Dyn = {
    'M': Nx,
    'linear': dstep_dx,
    'Noise': 0.0
}

EMDyn = dict(Dyn, model=em_ensemble_step)
RKDyn = dict(Dyn, model=rk_ensemble_step)
TruthDyn = dict(Dyn, model=truth_step)

# ensemble initial condition is shared between the truthe and the EM and RK ensembles
X0 = modelling.GaussRV(mu=x0, C=0.001)

## define the obs indices and the obs function for generating observations
jj = np.arange(Nx)  # obs_inds
Obs = modelling.partial_Id_Obs(Nx, jj)

# Dummy value for the observation noise is set here, to
# be re-defined upon instantiation in simulations
Obs['noise'] = 1

##########
# README #
##########
#
# This study uses no multiplicative inflation / localization or other
# regularization instead using a large ensemble size in the perturbed
# observation EnKF as a simple estimator to study the asymptotic filtering
# statistics under different model scenarios.
#
# The purpose of the study in grudzien2020 was to explore the relationships between:
#
#  (i)   numerical discretization error in model twins;
#  (ii)  model uncertainty in perfect-random models;
#  (iii) filter divergence and / or bias in filtering forecast statistics;
#
# Numerical discretization error increases with dt, with the strong / weak order of
# convergence discussed in the refs.  Although the orders of convergence of the
# stochastic Runge-Kutta and the Euler-Maruyama model match, it is shown that
# the step size configuration above keeps the discretization error for the model and
# truth twins bounded by approximately 10^-3 in expectation.
#
# Model uncertainty increases with the diffusion, representing the "instantaneous"
# standard deviation of the model noise at any moment. Larger diffusion
# thus corresponds to a wider variance of the relizations of the diffeomorphsims
# that generate the model / truth twin between observation times.
#
# It is demonstrated by grudzien2020 that the model error due to discretization of
# the SDE equations of motion is most detrimental to the filtering cycle when model
# uncertainty is low and observation precision is high.  In other
# configurations, such as those with high model uncertainty, the differences between
# ensembles with low discretization error (those using the Runge-Kutta scheme) and
# high discretization error (those using the Euler-Maruyama scheme) tend to be relaxed.
