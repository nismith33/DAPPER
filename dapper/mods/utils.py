"""Utilities to help define hidden Markov models."""

import functools
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from dapper.tools.rounding import is_whole


def rel2mods(path):
    mods = Path(__file__).parent
    path = Path(path).relative_to(mods).with_suffix("")
    return str(path)


# https://stackoverflow.com/q/22797580
# https://stackoverflow.com/q/10875442
class NamedFunc():
    "Provides custom repr for functions."

    def __init__(self, func, name):
        self._function = func
        self._new_name = name
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    def __str__(self):
        return self._new_name

    def __repr__(self):
        return str(self) + "\n" + repr(self._function)


def name_func(name):
    """Decorator for creating NamedFunc."""
    def namer(func):
        return NamedFunc(func, name)
    return namer


def ens_compatible(func):
    """Decorate to transpose before and after, i.e. `func(input.T).T`.

    This is helpful to make functions compatible with both 1d and 2d ndarrays.

    Note
    ----
    this is not the_way™ -- other tricks are sometimes more practical.

    Examples
    --------
    `dapper.mods.Lorenz63.dxdt`, `dapper.mods.DoublePendulum.dxdt`

    See Also
    --------
    np.atleast_2d, np.squeeze, np.vectorize
    """
    @functools.wraps(func)
    def wrapr(x, *args, **kwargs):
        return np.asarray(func(x.T, *args, **kwargs)).T
    return wrapr


def Id_op():
    """Id operator (named). Returns first argument."""
    return NamedFunc(lambda *args: args[0], "Id operator")


def linear_model_setup(ModelMatrix, dt0):
    r"""Make the Dyn/Obs field of a HMM representing a linear model.

    Let *M* be the model matrix. Then
    .. math::

      x(t+dt) = M^{dt/dt0} x(t),

    i.e.

    .. math::

      \frac{dx}{dt} = \frac{\log(M)}{dt0} x(t).

    In typical use, `dt0==dt` (where `dt` is defined by the chronology).
    Anyways, `dt` must be an integer multiple of `dt0`.

    Returns
    -------
    A `dict` with keys: 'M', 'model', 'linear'.
    """
    Mat = np.asarray(ModelMatrix)  # does not support sparse and matrix-class

    # Compute and cache ModelMatrix^(dt/dt0).
    @functools.lru_cache(maxsize=1)
    def MatPow(dt):
        assert is_whole(dt/dt0), "Mat. exponentiation unique only for integer powers."
        return sla.fractional_matrix_power(Mat, int(round(dt/dt0)))

    @ens_compatible
    def mod(x, t, dt): return MatPow(dt) @ x
    def lin(x, t, dt): return MatPow(dt)

    Dyn = {
        'M': len(Mat),
        'model': mod,
        'linear': lin,
    }
    return Dyn


def direct_obs_matrix(Nx, obs_inds):
    """Generate matrix that "picks" state elements `obs_inds` out of `range(Nx)`.

    Parameters
    ----------
    Nx: int
        Length of state vector
    obs_inds: ndarray
        Indices of elements of the state vector that are (directly) observed.

    Returns
    -------
    H: ndarray
        The observation matrix for direct partial observations.
    """
    Ny = len(obs_inds)
    H = np.zeros((Ny, Nx))
    H[range(Ny), obs_inds] = 1
    H = [h for h in H]

    # One-liner:
    # H = np.array([[i==j for i in range(M)] for j in jj],float)

    return H


def partial_Id_Obs(Nx, obs_inds):
    """Specify identity observations of a subset of obs. indices.

    It is not a function of time.

    Parameters
    ----------
    Nx: int
        Length of state vector
    obs_inds: ndarray
        The observed indices.

    Returns
    -------
    Obs: dict
        Observation operator including size of the observation space,
        observation operator/model and tangent linear observation operator
    """
    Ny = len(obs_inds)
    H = direct_obs_matrix(Nx, obs_inds)

    @name_func(f"Direct obs. at {obs_inds}")
    @ens_compatible
    def model(x, t): return x[obs_inds]
    @name_func(f"Constant matrix\n{H}")
    def linear(x, t): return H
    Obs = {
        'M': Ny,
        'model': model,
        'linear': linear,
    }
    return Obs

def var_Id_Obs(Nx) :
    """Specify identity observations of a subset of obs. indices.
    Size subset may vary in time.

    It is not a function of time.

    Parameters
    ----------
    Nx: int
        Length of state vector

    Returns
    -------
    Obs: dict
        Observation operator including size of the observation space,
        observation operator/model and tangent linear observation operator
    """
    
    #Number of observations as function of time. 
    Ny=lambda t: int(np.mod(np.floor(t),3)+1)

    @name_func(f"Direct time varying partial_id ")
    @ens_compatible
    def model(x, t): 
        obs_inds=np.arange(Ny(t))
        return x[obs_inds]
    @name_func(f"Time varying partial_id")
    def linear(x, t):        
        obs_inds=np.arange(Ny(t))
        H = direct_obs_matrix(Nx, obs_inds)
        return H
    Obs = {'M': 0, #Initial value.
           'model': model,
           'linear': linear,
           }
    return Obs

def model_Obs(model_obs, database):
    
    #Number of observations
    Ny = lambda t: sum(np.array(database['time']==t))
    
    @name_func(f"Point observations.")
    @ens_compatible
    def model(x, t):
        s = np.array(np.shape(x))
        if np.sum(s>1)==1:  
            return model_obs(database, x, t)
        elif np.sum(s>1)==2:
            Eo=[]
            for x1 in x.T:
                Eo.append(model_obs(database, x1, t))
            Eo=np.array(Eo)
            return Eo.T
        else:
            raise TypeError("State must be array of dim 1/2.")
        
    @name_func(f"H for point observations")
    def linear(x, t): 
        msg = "Observation operator H not implemented for point_obs."
        raise NotImplementedError(msg)
    
    Obs = {'M': 0, 'model': model, 'linear': linear}
    return Obs

def Id_Obs(Nx):
    """Specify identity observations of entire state.

    It is not a function of time.

    Parameters
    ----------
    Nx: int
        Length of state vector

    Returns
    -------
    Obs: dict
        Observation operator including size of the observation space,
        observation operator/model and tangent linear observation operator
    """
    return partial_Id_Obs(Nx, np.arange(Nx))


def linspace_int(Nx, Ny, periodic=True):
    """Provide a range of `Ny` equispaced integers between `0` and `Nx-1`.

    Parameters
    ----------
    Nx: int
        Range of integers
    Ny: int
        Number of integers
    periodic: bool, optional
        Whether the vector is periodic.
        Determines if `Nx == 0`.
        Default: True

    Returns
    -------
    integers: ndarray
        The list of integers.

    Examples
    --------
    >>> linspace_int(10, 10)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> linspace_int(10, 4)
    array([0, 2, 5, 7])
    >>> linspace_int(10, 5)
    array([0, 2, 4, 6, 8])
    """
    if periodic:
        jj = np.linspace(0, Nx, Ny+1)[:-1]
    else:
        jj = np.linspace(0, Nx-1, Ny)
    jj = jj.astype(int)
    return jj
