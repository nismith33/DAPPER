"""Localization tools, including distance and tapering comps.

A good introduction to localization:
Sakov (2011), Computational Geosciences:
'Relation between two common localisation methods for the EnKF'.
"""

# NB: Why is the 'order' argument not supported by this module? Because:
#  1) Assuming only order (orientation) 'C' simplifies the module's code.
#  2) It's not necessary, because the module only communicates to *exterior* via indices
#     [of what assumes to be X.flatten(order='C')], and not coordinates!
#     Thus, the only adaptation necessary if the order is 'F' is to reverse
#     the shape parameter passed to these functions (example: mods/QG/sakov2008).


import numpy as np
from abc import ABC, abstractmethod

def pairwise_distances(A, B=None, domain=None):
    """Euclidian distance (not squared) between pts. in `A` and `B`.

    Parameters
    ----------
    A: array of shape `(nPoints, nDims)`.
        A collection of points.

    B:
        Same as `A`, but `nPoints` can differ.

    domain: tuple
        Assume the domain is a **periodic** hyper-rectangle whose
        edges along dimension `i` span from 0 to `domain[i]`.
        NB: Behaviour not defined if `any(A.max(0) > domain)`, and likewise for `B`.

    Returns
    -------
    Array of of shape `(nPointsA, nPointsB)`.

    Examples
    --------
    >>> A = [[0, 0], [0, 1], [1, 0], [1, 1]]
    >>> with np.printoptions(precision=2):
    ...     print(pairwise_distances(A))
    [[0.   1.   1.   1.41]
     [1.   0.   1.41 1.  ]
     [1.   1.41 0.   1.  ]
     [1.41 1.   1.   0.  ]]

    The function matches `pdist(..., metric='euclidean')`, but is faster:
    >>> from scipy.spatial.distance import pdist, squareform
    >>> (pairwise_distances(A) == squareform(pdist(A))).all()
    True

    As opposed to `pdist`, it also allows comparing `A` to a different set of points,
    `B`, without the augmentation/block tricks needed for pdist.

    >>> A = np.arange(4)[:, None]
    >>> pairwise_distances(A, [[2]]).T
    array([[2., 1., 0., 1.]])

    Illustration of periodicity:
    >>> pairwise_distances(A, domain=(4, ))
    array([[0., 1., 2., 1.],
           [1., 0., 1., 2.],
           [2., 1., 0., 1.],
           [1., 2., 1., 0.]])

    NB: If an input array is 1-dim, it is seen as a single point.
    >>> pairwise_distances(np.arange(4))
    array([[0.]])
    """
    if B is None:
        B = A

    # Prep
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    mA, nA = A.shape
    mB, nB = B.shape
    assert nA == nB, "The last axis of A and B must have equal length."

    # Diff
    d = A[:, None] - B  # shape: (mA, mB, nDims)

    # Make periodic
    if domain:
        domain = np.reshape(domain, (1, 1, -1))  # for broadcasting
        d = abs(d)
        d = np.minimum(d, domain-d)

    distances = np.sqrt((d * d).sum(axis=-1))  # == sla.norm(d, axis=-1)

    return distances.reshape(mA, mB)

def dist2coeff(dists, radius, tag=None):
    """Compute tapering coefficients corresponding to a distances.

    NB: The radius is internally adjusted such that, independently of 'tag',
    `coeff==np.exp(-0.5)` when `distance==radius`.

    This is largely based on Sakov's enkf-matlab code. Two bugs have here been fixed:
    - The constants were slightly wrong, as noted in comments below.
    - It forgot to take sqrt() of coeffs when applying them through 'local analysis'.
    """
    coeffs = np.zeros(dists.shape)

    if tag is None:
        tag = 'GC'

    if tag == 'Gauss':
        R = radius
        coeffs = np.exp(-0.5 * (dists/R)**2)
    elif tag == 'Exp':
        R = radius
        coeffs = np.exp(-0.5 * (dists/R)**3)
    elif tag == 'Cubic':
        R            = radius * 1.87  # Sakov: 1.8676
        inds         = dists <= R
        coeffs[inds] = (1 - (dists[inds] / R) ** 3) ** 3
    elif tag == 'Quadro':
        R            = radius * 1.64  # Sakov: 1.7080
        inds         = dists <= R
        coeffs[inds] = (1 - (dists[inds] / R) ** 4) ** 4
    elif tag == 'GC':  # eqn 4.10 of Gaspari-Cohn'99, or eqn 25 of Sakov2011relation
        R = radius * 1.82  # =np.sqrt(10/3). Sakov: 1.7386
        # 1st segment
        ind1         = dists <= R
        r2           = (dists[ind1] / R) ** 2
        r3           = (dists[ind1] / R) ** 3
        coeffs[ind1] = 1 + r2 * (- r3 / 4 + r2 / 2) + r3 * (5 / 8) - r2 * (5 / 3)
        # 2nd segment
        ind2         = np.logical_and(R < dists, dists <= 2*R)
        r1           = (dists[ind2] / R)
        r2           = (dists[ind2] / R) ** 2
        r3           = (dists[ind2] / R) ** 3
        coeffs[ind2] = r2 * (r3 / 12 - r2 / 2) + r3 * (5 / 8) \
            + r2 * (5 / 3) - r1 * 5 + 4 - (2 / 3) / r1
    elif tag == 'Step':
        R            = radius
        inds         = dists <= R
        coeffs[inds] = 1
    else:
        raise KeyError('No such coeff function.')

    return coeffs

def inds_and_coeffs(dists, radius, cutoff=1e-3, tag=None):
    """Compute indices and coefficients of localization.

    - inds   : the indices of pts that are "close to" centre.
    - coeffs : the corresponding tapering coefficients.
    """
    coeffs = dist2coeff(dists, radius, tag)

    # Truncate using cut-off
    inds   = np.arange(len(dists))[coeffs > cutoff]
    coeffs = coeffs[inds]

    return inds, coeffs

def localization_setup(y2x_distances, batches):

    def localization_now(radius, direction, t, tag=None):
        """Provide localization setup for time t."""
        y2x = y2x_distances(t)

        if direction == 'x2y':
            def obs_taperer(batch):
                # Don't use `batch = batches[iBatch]`
                # (with iBatch as this function's input).
                # This would slow down multiproc.,
                # coz batches gets copied to each process.
                x2y = y2x.T
                dists = x2y[batch].mean(axis=0)
                return inds_and_coeffs(dists, radius, tag=tag)
            return batches, obs_taperer

        elif direction == 'y2x':
            def state_taperer(obs_idx):
                return inds_and_coeffs(y2x[obs_idx], radius, tag=tag)
            return state_taperer

    return localization_now

def no_localization(Nx, Ny):

    def obs_taperer(batch):
        return np.arange(Ny), np.ones(Ny)

    def state_taperer(obs_idx):
        return np.arange(Nx), np.ones(Nx)

    def localization_now(radius, direction, t, tag=None):
        """Returns all of the indices, with all tapering coeffs. set to 1.

        Used to validate local DA methods, eg. `LETKF<==>EnKF('Sqrt')`.
        """

        if direction == 'x2y':
            return [np.arange(Nx)], obs_taperer
        elif direction == 'y2x': 
            return state_taperer
        elif direction == 'x2x':
            return state_taperer
        else:
            msg = "{} is not a valid value for direction."
            raise ValueError(msg.format(direction))

    return localization_now

def rectangular_partitioning(shape, steps, do_ind=True):
    """N-D rectangular batch generation.

    Parameters
    ----------
    shape: (len(grid[dim]) for dim in range(ndim))
    steps: (step_len[dim]  for dim in range(ndim))

    Returns
    -------
    A list of batches,
    where each element (batch) is a list of indices.

    Example
    -------
    >>> shape   = [4, 13]
    ... batches = rectangular_partitioning(shape, [2, 4], do_ind=False)
    ... nB      = len(batches)
    ... values  = np.random.choice(np.arange(nB), nB, 0)
    ... Z       = np.zeros(shape)
    ... for ib, b in enumerate(batches):
    ...     Z[tuple(b)] = values[ib]
    ... plt.imshow(Z)  # doctest: +SKIP
    """
    import itertools
    assert len(shape) == len(steps)
    # ndim = len(steps)

    # An ndim list of (average) local grid lengths:
    nLocs = [round(n/d) for n, d in zip(shape, steps)]
    # An ndim list of (marginal) grid partitions
    # [array_split() handles non-divisibility]:
    edge_partitions = [np.array_split(np.arange(n), nLoc)
                       for n, nLoc in zip(shape, nLocs)]

    batches = []
    for batch_edges in itertools.product(*edge_partitions):
        # The 'indexing' argument below is actually inconsequential:
        # it merely changes batch's internal ordering.
        batch_rect  = np.meshgrid(*batch_edges, indexing='ij')
        coords      = [ii.flatten() for ii in batch_rect]
        batches    += [coords]

    if do_ind:
        def sub2ind(sub):
            return np.ravel_multi_index(sub, shape)
        batches = [sub2ind(b) for b in batches]

    return batches

# NB: Don't try to put the time-dependence of obs_inds inside obs_taperer().
# That would require calling ind2sub len(batches) times per analysis,
# and the result cannot be easily cached, because of multiprocessing.
def safe_eval(fun, t):
    try:
        return fun(t)
    except TypeError:
        return fun

def nd_Id_localization(shape,
                       batch_shape=None,
                       obs_inds=None,
                       periodic=True):
    """Localize Id (direct) point obs of an N-D, homogeneous, rectangular domain."""
    M = np.prod(shape)

    if batch_shape is None:
        batch_shape = (1,)*len(shape)
    if obs_inds is None:
        obs_inds = np.arange(M)

    def ind2sub(ind):
        return np.asarray(np.unravel_index(ind, shape)).T

    batches = rectangular_partitioning(shape, batch_shape)

    state_coord = ind2sub(np.arange(M))

    def y2x_distances(t):
        obs_coord = ind2sub(safe_eval(obs_inds, t))
        return pairwise_distances(obs_coord, state_coord, shape if periodic else None)

    return localization_setup(y2x_distances, batches)

class Localizer:
    """
    Class that carries out localisation. Should be provided to ens_method object. 
    This class follows strategy pattern. 
    
    Attributes
    ----------
    batcher : Batcher object 
        Object responsible for batching state vector. 
    taperer : Taper object 
        Object responsible for producing a function that calculates
        the tapering between one observation/state vector index 
        and the state/observation vector. 
    coorder : Coorder object 
        Object responsible for assigning space coordinates to the different
        indices in the state vector. 
    cutoff : float 
        Cut off radius. If distance between state vector/observation vector 
        radius is larger than correlation is assumed to be zero. 
        
    """
    
    def __init__(self, batcher, taperer, coorder, cutoff=-1e9):
        """ Class constructor. 
        
        Parameters
        ----------
        batcher : Batcher object 
            Object responsible for batching state vector. 
        taperer : Taper object 
            Object responsible for producing a function that calculates
            the tapering between one observation/state vector index 
            and the state/observation vector. 
        coorder : Coorder object 
            Object responsible for assigning space coordinates to the different
            indices in the state vector. 
        cutoff : float 
            Taper values below this cut off are assumed to be zero. 
            
        """
        self.batcher = batcher 
        self.taperer = taperer 
        self.coorder = coorder
        self.cutoff  = cutoff
    
    def __call__(self, radius, direction, time, tag):      
        """
        Function  called by ens_method. 
        
        Parameters
        ----------
        radius : float 
            Localization radius. 
        direction : x2y | y2x
            Type of localization used. 
        time : float 
            Timestep for localization. 
        tag : str 
            Indicate which localization function to use. 
        
        """
          
        #Update state batches 
        self.dyn_coords, self.batches = self.batcher(time)
        
        #Update taperer
        self.taper = self.taperer(radius, time, tag)
            
        #Update coordinates observations 
        self.obs_coords = self.coorder(time)
        
        #Distances between observations 
        if direction == 'x2y':
            return self.localize_x2y()
        elif direction == 'y2x':
            return self.localize_y2x()
        elif direction == 'x2x':
            return self.localize_x2x()
        else:
            msg = "{:s} unknown option for direction."
            raise ValueError(msg.format(direction))
    
    def localize_y2x(self):
        """Returns taper for state vector indices given observation vector index."""
        
        def state_taperer(obs_idx):
            """State taperer. 
            
            Parameters 
            ----------
            obs_idx : int 
                Index in observation vector used to generate taper. 
                
            Returns
            -------
            inds : array of int 
                State vector indices for which taper>0
            coeffs : array of float 
                Non-zero taper weights for state vector indices inds. 
                
            """ 
            coeffs = self.taper(self.obs_coords[obs_idx], self.dyn_coords)
            inds = np.where(coeffs >= self.cutoff)[0]
            return inds, coeffs[inds]
        
        return state_taperer
    
    def localize_x2y(self):
        """Returns taper for observation vector indices given state vector index."""
        
        def obs_taperer(batch):
            """Observation taperer. 
            
            Parameters 
            ----------
            batch : array of int
                State vector indices of batch used to generate taper. 
                
            Returns
            -------
            batches : list of int arrays
                List with state vector indices of different batches. 
            coeffs : array of float 
                Non-zero taper weights for observations vector indices inds. 
                
            """ 
            coeffs = self.taper(self.dyn_coords[batch], self.obs_coords)
            inds = np.where(coeffs >= self.cutoff)[0]
            return inds, coeffs[inds]
        
        return self.batches, obs_taperer
    
    def localize_x2x(self):
        """Returns taper for observation vector indices given state vector index."""
        
        def state_taperer(batch):
            """State taperer. 
            
            Parameters 
            ----------
            batch : array of int
                State vector index used to generate taperer. 
                
            Returns
            -------
            batches : list of int arrays
                List with state vector indices of different batches. 
            coeffs : array of float 
                Non-zero taper weights for
                
            """ 
            coeffs = self.taper(self.dyn_coords[batch], self.dyn_coords)
            inds = np.where(coeffs >= self.cutoff)[0]
            return inds, coeffs[inds]
        
        return state_taperer
    
    def update_ensemble(self, E):
        """ Update settings in taperer from the ensemble. """
        self.taperer.update_ensemble(E)
    
class Batcher(ABC):
    """ Class reponsible for dividing the state vector into batches. """
    
    def __call__(self, time):
        """ Batch state vector. 
        
        Parameters 
        ----------
        time : float 
            Time at which batching occurs. 
        
        Returns
        -------
        coords : array of float
            Spatial coordinates associated with all indices in state vector. 
        batches : list of int arrays
            List with state vector indices for each batch. 
            
        """

        if self.time != time:
            self.batches = self.batch(time)
            self.time = time 
            
        return self.coords, self.batches 
    
    @abstractmethod 
    def batch(self, time):
        """ Function carrying out the actual batching.
        
        Parameters 
        ----------
        time : float 
            Time at which batching is supposed to take place. 
        
        Returns
        -------
        List with batches as arrays of state vector indices. 
        
        """
        pass        
    
class RectrangularPartitioning(Batcher):
    """ Batch multidimensional grid into box-like batches of nearly equal size."""
    
    def __init__(self, shape, steps, do_int=True):
        """ Class constructor.
        
        shape : array of int>0
            Shape of state grid. 
        steps : array int>=0
            Number of batches in each dimension of grid. 
        
        """        
        self.time = None
        self.shape = np.array(shape, dtype=int) 
        self.steps = steps 
        self.do_int = do_int 
        
        self.coords = np.unravel_index(np.arange(self.shape.prod()), self.shape)
        self.coords = np.array(self.coords).T
    
    def batch(self, time):
        return rectangular_partitioning(self.shape, self.steps, do_ind=self.do_ind)
    
class SingleBatcher(Batcher):
    """ Treat each index in state vector as its own batch. Used by LETKF. """
    
    def __init__(self, coords):
        """ Class constructor. 
        
        Parameters
        ----------
        coords : array of float 
            Spatial coordinates for each state vector index. 
        
        """
        self.time = None 
        
        if np.ndim(coords)==1:
            self.coords = coords[...,None]
        else:
            self.coords = coords
        
    def batch(self, time):
        inds = np.arange(0, np.size(self.coords,0))
        return np.reshape(inds, (-1,1))
    
class LegendreBatcher(SingleBatcher):
    """ Express coordinate as spatial coordinate grid cell centre + order 
    Legendre polynomial.
    """
    
    def __init__(self, coords, order):
        self.time = None
        
        if np.ndim(coords)==1:
            coords = coords[...,None]
        self.coords = self.coords_orders(coords, order)
        
    def coords_orders(self, coords, order):
        """
        Add order of Legendre polynomial to spatial coordinate. 
        
        Parameters
        ----------
        coords : array of float 
            Spatial coordinates.
        order : int>=0
            Highest order Legendre polynomial. 
            
        Returns
        -------
        
        """
        orders = np.mod(np.arange(0, np.size(coords,0)), order+1)
        orders = orders[...,None]
        return np.concatenate((coords, orders), axis=1)
    
    def estimate_radius(self, E):
        """
        Estimate the optimal localisation radius. 

        Parameters
        ----------
        E : array of float
            Ensemble of model state.

        Returns
        -------
        None.

        """
        #Number of ensemble members
        N = np.size(E, 0)
        #Number of orders 
        orders = np.unique(self.coords[:,-1])
    
class BinBatcher(Batcher):
    """ Create batches by multidimensional binning. """
    
    def __init__(self, coords, bins):
        """Class constructor. 
        
        Parameters
        ----------
        coords : array of float 
            Spatial coordinates for each state vector index. 
        bins : list of float arrays
            List with each element of the list containing coordinates
            bin edges for that dimension. 
            
        """
        
        self.time = None 
        if np.ndim(coords)==1:
            self.coords = coords[...,None]
        else:
            self.coords = coords 
            
        self.bins = np.array(bins)
        
    def batch(self, time):
        from scipy.stats import binned_statistic_dd as binner
        
        _, _, bins = binner(self.coords, values=np.empty([]), statistic='count',
                            bins=self.bins, expand_binnumbers=False)
        
        bins = np.array(bins, dtype=int)
        inds = np.arange(0, np.size(self.coords,0))
        
        batches = []
        for bin in np.unique(bins):
            batches.append(inds[bins==bin])
        
        return batches
        
class Coorder(ABC):
    """
    Class producing spatial coordinates associated with indices of state or
    observation vector. 
    
    Attributes
    ----------
    coords : array of float 
        Coordinates for observation vector. 
    time : float 
        Time for which coordinates were calculated. 
        
    """
    
    def __call__(self, time):
        """ 
        Class for setting spatial coordinates. 
        
        Parameters 
        ----------
        time : float 
            Time for which coordinates are calculated. 
        
        Returns
        -------
        coords : array of floats
            Spatial coordinates. 
        
        """
        
        if self.time != time:
            self.coords = self.update_coords(time)
            self.time = time
            
        return self.coords
            
    @abstractmethod 
    def update_coords(self, time):
        """
        Function calculating the spatial coordinates. 
        
        Parameters 
        ----------
        time : float 
            Time for which coordinates are calculated. 
        
        Returns
        -------
        coords : array of floats
            Spatial coordinates. 
        """
        pass
    
class FunctionCoorder(Coorder):
    """ Use Python function to calculate the spatial coordinates."""
    
    def __init__(self, function):
        """ Class constructor. 
        
        Parameters
        ----------
        functionn : Python function 
            Function that returns coordinates given time. 
        
        """
        self.time = None
        self.function = function
    
    def update_coords(self, time):
        coords = self.function(time)  
        if np.ndim(coords)==1:
            coords = coords[...,None]
            
        return coords
    
def taper_factory(radius, tag=None):
    """
    Create taper function. 
    
    Parameters
    ----------
    radius : float | array of floats
        Localization radius. If array, different scaling is used
        for different dimensions. 
    tag : str 
        Type of function to use. 
        
    Returns
    -------
    Function that calculates tapering coefficients between two points. 
    
    """
        
    def distance(a,b,radius):
        """ Calculate scaled distance between coordinates.
        
        Parameters
        ----------
        a : array of float
            Coordinate of one point. 
        b : array of float 
            Coordinates of other points. 
        radius : float | array of floats 
            Scaling radius. If vector different dimensions are 
            scaled with different length scales. 
        
        Returns
        -------
        Distance scaled by radius. 
        
        """
        
        #Check input
        if np.ndim(a)!=1:
            raise ValueError("Argument a must be 1D array.")
        if np.ndim(radius)==0:
            radius = np.ones_like(a) * radius
        if np.any(radius<=0.):
            raise ValueError("Radii must be strictly positive.")
        
        #Calculate scaled difference
        if np.ndim(b)==1:
            return np.linalg.norm((a-b)/radius, axis=0)
        elif np.ndim(b)==2:
            return np.linalg.norm((a[None,...]-b)/radius[None,...], axis=1)
        else:
            raise ValueError("Argument b must be 1D or 2D array.")
    
    #Default option. 
    if tag is None or tag==None:
        tag = 'GC'
     
    if tag == 'Gauss':
        def taper(a,b):
            return np.exp(-0.5 * distance(a,b,radius)**2)
    elif tag == 'Exp':
        def taper(a,b):
            return np.exp(-0.5 * distance(a,b,radius)**3)
    elif tag == 'Cubic':
        R = radius * 1.87  # Sakov: 1.8676
        def taper(a,b):
            d = distance(a, b, R)
            return np.where(d<=1, (1-d**3)**3, 0.)
    elif tag == 'Quadro':
        R = radius * 1.64  # Sakov: 1.7080
        def taper(a,b):
            d = distance(a, b, R)
            return np.where(d<=1, (1-d**4)**4, 0.)
    elif tag == 'GC':  # eqn 4.10 of Gaspari-Cohn'99, or eqn 25 of Sakov2011relation
        R = radius * 1.82  # =np.sqrt(10/3). Sakov: 1.7386
        
        def s1(r1):
            r2, r3 = r1**2, r1**3
            return 1 + r2 * (- r3 / 4 + r2 / 2) + r3 * (5 / 8) - r2 * (5 / 3)
        
        def s2(r1):
            r2, r3 = r1**2, r1**3
            return r2 * (r3 / 12 - r2 / 2) + r3 * (5 / 8) + r2 * (5 / 3) - r1 * 5 + 4 - (2 / 3) / r1
            
        def taper(a,b):
            d = distance(a, b, radius)
            # 1st segment
            w = np.where(d<=1, s1(d), 0.)
            w = np.where(np.logical_and(d>1,d<=2), s2(d), w)
            return w
    elif tag == 'Step':
        
        def taper(a,b):
            d = distance(a, b, radius)
            return np.where(d<=1, 1., 0.)
    else:
        raise KeyError('{} is not a valid taper function.'.format(tag))  
    
    #Output of taper_factory
    return taper
    
class Taperer(ABC):
    """
    Class producing tapering functions.
    
    Attributes
    ----------
    radius : float | array of float 
        Localization radii used. 
    time : float 
        Time for which taper function is constructed. 
    tag : str 
        Localization method used.
    
    """
    
    def __call__(self, radius, time, tag=None):
        """
        Construct taper function.
        
        Parameters
        ----------
        radius : float | array of float | list of array of floats
            Localization radii used. 
        time : float 
            Time for which taper function is constructed. 
        tag : str | list of str
            Localization method used.
            
        Returns
        -------
        Function calculating tapering coefficients between points.  
            
        """
        radius = np.array(radius, dtype=float)
        tag = np.array(tag)
       
        must_update = False 
        if self.radius is None or np.any(radius != self.radius):
            must_update = True 
        if self.time is None or time != self.time:
            must_update = True
        if self.tag is None or np.any(tag != self.tag):
            must_update = True 
            
        if must_update:
            self.radius, self.time, self.tag = radius, time, tag 
            self.taper = self.update_taper(radius, time, tag)
      
        return self.taper
    
    def compress_coords(self, coords):
        """
        Compress a set of points into one using averaging. 
        
        Parameters
        ----------
        coords : array of float 
            Coordinates of points to be compressed. 
            
        Returns 
        -------
        Compressed point. 
        
        """
        if np.ndim(coords)==1:
            return coords
        if np.ndim(coords)==2:
            return np.mean(coords, axis=0)
        else:
            msg = "Invalid dimension of coords."
            raise TypeError(msg)
    
    @abstractmethod 
    def update_taper(self, radius, time, tag):
        """ Method that actually constructs new tapering function. """
        pass 
    
    def update_ensemble(self, E):
        """ If using an adaptive localizer, update localizer this point. """
        pass
    
class DistanceTaperer(Taperer):
    """
    Object constructing tapering function with values solely based 
    on scaled L2-distance between coordinates.
    """
    
    def __init__(self):
        """ Class constructor. """
        self.time, self.radius, self.tag = None, None, None
    
    def update_taper(self, radius, time, tag=None):
        taper = taper_factory(radius, tag)
        return lambda a,b : taper(self.compress_coords(a), b)
     
class LegendreTaperer(Taperer):   
    """
    Class that applies different tapering to coefficient associated with 
    different order polynomials. 
    
    Attributes 
    ----------
    order : int>=0
        Highest order Legendre polynomial.
        
    """
    
    def __init__(self, order):
        """ Class constructor. 
        
        Parameters
        ----------
        order : int>=0
            Highest order Legendre polynomial.
        """
        self.time, self.radius, self.tag = None, None, None
        self.order = order
        
    def update_taper(self, radius, time, tag):
        
        #Check shape radius
        if np.ndim(radius)==0:
            radius = np.ones((1, self.order+1)) * radius
        elif np.ndim(radius)==1:
            radius = np.array([radius for _ in range(self.order+1)])
        elif np.ndim(radius)==2:
            pass 
        else: 
            raise ValueError("Radius must be 0D,1D or 2D array.")
        
        #Check shape tag
        if tag is None:
            tag = [None for _ in range(self.order+1)]
        elif np.ndim(tag)==0:
            tag = np.array([tag for _ in range(self.order+1)])
        elif np.ndim(tag)==1:
            pass 
        else:
            raise ValueError("Tag must be 0D or 1D array.")
        
        def taper(a,b):
            #Determine polynomial order state vector
            order = np.unique(np.array(a[:,-1],dtype=int))
            if len(order)==1:
                order = int(order[0])
                a = self.compress_coords(a[:,:-1])
            else:
                msg = "Each batch may only contain 1 Legendre order."
                raise ValueError(msg)
            
            #Fetch taper function for this order
            taper_template = taper_factory(radius[order], tag[order])
            
            if np.size(b,-1)==np.size(a,-1):           
                #Evaluate function using only spatial part coordinates
                tapering = taper_template(a, b)
            elif np.size(b,-1)==np.size(a,-1)+1:
                orders = np.array(b[:,-1], dtype=int)
                tapering = taper_template(a, b[:,:-1])
                tapering[orders!=order] = 0
            else:
                print(np.shape(a), np.shape(b))
                raise ValueError("Dimension mismatch between a and b.")

            return tapering
                
            
        return taper
    
class OptimalTaperer(Taperer):
    """
    Calculate optimal taper based on  Menetrier et al. (2015), 10.1175/MWR-D-14-00157.1

    Parameters
    ----------
    E0, E1 : numpy 2D array
        Ensembles used to calculate the (cross-)covariance. E[n]
        must contain the nth ensemble members.
    dist : int
        The maximum distance for which localisation is calculated. 

    Returns
    -------
    L : numpy array of float 
        Localization taper for positions.
    var2 : numpy array of float
        Expectation of var[x]*var[x+k] 
    cov2 : numpy array of float 
        Expectation of cov[x,x+k]cov[x,x+k]

    """
    
    def __init__(self, order, period=None):
        self.time, self.radius, self.tag = None, None, None
        self.order = order
        self.period = period
        self.filter = self.filter_factory('identity')
        
    def update_ensemble(self, E):
        from matplotlib import pyplot as plt
        #Init storage 
        self.L = []
        
        #Separate by order polynomial
        E = np.reshape(E, (np.size(E,0), -1, self.order+1))
        
        #Calculate localization for each pair of orders
        self.L = np.zeros((self.order+1, self.order+1, np.size(E,1)))
        for i0 in range(self.order+1):
            E0 = E[:,:,i0]
            for i1 in range(self.order+1):
                E1 = E[:,:,i1]
                L1, _, _ = self.calculate_optimal_localization(E0, E1)
                self.L[i0,i1,:]=self.filter(L1)
        
    def filter_factory(self, filter_type):
        from scipy.signal import savgol_filter as savgol
        
        mirror = lambda x: np.append(np.flip(x[1:]),x)
        unmirror = lambda x: .5*x[int(len(x)/2)+1:] + .5*np.flip(x[:int(len(x)/2)])
        
        if filter_type == 'identity':
            def filter(y):
                return y
        elif filter_type =='loess':
            def filter(y):
                for _ in range(5):
                    y = savgol(y, window_length=5, polyorder=2, mode='wrap')
                return y
        elif filter_type == 'positive_definite':
            def filter(y):
                #y[1:] should already by equal to np.flip(y[1:]) because of periodicity, but here it is
                #enforced to remove any numerical errors                               
                y[1:] = 0.5*y[1:] + 0.5*np.flip(y[1:])
                                
                Fy = np.fft.fft(y)
                if not np.all(np.isclose(np.imag(Fy), 0.0)):
                    raise ValueError("Imaginary Fourier coefficients in symmetric function.")
                
                #Remove imaginary part. 
                Fy = np.real(Fy)
                #Ensure negative values can be removed
                sump = np.sum(Fy[Fy>0])
                summ = -np.sum(Fy[Fy<0])    
                if summ>sump:
                    return np.zeros_like(y)
                            
                #Remove negative values 
                it = 0 
                while not np.isclose(summ, 0.0) and summ>0.0:
                    mask = Fy>0
                    n = np.sum(mask)
                    m = min(summ/n, np.min(Fy[mask]))
                    Fy[mask] -= m 
                    summ -= m * n
                    if it>1000:
                        print('Exciting filter ',it,summ)
                        break
                Fy[Fy<0] *= 0
                
                #Convert back
                y1 = np.fft.ifft(Fy)            
                return np.real(y1)
                
        else:
            raise ValueError("{} is not a valid filter.".format(filter_type))
            
        return filter 
 
     
    def update_taper(self, radius, time, tag=None):
        self.radius = radius
        M = np.size(self.L, 2)
        
        def taper(a,b):
            #Determine polynomial order state vector
            order = np.unique(np.array(a[:,-1], dtype=int))
            if len(order)==1:
                order = int(order[0])
                a = self.compress_coords(a[:,:-1])
            else:
                msg = "Each batch may only contain 1 Legendre order."
                raise ValueError(msg)
            
            #Calculate weight
            orders = np.array(b[:,-1], dtype=int)
            if self.period is None:
                distances = np.linalg.norm(a[None,:]-b[:,:-1])
                r = np.arange(0, M) * self.radius
                distances = np.linalg.norm(distances, axis=1)
                
            else:
                distances = np.linalg.norm(a[None,:]-b[:,:-1], axis=1)
                distances = np.sin(np.pi*np.abs(distances)/self.period)
                
                r = np.arange(0,M) * self.radius 
                r = np.sin(np.pi*np.abs(r)/self.period)
                _, rind = np.unique(r, return_index=True)
            
            L = np.zeros_like(distances)
            for i1 in np.unique(orders):
                mask = orders==i1
                L[mask] = np.interp(distances[mask], xp=r[rind], fp=self.L[order,i1,rind])
                
            return L
        
        return taper        
    
    def lag_mean2(self, E0, E1, axes=None, max_lags=None):
        """
        Calculate cross-covariance <E0,E1>(lag) along specified axes 
        and for 0<=lag<=max_lags
        """
        from scipy.signal import fftconvolve as fft_convolve
        
        #If no axes specified convolution takes place along all axes. 
        if axes is None:
            axes = np.arange(0, np.ndim(E0))
        else:
            axes = np.array(axes, dtype=int)
        
        #If not specified convolution is calculated for lags 0 upto size of x0, x1
        #along axis of convolution. 
        if max_lags is None:
            max_lags = np.array([np.size(E0,d)-1 for d in axes], dtype=int)
        else:
            max_lags = np.array(max_lags, dtype=int)
            
        def calculate_lag_sum(axes,lim_lags,E0,E1,period):
            padding = [(0,0) for d in range(np.ndim(E0))]
            for ax, lag in zip(axes, lim_lags):
                if lag>=0:
                    padding[ax] = (0,lag)
                else:
                    padding[ax] = (-lag,0)
            
            #Expand 2nd ensemble 
            if period is None:
                raise NotImplemented
            else:
                scaling = np.prod([min(np.size(E0,d),np.size(E1,d)) for d in axes])
                E0 = np.pad(E0, padding, 'wrap')
            
            #Flip to turn lagging into convolution 
            for ax in axes:
                E0 = np.flip(E0, axis=ax)
        
            #Carry out convolution 
            S2 = fft_convolve(E0, E1, axes=axes, mode='valid')
        
            #Calculate normalization factor
            if period is None:
                raise NotImplemented
        
            #Scale 
            S2 = S2 / scaling 
            
            #convolution into lag 
            flip_iter = (ax for ax,lag in zip(axes,lim_lags) if lag>0)
            for ax in flip_iter:
                S2 = np.flip(S2,axis=ax)
                
            return S2

        #Combine positive and negative lags 
        S2  = 0.5*calculate_lag_sum(axes,  max_lags, E0, E1, self.period)
        S2 += 0.5*calculate_lag_sum(axes, -max_lags, E0, E1, self.period)
                        
        return S2
            
    def calculate_optimal_localization(self, E0, E1):
        """ 
        Calculate optimal taper based on Menetrier et al. (2015), 
        10.1175/MWR-D-14-00157.1

        Parameters
        ----------
        E0, E1 : numpy 2D array
            Ensembles used to calculate the (cross-)covariance. E[n]
            must contain the nth ensemble members.
        
        """
        #Size ensemble
        N = np.size(E0, axis=0)
        
        #Dimension ensemble
        D = np.ndim(E0)
        
        #Calculate ensemble anomalies
        E0  = E0 - np.mean(E0, axis=0, keepdims=True)
        E1  = E1 - np.mean(E1, axis=0, keepdims=True)
        E0 /= np.sqrt(N-1)
        E1 /= np.sqrt(N-1)
        
        #Calculate variance
        v0 = np.sum(E0**2, axis=0)
        v1 = np.sum(E1**2, axis=0)
        var2 = self.lag_mean2(v0, v1, axes=[0])
        
        #Instead of spatial average of sample covariance^2 we calculated
        #sample covariance after spatial averaging        
        cov2 = np.zeros_like(var2)
        for e0,e1 in zip(E0,E1):
            #element wise square of sample covariance matrix
            Ee0 = E0 * e0[None,...]
            Ee1 = E1 * e1[None,...]
            #spatial averaging
            cov01 = self.lag_mean2(Ee0, Ee1, axes=[1])
            #1st sample covariance
            cov01 = np.sum(cov01, axis=0)
            #sample covariance squard
            cov2 += cov01
                
        #Calculate localization function
        L = (N-1)/(N+1)/(N-2)*( N - 1 - var2/cov2)
                    
        return L, var2, cov2      
  
def smod(x1, x2, *args):  
    """ As mod but instead of returning remainder in [0,d) it is returned 
    in interval [-1/2d,1/2d).
    """    
    return np.mod(x1 + .5 * x2, x2, *args) - .5 * x2
    
