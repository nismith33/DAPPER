"""Variational DA methods (iEnKS, 4D-Var, etc)."""

from typing import Optional

import numpy as np
import scipy.linalg as sla
from copy import copy
from scipy.sparse.linalg import LinearOperator

from dapper.da_methods.ensemble import hyperprior_coeffs, post_process, zeta_a
from dapper.stats import center, inflate_ens, mean0
from dapper.tools.linalg import pad0, svd0, tinv
from dapper.tools.matrices import CovMat
from dapper.tools.progressbar import progbar

from . import da_method
from argparse import ArgumentTypeError

CALCULATE_STATS = True

@da_method
class var_method:
    """Declare default variational arguments."""

    Lag: int = 1
    nIter: int = 10
    wtol: float = 0


@var_method
class E3DVar:
    """
    Standard ensemble 3DVAR DA method without outer loops. Calculates DA correction as 
        B H^{T} R^{-1/2} ( \hat{B} + I )^{-1} d
    with H sample operator, R the observation error covariance (must be diagonal)
        B = \frac{1}{N-1} A A^{T} \dot L
        A = E - \overline{E}
        \hat{B} = R^{-1/2} H B H^{T} R^{-1/2}
        d = R^{-1/2} (y - H(x))

    """

    loc_rad: float
    N: int
    infl: float = 1.0
    wtol: float = 1e-5

    def assimilate(self, HMM, xx, yy):
        """ Carry out run with assimilation. """

        # Initial ensemble
        E = HMM.X0.sample(self.N)
        Efor, Eana = [copy(E)], []

        # Cycle
        for k, ko, t, dt in progbar(HMM.tseq.ticker, disable=True):
            # Move ensemble forward in time.
            E  = HMM.Dyn(E, t-dt, dt)
            E += HMM.Dyn.noise.sample(self.N)
            if hasattr(HMM.Dyn, 'limiter'):
                E  = HMM.Dyn.limiter(E)
            Efor.append(copy(E))

            # Analysis update
            if ko is not None and len(yy[ko]) > 0:
                self.stats.assess(k, ko, 'f', E=E)
                E = self.analyse(HMM, t, E, yy[ko])
                if hasattr(HMM.Dyn, 'limiter'):
                    E  = HMM.Dyn.limiter(E)
                Eana.append(copy(E))

            if CALCULATE_STATS:
                self.stats.assess(k, ko, E=E)

        return np.array(Efor), np.array(Eana)

    def analyse(self, HMM, time, E, yy):
        """ 
        Calculate analysis correction for a specific time and apply to ensemble. 

        Parameters
        ----------
        HMM : HiddenMarkovModel 
            Object containing all the DAPPER operators. 
        time : float 
            Current time. 
        E : 2D array of float 
            Array with each row representing an ensemble member. 
        yy : 1D array of float 
            Array with observation values. 

        """
        from scipy.sparse.linalg import cg
        from matplotlib import pyplot as plt

        # If using adaptive localization, update localization parameters using ensemble.
        if hasattr(HMM.Obs.localization, 'update_ensemble'):
            HMM.Obs.localization.update_ensemble(E)

        # Create ensemble perturbations
        N = np.size(E, 0)
        A = E - np.mean(E, axis=0, keepdims=True)
        A /= np.sqrt(N-1)
        Rinv = 1/np.sqrt(HMM.Obs.noise.C.diag)

        # Ensemble perturbations in observation space.
        D = HMM.Obs(A * np.sqrt(N-1), time) * Rinv[None, ...]

        # Calculate innovations
        xfor = np.mean(E, axis=0, keepdims=True)
        Hxfor = HMM.Obs(xfor, time)
        d = (yy - Hxfor[0]) * Rinv
        D = np.concatenate((d[None,...], D), axis=0)

        # Create operators
        self.B = self.create_B(HMM, time, A)
        
        if False:
            B = self.B(np.eye(np.size(A,1)))
            mineigs,maxeigs,trans = [],[],[]
            for n0 in np.arange(0,7):
                for n1 in np.arange(n0,7):
                    B1 = B[n0::7,n1::7]
                    B2 = B[n1::7,n0::7]
                    eig = np.linalg.eigvals(B1)
                    print('eig B',n0,n1,np.min(np.real(eig)),np.max(np.real(eig)))
                    print('sym B',np.linalg.norm(B1-B2.T)/np.linalg.norm(B1+B2.T))
                    mineigs+=[np.min(np.real(eig))]
                    maxeigs+=[np.max(np.real(eig))]
                    trans+=[np.linalg.norm(B1-B2)/np.linalg.norm(B1+B2)]
            
            print('vals',np.min(mineigs),np.max(maxeigs),np.min(trans),np.max(trans))
            eigs = np.linalg.eigvals(B)
            print('eigs',np.min(np.real(eigs)),np.max(np.real(eigs)))
            print('trans',np.linalg.norm(B-B.T)/np.linalg.norm(B+B.T))
       
        self.BHR = self.create_BHR(HMM, time, A, Rinv)
        self.RH = self.create_RH(HMM, time, Rinv)
        self.Bhat, self.Ahat = self.create_RHBHR()
        
        if False:
            B = self.Ahat(np.eye(self.Ahat.shape[0]))
            eigs = np.linalg.eigvals(B)
            print('eigs',np.min(np.real(eigs)),np.max(np.real(eigs)))
            print('trans',np.linalg.norm(B-B.T)/np.linalg.norm(B+B.T)) 
        
        self.solver = self.create_solver(self.Bhat)
        
        # Corrections average
        MU = self.solver.solve(D.T)

        # Correct ensemble mean
        E += self.BHR(MU[:,0])[None, ...]

        # Correct ensemble spread using Sakov and Oke (2008) DEnKF
        E -= 0.5 * self.BHR(MU[:, 1:]).T
        
        xana = np.mean(E,axis=0,keepdims=True)
        Hxana = HMM.Obs(xana, time)
        
        return E
    
    def create_B(self, HMM, time, A):
        # Point to localization
        loc = HMM.Obs.localization
        taperer = HMM.Obs.localization(self.loc_rad, 'x2x', time, None)
        
        # Linear operator on a matrix.
        def B(D):
            ndim = np.ndim(D)
            if ndim == 1:
                D = D[...,None]

            BD = np.zeros((np.size(A, 1), np.size(D, 1)))
            for i, a in enumerate(A.T):
                loc_ind, loc_coef = taperer([i])
                #loc_ind = np.arange(0,np.size(A,1))
                #loc_coef = np.ones_like(loc_ind)
                
                AD = A[:,loc_ind] @ (loc_coef[...,None] * D[loc_ind,:])
                BD[i] += a @ AD
                #BD[i] = loc_coef[None,...] @ D[loc_ind,:]


            if ndim == 1:
                return BD[:,0]
            else:
                return BD

        return LinearOperator((np.size(A,1),np.size(A,1)), matmat=B, matvec=B)

    def create_BHR(self, HMM, time, A, Rinv):
        """ Create linear operator B H^{T} R^{-1/2} with  B = \frac{1}{N-1} A A^{T} \dot L.

        Parameters
        ----------
        HMM : HiddenMarkovModel 
            Object containing all the DAPPER operators. 
        time : float 
            Current time. 
        A : array of float 
            Array with normalized ensemble perturbations. 
        Rinv : vector float 
            Diagional of the matrix R^{-1/2} with R being observational error covariance. 

        Returns
        -------
        Linear operator B H^{T} R^{-1/2}

        """
        shape = (HMM.Dyn.M, HMM.Obs.M)

        # Point to localization
        loc = HMM.Obs.localization
        taperer = HMM.Obs.localization(self.loc_rad, 'x2x', time, None)

        # If no linear observation operator is defined, create one as matrix.
        #IP: too slow fix this
        # if HMM.Obs.linear is None:
        #     Ht = np.array(HMM.Obs(np.eye(HMM.Dyn.M), time))
        #     HMM.Obs.linear = Ht.T
        # else:
        #     Ht = HMM.Obs.linear.T
        
        Ht = np.array(HMM.Obs(np.eye(HMM.Dyn.M), time))
        HMM.Obs.linear = Ht.T

        # Linear operator on a matrix.
        def BHR(D):
            ndim = np.ndim(D)
            if ndim == 1:
                D = D[..., None]

            BHRD = np.zeros((np.size(A, 1), np.size(D, 1)))
            for i, a in enumerate(A.T):
                loc_ind, loc_coef = taperer([i])
                AH = (A[:, loc_ind] * loc_coef[None, ...]) @ Ht[loc_ind, :]
                AHRD = AH @ (Rinv[..., None] * D)
                BHRD[i, ...] = a[None, ...] @ AHRD

            if ndim == 1:
                return BHRD[:, 0]
            else:
                return BHRD

        return LinearOperator(shape, matmat=BHR, matvec=BHR)

    def create_RH(self, HMM, time, Rinv):
        """ Create linear operator R^{-1/2} H

        Parameters
        ----------
        HMM : HiddenMarkovModel 
            Object containing all the DAPPER operators. 
        time : float 
            Current time. 
        A : array of float 
            Array with normalized ensemble perturbations. 
        Rinv : vector float 
            Diagional of the matrix R^{-1/2} with R being observational error covariance. 

        Returns
        -------
        Linear operator R^{-1/2} H

        """
        shape = (HMM.Obs.M, HMM.Dyn.M)

        # Linear operator working on matrix.
        def RH(X):
            ndim = np.ndim(X)
            if ndim == 1:
                X = X[:, None]

            RH = (HMM.Obs(X.T, time) * Rinv[None, ...]).T

            if ndim == 1:
                return RH[:, 0]
            else:
                return RH

        return LinearOperator(shape, matvec=RH, matmat=RH)

    def create_RHBHR(self):
        """ 
        Create linear operators Bhat, Ahat 

        Returns 
        -------
        Bhat : R^{-1/2} H B H^{T} R^{-1/2}
        Ahat : Bhat + I 

        """
        shape = (self.RH.shape[0], self.BHR.shape[1])
        def Bhat(X): return self.RH(self.BHR(X))
        def Ahat(X): return Bhat(X) + X
        return (LinearOperator(shape, matmat=Bhat, matvec=Bhat),
                LinearOperator(shape, matmat=Ahat, matvec=Ahat))

    def create_solver(self, Bhat):
        """
        Setup the linear solver. 

        Returns
        -------
        Solver. 

        """
        from dapper.tools.solvers import RBCG, RCG, IterationsChecker
        from dapper.tools.solvers import RankChecker
        from dapper.tools.solvers import RelativeSolutionChecker, RelativeResidualChecker, AbsResidualChecker

        def norm(x, axis=0):
            """ Bhat-norm """
            return x.T @ Bhat(x)

        eps = np.finfo(float).eps
        checkers = [IterationsChecker(40),
                    RelativeSolutionChecker(1e-2, norm=norm),
                    RelativeResidualChecker(1e-4),
                    AbsResidualChecker(10*eps),
                    RankChecker()]
        rbcg = RCG(Bhat, verbose=False, checkers=checkers, no_stored=2)

        return rbcg


@var_method
class iEnKS:
    """Iterative EnKS.

    Special cases: EnRML, ES-MDA, iEnKF, EnKF `bib.raanes2019revising`.

    As in `bib.bocquet2014iterative`, optimization uses Gauss-Newton.
    See `bib.bocquet2012combining` for Levenberg-Marquardt.
    If MDA=True, then there's not really any optimization,
    but rather Gaussian annealing.

    Args:
      upd_a (str):
        Analysis update form (flavour). One of:
        - "Sqrt"   : as in ETKF  , usin        - "Sqrt"   : as in ETKF  , using a deterministic matrix square root transform.
        - "Sqrt"   : as in ETKF  , using a deterministic matrix square root transform.
g a deterministic matrix square root transform.

        - "Sqrt"   : as in ETKF  , using a deterministic matrix square root transform.
        - "PertObs": as in EnRML , using stochastic, perturbed-observations.
        - "Order1" : as in DEnKF of `bib.sakov2008deterministic`.

      Lag:
        Length of the DA window (DAW), in multiples of dko (i.e. cycles).

        - Lag=1 (default) => iterative "filter" iEnKF `bib.sakov2012iterative`.
        - Lag=0           => maximum-likelihood filter `bib.zupanski2005maximum`.

      Shift : How far (in cycles) to slide the DAW.
              Fixed at 1 for code simplicity.

      nIter : Maximal num. of iterations used (>=1).
              Supporting nIter==0 requires more code than it's worth.

      wtol  : Rel. tolerance defining convergence.
              Default: 0 => always do nIter iterations.
              Recommended: 1e-5.

      MDA   : Use iterations of the "multiple data assimlation" type.
              Ref `bib.emerick2012history`

      bundle: Use finite-diff. linearization instead of of least-squares regression.
              Makes the iEnKS very much alike the iterative, extended KF (IEKS).

      xN    : If set, use EnKF_N() pre-inflation. See further documentation there.

    Total number of model simulations (of duration dto): N * (nIter*Lag + 1).
    (due to boundary cases: only asymptotically valid)

    Refs: `bib.bocquet2012combining`, `bib.bocquet2013joint`,
    `bib.bocquet2014iterative`.
    """

    upd_a: str
    N: int
    MDA: bool = False
    step: bool = False
    bundle: bool = False
    xN: float = None
    infl: float = 1.0
    rot: bool = False

    # NB It's very difficult to preview what should happen to
    # all of the time indices in all cases of nIter and Lag.
    # => Any changes to this function must be unit-tested via
    # scripts/test_iEnKS.py.

    # TODO 6:
    # - step length
    # - Implement quasi-static assimilation. Boc notes:
    #   * The 'balancing step' is complicated.
    #   * Trouble playing nice with '-N' inflation estimation.

    def assimilate(self, HMM, xx, yy):
        R, Ko = HMM.Obs.noise.C, HMM.tseq.Ko
        Rm12 = R.sym_sqrt_inv

        assert HMM.Dyn.noise.C == 0, (
            "Q>0 not yet supported."
            " See Sakov et al 2017: 'An iEnKF with mod. error'")

        if self.bundle:
            EPS = 1e-4  # Sakov/Boc use T=EPS*eye(N), with EPS=1e-4, but I ...
        else:
            EPS = 1.0  # ... prefer using  T=EPS*T, yielding a conditional cloud shape

        # Initial ensemble
        E = HMM.X0.sample(self.N)

        # Forward ensemble to ko = 0 if Lag = 0
        t = 0
        k = 0
        if self.Lag == 0:
            for k, t, dt in HMM.tseq.cycle(ko=0):
                self.stats.assess(k-1, None, 'u', E=E)
                E = HMM.Dyn(E, t-dt, dt)

        # Loop over DA windows (DAW).
        for ko in progbar(range(0, Ko+self.Lag+1)):
            kLag = ko-self.Lag
            DAW = range(max(0, kLag+1), min(ko, Ko) + 1)

            # Assimilation (if ∃ "not-fully-assimlated" Obs).
            if ko <= Ko:
                E = iEnKS_update(self.upd_a, E, DAW, HMM, self.stats,
                                 EPS, yy[ko], (k, ko, t), Rm12,
                                 self.xN, self.MDA, (self.nIter, self.wtol))
                E = post_process(E, self.infl, self.rot)

            # Slide/shift DAW by propagating smoothed ('s') ensemble from [kLag].
            if kLag >= 0:
                self.stats.assess(HMM.tseq.kko[kLag], kLag, 's', E=E)
            cycle_window = range(max(kLag+1, 0), min(max(kLag+1+1, 0), Ko+1))

            for kCycle in cycle_window:
                for k, t, dt in HMM.tseq.cycle(kCycle):
                    self.stats.assess(k-1, None, 'u', E=E)
                    E = HMM.Dyn(E, t-dt, dt)

        self.stats.assess(k, Ko, 'us', E=E)


def iEnKS_update(upd_a, E, DAW, HMM, stats, EPS, y, time, Rm12, xN, MDA, threshold):
    """Perform the iEnKS update.

    This implementation includes several flavours and forms,
    specified by `upd_a` (See `iEnKS`)
    """
    # distribute variable
    k, ko, t = time
    nIter, wtol = threshold
    N, Nx = E.shape

    # Init iterations.
    N1 = N-1
    HMM.X0, x0 = center(E)    # Decompose ensemble.
    w = np.zeros(N)  # Control vector for the mean state.
    T = np.eye(N)    # Anomalies transform matrix.
    Tinv = np.eye(N)
    # Explicit Tinv [instead of tinv(T)] allows for merging MDA code
    # with iEnKS/EnRML code, and flop savings in 'Sqrt' case.

    for iteration in np.arange(nIter):
        # Reconstruct smoothed ensemble.
        E = x0 + (w + EPS*T)@HMM.X0
        # Forecast.
        for kCycle in DAW:
            for k, t, dt in HMM.tseq.cycle(kCycle):  # noqa
                E = HMM.Dyn(E, t-dt, dt)
        # Observe.
        Eo = HMM.Obs(E, t)

        # Undo the bundle scaling of ensemble.
        if EPS != 1.0:
            E = inflate_ens(E, 1/EPS)
            Eo = inflate_ens(Eo, 1/EPS)

        # Assess forecast stats; store {Xf, T_old} for analysis assessment.
        if iteration == 0:
            stats.assess(k, ko, 'f', E=E)
            Xf, xf = center(E)
        T_old = T

        # Prepare analysis.
        Y, xo = center(Eo)         # Get HMM.Obs {anomalies, mean}.
        dy = (y - xo) @ Rm12.T  # Transform HMM.Obs space.
        Y = Y        @ Rm12.T  # Transform HMM.Obs space.
        Y0 = Tinv @ Y           # "De-condition" the HMM.Obs anomalies.
        V, s, UT = svd0(Y0)         # Decompose Y0.

        # Set "cov normlzt fctr" za ("effective ensemble size")
        # => pre_infl^2 = (N-1)/za.
        if xN is None:
            za = N1
        else:
            za = zeta_a(*hyperprior_coeffs(s, N, xN), w)
        if MDA:
            # inflation (factor: nIter) of the ObsErrCov.
            za *= nIter

        # Post. cov (approx) of w,
        # estimated at current iteration, raised to power.
        def Cowp(expo): return (V * (pad0(s**2, N) + za)**-expo) @ V.T
        Cow1 = Cowp(1.0)

        if MDA:  # View update as annealing (progressive assimilation).
            Cow1 = Cow1 @ T  # apply previous update
            dw = dy @ Y.T @ Cow1
            if 'PertObs' in upd_a:   # == "ES-MDA". By Emerick/Reynolds
                D = mean0(np.random.randn(*Y.shape)) * np.sqrt(nIter)
                T -= (Y + D) @ Y.T @ Cow1
            elif 'Sqrt' in upd_a:    # == "ETKF-ish". By Raanes
                T = Cowp(0.5) * np.sqrt(za) @ T
            elif 'Order1' in upd_a:  # == "DEnKF-ish". By Emerick
                T -= 0.5 * Y @ Y.T @ Cow1
            # Tinv = eye(N) [as initialized] coz MDA does not de-condition.

        else:  # View update as Gauss-Newton optimzt. of log-posterior.
            grad = Y0@dy - w*za                  # Cost function gradient
            dw = grad@Cow1                     # Gauss-Newton step
            # ETKF-ish". By Bocquet/Sakov.
            if 'Sqrt' in upd_a:
                # Sqrt-transforms
                T = Cowp(0.5) * np.sqrt(N1)
                Tinv = Cowp(-.5) / np.sqrt(N1)
                # Tinv saves time [vs tinv(T)] when Nx<N
            # "EnRML". By Oliver/Chen/Raanes/Evensen/Stordal.
            elif 'PertObs' in upd_a:
                D = mean0(np.random.randn(*Y.shape)) \
                    if iteration == 0 else D
                gradT = -(Y+D)@Y0.T + N1*(np.eye(N) - T)
                T = T + gradT@Cow1
                # Tinv= tinv(T, threshold=N1)  # unstable
                Tinv = sla.inv(T+1)           # the +1 is for stability.
            # "DEnKF-ish". By Raanes.
            elif 'Order1' in upd_a:
                # Included for completeness; does not make much sense.
                gradT = -0.5*Y@Y0.T + N1*(np.eye(N) - T)
                T = T + gradT@Cow1
                Tinv = tinv(T, threshold=N1)

        w += dw
        if dw@dw < wtol*N:
            break

    # Assess (analysis) stats.
    # The final_increment is a linearization to
    # (i) avoid re-running the model and
    # (ii) reproduce EnKF in case nIter==1.
    final_increment = (dw+T-T_old)@Xf
    # See docs/snippets/iEnKS_Ea.jpg.
    stats.assess(k, ko, 'a', E=E+final_increment)
    stats.iters[ko] = iteration+1
    if xN:
        stats.infl[ko] = np.sqrt(N1/za)

    # Final (smoothed) estimate of E at [kLag].
    E = x0 + (w+T)@HMM.X0

    return E


@var_method
class Var4D:
    """4D-Var.

    Cycling scheme is same as in iEnKS (i.e. the shift is always 1*ko).

    This implementation does NOT do gradient decent (nor quasi-Newton)
    in an inner loop, with simplified models.
    Instead, each (outer) iteration is computed
    non-iteratively as a Gauss-Newton step.
    Thus, since the full (approximate) Hessian is formed,
    there is no benefit to the adjoint trick (back-propagation).
    => This implementation is not suited for big systems.

    Incremental formulation is used, so the formulae look like the ones in iEnKS.
    """

    B: Optional[np.ndarray] = None
    xB: float = 1.0

    def assimilate(self, HMM, xx, yy):
        R, Ko = HMM.Obs.noise.C, HMM.tseq.Ko
        Rm12 = R.sym_sqrt_inv
        Nx = HMM.Dyn.M

        # Set background covariance. Note that it is static (compare to iEnKS).
        if isinstance(self.B, np.ndarray):
            # compare ndarray 1st to avoid == error for ndarray
            B = self.B.astype(float)
        elif self.B in (None, 'clim'):
            # Use climatological cov, estimated from truth
            B = np.cov(xx.T)
        elif self.B == 'eye':
            B = np.eye(Nx)
        else:
            raise ValueError("Bad input B.")
        B *= self.xB
        B12 = CovMat(B).sym_sqrt

        # Init
        x = HMM.X0.mu
        self.stats.assess(0, mu=x, Cov=B)

        # Loop over DA windows (DAW).
        for ko in progbar(np.arange(-1, Ko+self.Lag+1)):
            kLag = ko-self.Lag
            DAW = range(max(0, kLag+1), min(ko, Ko) + 1)

            # Assimilation (if ∃ "not-fully-assimlated" Obs).
            if 0 <= ko <= Ko:

                # Init iterations.
                w = np.zeros(Nx)  # Control vector for the mean state.
                x0 = x.copy()      # Increment reference.

                for iteration in np.arange(self.nIter):
                    # Reconstruct smoothed state.
                    x = x0 + B12@w
                    X = B12  # Aggregate composite TLMs onto B12
                    # Forecast.
                    for kCycle in DAW:
                        for k, t, dt in HMM.tseq.cycle(kCycle):  # noqa
                            X = HMM.Dyn.linear(x, t-dt, dt) @ X
                            x = HMM.Dyn(x, t-dt, dt)

                    # Assess forecast self.stats
                    if iteration == 0:
                        self.stats.assess(k, ko, 'f', mu=x, Cov=X@X.T)

                    # Observe.
                    Y = HMM.Obs.linear(x, t) @ X
                    xo = HMM.Obs(x, t)

                    # Analysis prep.
                    y = yy[ko]          # Get current HMM.Obs.
                    dy = Rm12 @ (y - xo)   # Transform HMM.Obs space.
                    Y = Rm12 @ Y          # Transform HMM.Obs space.
                    # Decomp for lin-alg update comps.
                    V, s, UT = svd0(Y.T)

                    # Post. cov (approx) of w,
                    # estimated at current iteration, raised to power.
                    Cow1 = (V * (pad0(s**2, Nx) + 1)**-1.0) @ V.T

                    # Compute analysis update.
                    grad = Y.T@dy - w          # Cost function gradient
                    dw = Cow1@grad           # Gauss-Newton step
                    w += dw                  # Step

                    if dw@dw < self.wtol*Nx:
                        break

                # Assess (analysis) self.stats.
                final_increment = X@dw
                self.stats.assess(k, ko, 'a', mu=x +
                                  final_increment, Cov=X@Cow1@X.T)
                self.stats.iters[ko] = iteration+1

                # Final (smoothed) estimate at [kLag].
                x = x0 + B12@w
                X = B12

            # Slide/shift DAW by propagating smoothed ('s') state from [kLag].
            if -1 <= kLag < Ko:
                if kLag >= 0:
                    self.stats.assess(HMM.tseq.kko[kLag], kLag, 's',
                                      mu=x, Cov=X@Cow1@X.T)
                for k, t, dt in HMM.tseq.cycle(kLag+1):
                    self.stats.assess(k-1, None, 'u', mu=x, Cov=Y@Y.T)
                    X = HMM.Dyn.linear(x, t-dt, dt) @ X
                    x = HMM.Dyn(x, t-dt, dt)

        self.stats.assess(k, Ko, 'us', mu=x, Cov=X@Cow1@X.T)
