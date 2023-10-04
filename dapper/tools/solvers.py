""" Linear solvers specially tailored to DA operations. """

from scipy.sparse.linalg import LinearOperator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from copy import copy
import re
from datetime import datetime

@dataclass
class Checker(ABC):
    """ Class that checks whether convergence criterium have been met and 
    still can be met. 
    
    """
    
    has_converged: bool = False 
    has_failed: bool = False 
    
    @abstractmethod
    def __call__(self, iterno, R, X):
        """
        Check if convergence criterium has been met. 

        Parameters
        ----------
        iterno : int
            Iteration number
        R : 2D numpy array of floats
            Array with residuals as columns.
        X : 2D numpy array floats
            Array with guesses as columns.

        Returns
        -------
            None
        """
        pass
    
    def set_solver(self, solver):
        """ Set solver after initialization."""
        self.solver = solver
    
    @abstractmethod 
    def _diag_message(self):
        """
        Print current state of ConvergenceCheck to string.

        Returns
        -------
        String with diagnostic message.

        """
        pass
    
    def _convergence_message(self):
        """
        Print convergence message to string. 
        
        Returns
        -------
        String with convergence message. 
        """
        return "CONVERGED"
    
    def _fail_message(self):
        return "FAILED"
    
    @property 
    def diag_message(self):
        """ Message to display with information about progress. """
        return self._diag_message()
    
    @property 
    def convergence_message(self):
        """ Message to display when convergence has occured. """
        if self.has_converged:
            return self._convergence_message()
        else:
            return ""
        
    @property 
    def fail_message(self):
        """ Message to display when a critical failure has occured. """
        if self.has_failed:
            return self._fail_message()
        else:
            return ""
        
    @property 
    def name(self):
        """ Classname of the object. """
        return self.__class__.__name__

class IterationsChecker(Checker):
    """ Checks if a predefined number of iterations has been carried out."""
    
    def __init__(self, max_iterations, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations 
        
    def __call__(self, iterno, R, X):
        self.iteration = iterno
        self.has_failed = iterno >= self.max_iterations

    def _diag_message(self):
        return "Iteration number: {:d}".format(self.iteration)
    
    def _fail_message(self):
        return "CONVERGED: maximum number of iterations ({:d}) reached.".format(self.max_iterations)
        
        
class RelativeResidualChecker(Checker):
    """ Checks if relative residual norm is smaller than a predefined value. """
    
    def __init__(self, max_norm, norm=np.linalg.norm, **kwargs):
        super().__init__(**kwargs)
        self.max_norm = max_norm 
        self.norm = norm 
        self.norms = []
        
    def __call__(self, iterno, R, X):
        self.norms.append(self.norm(R,axis=0))
        self.rnorm = np.max(self.norms[-1]/ self.norms[0])
        self.has_converged = self.rnorm <= self.max_norm
        
    def _diag_message(self):
        return "Relative norm residual: {:.2e}".format(self.rnorm)
    
    def _convergence_message(self):
        return "CONVERGED: max. relative residual norm smaller than {:.2e}.".format(self.max_norm)
  
class AbsResidualChecker(Checker):
    """ Checks if relative residual norm is smaller than a predefined value. """
    
    def __init__(self, max_norm, norm=np.linalg.norm, **kwargs):
        super().__init__(**kwargs)
        self.max_norm = max_norm 
        self.norm = norm 
        self.norms = []
        
    def __call__(self, iterno, R, X):
        self.norms.append(self.norm(R,axis=0))
        self.anorm = np.max(self.norms[-1])
        self.has_converged = self.anorm <= self.max_norm
        
    def _diag_message(self):
        return "Absolute norm residual: {:.2e}".format(self.anorm)
    
    def _convergence_message(self):
        return "CONVERGED: max. absolute residual norm smaller than {:.2e}.".format(self.max_norm)
    
        
class RelativeSolutionChecker(Checker):
    """ Checks if relative change in solution is smaller than a predefined value. """
    
    def __init__(self, max_error, norm=np.linalg.norm, **kwargs):
        super().__init__(**kwargs)
        self.max_error = max_error 
        self.norm = norm 
        self.norms = []
        self.x0, self.x1 = None, None
        
    def __call__(self, iterno, R, X):
        if iterno == 0:
            self.x0 = copy(X)
            self.has_converged = False
        else:
            self.x1 = self.x0
            self.x0 = copy(X) 
            
            norm = self.norm(self.x0 - self.x1 ,axis=0) / self.norm(self.x0, axis=0)
            self.norms.append(norm)
            self.rerror = np.max(norm)
        
            self.has_converged = self.rerror <= self.max_error
            
    def _diag_message(self):
        if self.x1 is None:
            return ""
        else:
            return "Relative error: {:.2e}".format(self.rerror)
    
    def _convergence_message(self):
        return "CONVERGED: max. relative error smaller than {:.2e}.".format(self.max_error)           

class RankChecker(Checker):
    
    def __init__(self):
        pass
        
    def __call__(self, iterno, R, X):
        self.ranks = np.sum(self.solver.ranks)            
        self.has_converged = self.ranks >= np.size(R,0)
        if len(self.solver.ranks)>0:
            self.has_converged = self.has_converged or (self.solver.ranks[-1]==0)
        
    def _convergence_message(self):
        return "CONVERGED: rank basis reached maximum size."
    
    def _diag_message(self):
        return "Rank basis is {:3d}.".format(self.ranks)

@dataclass
class KrylovSolver(ABC):
    """ 
    Class representing linear solver based on Krylov methods. 
    """
    
    no_stored: int = np.inf
    verbose: bool = False
    checkers: np.ndarray = field(default_factory=lambda : np.array([]))
    ranks: np.ndarray = field(default_factory=lambda : np.array([],dtype=int))

    def solve(self, B, X0=0):
        """
        Solve the system of linear equations AX = B
        
        Parameters
        ----------
        A : numpy.linalg.LinearOperator 
            Linear operator that generates the system of linear equations. 
        B : 1D/2D numpy array 
            Right-hand-side of the system of linear equations. 
        X0 : 1D/2D numpy array 
            Initial guesss solution. Default is 0. 
            
        Returns
        -------
        Solution X to the system.
            
        """
        B, X0 = self.format_input(B, X0)
        
        #Initial residual
        R = B - self.A(X0)
        
        #Default convergence
        if len(self.checkers)==0:
            self.checkers = [IterationsChecker(10)]            
            
        #Check if convergence criterium has been met. 
        iterno = 0
        for check in self.checkers:
            check.set_solver(self)
            check(iterno, R, X0)
            if self.verbose:
                print(check.diag_message)
        
        #Run iterations
        while not any([check.has_converged for check in self.checkers]):
            #Update iteration number
            iterno += 1
                
            #Expand Krylov basis
            self.update_basis(R, X0)
            
            #Project on new Krylov basis
            X0, R = self.update_solution(R, X0)
            
            #Check if convergence criterium has been met. 
            for check in self.checkers:
                check(iterno, R, X0)
                if self.verbose:
                    print(check.diag_message)
                if check.has_failed:
                    raise StopIteration(check.fail_message)
                    
        #convergence messages
        for check in self.checkers:
            if self.verbose and check.has_converged:
                print(check.convergence_message)
            
        X0 = self.format_output(X0)
        return X0        
            
    @abstractmethod 
    def update_basis(self, R, X0):
        """
        Add new vectors to Krylov basis and optionally remove old ones. 

        Parameters
        ----------
        R : 2D numpy array of floats
            Array with residuals as columns.
        X0 : 2D numpy array of floats
            Array with the latest approximations to the solution as columns.

        Returns
        -------
        None.

        """
        pass
    
    @abstractmethod 
    def orthonormalize(self):
        """
        Orthonomalizes the Krylov basis. 

        Returns
        -------
        None.

        """
        pass
    
    @abstractmethod 
    def update_solution(self, R, X0):
        """
        Calculates new approximation and new residuals using the latest
        Krylov basis. 

        Parameters
        ----------
        R : numpy array of floats
            Array with residuals before update as columns.
        X0 : numpy array of float
            Array with guesses before update as columns.

        Returns
        -------
        X0 : 
            Array with guesses after update as columns.
        R : 
            Array with residuals after update as columns.
        
        """
        pass 
        
    def format_input(self, B, X0):
        """ Reshape input and check dimensions. """
        if np.ndim(B)==1:
            self.is_vector = True 
            B = B[...,None]
        else:
            self.is_vector = False
            
        if np.ndim(X0)==0:
            X0 *= np.ones((self.A.shape[-1],np.size(B,1)))
        elif np.ndim(X0)==1:
            X0  = X0[...,None]
            
        if np.size(X0,0) != self.A.shape[-1]:
            raise ValueError("Incongruent dimensions for X0 and A.")
        if np.size(B,0) != self.A.shape[0]:
            raise ValueError("Incongruent dimensions for B and A.")
        if np.size(B,1) != np.size(X0,1):
            raise ValueError("Incongruent dimensions for B and X0.")
        
        return B, X0 
    
    def format_output(self, X0):
        """ Reshape output. """
        if self.is_vector:
            return X0[:,0]
        else:
            return X0 
        
class RCG(KrylovSolver):
    """ 
    Reduced Conjugate Gradient solver based on 
    
        Gürol, S., Weaver, A. T., Moore, A. M., Piacentini, A., Arango, 
        H. G., & Gratton, S. (2014). 
        B‐preconditioned minimization algorithms for variational data 
        assimilation with the dual formulation. Quarterly Journal of the 
        Royal Meteorological Society, 140(679), 539-556.
        
    By default this solver uses reothogonalization. Can be switch off by 
    specifying providing no_storage=1 as argument. 
    
    """
    
    def __init__(self, Bhat, **kwargs):
        """ Class constructor. 
        
        Parameters
        ----------
        Bhat : LinearOperator | numpy.array
            Linear operator representing R^{-1/2} H B H^{T} R^{-1/2}. 
            
        """
        #Overwrite defaults in KrylovSolver
        super().__init__(**kwargs)
        
        #Cast Bhat as operator.
        if isinstance(Bhat, LinearOperator):
            self.Bhat = Bhat
        elif isinstance(Bhat, np.ndarray):
            Bx = lambda x: Bhat@x 
            BX = lambda X: Bhat@X
            self.Bhat = LinearOperator(np.shape(Bhat), matmat=BX, matvec=Bx)
        else:
            raise TypeError("{} not a valid type.".format(type(Bhat)))
        
        #Bhat+I
        Ax = lambda x: self.Bhat(x) + x 
        AX = lambda X: self.Bhat(X) + X 
        self.A = LinearOperator(self.Bhat.shape, matvec=Ax, matmat=AX)
        
        D = self.A.shape[0]
        self.V, self.AV = np.empty((D,0)), np.empty((D,0))
        
    def update_basis(self, R, X0):
        #New application B
        BR = self.Bhat(R) 
        
        #Create new basisvector from residual.
        self.V  = np.concatenate((self.V , R     ), axis=1)
        self.AV = np.concatenate((self.AV, BR + R), axis=1)
        
        #Orthonormalize basisvectors 
        self.orthonormalize(np.size(R,1))
        
        #Calculate number of basis vectors to retain.
        if np.isinf(self.no_stored):
            N = np.size(self.V,0)
        else:
            N = np.sum(self.ranks[-self.no_stored-1:])
        
        #Only retain new basis vectors
        self.V  = self.V[:,-N:]
        self.AV = self.AV[:,-N:]
        
    def orthonormalize(self, N):
        actives = np.full((np.size(self.V,1),), True, dtype=bool)
        self.ranks = np.append(self.ranks, 0)
        
        # #Orthonormalize coefficients using Gramm-Schmidt
        # for n in np.arange(-N,0):            
        #     for m in np.arange(-np.size(self.V,1),n):
        #
        #         #projection coefficient
        #         p1 = np.dot(self.V[:,m], self.AV[:,n])
        #         p2 = np.dot(self.V[:,n], self.AV[:,m])
        #         p  = 0.5*p1 + 0.5*p2
        #
        #         if np.abs(p1-p2)/np.abs(p) > .01 and np.abs(p)>1e-16:
        #             print('P ',n,m,p1,p2)
        #             raise Exception('Non-symmetric A')
        #         elif np.abs(p1-p2)/np.abs(p) > .05:
        #             print('P ',n,m,p1,p2)
        #             p = 0.
        #
        #         #orthogonalize
        #         self.V[:,n]  -=  self.V[:,m]*p 
        #         self.AV[:,n] -= self.AV[:,m]*p
        #
        #     #machine precision
        #     eps = np.finfo(float).eps
        #
        #     #norm squared
        #     p2 = np.dot(self.V[:,n], self.AV[:,n])
        #
        #     #normalize
        #     if p2>eps:
        #         p = np.sqrt(p2)
        #         self.V[:,n]  /= p 
        #         self.AV[:,n] /= p
        #         self.ranks[-1] += 1
        #     else:
        #         print('P2 INACTIVE',n,p2)
        #         actives[n] = False   
                
        VAV = self.V.T @ self.AV
        U,S,Vt = np.linalg.svd(VAV)
        mask = S/np.max(S) > 10 * np.finfo(float).eps
        
        self.V = self.V @ (U[:,mask] @ np.diag(1/np.sqrt(S[mask])))
        self.AV = self.AV @ (Vt.T[:,mask] @ np.diag(1/np.sqrt(S[mask])))

        #self.V = np.compress(actives, self.V, axis=1)
        #self.AV = np.compress(actives, self.AV, axis=1)
        
        
    def update_solution(self, R, X0):        
        projection = self.V.T @ R
        X0 += self.V  @ projection 
        R  -= self.AV @ projection
        
        return X0, R
          
class RBCG(KrylovSolver):
    """ 
    Reduced B-Lanczos Conjugate Gradient solver based on 
    
        Gürol, S., Weaver, A. T., Moore, A. M., Piacentini, A., Arango, 
        H. G., & Gratton, S. (2014). 
        B‐preconditioned minimization algorithms for variational data 
        assimilation with the dual formulation. Quarterly Journal of the 
        Royal Meteorological Society, 140(679), 539-556.
        
    By default this solver uses reothogonalization. Can be switch off by 
    specifying providing no_storage=1 as argument. 
    
    """
    
    def __init__(self, Bhat, **kwargs):
        """ Class constructor. 
        
        Parameters
        ----------
        Bhat : LinearOperator | numpy.array
            Linear operator representing R^{-1/2} H B H^{T} R^{-1/2}. 
            
        """
        #Overwrite defaults in KrylovSolver
        super().__init__(**kwargs)
        
        #Cast Bhat as operator.
        if isinstance(Bhat, LinearOperator):
            self.Bhat = Bhat
        elif isinstance(Bhat, np.ndarray):
            Bx = lambda x: Bhat@x 
            BX = lambda X: Bhat@X
            self.Bhat = LinearOperator(np.shape(Bhat), matmat=BX, matvec=Bx)
        else:
            raise TypeError("{} not a valid type.".format(type(Bhat)))
        
        #Bhat+I
        Ax = lambda x: self.Bhat(x) + x 
        AX = lambda X: self.Bhat(X) + X 
        self.A = LinearOperator(self.Bhat.shape, matvec=Ax, matmat=AX)
        
        D = self.A.shape[0]
        self.V, self.BV, self.AV = np.empty((D,0)), np.empty((D,0)), np.empty((D,0))
        
    def update_basis(self, R, X0):
        #New application B
        BR = self.Bhat(R) 
        
        #Create new basisvector from residual.
        self.V  = np.concatenate((self.V , R     ), axis=1)
        self.BV = np.concatenate((self.BV, BR    ), axis=1)
        self.AV = np.concatenate((self.AV, BR + R), axis=1)
        
        #Orthonormalize basisvectors 
        self.orthonormalize(np.size(R,1))
        
        #Calculate number of basis vectors to retain.
        if np.isinf(self.no_stored):
            N = np.size(self.V,0)
        else:
            N = np.sum(self.ranks[-self.no_stored-1:])
        
        #Only retain new basis vectors
        self.V  = self.V[:,-N:]
        self.BV = self.BV[:,-N:]
        self.AV = self.AV[:,-N:]
        
    def orthonormalize(self, N):
        actives = np.full((np.size(self.V,1),), True, dtype=bool)
        self.ranks = np.append(self.ranks, 0)
        
        #Orthonormalize coefficients using Gramm-Schmidt
        for n in np.arange(-N,0):            
            for m in np.arange(-np.size(self.V,1),n):
                
                #projection coefficient
                p1 = np.dot(self.BV[:,m], self.AV[:,n])
                p2 = np.dot(self.BV[:,n], self.AV[:,m])
                p  = 0.5*p1 + 0.5*p2

                #orthogonalize
                self.V[:,n]  -=  self.V[:,m]*p 
                self.BV[:,n] -= self.BV[:,m]*p 
                self.AV[:,n] -= self.AV[:,m]*p
                
            #machine precision
            eps = np.finfo(float).eps
            
            #norm squared
            p2 = np.dot(self.BV[:,n], self.AV[:,n])
            
            #normalize
            if p2>eps:
                p = np.sqrt(p2)
                self.V[:,n]  /= p 
                self.BV[:,n] /= p 
                self.AV[:,n] /= p
                self.ranks[-1] += 1
            else:
                actives[n] = False
                
        self.V = np.compress(actives, self.V, axis=1)
        self.BV = np.compress(actives, self.BV, axis=1)
        self.AV = np.compress(actives, self.AV, axis=1)
    
    def update_solution(self, R, X0):        
        projection = self.BV.T @ R
        X0 += self.V  @ projection 
        R  -= self.AV @ projection
        
        return X0, R