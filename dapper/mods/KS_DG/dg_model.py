#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:27:55 2023

This model contains the DG part of the Kuramoto-Sivashinki model. 

@author: ivo
"""
from abc import ABC, abstractmethod 
import numpy as np
import firedrake as fem


class DgTransformer(ABC):
    
    @abstractmethod 
    def state2vector(self, state):
        pass
        
    @abstractmethod 
    def vector2state(self, state, vector):
        pass
    
class IdTransformer(DgTransformer):
    
    def state2vector(self, state):
        vector = state.dat.data[:]
        return vector
    
    def vector2state(self, state, vector):
        state.dat.data[:] = vector 
        return state
    

class KsModel:
    """
    Class containing all parts of a discontinous Galerkin Kuramoto-Sivashinki
    model. 

    This model solves for the PDE
    \partial_{t} u + u \partial_{x} u + \alpha \partial^{2}_{x} u + \beta \partial_{x}^{4} u = 0

    Parameters
    ----------
    nu : float 


    """

    def __init__(self, dt, alpha=1, beta=1, transformer = IdTransformer()):
        self.dt = dt
        self.alpha, self.beta = alpha, beta
        self.create_solver()
        self.transformer = transformer

    def create_space(self, order, length, n_cells):
        """
        Create periodic mesh on which the solution as well as function spaces
        for the solution. 

        Parameters
        ----------
        order : int>=0
            Order of the Legendre polynomials used the model  
        length : float > 0
            Distance after which the solution repeats. 
        n_cells : int > 0
            Number of grid cells. 

        """
        # Computational mesh.
        self.mesh = fem.PeriodicIntervalMesh(n_cells, length)

        # Functional spaces for solution and its derivative.
        spaces = []
        for n in np.arange(order,order-4,-1):
            print('order ',n)
            element = fem.FiniteElement("DG", cell=self.mesh.ufl_cell(), degree=int(n),
                                        variant="spectral")
            spaces += [fem.FunctionSpace(self.mesh, element)]

        self.space = fem.MixedFunctionSpace(spaces)

        # Spatial coordinates
        element = fem.VectorElement("DG", cell=self.mesh.ufl_cell(), 
                                    degree=int(order), variant="spectral")
        space = fem.FunctionSpace(self.mesh, element)
        self.coord_func = fem.project(fem.SpatialCoordinate(self.mesh), space)
        self.coords = self.coord_func.dat.data
        
        #States to store output. 
        self.states = [fem.Function(self.space) for _ in range(2)]
        
    def create_form(self):
        #Trial functions for form. 
        u0, u1, u2, u3 = fem.split(self.states[0])
        #Test functions for form.
        v0, v1, v2, v3 = fem.TestFunctions(self.space)
        #Set values. 
        _u0, _u1, _u2, _u3 = self.states[1].split()
        _u0.dat.data[:] = np.where(self.coords<=50.,self.coords,100.-self.coords)/50
        
        #Equations for the derivatives
        #\partial_{t}u
        self.form  = (1/self.dt)*(v0*u0 - v0*_u0)*fem.dx
        #1/2 * \partial_{x} u^2
        self.form += -(v0*u0).dx(0)*u0*fem.dx + fem.jump(v0)*(fem.avg(u0**2)/3 + 2*fem.avg(u0)**2/3)*fem.dS 
        #\alpha * \partial_{x}^2 u
        self.form += -self.alpha*v0.dx(0)*u1*fem.dx + self.alpha*fem.jump(v0)*fem.avg(u1)*fem.dS
        #\beta * \partial_{x}^4 u
        self.form += -self.beta*v0.dx(0)*u3*fem.dx + self.beta*fem.jump(v0)*fem.avg(u3)*fem.dS
        
        #\partial_{x}^{i} u
        for Dv, Du, U in zip([v1,v2,v3],[u1,u2,u3],[u0,u1,u2]):
            self.form += (Dv*Du + Dv.dx(0)*U) * fem.dx 
            self.form -= fem.jump(Dv) * fem.avg(U) * fem.dS
            
        #Build the solver. 
        problem = fem.NonlinearVariationalProblem(self.form, self.states[0])
        self.solver = fem.NonlinearVariationalSolver(problem, 
                                                     solver_parameters = self._solver)
        
    def create_solver(self, **kwargs):
        defaults = {"snes_type" : "newtonls", 
                    "snes_linesearch_type" : "bt", 
                    "ksp_type" : "preonly",
                    "pc_type" : "lu",
                    }
        
        self._solver = {**defaults, **kwargs}
        
        
    def run(self, duration):
        if not hasattr(self, "form"):
            self.create_form()
            
        time = 0 
        while time < duration:
            self.solver.solve()
            self.states[1].assign(self.states[0])
            time += self.dt
            
        return self.transformer.state2vector(self.states[0])
        
        

model = KsModel(.1)
model.create_space(4, 100, 20)
model.run(.6)
