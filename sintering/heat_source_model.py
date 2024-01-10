import torch
import torch.nn
import numpy as np
from parameters import param

class HeatSource():
    def __init__(self,power,omega,xsource,ysource):
        self.power = power
        self.omega = omega
        self.xsource = xsource
        self.ysource = ysource

    def heat_flux(self,Nx,Ny):
        x = np.linspace(0,Nx*param.dx,Nx)
        y = np.linspace(0,Ny*param.dy,Ny)
        
        X,Y = np.meshgrid(x,y)
        exponent = np.exp(-0.5*((X-X[self.xsource,self.ysource])**2 + (Y-Y[self.xsource,self.ysource])**2)/(self.omega**2))
        Q = (2*self.power/(np.pi* self.omega**2))*exponent
        Q = torch.tensor(Q,dtype=torch.float64)
        return Q