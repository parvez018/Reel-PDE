import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../")

from parameters import param
from diff_ops import LaplacianOp
from heat_source_model import HeatSource

class ThermalSingleTimestep(nn.Module):
        def __init__(self,dx=param.dx,dy=param.dy,dt=param.dt):
            super().__init__()
            self.cp = nn.Parameter(torch.randn(1)*5 + 0.1, requires_grad=True) # C_p
            self.k_air = nn.Parameter(torch.randn(1)*2 + 0.1, requires_grad=True) # Kappa_air
            self.k_metal = nn.Parameter(torch.randn(1)*5 + 0.1, requires_grad=True) # Kappa_metal
            self.rho = nn.Parameter(torch.randn(1)*5 + 0.1, requires_grad=True) # rho
            
            self.T0 = param.T0
            self.heat_source = HeatSource(power=param.power,omega=param.omega,xsource=param.xsource,ysource=param.ysource)
            
            
            self.lap = LaplacianOp()
            self.dx = dx
            self.dy = dy
            self.dt = dt
               
        def heat_flux(self,Tp):
            Nx,Ny = Tp.size()
            Q = self.heat_source.heat_flux(Nx,Ny)
            return Q
        
            
        def init_params(self,mparams):
            self.cp.data = torch.tensor([mparams['cp']])
            self.k_air.data = torch.tensor([mparams['k_air']])
            self.k_metal.data = torch.tensor([mparams['k_metal']])
            self.rho.data = torch.tensor([mparams['rho']])
            
            
        def forward(self,Tp,con):
            if Tp.dim()==1:
                print("Expected 2d grid, instead got 1d")
                return
            
            k_air = torch.abs(self.k_air)
            k_metal = torch.abs(self.k_metal)
            rho = torch.abs(self.rho)
            cp = torch.abs(self.cp)
            k_field = con*k_metal + (1-con)*k_air
            
            lap_Tp = self.lap(Tp,self.dx,self.dy,self.T0)
            Qflux = self.heat_flux(Tp)
            
            # print("types",type(k_field),type(lap_Tp),type(Qflux))
            delta_Tp = self.dt*(k_field*lap_Tp + Qflux)/(rho*cp)
            
            Tp_new = Tp + delta_Tp
            return Tp_new,Qflux