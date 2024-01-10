import torch
import torch.nn as nn

import sys
sys.path.append("../")
from parameters import param
from diff_ops import LaplacianOp

class ThermalSingleTimestep(nn.Module):
        def __init__(self,dx=param.dx,dy=param.dy,dt=param.dt):
            super().__init__()
            self.cp = nn.Parameter(torch.randn(1)*5 + 0.1, requires_grad=True) # C_p
            self.k_air = nn.Parameter(torch.randn(1) + 0.1, requires_grad=True) # Kappa_air
            self.k_metal = nn.Parameter(torch.randn(1) + 0.1, requires_grad=True) # Kappa_metal
            self.rho = nn.Parameter(torch.randn(1)*5 + 0.1, requires_grad=True) # rho

            self.T0 = param.T0
            
            self.lap = LaplacianOp()
            self.dx = dx
            self.dy = dy
            self.dt = dt

         
        def init_params(self,mparams):
            self.cp.data = torch.tensor([mparams['cp']])
            self.k_air.data = torch.tensor([mparams['k_air']])
            self.k_metal.data = torch.tensor([mparams['k_metal']])
            self.rho.data = torch.tensor([mparams['rho']])
            
            
        def forward(self,features):
            if features.dim()<3:
                print("Error, expected 4d tensors of size batch*features*X*Y")
        
            # unsqueezed = False
            # if features.dim()==3:
            #     unsqueezed = True
            #     features = features.unsqueeze(0)

            k_metal = torch.abs(self.k_metal)
            k_air = torch.abs(self.k_air)
            rho = torch.abs(self.rho)
            cp = torch.abs(self.cp)

            feat1 = features[:,0]
            feat2 = features[:,1]
            feat3 = features[:,2]
            
            
            delta_tp = (k_metal/(rho*cp))*feat1 + (k_air/(rho*cp))*feat2 + (1.0/(rho*cp))*feat3
            delta_tp = self.dt*delta_tp

            # if unsqueezed:
            #     delta_tp = delta_tp.squeeze()

            return delta_tp
            
            
