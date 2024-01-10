import torch


class SimulationParameters():
    pass
    
param = SimulationParameters()


param.Nx = 130
param.Ny = 130
param.dx = 1
param.dy = 1
param.nstep = 5000
param.nprint = 100
param.dt = 2e-2
param.eps = 1e-6
param.N = 1

param.fluct_norm = 100

param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# param.device = torch.device("cpu")
param.compression = 0.1

param.p_casc = 0.01