import torch

class Param():
    def __init__(self,name="thermal_model"):
        self.model = name
param = Param()

param.device = torch.device("cpu")

param.kB = 0.8625e-1
param.vm = 25

param.dx = 0.5
param.dy = 0.5
param.dt = 1e-4




# # source for 100*100 grid
param.Nx = 100
param.Ny = 100
param.xsource = 50
param.ysource = 65
param.power = 8e4
param.omega = 2 # effective laser spot radius

param.T0 = 298.0