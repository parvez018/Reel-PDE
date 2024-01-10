import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("../")

from parameters import param
from diff_ops import LaplacianOp

def fix_deviations(mat, lb=0.0, ub=1.0):
# def fix_deviations(mat, lb=0.0001, ub=0.9999):
    mat.masked_fill_(torch.ge(mat, ub).detach(), ub)
    mat.masked_fill_(torch.le(mat, lb).detach(), lb)
    return mat
    
class SinterSingleTimestep(nn.Module):
    def __init__(self):
        super(SinterSingleTimestep, self).__init__()
        self.coefm =  nn.Parameter(torch.randn(1)+0.1, requires_grad=True) 
        self.coefk =  nn.Parameter(torch.randn(1)+0.1, requires_grad=True) 
        self.coefl =  nn.Parameter(torch.randn(1)+0.1, requires_grad=True) 
        
        self.dvol0 =  nn.Parameter(torch.randn(1)+0.1, requires_grad=True) 
        self.dsur0 =  nn.Parameter(torch.randn(1)+0.1, requires_grad=True) 
        self.dgrb0 =  nn.Parameter(torch.randn(1)+0.1, requires_grad=True) 
        self.dvap0 =  torch.tensor([0.0])
        
        self.Qvol =  nn.Parameter(torch.randn(1)+0.1, requires_grad=True) 
        self.Qgrb =  nn.Parameter(torch.randn(1)+0.1, requires_grad=True) 
        self.Qsur =  nn.Parameter(torch.randn(1)+0.1, requires_grad=True) 

        self.vm = nn.Parameter(torch.randn(1)*50+0.1, requires_grad=True)

        self.lap = LaplacianOp()
        self.dt = param.dt
        self.dx = param.dx
        self.dy = param.dy
    
    def init_params(self,mparams):
        self.coefm.data = torch.tensor([mparams['coefm']])
        self.coefk.data = torch.tensor([mparams['coefk']])
        self.coefl.data = torch.tensor([mparams['coefl']])
        
        self.dvol0.data = torch.tensor([mparams['dvol0']])
        self.dvap0.data = torch.tensor([mparams['dvap0']])
        self.dsur0.data = torch.tensor([mparams['dsur0']])
        self.dgrb0.data = torch.tensor([mparams['dgrb0']])
        
        self.Qvol.data = torch.tensor([mparams['Qvol']])
        self.Qsur.data = torch.tensor([mparams['Qsur']])
        self.Qgrb.data = torch.tensor([mparams['Qgrb']])
        self.vm.data = torch.tensor([mparams['vm']])
        
        
        
    def forward(self,con,eta1,eta2,tp):
        # tp: temperature field
        # con: concentration, \phi in sintering model
        # eta: order parameters for each grain
        A = 16.0
        B = 1.0

        Qvol = torch.abs(self.Qvol)
        Qsur = torch.abs(self.Qsur)
        Qgrb = torch.abs(self.Qgrb)

        dvol0 = torch.abs(self.dvol0)
        dsur0 = torch.abs(self.dsur0)
        dgrb0 = torch.abs(self.dgrb0)

        coefl = torch.abs(self.coefl)
        coefk = torch.abs(self.coefk)
        coefm = torch.abs(self.coefm)
        
        vm = torch.abs(self.vm)
        
        dvol = dvol0*torch.exp(-Qvol/(param.kB*tp))
        dsur = dsur0*torch.exp(-Qsur/(param.kB*tp))
        dgrb = dgrb0*torch.exp(-Qgrb/(param.kB*tp))
        
        dvol = dvol0*torch.exp(-Qvol/(param.kB*tp))
        dsur = dsur0*torch.exp(-Qsur/(param.kB*tp))
        dgrb = dgrb0*torch.exp(-Qgrb/(param.kB*tp))
        dvap = self.dvap0

        
        mvol = dvol*vm/(param.kB*tp)
        msur = dsur*vm/(param.kB*tp)
        mgrb = dgrb*vm/(param.kB*tp)
        mvap = dvap*vm/(param.kB*tp)

        sum2 = eta1**2 + eta2**2
        sum3 = eta1**3 + eta2**3
        dfdcon = B*(2*con + 4*sum3 - 6*sum2) - 2*A*(con**2)*(1-con) + 2*A*con*((1-con)**2)
        
        phi = (con**3)*(10-15*con+6*(con**2))
        
        ssum = eta1*eta2 + eta2*eta1
                
        mobil = mvol*phi + mvap*(1.0-phi) + msur*con*(1.0-con) + mgrb*ssum
        
        lap_con = self.lap(con,self.dx,self.dy)
        lap_con2 = self.lap(dfdcon-coefm*lap_con,self.dx,self.dy)
        
        con_new = con + self.dt*mobil*lap_con2
        
        dfdeta1 = B*(-12*(eta1**2)*(2-con) + 12.0*eta1*(1-con) + 12*eta1*sum2)
        dfdeta2 = B*(-12*(eta2**2)*(2-con) + 12.0*eta2*(1-con) + 12*eta2*sum2)
        
        eta1_new = eta1 - self.dt*coefl*(dfdeta1 - coefk*self.lap(eta1,self.dx,self.dy))
        eta2_new = eta2 - self.dt*coefl*(dfdeta2 - coefk*self.lap(eta2,self.dx,self.dy))
        
        fix_deviations(con_new)
        fix_deviations(eta1_new)
        fix_deviations(eta2_new)
        
        return con_new,eta1_new,eta2_new