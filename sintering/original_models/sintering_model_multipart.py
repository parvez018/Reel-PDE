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
        
        
        
    def forward(self,con,etas,tp):
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
        

        # dfdeta = torch.zeros_like(con)
        
        # sum2 = torch.zeros_like(con)
        # sum3 = torch.zeros_like(con)
        
        # for ipart in range(npart):
        #     sum2 += etas[ipart,:,:]**2
        #     sum3 += etas[ipart,:,:]**3
        
        sum2 = (etas**2).sum(dim=0)
        sum3 = (etas**3).sum(dim=0)
        dfdcon = B*(2*con + 4*sum3 - 6*sum2) - 2*A*(con**2)*(1-con) + 2*A*con*((1-con)**2)
        
        phi = (con**3)*(10-15*con+6*(con**2))
        
        ssum = torch.zeros_like(con)
        npart,_,_ = etas.size()
        for ipart in range(npart):
            for jpart in range(npart):
                if ipart==jpart:
                    continue
                ssum += etas[ipart,:,:]*etas[jpart,:,:]
                
        mobil = mvol*phi + mvap*(1.0-phi) + msur*con*(1.0-con) + mgrb*ssum
        
        lap_con = self.lap(con,self.dx,self.dy)
        lap_con2 = self.lap(dfdcon-coefm*lap_con,self.dx,self.dy)
        
        
        con_new = con + self.dt*mobil*lap_con2
        etas_new = torch.clone(etas)
        
        # for ipart in range(npart):
        #     eta = etas[ipart,:,:]
        #     # dfdcon,dfdeta = free_energy_sint(Nx,Ny,con,eta,etas,npart)
        #     dfdeta = B*(-12*(eta**2)*(2-con) + 12.0*eta*(1-con) + 12*eta*sum2)
        #     etas_new[ipart,:,:] -= self.dtime*self.coefl*(dfdeta - 0.5*self.coefk*self.lap(eta,self.dx,self.dy))
        
        dfdeta = B*(-12*(etas**2)*(2-con[None,:,:]) + 12.0*etas*(1-con[None,:,:]) + 12*etas*sum2[None,:,:])
        etas_new = etas - self.dt*coefl*(dfdeta-coefk*self.lap(etas,self.dx,self.dy))

        fix_deviations(con_new)
        fix_deviations(etas_new)
        
        # print("dfdeta",dfdeta.size())
        # print("etas_new",etas_new.size())
        # print("con",con.size(),con[None,:,:].size())
        return con_new,etas_new