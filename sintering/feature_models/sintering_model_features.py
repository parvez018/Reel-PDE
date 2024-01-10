import torch
import torch.nn as nn

from parameters import param
from diff_ops import LaplacianOp

class SinterSingleTimestep(nn.Module):
    def __init__(self):
        super().__init__()
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
        
        
        
    def forward(self,features):
        if features.dim()<3:
            print("Error, expected 4d tensors of size batch*features*X*Y")
        
        # unsqueezed = False
        # if features.dim()==3:
        #     unsqueezed = True
        #     features = features.unsqueeze(0)


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
        
        paramlist_con = [dvol0,dvol0*Qvol,dvol0*(Qvol**2),dvol0*(Qvol**3),\
                        -coefm*dvol0,-coefm*dvol0*Qvol,-coefm*dvol0*(Qvol**2),-coefm*dvol0*(Qvol**3),
                        dsur0,dsur0*Qsur,dsur0*(Qsur**2),dsur0*(Qsur**3),\
                        -coefm*dsur0,-coefm*dsur0*Qsur,-coefm*dsur0*(Qsur**2),-coefm*dsur0*(Qsur**3),\
                        dgrb0,dgrb0*Qgrb,dgrb0*(Qgrb**2),dgrb0*(Qgrb**3),\
                        -coefm*dgrb0,-coefm*dgrb0*Qgrb,-coefm*dgrb0*(Qgrb**2),-coefm*dgrb0*(Qgrb**3)]
        paramlist_eta1 = [coefl,coefl*coefk]
        paramlist_eta2 = [coefl,coefl*coefk]

        delta_con = torch.zeros_like(features[:,0])
        delta_eta1 = torch.zeros_like(features[:,0])
        delta_eta2 = torch.zeros_like(features[:,0])
        
        for i in range(len(paramlist_con)):
            delta_con += paramlist_con[i]*features[:,i]
            
        for i in range(len(paramlist_eta1)):
            delta_eta1 += paramlist_eta1[i]*features[:,i+len(paramlist_con)]
        
        for i in range(len(paramlist_eta2)):
            delta_eta2 += paramlist_eta2[i]*features[:,i+len(paramlist_con)+len(paramlist_eta1)]
        
        delta_con = self.dt*delta_con*(vm/param.kB)
        delta_eta1 = self.dt*delta_eta1
        delta_eta2 = self.dt*delta_eta2
        
        # if unsqueezed:
        #     delta_con = delta_con.squeeze()
        #     delta_eta1 = delta_eta1.squeeze()
        #     delta_eta2 = delta_eta2.squeeze()
            
        return delta_con,delta_eta1,delta_eta2