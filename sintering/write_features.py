#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import sys
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.fft import fft2,fftshift,ifft2,ifftshift


# In[2]:


from parameters import param
from diff_ops import LaplacianOp
from heat_source_model import HeatSource


# In[3]:


device = param.device


# In[4]:


def sintering_features(con,eta1,eta2,T):
    A = 16.0
    B = 1.0
    
    h = (con**3)*(10-15*con+6*(con**2))
    
    sum2 = eta1**2 + eta2**2
    sum3 = eta1**3 + eta2**3
    dfdcon = B*(2*con + 4*sum3 - 6*sum2) - 2*A*(con**2)*(1-con) + 2*A*con*((1-con)**2)
    
    dfdeta1 = B*(-12*(eta1**2)*(2-con) + 12.0*eta1*(1-con) + 12*eta1*sum2)
    dfdeta2 = B*(-12*(eta2**2)*(2-con) + 12.0*eta2*(1-con) + 12*eta2*sum2)
    
    lap = LaplacianOp()
    lap_dfdcon = lap(dfdcon,param.dx,param.dy)
    lap4con = lap(lap(con,param.dx,param.dy),param.dx,param.dy)
    lapeta1 = lap(eta1,param.dx,param.dy)
    lapeta2 = lap(eta2,param.dx,param.dy)
    ssum = 2*eta1*eta2
    
    w1 = (h*lap_dfdcon)/T
    w2 = (h*lap4con)/T
    w3 = con*(1-con)*lap_dfdcon/T
    w4 = con*(1-con)*lap4con/T
    w5 = ssum*lap_dfdcon/T
    w6 = ssum*lap4con/T
    
    
    con_features = [w1,w1/(-param.kB*T),w1/(2*(-param.kB*T)**2),w1/(6*(-param.kB*T)**3),
               w2,w2/(-param.kB*T),w2/(2*(-param.kB*T)**2),w2/(6*(-param.kB*T)**3),
               w3,w3/(-param.kB*T),w3/(2*(-param.kB*T)**2),w3/(6*(-param.kB*T)**3),
               w4,w4/(-param.kB*T),w4/(2*(-param.kB*T)**2),w4/(6*(-param.kB*T)**3),
               w5,w5/(-param.kB*T),w5/(2*(-param.kB*T)**2),w5/(6*(-param.kB*T)**3),
               w6,w6/(-param.kB*T),w6/(2*(-param.kB*T)**2),w6/(6*(-param.kB*T)**3)]
    eta1_features = [-dfdeta1,lapeta1]
    eta2_features = [-dfdeta2,lapeta2]
    feature_tensor = torch.stack(con_features+eta1_features+eta2_features,dim=0)
    
    
    
    return feature_tensor
    


# In[5]:


def thermal_features(tp,con,power,omega,xsource,ysource):
    lap = LaplacianOp()
    heat_source = HeatSource(power=power,omega=omega,xsource=xsource,ysource=ysource)
    laptp = lap(tp,param.dx,param.dy,param.T0)
    feat1 = con*laptp
    feat2 = (1-con)*laptp
    nx,ny = tp.size()
    feat3 = heat_source.heat_flux(nx,ny)
    feat1 = feat1.unsqueeze(0)
    feat2 = feat2.unsqueeze(0)
    feat3 = feat3.unsqueeze(0)
    all_features = torch.cat((feat1,feat2,feat3),0)
    return all_features


# In[6]:

Nx = int(sys.argv[1])
param.Nx = Nx

datadir = "dataset/"
model_path = 'saved_models/'
pkl_path = "sintering_30apr23_temp_dt%r_dx%r_Nx%r"%(param.dt,param.dx,param.Nx)
featuredir = "features"
full_datadir = os.path.join(datadir,pkl_path)
full_featuredir = os.path.join(full_datadir,featuredir) 
if not os.path.exists(full_datadir):
    sys.exit(1)
if not os.path.exists(full_featuredir):
    os.makedirs(full_featuredir)

start_frame = 0
end_frame = 1500

sfeatures = None
tfeatures = None
for idx in tqdm(range(start_frame, end_frame)):
    data = torch.load(os.path.join(full_datadir,"data%d.pkl"%(idx)))
    
    con = data['con']
    eta1 = data['eta1']
    eta2 = data['eta2']
    tp = data['tp']
    
    sfeatures = sintering_features(con,eta1,eta2,tp)
    tfeatures = thermal_features(tp,con,param.power,param.omega,param.xsource,param.ysource)
    
    torch.save(sfeatures,os.path.join(full_featuredir,"sfeatures%d.pt"%(idx)))
    torch.save(tfeatures,os.path.join(full_featuredir,"tfeatures%d.pt"%(idx)))
print("sfeatures.size",sfeatures.size())
print("tfeatures.size",tfeatures.size())

