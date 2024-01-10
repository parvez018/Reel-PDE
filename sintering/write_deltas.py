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

Nx = int(sys.argv[1])
param.Nx = Nx



datadir = "dataset/"
pkl_path = "sintering_30apr23_temp_dt%r_dx%r_Nx%r"%(param.dt,param.dx,param.Nx)
deltadir = "deltas"
full_datadir = os.path.join(datadir,pkl_path)
full_deltadir = os.path.join(full_datadir,deltadir) 
if not os.path.exists(full_datadir):
    sys.exit(1)
if not os.path.exists(full_deltadir):
    os.makedirs(full_deltadir)

print("sys.argv",sys.argv,Nx,param.Nx,full_datadir,full_deltadir)

start_frame = 0
end_frame = 1499

drho, deta1, deta2, dtp = None, None, None, None
for idx in tqdm(range(start_frame, end_frame)):
    cur_data = torch.load(os.path.join(full_datadir,"data%d.pkl"%(idx)))
    next_data = torch.load(os.path.join(full_datadir,"data%d.pkl"%(idx+1)))
    
    dcon = next_data['con'] - cur_data['con']
    deta1 = next_data['eta1'] - cur_data['eta1']
    deta2 = next_data['eta2'] - cur_data['eta2']
    dtp = next_data['tp'] - cur_data['tp']
    
    deltas = {'dcon':dcon,'deta1':deta1,'deta2':deta2,'dtp':dtp}
    
    torch.save(deltas,os.path.join(full_deltadir,"deltas%d.pkl"%(idx)))
print("dcon",dcon.size())
print("deta1",deta1.size())
print("deta2",deta2.size())
print("dtp",dtp.size())

