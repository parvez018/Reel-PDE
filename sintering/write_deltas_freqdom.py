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
import argparse

from torch.fft import fft2,fftshift,ifft2,ifftshift
from parameters import param


# In[2]:


def write_deltas_freq(deltas):
    dcon = deltas['dcon']
    deta1 = deltas['deta1']
    deta2 = deltas['deta2']
    dtp = deltas['dtp']
    
    dtp_fft = fft2(dtp)
    return dcon,deta1,deta2,dtp_fft


# In[3]:

Nx = int(sys.argv[1])
datadir = "dataset/"
pkl_path = "sintering_30apr23_temp_dt%r_dx%r_Nx%r"%(param.dt,param.dx,Nx)
deltadir = "deltas"
full_datadir = os.path.join(datadir,pkl_path)
full_deltadir = os.path.join(full_datadir,deltadir) 


dtp_fft = None

start_frame = 0
end_frame = 1400

for idx in tqdm(range(start_frame, end_frame)):    
    deltas = torch.load(os.path.join(full_deltadir,"deltas%d.pkl"%(idx)))
    dcon,deta1,deta2,dtp_fft = write_deltas_freq(deltas)
    
    torch.save(dcon,os.path.join(full_deltadir,"fdcon%d.pt"%(idx)))
    torch.save(deta1,os.path.join(full_deltadir,"fdeta1%d.pt"%(idx)))
    torch.save(deta2,os.path.join(full_deltadir,"fdeta2%d.pt"%(idx)))
    torch.save(dtp_fft,os.path.join(full_deltadir,"fdtp%d.pt"%(idx)))
print("dtp_fft",dtp_fft.size())

