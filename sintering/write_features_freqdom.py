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


# In[3]:


device = param.device


# In[4]:


def write_features_freq(features):
    features_fft = fft2(features)
    return features_fft


# In[5]:

Nx = int(sys.argv[1])
datadir = "dataset/"
pkl_path = "sintering_30apr23_temp_dt%r_dx%r_Nx%r"%(param.dt,param.dx,Nx)
featuredir = "features"
full_datadir = os.path.join(datadir,pkl_path)
full_featuredir = os.path.join(full_datadir,featuredir) 
if not os.path.exists(full_datadir):
    sys.exit(1)
if not os.path.exists(full_featuredir):
    os.makedirs(full_featuredir)

fft_tfeatures = None

start_frame = 0
end_frame = 1400

for idx in tqdm(range(start_frame, end_frame)):
    tfeatures = torch.load(os.path.join(full_featuredir,"tfeatures%d.pt"%(idx)))
    fft_tfeatures = write_features_freq(tfeatures)
    
    torch.save(fft_tfeatures,os.path.join(full_featuredir,"ftfeatures%d.pt"%(idx)))
print("fft tfeature",fft_tfeatures.size())

