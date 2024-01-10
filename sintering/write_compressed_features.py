'''
used for compressing all features with and without Fourier
'''


import torch
import os
import sys
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from parameters import param


# In[2]:


device = param.device


# In[3]:


matrixdir = "matrix"
FILENAME_MATRIX_ROW_PREFIX = "row_"


# In[4]:


def compress_and_write_features(features,r):
    total_features,Nx,Ny = features.size()
    new_Nx2 = int(r*(Nx**2))
    compressed_features = None
    
    matrix_row_path_prefix = os.path.join(matrixdir,str(Nx),FILENAME_MATRIX_ROW_PREFIX)
    
    comp_features = None
    flat_features = features.flatten(start_dim = -2, end_dim = -1).type(torch.float64)
    for i in range(total_features):
        projection = None
        for nd in range(new_Nx2):
            row_path = os.path.join(matrix_row_path_prefix+"%d.pt"%nd)
            row = torch.load(row_path).type(torch.float64)
            
            dotp = torch.matmul(row.squeeze(),flat_features[i].squeeze()).unsqueeze(0)
            if projection is None:
                projection = dotp
            else:
                projection = torch.cat((projection,dotp),0)
        
        projection = projection.unsqueeze(0)
        if comp_features is None:
            comp_features = projection
        else:
            comp_features = torch.cat((comp_features,projection),0)
    
    return comp_features


# In[5]:

Nx = int(sys.argv[1])
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])

param.Nx = Nx
param.Ny = Nx


datadir = "dataset/"
pkl_path = "sintering_30apr23_temp_dt%r_dx%r_Nx%r"%(param.dt,param.dx,Nx)
featuredir = "features"
r = 0.01
full_datadir = os.path.join(datadir,pkl_path)
full_featuredir = os.path.join(full_datadir,featuredir) 


comp_tfeatures = None
comp_sfeatures = None



for idx in tqdm(range(start_frame, end_frame)):   
    sfeatures = torch.load(os.path.join(full_featuredir,"sfeatures%d.pt"%(idx)))
    tfeatures = torch.load(os.path.join(full_featuredir,"tfeatures%d.pt"%(idx)))
    
    comp_sfeatures = compress_and_write_features(sfeatures,r)
    comp_tfeatures = compress_and_write_features(tfeatures,r)
    
    torch.save(comp_sfeatures,os.path.join(full_featuredir,"comp_sfeatures%d_r%r.pt"%(idx,r)))
    torch.save(comp_tfeatures,os.path.join(full_featuredir,"comp_tfeatures%d_r%r.pt"%(idx,r)))

print("comp_tfeatures",comp_tfeatures.size())
print("comp_sfeatures",comp_sfeatures.size())

