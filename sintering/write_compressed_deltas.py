'''
used for compressing all deltas with and without Fourier
'''


import torch
import os
import sys
from tqdm.auto import tqdm
import numpy as np

from parameters import param


# In[ ]:


device = param.device


# In[ ]:


matrixdir = "matrix"
FILENAME_MATRIX_ROW_PREFIX = "row_"


# In[ ]:


def compress_and_write_deltas(features,r):
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


# In[ ]:

Nx = int(sys.argv[1])
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])


param.Nx = Nx
param.Ny = Nx

datadir = "dataset/"
pkl_path = "sintering_30apr23_temp_dt%r_dx%r_Nx%r"%(param.dt,param.dx,Nx)
deltadir = "deltas"
full_datadir = os.path.join(datadir,pkl_path)
full_deltadir = os.path.join(full_datadir,deltadir) 

r = 0.01

if not os.path.exists(full_datadir):
    sys.exit(1)
if not os.path.exists(full_deltadir):
    os.makedirs(full_deltadir)

comp_dcon, comp_deta1, comp_deta2, comp_dtp = None, None, None, None




for idx in tqdm(range(start_frame, end_frame)):   
    deltas = torch.load(os.path.join(full_deltadir,"deltas%d.pkl"%(idx)))
    comp_dcon = compress_and_write_deltas(deltas['dcon'].unsqueeze(0),r)
    comp_deta1 = compress_and_write_deltas(deltas['deta1'].unsqueeze(0),r)
    comp_deta2 = compress_and_write_deltas(deltas['deta2'].unsqueeze(0),r)
    comp_dtp = compress_and_write_deltas(deltas['dtp'].unsqueeze(0),r)
    
    torch.save(comp_dcon,os.path.join(full_deltadir,"comp_dcon%d_r%r.pt"%(idx,r)))
    torch.save(comp_deta1,os.path.join(full_deltadir,"comp_deta1%d_r%r.pt"%(idx,r)))
    torch.save(comp_deta2,os.path.join(full_deltadir,"comp_deta2%d_r%r.pt"%(idx,r)))
    torch.save(comp_dtp,os.path.join(full_deltadir,"comp_dtp%d_r%r.pt"%(idx,r)))
print("comp_dcon",comp_dcon.size())
print("comp_deta1",comp_deta1.size())
print("comp_deta2",comp_deta2.size())
print("comp_dtp",comp_dtp.size())

