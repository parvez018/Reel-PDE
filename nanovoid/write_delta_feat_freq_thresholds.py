import torch
import sys
import os

# for void_Nx100_8jan23 dataset
feat_thresholds = [50,100,10,60,100,\
                0.5,1,10,1000,20,\
                1,1,1,5,1,\
                10,50,50,100,10,\
                30,10]
delta_thresholds = [1,0.5,0.5]

# for void_Nx300_8jan23 dataset
# feat_thresholds = [100,100,10,40,200,\
#                     1,1,300,1,5,\
#                     0.5,1,1,5,1,\
#                     100,50,50,100,100,\
#                     60,100]
# delta_thresholds = [5,5,1]

tensor_feat_th = torch.tensor(feat_thresholds)
tensor_delta_th = torch.tensor(delta_thresholds)

# parent_datadir = "../dataset/"
# delta_savepath = os.path.join(parent_datadir,sys.argv[1],"delta_thresholds.pkl")
# feat_savepath = os.path.join(parent_datadir,sys.argv[1],"feat_thresholds.pkl")

datadir = sys.argv[1]
delta_savepath = sys.argv[2]
feat_savepath = sys.argv[3]

torch.save(tensor_delta_th,delta_savepath)
torch.save(tensor_feat_th, feat_savepath)
