import torch
import sys
sys.path.append("../")
import os

from datetime import datetime

from parameters import param

MAX_FRAME = 1000

DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"

FILENAME_COMPRESSED_DELTAS = "compressed_deltas_"
FILENAME_COMPRESSED_FEATURES = "compressed_features_"

ERROR = -1
OK = 1

def combine_features(filedir,split,r):
    feature_dir = os.path.join(filedir,DIRECTORY_FEATURE)
    start_frame = 0
    end_frame = split
    suffix = "start%d_end%d_cmp_%r.pkl"%(start_frame,end_frame,r)
    comb_feature_tensor = torch.load(os.path.join(feature_dir,FILENAME_COMPRESSED_FEATURES+suffix))
    for start_frame in range(split,MAX_FRAME,split):
        end_frame = start_frame + split
        suffix = "start%d_end%d_cmp_%r.pkl"%(start_frame,end_frame,r)
        cur_feature_tensor = torch.load(os.path.join(feature_dir,FILENAME_COMPRESSED_FEATURES+suffix))
        comb_feature_tensor = torch.cat((comb_feature_tensor,cur_feature_tensor),0)
        
    
    suffix = "Nx_%d_comp_%r.pt"%(param.Nx,r)
    savepath = os.path.join(feature_dir,FILENAME_COMPRESSED_FEATURES+suffix)
    torch.save(comb_feature_tensor,savepath)
    
    
    print("compressed features",comb_feature_tensor.size())
    print("File saved as",savepath)
    
    return OK
    

def combine_deltas(filedir,split,r):
    deltas_dir = os.path.join(filedir,DIRECTORY_DELTAS)
    start_frame = 0
    end_frame = split
    suffix = "start%d_end%d_cmp_%r.pkl"%(start_frame,end_frame,r)
    comb_deltas_tensor = torch.load(os.path.join(deltas_dir,FILENAME_COMPRESSED_DELTAS+suffix))
    for start_frame in range(split,MAX_FRAME,split):
        end_frame = start_frame + split
        suffix = "start%d_end%d_cmp_%r.pkl"%(start_frame,end_frame,r)
        cur_deltas_tensor = torch.load(os.path.join(deltas_dir,FILENAME_COMPRESSED_DELTAS+suffix))
        comb_deltas_tensor = torch.cat((comb_deltas_tensor,cur_deltas_tensor),0)
        
    
    suffix = "Nx_%d_comp_%r.pt"%(param.Nx,r)
    savepath = os.path.join(deltas_dir,FILENAME_COMPRESSED_DELTAS+suffix)
    torch.save(comb_deltas_tensor,savepath)
    
    
    print("compressed deltas size",comb_deltas_tensor.size())
    print("File saved as",savepath)
    
    return OK
    
    
if __name__=='__main__':
    param.Nx = int(sys.argv[1])
    split = int(sys.argv[2])
    comp = float(sys.argv[3])
    filepath = sys.argv[4]
    
    datadir = "../dataset"
    filedir = os.path.join(datadir,filepath)
    time0 = datetime.now()
    
    status = combine_deltas(filedir,split,comp)
    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    print("Deltas joining")
    print("Status %d, OK=%d, ERROR=%d"%(status,OK,ERROR))
    
    print("Source:",filedir)
    print("Compression",comp)
    print('Combination compute Time: %10f\n'%compute_time)