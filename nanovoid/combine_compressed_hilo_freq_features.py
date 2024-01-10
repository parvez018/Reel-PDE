import torch
import sys
sys.path.append("../")
import os

from datetime import datetime

from parameters import param

seed = 4321
MAX_FRAME = 1000


ERROR = -1
OK = 1
DIRECTORY_FEATURE = "features"
FILENAME_PREFIX_HIFREQ_FEATURE = "hifreq_features_"
FILENAME_PREFIX_LOFREQ_FEATURE = "lofreq_features_"


FILENAME_COMPRESSED_HIFREQ_FEATURES = "compressed_hifreq_features_"
FILENAME_COMPRESSED_LOFREQ_FEATURES = "compressed_lofreq_features_"


matrixdir = "matrix"
FILENAME_MATRIX_ROW_PREFIX = "row_"


def combine_features(filedir,split,r):
    feature_dir = os.path.join(filedir,DIRECTORY_FEATURE)
    start_frame = 0
    end_frame = split
    suffix = "start%d_end%d_cmp_%r.pkl"%(start_frame,end_frame,r)
    comb_hifreq_tensor = torch.load(os.path.join(feature_dir,FILENAME_COMPRESSED_HIFREQ_FEATURES+suffix))
    comb_lofreq_tensor = torch.load(os.path.join(feature_dir,FILENAME_COMPRESSED_LOFREQ_FEATURES+suffix))
    for start_frame in range(split,MAX_FRAME,split):
        end_frame = start_frame + split
        suffix = "start%d_end%d_cmp_%r.pkl"%(start_frame,end_frame,r)
        cur_hifreq_tensor = torch.load(os.path.join(feature_dir,FILENAME_COMPRESSED_HIFREQ_FEATURES+suffix))
        comb_hifreq_tensor = torch.cat((comb_hifreq_tensor,cur_hifreq_tensor),0)
        
        cur_lofreq_tensor = torch.load(os.path.join(feature_dir,FILENAME_COMPRESSED_LOFREQ_FEATURES+suffix))
        comb_lofreq_tensor = torch.cat((comb_lofreq_tensor,cur_lofreq_tensor),0)
    
    suffix = "Nx_%d_comp_%r.pt"%(param.Nx,r)
    savepath = os.path.join(feature_dir,FILENAME_COMPRESSED_HIFREQ_FEATURES+suffix)
    torch.save(comb_hifreq_tensor,savepath)
    print("File saved as",savepath)
    
    savepath = os.path.join(feature_dir,FILENAME_COMPRESSED_LOFREQ_FEATURES+suffix)
    torch.save(comb_lofreq_tensor,savepath)
    
    print("File saved as",savepath)
    
    print("compressed hifreq features",comb_hifreq_tensor.size())
    print("compressed lofreq features",comb_lofreq_tensor.size())
    
    return OK
    
    
if __name__=='__main__':
    param.Nx = int(sys.argv[1])
    split = int(sys.argv[2])
    comp = float(sys.argv[3])
    filepath = sys.argv[4]
    
    datadir = "../dataset"
    filedir = os.path.join(datadir,filepath)
    # filedir = os.path.join(datadir,'void_rapid_Nx%d_step%d_dt%r_dx%r'%(param.Nx,param.nstep,param.dt,param.dx))
    
    time0 = datetime.now()
    
    status = combine_features(filedir,split,comp)
    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    print("Feature joining took", compute_time)
    print("Status %d, OK=%d, ERROR=%d"%(status,OK,ERROR))
    
    
    print("Source:",filedir)
    print("Compression",comp)
    print('Combination compute Time: %10f\n'%compute_time)