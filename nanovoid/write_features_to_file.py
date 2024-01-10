
import os
import sys
sys.path.append("../")

from torch.fft import fft2
from datetime import datetime

import torch

# import parameters
from parameters import param
from feature_extraction_with_stored_randoms import get_features_one_step
# from eta_mask import get_one_boundary_mask

seed = 4321
MAX_FRAME = 1000


ERROR = -1
OK = 1
DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"
FILENAME_PREFIX_FRAME = "frame_"
FILENAME_PREFIX_FEATURE = "features_"


def compute_and_write_all_features(datapath,start_id=0,end_id=MAX_FRAME):
    feature_dir = os.path.join(datapath,DIRECTORY_FEATURE)
    
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    
    for index in range(start_id,end_id):
        suffix = "%d.pkl"%index
        
        cur_framepath = os.path.join(datapath,FILENAME_PREFIX_FRAME+suffix)
        cur_r1path = os.path.join(datapath,FILENAME_PREFIX_FRAME+"%d_r1.pkl"%index)
        cur_r2path = os.path.join(datapath,FILENAME_PREFIX_FRAME+"%d_r2.pkl"%index)
        
        if not os.path.isfile(cur_framepath):
            return ERROR
        
        cur_frame = torch.load(cur_framepath)
        cur_r1 = torch.load(cur_r1path)
        cur_r2 = torch.load(cur_r2path)
        
        
        all_features = get_features_one_step(cur_frame,param.p_casc,cur_r1,cur_r2)
        torch.save(all_features,os.path.join(feature_dir,FILENAME_PREFIX_FEATURE+suffix))
                
    return OK

def get_cmd_inputs():
    if len(sys.argv)<5:
        print("python %s dim start end datapath"%(sys.argv[0]))
        sys.exit(1)
    return int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]



if __name__=='__main__':
    datadir = "../dataset"
    param.Nx,start_id,end_id,filepath = get_cmd_inputs()
    filename_pkl = os.path.join(datadir,filepath)
    
    
    time0 = datetime.now()
    status = compute_and_write_all_features(filename_pkl,start_id,end_id)
    
    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    print("Status %d, OK=%d, ERROR=%d"%(status,OK,ERROR))
    print("Datafile source:",filename_pkl)
    print("Start frame %d, end frame %d"%(start_id,end_id))
    print('Only Feature compute Time: %10f\n'%compute_time)