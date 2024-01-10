
import os
import sys
sys.path.append("../")

from torch.fft import fft2
from datetime import datetime

import torch

import parameters
from parameters import param
from feature_extraction import get_features_one_step
# from eta_mask import get_one_boundary_mask

seed = 4321
MAX_FRAME = 1000


ERROR = -1
OK = 1
DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"
FILENAME_PREFIX_FRAME = "frame_"
FILENAME_PREFIX_DELTAS = "deltas_"
FILENAME_PREFIX_FEAUTURE = "features_"



def compute_and_write_all_deltas(datapath,start_id=0,end_id=MAX_FRAME):
    feature_dir = os.path.join(datapath,DIRECTORY_FEATURE)
    deltas_dir = os.path.join(datapath,DIRECTORY_DELTAS)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    if not os.path.isdir(deltas_dir):
        os.makedirs(deltas_dir)
    for index in range(start_id,end_id):
        suffix = "%d.pkl"%index
        cur_filepath = os.path.join(datapath,FILENAME_PREFIX_FRAME+suffix)
        next_filepath = os.path.join(datapath,FILENAME_PREFIX_FRAME+"%d.pkl"%(index+1))
        if not os.path.isfile(cur_filepath):
            print("error reading file",cur_filepath)
            return ERROR
        
        cur_data = torch.load(cur_filepath)
        next_data = torch.load(next_filepath)
        
    
    
        cv_delta = next_data['cv']-cur_data['cv']
        ci_delta = next_data['ci']-cur_data['ci']
        eta_delta = next_data['eta']-cur_data['eta']
        
        cv_delta = cv_delta.unsqueeze(0)
        ci_delta = ci_delta.unsqueeze(0)
        eta_delta = eta_delta.unsqueeze(0)
        
        
        all_deltas = torch.cat((cv_delta,ci_delta,eta_delta),0)
        torch.save(all_deltas,os.path.join(deltas_dir,FILENAME_PREFIX_DELTAS+suffix))
        
    return OK

def get_cmd_inputs():
    if len(sys.argv)<4:
        print("python %s dim start end"%(sys.argv[0]))
        sys.exit(1)
    return int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]



if __name__=='__main__':
    datadir = "../dataset"
    param.Nx,start_id,end_id,filepath = get_cmd_inputs()
    filename_pkl = os.path.join(datadir,filepath)
    
    
    time0 = datetime.now()
    
    status = compute_and_write_all_deltas(filename_pkl,start_id,end_id)
    
    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    print("Status %d, OK=%d, ERROR=%d"%(status,OK,ERROR))
    print("Datafile source ===>>>",filename_pkl)
    print("Start frame %d, end frame %d"%(start_id,end_id))
    print('Delta compute Time: %10f\n'%compute_time)