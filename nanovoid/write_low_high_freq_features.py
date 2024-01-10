import os
import sys
sys.path.append("../")

from torch.fft import fft2,fftshift,ifft2,ifftshift
# import scipy.fftpack as fp
from datetime import datetime

import torch

import parameters
from parameters import param
# from feature_extraction import get_features_one_step
# from eta_mask import get_one_boundary_mask

device = param.device
seed = 4321
MAX_FRAME = 1000


ERROR = -1
OK = 1
DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"
DIRECTORY_MASK = "masks"

FILENAME_PREFIX_FRAME = "frame_"
FILENAME_PREFIX_DELTAS = "deltas_"
FILENAME_PREFIX_FEATURE = "features_"
FILENAME_PREFIX_MASK = "mask_"

FILENAME_PREFIX_LOFREQ_FEATURE = "lofreq_features_"
FILENAME_PREFIX_HIFREQ_FEATURE = "hifreq_features_"

FILENAME_PREFIX_LOFREQ_DELTAS = "lofreq_deltas_"
FILENAME_PREFIX_HIFREQ_DELTAS = "hifreq_deltas_"

HILO_THRESHOLD = 0.5


def compute_and_write_all_features(datapath,start_id=0,end_id=MAX_FRAME):
    feature_dir = os.path.join(datapath,DIRECTORY_FEATURE)
    mask_dir = os.path.join(datapath,DIRECTORY_MASK)

    for index in range(start_id,end_id):
        suffix = "%d.pkl"%index
        filepath = os.path.join(feature_dir,FILENAME_PREFIX_FEATURE+suffix)
        all_features = torch.load(filepath).to(device)
        
        filepath = os.path.join(mask_dir,FILENAME_PREFIX_MASK + suffix)
        tmask = torch.load(filepath).to(device)
        
        all_features_fft = fft2(all_features)
        
        # print("all_features_fft.size",all_features_fft.size(),tmask.size())
        lowfreq_features = all_features_fft.masked_fill(mask=tmask[None,:,:],value=torch.complex(torch.tensor(0.0),torch.tensor(0.0)))
        
        highfreq_features = all_features_fft.masked_fill(mask=~tmask[None,:,:],value=torch.complex(torch.tensor(0.0),torch.tensor(0.0)))
        
        lowfreq_features_restored = ifft2(lowfreq_features).real

        torch.save(lowfreq_features_restored,os.path.join(feature_dir,FILENAME_PREFIX_LOFREQ_FEATURE+suffix))
        torch.save(highfreq_features,os.path.join(feature_dir,FILENAME_PREFIX_HIFREQ_FEATURE+suffix))
        
    return OK

def get_cmd_inputs():
    if len(sys.argv)<5:
        print("python %s dim start end filepath"%(sys.argv[0]))
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
    print('Low high frequency feature compute Time: %10f\n'%compute_time)