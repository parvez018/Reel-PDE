import os
import sys
sys.path.append("../")


import argparse
from torch.fft import fft2,fftshift,ifft2,ifftshift
import scipy.fftpack as fp
from datetime import datetime

import torch
from parameters import param

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

# HILO_THRESHOLD = 0.1
# feature at index 8 is always sparse in value domain, so we make its high freq component zero
# thresholds used for 8jan23_Nx100 dataset
# freq_thresholds = [50,100,10,60,100,\
#                 0.5,1,10,1000,20,\
#                 1,1,1,5,1,\
#                 10,50,50,100,10,\
#                 30,10]
# delta_thresholds = [1,0.5,0.5]

# THRESHOLDS_FEATS = torch.tensor(freq_thresholds)
# THRESHOLDS_DELTAS = torch.tensor(delta_thresholds)


def compute_masks(datapath,start_id=0,end_id=MAX_FRAME,THRESHOLDS_FEATS=None,THRESHOLDS_DELTAS=None):
    feature_dir = os.path.join(datapath,DIRECTORY_FEATURE)
    deltas_dir = os.path.join(datapath,DIRECTORY_DELTAS)
    mask_dir = os.path.join(datapath,DIRECTORY_MASK)
    
    if not os.path.isdir(mask_dir):
        os.makedirs(mask_dir)
    
    
    for index in range(start_id,end_id):
        suffix = "%d.pkl"%index
        
        filepath = os.path.join(feature_dir,FILENAME_PREFIX_FEATURE+suffix)
        all_features = torch.load(filepath).to(param.device)
        
        filepath = os.path.join(deltas_dir,FILENAME_PREFIX_DELTAS+suffix)
        all_deltas = torch.load(filepath).to(param.device)
        
        
        # all_features = get_features_one_step(cur_data)
        all_features_fft = fft2(all_features)
        all_deltas_fft = fft2(all_deltas)
        
        # print("features_fft.size",all_features_fft.size())
        # print("deltas_fft.size",all_deltas_fft.size())
        
        
        # deltas_mask = torch.abs(all_deltas_fft)<HILO_THRESHOLD
        # features_mask = torch.abs(all_features_fft)<HILO_THRESHOLD
        abs_all_deltas_fft = torch.abs(all_deltas_fft)
        abs_all_features_fft = torch.abs(all_features_fft)
        
        deltas_mask = abs_all_deltas_fft>THRESHOLDS_DELTAS[:,None,None]
        features_mask = abs_all_features_fft>THRESHOLDS_FEATS[:,None,None]  
        
        # fidx = 0
        # print("\n\nFeature %d histogram"%(fidx), torch.unique(features_mask[fidx],return_counts=True))
        # print(abs_all_features_fft[fidx].min(),abs_all_features_fft[fidx].max())
        # print("cv mask histogram",torch.unique(deltas_mask[0],return_counts=True))
        # print("ci mask histogram",torch.unique(deltas_mask[1],return_counts=True))
        # print("eta mask histogram",torch.unique(deltas_mask[2],return_counts=True))
        # print("deltas_mask.size=",deltas_mask.size())
        # print("features_mask.size=",features_mask.size())     
        
        deltas_mask = torch.any(deltas_mask,dim=0)
        features_mask = torch.any(features_mask,dim=0)
        
        # print("after OR")
        # print("deltas_mask.size=",deltas_mask.size())
        # print("features_mask.size=",features_mask.size()) 
        
        # print("\ndeltas_mask_histogram",torch.unique(deltas_mask,return_counts=True)) 
        # print("features_mask_histogram",torch.unique(features_mask,return_counts=True)) 
        
        final_mask = torch.logical_or(deltas_mask,features_mask)
        
        # print("final_mask.size=",final_mask.size())
        
        print("final_mask_histogram",torch.unique(final_mask,return_counts=True))
        torch.save(final_mask,os.path.join(mask_dir, FILENAME_PREFIX_MASK+suffix))

        
    return OK

def get_cmd_inputs():
    if len(sys.argv)<5:
        print("python %s dim start end filepath dthfile fthfile"%(sys.argv[0]))
        sys.exit(1)
    return int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4],sys.argv[5]


if __name__=='__main__':
    datadir = "../dataset"
    
    
    # param.Nx,start_id,end_id,filepath, delta_threshold_file, feat_threshold_file = get_cmd_inputs()
    # dirname = os.path.join(datadir,filepath)
    # thresholds = torch.load(threshold_file)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nx", type=int, default=100, required=True)
    parser.add_argument("--start","-s", type=int, default=0, required=True)
    parser.add_argument("--end","-e", type=int, default=1000, required=True)
    parser.add_argument("--datapath","-p", type=str, default="", required=True)
    parser.add_argument("--delta_thresholds_file","-d", type=str, default="", required=True)
    parser.add_argument("--feat_thresholds_file","-f", type=str, default="", required=True)
    
    args = parser.parse_args()
    
    param.Nx = args.Nx
    start_id = args.start
    end_id = args.end
    dirname = args.datapath
    delta_tfile = args.delta_thresholds_file
    feat_tfile = args.feat_thresholds_file
    
    delta_thresholds = torch.load(delta_tfile).to(param.device)
    feat_thresholds = torch.load(feat_tfile).to(param.device)
    
    
    
    time0 = datetime.now()
    
    status = compute_masks(dirname,start_id,end_id,feat_thresholds,delta_thresholds)
    
    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    print("Status %d, OK=%d, ERROR=%d"%(status,OK,ERROR))
    print("Datafile source:",dirname)
    print("Start frame %d, end frame %d"%(start_id,end_id))
    print("Mask compute Time: %10f\n"%compute_time)