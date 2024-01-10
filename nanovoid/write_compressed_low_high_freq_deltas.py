import os
import sys
sys.path.append("../")

from torch.fft import fft2
from datetime import datetime

import torch

# import parameters
from parameters import param
# from feature_extraction import get_features_one_step
# from eta_mask import get_one_boundary_mask

seed = 4321
MAX_FRAME = 1000


ERROR = -1
OK = 1
DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"
FILENAME_PREFIX_FRAME = "frame_"
FILENAME_PREFIX_DELTAS = "deltas_"
# FILENAME_PREFIX_FEATURE = "features_"
# FILENAME_PREFIX_HIFREQ_FEATURE = "hifreq_features_"
# FILENAME_PREFIX_LOFREQ_FEATURE = "lofreq_features_"


FILENAME_PREFIX_HIFREQ_DELTAS = "hifreq_deltas_"
FILENAME_PREFIX_LOFREQ_DELTAS = "lofreq_deltas_"

# FILENAME_COMPRESSED_HIFREQ_FEATURES = "compressed_hifreq_featrues_"
# FILENAME_COMPRESSED_LOFREQ_FEATURES = "compressed_lofreq_featrues_"

FILENAME_COMPRESSED_HIFREQ_DELTAS = "compressed_hifreq_deltas_"
FILENAME_COMPRESSED_LOFREQ_DELTAS = "compressed_lofreq_deltas_"


matrixdir = "../matrix"
FILENAME_MATRIX_ROW_PREFIX = "row_"

def compress_and_write_features(datapath,start_frame,end_frame,old_dim,new_dim,r):
    deltas_dir = os.path.join(datapath,DIRECTORY_DELTAS)
    
    compressed_lofreq_deltas = None
    compressed_hifreq_deltas = None
    
    matrix_row_path_prefix = os.path.join(matrixdir,str(param.Nx),FILENAME_MATRIX_ROW_PREFIX)
    
    
    for nd in range(new_dim):
        row_path = os.path.join(matrix_row_path_prefix+"%d.pkl"%nd)
        row = torch.load(row_path)
        
        curr_lofreq_deltas_col_product = None
        curr_hifreq_deltas_col_product = None
        
        for index in range(start_frame,end_frame):
            suffix = "%d.pkl"%index
            
            lofreq_delta_filepath = os.path.join(datapath,DIRECTORY_DELTAS,FILENAME_PREFIX_LOFREQ_DELTAS+suffix)
            hifreq_delta_filepath = os.path.join(datapath,DIRECTORY_DELTAS,FILENAME_PREFIX_HIFREQ_DELTAS+suffix)
            
            if not (os.path.isfile(lofreq_delta_filepath) and os.path.isfile(hifreq_delta_filepath)):
                return ERROR
            
            
            lofreq_deltas = torch.load(lofreq_delta_filepath).flatten(start_dim=-2,end_dim=-1)
            hifreq_deltas = torch.load(hifreq_delta_filepath).flatten(start_dim=-2,end_dim=-1)
            
            product_lofreq = torch.matmul(lofreq_deltas,row).unsqueeze(0)
            product_hifreq = torch.matmul(hifreq_deltas,row.type(torch.complex64)).unsqueeze(0)
            
            if curr_lofreq_deltas_col_product is None:
                curr_lofreq_deltas_col_product = product_lofreq
            else:
                curr_lofreq_deltas_col_product = torch.cat((curr_lofreq_deltas_col_product,product_lofreq),0)
            
            if curr_hifreq_deltas_col_product is None:
                curr_hifreq_deltas_col_product = product_hifreq
            else:
                curr_hifreq_deltas_col_product = torch.cat((curr_hifreq_deltas_col_product,product_hifreq),0)
            
        if compressed_lofreq_deltas is None:
            compressed_lofreq_deltas = curr_lofreq_deltas_col_product
        else:
            compressed_lofreq_deltas = torch.cat((compressed_lofreq_deltas,curr_lofreq_deltas_col_product),-1)
        
        if compressed_hifreq_deltas is None:
            compressed_hifreq_deltas = curr_hifreq_deltas_col_product
        else:
            compressed_hifreq_deltas = torch.cat((compressed_hifreq_deltas,curr_hifreq_deltas_col_product),-1)
        
    suffix = "start%d_end%d_cmp_%r.pkl"%(start_frame,end_frame,r)

    torch.save(compressed_lofreq_deltas,os.path.join(deltas_dir,FILENAME_COMPRESSED_LOFREQ_DELTAS+suffix))
    torch.save(compressed_hifreq_deltas,os.path.join(deltas_dir,FILENAME_COMPRESSED_HIFREQ_DELTAS+suffix))
    
    return OK


def get_cmd_inputs():
    if len(sys.argv)<6:
        print("Argument length",len(sys.argv),sys.argv)
        print("python %s dim start end compression filepath"%(sys.argv[0]))
        sys.exit(1)
    return int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]),sys.argv[5]


if __name__=='__main__':
    datadir = "../dataset"
    param.Nx,start_frame,end_frame,compression,filepath = get_cmd_inputs()
    filename_pkl = os.path.join(datadir,filepath)
    # filename_pkl = os.path.join(datadir,'void_rapid_Nx%d_step%d_dt%r_dx%r'%(param.Nx,param.nstep,param.dt,param.dx))
    
    
    time0 = datetime.now()
    
    # status = compute_and_write_all_features(filename_pkl,start_id,end_id)
    old_dim = param.Nx*param.Nx
    new_dim = int(old_dim*compression)
    status = compress_and_write_features(filename_pkl,start_frame,end_frame,old_dim,new_dim,compression)
    
    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    print("Status %d, OK=%d, ERROR=%d"%(status,OK,ERROR))
    print("Datafile source:",filename_pkl)
    print("Start frame %d, end frame %d"%(start_frame,end_frame))
    print("Compression",compression)
    print('Higl low freq deltas compression compute Time: %10f\n'%compute_time)