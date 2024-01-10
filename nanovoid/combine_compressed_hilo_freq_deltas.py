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
DIRECTORY_DELTAS = "deltas"
FILENAME_PREFIX_DELTAS = "deltas_"

# FILENAME_PREFIX_HIFREQ_DELTAS = "hifreq_deltas_"
# FILENAME_PREFIX_LOFREQ_DELTAS = "lofreq_deltas_"

FILENAME_COMPRESSED_HIFREQ_DELTAS = "compressed_hifreq_deltas_"
FILENAME_COMPRESSED_LOFREQ_DELTAS = "compressed_lofreq_deltas_"

matrixdir = "matrix"
FILENAME_MATRIX_ROW_PREFIX = "row_"


def combine_deltas(filedir,split,r):
    deltas_dir = os.path.join(filedir,DIRECTORY_DELTAS)
    start_frame = 0
    end_frame = split
    suffix = "start%d_end%d_cmp_%r.pkl"%(start_frame,end_frame,r)
    comb_hifreq_tensor = torch.load(os.path.join(deltas_dir,FILENAME_COMPRESSED_HIFREQ_DELTAS+suffix))
    comb_lofreq_tensor = torch.load(os.path.join(deltas_dir,FILENAME_COMPRESSED_LOFREQ_DELTAS+suffix))
    for start_frame in range(split,MAX_FRAME,split):
        end_frame = start_frame + split
        suffix = "start%d_end%d_cmp_%r.pkl"%(start_frame,end_frame,r)
        cur_hifreq_tensor = torch.load(os.path.join(deltas_dir,FILENAME_COMPRESSED_HIFREQ_DELTAS+suffix))
        comb_hifreq_tensor = torch.cat((comb_hifreq_tensor,cur_hifreq_tensor),0)
        
        cur_lofreq_tensor = torch.load(os.path.join(deltas_dir,FILENAME_COMPRESSED_LOFREQ_DELTAS+suffix))
        comb_lofreq_tensor = torch.cat((comb_lofreq_tensor,cur_lofreq_tensor),0)
    
    suffix = "Nx_%d_comp_%r.pt"%(param.Nx,r)
    savepath = os.path.join(deltas_dir,FILENAME_COMPRESSED_HIFREQ_DELTAS+suffix)
    torch.save(comb_hifreq_tensor,savepath)
    
    print("File saved as",savepath)
    
    savepath = os.path.join(deltas_dir,FILENAME_COMPRESSED_LOFREQ_DELTAS+suffix)
    torch.save(comb_lofreq_tensor,savepath)
    
    print("File saved as",savepath)
    
    print("compressed hifreq deltas",comb_hifreq_tensor.size())
    print("compressed lofreq deltas",comb_lofreq_tensor.size())
    
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