import torch
import torch.nn as nn
import sys
import argparse
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import numpy as np

import math
import os
import shutil

sys.path.append("../")

from irradiation_model_features_noiseless_riv_piv import IrradiationSingleTimestep
from parameters import param


# param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = param.device
torch.set_num_threads(1)


seed = 4321
MAX_FRAME = 1000
DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"
FILENAME_PREFIX_FRAME = "frame_"
FILENAME_PREFIX_DELTAS = "deltas_"
FILENAME_PREFIX_FEAUTURE = "features_"

FILENAME_PREFIX_LOFREQ_FEATURE = "lofreq_features_"
FILENAME_PREFIX_HIFREQ_FEATURE = "hifreq_features_"

FILENAME_PREFIX_LOFREQ_DELTAS = "lofreq_deltas_"
FILENAME_PREFIX_HIFREQ_DELTAS = "hifreq_deltas_"

from set_seed import seed_torch


class IrradiationVideoDataset(Dataset):
    def __init__(self, datapath):
        super(IrradiationVideoDataset, self).__init__()
        self.datapath = datapath
        self.start_skip = 0
        
        ########## change later
        self.cnt = MAX_FRAME
        


    def __getitem__(self, index):
        suffix = "%d.pkl"%index
        feature_dir = os.path.join(self.datapath,DIRECTORY_FEATURE)
        deltas_dir = os.path.join(self.datapath,DIRECTORY_DELTAS)
        
        # all_deltas = torch.load(os.path.join(deltas_dir,FILENAME_PREFIX_DELTAS+suffix))
        # all_features = torch.load(os.path.join(feature_dir,FILENAME_PREFIX_FEAUTURE+suffix))
        
        hifreq_deltas = torch.load(os.path.join(deltas_dir,FILENAME_PREFIX_HIFREQ_DELTAS+suffix))
        lofreq_deltas = torch.load(os.path.join(deltas_dir,FILENAME_PREFIX_LOFREQ_DELTAS+suffix))
        
        hifreq_features = torch.load(os.path.join(feature_dir,FILENAME_PREFIX_HIFREQ_FEATURE+suffix))
        lofreq_features = torch.load(os.path.join(feature_dir,FILENAME_PREFIX_LOFREQ_FEATURE+suffix))

        
        return {
            'hi_features':hifreq_features,
            'lo_features':lofreq_features,
            'hi_deltas':hifreq_deltas,
            'lo_deltas':lofreq_deltas
        }


    def __len__(self):
        return self.cnt


def get_cmd_inputs():
    if len(sys.argv)<8:
        print("python %s batch_size epochs dims datadir model gtmodel lrate"%(sys.argv[0]))
        sys.exit(1)
    return int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4],sys.argv[5],sys.argv[6],float(sys.argv[7])
    

def compare_parameters(trained_modelname,gt_modelname):
    print("Learned parameters:")
    learned_model = IrradiationSingleTimestep()
    learned_model.load_state_dict(torch.load(trained_modelname,map_location=device))
    learned_model.eval()
    print(learned_model.state_dict())
    
    print("Ground truth parameters:")
    gt_model = IrradiationSingleTimestep()
    gt_model.load_state_dict(torch.load(gt_modelname,map_location=device))
    gt_model.eval()
    print(gt_model.state_dict())
    
    



if __name__=='__main__':

    datadir = "../dataset/"
    modeldir = "../models/"
    
    
    # batch_size, epoch, param.Nx, filedir, tmodel, gtmodel, lrate = get_cmd_inputs()
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--epoch", "-e", type=int, default=10)
    parser.add_argument("--Nx", type=int, default=100,required=True)
    parser.add_argument("--learn_rate","-lr", type=float, default=0.0001, required=True)
    parser.add_argument("--filespath","-f", type=str, default="", required=True)
    parser.add_argument("--tr_modelname","-t", type=str, default="", required=True)
    parser.add_argument("--gt_modelname","-g", type=str, default="", required=True)
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    epoch = args.epoch
    param.Nx = args.Nx
    lrate = args.learn_rate
    filespath = args.filespath
    tr_modelname = args.tr_modelname
    gt_modelname = args.gt_modelname
    
    fourier_normalizer = (param.Nx*param.Nx)
    
    filename_pkl = os.path.join(datadir,filespath)
    trained_modelname = os.path.join(modeldir,tr_modelname)
    gt_modelname = os.path.join(modeldir,gt_modelname)
    
    nprint = 100
        
    time0 = datetime.now()
    
    
    dataset = IrradiationVideoDataset(filename_pkl)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    ts_model = IrradiationSingleTimestep()
    
    # # for debuggin purpuose, we first start with the ground truth model and check
    # # if the learned parameters deviate from here, if so then there is bug in the training code
    # ts_model.load_state_dict(torch.load(gt_modelname,map_location=device))
    
    ts_model = ts_model.to(device)
    
    initial_model = ts_model.state_dict()
    print("Before training initial model:")
    print(initial_model)
    
    mse = nn.MSELoss(reduction='sum').to(device)

    lr = lrate
    optimizer = torch.optim.Adam(ts_model.parameters(), lr=lr)

    minloss = -1
    lambda_cv = 1
    lambda_ci = 1
    lambda_eta = 1
    
    epsilon = 1e-5
    for istep in range(epoch):
        loss = 0.0
        total_size = 0
        
        for batch in loader:
            hifreq_features = batch['hi_features'].to(device)
            lofreq_features = batch['lo_features'].to(device)
            
            gt_hifreq_deltas = batch['hi_deltas'].to(device)
            gt_lofreq_deltas = batch['lo_deltas'].to(device)
            
            cv_gt_hdelta = gt_hifreq_deltas[:,0]
            ci_gt_hdelta = gt_hifreq_deltas[:,1]
            eta_gt_hdelta = gt_hifreq_deltas[:,2]
            
            cv_gt_ldelta = gt_lofreq_deltas[:,0]
            ci_gt_ldelta = gt_lofreq_deltas[:,1]
            eta_gt_ldelta = gt_lofreq_deltas[:,2]
            
            
            cv_hdelta, ci_hdelta, eta_hdelta = ts_model(hifreq_features)
            cv_ldelta, ci_ldelta, eta_ldelta = ts_model(lofreq_features)
            

            cv_hloss = (cv_hdelta-cv_gt_hdelta).abs().sum()/fourier_normalizer
            ci_hloss = (ci_hdelta-ci_gt_hdelta).abs().sum()/fourier_normalizer
            eta_hloss = (eta_hdelta-eta_gt_hdelta).abs().sum()/fourier_normalizer
            
            cv_lloss = (cv_ldelta-cv_gt_ldelta).abs().sum()
            ci_lloss = (ci_ldelta-ci_gt_ldelta).abs().sum()
            eta_lloss = (eta_ldelta-eta_gt_ldelta).abs().sum()
            
            cv_batch_loss = lambda_cv*(cv_hloss+cv_lloss)
            ci_batch_loss = lambda_ci*(ci_hloss+ci_lloss)
            eta_batch_loss = lambda_eta*(eta_hloss+eta_lloss)
            batch_loss = cv_batch_loss + ci_batch_loss + eta_batch_loss
                       

            optimizer.zero_grad()

            batch_loss.backward()

            optimizer.step()

            loss += batch_loss.item()

        
        if loss < minloss or minloss<0:
            minloss = loss
            print('Get minimal loss',istep)
            torch.save(ts_model.state_dict(), trained_modelname)
        
        
        if istep % nprint == 0:
            print("Epoch:",istep,"current minloss:",minloss)
            print('current loss:', loss)
            # print(ts_model.state_dict())
        
        if loss<epsilon:
            break
            
                
        if istep % 100 == 0:
            torch.save(ts_model.state_dict(), trained_modelname + "-" + str(istep))
            


    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    
    
    compare_parameters(trained_modelname,gt_modelname)
    
    
    print('Compute Time: %10f\n'%compute_time)
    
    print("trained model:",trained_modelname)
    print("trained from data:",filename_pkl)
    