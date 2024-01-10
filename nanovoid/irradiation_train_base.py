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


device = param.device
torch.set_num_threads(1)


seed = 4321
MAX_FRAME = 1000
DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"
FILENAME_PREFIX_FRAME = "frame_"
FILENAME_PREFIX_DELTAS = "deltas_"
FILENAME_PREFIX_FEAUTURE = "features_"

from set_seed import seed_torch


class IrradiationVideoDataset(Dataset):
    def __init__(self, datapath):
        super(IrradiationVideoDataset, self).__init__()
        self.datapath = datapath
        self.skip_step = 1
        self.start_skip = 0
        
        ########## change later
        self.cnt = MAX_FRAME
        


    def __getitem__(self, index):
        suffix = "%d.pkl"%index
        feature_dir = os.path.join(self.datapath,DIRECTORY_FEATURE)
        deltas_dir = os.path.join(self.datapath,DIRECTORY_DELTAS)
        
        all_deltas = torch.load(os.path.join(deltas_dir,FILENAME_PREFIX_DELTAS+suffix))
        all_features = torch.load(os.path.join(feature_dir,FILENAME_PREFIX_FEAUTURE+suffix))
        
        # pfields_current = torch.load(os.path.join(self.datapath,FILENAME_PREFIX_FRAME+suffix))
        # pfields_next = torch.load(os.path.join(self.datapath,FILENAME_PREFIX_FRAME+"%d.pkl"%(index+1)))
        
        # cv = pfields_current['cv']
        # ci = pfields_current['ci']
        # eta = pfields_current['eta']
        
        # cv_new = pfields_next['cv']
        # ci_new = pfields_next['ci']
        # eta_new = pfields_next['eta']
        
        
        return {
            'deltas':all_deltas,
            'features': all_features
            # 'cv':cv,
            # 'ci':ci,
            # 'eta':eta,
            # 'cv_new':cv_new,
            # 'ci_new':ci_new,
            # 'eta_new':eta_new
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
    # seed_torch(125478)

    datadir = "../dataset/"
    modeldir = "../models/"
    

    
    param.nstep = 5000
   
    
    batch_size, epoch, param.Nx, filedir, tmodel, gtmodel, lrate = get_cmd_inputs()
    skip_step = 1
    
    filename_pkl = os.path.join(datadir,filedir)
    trained_modelname = os.path.join(modeldir,tmodel)
    gt_modelname = os.path.join(modeldir,gtmodel)
    
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
            # print("batch:\n",batch)
            features = batch['features'].to(device)
            gt_deltas = batch['deltas'].to(device)
            
            # cv = batch['cv'].to(device)
            # ci = batch['ci'].to(device)
            # eta = batch['eta'].to(device)
            
            # cv_new = batch['cv_new'].to(device)
            # ci_new = batch['ci_new'].to(device)
            # eta_new = batch['eta_new'].to(device)
            
            cv_gt_deltas = gt_deltas[:,0]
            ci_gt_deltas = gt_deltas[:,1]
            eta_gt_deltas = gt_deltas[:,2]
            
            # cv_pred, ci_pred, eta_pred = ts_model(features,cv,ci,eta)
            # cv_batch_loss = lambda_cv * mse(cv_pred,cv_new)
            # ci_batch_loss = lambda_ci * mse(ci_pred, ci_new)
            # eta_batch_loss = lambda_eta * mse(eta_pred,eta_new)
            
            cv_delta,ci_delta,eta_delta = ts_model(features)
            cv_batch_loss = lambda_cv * mse(cv_delta, cv_gt_deltas)
            ci_batch_loss = lambda_ci * mse(ci_delta, ci_gt_deltas)
            eta_batch_loss = lambda_eta * mse(eta_delta, eta_gt_deltas)
            
            
            batch_loss = cv_batch_loss + ci_batch_loss + eta_batch_loss
                       

            optimizer.zero_grad()

            batch_loss.backward()

            optimizer.step()

            loss += batch_loss.item()

        
        if loss < minloss or minloss<0:
            minloss = loss
            print('Get minimal loss')
            torch.save(ts_model.state_dict(), trained_modelname)
        
        
        if istep % nprint == 0:
            print("Epoch:",istep,"current minloss:",minloss)
            print('current loss:', loss)
            # print(ts_model.state_dict())
        
        if loss<epsilon:
            break
            


    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    
    
    compare_parameters(trained_modelname,gt_modelname)
    
    
    print('Compute Time: %10f\n'%compute_time)
    
    print("trained model:",trained_modelname)
    print("trained from data:",filename_pkl)
    print("simulation parameters:",vars(param))
    