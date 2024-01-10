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
import argparse

sys.path.append("../")
from irradiation_model_features import IrradiationSingleTimestep
from parameters import param


device = param.device


seed = 4321

from set_seed import seed_torch


class IrradiationVideoDataset(Dataset):
    def __init__(self, features, deltas):
        super(IrradiationVideoDataset, self).__init__()
        self.features = features
        self.deltas = deltas
        
        self.skip_step = 1
        self.cnt = len(deltas)-self.skip_step

    def __getitem__(self, index):
        # index = self.all_data[idx]['step']
        return {
            'cdeltas':self.deltas[index],
            'cfeatures': self.features[index]
        }


    def __len__(self):
        return self.cnt



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
    
    
def get_cmd_inputs():
    if len(sys.argv)<7:
        print("python %s batch_size epochs dims compression learning_rate datadir modelname"%(sys.argv[0]))
        sys.exit(1)
    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    dim = int(sys.argv[3])
    compression = float(sys.argv[4])
    lrate = float(sys.argv[5])
    rid = int(sys.argv[6])
    return batch_size,epochs,dim,compression,lrate,rid


DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"
FILENAME_COMPRESSED_FEATURES = "compressed_features_"
FILENAME_COMPRESSED_DELTAS = "compressed_deltas_"


if __name__=='__main__':
    # seed_torch(125478)

    datadir = "../dataset/"
    modeldir = "../models/"
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--epoch", "-e", type=int, default=10)
    parser.add_argument("--Nx", type=int, default=100)
    parser.add_argument("--compression","-c", type=float, default=0.005, required=True)
    parser.add_argument("--learn_rate","-lr", type=float, default=0.0001, required=True)
    parser.add_argument("--filespath","-f", type=str, default="", required=True)
    parser.add_argument("--tr_modelname","-t", type=str, default="", required=True)
    parser.add_argument("--gt_modelname","-g", type=str, default="", required=True)
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    epoch = args.epoch
    param.Nx = args.Nx
    compression = args.compression
    lrate = args.learn_rate
    filespath = args.filespath
    tr_modelname = args.tr_modelname
    gt_modelname = args.gt_modelname
    skip_step = 1
    
    filedir = os.path.join(datadir,filespath)
    trained_modelname = os.path.join(modeldir,tr_modelname)
    gt_modelname = os.path.join(modeldir,gt_modelname)
    
    nprint = 50
        
    time0 = datetime.now()
    
    suffix = "Nx_%d_comp_%r.pt"%(param.Nx,compression)
    feature_dir = os.path.join(filedir,DIRECTORY_FEATURE)
    deltas_dir = os.path.join(filedir,DIRECTORY_DELTAS)
    
    feature_path = os.path.join(feature_dir,FILENAME_COMPRESSED_FEATURES+suffix)
    delta_path = os.path.join(deltas_dir,FILENAME_COMPRESSED_DELTAS+suffix)
    
    compressed_features = torch.load(feature_path)
    compressed_deltas = torch.load(delta_path)
    
    print("compressed_features=",compressed_features.size())
    print("compressed_deltas=",compressed_deltas.size())
    
    
    dataset = IrradiationVideoDataset(compressed_features,compressed_deltas)
    
    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    print("\n\nTime for data prep:",compute_time,"\n\n")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    ts_model = IrradiationSingleTimestep()
    # ts_model.load_state_dict(torch.load(gt_modelname))
    ts_model = ts_model.to(device)
    print("Before training initial model:")
    print(ts_model.state_dict())
    
    mse = nn.MSELoss(reduction='sum').to(device)

    lr = lrate
    optimizer = torch.optim.Adam(ts_model.parameters(), lr=lr)

    minloss = -1
    lambda_cv = 1
    lambda_ci = 1
    lambda_eta = 1

    for istep in range(epoch):
        loss = 0.0
        total_size = 0
        
        for batch in loader:            
            features = batch['cfeatures'].to(device)
            gt_deltas = batch['cdeltas'].to(device)

            cv_gt_deltas = gt_deltas[:,0]
            ci_gt_deltas = gt_deltas[:,1]
            eta_gt_deltas = gt_deltas[:,2]

            cv_delta, ci_delta, eta_delta = ts_model(features)
            
            cv_batch_loss = lambda_cv * mse(cv_delta,cv_gt_deltas)
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
            


    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    
    
    compare_parameters(trained_modelname,gt_modelname)
    
    
    print('Compute Time: %10f\n'%compute_time)
    
    print("trained model:",trained_modelname)
    print("trained from data:",filedir)
    