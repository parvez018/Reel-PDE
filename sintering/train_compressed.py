'''
used for training with -
1. Taylor approx + Fourier
2. Taylor approx + compression
3. Taylor approx + Fourier + Compression
Fourier transfomr on T
'''

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
from tqdm.auto import tqdm


from feature_models.sintering_model_features import SinterSingleTimestep
from feature_models.thermal_model_features import ThermalSingleTimestep
from parameters import param


device = param.device
torch.set_num_threads(1)


seed = 4321
MAX_FRAME = 1000
DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"
FILENAME_PREFIX_DELTAS = "deltas"
FILENAME_PREFIX_SFEAUTURE = "sfeatures"
FILENAME_PREFIX_TFEAUTURE = "tfeatures"

# from set_seed import seed_torch


class SinteringDataset(Dataset):
    def __init__(self,datapath,comp,start_skip,total_frame,inmemory=False):
        super().__init__()
        self.datapath = datapath
        self.start_skip = start_skip
        self.cnt = total_frame
        self.r = comp
        self.inmemory = inmemory
        if self.inmemory:
            self.all_sfeatures,self.all_tfeatures = self.load_features()
            self.all_dcon,self.all_deta1,self.all_deta2,self.all_dtp = self.load_deltas()

    def load_features(self):
        feature_dir = os.path.join(self.datapath,DIRECTORY_FEATURE)
        all_sfeatures = []
        all_tfeatures = []

        for i in range(self.start_skip,self.start_skip+self.cnt):
            sfeat = torch.load(os.path.join(feature_dir,"comp_sfeatures%d_r%r.pt"%(i,self.r)))
            tfeat = torch.load(os.path.join(feature_dir,"comp_tfeatures%d_r%r.pt"%(i,self.r)))
            all_sfeatures.append(sfeat)
            all_tfeatures.append(tfeat)

        return all_sfeatures,all_tfeatures
    
    def load_deltas(self):
        deltas_dir = os.path.join(self.datapath,DIRECTORY_DELTAS)
        all_dcon = []
        all_deta1 = []
        all_deta2 = []
        all_dtp = []
        for i in range(self.start_skip,self.start_skip+self.cnt):
            dcon = torch.load(os.path.join(deltas_dir,"comp_dcon%d_r%r.pt"%(i,self.r)))
            deta1 = torch.load(os.path.join(deltas_dir,"comp_deta1%d_r%r.pt"%(i,self.r)))
            deta2 = torch.load(os.path.join(deltas_dir,"comp_deta2%d_r%r.pt"%(i,self.r)))
            dtp = torch.load(os.path.join(deltas_dir,"comp_dtp%d_r%r.pt"%(i,self.r)))

            all_dcon.append(dcon)
            all_deta1.append(deta1)
            all_deta2.append(deta2)
            all_dtp.append(dtp)
        return all_dcon,all_deta1,all_deta2,all_dtp
    

    def __getitem__(self, index):
        sfeatures = None
        tfeatures = None
        dcon = None
        deta1 = None
        deta2 = None
        dtp = None

        if self.inmemory:
            sfeatures = self.all_sfeatures[index]
            tfeatures = self.all_tfeatures[index]
            dcon = self.all_dcon[index]
            deta1 = self.all_deta1[index]
            deta2 = self.all_deta2[index]
            dtp = self.all_dtp[index]
        else:
            index = index + self.start_skip
            suffix = "%d.pkl"%index
            data = torch.load(os.path.join(self.datapath,"data%d.pkl"%(index)))
            feature_dir = os.path.join(self.datapath,DIRECTORY_FEATURE)
            deltas_dir = os.path.join(self.datapath,DIRECTORY_DELTAS)
            
            sfeatures = torch.load(os.path.join(feature_dir,"comp_sfeatures%d_r%r.pt"%(index,self.r)))
            tfeatures = torch.load(os.path.join(feature_dir,"comp_tfeatures%d_r%r.pt"%(index,self.r)))
            dcon = torch.load(os.path.join(deltas_dir,"comp_dcon%d_r%r.pt"%(index,self.r)))
            deta1 = torch.load(os.path.join(deltas_dir,"comp_deta1%d_r%r.pt"%(index,self.r)))
            deta2 = torch.load(os.path.join(deltas_dir,"comp_deta2%d_r%r.pt"%(index,self.r)))
            dtp = torch.load(os.path.join(deltas_dir,"comp_dtp%d_r%r.pt"%(index,self.r)))

        return {
            'sfeatures': sfeatures,
            'tfeatures': tfeatures,
            'dcon':dcon,
            'deta1':deta1,
            'deta2':deta2,
            'dtp':dtp
        }


    def __len__(self):
        return self.cnt


def compare_parameters(trained_modelname,gt_modelname):
    print("Learned parameters:")
    print(torch.load(trained_modelname))
    
    print("Ground truth parameters:")
    print(torch.load(gt_modelname))
    

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--epoch", "-e", type=int, default=10)
    parser.add_argument("--Nx", type=int, default=100,required=True)
    parser.add_argument("--comp", type=float, default=0.005,required=True)
    parser.add_argument("--lrate","-lr", type=float, default=0.0001, required=True)
    parser.add_argument("--filespath","-f", type=str, default="", required=True)
    parser.add_argument("--tr_tmodel","-tt", type=str, default="", required=True)
    parser.add_argument("--gt_tmodel","-gt", type=str, default="", required=True)
    parser.add_argument("--tr_smodel","-ts", type=str, default="", required=True)
    parser.add_argument("--gt_smodel","-gs", type=str, default="", required=True)
    parser.add_argument("--start_skip","-sk", type=int, default=10, required=True)
    parser.add_argument("--total_frame","-tf", type=int, default=10, required=True)
    parser.add_argument("--nprint",type=int, default=100, required=False)
    parser.add_argument("--inmemory",type=bool, default=False, required=True)
    
    args = parser.parse_args()
    
    batch_size = args.batch_size
    epoch = args.epoch
    param.Nx = args.Nx
    lrate = args.lrate
    filespath = args.filespath
    tr_tmodelname = args.tr_tmodel
    gt_tmodelname = args.gt_tmodel
    tr_smodelname = args.tr_smodel
    gt_smodelname = args.gt_smodel
    comp = args.comp
    start_skip = args.start_skip
    total_frame = args.total_frame
    inmemory = args.inmemory
    
    nprint = args.nprint
    
    fourier_normalizer = (param.Nx*param.Nx)
    
    filename_pkl = filespath
        
    time0 = datetime.now()
    
    
    dataset = SinteringDataset(filespath,comp,start_skip,total_frame,inmemory)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    tmodel = ThermalSingleTimestep()
    smodel = SinterSingleTimestep()

    print("Before training initial model:")
    print("thermal model",tmodel.state_dict())
    print("sinter model",smodel.state_dict())
    
    mse = nn.MSELoss(reduction='sum').to(device)

    lr = lrate
    optimizer1 = torch.optim.Adam(tmodel.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(smodel.parameters(), lr=lr)

    minloss = -1
    epsilon = 1e-5

    time0 = datetime.now()
    lambda_t = 0.1
    lambda_s = 1

    min_sloss = -1
    min_tloss = -1
    
    for istep in tqdm(range(epoch)):
        loss = 0.0
        total_sinter_loss = 0.0
        total_thermal_loss = 0.0
        
        for batch in loader:
            sfeatures = batch['sfeatures'].to(device)
            tfeatures = batch['tfeatures'].to(device)
            dcon = batch['dcon'].to(device)
            deta1 = batch['deta1'].to(device)
            deta2 = batch['deta2'].to(device)
            dtp = batch['dtp'].to(device)

            pred_dcon, pred_deta1, pred_deta2 = smodel(sfeatures)
            pred_dtp = tmodel(tfeatures)

            loss_sinter = torch.sum((dcon-pred_dcon)**2 + (deta1-pred_deta1)**2 + (deta2-pred_deta2)**2)
            # uncomment the following line for training compressed without Fourier
            loss_tp = torch.sum((dtp-pred_dtp)**2)
            # uncomment the following line for training compressed with Fourier
            # loss_tp = (dtp-pred_dtp).abs().sum()/fourier_normalizer
            batch_loss = (lambda_s*loss_sinter + lambda_t*loss_tp)


            optimizer1.zero_grad()
            optimizer2.zero_grad()

            batch_loss.backward()

            optimizer1.step()
            optimizer2.step()

            loss += batch_loss.item()
            total_sinter_loss += loss_sinter.item()
            total_thermal_loss += loss_tp.item()

        
        if loss < minloss or minloss<0:
            minloss = loss
            min_tloss = total_thermal_loss
            min_sloss = total_sinter_loss
            print('Step: %d, minimal loss %r, tp %r, sinter %r'%(istep, minloss,min_tloss,min_sloss))
            torch.save(tmodel.state_dict(), tr_tmodelname)
            torch.save(smodel.state_dict(), tr_smodelname)
        
        
        if istep % nprint == 0:
            print('Step: %d, minimal loss %r, tp %r, sinter %r'%(istep, minloss,min_tloss,min_sloss))
            print('current loss:', loss)
        
        if loss<epsilon:
            break


    timeN = datetime.now()
    compute_time = (timeN-time0).total_seconds()
    
    compare_parameters(tr_smodelname,gt_smodelname)
    compare_parameters(tr_tmodelname,gt_tmodelname)
    
    print('Compute Time: %10f\n'%compute_time)

    print("file path",filespath)
    print("saved models",tr_tmodelname,tr_smodelname)
    print("learning rate",lrate)
    print("Parameters",vars(param))
    

    