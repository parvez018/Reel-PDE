'''
used for training with Taylor series approx.
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


seed = 4321
MAX_FRAME = 1000
DIRECTORY_FEATURE = "features"
DIRECTORY_DELTAS = "deltas"
FILENAME_PREFIX_DELTAS = "deltas"
FILENAME_PREFIX_SFEAUTURE = "sfeatures"
FILENAME_PREFIX_TFEAUTURE = "tfeatures"


class SinteringDataset(Dataset):
    def __init__(self, datapath, start_skip, total_frame):
        super().__init__()
        self.datapath = datapath
        self.start_skip = start_skip
        
        ########## change later
        self.cnt = total_frame
        


    def __getitem__(self, index):
        index = index + self.start_skip
        suffix = "%d.pkl"%index
        data = torch.load(os.path.join(self.datapath,"data%d.pkl"%(index)))
        feature_dir = os.path.join(self.datapath,DIRECTORY_FEATURE)
        deltas_dir = os.path.join(self.datapath,DIRECTORY_DELTAS)
        
        sfeatures = torch.load(os.path.join(feature_dir,"sfeatures%d.pt"%index))
        tfeatures = torch.load(os.path.join(feature_dir,"tfeatures%d.pt"%index))
        deltas = torch.load(os.path.join(deltas_dir,"deltas%d.pkl"%(index)))
        con = data['con']
        dcon = deltas['dcon']
        deta1 = deltas['deta1']
        deta2 = deltas['deta2']
        dtp = deltas['dtp']
        
        return {
            'sfeatures': sfeatures,
            'tfeatures': tfeatures,
            'dcon':dcon,
            'deta1':deta1,
            'deta2':deta2,
            'dtp':dtp,
            'con':con
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
    print(torch.load(trained_modelname))
    
    print("Ground truth parameters:")
    print(torch.load(gt_modelname))
    

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--epoch", "-e", type=int, default=10)
    parser.add_argument("--Nx", type=int, default=100,required=True)
    parser.add_argument("--lrate","-lr", type=float, default=0.0001, required=True)
    parser.add_argument("--filespath","-f", type=str, default="", required=True)
    parser.add_argument("--tr_tmodel","-tt", type=str, default="", required=True)
    parser.add_argument("--gt_tmodel","-gt", type=str, default="", required=True)
    parser.add_argument("--tr_smodel","-ts", type=str, default="", required=True)
    parser.add_argument("--gt_smodel","-gs", type=str, default="", required=True)
    parser.add_argument("--start_skip","-sk", type=int, default=10, required=True)
    parser.add_argument("--total_frame","-tf", type=int, default=10, required=True)
    
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
    start_skip = args.start_skip
    total_frame = args.total_frame
    
    fourier_normalizer = (param.Nx*param.Nx)
    
    filename_pkl = filespath
        
    time0 = datetime.now()
    
    
    dataset = SinteringDataset(filespath,start_skip,total_frame)
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
    nprint = 100

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
            con = batch['con'].to(device)

            pred_con, pred_deta1, pred_deta2 = smodel(sfeatures)
            pred_dtp = tmodel(tfeatures)

            loss_sinter = torch.sum((dcon-pred_con)**2 + (deta1-pred_deta1)**2 + (deta2-pred_deta2)**2)
            loss_tp = torch.sum((dtp-pred_dtp)**2)
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
    
    print("file path:",filespath)
    print("trained models:",tr_smodelname,tr_tmodelname)
    print("learning rate",lrate)
    