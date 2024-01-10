import torch
import os
import sys
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.fft import fft2,fftshift,ifft2,ifftshift
import argparse

from parameters import param
from original_models.thermal_model import ThermalSingleTimestep
from original_models.sintering_model import SinterSingleTimestep

device = param.device

parser = argparse.ArgumentParser()
parser.add_argument("--tr_tmodel","-tt", type=str, default="", required=True)
parser.add_argument("--tr_smodel","-ts", type=str, default="", required=True)
parser.add_argument("--filespath","-f", type=str, default="", required=True)

args = parser.parse_args()

filespath = args.filespath
tr_tmodelname = args.tr_tmodel
tr_smodelname = args.tr_smodel

nstep = 200

tmodel = ThermalSingleTimestep()
smodel = SinterSingleTimestep()

tmodel.load_state_dict(torch.load(tr_tmodelname))
smodel.load_state_dict(torch.load(tr_smodelname))


delta_tp = None
qflux = None
all_data = []

eval_start = 1000
eval_end = 1200
with torch.no_grad():
    tmodel.eval()
    smodel.eval()

    diff_norm_con = []
    diff_norm_tp = []
    for data_idx in range(eval_start,eval_end):
        data_item = torch.load(os.path.join(filespath,"data%d.pkl"%(data_idx)))
        tp = data_item['tp']
        con,eta1,eta2 = data_item['con'],data_item['eta1'],data_item['eta2']
        
        gt_data_item = torch.load(os.path.join(filespath,"data%d.pkl"%(data_idx+nstep)))
        gt_tp = gt_data_item['tp']
        gt_con,gt_eta1,gt_eta2 = gt_data_item['con'],gt_data_item['eta1'],gt_data_item['eta2']
        

        for istep in range(nstep):
            tp_new, qflux = tmodel(tp,con)
            con_new, eta1_new, eta2_new = smodel(con,eta1,eta2,tp_new)
            
            delta_tp = tp_new - tp
            dcon = con_new - con
            deta1 = eta1_new - eta1
            deta2 = eta2_new - eta2
            
            del tp
            del con
            del eta1
            del eta2
            
            tp = tp_new
            con, eta1, eta2 = con_new, eta1_new, eta2_new
        
        diff_norm_con.append(((con-gt_con)**2).sum())
        diff_norm_tp.append(((tp-gt_tp)**2).sum())
    
    print("con_errors",diff_norm_con)
    print("tp errors",diff_norm_tp)
    print("MSE for con",np.mean(diff_norm_con),np.std(diff_norm_con))
    print("MSE for tp",np.mean(diff_norm_tp),np.std(diff_norm_tp))
    print("data",filespath)
    print("sim step",nstep,"eval start,end:",eval_start,eval_end)
    print("Models",tr_smodelname,tr_tmodelname)
    