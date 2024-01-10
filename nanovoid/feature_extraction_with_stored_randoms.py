import torch

import sys
sys.path.append("../")
from diff_ops import LaplacianOp
from parameters import param


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
device = param.device

def log_with_mask(mat, eps=param.eps):
    mask = (mat < eps).detach()
    mat = mat.masked_fill(mask=mask, value=eps)
    return torch.log(mat)

# def project(projector,vec):
#     flat_vec = vec.clone().flatten()
#     prod = torch.matmul(projector,flat_vec)
#     return prod
    

def _get_pfields(all_data):
    cv = None
    ci = None
    eta = None
    # print("all data",all_data)
    for item in all_data:
        # print("item",item)
        current_cv = item['cv'].unsqueeze(0)
        current_ci = item['ci'].unsqueeze(0)
        current_eta = item['eta'].unsqueeze(0)
        if cv is None:
            cv = current_cv
            ci = current_ci
            eta = current_eta
        else:
            cv = torch.cat((cv,current_cv),0)
            ci = torch.cat((ci,current_ci),0)
            eta = torch.cat((eta,current_eta),0)
    
    return cv,ci,eta


def source_term_feature(eta,p_casc,R1,R2):
    # R1 = torch.rand_like(eta)
    # R2 = torch.rand_like(eta)
    mask = (eta >= 0.8) | (R1 > p_casc)
    R2.masked_fill_(mask=mask,value=0.0)
    return R2
    

def get_features_one_step(one_data,p_casc,R1,R2):
    cv = one_data['cv']
    ci = one_data['ci']
    eta = one_data['eta']
    
    cv = cv.to(device)
    ci = ci.to(device)
    eta = eta.to(device)
    
    R1 = R1.to(device)
    R2 = R2.to(device)
    
    lap = LaplacianOp()
    xdim, ydim = cv.size()
    
    
    
    h = (eta-1)
    h2 = (eta-1)**2
    cv_ci = cv*ci
    eta2_cv_ci = (eta**2)*cv_ci
    lap_h2 = lap(h2)
    j = eta**2
    log_cv = log_with_mask(cv)
    log_ci = log_with_mask(ci)
    log_cvi = log_with_mask(1-cv-ci)
    
    fs_mask = (1 - cv - ci < param.eps)
    random_feature = source_term_feature(eta,p_casc,R1,R2)
    
    
    # feature 0
    cur_feature = cv * (lap_h2).masked_fill(mask=fs_mask, value=0.0) # (Ev*Dv)/kBT, 0
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = cur_feature
    
    # feature 1
    cur_feature = (cv * lap(h2*(log_cv))).masked_fill(mask=fs_mask, value=0.0) # Dv
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 2
    cur_feature = (cv * lap(h2*(log_cvi))).masked_fill(mask=fs_mask, value=0.0) # -Dv
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 3
    cur_feature = lap(j*2*(cv-1)) * cv # Dv/kBT
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 4
    cur_feature = lap(lap(cv)) * cv # -Kv * Dv / kBT
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    
    
    # feature 5
    cur_feature = cv_ci # -r_bulk
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 6
    cur_feature = eta2_cv_ci # -r_surf
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    
    # random source term features
    # feature 7
    cur_feature = random_feature.clone() # bias*vg
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
        
    # print("feature for cv computed")
    
    
    # feature 8
    cur_feature = ci * (lap_h2).masked_fill(mask=fs_mask, value=0.0) # (Ei*Di)/kBT, 0
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 9
    cur_feature = ci * lap( h2 * ((log_ci).masked_fill(mask=fs_mask, value=0.0))) # Di, 0
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 10
    cur_feature = ci * lap( h2 * ((log_cvi).masked_fill(mask=fs_mask, value=0.0))) # -Di, 0
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 11
    cur_feature = (lap(j*2*(ci)) * ci) # Di/kBT
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 12
    cur_feature = lap(lap(ci)) * ci # -(Ki*Di) / kBT
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 13
    cur_feature = cv_ci # -r_bulk
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 14
    cur_feature = eta2_cv_ci # -r_surf
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    
    # random source term feature
    # feature 15
    cur_feature = random_feature.clone() # bias*vg
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # ci feature computation complete
    
    # eta feature computation begin
    
    # feature 16
    cur_feature = 2*(cv).masked_fill(mask=fs_mask, value=0.0)*(eta-1) # Ev, 0
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    
    # feature 17
    cur_feature = 2*(ci).masked_fill(mask=fs_mask, value=0.0)*(eta-1) # Ei, 0
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    

    # feature 18 # kBT
    cur_feature = 2*((cv*log_cv + ci*log_ci + (1 - cv - ci) * log_cvi)).masked_fill(mask=fs_mask, value=0.0) * (eta-1)
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 19
    cur_feature = ((cv-1)**2 + ci**2)*2*eta # 1
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # feature 20
    cur_feature = lap(eta) # kappa_eta
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    # random source term features
    # feature 21
    cur_feature = random_feature.clone() # bias*vg
    cur_feature = cur_feature.unsqueeze(0)
    batch_features = torch.cat((batch_features,cur_feature),0)
    
    
    return batch_features
    