import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math

class FidelityLoss(nn.Module): 
    def __init__(self): 
        super(FidelityLoss, self).__init__()  

    def forward(self, p, q):
        if q.sum() == 0.: 
            return q.sum() * 0. # The gt is not labeled. Gradient and loss is 0. 

        loss = 1 - torch.sqrt(p*q + 1e-12).sum(dim=-1) 
        loss = loss.mean()  
        return loss 
    
class MultiClassFidelityLoss(nn.Module): 
    def __init__(self): 
        super(MultiClassFidelityLoss, self).__init__() 

    def forward(self, p, q): 
        if q.sum() == 0.: 
            return q.sum() * 0.

        loss = 1 - torch.sqrt(p*q + 1e-12) - torch.sqrt((1-p)*(1-q) + 1e-12)  
        loss = loss.mean() 
        return loss   

class ThurstoneQALoss(nn.Module):
    def __init__(self, std=0.01): 
        super(ThurstoneQALoss, self).__init__() 
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32)) 

    def forward(self, prediction, gt):
        B = gt.shape[0]
        prediction = prediction.unsqueeze(dim=0)
        gt = gt.unsqueeze(dim=0)   
        prediction = prediction - prediction.T     
        gt = gt - gt.T  

        triu_indices = torch.triu_indices(B, B, offset=1) # Discard the lower triangle.  
        prediction = prediction[triu_indices[0], triu_indices[1]] 
        gt = gt[triu_indices[0], triu_indices[1]] 
        
        prediction = (1 + torch.erf(prediction/self.std)) / 2
        gt = (torch.sign(gt) + 1) / 2
        
        loss = 1 - torch.sqrt(prediction*gt + 1e-12) - torch.sqrt((1-prediction)*(1-gt) + 1e-12) 
        loss = loss.mean() 
        return loss
    
class MarginRankingLoss(nn.Module):
    def __init__(self, margin=0.01): 
        super(MarginRankingLoss, self).__init__()  
        self.register_buffer("margin", torch.tensor(margin, dtype=torch.float32)) 

    def forward(self, prediction, gt):
        B = gt.shape[0]
        prediction = prediction.unsqueeze(dim=0)
        gt = gt.unsqueeze(dim=0)   
        prediction = prediction - prediction.T     
        gt = gt - gt.T  

        triu_indices = torch.triu_indices(B, B, offset=1) # Discard the lower triangle.  
        prediction = prediction[triu_indices[0], triu_indices[1]] 
        gt = gt[triu_indices[0], triu_indices[1]] 
        
        mask = (gt.abs() > self.margin/2) # Eliminate ambiguous pairs.  
        gt = torch.sign(gt)
        #mask = (gt != 0) # Eliminate ties. 
        
        prediction = prediction[mask]
        gt = gt[mask]

        if prediction.numel() == 0:  
            return torch.tensor(0.0, device=gt.device, dtype=gt.dtype, requires_grad=True)
        else: 
            loss = F.margin_ranking_loss(prediction, torch.zeros_like(prediction), gt, margin=self.margin) 
            return loss    
    
class ListRankingLoss(nn.Module):
    def __init__(self): 
        super(ListRankingLoss, self).__init__()  

    def forward(self, prediction, gt):
        prediction = prediction / (torch.sum(prediction) + 1e-12) 
        gt = gt / (torch.sum(gt) + 1e-12)   
        prediction = torch.log(prediction + 1e-12)    

        loss = -torch.mean(gt*prediction) # Normalize by the length of the list.
        return loss

class AdaptiveMarginRankingLoss(nn.Module):
    def __init__(self, tolerance_fator=0.95, eps=0.01): 
        super(AdaptiveMarginRankingLoss, self).__init__() 
        self.tolerance_factor = tolerance_fator 
        self.eps = eps  

    def forward(self, prediction, gt):
        B = gt.shape[0]
        prediction = prediction.unsqueeze(0) - prediction.unsqueeze(1)  # [B,B]
        gt = gt.unsqueeze(0) - gt.unsqueeze(1)  # [B,B]

        triu_indices = torch.triu_indices(B, B, offset=1)
        prediction = prediction[triu_indices[0], triu_indices[1]]
        gt = gt[triu_indices[0], triu_indices[1]]

        y = torch.sign(gt)
        mask = (y != 0) 
        if mask.sum() == 0:
            print("mask fail") 
            return gt.sum() * 0.0

        prediction = prediction[mask]
        y = y[mask]
        gt = gt[mask]

        adaptive_margin = torch.clamp(torch.abs(gt)*self.tolerance_factor, min=self.eps)
        loss = F.relu(-y * prediction + adaptive_margin)
        return loss.mean()
    
class ContrastiveRankingLoss(nn.Module): 
    def __init__(self, margin=None): 
        super(ContrastiveRankingLoss, self).__init__() 
        self.margin = margin

    def forward(self, prediction, gt):
        B, D = prediction.shape 

        sorted_indices = torch.argsort(gt, descending=True)  
        prediction = prediction[sorted_indices]  # (B, D) -> (B, D)  

        prediction = prediction / prediction.norm(dim=-1, keepdim=True) # (B, D) -> (B, D)   
        prediction = prediction @ prediction.T # (1) * (B, D) @ (D, B) -> (B, B)    

        idx = torch.arange(B)
        i_idx, j_idx, k_idx = torch.meshgrid(idx, idx, idx, indexing="ij")

        mask = (j_idx >= i_idx) & (k_idx > j_idx) 
        prediction_ij = prediction[i_idx[mask], j_idx[mask]]  
        prediction_ik = prediction[i_idx[mask], k_idx[mask]]
        
        target = torch.ones_like(prediction_ij)
        margin = self.margin if self.margin is not None else 1 / (torch.sqrt(torch.tensor(D, dtype=prediction.dtype)))     
        loss = F.margin_ranking_loss(prediction_ij, prediction_ik, target, margin) 
        return loss

class PLCCLoss(nn.Module): 
    def __init__(self): 
        super(PLCCLoss, self).__init__() 

    def forward(self, prediction, gt): 
        prediction_mean = torch.mean(prediction)
        gt_mean = torch.mean(gt)
        prediction_std = torch.std(prediction, unbiased=False)
        gt_std = torch.std(gt, unbiased=False)  
        
        covariance = torch.mean( (prediction-prediction_mean) * (gt - gt_mean) )
        plcc = covariance / (prediction_std * gt_std + 1e-12) 
        loss = 1-plcc  
        return loss
