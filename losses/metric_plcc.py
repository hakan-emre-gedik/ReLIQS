import torch.nn as nn 

from losses.losses import * 

class Metric_PLCC(nn.Module): 
    def __init__(self): 
        super(Metric_PLCC, self).__init__()  
        self.PLCC_loss = PLCCLoss() 
        
    def forward(self, predictions, gts): 
        loss_plcc = 0.0 
        zipped = list(zip(predictions, gts))
        
        for (prediction, gt) in zipped: 
            loss_plcc += self.PLCC_loss(prediction, gt) 
        
        loss_plcc /= len(zipped)  

        loss = loss_plcc
        meta = {"q_plcc": loss_plcc.detach()} 
        
        return loss, meta