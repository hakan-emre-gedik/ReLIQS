import torch.nn as nn 

from losses.losses import * 

class MetricLoss(nn.Module): 
    def __init__(self, tradeoff=1.0): 
        super(MetricLoss, self).__init__()  
        self.tradeoff = tradeoff 
        self.PLCC_loss = PLCCLoss() 
        self.SRCC_loss = MarginRankingLoss()
        
    def forward(self, predictions, gts): 
        loss_srcc, loss_plcc = 0.0, 0.0 
        zipped = list(zip(predictions, gts))
        
        for (prediction, gt) in zipped: 
            loss_srcc += self.SRCC_loss(prediction, gt) 
            loss_plcc += self.PLCC_loss(prediction, gt) 
        loss_srcc /= len(zipped)
        loss_plcc /= len(zipped)  

        loss = (1-self.tradeoff)*loss_srcc + self.tradeoff*loss_plcc
        meta = {"q_plcc": loss_plcc.detach(), "q_srcc": loss_srcc.detach()} 
        
        return loss, meta
    
