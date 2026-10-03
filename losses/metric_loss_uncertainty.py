import torch.nn as nn 

from losses.losses import * 

class MetricLossUncertainty(nn.Module): 
    def __init__(self): 
        super(MetricLossUncertainty, self).__init__()  
        self.PLCC_loss = PLCCLoss() 
        self.SRCC_loss = MarginRankingLoss()

        self.log_sigma_plcc = nn.Parameter(torch.tensor(0.0)) 
        self.log_sigma_srcc = nn.Parameter(torch.tensor(0.0)) 
        
    def forward(self, predictions, gts): 
        zipped = list(zip(predictions, gts))
        
        loss_plcc = [self.PLCC_loss(prediction, gt) for (prediction, gt) in zipped] 
        loss_srcc = [self.SRCC_loss(prediction, gt) for (prediction, gt) in zipped] 
        loss_plcc = torch.mean(torch.stack(loss_plcc)) 
        loss_srcc = torch.mean(torch.stack(loss_srcc)) 

        coeff_plcc = torch.exp(-2 * self.log_sigma_plcc) / 2  
        coeff_srcc = torch.exp(-2 * self.log_sigma_srcc) / 2  
        
        loss = (coeff_plcc*loss_plcc + self.log_sigma_plcc + 
                coeff_srcc*loss_srcc + self.log_sigma_srcc)  

        meta = {"q_plcc": loss_plcc.detach(),
                "q_srcc": loss_srcc.detach(),
                "coeff_plcc": coeff_plcc.detach(),
                "coeff_srcc": coeff_srcc.detach(),
                "loss": loss.detach()} 
        return loss, meta
    
