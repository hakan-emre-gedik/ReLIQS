import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.transforms import Normalize 

from utils.utils import * 
import clip 


import torch
import torch.nn as nn
import torch.nn.functional as F

class IQA_Model(nn.Module): 
    def __init__(self, device: torch.device, options: dict): 
        super(IQA_Model, self).__init__()  
        self.options = options
        self.device = device 
        self.num_attributes = options["num_attributes"] 

        # Instantiate the CLIP model. 
        self.image_preprocess = Normalize( mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711) ) # CLIP   
        self.CLIP, _ = clip.load(options["model_name"], device=device)
        self.CLIP = self.CLIP.float() 

        # Detect the output dimension. 
        dummy = torch.rand(1, 3, 224, 224).to(device)
        with torch.no_grad():  
            dummy = self.CLIP.encode_image(dummy)
        self.ve_dim = dummy.shape[-1]   
        self.ve_scaling = torch.tensor(self.ve_dim).sqrt().to(device)  
        
        # Multiscale pooling and quality mapping.
        self.log_T_ms = nn.Parameter(torch.zeros(self.num_attributes))
        self.Q_ms = nn.Parameter(torch.randn(self.num_attributes, self.ve_dim))  
        self.K_ms = nn.Linear(self.ve_dim, self.num_attributes*self.ve_dim)
        self.V_ms = nn.Linear(self.ve_dim, self.num_attributes*self.ve_dim) 

        self.log_T_quality = nn.Parameter(torch.zeros(self.num_attributes)) 
        self.quality_queries = nn.Parameter(torch.randn(self.num_attributes, 2, self.ve_dim))
        self.logit_aspect_weights = nn.Parameter(torch.zeros(self.num_attributes)) 
    
    def freeze_vision_encoder(self): 
        for param in self.CLIP.parameters(): 
            param.requires_grad_(False) 

    def unfreeze_vision_encoder(self): 
        for param in self.CLIP.parameters(): 
            param.requires_grad_(True) 

    def freeze_model(self): 
        for param in self.parameters(): 
            param.requires_grad_(False) 
    
    def unfreeze_pooling(self): 
        self.log_T_ms.requires_grad_(True) 
        self.Q_ms.requires_grad_(True)
        self.K_ms.weight.requires_grad_(True)
        self.K_ms.bias.requires_grad_(True)
        self.V_ms.weight.requires_grad_(True)
        self.V_ms.bias.requires_grad_(True)
        self.log_T_quality.requires_grad_(True)
        self.quality_queries.requires_grad_(True)
        self.logit_aspect_weights.requires_grad_(True)
        

    def forward_clip(self, x: torch.Tensor):
        if len(x.shape) == 4: 
            # If (B C H W) is passed, only 1 patch exists. 
            x = x.unsqueeze(dim=1) # (B 1 C H W) 

        B, N, C, H, W = x.shape # (B, N, C, H, W) = (B, N, 3, 224, 224) N:= # of patches from an image. B:= # of images. 

        x = x.contiguous().view(-1, C, H, W) # (B*N, C, H, W) = (B*N, 3, 224, 224). N was 3 for LIQE, we adopted 9.   
        x = self.image_preprocess(x)  

        # Obtain image features. 
        image_features = self.CLIP.encode_image(x) # (B*N, C, H, W) -> (B*N, D), D:=512   
        image_features = image_features.view(B, N, -1) # (B*N, D) -> (B, N, D) 
        return image_features
    
    
    def forward_pooling(self, image_features: torch.Tensor, weights: torch.Tensor, scale_counts: torch.Tensor): 
        B, _, D = image_features.shape
        
        image_features_ = [] 
        weights_ = []

        indices = [0]   
        for i in range(scale_counts.shape[-1]): 
            lo = indices[-1] 
            hi = lo + int(scale_counts[i].item())  
            indices.append(hi)  

        for i in range(len(indices)-1):
            down = indices[i] 
            up = indices[i+1] 
            image_features_scale = image_features[:, down:up, :]  # (B, N, D) 
            weights_scale = weights[:, down:up] # (B, N) 
            image_features_.append(image_features_scale) 
            weights_.append(weights_scale) 
            
        scale_features = [(features*weight[..., None]).sum(dim=1) for (features, weight) in zip(image_features_, weights_)] # [(B, D)] # SALIENCY POOLING

        query_ms = self.Q_ms / self.Q_ms.norm(dim=-1, keepdim=True) # (A, D) 
        T_ms = self.log_T_ms.exp() # (A)  
        T_quality = self.log_T_quality.exp() # (A)  
        quality_scale = torch.tensor([0.0, 1.0],
                                     dtype=T_quality.dtype,
                                     device=T_quality.device)
        aspect_weights = F.softmax(self.logit_aspect_weights, dim=0) 
        
        quality_prediction_scale = [] 

        for i, scale_feature in enumerate(scale_features): 
            key = self.K_ms(scale_feature).reshape(B, self.num_attributes, D) # (B, A, D)  
            value = self.V_ms(scale_feature).reshape(B, self.num_attributes, D) # (B, A, D)
            #key = self.K_ms(scale_feature).reshape(B, 1, D) # (B, A, D)  
            #value = self.V_ms(scale_feature).reshape(B, 1, D) # (B, A, D)
            
            key = key / key.norm(dim=-1, keepdim=True)  
            #key = key / self.ve_scaling
            value = value / value.norm(dim=-1, keepdim=True) 
            
            attn = (T_ms[None, :, None] * query_ms[None, ...] * key).sum(dim=-1) # ( (1, A, 1) * (1, A, D) * (B, A, D) ).sum(-1) -> (B, A) ?  
            attn = F.softmax(attn, dim=-1) # (B, A) -> (B, A)   

            quality_queries = self.quality_queries / self.quality_queries.norm(dim=-1, keepdim=True) # (A, 2, D) 
            quality_queries_0, quality_queries_1 = torch.chunk(quality_queries, 2, dim=1) # (A, 1, D), (A, 1, D)    
            
            zero_logits = (T_quality[None, :, None] * quality_queries_0[:, 0, :][None, ...] * value).sum(-1) # ( (1, A, 1) * (1, A, D) * (B, A, D) ).sum(-1) -> (B, A) ?
            one_logits = (T_quality[None, :, None] * quality_queries_1[:, 0, :][None, ...] * value).sum(-1) # ( (1, A, 1) * (1, A, D) * (B, A, D) ).sum(-1) -> (B, A) ?
            logits = torch.stack((zero_logits, one_logits), dim=-1) # (B, A) | (B, A) -> (B, A, 2) 
            probs = F.softmax(logits, dim=-1) # (B, A, 2) -> (B, A, 2)   
            
            quality_predictions = (probs * quality_scale[None, None, :]).sum(dim=-1) # ( (B, A, 2) * (1, 1, 2) ).sum(-1) -> (B, A)   
            combined_weights = attn * aspect_weights[None, :] # (B, A) * (1, A) -> (B, A) 
            combined_weights = combined_weights / combined_weights.sum(dim=-1, keepdim=True) 
            quality_prediction = (combined_weights * quality_predictions).sum(dim=-1) # ( (B, A) * (B, A) ).sum(-1) -> (B)
            quality_prediction_scale.append(quality_prediction)
        
        quality = torch.stack(quality_prediction_scale, dim=-1).mean(dim=-1)   
        return quality
    
    def forward(self,
                x: torch.Tensor,
                weights: torch.Tensor,
                scale_counts: torch.Tensor):

        image_features = self.forward_clip(x)   
        quality_prediction = self.forward_pooling(image_features, weights, scale_counts) 

        return quality_prediction  
