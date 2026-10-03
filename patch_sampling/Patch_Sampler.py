import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.transforms import Normalize
import timm  

import tiny_clip 
from utils.utils import * 


class LayerNorm2d(nn.Module):
    """
    LayerNorm over channel dimension for NCHW tensors.
    Normalizes per spatial location (H,W) across channels C.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B,H,W,C) -> LN(C) -> (B,C,H,W)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DWSeparableConv(nn.Module):
    """
    Depthwise 3x3 (spatial mixing) + pointwise 1x1 (channel mixing),
    with LN2d placed after the pointwise conv (ideal for stability).
    """
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, eps: float = 1e-6):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(out_ch, eps=eps)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7):
        super(ResidualBlock, self).__init__()
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.dwconv_1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.dwconv_2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x) 
        x = self.dwconv_1(x)
        x = F.gelu(x) 
        x = self.dwconv_2(x)    
        x = x * self.gamma.view(1, -1, 1, 1)  

        return residual + x

class UpStageV1(nn.Module):
    """
    One pyramid stage:
      upsample x2 -> 1x1 proj (C_in -> C_out) -> N ConvNeXt v1 blocks at C_out
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        num_blocks: int = 1,
        mlp_ratio: float = 2.0,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1, bias=True)
        self.blocks = nn.Sequential(*[
            ResidualBlock(dim=c_out, kernel_size=kernel_size)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = self.blocks(x)
        return x

class ProgressiveConvNeXtV1Decoder(nn.Module):
    """
    Progressive upsampler decoder for ViT-16 dense features.

    Input:  (B, D, Ht, Wt) e.g., (B,512,14,14)
    Output: (B, out_ch, H, W) if out_hw provided (upsamples logits to out_hw)
    """
    def __init__(
        self,
        in_dim: int = 512,
        blocks_per_stage=(1, 1, 1, 1),
        mlp_ratio: float = 2.0,
        kernel_size: int = 7,
        layer_scale_init_value: float = 1e-6,
        out_ch: int = 1,
    ):
        super().__init__()
        channel_schedule = [in_dim // (2**i) for i in range(1, len(blocks_per_stage)+1, 1) ]   

        stages = []
        c = in_dim
        for c_next, nb in zip(channel_schedule, blocks_per_stage):
            stages.append(
                UpStageV1(
                    c_in=c,
                    c_out=c_next,
                    num_blocks=nb,
                    mlp_ratio=mlp_ratio,
                    kernel_size=kernel_size,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
            c = c_next
        self.stages = nn.ModuleList(stages)
        self.head = nn.Conv2d(c, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for st in self.stages:
            x = st(x)
        logits = self.head(x)  # (B,out_ch,h,w)
        return logits


class SaliencyPatchSampler(nn.Module): 
    def __init__(self,   
                 patch_size: int,
                 num_scales: int,
                 min_short_edge: int,  
                 device: torch.device): 
        super(SaliencyPatchSampler, self).__init__() 
        self.patch_size = patch_size   
        self.num_scales = num_scales
        self.min_short_edge = min_short_edge

        arch = "TinyCLIP-ViT-8M-16-Text-3M"
        model, _, _ = tiny_clip.create_model_and_transforms(arch, pretrained='YFCC15M', device=device)
        self.saliency_model = model.visual
        self.image_preprocess = Normalize( mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711) ) # CLIP 
 
        self.ln_post = nn.LayerNorm(self.saliency_model.embed_dim) 
        
        #self.projection = nn.Sequential(nn.Conv2d(self.saliency_model.embed_dim, self.saliency_model.embed_dim, 5, 1, 2, groups=self.saliency_model.embed_dim),
        #                                nn.GELU(),
        #                                nn.Conv2d(self.saliency_model.embed_dim, self.saliency_model.embed_dim, 5, 1, 2, groups=self.saliency_model.embed_dim),
        #                                nn.GELU(),
        #                                nn.Conv2d(self.saliency_model.embed_dim, 1, 1))
         
        self.projection = ProgressiveConvNeXtV1Decoder(in_dim=self.saliency_model.embed_dim) 
        self.log_T = nn.Parameter(torch.zeros(())) 

        # Hook 
        self.saliency_map = None

    def freeze(self): 
        for param in self.parameters(): 
            param.requires_grad_(False) 
    
    def freeze_ve(self): 
        for param in self.saliency_model.parameters():  
            param.requires_grad_(False)

    def unfreeze(self): 
        for param in self.parameters():  
            param.requires_grad_(True)

    def forward_single_scale(self,
                             image: torch.Tensor,
                             saliency_map: torch.Tensor,
                             num_patches: int,
                             stride_ratio: float,
                             random: bool):  
        C, H, W = image.shape
        H_s, W_s = saliency_map.shape 
        if not (H_s == H and W_s == W):
            saliency_map = F.interpolate(saliency_map[None, None, ...], (H, W), mode="bilinear", align_corners=False)[0, 0, ...]   
        
        if random: 
            max_y = max(H - self.patch_size, 0)
            max_x = max(W - self.patch_size, 0)
            ys = torch.randint(0, max_y + 1, (num_patches,), device=image.device)
            xs = torch.randint(0, max_x + 1, (num_patches,), device=image.device)

            y_idx = torch.arange(self.patch_size, device=image.device).view(1, -1, 1)
            x_idx = torch.arange(self.patch_size, device=image.device).view(1, 1, -1)
            ys = ys.view(-1, 1, 1) + y_idx
            xs = xs.view(-1, 1, 1) + x_idx

            with torch.no_grad():  
                patches = image[:, ys, xs].permute(1, 0, 2, 3)  # (N, C, P, P)
            
            saliency_map = saliency_map[ys, xs] # (N, P, P) 
            weights = saliency_map.sum(dim=(1, 2))
            weights = weights / (weights.sum() + 1e-12)

        else: 
            stride_y = self.patch_size * stride_ratio
            stride_x = self.patch_size * stride_ratio
            n_h = int((H - self.patch_size) / stride_y) + 1
            n_w = int((W - self.patch_size) / stride_x) + 1

            ys = torch.linspace(0, max(H - self.patch_size, 0), n_h, device=image.device)
            xs = torch.linspace(0, max(W - self.patch_size, 0), n_w, device=image.device)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")

            coords = torch.stack([yy.flatten(), xx.flatten()], dim=1).round().long()
            
            y_idx = torch.arange(self.patch_size, device=image.device).view(1, -1, 1)
            x_idx = torch.arange(self.patch_size, device=image.device).view(1, 1, -1)
            ys = coords[:, 0].view(-1, 1, 1) + y_idx
            xs = coords[:, 1].view(-1, 1, 1) + x_idx

            with torch.no_grad(): 
                patches = image[:, ys, xs].permute(1, 0, 2, 3)
            
            saliency_map = saliency_map[ys, xs] # (N, P, P) 
            weights = saliency_map.sum(dim=(1, 2))
            weights = weights / (weights.sum() + 1e-12)     


        return patches.detach(), weights 
    
    def forward(self,
                image: torch.Tensor,
                num_patches: list,
                stride_ratio: float, 
                random: bool): 
        C, H, W = image.shape 
        # Multi-scale representation. 
        if min(H, W) < self.min_short_edge: 
            image = resize_shorter_edge(image, self.min_short_edge, mode="bilinear")
            C, H, W = image.shape 
        
        # Predict the IQA-saliency map.
        image_resized = resize_shorter_edge(image, short_edge=self.min_short_edge, mode="area") 
        image_resized = image_resized[None, ...]
        _, C, H_r, W_r = image_resized.shape
        pad_w, pad_h = (16-W_r%16)%16, (16-H_r%16)%16  
        image_resized = F.pad(image_resized, (0, pad_w, 0, pad_h), mode="reflect") 
        _, C, H_padded, W_padded = image_resized.shape
        num_H_blocks, num_W_blocks = H_padded//16, W_padded//16   

        saliency_image = self.image_preprocess(image_resized) 
        saliency_features = self.saliency_model(saliency_image, dense_features=True) # (1, N, D)  
        saliency_features = self.ln_post(saliency_features) # (1, N, D) 
        saliency_features = saliency_features.view(1, num_H_blocks, num_W_blocks, -1).permute(0, 3, 1, 2) # (1, D, H, W) 
        saliency_map = self.projection(saliency_features)[0, 0, :H_r, :W_r] 
        T = self.log_T.exp() 
        saliency_map = F.softmax(saliency_map.flatten()/T, dim=0).view(H_r, W_r)

        # Hook 
        self.saliency_map = saliency_map.detach().cpu() 

        scale_resolutions = torch.linspace(min(H, W), self.min_short_edge, self.num_scales).round().long().tolist()   
        image_scales = [] 
        weights_scales = [] 
        scale_counts = [] 
        for idx, scale_resolution in enumerate(scale_resolutions): 
            image_scale = resize_shorter_edge(image, short_edge=scale_resolution, mode="area")

            image_scale, weights_scale = self.forward_single_scale(image_scale, saliency_map, num_patches[idx], stride_ratio, random)
            image_scales.append(image_scale)
            weights_scales.append(weights_scale)  
            scale_counts.append(weights_scale.shape[-1]) 

        patches = torch.cat(image_scales, dim=0) # (K, C, patch_size, patch_size) 
        weights = torch.cat(weights_scales , dim=0) # (K) 
        scale_counts = torch.tensor(scale_counts, dtype=torch.long, device=patches.device) # (num_scales) 
        return patches, weights, scale_counts  
