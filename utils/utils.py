import numpy as np 
import os   
import cv2 as cv 
import torch 
import pandas as pd 
import subprocess 
import torch.nn.functional as F 
import gc 


def imagePaths(dataroot: str) -> list: 
    extensions = (".jpg", ".png", ".jpeg", ".bmp") 
    image_paths = [os.path.join(dataroot, f) for f in os.listdir(dataroot) if f.lower().endswith(extensions)]
    return image_paths

def imagePathsRecursive(root_dir, exts=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(exts):
                image_paths.append(os.path.join(root, file))
    return image_paths

def imread(path: str) -> np.ndarray: 
    img = cv.imread(path)  
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0   
    return img 

def resize_shorter_edge(x, short_edge=224, mode='bilinear'):
    if short_edge < 1: 
        return x

    # x: tensor [C, H, W] or [B, C, H, W]
    is_batched = x.dim() == 4
    if not is_batched:
        x = x.unsqueeze(0)  # make batch [1, C, H, W]
    
    _, _, H, W = x.shape
    scale = short_edge / min(H, W)
    
    if scale == 1: 
        return x if is_batched else x.squeeze(0)  
    
    new_H = int(round(H * scale))
    new_W = int(round(W * scale))
    
    x = F.interpolate(x, size=(new_H, new_W), mode=mode)
    if not is_batched:
        x = x.squeeze(0)
    return x

def resize_shorter_edge_np(image, target):
    h, w = image.shape[:2]
    if h < w:
        new_h = target
        new_w = int((w / h) * target)
    else:
        new_w = target
        new_h = int((h / w) * target)
    
    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    return resized


def randomHorizontalFlip(img: np.ndarray):
    if np.random.rand() < 0.5:
        # Handle CHW or HWC automatically
        if img.ndim == 3:  
            return np.ascontiguousarray(img[:, ::-1, :] if img.shape[0] != 3 else img[:, :, ::-1])
        elif img.ndim == 2:  # grayscale HxW
            return np.ascontiguousarray(img[:, ::-1])
    return img
    
def pad_to_divisor(img, divisor=32):
    H, W, C = img.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor

    padded_img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    return padded_img

def pad_to_divisor_torch(x, divisor=32):
    is_batched = x.dim() == 4
    if not is_batched:
        x = x.unsqueeze(0)  # make [1, C, H, W]
    
    _, _, H, W = x.shape
    pad_h = (divisor - H % divisor) % divisor
    pad_w = (divisor - W % divisor) % divisor

    # pad format: (left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    
    if not is_batched:
        x = x.squeeze(0)
    return x


def toTensor(x) -> torch.Tensor: 
    if isinstance(x, np.ndarray):  
        if len(x.shape) == 3: 
            x = x.transpose(2, 0, 1)
        x = torch.from_numpy(x) 
    else: 
        x = torch.tensor(x, dtype=torch.float32) # Torch will raise error if x cannot be converted. 
    
    return x 

def readCSV(path) -> dict: 
    df = pd.read_csv(path, sep="\t", header=None, names=["name", "mos", "distortion", "scene1", "scene2", "scene3"])
    # Normalize the MOS scale.  
    min_mos = df["mos"].min()
    max_mos = df["mos"].max()
    df["mos"] = (df["mos"] - min_mos) / (max_mos - min_mos)
    
    data = df.to_dict(orient='list') 
    return data

def unwrap_collate(batch):
    return batch[0]

def identity_collate(batch):
    return batch  