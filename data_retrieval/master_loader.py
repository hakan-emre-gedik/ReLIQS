import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.utils import * 
from data_retrieval.Annotated_Dataset import Annotated_Dataset

class MasterLoader: 
    def __init__(self,
                 dataset_names: list,
                 samples_per_dataset: list,
                 device: torch.device, 
                 options: dict): 
        self.dataset_names = dataset_names
        self.samples_per_dataset = samples_per_dataset 
        self.device = device 
        self.indices_per_dataset = [0] 
        
        for sample_count in samples_per_dataset: # Precreate the batch indices per dataset for decoupling. 
            previous_indice = self.indices_per_dataset[-1] 
            if sample_count == 0: 
                continue 
            else: 
                self.indices_per_dataset.append(previous_indice+sample_count)    
        
        self.train_dataloaders = [] 
        self.val_dataloaders   = [] 
        self.test_dataloaders  = []  

        for (dataset_name, batch_size) in zip(dataset_names, samples_per_dataset): 
            train_dataset = Annotated_Dataset(dataset_name,
                                              split_id=int(options["split"]),
                                              mode="train") 
            val_dataset   = Annotated_Dataset(dataset_name,
                                              split_id=int(options["split"]),
                                              mode="val") 
            test_dataset  = Annotated_Dataset(dataset_name,
                                              split_id=int(options["split"]),
                                              mode="test") 
            
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler   = DistributedSampler(val_dataset, shuffle=False) 
            test_sampler  = DistributedSampler(test_dataset, shuffle=False)
            
            if batch_size > 0: 
                train_loader = DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          sampler=train_sampler,
                                          num_workers=1, 
                                          drop_last=True,
                                          pin_memory=True,
                                          collate_fn=identity_collate)
            else: 
                train_loader = None
            
            val_loader   = DataLoader(val_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      sampler=val_sampler,
                                      num_workers=1, 
                                      drop_last=False,
                                      pin_memory=False,
                                      collate_fn=identity_collate) 
            test_loader  = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      sampler=test_sampler,
                                      num_workers=1, 
                                      drop_last=False,
                                      pin_memory=False, 
                                      collate_fn=identity_collate) 

            if train_loader: 
                self.train_dataloaders.append(train_loader) 
            self.val_dataloaders.append(val_loader) 
            self.test_dataloaders.append(test_loader)
            
        self.epoch_length = max([len(loader) for loader in self.train_dataloaders])
    
    def TrainLoaders(self): 
        return self.train_dataloaders  

    def ValLoaders(self): 
        return self.val_dataloaders
    
    def TestLoaders(self): 
        return self.test_dataloaders  

    def setEpoch(self, epoch): 
        for loader in self.train_dataloaders: 
            loader.sampler.set_epoch(epoch)

    def epochLength(self): 
        return self.epoch_length  