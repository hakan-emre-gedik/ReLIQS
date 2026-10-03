from torch.utils.data import Dataset 
from torchvision.transforms import RandomHorizontalFlip
import os 

from utils.utils import * 

class Annotated_Dataset(Dataset): 
    def __init__(self,
                 dataset_name: str,
                 split_id: int,
                 mode="train"): 
        super(Annotated_Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.mode = mode
        self.resize_resolution = 512 
        self.patch_size = 224 

        self.path_dataset = os.path.join("..", "vlm_iqa_datasets", dataset_name) 
        path_labels_root  = os.path.join(self.path_dataset, "splits", str(split_id))    
        
        # Get the label paths under the specified split folder. 
        for filename in os.listdir(path_labels_root): 
            if "train" in filename.lower(): 
                path_labels_train = os.path.join(path_labels_root, filename) 
            elif "val" in filename.lower():
                path_labels_val = os.path.join(path_labels_root, filename) 
            elif "test" in filename.lower(): 
                path_labels_test = os.path.join(path_labels_root, filename) 
            else: 
                pass 
        
        # Read data. 
        if mode == "train":  
            train_set = readCSV(path_labels_train)
            self.dataset = train_set
        elif mode == "val": 
            val_set = readCSV(path_labels_val) 
            self.dataset = val_set 
        elif mode == "test":     
            test_set = readCSV(path_labels_test)
            self.dataset = test_set
        else: 
            raise Exception(f"For IQA dataset, the mode: {mode} is not defined.") 


    def __getitem__(self, index):
        # Read the image. 
        image_path = os.path.join(self.path_dataset, self.dataset["name"][index]) 
        image = imread(image_path) 
                
        # Convert the string and float labels into one-hot encoding.  
        mos = toTensor(self.dataset["mos"][index]) 

        if self.dataset_name == "SPAQ": 
            image = resize_shorter_edge_np(image, 512) # SPAQ lab study conducted with short edge 512 resized.  
                    
        if self.mode == "train": 
            image = randomHorizontalFlip(image) 

        image = toTensor(image)

        # Dictionary to be returned. 
        datum = {"image": image, "quality": mos} 
        return datum 
        
    def __len__(self): 
        return len(self.dataset["name"])  



