from torch.utils.data import Subset
from base.torch_dataset import TorchDataset
from .preprocessing import create_semisupervised_setting
from glob import glob 


import logging
import torch
import pickle
import random

import os.path as osp
import numpy as np


class shearBuilding_Dataset(TorchDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class : int = 1, 
                n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0, 
                ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0) : 

        super().__init__(root)

        # Define normal and outlier classes 

        self.n_classes = 2 # 0: normal, 1: outlier 
        self.normal_classes = tuple([normal_class]) 
        self.outlier_classes = list(range(0, 1))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0: 
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1: 
            self.known_outlier_classes = tuple([known_outlier_class])
        else : 
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # Preprocessing : feature s

        train_set = shearBuilding(root=self.root, train_type='train')
        train_set.load_data()   
        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes, 
                                                            self.outlier_classes, self.known_outlier_classes, 
                                                            ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets) # set respective semi-supervised labels 

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set 
        self.test_set = shearBuilding(root=self.root, train_type='test')
        self.test_set.load_data()   



class shearBuilding: 

    def __init__(self, root: str,  train_type : str = 'train') : 

        self.root = root 
        self.train_type = train_type
        self.data_files = glob(osp.join(self.root, self.train_type, '*.pkl'))
        self.data_files.sort()
        
        
    def __len__(self) : 
        return len(self.data_files)

    def __getitem__(self, idx) : 
        """
        Return bridge response data 

        Args : 
            idx (int)  : Index 

        Returns : 
            tuple: (response, target, semi_target, index)

        """
        data = self.read_data(idx)
        target, semi_target = int(self.targets[idx]), int(self.semi_targets[idx])
        
        data = torch.tensor(data, dtype=torch.float)
        target = torch.tensor(target,)
        semi_target = torch.tensor(semi_target,)

        return data, target, semi_target, idx 


    def read_data(self, idx):
        """
        Load data from pkl data 

        Args : 
            idx (int) : index 

        Returns : 
            data (list)
        
        """
        data_file_path = self.data_list[idx][0] 

        with open(data_file_path, 'rb') as f: 
            data_dict = pickle.load(f)   

        data = data_dict['amplitude']

        return data[..., np.newaxis].T  

    def load_data(self) : 
        """
        Load target from pkl data 

        Args : 

        Returns : 
            None 
        
        """
        
        self.data_list = []
        self.targets = []

        for data_file in self.data_files : 
            datum = [data_file]
        
            with open(data_file, 'rb') as f: 
                datum_dict = pickle.load(f)   
            
            anomal_target = np.int(datum_dict['is_anomal'])

            self.targets.append(anomal_target)
            datum.append(anomal_target)
            self.data_list.append(datum)
        
        self.targets = np.array(self.targets)
        self.targets = torch.from_numpy(self.targets)
        self.targets = torch.squeeze(self.targets)
        self.semi_targets = torch.zeros_like(self.targets)


# def preprocessing(file_path: str, save_path: str):
#     # read lvm 
#     # return 



# if __name__ == "__main__":
#     preprocessing()