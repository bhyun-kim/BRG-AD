from glob import glob 
from torch.utils.data import Subset

from .preprocessing import create_semisupervised_setting
from base.torch_dataset import TorchDataset

import torch
import pickle
import random
import struct

import os.path as osp
import numpy as np


class hanBridge_Dataset(TorchDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class : int = 1, 
                n_known_outlier_classes: int = 0, ratio_known_normal: float = 0.0, 
                ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0) : 

        super().__init__(root)

        # Define normal and outlier classes 

        self.n_classes = 2 # 0: normal, 1: outlier 
        self.normal_classes = tuple([normal_class]) 
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)
        self.outlier_classes = tuple(self.outlier_classes)

        if n_known_outlier_classes == 0: 
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1: 
            self.known_outlier_classes = tuple([known_outlier_class])
        else : 
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))

        # Preprocessing : feature s

        train_set = hanBridge(root=self.root, train_type='train')
        train_set.load_data()   
        # Create semi-supervised setting
        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes, 
                                                            self.outlier_classes, self.known_outlier_classes, 
                                                            ratio_known_normal, ratio_known_outlier, ratio_pollution)
        train_set.semi_targets[idx] = torch.tensor(semi_targets) # set respective semi-supervised labels 

        # Subset train_set to semi-supervised setup
        self.train_set = Subset(train_set, idx)

        # Get test set 
        self.test_set = hanBridge(root=self.root, train_type='test')
        self.test_set.load_data()   



class hanBridge: 

    def __init__(self, root: str,  sns_loc :  list = [0], data_type: str = 'acc', 
                train_type : str = 'train') : 

        self.root = root 
        self.sns_loc = sns_loc 
        self.data_type = data_type 
        self.train_type = train_type

        self.file_ext = '*_DG_01.dam'
        self.year = '2017'
        if train_type == 'train' :
            self.data_normal_files = []
            date_normal_list = ['0910', '0911', '0912', '0913', '0914']
            for date in date_normal_list :
                self.data_normal_files = self.data_normal_files + glob(osp.join(self.root, self.year, self.year + date, self.file_ext))

            self.data_anomaly_files = []
            date_anomaly_list = ['0923']

            for date in date_anomaly_list :
                self.data_anomaly_files = self.data_anomaly_files + glob(osp.join(self.root, self.year, self.year + date, self.file_ext))
            
            self.data_files = self.data_normal_files + self.data_anomaly_files

        elif train_type == 'test' : 
            self.data_normal_files = []
            date_normal_list = ['0915', '1003']
            for date in date_normal_list :
                self.data_normal_files = self.data_normal_files + glob(osp.join(self.root, self.year, self.year + date, self.file_ext))

            self.data_anomaly_files = []
            date_anomaly_list = ['0924']

            for date in date_anomaly_list :
                self.data_anomaly_files = self.data_anomaly_files + glob(osp.join(self.root, self.year, self.year + date, self.file_ext))
            
            self.data_files = self.data_normal_files + self.data_anomaly_files
        
        self.data_anomaly_files.sort()
        self.data_normal_files.sort()
        
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

    def read_dam_file(self, filename):    
        """Convert dam file to np.array
        Args : 
            filename (str) 

        Returns : 
            output (np.arr, np.float32)  
        """
        input = np.fromfile(filename, dtype='<i4')
        output = [struct.unpack('f', x) for x in input]

        return np.array(output)

    def read_data(self, idx):
        """
        Load data from pkl data 

        Args : 
            idx (int) : index 
            sns_loc (int) : sensor location 

        Returns : 
            data 
        
        """
        data_file_path = self.data_list[idx][0] # The first element of data_list is the filename

        data = self.read_dam_file(data_file_path)

        data = np.array(np.squeeze(data[:2560])) # TODO check index  
        # data_norm = data / np.max(np.abs(data),axis=0)
        data_norm = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        return data_norm[:16*16*8]


    def load_data(self) : 
        """
        Load target from pkl data 

        Args : 

        Returns : 
            None 
        
        """

        
        self.data_list = []
        self.targets = []
        for data_file in self.data_normal_files : 
            datum = [data_file] 
            self.targets.append(0)
            datum.append(0)
            self.data_list.append(datum)

        for data_file in self.data_anomaly_files : 
            datum = [data_file] 
            self.targets.append(1)
            datum.append(1)
            self.data_list.append(datum)
        
        self.targets = np.array(self.targets)
        self.targets = torch.from_numpy(self.targets)
        self.targets = torch.squeeze(self.targets)
        self.semi_targets = torch.zeros_like(self.targets)