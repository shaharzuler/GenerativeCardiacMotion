import os

import torch

import numpy as np
from torch.utils.data import DataLoader



class cvae_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num_layers=6, device=None): 
        self.flow_dir = f"{dataset_path}/flows"
        self.img_features_dir = f"{dataset_path}/img_features"
        self.num_layers = num_layers
        self.device = device

    def __getitem__(self, idx):
        flow = torch.tensor(np.load(f"{self.flow_dir}/flow_sample_{idx}.npy"), device=self.device).float()
        conditions_pyramid = []
        data_sample = {"flow": flow}
        for num_layer in range(self.num_layers):
            layer = torch.tensor(np.load(f"{self.img_features_dir}/img_features_layer_{num_layer}_sample_{idx}.npy"), device=self.device).float()
            conditions_pyramid.append(layer)
            data_sample[num_layer] = layer
        return  flow, conditions_pyramid # data_sample #

    def __len__(self):
        return len(os.listdir((self.flow_dir)))
