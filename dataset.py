import os

import torch
import torch.nn.functional as F

import numpy as np

from scipy.ndimage import zoom




class cvae_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num_layers=6, device=None, l1o_idx=None, train=True, augmentations=False, augmentation_params=None): 
        self.flow_dir = f"{dataset_path}/flows"
        self.img_features_dir = f"{dataset_path}/img_features"
        self.num_layers = num_layers
        self.device = device
        self.l1o_idx = l1o_idx
        self.train = train
        self.augmentations = augmentations
        self.augmentation_params = augmentation_params

    def __getitem__(self, idx):
        if self.train:
            actual_idx = idx
            if self.l1o_idx is not None:
                if idx >= self.l1o_idx:
                    actual_idx = idx + 1
        else:
            actual_idx = self.l1o_idx
        flow = torch.tensor(np.load(f"{self.flow_dir}/flow_sample_{actual_idx}.npy"), device=self.device).float()
        conditions_pyramid = []
        data_sample = {"flow": flow}
        for num_layer in range(self.num_layers):
            layer = torch.tensor(np.load(f"{self.img_features_dir}/img_features_layer_{num_layer}_sample_{actual_idx}.npy"), device=self.device).float()
            conditions_pyramid.append(layer)
            data_sample[num_layer] = layer

        if self.augmentations:
            # aug_params = {
            #     "intensity_noise_var": 0.015,
            #     "max_shift": 10,
            #     "zoom_vals": (0.9, 1.1)

            # }
            flow, conditions_pyramid = self.apply_augmentations(flow, conditions_pyramid, self.augmentation_params)
        
        return  flow, conditions_pyramid



    def apply_augmentations(self, flow, conditions_pyramid, aug_params):

        flow, conditions_pyramid = self.apply_random_movement(flow, conditions_pyramid, aug_params["max_shift"])

        flow, conditions_pyramid = self.apply_random_zoom(flow, conditions_pyramid, aug_params["zoom_vals"])

        conditions_pyramid = [layer + torch.randn_like(layer) * aug_params["intensity_noise_var"] for layer in conditions_pyramid]
        
        return flow, conditions_pyramid
    
    def apply_random_movement(self, flow, conditions_pyramid, max_shift):
        shift_x = np.random.randint(-max_shift, max_shift + 1)
        shift_y = np.random.randint(-max_shift, max_shift + 1)
        shift_z = np.random.randint(-max_shift, max_shift + 1)
        
        flow = self.shift_tensor(flow, shift_x, shift_y, shift_z)
        
        for i, layer in enumerate(conditions_pyramid):
            scale = 2 ** (self.num_layers - 1 - i)
            if any((shift_x // scale, shift_y // scale, shift_z // scale)) != 0:
                conditions_pyramid[i] = self.shift_tensor(layer, shift_x // scale, shift_y // scale, shift_z // scale)
        
        return flow, conditions_pyramid

    def shift_tensor(self, tensor, shift_x, shift_y, shift_z):
        tensor = F.pad(tensor, pad=(-shift_z, shift_z, -shift_y, shift_y, -shift_x, shift_x, 0, 0))
        return tensor

    def apply_random_zoom(self, flow, conditions_pyramid, zoom_vals):
        zoom_factor = np.random.uniform(zoom_vals[0], zoom_vals[1])

        orig_flow_shape = flow.shape

        flow = F.interpolate(flow, scale_factor=zoom_factor, mode='bilinear', align_corners=True)
        diff = np.array(orig_flow_shape) - np.array(flow.shape)
        flow = F.pad(flow,(diff[3]//2,(diff[3]-diff[3]//2),diff[2]//2,(diff[2]-diff[2]//2),diff[1]//2,(diff[1]-diff[1]//2),0,0))
        flow *= zoom_factor

        for i, layer in enumerate(conditions_pyramid):
            orig_layer_shape = layer.shape
            layer = F.interpolate(layer, scale_factor=zoom_factor, mode='bilinear', align_corners=True)
            diff = np.array(orig_layer_shape) - np.array(layer.shape)
            layer = F.pad(layer,(diff[3]//2,(diff[3]-diff[3]//2),diff[2]//2,(diff[2]-diff[2]//2),diff[1]//2,(diff[1]-diff[1]//2),0,0))

            conditions_pyramid[i] = layer
        
        return flow, conditions_pyramid





    def __len__(self):
        if self.train:
            orig_len = len(os.listdir((self.flow_dir)))
            if self.l1o_idx is not None:
                return orig_len - 1 
            else:
                return orig_len 
        else:
            return 1
