import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from utils import run_length_decode
import pickle
import os
import pywt

def get_decomosition(img, mean, std):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    wavelet_decomp = np.zeros(LL.shape+(2,), dtype=np.float32)
    for idx,el in enumerate([LH, HL]):
        wavelet_decomp[...,idx] = (((el - el.min()) / (el.max() - el.min())) - mean[idx+1]) / std[idx+1]
    return wavelet_decomp

class SegDataset(Dataset):
    def __init__(self,  data_folder, rle_dict_path, fold_keys_path, size, mean, std, transforms = None, mode = 'train', tta_func = None):
        
        with open(rle_dict_path, 'rb') as handle:
            self.rle_dict = pickle.load(handle)
            
        with open(fold_keys_path, 'rb') as handle:
            self.fold_keys = pickle.load(handle)
            
        self.data_folder = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.transforms = transforms
        self.mode = mode
        self.tta_func = tta_func
        
    def __len__(self):
        return len(self.fold_keys)
    
    def __getitem__(self, idx):
        image_id = self.fold_keys[idx]
        image_path = os.path.join(self.data_folder, image_id + ".png")
        image = cv2.imread(image_path)
        mask = np.zeros([1024, 1024])
        if self.mode != 'test':
            if self.rle_dict[image_id][0] != ' -1':
                for rle in self.rle_dict[image_id]:
                    mask += run_length_decode(rle)
                    
        if self.mode == 'test' and self.tta_func is not None:
            image = self.tta_func(image)
        mask = (mask >= 1).astype('float32') # for overlap cases
        if self.transforms is not None:
            aug_fun = self.transforms(1,self.size, self.mean, self.std)
            augmented = aug_fun(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        image = torch.from_numpy(np.transpose(image,(2,0,1))).type(torch.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).type(torch.float32)
        return image, mask
    
class SegWaveletDataset(Dataset):
    def __init__(self,  data_folder, rle_dict_path, fold_keys_path, size, mean, std, transforms = None, mode = 'train', tta_func = None):
        
        with open(rle_dict_path, 'rb') as handle:
            self.rle_dict = pickle.load(handle)
            
        with open(fold_keys_path, 'rb') as handle:
            self.fold_keys = pickle.load(handle)
            
        self.data_folder = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.transforms = transforms
        self.mode = mode
        self.tta_func = tta_func
        
    def __len__(self):
        return len(self.fold_keys)
    
    def __getitem__(self, idx):
        image_id = self.fold_keys[idx]
        image_path = os.path.join(self.data_folder, image_id + ".png")
        image = cv2.imread(image_path)
        mask = np.zeros([1024, 1024])
        if self.mode != 'test':
            if self.rle_dict[image_id][0] != ' -1':
                for rle in self.rle_dict[image_id]:
                    mask += run_length_decode(rle)
                   
        if self.mode == 'test' and self.tta_func is not None:
            image = self.tta_func(image)
            
        mask = (mask >= 1).astype('float32') # for overlap cases
        if self.transforms is not None:
            aug_fun = self.transforms(1,self.size, self.mean, self.std)
            augmented = aug_fun(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        img_wav_tr = cv2.resize(np.round(((image[...,0]*self.std[0]) + self.mean[0])*255).astype(np.uint8), (self.size*2,self.size*2), interpolation=cv2.INTER_CUBIC)
        wavelets = get_decomosition(img_wav_tr, self.mean, self.std)
        image[...,1:] = wavelets
        
        image = torch.from_numpy(np.transpose(image,(2,0,1))).type(torch.float32)
        mask = torch.from_numpy(mask).unsqueeze(0).type(torch.float32)
        return image, mask