from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import random
import json 

class EDFDataset(Dataset):
    def __init__(self, root, download=True, standard_scaler=True, train=True, crop_length=250, valid=False):
        self.root = root
        self.standard_scaler = standard_scaler
        self.train = train
        self.crop_length = crop_length
        self.time_length = crop_length
        
        self.label2idx = {
            'W': 0, '1': 1, '2': 2, '3': 3, '4': 3, 'R': 4}

        # if not os.path.exists(self.root):
        #     if download:
        #         print('Downloading EDF Dataset...', end=' ')
        #         os.makedirs(root, exist_ok=True)
        #         os.system('bash ./benchmark/ptbxl_classification_lm/download.sh')
        #         print('done!')
        #     else:
        #         raise RuntimeError('Dataset not found. You can use download=True to download it')
        
        with open(os.path.join(self.root, 'metadata.json')) as f:
            self.labels = json.load(f)
        
        if self.train:
            with open(os.path.join(self.root, 'train_metadata.json')) as f:
                self.indices = json.load(f)
        else:
            with open(os.path.join(self.root, 'test_metadata.json')) as f:
                self.indices = json.load(f)
        self.y = np.array(
            [self.label2idx[self.labels[oid]] for oid in self.indices])
        
    def standardize(self, x):
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        return (x - mean) / std
    
    def adjust_crop_length(self, crop_length):
        print("Reset crop length to: ", crop_length)
        self.crop_length = crop_length
        return
            
    def normalize(self, x): 
        pmax = np.max(x, axis=1, keepdims=True) 
        pmin = np.min(x, axis=1, keepdims=True) 
        return (x - pmin) / (pmax - pmin + 1e-9)
    
    def interpolate(self, x, time_length):
        old_length = x.shape[0] # LxC
        old_time_stamps = np.arange(old_length)
        new_time_stamps = np.linspace(0, old_length, time_length)
        x_new = np.interp(new_time_stamps, old_time_stamps, x)
        return x_new
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        # x = self.x[index]
        orig_idx = self.indices[index]
        x = np.load(os.path.join(self.root, f"{orig_idx}.npy"))
        
        x = x[:-2, :]   # remove event marker, rectal temperature
        x = x.transpose(1, 0)   # CxL -> LxC
        
        x_new = np.zeros([self.time_length, x.shape[1]])
        
        for i in range(x.shape[1]):
            x_new[:, i] = self.interpolate(x[:, i], self.time_length)
        x = x_new
        
        y = self.label2idx[self.labels[orig_idx]]
        if x.shape[0] >= self.crop_length:
            # start_idx = random.randint(0, x.shape[0] - self.crop_length - 1)
            start_idx = 0
            x = x[start_idx:start_idx + self.crop_length, :].transpose().astype(np.float32)
            x = self.standardize(x).astype(np.float32)
            # x = x[: self.crop_length, :].transpose().astype(np.float32)
        else:
            x = self.standardize(x).astype(np.float32)
            x = np.pad(x, ((0, self.crop_length - x.shape[0]), (0, 0)),
                       mode='constant', constant_values=-1).transpose()
        x = x.astype(np.float32)
        # y = y.astype(np.float32)
        
        return x, y
    
if __name__ == '__main__':
    dataset = EDFDataset(root='./benchmark/RAW_DATA/SLEEP_EDF/final_data', standard_scaler=True, train=True)
    for i in range(1):
        x, y = dataset[i]
        # print(x.shape)
        for i in range(x.shape[0]):
            print(x[i])
