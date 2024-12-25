import os
import numpy as np
import mne  
import json
from tqdm import tqdm
import random

def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def random_split(indices, rate):
    random.shuffle(indices)
    border_idx = int(len(indices)*rate)
    train_indices = indices[:border_idx]
    test_indices = indices[border_idx:]
    return train_indices, test_indices

def write2json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=4)

if __name__=='__main__':
    data_folder = './benchmark/RAW_DATA/SLEEP_EDF/final_data'
    data_json = os.path.join(data_folder, 'metadata.json')
    out_train_json = os.path.join(data_folder, 'train_metadata.json')
    out_test_json = os.path.join(data_folder, 'test_metadata.json')
    
    data = read_json(data_json)
    num_samples = len(data)
    
    count_dict = {}
    
    trains = []
    tests = []
    rate = 0.8
    
    for i in tqdm(range(num_samples), total=num_samples):
        label = data[i]
        count_dict[label] = count_dict.get(label, []) + [i]
    for label, indices in count_dict.items():
        train_indices, test_indices = random_split(indices, rate=rate)
        print(label, len(train_indices), len(test_indices))
        trains += train_indices
        tests += test_indices
        
    print('train', len(trains))
    print('test', len(tests))
    
    write2json(trains, out_train_json)
    write2json(tests, out_test_json)
    
    