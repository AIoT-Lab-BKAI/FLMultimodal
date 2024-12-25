import os
import numpy as np
import mne  
import json

from tqdm import tqdm

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def read_np(file):
    return np.load(file)

def cut_and_merge(input, label, input_folder):
    global gid, min_duration
    gts = []
    for (start, duration, gt) in zip(label['start'], label['duration'], label['label']):
        start, duration = int(start), int(duration)
        min_duration = min(min_duration, duration)
        section = input[:, start: start + duration]
        
        if duration < threshold_duration: continue
        if gt in ['e', '?']: continue
        
        np.save(os.path.join(out_folder, f"{gid}.npy"), section)
        gts.append(gt)
        gid += 1
        
    return gts 

def run(input_folder, label_folder):
    global gid
    all_gts = []
    label_names = os.listdir(label_folder)
    for label_name in tqdm(label_names, total=len(label_names)):
        input_name = label_name.split('.json')[0] + '.npy'
        input_path = os.path.join(input_folder, input_name)
        label_path = os.path.join(label_folder, label_name)
        
        inputs = read_np(input_path)
        labels = read_json(label_path)
        
        gts = cut_and_merge(inputs, labels, input_folder)
        all_gts += gts
    
    with open(os.path.join(out_folder, 'metadata.json'), 'w') as f:
        json.dump(all_gts, f, indent=4)
        
    print(gid, len(all_gts))

if __name__=='__main__':
    temp_folder = './benchmark/RAW_DATA/SLEEP_EDF/_temp'
    out_folder = './benchmark/RAW_DATA/SLEEP_EDF/final_data'
    
    os.makedirs(out_folder, exist_ok=True)
    
    gid = 0
    min_duration = 1e9
    threshold_duration = 100
    
    input_folder = os.path.join(temp_folder, 'inputs')
    label_folder = os.path.join(temp_folder, 'labels')
    
    run(input_folder, label_folder)
    print('min duration: ', min_duration)