import os
import mne 
import numpy as np
import json

folder_path = './benchmark/RAW_DATA/SLEEP_EDF/sleep-edfx/1.0.0/sleep-cassette'
out_folder = './benchmark/RAW_DATA/SLEEP_EDF/_temp_cassette'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def write2json(file, path):
    with open(path, 'w') as f:
        json.dump(file, f, indent=4, 
                  cls=NumpyEncoder)
    return


if __name__=='__main__':
    
    label_dir = os.path.join(out_folder, 'labels')
    input_dir = os.path.join(out_folder, 'inputs')
    
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    
    # input
    for file_name in os.listdir(folder_path):
        if '.edf' not in file_name: continue 
        # print(file_name)
        if 'Hypnogram' not in file_name:
            file_path = os.path.join(folder_path, file_name)
            data = mne.io.read_raw_edf(file_path)
            
            patient_name = file_name.split('-PSG')[0][:-2]
            raw_data = np.array(data.get_data())
            # print(raw_data.shape)
            np.save(os.path.join(input_dir, patient_name+'.npy'), raw_data)
            
        elif 'Hypnogram' in file_name:
            label_name = file_name.split('-Hypnogram')[0][:-2] +'.json'
            label_path = os.path.join(folder_path, file_name)
            
            new_label_path = os.path.join(label_dir, label_name)
            label = mne.read_annotations(label_path)
            # mne.export.export_raw(new_label_path, label, fmt='edf', overwrite=False)
            # print(label.onset)
            # print(label.duration)
            # print(label.description)
            
            label_dict = {
                'start': label.onset,
                'duration': label.duration,
                'label': [d[-1] for d in label.description]
            }
            
            write2json(label_dict, new_label_path)
            # print(label.orig_time)
            # print(label.ch_names)
            # break