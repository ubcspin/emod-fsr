import os
import re

import pandas as pd
import numpy as np
import scipy.io as sp_io
import glob

from tqdm import tqdm

def extract_pnum(filename: str):
    match = re.search('[0-9]?[0-9]', filename)
    return 'p' + "%02d" % int(match.group(0))


def create_dataset(src_dir: str, out_dir = 'COMBINED_DATA'):
    os.makedirs(out_dir, exist_ok=True) 
    subject_data_dir = glob.glob(os.path.join(src_dir, 'p*'))
    all_subjects_files = [glob.glob(os.path.join(x, '*')) for x in subject_data_dir]
    
    file_dict = {
        "fsr.csv": ['timestamps', 'a0', 'a1', 'a2', 'a3', 'a4'],
        "joystick.csv": ['timestamps', 'feeltrace'],
        "events_game_controlled_visual.csv": ['timestamps', 'label'],
        "events_game_controlled_sound.csv": ['timestamps', 'label'],
        "events_player_controlled.csv": ['timestamps', 'label'],
        "calibrated_words_calibrated_values.csv": ['calibrated_values'],
        "calibrated_words_calibrated_words.csv": ['calibrated_words'],
        "calibrated_words_timestamp_ms.csv": ['timestamps'],
        "interview.csv": ['timestamps', 'quote'],
        "scenes.csv": ['scene', 'stream', 'scene_begin', 'scene_peak', 
                       'scene_end', 'is_death', 'scene_tag', 'scene_tag_ind', 
                       'timestamps', 'scene_first_start']
    }
    
    read_order = list(file_dict.keys())
    print(read_order)
    
    subjects = {}
    
    for subject_files in all_subjects_files:
        subject_data = []
        subject_files = sorted(subject_files)

        calibrated_words = pd.DataFrame()
        
        for file in tqdm(subject_files):
            filename = file.split('/')[-1]
            if not filename in file_dict.keys():
                continue
            x = pd.read_csv(file, names=file_dict[filename], header=0)
            
            if 'calibrated' in filename:
                calibrated_words[x.columns] = x.values
                if 'timestamp' in filename:
                    subject_data.append({'filename': 'calibrated_words.csv', 'df': calibrated_words})
                else:
                    continue
            else:
                subject_data.append({'filename': filename, 'df': x})
        pnum = extract_pnum(file)
        subjects[pnum] = subject_data
        
    return subjects


if __name__ == "__main__":
    subject_data = create_dataset('../data/trial_data_split-anon')
