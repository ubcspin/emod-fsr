import os
import re

import pandas as pd
import numpy as np
import scipy.io as sp_io
import glob

from tqdm import tqdm

def calculate_scene_restart_index(df, death_index):
    scene_parts_index = df[(df.scene_begin == 1) | (df.scene_peak == 1) | (df.scene_end == 1)].index    
    scene_restart_index = list(filter(lambda x: x >= death_index, scene_parts_index))[0]
    return scene_restart_index

def fix_is_death_nan(df):
    death_index_list = df[(df.is_death == 1)].index
    
    for death_index in death_index_list:
        scene_restart_index = calculate_scene_restart_index(df, death_index)
        df.loc[death_index:scene_restart_index, 'is_death'] = 1
        df.loc[death_index:scene_restart_index, 'scene_tag'] = df.loc[death_index, 'scene_tag']
        df.loc[death_index:scene_restart_index, 'scene_tag_ind'] = df.loc[death_index, 'scene_tag_ind']
        df.loc[death_index:scene_restart_index, 'scene_first_start'] = df.loc[death_index, 'scene_first_start']
        df.loc[death_index:scene_restart_index, 'stream'] = df.loc[death_index, 'stream']
    df.is_death.fillna(0, inplace=True)
    
    df.is_death = df.is_death.astype('boolean')
    
    return df


def fix_scene_begin_nan(df):
    df.scene_begin = ~(df.scene == df.scene.shift())
    return df

def fix_scene_end_nan(df):
    df.scene_end = ~(df.scene == df.scene.shift(-1))
    return df

def fix_scene_peak_nan(df):
    df.scene_peak = (df.scene_begin == False) & (df.scene_end == False)
    return df

def fix_scenes(df):
    row = df[df.scene_begin == True].iterrows()
    scene_label_set = set()
    
    prev_row = current_row = next(row)
    available_rows = True

    while(available_rows):
        if current_row[1].scene != prev_row[1].scene:
            if current_row[1].scene in scene_label_set:
                df.loc[current_row[0], 'scene'] = prev_row[1].scene
                df.loc[current_row[0], 'scene_begin'] = False
            else:
                scene_label_set.add(current_row[1].scene)
                df.loc[current_row[0], 'scene_first_start'] = 1
        try:
            prev_row = current_row
            current_row = next(row)
        except StopIteration:
            break
    
    df.loc[0, 'scene_first_start'] = 1
    df.scene_first_start.fillna(0, inplace=True)
    df.scene_first_start = df.scene_first_start.astype('boolean')
    return df

def fix_nan_values(df):
    df = fix_is_death_nan(df)
    df = fix_scenes(df)
    df = fix_scene_begin_nan(df)
    df = fix_scene_end_nan(df)
    df = fix_scene_peak_nan(df)
    return df

def create_keystroke_flags(df):
    keys = list(map(lambda x: 'a' + str(x), range(7)))
    for key in keys:
        df[key + '_pressed'] = (df[key].astype('int64') > 0)
        df[key + '_press_start'] = (df[key + '_pressed']) & (df[key + '_pressed'] != df[key + '_pressed'].shift()) 
        df[key + '_down'] = (df[key].astype('int64') == 0)
        df[key + '_press_end'] = (df[key + '_down']) & (df[key + '_down'] != df[key + '_down'].shift()) 
        df.loc[0, key + '_press_end'] = False
        assert len(pd.Index(df[key + '_press_start']).get_indexer_for([True])) == len(pd.Index(df[key + '_press_end']).get_indexer_for([True]))
    
    return df

if __name__ == "__main__":
    merged_data = {}
    cols = ['timestamps', 'a0', 'a1', 'a2', 'a3', 'a4', 'scene', 'feeltrace', 'calibrated_values']
    lowpass_cols = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'feeltrace']
    order = {v: i for i, v in enumerate(file_order)}
    time_index='timedelta'
    time_interval='1s'

    for pnum in subject_data.keys():
        print('[INFO] Participant: ', pnum)
        
        subject_data[pnum].sort(key=lambda x: order[x['filename']])
        
        subject_data_iter = iter(subject_data[pnum])
        
        df = pd.DataFrame.from_dict(next(subject_data_iter)['df'])
        
        for subject in tqdm(subject_data_iter):
            subject_df = subject['df']
            df = pd.merge_ordered(df, subject_df, on='timestamps', suffixes=(None, '_'+subject['filename'].split('.')[0]))
            
        # backward and foward fill NAs with valid observations
        df['pnum'] = pnum
        print('[INFO] Sorting')
        df.sort_values(by='timestamps', inplace=True)
        df.reset_index(inplace=True, drop=True)

        print('[INFO] Filling NaNs')
        df[cols] = df[cols].bfill().ffill()
        df = df[~df.timestamps.duplicated()]

        df = fix_nan_values(df)

        print('[INFO] Combined keypress')
        fsr_cols = list(map(lambda x: 'a' + str(x), range(5)))
        df['a5'] = np.sum(df[fsr_cols], axis=1)
        df['a6'] = np.max(df[fsr_cols], axis=1)
        
        print('[INFO] Resampling')
        df['timedelta'] = pd.TimedeltaIndex(df.timestamps, unit='ms')
        df = df.set_index('timedelta').resample('33ms').nearest()
        df.reset_index(inplace=True, drop=False)
        df['timestamps'] = df.timedelta.astype('int64') / 1e9
        
        print('[INFO] Smoothing')
        df = ewm(df, lowpass_cols)  
        df = filter_signals(df, lowpass_cols) 
        
        print('[INFO] Normalizing feeltrace')
        df.loc[:, df.columns.str.contains('feeltrace')] = df.loc[:, df.columns.str.contains('feeltrace')] / 250
        
        print('[INFO] Creating keystroke flags')
        df = create_keystroke_flags(df)
        
        merged_data[pnum] = df