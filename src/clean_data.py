import os
import re
import glob
import logging
import utils

import pandas as pd
import numpy as np

from config import TIME_INDEX, TIME_INTERVAL, FS, SAMPLE_PERIOD

INPUT_PICKLE_FILE = True
INPUT_DIR = 'COMBINED_DATA'
INPUT_PICKLE_NAME = 'subject_data.pk'

SAVE_PICKLE_FILE = True
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'merged_fsr_data.pk'


def calculate_scene_restart_index(df, is_death_index):
    scene_parts_index = df[(df.scene_begin == 1) | (
        df.scene_peak == 1) | (df.scene_end == 1)].index
    scene_restart_index = list(
        filter(lambda x: x >= is_death_index, scene_parts_index))[0]
    return scene_restart_index


def fix_is_death_nan(df):
    is_death_index_list = df[(df.is_death == 1)].index

    for is_death_index in is_death_index_list:
        scene_restart_index = calculate_scene_restart_index(df, is_death_index)
        df.loc[is_death_index:scene_restart_index, 'is_death'] = 1
        df.loc[is_death_index:scene_restart_index,
               'scene_tag'] = df.loc[is_death_index, 'scene_tag']
        df.loc[is_death_index:scene_restart_index,
               'scene_tag_ind'] = df.loc[is_death_index, 'scene_tag_ind']
        df.loc[is_death_index:scene_restart_index,
               'scene_first_start'] = df.loc[is_death_index, 'scene_first_start']
        df.loc[is_death_index:scene_restart_index,
               'stream'] = df.loc[is_death_index, 'stream']
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
        df[key + '_press_start'] = (df[key + '_pressed']) & (
            df[key + '_pressed'] != df[key + '_pressed'].shift())
        df[key + '_down'] = (df[key].astype('int64') == 0)
        df[key + '_press_end'] = (df[key + '_down']
                                  ) & (df[key + '_down'] != df[key + '_down'].shift())
        df.loc[0, key + '_press_end'] = False
        assert len(pd.Index(df[key + '_press_start']).get_indexer_for([True])
                   ) == len(pd.Index(df[key + '_press_end']).get_indexer_for([True]))

    return df

def parse_data(subject_data: dict):
    merged_data = {}
    cols = ['timestamps', 'a0', 'a1', 'a2', 'a3', 'a4',
            'scene', 'feeltrace', 'calibrated_values']
    lowpass_cols = ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'feeltrace']
    order = {v: i for i, v in enumerate(file_order)}
    
    for pnum in subject_data.keys():
        logging.info(f'Parsing data for {pnum}')

        subject_data[pnum].sort(key=lambda x: order[x['filename']])

        subject_data_iter = iter(subject_data[pnum])

        df = pd.DataFrame.from_dict(next(subject_data_iter)['df'])

        for subject in tqdm(subject_data_iter):
            subject_df = subject['df']
            df = pd.merge_ordered(df, subject_df, on='timestamps', suffixes=(
                None, '_'+subject['filename'].split('.')[0]))
        df['pnum'] = pnum

        logging.info('Sorting DataFrame')
        df.sort_values(by='timestamps', inplace=True)
        df.reset_index(inplace=True, drop=True)

        logging.info('Filling NaNs')
        df[cols] = df[cols].bfill().ffill()
        df = df[~df.timestamps.duplicated()]
        df = fix_nan_values(df)

        logging.info('Creating combined keypress keys')
        fsr_cols = list(map(lambda x: 'a' + str(x), range(5)))
        df['a5'] = np.sum(df[fsr_cols], axis=1)
        df['a6'] = np.max(df[fsr_cols], axis=1)

        logging.info('Fix sampling to {FS}Hz')
        df['timedelta'] = pd.TimedeltaIndex(df.timestamps, unit='ms')
        df = df.set_index('timedelta').resample(SAMPLE_PERIOD).nearest()
        df.reset_index(inplace=True, drop=False)
        df['timestamps'] = df.timedelta.astype('int64') / 1e9
    
        logging.info('Scaling feeltrace to 0-1 range')
        df.loc[:, df.columns.str.contains(
            'feeltrace')] = df.loc[:, df.columns.str.contains('feeltrace')] / MAX_FEELTRACE

        logging.info('Creating keystroke flags')
        df = create_keystroke_flags(df)

        merged_data[pnum] = df

    return merged_data


if __name__ == "__main__":
    if INPUT_PICKLE_FILE:
        input_pickle_file_path = os.path.join(INPUT_DIR, INPUT_PICKLE_NAME)
        subject_data = utils.load_pickle(file_path=input_pickle_file_path)
    

    merged_fsr_data = parse_data(subject_data)
    
    if SAVE_PICKLE_FILE:
        output_pickle_file_path = os.path.join(OUTPUT_DIR, OUTPUT_PICKLE_NAME)
        utils.pickle_data(data=merged_fsr_data, file_path=output_pickle_file_path)
