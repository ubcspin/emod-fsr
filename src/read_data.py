import os
import re
import glob
import utils
import logging

import pandas as pd
import pickle

from tqdm import tqdm

# Filenames and columns
FILES_DICT = {
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

RAW_DATA_PATH = '../data/trial_data_split-anon'
OUTPUT_DIR = 'COMBINED_DATA'
OUTPUT_PICKLE_NAME = 'subject_data.pk'
SAVE_PICKLE_FILE = True

logging.basicConfig(
    filename=None, format='%(asctime)-6s: %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s', level=logging.DEBUG)


def read_dataset(src_dir=RAW_DATA_PATH, output_dir=OUTPUT_DIR, file_dict=FILES_DICT):
    logging.info(f'Reading data from {src_dir}')

    os.makedirs(output_dir, exist_ok=True)
    subject_data_dir = glob.glob(os.path.join(src_dir, 'p*'))
    all_subjects_files = [glob.glob(os.path.join(x, '*'))
                          for x in subject_data_dir]

    file_dict = FILES_DICT

    read_order = list(file_dict.keys())

    subjects = {}

    for subject_files in all_subjects_files:
        subject_files = sorted(subject_files)
        pnum = extract_pnum(subject_files[0])

        logging.info(f'Parsing files for {pnum}')

        subjects[pnum] = parse_files(subject_files, file_dict)

    return subjects


def extract_pnum(filename: str):
    match = re.search('[0-9]?[0-9]', filename)
    return 'p' + "%02d" % int(match.group(0))


def parse_files(subject_files: list, file_dict=FILES_DICT):
    subject_data = []
    calibrated_words = pd.DataFrame()

    for file in tqdm(subject_files):
        filename = file.split('/')[-1]

        if not filename in file_dict.keys():
            continue

        x = pd.read_csv(file, names=file_dict[filename], header=0)

        if 'calibrated' in filename:
            calibrated_words[x.columns] = x.values
            if 'timestamp' in filename:
                subject_data.append(
                    {'filename': 'calibrated_words.csv', 'df': calibrated_words})
            else:
                continue
        else:
            subject_data.append({'filename': filename, 'df': x})

        return subject_data


if __name__ == "__main__":
    subject_data = read_dataset(RAW_DATA_PATH)

    if SAVE_PICKLE_FILE:
        pickle_file_path = os.path.join(OUTPUT_DIR, OUTPUT_PICKLE_NAME)
        utils.pickle_data(data=subject_data, file_path=pickled_file_path)
