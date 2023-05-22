from __future__ import absolute_import, print_function

import argparse
import os
import sys

import ipdb
from tqdm import tqdm

from data_utils.preprocessing.mimic3benchmark.clean_extracts import \
    extract_episodes
from data_utils.preprocessing.mimic3benchmark.subject import (read_diagnoses,
                                                              read_events,
                                                              read_stays)


def main(args):
    for subject_dir in tqdm(os.listdir(args.subjects_root_path), desc='Iterating over subjects'):
        dn = os.path.join(args.subjects_root_path, subject_dir)
        try:
            subject_id = int(subject_dir)
            if not os.path.isdir(dn):
                raise Exception
        except:
            continue

        try:
            # reading tables of this subject
            stays = read_stays(os.path.join(args.subjects_root_path, subject_dir))
            diagnoses = read_diagnoses(os.path.join(args.subjects_root_path, subject_dir))
            events = read_events(os.path.join(args.subjects_root_path, subject_dir))
        except:
            sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject_id))
            continue
        res = extract_episodes(stays, diagnoses, events)
        for i, (hadm_id, episodic_data, episode) in enumerate(res):
            episodic_data.to_csv(os.path.join(args.subjects_root_path, subject_dir, f'episode{i+1}_{hadm_id}.csv'), index_label='Icustay')
            episode.to_csv(os.path.join(args.subjects_root_path, subject_dir, f'episode{i+1}_{hadm_id}_timeseries.csv'), index_label='Hours')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract episodes from per-subject data.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    parser.add_argument('--variable_map_file', type=str,
                        default=os.path.join(os.path.dirname(__file__), './resources/itemid_to_variable_map.csv'),
                        help='CSV containing ITEMID-to-VARIABLE map.')
    parser.add_argument('--reference_range_file', type=str,
                        default=os.path.join(os.path.dirname(__file__), './resources/variable_ranges.csv'),
                        help='CSV containing reference ranges for VARIABLEs.')
    args, _ = parser.parse_known_args()
    main(args)

