from __future__ import absolute_import, print_function

import argparse
import functools
import json
import os
import sys

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import tqdm

from data_utils.preprocessing.mimic3benchmark.preprocessing import (
    assemble_episodic_data, clean_events, map_itemids_to_variables,
    read_itemid_to_variable_map)
from data_utils.preprocessing.mimic3benchmark.subject import (
    add_hours_elpased_to_events, convert_events_to_timeseries,
    get_events_for_stay, get_first_valid_from_timeseries, read_diagnoses,
    read_events, read_stays)


@functools.lru_cache()
def read_varmaps(variable_map_file=None):
    if variable_map_file is None:
        variable_map_file = os.path.join(os.path.dirname(__file__), './resources/itemid_to_variable_map.csv')
    var_map = read_itemid_to_variable_map(variable_map_file)
    variables = var_map.VARIABLE.unique()
    return var_map, variables

@functools.lru_cache()
def read_feature_map_fns():
    with open(os.path.join(os.path.dirname(__file__), "mimic3models/resources/channel_info.json")) as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    ret = {}
    for channel, info in channel_info.items():
        if len(info['possible_values']) > 0:
            ret[channel] = lambda x: info['values'].get(x, np.NaN)
        else:
            ret[channel] = lambda x: x
    return ret

def extract_episodes(stays, diagnoses, events, variable_map_file=None):
    var_map, variables = read_varmaps(variable_map_file)

    episodic_data = assemble_episodic_data(stays, diagnoses)
    events = map_itemids_to_variables(events, var_map)
    events = clean_events(events)
    ret = []
    if events.shape[0] == 0:
        # no valid events for this subject
        return ret
    timeseries = convert_events_to_timeseries(events, variables=variables)

    # extracting separate episodes
    for i in range(stays.shape[0]):
        stay_id = stays.ICUSTAY_ID.iloc[i]
        hadm_id = stays.HADM_ID.iloc[i]
        intime = stays.INTIME.iloc[i]
        outtime = stays.OUTTIME.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            # no data for this episode
            continue

        episode = add_hours_elpased_to_events(episode, intime).set_index('HOURS').sort_index(axis=0)
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'Weight'] = get_first_valid_from_timeseries(episode, 'Weight')
            episodic_data.loc[stay_id, 'Height'] = get_first_valid_from_timeseries(episode, 'Height')
        episodic_data = episodic_data.loc[episodic_data.index == stay_id]
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "Hours" else x))
        episode = episode[columns_sorted]
        for col in episode.columns:
            try:
                episode[col] = episode[col].astype(float)
            except:
                pass
        ret.append((hadm_id, episodic_data, episode))
    return ret


from data_utils.preprocessing.mimic3benchmark.mimic3models.feature_extractor import \
    extra_features_clean


def extract_features_from_df(df, period='all', features='all'):
    df = df.set_index('Hours').sort_index()
    map_fns = read_feature_map_fns()
    ret = []
    for col in df.columns:
        map_fn = map_fns[col]
        ret.append(extra_features_clean(df[col].map(map_fn), name=col, period=period,features=features))
    return pd.concat(ret)

def read_all_episodes(subjects_root_path, max_cnt=None):
    subjects = sorted(pd.Index(list(filter(str.isdigit, os.listdir(subjects_root_path)))))
    ret = {}
    for subject in tqdm.tqdm(subjects[:max_cnt] if max_cnt is not None else subjects):
        subject_path = os.path.join(subjects_root_path, subject)
        stays = read_stays(subject_path)
        diagnoses = read_diagnoses(subject_path)
        episodes = read_events(subject_path)
        for i, (hadm_id, _, episode) in enumerate(extract_episodes(stays, diagnoses, episodes)):
            ret[hadm_id] = episode.reset_index().rename(columns={"HOURS":"Hours"})
    return ret