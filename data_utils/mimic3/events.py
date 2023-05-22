import functools
import os

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import tqdm

import _settings
import data_utils.preprocessing.mimic3benchmark.clean_extracts as event_prep


@functools.lru_cache()
def normalization_constants(period, features, max_cnt=None):
    events = MIMIC_III_Event_Reader._read_all_unnormalized_events(period, features, max_cnt)
    return events.mean(), events.std()

class MIMIC_III_Event_Reader():
    def __init__(self, episodes=None, period='all', features='all') -> None:
        assert period in ['first4days', 'first8days', 'last12hours', 'first25percent', 'first50percent', 'all']
        if episodes is None:
            events = self._read_all_unnormalized_events(period, features)
            _mean, _std = normalization_constants(period, features)
            events = ((events - _mean) / _std).fillna(0.)
        else:
            events = {}
            for hadm_id, episode in tqdm.tqdm(episodes.items()):
                events[hadm_id] = event_prep.extract_features_from_df(episode, period=period, features=features)
            events = pd.DataFrame(events).T.dropna(how='all', axis=1)
            _mean, _std = normalization_constants(period, features)
            events = ((events - _mean) / _std).fillna(0.)
        # ipdb.set_trace() # events.shape==(41919,609)
        events = events.dropna(how='all', axis=1) #some have 0 std. leaving 589 features with default
        self.data = events.T
        self.hadm_ids = events.index

    @classmethod
    def _read_all_unnormalized_events(cls, period, features, max_cnt=None,
                                      subjects_root_path=_settings.MIMIC_PREPROCESS_OUTPUT):
        key = f'main_data|{period}|{features}|{max_cnt}'
        events = ptd.manual_cache(key)
        if events is None:
            episodes = event_prep.read_all_episodes(subjects_root_path,max_cnt=max_cnt)
            events = {}
            for hadm_id, tdf in tqdm.tqdm(episodes.items()):
                events[hadm_id] = event_prep.extract_features_from_df(tdf, period=period, features=features)
            events = pd.DataFrame(events).T.dropna(how='all', axis=1)
            ptd.manual_cache(key, events, write=True)
        return events

    def __getitem__(self, hadm_id):
        ret = self.data[hadm_id]
        return ret
if __name__ == '__main__':
    from importlib import reload

    o = MIMIC_III_Event_Reader()