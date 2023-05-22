import os
from collections import defaultdict

import numpy as np
import pandas as pd
import persist_to_disk as ptd
import tqdm

import _settings
import data_utils.preprocessing.clinicalBERT_preprocess as note_prep


@ptd.persistf()
def read_cached_bert_features():
    features = note_prep.get_features(os.path.join(_settings.WORKSPACE, 'discharge_new.csv'))
    res = {'input_ids':[], 'input_mask': [], 'segment_ids': [], 'guid': []}
    for feature in tqdm.tqdm(features):
        for key in res.keys():
            res[key].append(getattr(feature, key))
    meta = pd.DataFrame(np.asarray([_.split("-") for _ in res['guid']]), columns=['HADM_ID', 'Seq'])
    return meta, np.asarray(res['input_ids']), np.asarray(res['segment_ids']), np.asarray(res['input_mask'])


class MIMIC_III_Notes_BertFeature_Reader():
    def __init__(self, df=None, max_blocks=15) -> None:
        if df is None:
            data = ptd.manual_cache('main')
            if data is None:
                data = note_prep.read_bert_features(os.path.join(_settings.WORKSPACE, 'discharge_new.csv'))
                ptd.manual_cache('main', data, write=True)
        else:
            data = note_prep.read_bert_features(df)
        meta, self.input_ids, self.segment_ids, self.input_mask = data
        meta['HADM_ID'] = meta['HADM_ID'].astype(int)
        self.hadm_ids = pd.Index(sorted(meta['HADM_ID'].unique()))
        self.meta = meta.astype(int).reset_index().sort_values(['HADM_ID', 'Seq']).set_index("HADM_ID")
        if max_blocks is None:
            max_blocks = meta.groupby("HADM_ID").size().max()
        self.max_blocks = max_blocks

    def pad(self, x):
        if len(x) == self.max_blocks: return x
        padding = np.zeros(shape=(self.max_blocks-len(x), x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, padding], 0)

    def __getitem__(self, hadm_id):
        tdf = self.meta.loc[hadm_id].iloc[:self.max_blocks]
        if isinstance(tdf, pd.Series):
            tdf = pd.DataFrame(tdf).T

        return {
            "num_blocks": len(tdf),
            'input_ids': self.pad(self.input_ids[tdf['index']]),
            'segment_ids': self.pad(self.segment_ids[tdf['index']]),
            'input_mask': self.pad(self.input_mask[tdf['index']]),
            }


if __name__ == '__main__':
    o = MIMIC_III_Notes_BertFeature_Reader()
    #res = read_cached_bert_features()
    #res2 = read_all_bert_features()