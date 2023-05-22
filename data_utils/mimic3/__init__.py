from typing import Optional

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
from torch.utils.data import Dataset

import _settings
from data_utils.common import TEST, TRAIN, VALID, VALIDTEST, get_split_indices
from data_utils.preprocessing.code_utils import HCC2014

from .code import MIMICIII_DiagCodes_Reader, get_full_diags
from .events import MIMIC_III_Event_Reader
from .notes import MIMIC_III_Notes_BertFeature_Reader


@ptd.persistf()
def get_filtered_hadm_ids(
    icd_3digit: bool=True,
    icd_cutoff: Optional[float]=0.97,
    subjects_root_path=_settings.MIMIC_PREPROCESS_OUTPUT):
    diags, _, _ = get_full_diags(subjects_root_path, icd_3digit, icd_cutoff)

    X_events = MIMIC_III_Event_Reader()
    X_bert_features = MIMIC_III_Notes_BertFeature_Reader(max_blocks=1)
    cidx = pd.Index(diags['HADM_ID'].unique()).intersection(X_events.hadm_ids)
    cidx = cidx.intersection(X_bert_features.hadm_ids)
    return cidx.sort_values()

class MIMICIII_CompletionOnlineDynamic(Dataset):
    DATASET = _settings.MIMICIIICompletion_NAME

    MODE_STATIC, MODE_LINEAR, MODE_SINE = 0, 1, 2
    def __init__(self,  split=TRAIN, train_ratio=0.9, test_ratio = 0.15,
                 seed=_settings.RANDOM_SEED, keep_proba_mode= MODE_STATIC, keep_proba=0.7,
                 use_notes=True, icd_cutoff=0.97,
                 hcc_choice: str='few', max_blocks=1):
        super().__init__()


        self.X_events = MIMIC_III_Event_Reader(episodes=None)

        full_indices = get_filtered_hadm_ids(icd_cutoff=icd_cutoff)
        self.keep_proba_mode, self.keep_proba = keep_proba_mode, keep_proba
        self.diags_reader = MIMICIII_DiagCodes_Reader(keep_proba=keep_proba if keep_proba_mode == self.MODE_STATIC else None,
                                                      hcc_choice=hcc_choice, icd_cutoff=icd_cutoff)
        self.bert_features = MIMIC_III_Notes_BertFeature_Reader(max_blocks=max_blocks)
        self.LABEL_MAP = {_[1]: _[0] for _ in self.diags_reader.code_mapping.items()}
        self.use_notes = use_notes

        _idx_splits = get_split_indices(seed, [train_ratio * (1-test_ratio), (1-train_ratio) * (1-test_ratio), test_ratio], n=len(full_indices))
        self.indices = full_indices[_idx_splits[split] if split != VALIDTEST else np.concatenate([_idx_splits[_] for _ in [VALID, TEST]])]

        self.CLASSES = [self.LABEL_MAP[i] for i in sorted(self.LABEL_MAP.keys())]

    def __len__(self):
        return len(self.indices)

    def get_keep_proba(self, idx):
        if self.keep_proba_mode == self.MODE_STATIC:
            return None
        elif self.keep_proba_mode == self.MODE_LINEAR:
            t = idx / len(self)
            return self.keep_proba + (1-self.keep_proba) * t
        elif self.keep_proba_mode == self.MODE_SINE:
            t = idx / len(self)
            return self.keep_proba + (1-self.keep_proba) * np.sin(t * 2 * np.pi)
        # random walk?
        raise ValueError()

    def __getitem__(self, idx):
        hadm_id = self.indices[idx]
        all_data = {'event': self.X_events[hadm_id].values.astype(np.float32)} #NOTE: a vector for now,
        if self.use_notes: all_data.update(self.bert_features[hadm_id])
        out = {"data": all_data,  "index": hadm_id}
        keep_proba = self.get_keep_proba(idx)
        all_data['partialHCC'], out['target'] = self.diags_reader[(hadm_id, keep_proba)]
        out['target'] = out['target'].astype(np.float32)
        out['keep_proba'] = all_data['keep_proba'] = self.keep_proba if keep_proba is None else keep_proba
        return out
