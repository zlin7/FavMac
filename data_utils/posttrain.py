import functools
import os
from functools import lru_cache
from importlib import reload
from typing import Optional

import ipdb
import numpy as np
import pandas as pd
import torch.nn as nn
import tqdm

import _settings
from data_utils.common import (TEST, TRAIN, VALID, VALIDTEST, DatasetWrapper,
                               _sample_classes_with_seed, get_split_indices,
                               get_split_indices_by_group, numpy_one_hot,
                               onehot_to_cond_prob)


class Logits2numFP(DatasetWrapper):
    DATASET = None
    LABEL_MAP = None

    def __init__(self, logits, labels, index, dataset, niters=5000, seed=_settings.RANDOM_SEED, split=VALIDTEST):
            super().__init__(split)
            self.logits = logits.astype(np.float32)
            self.labels = labels.astype(int)
            self.indices = index

            self.DATASET = dataset
            self.LABEL_MAP = {i: i for i in range(labels.shape[1])}

            self.masks = np.random.RandomState(seed).randint(0, 2, size=(niters, labels.shape[1]))

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        s = self.masks[idx]
        midx = idx % len(self.labels)
        y = self.labels[midx]
        data = {"data": self.logits[midx], 'mask': s}
        return {"data": data, "target": ((1-y)*s).sum(), "index": f"{self.indices[midx]}-{idx}"}

    @classmethod
    def initialize(cls, keymode, split, dataset, datakwargs, **kwargs):
        import pipeline.main
        key, mode = keymode.split("|") # key|last
        preds = pipeline.main.read_prediction(key, dataset, split, datakwargs, mode=mode)
        nclass = 0
        while 'S%d'%nclass in preds.columns: nclass += 1
        logits = preds.reindex(columns=['S%d'%i for i in range(nclass)])
        labels = preds.reindex(columns=['Y%d'%i for i in range(nclass)])
        return cls(logits.values, labels.values, logits.index, dataset, split=split, **kwargs)

class Logits2SetFn(DatasetWrapper):
    DATASET = None
    LABEL_MAP = None

    def __init__(self, logits, labels, index, dataset, set_fn, niters=5000, seed=_settings.RANDOM_SEED, split=VALIDTEST):
            super().__init__(split)
            from data_utils.set_function import get_set_fn
            set_fn = get_set_fn(set_fn)
            self.logits = logits.astype(np.float32)
            self.labels = labels.astype(int)
            self.indices = index

            self.DATASET = dataset
            self.LABEL_MAP = {i: i for i in range(labels.shape[1])}

            self.masks = np.random.RandomState(seed).randint(0, 2, size=(niters, labels.shape[1]))
            self.set_fn = set_fn

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        s = self.masks[idx]
        midx = idx % len(self.labels)
        y = self.labels[midx]
        data = {"data": self.logits[midx], 'mask': s}
        ret = {"data": data, "target": np.float32(self.set_fn(s, Y=y)), "index": f"{self.indices[midx]}-{idx}"}
        return ret

    @classmethod
    def initialize(cls, keymode, split, dataset, datakwargs, set_fn, **kwargs):
        import pipeline.main
        key, mode = keymode.split("|") # key|last
        preds = pipeline.main.read_prediction(key, dataset, split, datakwargs, mode=mode)
        nclass = 0
        while 'S%d'%nclass in preds.columns: nclass += 1
        logits = preds.reindex(columns=['S%d'%i for i in range(nclass)])
        labels = preds.reindex(columns=['Y%d'%i for i in range(nclass)])
        return cls(logits.values, labels.values, logits.index, dataset, set_fn, split=split, **kwargs)

class LogitsWrapper(DatasetWrapper):
    DATASET = None
    LABEL_MAP = None

    def __init__(self, logits:np.ndarray, labels:np.ndarray, index, dataset, seed=_settings.RANDOM_SEED, split=VALIDTEST):
            super().__init__(split)
            self.logits = logits.astype(np.float32)
            self.labels = labels.astype(int)
            self.indices = index

            self.DATASET = dataset
            self.LABEL_MAP = {i: i for i in range(labels.shape[1])}

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, idx):
        return {
            'data': self.logits[idx],
            'target': self.labels[idx],
            'index': f"{self.indices[idx]}"
        }

    @classmethod
    def initialize(cls, keymode, split, dataset, datakwargs, **kwargs):
        import pipeline.main
        key, mode = keymode.split("|") # key|last
        preds = pipeline.main.read_prediction(key, dataset, split, datakwargs, mode=mode)
        nclass = 0
        while 'S%d'%nclass in preds.columns: nclass += 1
        logits = preds.reindex(columns=['S%d'%i for i in range(nclass)])
        labels = preds.reindex(columns=['Y%d'%i for i in range(nclass)])
        return cls(logits.values, labels.values, logits.index, dataset, split=split, **kwargs)

if __name__ == '__main__':
    pass