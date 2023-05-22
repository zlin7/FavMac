import bisect
import functools
import os
from importlib import reload

import ipdb
import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset

import _settings
from data_utils.common import *


def _get_default_dataset_class(dataset):
    if dataset.startswith('LDS-'):
        import data_utils.posttrain
        return data_utils.posttrain.Logits2SetFn
    if dataset.startswith('LDSFP-'):
        import data_utils.posttrain
        return data_utils.posttrain.Logits2numFP
    if dataset.startswith('LOGIT-'):
        import data_utils.posttrain
        return data_utils.posttrain.LogitsWrapper
    if dataset == _settings.MNISTSup_NAME:
        import data_utils.preprocessing.mnist_multilabel
        return data_utils.preprocessing.mnist_multilabel.SuperimposeSyntheticData
    if dataset.startswith(_settings.CLAIMDEMO_NAME):
        import data_utils.preprocessing.claim_demo
        return data_utils.preprocessing.claim_demo.ClaimDemoData
    if dataset == _settings.MIMICIIICompletion_NAME:
        import data_utils.mimic3
        return data_utils.mimic3.MIMICIII_CompletionOnlineDynamic
    raise ValueError(dataset)

@functools.lru_cache()
def _cached_get_default_dataset(dataset, split=VALID, seed=_settings.RANDOM_SEED, **kwargs):
    kwargs = kwargs.copy()
    _dataset_class = _get_default_dataset_class(dataset)
    return _dataset_class(split=split, seed=seed, **kwargs)

def get_default_dataset(dataset, split=VALID, seed=_settings.RANDOM_SEED, **kwargs):
    kwargs = kwargs.copy()
    _dataset_class = _get_default_dataset_class(dataset)
    prefix, dataset = _clean_dataset_name(dataset, get_prefix=True)
    if dataset == _settings.CLAIMDEMOSeq_NAME:
        seed = None
    if prefix is None:
        return _cached_get_default_dataset(dataset, split, seed, **kwargs)
    if prefix in {'LDS', 'LDSFP', 'LOGIT'}:
        _base_datakwargs = kwargs.pop('datakwargs', {})
        if seed != _settings.RANDOM_SEED and dataset != _settings.CLAIMDEMOSeq_NAME:
            assert 'seed'  not in _base_datakwargs
            _base_datakwargs['seed'] = seed
        return _dataset_class.initialize(split=split, dataset=dataset, datakwargs=_base_datakwargs, **kwargs)
    return _cached_get_default_dataset(dataset, split, seed, **kwargs)

def _clean_dataset_name(dataset, get_prefix=False):
    for prefix in ['LDSFP', 'LDS', 'LOGIT']:
        if dataset.startswith(prefix+"-"):
            dataset = dataset.split(prefix+"-")[1]
            if get_prefix: return prefix, dataset
    if get_prefix: return None, dataset
    return dataset

def get_class_names(dataset):
    _dataset_class = _get_default_dataset_class(_clean_dataset_name(dataset))
    return _dataset_class.CLASSES

def get_nclasses(dataset):
    return len(get_class_names(dataset))

if __name__ == '__main__':
    o1 = get_default_dataset(_settings.MIMICIIICompletion_NAME, **{"use_notes": True, 'hcc_choice': 'more'})
    pass
