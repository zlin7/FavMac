

import bisect
import os
from importlib import reload

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import torch
import tqdm
from torch.utils.data import Dataset

import _settings

TRAIN = 'train'
VALID = 'val'
TEST = 'test'
VALIDTEST = 'valtest'

TRAIN1 = 'train1'
VALID1 = 'val1'

SEED_OFFSETS = {TRAIN: 101, VALID: 202, VALID1: 203, TEST: 303, TRAIN1:102}


class DatasetWrapper(Dataset):
	def __init__(self, split):
		super(DatasetWrapper, self).__init__()
		self.split = split
		assert hasattr(self, 'DATASET'), "Please give this dataset a name"
		assert hasattr(self, 'LABEL_MAP'), "Please give a name to each class {NAME: class_id}"

		self._mem = {}

		self.labels = None #the "Y"
		self.groups = None #e.g. patient

		self.multilabel = None #NOTE: the portion of labels used to compute matrices etc.
		self.indices = None

	def is_train(self):
		return self.split == TRAIN or self.split == TRAIN1
	def is_test(self):
		return self.split == TEST
	def is_valid(self):
		return self.split == VALID or self.split == VALID1
	def idx2pid(self, idx):
		if torch.is_tensor(idx): idx = idx.tolist()
		return idx

	def get_num_classes(self):
		return len(self.CLASSES)

	def get_class_frequencies(self):
		if hasattr(self, 'CLASS_FREQ'): return self.CLASS_FREQ
		raise NotImplementedError()

	#This function will be used for sampling purposes - that is, we want to reverse look-up indices that belong to a class
	def label_to_indices(self, k, **kwargs):
		if not hasattr(self, 'labels'): raise NotImplementedError()
		if '_label2indices_maps' not in self._mem:
			_label2indices_maps = {k: [] for k in range(len(self.CLASSES))}
			for i, y in enumerate(self.labels):
				_label2indices_maps[y].append(i)
			self._mem['_label2indices_maps'] = _label2indices_maps
		if k is None: return self._mem['_label2indices_maps']
		return self._mem['_label2indices_maps'][k]


	def get_group_ids(self): #For things like patient data, each patient is a group
		raise NotImplementedError()

	def get_all_X_Y_indices(self):
		X, Y, indices = [], [], []
		for x, y, idx in self:
			X.append(x)
			Y.append(y)
			indices.append(idx)
		return np.asarray(X), np.asarray(Y), np.asarray(indices)

	def get_labels(self):
		raise NotImplementedError() #This is too slow

@ptd.persistf()
def _get_perm(seed, n):
	assert seed is not None
	return np.random.RandomState(seed).permutation(n)

def get_split_indices(seed, split_ratio, n, names=None):
	# TRAIN1 and VALID1 split the training set into 80% and 20%.
	# VALID1 is used to train additional models, such as the DeepSet for FPCP baseline
	perm = _get_perm(seed, n) if seed is not None else np.arange(n)
	split_ratio = np.asarray(split_ratio).cumsum() / sum(split_ratio)
	cuts = [int(_s* n) for _s in split_ratio]
	if names is not None and len(names) == len(split_ratio):
		return {k: perm[cuts[i-1]:cuts[i]] if i > 0 else perm[:cuts[0]] for i, k in enumerate(names)}
	if len(split_ratio) == 3:
		return {TRAIN: perm[:cuts[0]], VALID:perm[cuts[0]:cuts[1]], TEST: perm[cuts[1]:],
				TRAIN1: perm[:int(0.8*cuts[0])], VALID1: perm[int(0.8*cuts[0]):cuts[0]]}
	else:
		assert len(split_ratio) == 2
		return {TRAIN: perm[:cuts[0]], VALID: perm[cuts[0]:],
				TRAIN1: perm[:int(0.8*cuts[0])], VALID1: perm[int(0.8*cuts[0]):cuts[0]]}

def get_split_indices_by_group(seed, split_ratio, groups, names=None):
	to_split = np.unique(groups)
	res = get_split_indices(seed, split_ratio, len(to_split), names=names)
	ret = {k: [] for k in res.keys()}
	group2indices = pd.DataFrame({"group": groups}).reset_index().groupby('group')['index'].agg(list)
	for key, group_ids in res.items():
		for group_id in group_ids:
			ret[key].extend(group2indices[to_split[group_id]])
	return ret


from functools import lru_cache


@lru_cache(10)
def _cached_np_load(path):
	return np.load(path)
def cached_np_load(path):
	return _cached_np_load(os.path.abspath(path))


def onehot_to_cond_prob(multilabel):
	"""
	Input: numpy array, where multilabel[i, k] means the i-th sample has label k
	Output: a matrix *mat*, where mat[k, k1] = P(k1 \in Y | k \in Y)
	"""
	def _onehot_to_cond_prob_np(multilabel):
		mat = []
		for k in range(multilabel.shape[1]):
			msk = multilabel[:, k] == 1
			mat.append(multilabel[msk].sum(0) / msk.sum())
		mat = np.stack(mat)
		return mat
	if isinstance(multilabel, pd.DataFrame):
		return pd.DataFrame(_onehot_to_cond_prob_np(multilabel.values),
					index=multilabel.columns, columns=multilabel.columns)
	return _onehot_to_cond_prob_np(multilabel)


def numpy_one_hot(x, depth, exclude=-1):
	x = np.copy(x)
	x[x >= depth] = exclude
	msk = x==exclude
	ret = np.eye(depth)[x]
	ret[msk] = 0
	return ret


def _sample_classes_with_seed(seed, labels, n_labels_keep):
	BIG_PRIME = 100003
	BIG_PRIME2 = 1747591
	N, K = labels.shape
	assert n_labels_keep < K
	offsets = np.random.RandomState(seed).permutation(N)

	new_labels = np.zeros_like(labels)

	for j, offset in tqdm.tqdm(enumerate(offsets), desc='generating sampled labels', total=N):
		_seed = (seed + BIG_PRIME * offset) % BIG_PRIME2
		perm = np.random.RandomState(_seed).permutation(K)
		cnt = 0
		for i, p_i in enumerate(perm):
			if labels[j, p_i] == 1:
				cnt += 1
				new_labels[j, p_i] = 1
			if cnt >= n_labels_keep: break
	return new_labels