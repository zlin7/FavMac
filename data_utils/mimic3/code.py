import functools
import os
from importlib import reload
from typing import Optional, Union

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import tqdm

import _settings
from data_utils.mimic3.utils import read_table
from data_utils.preprocessing.code_utils import (HCC2014, icd9_to_3digits,
                                                 icd9_to_hcc)


def sample_one_record(df, keep_proba, target_hccs_mapping, ps=None, rs=None):
	if rs is None:
		rs = np.random.RandomState()
	curr_hidden_hccs = df['Y'].unique()
	curr_input_hccs = set()
	y_hidden = np.zeros(len(target_hccs_mapping) + 1, dtype=int)
	y_input = np.zeros(len(target_hccs_mapping) + 1, dtype=int)
	for yi in curr_hidden_hccs:
		y_hidden[yi] = 1
		if ps is None:
			if yi != 0 and rs.uniform(0,1) > keep_proba: continue
		else:
			if yi != 0 and ps[yi] > keep_proba: continue
		curr_input_hccs.add(yi)
		y_input[yi] = 1
	x_diag_input = sorted(df[df['Y'].isin(curr_input_hccs)]['DIAG'].unique())
	return y_hidden, y_input, x_diag_input

@functools.lru_cache()
def get_ICD9_codes(perc: float = 0.97, add_hcc_targets: bool=True, icd_3digit: bool=True):
	df = read_table("DIAGNOSES_ICD")
	if icd_3digit:
		df['ICD9_CODE'] = df['ICD9_CODE'].map(icd9_to_3digits)
	cnt = df.groupby('ICD9_CODE').size().sort_values(ascending=False)
	cumu_perc = (cnt / cnt.sum()).cumsum()
	codes = cumu_perc[cumu_perc < perc].index
	if add_hcc_targets:
		hcc_targets = pd.Index(icd9_to_hcc(icd_3digit=False).dropna(subset=['HCC_DESC'])['ICD9_CODE'].unique())
		if icd_3digit:
			hcc_targets = pd.Index(hcc_targets.map(icd9_to_3digits).unique())
		codes = codes.union(hcc_targets)
	return pd.Series({k: i+1 for i,k in enumerate(sorted(codes))})

@ptd.persistf()
def _get_all_diags(subjects_root_path=_settings.MIMIC_PREPROCESS_OUTPUT):
	subjects = sorted(pd.Index(list(filter(str.isdigit, os.listdir(subjects_root_path)))))
	diags = []
	for subject in tqdm.tqdm(subjects):
		tdf = pd.read_csv(os.path.join(subjects_root_path, subject, 'diagnoses.csv'), dtype={"ICD9_CODE":str})
		diags.append(tdf.reindex(columns=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE', 'ICUSTAY_ID']))
	diags = pd.concat(diags, ignore_index=True)
	hcc_mapping = icd9_to_hcc(icd_3digit=False)
	diags = diags.reindex(columns=['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE', 'ICUSTAY_ID']).reset_index()
	diags['ICD9_CODE_3'] = diags['ICD9_CODE'].map(icd9_to_3digits)
	diags = diags.merge(hcc_mapping.reindex(columns=['ICD9_CODE', 'HCC_CODE']), on='ICD9_CODE', how='left')
	return diags

def get_full_diags(
	subjects_root_path=_settings.MIMIC_PREPROCESS_OUTPUT,
	icd_3digit: bool=True,
	icd_cutoff: Optional[Union[float,int]]=0.97,
	hcc_choice: str='few'
	):
	diags = _get_all_diags(subjects_root_path)
	icd_col = 'ICD9_CODE_3' if icd_3digit else 'ICD9_CODE'
	if isinstance(icd_cutoff, float):
		keep_codes = get_ICD9_codes(icd_cutoff, icd_3digit=icd_3digit)
		#TODO: Change this
	else:
		assert isinstance(icd_cutoff, int) and icd_cutoff > 1
		keep_codes = diags[icd_col].value_counts()
		keep_codes = keep_codes[keep_codes >= icd_cutoff]
	diags['DIAG'] = diags[icd_col].map(lambda c: keep_codes.get(c, 0))
	if hcc_choice is None:
		# ICD
		code_mapping = {code: i for i, code in enumerate(keep_codes.index)}
		diags['Y'] = diags[icd_col].map(lambda c: code_mapping.get(c, 0))
	else:
		code_mapping = HCC2014.encode_dict(hcc_choice)
		diags['Y'] = diags['HCC_CODE'].map(lambda c: code_mapping.get(c, 0))
	return diags.drop('index', axis=1), keep_codes, code_mapping

@ptd.persistf()
def generate_label_completion(
	seed=_settings.RANDOM_SEED,
	icd_3digit: bool=True,
	icd_cutoff: Optional[float]=1.0,
	keep_proba: float=0.7,
	subjects_root_path=_settings.MIMIC_PREPROCESS_OUTPUT,
	hcc_choice: str='few'):
	"""
	Generate synthetic data with missing labels
	"""
	diags, keep_codes, code_mapping = get_full_diags(subjects_root_path, icd_3digit, icd_cutoff, hcc_choice=hcc_choice)
	rs = np.random.RandomState(seed)
	# diags.groupby("HADM_ID")['HCC_DESC'].nunique().mean() = 0.88
	# 0.5475 of HADM has any target HCC. Mean is 1.623 for the ones WITH any target HCC
	x_diags = {}
	input_codes = pd.DataFrame(columns=np.arange(len(code_mapping)+1))
	hidden_codes = pd.DataFrame(columns=np.arange(len(code_mapping)+1))
	for hadm_id, df in tqdm.tqdm(diags.groupby("HADM_ID")):
		hidden_codes.loc[hadm_id], input_codes.loc[hadm_id], x_diags[hadm_id] = sample_one_record(df, keep_proba, code_mapping, rs=rs)
	return x_diags, input_codes, hidden_codes, keep_codes, code_mapping

class MIMICIII_DiagCodes_Reader():
	def __init__(self, keep_proba: Optional[float]=0.7,
		seed=_settings.RANDOM_SEED,
		icd_3digit: bool=True, icd_cutoff: Optional[float]=1., subjects_root_path=_settings.MIMIC_PREPROCESS_OUTPUT,
		hcc_choice: str='few') -> None:
		self.keep_proba = keep_proba
		self.seed = seed
		self.rs = np.random.RandomState(seed)
		if keep_proba is not None:
			X_diag, self.X_partial, self.label_hidden, diag_mapping, self.code_mapping = generate_label_completion(seed=seed, icd_3digit=icd_3digit, icd_cutoff=icd_cutoff, keep_proba=keep_proba, subjects_root_path=subjects_root_path, hcc_choice=hcc_choice)
			max_num_diags = pd.Series(X_diag).map(len).max()
			self.X_diag = pd.DataFrame({hadm_id: np.asarray(_ + [np.NaN] * (max_num_diags-len(_)))for hadm_id, _ in X_diag.items()}).T
			self.label_hidden = self.label_hidden - self.X_partial
			self.diags, self.keep_codes = None, None
		else:
			self.diags, self.keep_codes, self.code_mapping = get_full_diags(subjects_root_path, icd_3digit, icd_cutoff, hcc_choice=hcc_choice)
			self.max_num_diags = self.diags.groupby('HADM_ID')['DIAG'].nunique().max()
			self.diags = {hadm_id: _ for hadm_id, _ in self.diags.groupby('HADM_ID')}

			self.X_partial = {}
			self.label_hidden = {}
			self.ps = np.random.RandomState(seed).uniform(size=(len(self.diags), len(self.code_mapping)+1))
			self.ps = {hadm_id: self.ps[_] for _, hadm_id in enumerate(self.diags.keys())}


	def sample_missing_labels(self, keep_proba, hadm_id):
		y_hidden, y_input, x_diag_input = sample_one_record(self.diags[hadm_id], keep_proba, self.code_mapping, ps=self.ps[hadm_id], rs=self.rs)
		x_diag_input = np.asarray(x_diag_input + [np.NaN] * (self.max_num_diags-len(x_diag_input)))
		return y_input, y_hidden - y_input

	def __getitem__(self, hadm_id):
		if self.keep_proba is not None:
			assert hadm_id[1] is None
			hadm_id = hadm_id[0]
			x = self.X_partial.loc[hadm_id].values
			y = self.label_hidden.loc[hadm_id].values
		else:
			hadm_id, keep_proba = hadm_id
			if hadm_id not in self.X_partial:
				x, y = self.sample_missing_labels(keep_proba, hadm_id)
				self.X_partial[hadm_id] = (x, keep_proba)
				self.label_hidden[hadm_id] = y
			else:
				assert keep_proba == self.X_partial[hadm_id][1]
				x, y = self.X_partial[hadm_id][0], self.label_hidden[hadm_id]
		return x[1:].astype(int), y[1:].astype(int)

if __name__ == '__main__':
	#o = MIMICIII_DiagCodes_Reader(icd_cutoff=1.0)

	# predicting from scratch, all HCC code, all ICD codes are kept.
	#o = MIMICIII_DiagCodes_Reader(keep_proba=0., hcc_choice='all', icd_cutoff=1.)
	# res = get_full_diags(hcc_choice=None, icd_cutoff=100)
	o = MIMICIII_DiagCodes_Reader(hcc_choice=None, icd_cutoff=100)