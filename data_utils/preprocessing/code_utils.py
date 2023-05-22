

from functools import lru_cache
import os
import openpyxl # pd.read_excel

import pandas as pd
import numpy as np
import ipdb
# https://www.cms.gov/medicare/health-plans/medicareadvtgspecratestats/risk-adjustors/2023-model-software/icd-10-mappings
# 2023 Midyear software for ICD-10
MAPPING_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
ICD10TOHCC_MAPPING_PATH = os.path.join(MAPPING_DATA_PATH, '2022 Midyear_Final ICD-10-CM Mappings.xlsx') # https://www.cms.gov/medicarehealth-plansmedicareadvtgspecratestatsrisk-adjustors/2022-model-softwareicd-10-mappings
RXHCC_RISKFACTOR_PATH = os.path.join(MAPPING_DATA_PATH, '2022_riskadjustmentmodelcoef.txt')
HCC_RISKFACTOR2014_PATH = os.path.join(MAPPING_DATA_PATH, '2014_riskadjustmentmodelcoef.txt')
HCCV24_RISKFACTOR_PATH = os.path.join(MAPPING_DATA_PATH, 'model', 'c2419p1m.csv')

ICD10TOHCCV24_RISKFACTOR_PATH = os.path.join(MAPPING_DATA_PATH, 'model', 'F2423P1M.TXT')
HCCV24_DESC_PATH = os.path.join(MAPPING_DATA_PATH, 'model', 'V24H86L1.TXT')

def normalize_icd10(code: str):
	"""Standardizes ICD10CM code."""
	if "." in code:
		return code
	if len(code) <= 3:
		return code
	return code[:3] + "." + code[3:]

def icd10_to_3digits(code: str):
	if not isinstance(code, str): return code
	return normalize_icd10(code).split(".")[0]
	
def normalize_icd9(code: str):
	"""Normalize ICD9 code"""
	if not isinstance(code, str): return code
	if code.startswith('E'):
		assert len(code) >= 4
		if len(code) == 4:
			return code
		return code[:4] + '.' + code[4:]
	else:
		assert len(code) >= 3
		if len(code) == 3:
			return code
		return code[:3] + '.' + code[3:]

def icd9_to_3digits(code: str):
	if not isinstance(code, str): return code
	return normalize_icd9(code).split(".")[0]

def icd10_to_hccv24():
	#NOTE: The following file somehow misses one mapping (out of ~2600). 
	# df = pd.read_csv(ICD10TOHCCV24_RISKFACTOR_PATH, sep='\t', dtype=str, header=None).set_index(0).rename(columns={0:"HCC_CODE"})[1]
	# return df
	df = pd.read_excel(ICD10TOHCC_MAPPING_PATH, sheet_name='FY21-FY22 ICD10 Payment Codes', skiprows=3, dtype=str)
	df.columns = [_.replace("\n", ' ') for _ in df.columns]
	col_maps = {
		'Diagnosis Code': 'ICD10', 
		'Description': 'ICD_DESC',
		'CMS-HCC Model Category V24': 'CMSHCC_V24'
		}
	df = df.rename(columns=col_maps)
	return df.reindex(columns=list(col_maps.values())).dropna(how='all')
	
def icd10_to_hcc():
	df = pd.read_excel(ICD10TOHCC_MAPPING_PATH, sheet_name='FY21-FY22 ICD10 Payment Codes', skiprows=3, dtype=str)
	df.columns = [_.replace("\n", ' ') for _ in df.columns]
	col_maps = {
		"RxHCC Model Category V05": "RxHCC", 
		'Diagnosis Code': 'ICD10', 
		'Description': 'ICD_DESC',
		'CMS-HCC Model Category V24': 'CMSHCC_V24'
		}
	df = df.rename(columns=col_maps)
	return df.reindex(columns=list(col_maps.values())).dropna(how='all')

def hcc_risk_factors_2014():
	with open(HCC_RISKFACTOR2014_PATH) as fin:
		lines = fin.readlines()
	assert len(lines) % 4 == 0
	lines = np.asarray([_.strip() for _ in lines]).reshape(-1, 4)
	df = pd.DataFrame(lines[1:], columns=lines[0])
	df['HCC_CODE'] = df['Disease Coefficients'].map(lambda x: x.replace("HCC", ""))
	df = df.rename(columns={"Description Label": "HCC_DESC"})
	df['RISK_FACTOR'] = df['Community'].astype(float)
	return df.drop('Disease Coefficients', axis=1).reindex(columns=['HCC_CODE', 'HCC_DESC', 'RISK_FACTOR'])

def rxhcc_risk_factors():
	with open(RXHCC_RISKFACTOR_PATH) as fin:
		lines = fin.readlines()
	assert len(lines) % 7 == 0
	lines = np.asarray([_.strip() for _ in lines]).reshape(-1, 7)
	df = pd.DataFrame(lines[1:], columns=lines[0])
	df['HCC'] = df['HCC or RXC No.'].map(lambda x: x.replace("HCC", ""))
	return df


def _read_cmshccv24_desc():
	with open(HCCV24_DESC_PATH, 'r') as fin:
		lines = fin.readlines()
	idx = lines.index(' LABEL\n')
	ret = {}
	for line in lines[idx+1:]:
		if not line.startswith(" HCC"): break
		line = line[4:].strip().split(" =")
		ret[line[0]] = line[1].strip('"')
	return pd.Series(ret)

@lru_cache()
def read_cmshccv24_coefs(col='CNA'):
	"""
	Reference:
	https://github.com/galtay/hcc_risk_models/blob/master/docs/index.md
	https://www.cms.gov/Medicare/Health-Plans/MedicareAdvtgSpecRateStats/Downloads/RiskAdj2017ProposedChanges.pdf

    "CFA": "Community Full Benefit Dual Aged",
    "CFD": "Community Full Benefit Dual Disabled",
    "CNA": "Community NonDual Aged", #This is 70.9% of population, mean of actual cost = $8,932
    "CND": "Community NonDual Disabled",
    "CPA": "Community Partial Benefit Dual Aged",
    "CPD": "Community Partial Benefit Dual Disabled",
    "INS": "Long Term Institutional"
	"""
	df = {}
	for key, v in pd.read_csv(HCCV24_RISKFACTOR_PATH).set_index('_NAME_')['COL1'].items():
		key = key.split("_")
		_model = key[0]
		key = "_".join(key[1:])
		if _model not in df: df[_model] = {}
		df[_model][key] = v
	df = pd.DataFrame(df).reindex(columns=['CFA', 'CFD', 'CPA', 'CPD', 'CND', 'CNA', 'INS'])
	df = df[df.index.map(lambda x: x.startswith("HCC"))]
	df.index = df.index.str.replace("HCC", "")
	df['HCC_DESC'] = _read_cmshccv24_desc()
	return df.dropna().reindex(columns=[col, 'HCC_DESC'])

def icd9_to_hcc(icd_3digit=False):
	icd9_to_hcc = pd.read_csv(os.path.join(MAPPING_DATA_PATH, 'icd2hccxw2014.csv'), dtype={"hcc": str})
	hccrf = hcc_risk_factors_2014()
	ret = icd9_to_hcc.rename(columns={"icd": "ICD9_CODE", "hcc": "HCC_CODE"}).reindex(columns=['ICD9_CODE', 'HCC_CODE'])
	ret = ret.merge(hccrf, on='HCC_CODE')
	if icd_3digit:
		ret['ICD9_CODE'] = ret['ICD9_CODE'].map(icd9_to_3digits)
		ret = ret.drop_duplicates()
	return ret

class HCC2014:
	CHOICE_CODES = ['96', '85', '135', '84', '19', '8', '18', '9', '136', '27']
	FREQ_CODES = ["96","85","135","84","19","111","2","86","114","48","108","79","8","18","167","9","107","55","99","87","33","176","80","136","28","100","21","58","54","27","23","40","169","29","10"]

	@classmethod
	def encode_dict(cls, mode='few'):
		if mode == 'few':
			hcc_codes = cls.CHOICE_CODES
		else:
			assert mode == 'more'
			hcc_codes = cls.FREQ_CODES
		target_hccs_mapping = pd.Series({k: i+1 for i, k in enumerate(sorted(hcc_codes))})
		return target_hccs_mapping

	@classmethod
	def decode_dict(cls, mode='few'):
		return {i:k for k,i in cls.encode_dict(mode).items()}

	@classmethod
	def get_risk_factors(cls, mode='few'):
		rf = hcc_risk_factors_2014().set_index('HCC_CODE')['RISK_FACTOR']
		if mode is None: return rf
		decode_ser = cls.decode_dict(mode)
		rf = rf.reindex(['Other'] + list(pd.Series(decode_ser).sort_index().values))
		return rf.dropna()
		# C_max = 6.558 or 16.475

class HCCV24:
	FREQ_CODES = ["19","18","55","85","59","12","136","22","40","111","96","48","79","23","108","8","11","9","84","10","134","2","135","35","52","161","103","47","100","77","46"]
	CHOICE_CODES = ["19","18","85","136","96","8","9","84","135","46"]
	@classmethod
	def encode_dict(cls, mode='few'):
		if mode == 'few':
			hcc_codes = cls.CHOICE_CODES
		else:
			assert mode == 'more'
			hcc_codes = cls.FREQ_CODES
		target_hccs_mapping = pd.Series({k: i+1 for i, k in enumerate(sorted(hcc_codes))})
		return target_hccs_mapping

	@classmethod
	def decode_dict(cls, mode='few'):
		return {i:k for k,i in cls.encode_dict(mode).items()}
	
	@classmethod
	def get_risk_factors(cls, mode='few'):
		rf = read_cmshccv24_coefs('CNA')['CNA']
		if mode is None: return rf
		decode_ser = cls.decode_dict(mode)
		rf = rf.reindex(['Other'] + list(pd.Series(decode_ser).sort_index().values))
		rf['Other'] = {"more":  0.425121795, "few": 0.342027118}[mode]
		return rf
	# C_max = 7.41 or 14.87

if __name__ == '__main__':
	pass