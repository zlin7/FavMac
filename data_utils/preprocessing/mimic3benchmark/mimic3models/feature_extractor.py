from __future__ import absolute_import, print_function

import ipdb
import numpy as np
import pandas as pd
import tqdm
from scipy.stats import skew

all_functions = [min, max, np.mean, np.std, skew, len]
all_function_names = ['min', 'max', 'mean', 'std', 'skew', 'len']

def _quiet_skew(x):
	if np.std(x) == 0:
		return np.NaN
	return skew(x)

agg_funcs = {'min': np.min, 'max': np.max, 'mean': np.mean, 'std': np.std, 'skew': _quiet_skew, 'len': len}
agg_map = {
	"all": agg_funcs,
	"len": {'len': agg_funcs['len']},
	"all_but_len": {_[0]: _[1] for _ in agg_funcs.items() if _[0] != 'len'}
}
functions_map = {
	"all": all_functions,
	"len": [len],
	"all_but_len": all_functions[:-1]
}

periods_map = {
	"all": (0, 0, 1, 0),
	"first4days": (0, 0, 0, 4 * 24),
	"first8days": (0, 0, 0, 8 * 24),
	"last12hours": (1, -12, 1, 0),
	"first25percent": (2, 25),
	"first50percent": (2, 50)
}

sub_periods = [(2, 100), (2, 10), (2, 25), (2, 50),
			   (3, 10), (3, 25), (3, 50)]

sub_periods_dict = {
	'f100': (2, 100), 'f10': (2, 10), 'f25': (2, 25), 'f50': (2, 50),
	'l10': (3, 10),  'l25': (3, 25), 'l50': (3, 50)
}


def get_range(begin, end, period):
	# first p %
	if period[0] == 2:
		return (begin, begin + (end - begin) * period[1] / 100.0)
	# last p %
	if period[0] == 3:
		return (end - (end - begin) * period[1] / 100.0, end)

	if period[0] == 0:
		L = begin + period[1]
	else:
		L = end + period[1]

	if period[2] == 0:
		R = begin + period[3]
	else:
		R = end + period[3]
	return (L, R)



def calculate_clean(ser, period, sub_period, functions, eps=1e-6):
	ser = ser.dropna()
	if len(ser) == 0:
		return pd.Series(np.NaN, index=functions.keys())
	L, R = get_range(ser.index[0], ser.index[-1], period)
	L, R = get_range(L, R, sub_period)
	ser = ser[(ser.index < R+eps) & (ser.index > L - eps)]
	if len(ser) == 0:
		return pd.Series(np.NaN, index=functions.keys())
	return pd.Series({key: _fn(ser.values) for key, _fn in functions.items()})


def extra_features_clean(ser:pd.Series, name=None, period='all', features='all'):
	name = name or ser.name
	functions = agg_map[features]
	_period = periods_map[period]
	ret = []
	for sub_period, _sub_period in sub_periods_dict.items():
		tser = calculate_clean(ser, _period, _sub_period, functions)
		tser.index = tser.index.map(lambda s: f'{name}|{period}|{sub_period}|{s}')
		ret.append(tser)
	return pd.concat(ret)