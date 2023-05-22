import functools
import os

import pandas as pd

import _settings


@functools.lru_cache()
def read_table(table_name, nrows=None):
	fpath = os.path.join(_settings.MIMIC_PATH, f"{table_name}.csv.gz")
	df = pd.read_csv(fpath, compression='gzip', on_bad_lines=None, nrows=nrows)
	return df.set_index("ROW_ID", verify_integrity=True)

