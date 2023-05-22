import os

import ipdb
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import tqdm

import _settings
import data_utils.preprocessing.code_utils as code_utils
from data_utils.common import (TEST, TRAIN, TRAIN1, VALID, VALID1, VALIDTEST,
                               DatasetWrapper, _sample_classes_with_seed,
                               get_split_indices, get_split_indices_by_group,
                               numpy_one_hot, onehot_to_cond_prob)

CLAIM_ICD10_FIRST_DAY = '20151001'
CLAIM_LAST_DAY = '20191231'
MAX_OFFSET_DAYS = 9999

@ptd.persistf()
def _read_claim_df(fpath=os.path.join(_settings.CLAIMDEMO_PATH, f'claims_2019.dat')):
    print("Reading", fpath)
    dtypes = {c: str for c in ['diag%d'%i for i in range(1, 13)]}
    dtypes.update({c: str for c in ["pat_id"]})
    dtypes.update({'diagprc_ind': int})

    keep_cols = list(dtypes.keys()) + ['from_dt', 'to_dt']
    # keep_cols += ['pos'] # place of service. Need to perform additional filtering
    df = pd.read_csv(fpath, sep='|', dtype=dtypes)
    df['from_dt'] = df['from_dt'].map(pd.to_datetime).dt.date
    df['to_dt'] = df['to_dt'].map(pd.to_datetime).dt.date
    return df.reindex(columns=keep_cols)

def get_full_raw_df():
    fpaths = [os.path.join(_settings.CLAIMDEMO_PATH, f'claims_{yr}.dat') for yr in range(2015, 2020)]
    df = pd.concat([_read_claim_df(_) for _ in fpaths], ignore_index=True)
    return df

def get_aggregated_df(freq='daily'):
    cache_key = freq
    df = ptd.manual_cache(cache_key, obj=None)
    if df is None:
        df = get_full_raw_df()
        df = df[df['diagprc_ind'] == 2].drop(['diagprc_ind', 'to_dt'], axis=1)
        if freq == 'daily':
            df['date'] = df['from_dt']
        elif freq == 'weekly':
            offset = pd.offsets.Week(weekday=6)
            df['date'] = df['from_dt'].map(lambda x: (pd.to_datetime(x) + offset).date())
        else:
            raise NotImplementedError()
        df = df.drop('from_dt', axis=1)
        df = df.fillna('').groupby(['pat_id', 'date']).agg(lambda xs: set(filter(len, xs)))
        ptd.manual_cache(cache_key, obj=df, write=True)
    return df

def encode_icd10(min_cnt=200, to3digits=True):
    df = pd.DataFrame({"cnt": _icd_freq()[2]})
    df['3digit'] = df.index.map(code_utils.icd10_to_3digits)
    if to3digits:
        cnts = df.groupby('3digit')['cnt'].sum().sort_values(ascending=False)
        mapping = cnts[cnts >= min_cnt]
        mapping = pd.Series(np.arange(len(mapping)), mapping.index) + 1
        df['code'] = df['3digit'].map(lambda x: mapping.get(x, 0))
    else:
        mapping = df['cnt'][df['cnt'] >= min_cnt]
        mapping = pd.Series(np.arange(len(mapping)), mapping.index) + 1
        df['code'] = df.index.map(lambda x: mapping.get(x, 0))
    return dict(df['code'])

def get_processed_claim_df(freq='daily', min_cnt=200, to3digits=True):
    cache_key = f"{freq}_{min_cnt}_{to3digits}"

    ret = ptd.manual_cache(cache_key)
    if ret is None:
        icd10tohccv24_mapping = code_utils.icd10_to_hccv24().dropna(subset='CMSHCC_V24').groupby('ICD10')['CMSHCC_V24'].agg(set)
        icd10encoder = encode_icd10(min_cnt, to3digits)
        ret = []
        df = get_aggregated_df(freq)
        ndiags = len(df.columns)
        df.columns = df.columns.map(lambda c: int(c[4:]))
        for (pat_id, dt), r in tqdm.tqdm(df.iterrows(), total=len(df)):
            nr = [set() for _ in range(ndiags + 2)] + [pat_id, dt]
            min_seq = {}
            for seq, codes in r.items():
                if len(codes) == 0: continue
                for code in codes:
                    hcccodes = icd10tohccv24_mapping.get(code, set())
                    code = icd10encoder[code]
                    nr[0].add(code)
                    min_seq[code] = min(seq, min_seq.get(code, np.inf))
                    if len(hcccodes):
                        nr[ndiags+1] = nr[ndiags+1].union(hcccodes)
            for code, seq in min_seq.items():
                nr[seq].add(code)
            ret.append(nr)
        ret = pd.DataFrame(ret).rename(columns={ndiags + 2: 'pat_id', ndiags+3: 'date', 0: 'all', ndiags+1: 'hcc'})
        ret = ret.set_index(['pat_id', 'date'])
        ptd.manual_cache(cache_key, obj=ret, write=True)
    return ret

# Need a pre_window when we predict
def _add_hcc_future_label(df, post_window=90, label_only=True):
    if post_window is None:
        df['label_next_hcc'] = df['hcc'].shift(-1)
        df.loc[df.index[:-1], 'label_date'] = df.index[1:]
        if label_only: df = df.reindex(columns=['hcc', 'label_next_hcc', 'label_date'])
        return df
    post_window = pd.DateOffset(days=post_window)
    last_date = (pd.to_datetime(CLAIM_LAST_DAY) - post_window).date()
    # df belongs to one patient
    df = df.sort_index()
    labels = {}
    hccser = df['hcc']
    for i in range(len(df)):
        dt = df.index[i]
        if dt > last_date: break
        edt = (dt + post_window).date()
        currlabel = set()
        for cdt, _hccs in hccser.iloc[i+1:].items():
            assert cdt > dt
            if cdt > edt: break
            if len(_hccs) > 0:
                currlabel = currlabel.union(_hccs)
        labels[dt] = currlabel
    if label_only: df = df.reindex(columns=['hcc'])
    if len(labels) > 0:
        df[f'label_{post_window.days}_hcc'] = pd.Series(labels)
    return df

def get_processed_hcc_label(freq='daily', post_window=90, min_cnt=200, to3digits=True):
    cache_key = f"{freq}_{min_cnt}_{to3digits}_pw{post_window}"
    ret = ptd.manual_cache(cache_key)
    if ret is None:
        df = get_processed_claim_df(freq, min_cnt, to3digits)
        ret = []
        all_pats = df.groupby(level=0).size()
        all_pats = all_pats[all_pats > 1].index
        for pat_id in tqdm.tqdm(all_pats):
            pat_df = df.loc[pat_id]
            pat_df_wlabel = _add_hcc_future_label(pat_df, post_window, label_only=True)
            pat_df_wlabel['pat_id'] = pat_id
            ret.append(pat_df_wlabel.reset_index())
        ret = pd.concat(ret, ignore_index=True).set_index(['pat_id', 'date'])
        ptd.manual_cache(cache_key, obj=ret, write=True)
    return ret

def get_pat_df():
    return pd.read_csv(os.path.join(_settings.CLAIMDEMO_PATH, f'enroll_synth.dat'), sep='|').set_index('pat_id', verify_integrity=True)

@ptd.persistf()
def _icd_freq():
    df = get_full_raw_df()
    cols = ['diag%d'%i for i in range(1, 13)]
    df = df.reindex(columns=cols + ['diagprc_ind'])
    tdf10 = df[df['diagprc_ind'] == 2].drop('diagprc_ind', axis=1).stack().value_counts() #ICD-10
    tdf9 = df[df['diagprc_ind'] == 1].drop('diagprc_ind', axis=1).stack().value_counts()  # ICD-9
    tdf0 = df[df['diagprc_ind'] == -1].drop('diagprc_ind', axis=1).stack().value_counts()  # ICD-?
    return tdf0, tdf9, tdf10

#=================================================

class PatientDataReader():
    def __init__(self) -> None:
        self.data = get_pat_df().reindex(columns=['der_sex', 'der_yob', 'pat_region'])
        self.data['pat_region'].fillna('U', inplace=True)
        self.data['der_yob'].fillna(self.data['der_yob'].median(), inplace=True)
        assert self.data.count().min() == len(self.data)
        self.data['pat_region'] = self.data['pat_region'].map({'W':4, 'MW': 1, 'S': 2, 'E':3, 'U': 0})
        self.data['der_sex'] = self.data['der_sex'].map({'M':0, 'F': 1, 'U': 0}) # there is only 1 U

    def get_data(self, pat_id, date):
        age = (date.year-self.data.loc[pat_id, 'der_yob'])/100.
        return [float(age), int(self.data.loc[pat_id,'der_sex']), int(self.data.loc[pat_id,'pat_region'])]

def cache_data_by_patient(data_window=30, pooled=False, seq=False, freq='daily', min_cnt=200, to3digits=True):
    cache_key = f"{freq}_{min_cnt}_{to3digits}_dw{data_window}_pd{'Y' if pooled else 'N'}_seq{'Y' if seq else 'N'}"
    ret = ptd.manual_cache(cache_key)
    if ret is None:
        from collections import Counter
        df = get_processed_claim_df(freq, min_cnt, to3digits)
        df = df.reindex(columns=['all'] if pooled else list(range(1,13)))
        if data_window is None:
            assert not seq
            num_codes = df.apply(lambda r: r.map(len)).stack().value_counts()
            return df.apply(lambda r: r.map(list)), num_codes, None
        num_codes = Counter()
        if seq:
            num_codes = df.apply(lambda r: r.map(len)).stack().value_counts()
        seq_lens = Counter()
        data_window = pd.DateOffset(days=data_window)
        ret = {}
        all_pats = df.groupby(level=0).size()
        all_pats = all_pats[all_pats > 1].index
        for pat_id in tqdm.tqdm(all_pats):
            pat_df = df.loc[pat_id].sort_index()
            pat_df.index = pat_df.index.map(pd.to_datetime)
            seq_len = pat_df.index.map(lambda dt: len(pat_df.loc[dt-data_window:dt]))
            seq_lens += Counter(seq_len)
            if seq:
                pat_df = pat_df.apply(lambda r: r.map(list))
                pat_df['seq_len'] = seq_len
                for dt, r in pat_df.iterrows():
                    ret[(pat_id, dt)] = r
                continue
            for dt in pat_df.index:
                tdf = pat_df.loc[dt-data_window:dt]
                #nr = tdf.apply(lambda c: list(set.union(*c.values.tolist())))
                nr = pd.Series({c: list(set.union(*c_.values.tolist())) for c, c_ in tdf.items()})
                num_codes += Counter(nr.map(len))
                ret[(pat_id, dt)] = nr
        ret = {(k[0], k[1].date()): v for k, v in ret.items()}
        ret = (pd.DataFrame(ret).T, pd.Series(num_codes), pd.Series(seq_lens))
        ptd.manual_cache(cache_key, obj=ret, write=True)
    return ret


class ClaimDataReader():
    def __init__(self,
                 freq='weekly', pred_window=None, data_window=None,
                 topndiags=None, seq=False, rev_seq=True) -> None:
        _valid_offsets =  {'daily': [30, 90, 180, 365], 'weekly': [28, 13*7, 26*7, 52*7]}[freq]
        assert pred_window is None or pred_window in _valid_offsets
        self.labels = get_processed_hcc_label(freq=freq, post_window=pred_window).sort_index(level=1)
        self.data, _num_codes, _seq_lens = cache_data_by_patient(data_window, topndiags is None, seq, freq=freq)

        self.seq_len, self.max_seq_len = None, None
        if seq:
            self.seq_len = self.data.pop('seq_len')
            self.max_seq_len = self.seq_len.max()
        if topndiags is not None:
            self.data = self.data.sort_index(axis=1).iloc[:, :topndiags]
        self._data_shape = [topndiags or 1, _num_codes.index.max()] # (topndiags or 1, num_codes)
        self.labels = self.labels.rename(columns=lambda x: x if x in {'hcc', 'label_date'} else 'Y')
        if 'label_date' not in self.labels.columns:
            assert pred_window is not None
            pred_window = pd.DateOffset(days=pred_window)
            self.labels['label_date'] = self.labels.index.map(lambda x: (x[1]+pred_window).date())
        self.labels['Y'] = self.labels['Y'].dropna().map(list)

        self.full_indices = self.labels.dropna(how='any').index.intersection(self.data.index)

        self.rev_seq = rev_seq #If rev_seq, the vists are reversed.
        """
        full_indices = self.labels.dropna(how='any').index.intersection(self.data.index).sortlevel(1)[0]
        _idx_splits = get_split_indices(None, [train_ratio, 1-train_ratio-test_ratio, test_ratio], n=len(full_indices))
        self.indices = full_indices[_idx_splits[split] if split != VALIDTEST else np.concatenate([_idx_splits[_] for _ in [VALID, TEST]])]
        """

    def get_data(self, pat_id='03d2AAAAABEGKIQE', dt=pd.to_datetime('20160803').date()):
        y = self.labels.loc[(pat_id, dt), 'Y']
        ydate = self.labels.loc[(pat_id, dt), 'label_date']
        if self.seq_len is None: # not seq, but aggregated
            x = -np.ones(shape=self._data_shape, dtype=int)
            for diag_j, v in enumerate(self.data.loc[(pat_id, dt)].values):
                x[diag_j, :len(v)] = v
            x = np.expand_dims(x, 0)
        else:
            seq_len = self.seq_len[(pat_id, dt)]
            pat_df = self.data.loc[pat_id].loc[:dt]
            x = -np.ones(shape=[seq_len] + self._data_shape, dtype=int)
            for i in range(seq_len):
                if not self.rev_seq: raise NotImplementedError()
                curr_visit = pat_df.iloc[-i-1].values
                for diag_j in range(x.shape[1]):
                    x[i, diag_j, :len(curr_visit[diag_j])] = curr_visit[diag_j]
        # x: [ndiags, codes] or [seq_len, ndiags, codes]
        # y: set of labels (un-encoded HCC codes)
        # ydate: date when y is observed
        #NOTE: The +1 here is because I used -1 as padding, but obviousy torch does not understand this so I have to change it to 0
        return x + 1, y, ydate

class ClaimDemoData(DatasetWrapper):
    DATASET = _settings.CLAIMDEMO_NAME
    def __init__(self, split, train_ratio=0.75, test_ratio = 0.15, seed=_settings.RANDOM_SEED,
                 freq='weekly', pred_window=182, data_window=182, topndiags=None, seq=True,
                 hcc_choice: str='few'):
        self.LABEL_MAP = code_utils.HCCV24.decode_dict(hcc_choice)
        self.LABEL_MAP[0] = 'Other'
        self.labelencoder = code_utils.HCCV24.encode_dict(hcc_choice)
        self.CLASSES = [self.LABEL_MAP[i] for i in sorted(self.LABEL_MAP.keys())]
        super().__init__(split)

        self.pat_reader = PatientDataReader()
        self.claim_reader = ClaimDataReader(freq=freq, pred_window=pred_window, data_window=data_window, topndiags=topndiags, seq=seq)

        #indices
        full_indices = self.claim_reader.full_indices
        if seed is None: #sequential
            full_indices = full_indices.sortlevel(1)[0]
            _idx_splits = get_split_indices(seed, [train_ratio, 1-train_ratio-test_ratio, test_ratio], n=len(full_indices))
        else:
            _idx_splits = get_split_indices_by_group(seed, [train_ratio, 1-train_ratio-test_ratio, test_ratio], groups = [_[0] for _ in full_indices])
        self.indices = full_indices[_idx_splits[split] if split != VALIDTEST else np.concatenate([_idx_splits[_] for _ in [VALID, TEST]])]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pat_id, dt = self.indices[idx]
        y = np.zeros(len(self.labelencoder) + 1, dtype=np.float32)
        x, hccs, ydate = self.claim_reader.get_data(pat_id, dt)
        for _y_hcc in hccs:
            _y_hcc = self.labelencoder.get(_y_hcc, 0)
            y[_y_hcc] = 1.

        pat_info = self.pat_reader.get_data(pat_id, dt) # [age, r['der_sex'], r['pat_region']]
        ret = {'data': {k: v for k, v in zip(['pat_age', 'pat_sex', 'pat_region'], pat_info)}}
        ret['data']['diags'] = x
        ret['target'] = y
        ret['index'] = pat_id + '_' + dt.strftime("%Y%m%d")
        ret['target_date'] = ydate.strftime("%Y%m%d")
        return ret

    @classmethod
    def _collate_func(cls, batch):
        import torch
        max_seq_len = max([len(_['data']['diags']) for _ in batch])
        shape = list(batch[0]['data']['diags'].shape)
        shape[0] = max_seq_len
        padding = np.zeros(shape=shape, dtype=int)
        for i in range(len(batch)):
            old_diags = batch[i]['data']['diags']
            batch[i]['data']['diags'] = np.concatenate([old_diags, padding[len(old_diags):]], 0)
        return torch.utils.data.default_collate(batch)

if __name__ == '__main__':
    if False:
        for freq in ['daily', 'weekly']:
            df = get_aggregated_df(freq)
            df = get_processed_claim_df(freq)
            offsets = {'daily': [30, 90, 180, 365], 'weekly': [28, 13*7, 26*7, 52*7]}[freq]

            for window in [None] + offsets + [MAX_OFFSET_DAYS]:
                if window != MAX_OFFSET_DAYS:
                    df = get_processed_hcc_label(freq=freq, post_window=window)
                for pooled in [False, True]:
                    res = cache_data_by_patient(seq=False, pooled=pooled, data_window=window, freq=freq)
                    if window is not None:
                        res = cache_data_by_patient(seq=True, pooled=pooled, data_window=window, freq=freq)
    if False:
        #o = ClaimDataReaderSeq(pred_window=None, data_window=None, topndiags=None)
        ot = ClaimDemoData('train', pred_window=30, data_window=30, topndiags=5, seq=True, freq='daily')
        #ot2 = ClaimDemoData('test', pred_window=30, data_window=30, topndiags=5, seq=False, freq='daily')
        #ov = ClaimDemoData('val', pred_window=30, data_window=30, topndiags=5, seq=False, freq='daily')
        for i in tqdm.tqdm(range(len(ot))):
            ot[i]
    if False:
        # claim few cost: ~0.7, 90%->0.45 (unique subsets, not accounting for frequency)
        # claim more cost: ~0.33, 90%->0.14 (unique subsets, not accounting for frequency)
        rf = code_utils.HCCV24.get_risk_factors(mode='more')
        labels = get_processed_hcc_label(freq='daily', post_window=365)['label_365_hcc'].dropna()
        costs = {}
        for _pre_S in tqdm.tqdm(labels.map(lambda x: tuple(sorted(list(x)))).unique()):
            _curr_rf = rf.reindex(_pre_S)
            _curr_cost = rf['Other'] if _curr_rf.isnull().any() else 0.
            _curr_cost += _curr_rf.sum()
            costs["|".join(_pre_S)] = _curr_cost / rf.sum()
        costs = pd.Series(costs)
