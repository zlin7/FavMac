import glob
import os
from importlib import reload
from typing import Union

import ipdb
import pandas as pd
import scipy.stats

import utils.eval_utils as eutils

reload(eutils)

INTEGER_SAFE_DELTA = 0.1

#Full
_METHOD_MAPPING = [
    ("GV", "\\baselineGreedyVal"),
    ("GV+DS", "\\baselineGreedyVal(NN)"),
    ("GV+MC", "\\baselineGreedyVal+MC"),
    ("GV+exact", "\\baselineGreedyVal+exact"),
    ("GV+gauss", "\\baselineGreedyVal+gauss"),

    ("GR", "\\methodname"),
    ("GR+DS", "\\methodname(NN)"),
    ("GR+MC", "\\methodname+MC"),
    ("GR+exact", "\\methodname+exact"),
    ("GR+gauss", "\\methodname+gauss"),

    ("GP", "\\baselineGreedyProb"),
    ("GP+DS", "\\baselineGreedyProb(NN)"),
    ("GP+DS", "\\baselineFPCP"),

    ("Full", "\\baselineFull"),

    ('ClassWise', '\\baselineClasswise'),
    ('InnerSet', '\\baselineInnerSet'),
    ('RCPS', 'RCPS'),
]

_FP_DATASET_MAPPING = [
    ('mimic3_few', '\\dataMIMICThree(select)'),
    ('mimic3_more', '\\dataMIMICThree'),

    ('claim_few_random', '\\dataClaim(select)'),
    ('claim_more_random', '\\dataClaim'),

    ('claim_few', '\\dataClaimSeq(select)'),
    ('claim_more', '\\dataClaimSeq'),

    ('mnist_add', '\\dataMNIST'),
    ('mnist_add_mult_util', '\\dataMNIST(GEN)'),
]

_CTS_DATASET_MAPPING = [
    ('mimic3_few_cost_TP', '\\dataMIMICThree(select,TP)'),
    ('mimic3_more_cost_TP', '\\dataMIMICThree(TP)'),
    ('claim_few_cost_TP_random', '\\dataClaim(select,TP)'),
    ('claim_more_cost_TP_random', '\\dataClaim(TP)'),
    ('claim_few_cost_TP', '\\dataClaimSeq(select,TP)'),
    ('claim_more_cost_TP', '\\dataClaimSeq(TP)'),
    ('mnist_add_cost_TP', '\\dataMNIST(TP)'),

    ('mimic3_few_cost', '\\dataMIMICThree(select,TPC)'),
    ('mimic3_more_cost', '\\dataMIMICThree(TPC)'),
    ('claim_few_cost_random', '\\dataClaim(select,TPC)'),
    ('claim_more_cost_random', '\\dataClaim(TPC)'),
    ('claim_few_cost', '\\dataClaimSeq(select,TPC)'),
    ('claim_more_cost', '\\dataClaimSeq(TPC)'),
    ('mnist_add_cost', '\\dataMNIST(TPC)'),

    ('mnist_add_cost_mult_util','\\dataMNIST(GEN)')
]

_NONCONFORMALMETHOD_MAPPING = [

]

def g_METHODs():
    return [_[0] for _ in _METHOD_MAPPING + _NONCONFORMALMETHOD_MAPPING]
def g_METHOD_MAPs():
    ret = dict(_METHOD_MAPPING + _NONCONFORMALMETHOD_MAPPING)
    return {k: v.replace("_", "\_") for k, v in ret.items()}
def g_FP_DATAs():
    return [_[0] for _ in _FP_DATASET_MAPPING]
def g_CTS_DATAs():
    return [_[0] for _ in _CTS_DATASET_MAPPING]
def g_DATA_MAPs():
    ret = dict(_FP_DATASET_MAPPING + _CTS_DATASET_MAPPING)
    return {k: v.replace("_", "\_") for k, v in ret.items()}
def g_NONCONFMETHODs():
    return [_[0] for _ in _NONCONFORMALMETHOD_MAPPING]

def g_METHOD_ORDER():
    _map = g_METHOD_MAPs()
    return {_map[_]: i for i, _ in enumerate(g_METHODs())}

def g_DATA_ORDER():
    return {_: i for i, _ in enumerate(g_FP_DATAs()+g_CTS_DATAs())}

def retrieve(dfs, dataset, delta, metric, describe, method=None, seed=None, target_cost=None):
    df = dfs[dataset]
    df = df[df['delta'].isnull() if delta is None else df['delta'] == delta]#.drop('delta', axis=1)
    #if delta is None and metric == 'util' and dataset =='mimic3_few_cost':
    #    ipdb.set_trace()
    if metric is not None:
        df = df[df['metric'] == metric]#.drop('metric', axis=1)
    if describe is not None:
        df = df[df['describe'] == describe]#.drop('describe', axis=1)
    if method is not None:
        if isinstance(method, list):
            df = df[df['method'].isin(method)]#.drop('method', axis=1)
        else:
            df = df[df['method'] == method]#.drop('method', axis=1)
    if seed is not None:
        df = df[df['seed'] == seed]#.drop('seed', axis=1)
    if target_cost is not None:
        df = df[df['target_cost'] == target_cost]#.drop('target_cost', axis=1)

    return df

def _default_formatter(mean, std):
    if pd.isnull(std): return f"{mean:.2f}"
    return f"{mean:.2f}$\pm${std:.2f}"
def _default_formatter3(mean, std):
    if pd.isnull(std): return f"{mean:.3f}"
    return f"{mean:.3f}$\pm${std:.3f}"
def create_printable_df(summs, create_ser_fn, formatter=_default_formatter, scale:Union[float,int,dict]=1, pval=0.01):
    METHOD_ORDER = g_METHOD_ORDER()
    ret = {}
    mask = {}
    for dataset, summ in summs.items():
        _scale = scale[dataset] if isinstance(scale, dict) else scale
        ret[dataset], mask[dataset] = create_ser_fn(summ, formatter, _scale, pval=pval)
    ret, mask = pd.DataFrame(ret), pd.DataFrame(mask)
    sidx = ret.index[ret.index.map(METHOD_ORDER).argsort()]
    return ret.reindex(sidx), mask.reindex(sidx)

#=====================================================Latex Handling

class LatexPrinter:
    _MIDRULE  = '\\midrule'
    _BOTTOM = '\\bottomrule'
    def __init__(self, mask_format=None, fill_nan = '\\textendash', pad=True) -> None:
        if mask_format is None:
            def mask_format(s, flag):
                if flag == 0: return s
                assert flag == 1
                return "\\textbf{%s}"%s
        self.mask_format = mask_format
        self.fill_nan = fill_nan
        self.pad = pad


    def _get_formatted_cells(self, df, mask_df):
        _fmt = lambda s, flag: self.fill_nan if pd.isnull(s) else self.mask_format(s, flag)
        assert (df.dtypes == 'O').all(), "Expect the dataframe to be full of strings only."
        assert df.shape == mask_df.shape
        strs = []
        for idx in df.index:
            strs.append([str(idx)] + [_fmt(df.loc[idx, c], mask_df.loc[idx, c]) for c in df.columns])
        return strs

    def _compute_column_widths(self, cells):
        formattable_cells = [line for line in cells if isinstance(line, list)]
        if not self.pad:
            return [1] * len(formattable_cells[0])
        return [max([len(_[j]) for _ in formattable_cells]) for j in range(len(formattable_cells[0]))]

    def _prints_df_helper(self, df, mask_df, table_name = '', skip_header=False, colwidths=None):
        # Can repeatedly call this function with different formatter and masks
        lines = [[table_name] + list(map(str, df.columns))]
        lines.extend(self._get_formatted_cells(df, mask_df))
        new_lines = []
        if colwidths is None: colwidths = self._compute_column_widths(lines)
        for i, line in enumerate(lines):
            if i == 0 and skip_header: continue
            new_lines.append(" & ".join([_.rjust(colwidths[j]) for j, _ in enumerate(line)]) + "\\\\")
            if i == 0: new_lines.append(self._MIDRULE)
        return new_lines

    def print_df(self, df, mask_df, table_name = '', skip_header=False):
        print("\n".join(self._prints_df_helper(df, mask_df, table_name, skip_header)))

    def _prints_dfs(self, dfs, mask_dfs, column_names=None, row_names = None, multirow=False):
        assert len(dfs) == len(mask_dfs)
        assert all([len(dfs1) == len(mask_dfs[i]) for i, dfs1 in enumerate(dfs)])
        # M x N
        M, N = len(dfs), len(dfs[0])
        if column_names is None: column_names = [''] * (N)
        if row_names is None: row_names = [''] * (M )

        middle_cells = [[self._get_formatted_cells(dfs[i][j], mask_dfs[i][j]) for j in range(N)] for i in range(M)]

        strs = []
        if N > 1:
            strs.append(" & " + " & ".join(["\\multicolumn{%d}{|c}{%s}"%(dfs[0][j].shape[1], column_names[j]) for j in range(N)]) + "\\\\")
            #strs.append(self._MIDRULE)
        strs.append([''] * (2 if M > 1 and multirow else 1) + [str(_) for j in range(N) for _ in dfs[0][j].columns])
        strs.append(self._MIDRULE)
        for i in range(M):
            middle_cells_i = middle_cells[i]
            if M > 1 and not multirow:
                strs.extend([f"{row_names[i]}\\\\", self._MIDRULE])
            for ii in range(len(middle_cells_i[0])):
                curr_line = []
                if M > 1 and multirow:
                    curr_line.append("\multirow{%d}{*}{%s}"%(len(dfs[i][0]), row_names[i]) if ii == 0 else '')
                for j in range(N):
                    curr_line.extend(middle_cells_i[j][ii][min(1,j):])
                strs.append(curr_line)
            if i < M - 1:
                strs.append(self._MIDRULE)
        colwidths = self._compute_column_widths(strs)
        new_lines = []
        for i, line in enumerate(strs):
            if isinstance(line, list):
                new_lines.append(" & ".join([_.rjust(colwidths[j]) for j, _ in enumerate(line)]) + "\\\\")
            else:
                new_lines.append(line)
        new_lines.append(self._BOTTOM)
        return new_lines


    def print_dfs(self, dfs, mask_dfs, column_names=None, row_names = None, multirow=False):
        print("\n".join(self._prints_dfs(dfs, mask_dfs, column_names, row_names, multirow=multirow)))

    @classmethod
    def test(cls):
        test_df = pd.DataFrame([['1', '2'],['3', '4']], index=['asdasdasa', 'ad'])
        test_mask = pd.DataFrame(0, index=test_df.index, columns=test_df.columns)
        o = cls(pad=True)
        o.print_dfs([[test_df, test_df]], [[test_mask, test_mask]], column_names=['Datafram1', 'dataframe2'])
        o.print_dfs([[test_df], [test_df]], [[test_mask], [test_mask]])
        o.print_dfs([[test_df], [test_df]], [[test_mask], [test_mask]], multirow=True)

def printable_df_to_latex(df, mask_df, mask_format=None, table_name='', fill_nan = '\\textendash',
                          skip_header=False, pad=True):
    LatexPrinter(mask_format, fill_nan=fill_nan, pad=pad).print_df(df, mask_df, table_name=table_name, skip_header=skip_header)
    return

#==================================================================================
def read_results(_dir): #_dir=notebook/output
    files = glob.glob(os.path.join(_dir, "*(summ).csv"))
    #print(files)
    dfs = {
        os.path.basename(f).replace("(summ).csv", ""): pd.read_csv(f) for f in files
    }
    return dfs

def create_printable_ser_util(summ, formatter=_default_formatter, scale=1, pval=0.01):
    NONCONFMETHODs = g_NONCONFMETHODs()
    ret = {}
    mask = {}
    best_method = summ['rank'][~summ.index.isin(NONCONFMETHODs)].idxmax()
    num_with_reps = summ['std'].notnull().sum()
    assert num_with_reps == 0 or num_with_reps == summ['mean'].count()
    for method in summ.index:
        assert isinstance(method, str)
        ret[method] = formatter(summ.loc[method, 'mean'] * scale, summ.loc[method, 'std'] * scale)
        if num_with_reps == 0:
            mask[method] = 1 if summ.loc[method, 'mean'] == summ.loc[best_method, 'mean'] else 0
        else:
            mask[method] = 1 if summ.loc[method, 'pval'] > pval and method not in NONCONFMETHODs else 0
    return pd.Series(ret).astype(str).reindex(summ.index), pd.Series(mask).astype(int).reindex(summ.index)


def create_printable_ser_cost(summ, formatter=_default_formatter, scale=1, pval=0.01):
    NONCONFMETHODs = g_NONCONFMETHODs()
    ret = {}
    mask = {}
    best_method = summ['rank'][~summ.index.isin(NONCONFMETHODs)].idxmax()
    num_with_reps = summ['std'].notnull().sum()
    assert num_with_reps == 0 or num_with_reps == summ['mean'].count()
    for method in summ.index:
        assert isinstance(method, str)
        ret[method] = formatter(summ.loc[method, 'mean'] * scale, summ.loc[method, 'std'] * scale)
        #We mark bad entries this time
        mask[method] = 1 if summ.loc[method, 'pval'] < pval and method not in NONCONFMETHODs else 0
    return pd.Series(ret).astype(str).reindex(summ.index), pd.Series(mask).astype(int).reindex(summ.index)

def get_util_summs(dfs, max_target=0.54, min_target=0.011):
    from collections import defaultdict
    METHODs, METHOD_MAPs, FP_DATAs, CTS_DATAs, DATA_MAPs, NONCONFMETHODs, METHOD_ORDER, DATA_ORDER = g_METHODs(), g_METHOD_MAPs(), g_FP_DATAs(), g_CTS_DATAs(), g_DATA_MAPs(), g_NONCONFMETHODs(), g_METHOD_ORDER(), g_DATA_ORDER()
    util_summs = {"FP": defaultdict(dict), "CTS": defaultdict(dict)}
    all_target_costs = []
    for delta in [None, 0.1]:
        for _mode, datasets in zip(['FP', 'CTS'], [FP_DATAs, CTS_DATAs]):
            for dataset in datasets:
                if dataset not in dfs: continue
                tdf = retrieve(dfs, dataset, delta, metric='util', describe='mean', method=METHODs)
                relevant_methods = [_ for _ in METHODs if _ in tdf['method'].unique()]
                tdf = tdf.reindex(columns=['seed', 'target_cost', 'method', 'value'])
                if max_target is not None: tdf = tdf[tdf['target_cost'] < max_target]
                if min_target is not None:tdf = tdf[tdf['target_cost'] > min_target]
                assert tdf.groupby('method').size().max() == tdf.groupby('method').size().min(), f"{tdf.groupby('method').size()}"
                all_target_costs.append(tdf['target_cost'])
                tdf = tdf.groupby(['method','seed'])['value'].mean().unstack().T
                summ = eutils.summarize_mean_std_pval(tdf, paired=True, higher_better=True)
                summ = summ.reindex(relevant_methods).rename(METHOD_MAPs)
                util_summs[_mode][delta][DATA_MAPs[dataset]] = summ
    print(f"Target Summary: {pd.concat(all_target_costs, ignore_index=True).describe()}")
    return util_summs

def get_util_plot_summs(dfs, max_target=0.54, min_target=0.011):
    from collections import defaultdict
    METHODs, METHOD_MAPs, FP_DATAs, CTS_DATAs, DATA_MAPs, NONCONFMETHODs, METHOD_ORDER, DATA_ORDER = g_METHODs(), g_METHOD_MAPs(), g_FP_DATAs(), g_CTS_DATAs(), g_DATA_MAPs(), g_NONCONFMETHODs(), g_METHOD_ORDER(), g_DATA_ORDER()
    util_summs = {"FP": defaultdict(dict), "CTS": defaultdict(dict)}
    all_target_costs = []
    for delta in [None, 0.1]:
        for _mode, datasets in zip(['FP', 'CTS'], [FP_DATAs, CTS_DATAs]):
            for dataset in datasets:
                if dataset not in dfs: continue
                tdf = retrieve(dfs, dataset, delta, metric='util', describe='mean', method=METHODs)
                relevant_methods = [_ for _ in METHODs if _ in tdf['method'].unique()]
                tdf = tdf.reindex(columns=['seed', 'target_cost', 'method', 'value'])
                if max_target is not None: tdf = tdf[tdf['target_cost'] < max_target]
                if min_target is not None:tdf = tdf[tdf['target_cost'] > min_target]
                assert tdf.groupby('method').size().max() == tdf.groupby('method').size().min(), f"{tdf.groupby('method').size()}"
                all_target_costs.append(tdf['target_cost'])
                tdf = pd.pivot_table(tdf, values='value', columns='method', index='target_cost')
                util_summs[_mode][delta][DATA_MAPs[dataset]] = tdf.reindex(columns=relevant_methods).rename(columns=METHOD_MAPs)
    print(f"Target Summary: {pd.concat(all_target_costs, ignore_index=True).describe()}")
    return util_summs

def get_cost_summs(dfs, max_target=0.54, min_target=0.011, twosided=False, retrieve_violation=False):
    from collections import defaultdict
    METHODs, METHOD_MAPs, FP_DATAs, CTS_DATAs, DATA_MAPs, NONCONFMETHODs, METHOD_ORDER, DATA_ORDER = g_METHODs(), g_METHOD_MAPs(), g_FP_DATAs(), g_CTS_DATAs(), g_DATA_MAPs(), g_NONCONFMETHODs(), g_METHOD_ORDER(), g_DATA_ORDER()
    cost_summs = {"FP": defaultdict(dict), "CTS": defaultdict(dict)}
    all_target_costs = []
    for delta in [None, 0.1]:
        for _mode, datasets in zip(['FP', 'CTS'], [FP_DATAs, CTS_DATAs]):
            for dataset in datasets:
                if dataset not in dfs: continue
                if delta is None and not retrieve_violation:
                    tdf = retrieve(dfs, dataset, delta, metric='cost', describe='mean', method=METHODs)
                    tdf['value'] = tdf['value'] - tdf['target_cost']
                else:
                    tdf = retrieve(dfs, dataset, delta, metric='violation', describe='mean', method=METHODs)
                relevant_methods = [_ for _ in METHODs if _ in tdf['method'].unique()]
                tdf = tdf.reindex(columns=['seed', 'target_cost', 'method', 'value'])
                if max_target is not None: tdf = tdf[tdf['target_cost'] < max_target]
                if min_target is not None:tdf = tdf[tdf['target_cost'] > min_target]
                assert tdf.groupby('method').size().max() == tdf.groupby('method').size().min(), f"{tdf.groupby('method').size()}"
                all_target_costs.append(tdf['target_cost'])
                tdf = tdf.groupby(['method','seed'])['value'].mean().unstack().T
                summ = eutils.summarize_mean_std_pval(tdf, paired=False, higher_better=False, target=0 if delta is None else delta, twosided=twosided)
                summ = summ.reindex(relevant_methods).rename(METHOD_MAPs)
                cost_summs[_mode][delta][DATA_MAPs[dataset]] = summ
    print(f"Target Summary: {pd.concat(all_target_costs, ignore_index=True).describe()}")
    return cost_summs