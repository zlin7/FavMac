import itertools
from collections import defaultdict
from importlib import reload

import ipdb
import numpy as np
import pandas as pd
import tqdm
from scipy import stats

import _settings
import conformal
import data_utils
import pipeline.main as pm
import utils.eval_utils as eutils
import utils.utils as utils
from data_utils import INTEGER_SAFE_DELTA


class Evaluator:
    def __init__(self, calib_cls, base_keymode, cost_fn, value_fn, proxy_fn, dataset, datakwargs,
                 *,
                 split=data_utils.VALIDTEST, burn_in=1000, calib_cls_kwargs=None, **kwargs) -> None:
        self.call_kwargs = {
            'calib_cls': calib_cls,
            'base_keymode': base_keymode,
            'cost_fn': cost_fn,
            'value_fn': value_fn,
            'proxy_fn': proxy_fn,
            'dataset': dataset,
            'datakwargs': datakwargs,
            'split': split,
            'burn_in': burn_in,
            'calib_cls_kwargs': calib_cls_kwargs
        }
        self.call_kwargs.update(kwargs)
        if dataset == _settings.MNISTSup_NAME:
            self.call_kwargs['split'] = data_utils.TEST
        if dataset != _settings.CLAIMDEMOSeq_NAME:
            self.call_kwargs['dataset'] = f"{dataset}|cap=3000"
            # use only 3000 samples as "test" set for each random permutation
    def __call__(self, target_cost, delta, *, perm_seed=None, **kwargs):
        assert type(target_cost) == float and (delta is None or type(delta) == float)
        call_kwargs = self.call_kwargs.copy()
        call_kwargs.update({"target_cost": target_cost, "delta": delta})
        call_kwargs.update({"perm_seed": perm_seed})
        call_kwargs.update(kwargs)
        # print(call_kwargs)
        res = pm._evaluate_main_cache(**call_kwargs)
        res = pd.DataFrame(res).iloc[1:]
        return res, None


    def eval_targetcosts(self, target_costs, delta):
        ret = {_: self(_, delta)[0] for _ in tqdm.tqdm(target_costs)}
        return ret

    @classmethod
    def _eval_target_to_realized(cls, results, col='cost'):
        ret = {}
        for target_cost, _res in results.items():
            ret[target_cost] = _res[col]
        ret = pd.DataFrame(ret).T
        return ret

    @classmethod
    def _eval_target_to_realized_mean_by_seed(cls, results, col='cost'):
        ret = {}
        for target_cost, _res in results.items():
            if '_' in _res['index'].iloc[0]:
                gb = _res.groupby(_res['index'].map(lambda x: x.split("_")[-1]))
            else:
                gb = _res
            ret[target_cost] = gb[col].mean()
        ret = pd.DataFrame(ret).T.mean(0)
        return ret

    @classmethod
    def eval_target2meanutil_by_seed(cls, results, col='util'):
        return cls._eval_target_to_realized_mean_by_seed(results, col)

    @classmethod
    def eval_target2meanutil(cls, results, col='util'):
        return cls._eval_target_to_realized(results, col).mean(1)

    @classmethod
    def eval_target2cost(cls, results, col='cost'):
        return cls._eval_target_to_realized(results, col).T

    @classmethod
    def _eval_grouped_cost(cls, df, window=100):
        df = df.iloc[:(len(df)//window) * window]
        assert len(df)%window == 0
        mcosts = df.groupby(np.arange(len(df)) // window)['cost'].mean()
        return mcosts

    @classmethod
    def eval_grouped_cost(cls, results, window=100):
        ret = {}
        summ = {}
        all_excess = []
        def _summ_excess(excess):
            return pd.Series({
                "mean": excess.mean(),
                'std': excess.std(),
                '95p': excess.quantile(0.95)})
        for target_cost, df in results.items():
            tser = cls._eval_grouped_cost(df, window=window)
            excess = (tser - target_cost).clip(0)
            ret[target_cost] = tser
            summ[target_cost] = _summ_excess(excess)
            all_excess.append(excess)
        return pd.DataFrame(ret), pd.DataFrame(summ).T, pd.Series(_summ_excess(pd.concat(all_excess, ignore_index=True)))

    @classmethod
    def test_violation(cls, results, delta):
        from scipy import stats
        ret = {}
        all_sers = []
        def _summ_violation(violation):
            return pd.Series({'pval': stats.binom_test(violation.sum(), len(violation), p=delta, alternative='greater'),
            'mean': violation.mean(), 'std': violation.std(),
            })
        def _summ_cost(cost):
            return pd.Series({'pval': stats.ttest_1samp(cost.values, 0, alternative='greater').pvalue,
            'mean': cost.mean(), 'std': cost.std(),
            })
        _summ = _summ_cost if delta is None else _summ_violation

        tc2cost = cls.eval_target2cost(results)
        for target_cost, realized_costs in tc2cost.items():
            curr_ser = realized_costs - target_cost
            if delta is not None: curr_ser = (curr_ser > 0).astype(int)
            ret[target_cost] = _summ(curr_ser)
            all_sers.append(curr_ser)
        return pd.DataFrame(ret), _summ(pd.concat(all_sers, ignore_index=True))

    @classmethod
    def compare_value_mean_std(cls, results, paired_pval=True, col='util', higher_better=True):
        # results[method][cost] is a dataframe with columns[thres, cost, util, index]
        values = {}
        num_tcs = None
        for method, res in results.items():
            if num_tcs is None: num_tcs = len(res)
            assert len(res) == num_tcs, "Different number of target_costs is not OK"
            tdf = {}
            for target_cost, df in res.items():
                if '_' in df['index'].iloc[0]:
                    gb = df.groupby(df['index'].map(lambda x: x.split("_")[-1]))
                else:
                    gb = df.groupby(np.ones(len(df)))
                tdf[target_cost] = gb[col].mean()
            values[method] = pd.DataFrame(tdf).mean(1)
        values = pd.DataFrame(values)
        values.index.name = 'seed'
        # summary
        summ = eutils.summarize_mean_std_pval(values, paired=paired_pval, higher_better=higher_better)
        return summ, values

    @classmethod
    def flatten_summary(cls, results):
        ret = []
        for delta, method_target_seed_results in results.items():
            for method, target_seed_results in method_target_seed_results.items():
                for target_cost, df in target_seed_results.items():
                    df['violation'] = (df['cost'] > target_cost).astype(int)
                    if '_' in df['index'].iloc[0]:
                        gb = df.groupby(df['index'].map(lambda x: x.split("_")[-1]))
                    else:
                        gb = df.groupby(np.ones(len(df)))
                    tdf = gb.describe().stack().stack().reset_index()
                    tdf['target_cost'] = target_cost
                    tdf['method'] = method
                    tdf['delta'] = delta
                    ret.append(tdf)
        return pd.concat(ret, ignore_index=True).rename(columns={"index": 'seed', 'level_1': 'describe', 'level_2': 'metric', 0: 'value'})


class Config:
    KEYS = {
        _settings.CLAIMDEMO_NAME:{
            'train_few': '',
            'train1_few': '',
            'dsproxy_few': '',
            'train_more':  '',
            'train1_more': '',
            'dsproxy_more': '',
        },
        _settings.MIMICIIICompletion_NAME: {
            'train_few': 'EHRModel-MIMIC-IIICompletion-20230521_160512091696',
            'train1_few': 'EHRModel-MIMIC-IIICompletion-20230521_181249342352',
            'dsproxy_few': 'DeepSets-MIMIC-IIICompletion-20230522_004219423813',
            'train_more':  'EHRModel-MIMIC-IIICompletion-20230521_172120015213',
            'train1_more': 'EHRModel-MIMIC-IIICompletion-20230521_161730874913',
            'dsproxy_more': 'DeepSets-MIMIC-IIICompletion-20230522_004548627063',
        },
        _settings.MNISTSup_NAME: {
            'train': "MNISTCNN-MNISTSup-20230521_012403087576",
            'train1': "MNISTCNN-MNISTSup-20230521_012820281502",
            'dsproxy': 'DeepSets-MNISTSup-20230521_210137626025',
        }
    }
    NCLASS = {
        (_settings.MIMICIIICompletion_NAME, '_few'): 10,
        (_settings.MIMICIIICompletion_NAME, '_more') : 35,
        (_settings.CLAIMDEMO_NAME, '_few'): 10+1,
        (_settings.CLAIMDEMOSeq_NAME, '_few'): 10+1,
        (_settings.CLAIMDEMO_NAME, '_more'): 31+1,
        (_settings.CLAIMDEMOSeq_NAME, '_more'): 31+1,
        (_settings.MNISTSup_NAME, ''): 10,
    }
    @classmethod
    def tidy_run(cls, task_runner):
        from collections import defaultdict
        ret_mid = defaultdict(list)
        for (method, tc, delta, perm_seed), (df, predsets) in task_runner.run().items():
            suffix = '_None' if perm_seed is None else f'_{perm_seed}'
            df['index'] = df.index.map(lambda x: f"{x}{suffix}")
            ret_mid[(method, tc, delta)].append(df)
        ret = defaultdict(lambda : defaultdict(dict))
        for (method, tc, delta,), dfs in ret_mid.items():
            ret[delta][method][tc] = pd.concat(dfs, ignore_index=True)
        return ret

    @classmethod
    def _add_tasks(cls, evaluators, task_runner:utils.TaskPartitioner = None, add_key=False, perm_seeds=[None], nclass=None):
        if task_runner is None: task_runner = utils.TaskPartitioner()
        costs = [0.05 * (i) for i in range(1, 20)]
        if nclass is None:
            costs = [float(np.round(c, 3)) for c in costs]
        else:
            costs = [(np.round(c*nclass, 0)+INTEGER_SAFE_DELTA)/nclass for c in costs] #ignore the first cost
            costs = sorted(list(set([float(np.round(c, 5)) for c in costs])))
        deltas = [None, 0.1]
        for perm_seed in perm_seeds:
            for target_cost, delta in itertools.product(costs, deltas):
                if target_cost >= 1 : continue
                for method, obj in evaluators.items():
                    if method == 'ClassWise' and delta is not None: continue
                    if nclass is not None and target_cost < INTEGER_SAFE_DELTA/nclass * 2 and method in {"ClassWise"}: continue
                    call_kwargs = {"target_cost": target_cost, 'delta': delta, 'perm_seed': perm_seed}
                    if add_key:
                        task_runner.add_task_with_key( (method, target_cost, delta, perm_seed), obj, **call_kwargs)
                    else:
                        task_runner.add_task(obj, **call_kwargs)
        return task_runner


    @classmethod
    def _FPcost_CTSutil(cls, task_runner, dataset, datakwargs, suffix, cost_fn, util_fn, proxy_fn=None,
            mode='last', skip=[], nseeds=3):
        assert cost_fn == 'sf_FP_cost' and util_fn.endswith("_util") and proxy_fn is None
        nclass = cls.NCLASS[(dataset, suffix)]
        train_key, train1_key, ds_key = [cls.KEYS[dataset][f"{_}{suffix}"] for _ in ['train', 'train1', 'dsproxy']]
        evaluators = {}

        kwargs = {"dataset": dataset, 'datakwargs': datakwargs, 'cost_fn': cost_fn, 'value_fn': util_fn}
        kwargs.update({"base_keymode": f"{train_key}|{mode}", 'proxy_fn': data_utils.SF_FP_proxy})
        evaluators['GV'] = Evaluator(conformal.FavMac_GreedyValue, **kwargs.copy())
        evaluators['Full'] = Evaluator(conformal.FullUniverse, **kwargs.copy())
        evaluators['GP'] = Evaluator(conformal.FavMac_GreedyProb, **kwargs.copy())
        evaluators['GR'] = Evaluator(conformal.FavMac_GreedyRatio, **kwargs.copy())


        kwargs.pop('proxy_fn')
        kwargs.update({"base_keymode": f"{train1_key}|{mode}"})
        evaluators['GP+DS'] = Evaluator(conformal.FPCP_fast, proxy_fn=f'{ds_key}', **kwargs.copy())

        # Other naiver baselines
        kwargs['proxy_fn'] = None
        evaluators["ClassWise"] = Evaluator(conformal.IndividualCPSet, **kwargs.copy())

        evaluators = {k: _ for k, _ in evaluators.items() if k.split("@")[0] not in skip}
        if nclass > 11:  evaluators = {k: _ for k, _ in evaluators.items() if not k.endswith("+exact") and k != 'Full'}
        perm_seeds = list(np.arange(nseeds)) if dataset != _settings.CLAIMDEMOSeq_NAME else [None]
        return cls._add_tasks(evaluators, task_runner, add_key=task_runner is None, nclass=nclass, perm_seeds=perm_seeds)


    @classmethod
    def _CTScost(cls,  task_runner, dataset, datakwargs, suffix, cost_fn, util_fn, proxy_fn, mode='last', skip=[], nseeds=3):
        nclass = cls.NCLASS[(dataset, suffix)]
        train_key = cls.KEYS[dataset][f"train{suffix}"]
        evaluators = {}
        kwargs = {"dataset": dataset, 'datakwargs': datakwargs, 'cost_fn': cost_fn, 'value_fn': util_fn}
        kwargs.update({"base_keymode": f"{train_key}|{mode}", 'proxy_fn': proxy_fn})
        evaluators['GV'] = Evaluator(conformal.FavMac_GreedyValue, **kwargs.copy())
        evaluators['Full'] = Evaluator(conformal.FullUniverse, **kwargs.copy())
        evaluators['GP'] = Evaluator(conformal.FavMac_GreedyProb, **kwargs.copy())
        evaluators['GR'] = Evaluator(conformal.FavMac_GreedyRatio, **kwargs.copy())

        evaluators = {k: _ for k, _ in evaluators.items() if k.split("@")[0] not in skip}
        if nclass > 11:  evaluators = {k: _ for k, _ in evaluators.items() if not k.endswith("+exact") and k != 'Full'}
        perm_seeds = list(np.arange(nseeds)) if dataset != _settings.CLAIMDEMOSeq_NAME else [None]
        return cls._add_tasks(evaluators, task_runner, add_key=task_runner is None, perm_seeds=perm_seeds)


    @classmethod
    def _CTScost_CTSutil(cls, task_runner, dataset, datakwargs, suffix, cost_fn, util_fn, proxy_fn, mode='last', skip=[], **kwargs):
        assert cost_fn.endswith("_cost") and util_fn.endswith("_util") and proxy_fn.endswith("_proxy"), f"{cost_fn}, {util_fn}, {proxy_fn}"
        return cls._CTScost(task_runner, dataset, datakwargs, suffix,
            cost_fn=cost_fn, util_fn=util_fn, proxy_fn=proxy_fn, mode=mode, skip=skip, **kwargs)

    @classmethod
    def _CTScost_TPutil(cls, task_runner, dataset, datakwargs, suffix, cost_fn, util_fn, proxy_fn, mode='last', skip=[]):
        assert cost_fn.endswith("_cost") and util_fn == 'sf_TP_util' and proxy_fn.endswith("_proxy"), f"{cost_fn}, {util_fn}, {proxy_fn}"
        return cls._CTScost(task_runner, dataset, datakwargs, suffix,
                            cost_fn=cost_fn, util_fn=util_fn, proxy_fn=proxy_fn, mode=mode, skip=skip)
    #==================================================================================================================================
    @classmethod
    def mnist_add(cls, task_runner=None):
        # MNIST, FP control, weighted value
        return cls._FPcost_CTSutil(task_runner, _settings.MNISTSup_NAME, {"sample_proba": 0.4, 'noise_level': None}, '',
            data_utils.SF_FP_cost, data_utils.SF_mnistadd_util, mode='loss')

    @classmethod
    def mnist_add_cost_TP(cls, task_runner=None):
        # MNIST, continous cost control, TP value
        return cls._CTScost_TPutil(task_runner, _settings.MNISTSup_NAME, {"sample_proba": 0.4, 'noise_level': None}, '',
            data_utils.SF_mnistadd_cost, data_utils.SF_TP_util, data_utils.SF_mnistadd_proxy, mode='loss')

    @classmethod
    def mnist_add_cost(cls, task_runner=None):
        # MNIST, continous cost control, weighted value
        return cls._CTScost_CTSutil(task_runner, _settings.MNISTSup_NAME, {"sample_proba": 0.4, 'noise_level': None}, '',
            data_utils.SF_mnistadd_cost, data_utils.SF_mnistadd_util, data_utils.SF_mnistadd_proxy, mode='loss')

    @classmethod
    def mnist_add_mult_util(cls, task_runner=None):
        # MNIST, FP control, GEN value
        return cls._FPcost_CTSutil(task_runner, _settings.MNISTSup_NAME, {"sample_proba": 0.4, 'noise_level': None}, '',
            data_utils.SF_FP_cost, data_utils.SF_mnistmult_util2, mode='loss')

    @classmethod
    def mnist_add_cost_mult_util(cls, task_runner=None):
        # MNIST, continous cost control, GEN value
        return cls._CTScost_CTSutil(task_runner, _settings.MNISTSup_NAME, {"sample_proba": 0.4, 'noise_level': None}, '',
            data_utils.SF_mnistadd_cost, data_utils.SF_mnistmult_util2, data_utils.SF_mnistadd_proxy, mode='loss')


    #==================================================================================================================================
    @classmethod
    def mimic3_few(cls, task_runner=None):
        # MIMIC(select), FP control, weighted value
        return cls._FPcost_CTSutil(task_runner, _settings.MIMICIIICompletion_NAME, {"use_notes": True, 'hcc_choice': 'few'}, '_few',
            data_utils.SF_FP_cost, data_utils.SF_mimic3few_util)

    @classmethod
    def mimic3_more(cls, task_runner=None):
        # MIMIC, FP control, weighted value
        return cls._FPcost_CTSutil(task_runner, _settings.MIMICIIICompletion_NAME, {"use_notes": True, 'hcc_choice': 'more'}, '_more',
            data_utils.SF_FP_cost, data_utils.SF_mimic3more_util)

    @classmethod
    def mimic3_few_cost_TP(cls, task_runner=None):
        # MIMIC(select), continous cost control, TP value
        return cls._CTScost_TPutil(task_runner, _settings.MIMICIIICompletion_NAME, {"use_notes": True, 'hcc_choice': 'few'}, '_few',
            data_utils.SF_mimic3few_cost, data_utils.SF_TP_util, data_utils.SF_mimic3few_proxy)
    @classmethod
    def mimic3_few_cost(cls, task_runner=None):
        # MIMIC(select), continous cost control, weighted value
        return cls._CTScost_CTSutil(task_runner, _settings.MIMICIIICompletion_NAME, {"use_notes": True, 'hcc_choice': 'few'}, '_few',
            data_utils.SF_mimic3few_cost, data_utils.SF_mimic3few_util, data_utils.SF_mimic3few_proxy)

    @classmethod
    def mimic3_more_cost_TP(cls, task_runner=None):
        # MIMIC, continous cost control, TP value
        return cls._CTScost_TPutil(task_runner, _settings.MIMICIIICompletion_NAME, {"use_notes": True, 'hcc_choice': 'more'}, '_more',
            data_utils.SF_mimic3more_cost, data_utils.SF_TP_util, data_utils.SF_mimic3more_proxy)
    @classmethod
    def mimic3_more_cost(cls, task_runner=None):
        # MIMIC, continous cost control, weighted value
        return cls._CTScost_CTSutil(task_runner, _settings.MIMICIIICompletion_NAME, {"use_notes": True, 'hcc_choice': 'more'}, '_more',
            data_utils.SF_mimic3more_cost, data_utils.SF_mimic3more_util, data_utils.SF_mimic3more_proxy)
    #====================================================================================================================================
    @classmethod
    def claim_few_random(cls, task_runner=None):
        return cls._FPcost_CTSutil(task_runner, _settings.CLAIMDEMO_NAME, {'hcc_choice': 'few', 'seq': True, 'topndiags': None}, '_few',
            data_utils.SF_FP_cost, data_utils.SF_claimfew_util, mode='loss')
    @classmethod
    def claim_few_cost_random(cls, task_runner=None):
        return cls._CTScost_CTSutil(task_runner, _settings.CLAIMDEMO_NAME, {'hcc_choice': 'few', 'seq': True, 'topndiags': None}, '_few',
            data_utils.SF_claimfew_cost, data_utils.SF_claimfew_util, data_utils.SF_claimfew_proxy, mode='loss')
    @classmethod
    def claim_few_cost_TP_random(cls, task_runner=None):
        return cls._CTScost_TPutil(task_runner, _settings.CLAIMDEMO_NAME, {'hcc_choice': 'few', 'seq': True, 'topndiags': None}, '_few',
            data_utils.SF_claimfew_cost, data_utils.SF_TP_util, data_utils.SF_claimfew_proxy, mode='loss')

    @classmethod
    def claim_more_random(cls, task_runner=None):
        return cls._FPcost_CTSutil(task_runner, _settings.CLAIMDEMO_NAME, {'hcc_choice': 'more', 'seq': True, 'topndiags': None}, '_more',
            data_utils.SF_FP_cost, data_utils.SF_claimmore_util, mode='loss', skip=[])
    @classmethod
    def claim_more_cost_random(cls, task_runner=None):
        return cls._CTScost_CTSutil(task_runner, _settings.CLAIMDEMO_NAME, {'hcc_choice': 'more', 'seq': True, 'topndiags': None}, '_more',
            data_utils.SF_claimmore_cost, data_utils.SF_claimmore_util, data_utils.SF_claimmore_proxy, mode='loss', skip=[])
    @classmethod
    def claim_more_cost_TP_random(cls, task_runner=None):
        return cls._CTScost_TPutil(task_runner, _settings.CLAIMDEMO_NAME, {'hcc_choice': 'more', 'seq': True, 'topndiags': None}, '_more',
            data_utils.SF_claimmore_cost, data_utils.SF_TP_util, data_utils.SF_claimmore_proxy, mode='loss', skip=[])

if __name__ == '__main__':
    Config.mnist_add().run_multi_process(32)
    Config.mimic3_more().run_multi_process(32)
    pass