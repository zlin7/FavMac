import math
import time
from typing import List, Optional, Union

import ipdb
import numpy as np
import pandas as pd

import data_utils.preprocessing.code_utils as code_utils

INTEGER_SAFE_DELTA = 0.1
# INTEGER_SAFE_DELTA is added to the cost target so it's between two integers.

class SetFunction:
    def __init__(self, name='unknown') -> None:
        self.name = name

    def __call__(self, S: np.ndarray) -> float:
        raise NotImplementedError()

    def is_additive(self):
        return False

def generate_all_pred_sets_masked(K, mask=None):
    assert K < 12
    S = np.zeros(K, dtype=int)
    K = mask.sum()
    locs = [i for i, _ in enumerate(mask) if _]
    bases = [2 ** _ for _ in range(K)]
    for i in range(2 ** K):
        ret = S.copy()
        for j, base in enumerate(bases):
            if base & i > 0:
                ret[locs[j]] = 1
        yield ret

def montecarlo_eval(set_fn, S: np.ndarray, pred: np.ndarray, sample: int=1000, tail_value=None) -> float:
    vv = np.random.uniform(0, 1, size=(sample, len(pred)))
    if tail_value is None:
        vhats = [set_fn(S * Y) for Y in (vv < pred).astype(int)]
        return np.mean(vhats)
    else:
        excess = [set_fn(S * Y) > tail_value for Y in (vv < pred).astype(int)]
        return np.mean(excess)

def exact_pred_eval(set_fn, S: np.ndarray, pred: np.ndarray, tail_value=None, **kwargs):
    ret = 0.
    mask = S.astype(bool)
    if tail_value is None:
        for _S in generate_all_pred_sets_masked(len(pred), mask=mask):
            joint_p = np.product((_S * pred + (1-_S) * (1-pred))[mask])
            val = set_fn(_S)
            ret += joint_p * val
    else:
        for _S in generate_all_pred_sets_masked(len(pred), mask=mask):
            if set_fn(_S) > tail_value:
                ret += np.product((_S * pred + (1-_S) * (1-pred))[mask])
    return ret

class GeneralSetFunction(SetFunction):
    def __init__(self, mode, sample=500, revert_to_exact=11, name='unknown') -> None:
        super().__init__(name=name)
        self.mode = mode
        self.sample = sample
        self.revert_to_exact = revert_to_exact
        self.dummy_greedy_maximize_seq = False

    def is_additive(self):
        return False

    def naive_call(self, S: np.ndarray) -> float:
        raise NotImplementedError()

    def __call__(self, S: np.ndarray, Y:np.ndarray=None, pred:np.ndarray=None, sample=None, target_cost=None) -> float:
        if sample is None: sample = self.sample
        if self.mode == 'cost':
            assert pred is None
            return self.cost_call(S, Y)
        if self.mode == 'proxy':
            assert Y is None
            return self.proxy_call(S, pred, target_cost=target_cost, sample=sample)
        assert self.mode == 'util'
        return self.util_call(S, Y, pred, sample=sample)

    def cost_call(self, S: np.ndarray, Y:np.ndarray) -> float:
        return self.naive_call(S * (1-Y))

    def util_call(self, S: np.ndarray, Y:np.ndarray=None, pred:np.ndarray=None, sample=None) -> float:
        if sample is None: sample = self.sample
        assert Y is None or pred is None
        if pred is not None:
            if len(S) <= self.revert_to_exact: return exact_pred_eval(self.naive_call, S, pred)
            return montecarlo_eval(self.naive_call, S, pred, sample)
        if Y is not None: return self.naive_call(S * Y)
        return self.naive_call(S)

    def proxy_call(self, S: np.ndarray, pred: np.ndarray, target_cost: float=None, sample=None) -> float:
        if sample is None: sample = self.sample
        if len(S) <= self.revert_to_exact: return exact_pred_eval(self.naive_call, S, 1-pred, tail_value=target_cost)
        return montecarlo_eval(self.naive_call, S, 1-pred, sample, tail_value=target_cost)

    def greedy_maximize(self, S: np.ndarray, pred: np.ndarray, prev_util: float=None, sample: int=None, d_proxy:np.ndarray=None):
        assert self.mode == 'util'
        if sample is None: sample = self.sample
        # Find the element to add that results in most value increse.
        if S.min() == 1: return [None, None]
        if prev_util is None: prev_util = self(S, pred=pred, sample=sample)
        S = S.copy()
        best = [None, -np.inf] #
        eps = 1e-8
        d_utils = {}
        for k in range(len(S)):
            if S[k] == 1: continue
            S[k] = 1
            d_util = self(S, pred=pred, sample=sample) - prev_util
            S[k] = 0
            if d_util <= 0: continue

            d_utils[k] = d_util
            val = d_util if d_proxy is None else (d_util / max(eps, d_proxy[k]))
            if val > best[1]: best = [k, val]
        return [best[0], d_utils.get(best[0], 0) + prev_util]

    def greedy_maximize_seq(self, pred: np.ndarray=None, S_base:np.ndarray=None, sample=None, d_proxy:np.ndarray=None):
        assert self.mode == 'util'
        if self.dummy_greedy_maximize_seq: #This is a hack....
            ks = np.argsort(-pred)
            Ss = [np.zeros(len(pred), dtype=int)]
            for k in ks:
                Ss.append(Ss[-1].copy())
                Ss[-1][k] = 1
            return Ss, {"ks": ks}
        if sample is None: sample = self.sample
        Ss = [S_base]
        if S_base is None:
            assert pred is not None
            Ss = [np.zeros(len(pred), dtype=int)]
        utils = [self(Ss[-1], pred=pred)]
        ks = []
        new_k, new_util = self.greedy_maximize(Ss[-1], pred=pred, prev_util=utils[-1], sample=sample, d_proxy=d_proxy)
        while new_k is not None:
            Ss.append(Ss[-1].copy())
            Ss[-1][new_k] = 1
            utils.append(new_util)
            ks.append(new_k)
            new_k, new_util = self.greedy_maximize(Ss[-1], pred=pred, prev_util=utils[-1], sample=sample, d_proxy=d_proxy)
        return Ss, {'ks': ks, 'utils': utils}

class MNISTMult_util2(GeneralSetFunction):
    def __init__(self, weight_on_value=None, name='unknown') -> None:
        super().__init__('util', name=name)
        self.ks = np.arange(10)
        self.ks[0] = 10

        self.prod_vals = 1 + (self.ks - 5)/10
        self.sum_vals = np.square((self.ks-5))
        self._C_max = 86.09

        assert weight_on_value is None or weight_on_value == 0
        self.dummy_greedy_maximize_seq = weight_on_value == 0

    def naive_call(self, S: np.ndarray, normalize=True) -> float:
        S = S.astype(bool)
        ret = np.product(self.prod_vals[S]) + np.sum(self.sum_vals[S])
        if not normalize: return ret
        ret = ret / self._C_max
        assert ret <= 1, "util too large"
        return ret

def generate_all_pred_sets(K, mask=None):
    assert K < 12
    bases = [2 ** _ for _ in range(K)]
    for i in range(2 ** K):
        S = np.zeros(K, dtype=int)
        for j, base in enumerate(bases):
            if base & i > 0:
                S[j] = 1
        yield S

class AdditiveViolationEstimation: #This does NOT need to be normalized!!
    def __init__(self, values, mode='exact', revert_to_exact=11) -> None:
        self.values = values
        assert type(values) in {np.ndarray, float, int}
        self.mode = mode
        self.revert_to_exact = 0 if revert_to_exact is None else revert_to_exact
        assert mode in {'exact', 'gaussian', 'montecarlo', 'expectation'}, mode

    def __call__(self, S: Union[np.ndarray, List[np.ndarray]], pred:np.ndarray, tc:float) -> float:
        if isinstance(S, list): return [self(_S, pred) for _S in S]
        mask = S.astype(bool)
        phat = 1-pred[mask]
        if isinstance(self.values, np.ndarray):
            weights = self.values[mask]
        else:
            weights = self.values * np.ones(phat.shape)
        if len(phat) == 0: return 0.
        if self.mode == 'expectation':
            return np.sum(weights * phat)
        if self.mode == 'exact' or len(phat) <= self.revert_to_exact:
            return self.exact_pred_violation(phat, weights, tc)
        if self.mode == 'gaussian':
            return self.gaussian_approximation(phat, weights, tc)
        if self.mode == 'montecarlo':
            return self.montecarlo_approximation(phat, weights, tc)

    @classmethod
    def gaussian_approximation(cls, phat, weights, tc):
        from scipy.stats import norm
        var_each = np.square(weights) * phat * (1-phat)
        return 1-norm.cdf(tc, loc=np.sum(phat * weights), scale=np.sqrt(np.sum(var_each)))

    @classmethod
    def exact_pred_violation(cls, phat, weights, tc):
        ret = 0.
        for S in generate_all_pred_sets(len(phat)):
            if np.sum(S * weights) > tc:
                ret += np.product(S * phat + (1-S) * (1-phat))
        return ret

    @classmethod
    def montecarlo_approximation(cls, pred, weights, tc, n=10000, seed=None):
        vv = np.random.RandomState(seed).uniform(0, 1, size=(n, len(pred)))
        return np.mean(np.sum((vv < pred) * weights, 1) > tc)

class AdditiveSetFunction(SetFunction):
    def __init__(self, values: Union[float, np.ndarray, int], mode=None, weight_on_value=1, name='unknown',
        quantile_method='expectation', revert_to_exact=0) -> None:
        super().__init__(name=name)
        self.values = values
        self.weight_on_value = weight_on_value
        assert weight_on_value in {1, 0}
        assert mode is None or mode in {'util', 'cost', 'proxy'}
        self.mode = mode
        self._C_max = self.values.sum() if isinstance(self.values, np.ndarray) else None
        self.quantile_method = AdditiveViolationEstimation(self.values, quantile_method, revert_to_exact=revert_to_exact)

    def is_additive(self):
        if self.mode == 'proxy' and self.quantile_method.mode != 'expectation': return False
        return True

    def __call__(self, S: np.ndarray, Y:np.ndarray=None, pred:np.ndarray=None, sample=100, target_cost=None) -> float:
        if self.mode == 'cost':
            assert pred is None
            return self.cost_call(S, Y)
        if self.mode == 'proxy':
            assert Y is None
            return self.proxy_call(S, pred, target_cost=target_cost)
        assert self.mode == 'util'
        return self.util_call(S, Y, pred, sample=sample)

    def naive_call(self, S: np.ndarray) -> float:
        C_max = self._C_max or len(S) * self.values
        return np.sum(S * self.values) / C_max

    def util_call(self, S: np.ndarray, Y:np.ndarray=None, pred:np.ndarray=None, sample=1000) -> float:
        assert Y is None or pred is None
        if pred is not None:
            return self.naive_call(S * pred) # THis is because this is additive.
        if Y is not None: return self.naive_call(S * Y)
        return self.naive_call(S)

    def cost_call(self, S: np.ndarray, Y:np.ndarray) -> float:
        return self.naive_call(S * (1-Y))

    def proxy_call(self, S: np.ndarray, pred: np.ndarray, target_cost: float=None) -> float:
        if target_cost is None: #
            return self.naive_call(S * (1-pred))
        return self.quantile_method(S, pred, target_cost) #Does not need to be normalized

    def greedy_maximize(self, S: np.ndarray, pred: np.ndarray=None, weight_on_value: float=None, d_proxy:np.ndarray=None, prev_util_and_proxy=None):
        # (prev_u, prev_p) = prev_util_and_proxy
        assert self.mode == 'util', "This is only used for util function"
        if (1-S).sum() == 0: return None
        if weight_on_value is None: weight_on_value = self.weight_on_value
        d_util = weight_on_value * self.values + (1-weight_on_value) * np.mean(self.values)
        d_util = self.weight_on_value * self.values + (1-self.weight_on_value) * np.mean(self.values)
        if pred is not None: d_util = d_util * pred

        # if min(d_proxy) < 1e-8: ipdb.set_trace() montecarlo could create all 0s
        objective = d_util / (1 if d_proxy is None else d_proxy.clip(1e-8))
        k = pd.Series((1-S) * objective).dropna().idxmax()
        return k, objective[k]

    def greedy_maximize_seq(self, pred: np.ndarray=None, weight_on_value: float=None, d_proxy:np.ndarray=None):
        # if cost is also additive, then cost_proxy is fixed: weight * (1-p)
        assert self.mode == 'util', "This is only used for util function"
        if weight_on_value is None: weight_on_value = self.weight_on_value
        d_util = weight_on_value * self.values + (1-weight_on_value) * np.mean(self.values)
        if pred is not None: d_util = d_util * pred

        # if min(d_proxy) < 1e-8: ipdb.set_trace() montecarlo could create all 0s
        objective = d_util / (1 if d_proxy is None else d_proxy.clip(1e-8))

        assert np.isnan(objective).sum() == 0
        ks = np.argsort(-objective)
        Ss = [np.zeros(len(objective), dtype=int)]
        for k in ks:
            Ss.append(Ss[-1].copy())
            Ss[-1][k] = 1
        return Ss, ks

# importable things ('sf_[data]_[cost/util/proxy]', sf=set function)
SF_mimic3few_cost = 'sf_mimic3few_cost'
SF_mimic3more_cost = 'sf_mimic3more_cost'
SF_mimic3few_proxy = 'sf_mimic3few_proxy'
SF_mimic3more_proxy = 'sf_mimic3more_proxy'
SF_mimic3few_util = 'sf_mimic3few_util'
SF_mimic3more_util = 'sf_mimic3more_util'

SF_claimfew_cost = 'sf_claimfew_cost'
SF_claimmore_cost = 'sf_claimmore_cost'
SF_claimfew_proxy = 'sf_claimfew_proxy'
SF_claimmore_proxy = 'sf_claimmore_proxy'
SF_claimfew_util = 'sf_claimfew_util'
SF_claimmore_util = 'sf_claimmore_util'

SF_mnistadd_util = 'sf_mnistadd_util'
SF_mnistadd_cost = 'sf_mnistadd_cost'
SF_mnistadd_proxy = 'sf_mnistadd_proxy'

SF_mnistmult_util2 = 'sf_mnistmult2_util'

SF_FP_proxy = 'sf_FP_proxy'
SF_FP_cost = 'sf_FP_cost'
SF_TP_util = 'sf_TP_util'

class TrainedProxy(SetFunction):
    def __init__(self, name:str) -> None:
        assert name.startswith("TrainedProxy|")
        super().__init__(name)
        assert name.endswith("_regcost")
        key = name.replace("TrainedFP|", "")
        import pipeline.trainer as tr
        self.model, settings, _ = tr.CallBack.load_state(None, key, mode='last', device='cpu')
        self.model.eval()

    def __call__(self, S: np.ndarray, pred: np.ndarray) -> float:
        import torch
        logit = -np.log(1/pred - 1)
        with torch.no_grad():
            logit = torch.tensor(logit, dtype=torch.float).unsqueeze(0)
            S = torch.tensor(S, dtype=torch.float).unsqueeze(0)
            ret = self.model({"data": logit, "mask": S})[0].numpy()
        return float(ret)

class FPProxy(SetFunction):
    def __init__(self, name:str) -> None:
        assert name.startswith("TrainedFP|")
        super().__init__(name)
        key = name.replace("TrainedFP|", "")
        import pipeline.trainer as tr
        self.model, settings, _ = tr.CallBack.load_state(None, key, mode='last', device='cpu')
        self.model.eval()

    def __call__(self, S: np.ndarray, pred: np.ndarray, target_cost: float=None) -> float:
        import torch
        from scipy.special import expit, softmax
        logit = -np.log(1/pred.clip(1e-5,1-1e-5) - 1)
        with torch.no_grad():
            logit = torch.tensor(logit, dtype=torch.float).unsqueeze(0)
            S = torch.tensor(S, dtype=torch.float).unsqueeze(0)
            ret = self.model({"data": logit, "mask": S})[0].numpy()
        deepset_pred = softmax(ret)
        K = len(deepset_pred) - 1
        if target_cost is not None:
            kint = int(np.floor(target_cost * K))
            #assert abs(target_cost * K - kint - INTEGER_SAFE_DELTA) < 1e-3
            return 1 - deepset_pred[:kint + 1].sum()
        else:
            return (np.arange(K+1) * deepset_pred).sum() / K

def get_set_fn(set_fn_or_name, **kwargs):
    if not isinstance(set_fn_or_name, str): return set_fn_or_name
    kwargs.setdefault('name', set_fn_or_name)
    if set_fn_or_name.startswith("sf_claim"):
        values = code_utils.HCCV24.get_risk_factors(set_fn_or_name.replace("sf_claim", "").split("_")[0]).values
        return AdditiveSetFunction(values, mode=set_fn_or_name.split("_")[-1], **kwargs)
    if set_fn_or_name.startswith("sf_mimic3"):
        values = code_utils.HCC2014.get_risk_factors(set_fn_or_name.replace("sf_mimic3", "").split("_")[0]).values
        return AdditiveSetFunction(values, mode=set_fn_or_name.split("_")[-1], **kwargs)
    if set_fn_or_name.startswith("sf_mnistadd"):
        values = np.arange(10)
        values[0] = 10
        return AdditiveSetFunction(values, mode=set_fn_or_name.split("_")[-1], **kwargs)
    if set_fn_or_name == SF_mnistmult_util2: return MNISTMult_util2(**kwargs)

    #basic ones related to FP/TP
    if set_fn_or_name == SF_FP_cost: return AdditiveSetFunction(1., mode='cost', **kwargs)
    if set_fn_or_name == SF_FP_proxy: return AdditiveSetFunction(1., mode='proxy', **kwargs)
    if set_fn_or_name == SF_TP_util: return AdditiveSetFunction(1., mode='util', **kwargs)
    if set_fn_or_name.startswith("TrainedFP|"):
        return FPProxy(**kwargs)
    print(f"Unknown proxy: {set_fn_or_name}")
    return set_fn_or_name # Unkonwn, but it's OK for DeepSetBased_fast

def is_additive(set_fn_or_name):
    set_fn_or_name = get_set_fn(set_fn_or_name)
    if isinstance(set_fn_or_name, str): return False
    return set_fn_or_name.is_additive()

if __name__ == '__main__':
    pass