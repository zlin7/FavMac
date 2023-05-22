from collections import deque
from importlib import reload

import numpy as np
import pandas as pd
from scipy.special import expit

import conformal.common as cc
from conformal.queryquantile import QuantileTreeWrap
from data_utils import INTEGER_SAFE_DELTA


class FavMac(cc.Calibrator):
    def __init__(self, cost_fn, util_fn, proxy_fn, target_cost, delta=None) -> None:
        super().__init__(cost_fn, util_fn, proxy_fn, target_cost, delta=delta)

        self.quantile_query = QuantileTreeWrap()
        self._queue = deque()
        self.C_max = 1.

    def _add_sample(self, predset, extra_info):
        costs, proxies = extra_info
        _sidx = np.argsort(proxies)
        costs = np.asarray(costs)[_sidx]
        proxies = np.asarray(proxies)[_sidx]
        assert max(costs) <= 1
        if self.delta is not None:
            costs = pd.Series(costs).cummax().values
            _valid_ts = [score for cost, score in zip(costs, proxies) if cost > self.target_cost] # more like invalid ts
            t_k_i = min(_valid_ts) if len(_valid_ts) > 0 else np.inf # max_{t}\{C^+(Chat<t)\leq c}
            self.quantile_query.add(t_k_i, gap=1)
            self._queue.append(t_k_i)
        else:
            curr_cost = 0
            for cost, score in zip(costs, proxies):
                if cost > curr_cost:
                    self.quantile_query.add(score, gap = cost - curr_cost)
                    curr_cost = cost
            self._queue.append((costs, proxies))

    def _query_threshold(self):
        n = len(self._queue)
        if self.delta is None:
            cutoff = self.target_cost * (n+1) - self.C_max
            return self.quantile_query.query(cutoff, inclusive=False)
        else:
            cutoff = self.delta * (n+1) - 1# We should assume a violation for the next point? Should we minus 1??
            return self.quantile_query.query(cutoff, inclusive=False)

    def _greedy_sequence(self, pred:np.ndarray):
        raise NotImplementedError()

    def _forward(self, logit, label=None):
        # return predset, (costs, cost_proxies)
        # costs[j] or cost_proxies[j] is for S_j
        # (S_0 \subset S_1 \ldots S_K)
        pred = expit(logit)
        Ss, proxies = self._greedy_sequence(pred)
        costs, predset = None, None
        if label is not None:
            costs = [self.cost_fn(S, label) for S in Ss]
        if self.t is not None:
            candidates = [S for S,v in zip(Ss, proxies) if v < self.t]
            predset = Ss[0] if len(candidates) == 0 else candidates[-1]
        return predset, (costs, proxies)

class FavMac_GreedyRatio(FavMac):
    #In each step, maximize dValue/dProxy.
    def _greedy_sequence(self, pred:np.ndarray):
        proxy_fn = lambda _S: self.proxy_fn(_S, pred=pred, target_cost = None if self.delta is None else self.target_cost)
        try:
            if self.proxy_fn.is_additive():
                Ss, _ = self.util_fn.greedy_maximize_seq(pred=pred, d_proxy = self.proxy_fn.values * (1-pred))
                return Ss, list(map(proxy_fn, Ss))
        except:
            pass

        Ss = [np.zeros(len(pred), dtype=int)]
        proxies = [proxy_fn(Ss[0])]
        while Ss[-1].min() == 0:
            S = Ss[-1].copy()
            curr_d_proxies = [np.nan] * len(S)
            for k in range(len(S)):
                if S[k] == 1: continue
                S[k] = 1
                curr_d_proxies[k] = proxy_fn(S) - proxies[-1]
                S[k] = 0
            k, du_div_dp = self.util_fn.greedy_maximize(S, pred=pred, d_proxy = np.asarray(curr_d_proxies))
            if k is None: break
            S[k] = 1
            Ss.append(S)
            proxies.append(curr_d_proxies[k] + proxies[-1])
        return Ss, proxies


class FavMac_GreedyProb(FavMac):
    def _forward(self, logit, label=None):
        pred = expit(logit)
        Ss = cc.create_nested_sets_naive(pred)
        costs, predset = None, None
        proxy_kwargs = {'pred': pred}
        if self.delta is not None: proxy_kwargs['target_cost'] = self.target_cost
        proxies = [self.proxy_fn(S, **proxy_kwargs) for S in Ss]
        if label is not None:
            costs = [self.cost_fn(S, label) for S in Ss]
        if self.t is not None:
            candidates = [S for S,v in zip(Ss, proxies) if v < self.t]
            predset = Ss[0] if len(candidates) == 0 else candidates[-1]
        return predset, (costs, proxies)


class FavMac_GreedyValue(FavMac):
    def _forward(self, logit, label=None):
        pred = expit(logit)
        Ss, _ = self.util_fn.greedy_maximize_seq(pred=pred)
        costs, predset = None, None
        proxy_kwargs = {'pred': pred}
        if self.delta is not None: proxy_kwargs['target_cost'] = self.target_cost
        proxies = [self.proxy_fn(S, **proxy_kwargs) for S in Ss]
        if label is not None:
            costs = [self.cost_fn(S, label) for S in Ss]
        if self.t is not None:
            candidates = [S for S,v in zip(Ss, proxies) if v < self.t]
            predset = Ss[0] if len(candidates) == 0 else candidates[-1]
        return predset, (costs, proxies)


def generate_all_pred_sets(K):
    bases = [2 ** _ for _ in range(K)]
    for i in range(2 ** K):
        S = np.zeros(K, dtype=int)
        for j, base in enumerate(bases):
            if base & i > 0:
                S[j] = 1
        yield S

class FullUniverse(FavMac):
    def _forward(self, logit, label=None):
        assert len(logit) <= 11, "Otherwise too expensive"
        pred = expit(logit)
        Ss = [S for S in generate_all_pred_sets(len(pred))]
        costs, predset = None, None
        proxies = pd.Series([self.proxy_fn(S, pred=pred) for S in Ss]).sort_values()
        Ss = [Ss[_] for _ in proxies.index]
        if label is not None:
            costs = pd.Series([self.cost_fn(S, label) for S in Ss]).cummax().values
        if self.t is not None:
            main = pd.DataFrame({"set": Ss, 'proxy': proxies})
            main = main[main['proxy'] < self.t]
            #NOTE: This leveraging the additive property
            main['u'] = main['set'].map(lambda S: self.util_fn(S, pred=pred))
            predset = main.loc[main['u'].idxmax(), 'set']
        return predset, (costs, proxies)
if __name__ == '__main__':
    pass