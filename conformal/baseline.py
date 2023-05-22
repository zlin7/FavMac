from collections import deque
from importlib import reload

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

import conformal.common as cc
import data_utils
from conformal.favmac import INTEGER_SAFE_DELTA, FavMac


class NaiveMethod(cc.Calibrator):
    def __init__(self, cost_fn, util_fn, proxy_fn, target_cost, delta=None) -> None:
        super().__init__(cost_fn, util_fn, proxy_fn, target_cost, delta)
        self.C_max = 1.
        self.cache_costsplus = deque([])

    @classmethod
    def _find_cost(cls, costs, t):
        costs = costs[costs.index < t]
        return 0. if len(costs) == 0 else costs.values[-1]

    def _forward(self, logit, label=None):
        pred = expit(logit)
        Ss = cc.create_nested_sets_naive(pred)

        fps, predset = None, None
        cost_hats = [self.proxy_fn(S, pred=pred) for S in Ss]
        if label is not None:
            fps = [self.cost_fn(S, label) for S in Ss]
        if self.t is not None:
            predset = [S for S,chat in zip(Ss, cost_hats) if chat < self.t][-1]
        return predset, (fps, cost_hats)

    def _add_sample(self, predset, extra_info):
        costs, scores = extra_info
        _cost_plus = pd.Series(costs, index=scores).cummax()
        self.cache_costsplus.append(_cost_plus)

    def _query_threshold(self):
        all_ts = np.sort(np.concatenate([_.index for _ in self.cache_costsplus]))
        n = len(self.cache_costsplus)
        if self.delta is None:
            def _eval_t(t):
                _sum = 0.
                for _cost_plus in self.cache_costsplus:
                    _sum += self._find_cost(_cost_plus, t)
                return _sum <= (n+1)* self.target_cost - self.C_max
        else:
            def _eval_t(t):
                cnt = 0
                for _cost_plus in self.cache_costsplus:
                    if self._find_cost(_cost_plus, t) <= self.target_cost:
                        cnt += 1
                return cnt >= (n + 1) * (1-self.delta)
        return cc.bin_search_sup(_eval_t, all_ts)

class FPCP_fast(FavMac):
    def __init__(self, cost_fn, util_fn, proxy_fn, target_cost, delta=None, device='cpu') -> None:
        assert cost_fn.name == data_utils.SF_FP_cost
        assert isinstance(proxy_fn, str)
        import pipeline.trainer as tr
        self.model, settings, _ = tr.CallBack.load_state(None, proxy_fn, mode='last', device=device)
        self.model.eval()
        super().__init__(cost_fn, None, self.proxy_fn, target_cost, delta=delta)

    def proxy_fn(self, S, pred):
        logit = -np.log(1/pred.clip(1e-5,1-1e-5) - 1)
        deepset_pred = softmax(self.model.np_pred(logit, S))
        K = len(deepset_pred) - 1
        if self.delta is not None:
            kint = int(np.floor(self.target_cost * K))
            #assert abs(self.target_cost * K - kint - INTEGER_SAFE_DELTA) < 1e-3
            return 1 - deepset_pred[:kint + 1].sum()
        else:
            return (np.arange(K+1) * deepset_pred).sum() / K

    def _forward(self, logit, label=None):
        pred = expit(logit)
        Ss = cc.create_nested_sets_naive(pred)

        fps, predset = None, np.zeros(len(logit), dtype=int)
        cost_hats = [self.proxy_fn(S, pred=pred) for S in Ss]
        if label is not None:
            fps = [self.cost_fn(S, label) for S in Ss]
        if self.t is not None:
            for S, chat in zip(Ss, cost_hats):
                if chat >= self.t: continue
                if S.sum() > predset.sum():
                    predset = S
        return predset, (fps, cost_hats)

class FPCP(NaiveMethod):
    def __init__(self, cost_fn, util_fn, proxy_fn, target_cost, delta=None, device='cpu') -> None:
        assert cost_fn.name == data_utils.SF_FP_cost

        assert isinstance(proxy_fn, str)
        import pipeline.trainer as tr
        self.model, settings, _ = tr.CallBack.load_state(None, proxy_fn, mode='last', device=device)
        self.model.eval()
        super().__init__(cost_fn, None, self.proxy_fn, target_cost, delta=delta)

        self.cache_FPmax = deque([])

    def proxy_fn(self, S, pred):
        logit = -np.log(1/pred.clip(1e-5,1-1e-5) - 1)
        deepset_pred = softmax(self.model.np_pred(logit, S))
        K = len(deepset_pred) - 1
        if self.delta is not None:
            kint = int(np.floor(self.target_cost * K))
            #assert abs(self.target_cost * K - kint - INTEGER_SAFE_DELTA) < 1e-3
            return 1 - deepset_pred[:kint + 1].sum()
        else:
            return (np.arange(K+1) * deepset_pred).sum() / K

class IndividualCPSet(cc.Calibrator):
    def __init__(self, cost_fn, util_fn, proxy_fn, target_cost, delta=None) -> None:
        assert cost_fn.name == data_utils.SF_FP_cost and proxy_fn is None
        super().__init__(cost_fn, None, None, target_cost, delta=delta)

        self.quantile_querys = {}
        self._queue = deque()

        assert delta is None, "Otherwise this is the same as GreedyProb"

    def _get_query_obj(self, i):
        from conformal.queryquantile import SortedQuantile
        if i not in self.quantile_querys:
            self.quantile_querys[i] = SortedQuantile()
        return self.quantile_querys[i]

    def _add_sample(self, predset, extra_info):
        logit, label = extra_info
        self._queue.append((logit, label))
        if self.delta is None:
            for k, y in enumerate(label):
                if y == 1: continue
                self._get_query_obj(k).add(logit[k], 1)
        else:
            #This is unncessary. It's the same as GreedyProb
            raise NotImplementedError()

    def _query_threshold(self):
        n = self._cnt

        if self.delta is None:
            K = len(self.quantile_querys)
            #NOTE: We need this adjustment because we used INTEGER_SAFE_DELTA
            actual_tc = np.floor(self.target_cost * K) / K
            ret = []
            for k in range(K):
                obj = self._get_query_obj(k)
                n = len(obj.sl)
                cutoff = int(np.ceil((n+1) * (1-actual_tc)))
                ret.append(obj.query(cutoff))
            return np.asarray(ret)
        else:
            raise NotImplementedError()

    def _forward(self, logit, label=None):
        predset = None
        #logit, label = logit[1:2], label[1:2]
        if self.t is not None:
            predset = (logit > self.t).astype(int)
        return predset, (logit, label)

if __name__ == '__main__':
    pass