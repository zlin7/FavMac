#Code for bounds from #https://github.com/aangelopoulos/rcps/blob/main/core/bounds.py
from collections import deque
from importlib import reload

import ipdb
import numpy as np
import numpy.ma as ma
import pandas as pd
import tqdm
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import binom

import conformal.common as cc
import data_utils


class RCPSMultiLabel(cc.Calibrator):
    #It is computationally way too expensiver to re-evaluate the threshold in an online manner for this method.
    #Thus, we use QuantileTree for faster online updates
    def __init__(self, cost_fn, util_fn, proxy_fn, target_cost, delta=None) -> None:
        from sortedcontainers import SortedList

        from conformal.queryquantile import QuantileTreeWrap
        assert proxy_fn is None
        assert delta is not None, "Otherwise this cannot work"
        super().__init__(cost_fn, None, None, target_cost, delta=delta)

        self.quantile_query = QuantileTreeWrap()
        self.C_max = 1.
        self._queue = deque()
        self.all_neg_ts = SortedList()
        self.actual_target_cost = None if self.cost_fn.name == data_utils.SF_FP_cost else self.target_cost

    def _add_sample(self, predset, extra_info):
        assert self.delta is not None

        costs, logit_thresholds, label, ks = extra_info
        assert pd.Series(costs).equals(pd.Series(costs).cummax())
        neg_ts = -logit_thresholds[ks]

        curr_cost = 0.
        for cost, neg_t in zip(costs, neg_ts):
            if cost > curr_cost:
                self.quantile_query.add(neg_t, gap = cost - curr_cost)
                curr_cost = cost
            self.all_neg_ts.add(neg_t)
        self._queue.append(extra_info)

    def _debug_eval_muhat(self, neg_t):
        past_logits = np.stack([_[1] for _ in self._queue],0)
        past_labels = np.stack([_[2] for _ in self._queue],0).astype(int)
        pred = past_logits >= -neg_t
        ccs = np.asarray([self.cost_fn(S.astype(int), Y) for S, Y in zip(pred, past_labels)])
        muhat = np.mean(ccs)
        print(muhat, muhat > self.target_cost)

    def _query_threshold(self):
        n = self._cnt
        assert self.delta is not None
        def _eval_t(neg_t):
            # inclusive so the predicton is >= t
            muhat = self.quantile_query.tree.query_sum(neg_t, inclusive=True) / n
            bd = HB_mu_plus(muhat, n, self.delta)
            return bd < self.actual_target_cost
        neg_t = cc.bin_search_sup(_eval_t, self.all_neg_ts)
        return -neg_t

    def _forward(self, logit, label=None):
        if self.actual_target_cost is None:
            self.actual_target_cost = np.floor(self.target_cost * len(logit)) / len(logit)
        pred = expit(logit)
        predset = None
        Ss, ks = cc.create_nested_sets_naive(pred, include_ks=True)
        if self.t is not None:
            predset = (logit >= self.t).astype(int)
        if label is not None:
            costs = [self.cost_fn(S, label) for S in Ss[1:]]
        return predset, (costs, logit, label, ks)


#https://github.com/aangelopoulos/rcps/blob/4e054066e3746450a69f4f0730a81dce9898d6da/core/bounds.py
def h1(y, mu):
    return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))

def hoeffding_plus(mu, x, n):
    return -n * h1(np.maximum(mu,x),mu)

def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n*x),n,mu),1e-10))+1

def HB_mu_plus(muhat, n, delta, maxiters=1000):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n)
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

if __name__ == "__main__":
    pass