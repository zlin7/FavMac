from collections import deque
from importlib import reload

import ipdb
import numpy as np
import numpy.ma as ma
import pandas as pd
import tqdm
from scipy.special import expit

import conformal.common as cc
import data_utils


class InnerSet(cc.Calibrator):
    def __init__(self, cost_fn, util_fn, proxy_fn, target_cost, delta=None) -> None:
        from conformal.queryquantile import SortedQuantile
        assert cost_fn.name == data_utils.SF_FP_cost
        import pipeline.trainer as tr
        self.model, settings, _ = tr.CallBack.load_state(None, proxy_fn, mode='last')
        self.model.eval()
        super().__init__(cost_fn, None, None, target_cost, delta=delta)

        self.quantile_query = SortedQuantile()

    def _add_sample(self, predset, extra_info):
        S_i = extra_info
        self.quantile_query.add(S_i, gap=1)

    def _query_threshold(self):
        n = self._cnt
        if self.delta is None:
            cutoff = (n+1) * (1-self.target_cost)
        else:
            cutoff = (n+1) * (1-self.delta)
        return self.quantile_query.query(int(np.ceil(cutoff)), inclusive=False)

    def _forward(self, logit, label=None):
        import torch
        pred = expit(logit)
        with torch.no_grad():
            t = self.model(torch.tensor(logit, dtype=torch.float).unsqueeze(0))['out'].numpy()
        t_in = - t[:, 0]
        #t_out = t[:, 1]
        predset, s_i = None, None
        if label is not None:
            s_i = np.inf
            #if label.max() == 1:
            #    s_i = min(s_i, ma.masked_array(logit, mask=label == 0).min() - t_out)
            if label.min() == 0:
                s_i = min(s_i, t_in - ma.masked_array(logit, mask=label == 1).max())# mask =1 mean invalid -> y = 1
        if self.t is not None:
            t_in_adj = t_in + self.t
            predset = (logit > t_in_adj).astype(int)
        return predset, -s_i

if __name__ == '__main__':
    pass