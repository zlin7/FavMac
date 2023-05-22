import numpy as np
import tqdm

import data_utils.set_function as set_function


def bin_search_sup(fn, vals, debug=False):
    """
        fn(v) = True -> v is good
        there is a switching point for fn to turn True from False.
        We need to find the sup of v
    """
    lb, ub = 0, len(vals) - 1
    mem = {}
    def _fn(loc):
        v = vals[loc]
        res = mem.get(v, None)
        if res is None:
            res = mem[v] = fn(v)
        return res
    assert _fn(lb), "lower bound is not low enough it seems"
    good_locs = set([lb])
    #with tqdm.tqdm(total=int(np.log2(len(vals)))) as pbar:
    if True:
        while lb <= ub:
            if debug:
                print(vals[lb], vals[ub])
            if _fn(ub):
                good_locs.add(ub)
                break
            if not _fn(lb):
                break
            if lb == ub: break
            mid = (lb + ub) // 2
            if _fn(mid):
                good_locs.add(mid)
                lb = mid + 1
            else:
                ub = mid - 1
    if debug:
        print([vals[_] for _ in good_locs])
        print(mem)
    ret_loc = max(good_locs)
    if ret_loc < ub:
        assert not _fn(ret_loc + 1)
    return vals[ret_loc]


# Nested Set creation
def create_nested_sets_naive(pred, include_ks=False):
    S0 = np.zeros(len(pred), dtype=int)
    Ss = [S0]
    for k in np.argsort(-pred):
        Scurr = Ss[-1].copy()
        Scurr[k] = 1
        Ss.append(Scurr)
    if include_ks: return Ss, np.argsort(-pred)
    return Ss


class Calibrator:
    def __init__(self, cost_fn:set_function.SetFunction, util_fn:set_function.SetFunction, proxy_fn:set_function.SetFunction, target_cost, delta=None) -> None:
        self.target_cost = target_cost
        self.delta = delta

        self.cost_fn = cost_fn # (S, Y)
        self.util_fn = util_fn # (S, Y=None, pred=None)
        self.proxy_fn = proxy_fn # (S, pred)

        #Threshold
        self.t = None

        # reserved parameters to avoid repeating queries
        self._thresholds_mem = {}
        self._cnt = 0

    def _forward(self, logit, label=None):
        # return predset, extra_info
        raise NotImplementedError()

    def _add_sample(self, predset, extra_info):
        raise NotImplementedError()

    def _query_threshold(self):
        raise NotImplementedError()

    def query_threshold(self, target_cost=None):
        if target_cost is not None:
            old_target_cost = self.target_cost
            self.target_cost = target_cost
            t = self._query_threshold()
            self.target_cost = old_target_cost
            return t
        if self._cnt not in self._thresholds_mem:
            self._thresholds_mem[self._cnt] = self._query_threshold()
        # print(self._thresholds_mem); ipdb.set_trace()
        return self._thresholds_mem[self._cnt]

    def update(self, logit, label):
        predset, extra_info = self._forward(logit, label)
        self._add_sample(predset, extra_info)
        self._cnt += 1
        self.t = self.query_threshold()
        return predset, extra_info

    def init_calibrate(self, logits, labels):
        n = len(logits)
        for i, (_logit, _y) in tqdm.tqdm(enumerate(zip(logits, labels)), desc='initial calibration...', total=n):
            predset, extra_info = self._forward(_logit, _y)
            self._add_sample(predset, extra_info)
            self._cnt += 1
        self.t = self.query_threshold()

    def __call__(self, logit, label=None, update=True):
        if update and label is not None:
            return self.update(logit, label)
        return self._forward(logit, label)

if __name__ == '__main__':
    res = bin_search_sup(lambda v: v <= 0.3421, np.linspace(0, 1, 1000))