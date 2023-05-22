import bisect
import math

import ipdb
import numpy as np
import pandas as pd

from quantiletree.rbwbst import QuantileTree


class NaiveSlowWeightedQuery:
    def __init__(self, ):
        self.mem = pd.Series()
        self.sorted = True

    def insert(self, val, weight=1):
        if val in self.mem:
            self.mem[val] += weight
        else:
            self.mem.loc[val] = weight
        self.sorted = False

    def delete(self, val, weight=1):
        assert val in self.mem
        if self.mem[val] < weight:
            raise ValueError()
        if self.mem[val] == weight:
            self.mem.pop(val)
        else:
            self.mem[val] -= weight
        self.sorted = False

    def query_cumu_weight(self, w, prev=True):
        if not self.sorted:
            self.mem = self.mem.sort_index()
            self.sorted = True
        cumusum = self.mem.cumsum()
        idx = bisect.bisect(cumusum.values, w)
        if prev:
            if idx == 0: return -math.inf
            return self.mem.index[idx - 1]
        else:
            return self.mem.index[min(len(self.mem)-1, idx)]

class Tester:
    def __init__(self) -> None:
        self.ref = NaiveSlowWeightedQuery()
        self.imp = QuantileTree()
    def insert(self, val, weight=1):
        self.ref.insert(val, weight)
        self.imp.insert(val, weight)
    def delete(self, val, weight=1):
        self.ref.delete(val, weight)
        self.imp.delete(val, weight)
    def query_cumu_weight(self, w, prev=True):
        ansref = self.ref.query_cumu_weight(w, prev=prev)
        ans = self.imp.query_cumu_weight(w, prev=prev)
        assert ans == ansref
        return ans


def test_random(seed=7):
    np.random.seed(seed)

    o = Tester()
    records = []
    s = 0
    for _ in range(1000):
        p = np.random.uniform(0, 1)
        if p < 0.8 or len(records) < 10:
            t = np.random.randint(0, 100)
            c = int(np.random.randint(1, 101))
            o.insert(t, c)
            records.append((t,c))
            s += c
        else:
            idx = np.random.choice(len(records), 1)[0]
            t, c = records[idx]
            records = records[:idx] + records[idx+1:]
            o.delete(t, c)
            s -= c
        # query
        for _i in range(10):
            w = np.random.uniform(0, s + 5)
            o.query_cumu_weight(w)
            o.query_cumu_weight(w, prev=False)

if __name__ == '__main__':
    o = NaiveSlowWeightedQuery()
    o.insert(0.1, 1)
    o.insert(0.2, 1)
    o.insert(0.4, 1)
    o.insert(0.5, 4)
    # o.query_cumu_weight()
    test_random()