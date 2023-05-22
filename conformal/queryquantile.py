import numpy as np

import quantiletree


class BaseQuery:
    def __init__(self) -> None:
        pass

    def add(self, thres, gap=1):
        raise NotImplementedError()

    def delete(self, thres, gap=1):
        raise NotImplementedError()

    def query(self, cumu):
        raise NotImplementedError()



class QuantileTreeWrap:
    def __init__(self) -> None:
        self.tree = quantiletree.QuantileTree()

    def add(self, thres, gap=1):
        self.tree.insert(thres, gap)

    def delete(self, thres, gap=1):
        self.tree.delete(thres, gap)

    def query(self, cumu, inclusive=False):
        # inclusive: whether the prediction set is proxy/t_k_i < t or proxy <= t. "<=" is inclusive
        return self.tree.query_cumu_weight(cumu, prev=inclusive)

class SortedQuantile:
    def __init__(self) -> None:
        from sortedcontainers import SortedList
        self.sl = SortedList()

    def add(self, thres, gap=1):
        assert gap == 1
        self.sl.add(thres)

    def query(self, cumu, normalized=False, **kwargs):
        #if normalized: cumu = int(np.ceil(cumu/len(self.sl)))
        return self.sl[cumu]



if __name__ == '__main__':
    o = SortedQuantile()

