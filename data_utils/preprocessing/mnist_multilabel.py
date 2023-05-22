
import numpy as np
import pandas as pd
import persist_to_disk as ptd
import torch
import torchvision
import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

import _settings
from data_utils.common import (SEED_OFFSETS, TEST, TRAIN, TRAIN1, VALID,
                               VALID1, VALIDTEST, DatasetWrapper,
                               get_split_indices)


def _get_transforms(
    normalize: torchvision.transforms.Normalize=None,
    ):
    all_transforms = [torchvision.transforms.ToTensor()]
    if normalize is not None:
        all_transforms.append(normalize)
    return transforms.Compose(all_transforms)

class MNISTData(Dataset):
    def __init__(self, split=TRAIN, seed=_settings.RANDOM_SEED):
        super().__init__()
        self._seed = seed
        self._data = torchvision.datasets.MNIST(_settings.MNIST_PATH, train=split !=TEST, download=True, transform=_get_transforms())
        if split == TEST:
            self.indices = np.arange(len(self._data))
        else:
            self.indices = sorted(get_split_indices(seed, [0.9, 0.1], n=len(self._data))[split])
        self.n = len(self.indices)
        self._labels = pd.Series(self._data.targets).reindex(self.indices)
        self.split = split

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        idx = self.indices[idx]
        x, y = self._data[idx]
        if hasattr(self, '_labels'):
            assert y == self._labels[idx]
        return {"data": x, "target": y, "index":  (0 if self.split != TEST else 9900000) + idx}


class SuperimposeSyntheticData(DatasetWrapper):
    DATASET = _settings.MNISTSup_NAME
    LABEL_MAP = dict(zip(range(10), range(10)))
    def __init__(self, split=TRAIN, sample_proba=0.4,
                 niters_per_epoch=None, imagesize=48, noise_level=None,
                 seed=_settings.RANDOM_SEED,
                 debug=False,
                 ):
        super().__init__(split)
        assert split != VALIDTEST, "valtest should be test for MNIST"
        self.basedata = MNISTData(split, seed)
        self.indices_by_class = dict(self.basedata._labels.reset_index().reset_index().groupby(0)['level_0'].agg(list))

        if niters_per_epoch is None: niters_per_epoch = len(self.basedata)
        self.niters_per_epoch = niters_per_epoch

        key = f'{self.DATASET}-{self.split}-{sample_proba}-{imagesize}-{seed}-{self.niters_per_epoch}'
        self.noise_level = noise_level
        if noise_level is not None:
            key = key + f'-N{noise_level}'

        self._data_shape = (3, imagesize, imagesize)

        self.pad = imagesize - 28
        self.random_states = np.random.RandomState(seed + SEED_OFFSETS[split]).randint(0, 393241, self.niters_per_epoch)
        self.labels = (np.random.RandomState(seed + SEED_OFFSETS[split]).uniform(0,1, (niters_per_epoch, len(self.LABEL_MAP))) < sample_proba).astype(int)

        if debug:
            self.all_data = None
            return

        self.all_data = ptd.manual_cache(key)
        if self.all_data is None:
            print(key)
            self.all_data = [self._get_item_cache(_) for _ in tqdm.tqdm(range(self.niters_per_epoch))]
            ptd.manual_cache(key, self.all_data, write=True)

    def __len__(self):
        return self.niters_per_epoch

    def _get_item_cache(self, idx):
        rs = np.random.RandomState(self.random_states[idx])
        y = self.labels[idx]
        x = torch.zeros(self._data_shape)
        indices = []
        for k, present in enumerate(y):
            if present == 0: continue

            channel = rs.randint(0, self._data_shape[0])
            st_w, st_h = rs.randint(0, self.pad+1, size=2)
            res = self.basedata[rs.choice(self.indices_by_class[k])]

            x[channel, st_w:st_w+28, st_h:st_h+28] += res['data'][0]
            if self.noise_level is not None:
                x[channel, st_w:st_w+28, st_h:st_h+28] += torch.tensor(rs.normal(size=(28, 28))) * self.noise_level
            indices.append(res['index'])
        return x, y, "|".join(map(str, indices))

    def __getitem__(self, idx):
        if self.all_data is not None:
            x, y, index = self.all_data[idx]
        else:
            x, y, index = self._get_item_cache(idx)
        return {"data": x, "target": y.astype(np.float32), "index": index}

def cache_(*args, **kwargs):
    return SuperimposeSyntheticData(*args, **kwargs)
if __name__ == '__main__':
    import utils.utils as utils
    taskrunner = utils.TaskPartitioner()
    for split in [TRAIN, VALID, TRAIN1, VALID1]:
        taskrunner.add_task(cache_, split, sample_proba=0.4, noise_level=None)
    taskrunner.run_multi_process(1)
