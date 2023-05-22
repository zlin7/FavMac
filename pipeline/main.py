from collections import defaultdict

import numpy as np
import pandas as pd
import persist_to_disk as ptd
import torch
import tqdm
from torch.utils.data import DataLoader

import _settings
import data_utils as data_utils
import pipeline.trainer as trainer
import utils.utils as utils


def infer_nclass(df):
    for i in range(len(df.columns)):
        if 'Y%d'%i not in df.columns: return i
    return i + 1

@ptd.persistf(expand_dict_kwargs='all', skip_kwargs=['device'])
def _read_prediction(key, dataset, split, datakwargs,
                         device='cuda:0', **kwargs):
    import pipeline.trainer as tr
    mode = kwargs.pop('mode', 'last')
    model, settings, _ = tr.CallBack.load_state(None, key, mode=mode, device=device)
    model.eval()

    dataset = data_utils.get_default_dataset(dataset, split=split, **datakwargs)
    ret = defaultdict(list)
    def _transform(x):
        if isinstance(x, torch.Tensor):
            if len(x.size()) == 0: return x.item()
            return x.cpu().numpy()
        return x
    with torch.no_grad():
        for all_input_dict in tqdm.tqdm(DataLoader(dataset, batch_size=1, shuffle=False)):
            all_input_dict = tr._to_device(all_input_dict, device)
            out = model(all_input_dict['data'])
            ret['logit'].append(_transform(out[0]))
            for k, v in all_input_dict.items():
                if k != 'data': ret[k].append(_transform(v[0]))
    ret = {k:np.stack(v) if isinstance(v, np.ndarray) else np.asarray(v) for k, v in ret.items()}
    return ret

def read_prediction(key, dataset, split, datakwargs, device='cuda:0', **kwargs):
    from scipy.special import expit
    res = _read_prediction(key, dataset, split, datakwargs, device=device, **kwargs)
    S = res['logit']
    Y = res['target']
    P = expit(S)
    preds = pd.DataFrame(np.concatenate([P, S, Y], 1), columns=[f'{_}{k}' for _ in ['P', 'S', 'Y'] for k in range(S.shape[1])], index=res['index'])
    if dataset == _settings.MNISTSup_NAME:  preds = preds[~preds.index.duplicated(keep='first')]
    if 'target_date' in res.keys():
        preds['target_date'] = res['target_date']
    return preds

@ptd.persistf(expand_dict_kwargs='all', skip_kwargs=['gpu_id', 'debug'], groupby=['dataset'], switch_kwarg='cache')
def _train(dataset, datakwargs={},  **kwargs):
    seed = kwargs.get('seed', _settings.RANDOM_SEED)
    utils.set_all_seeds(seed)
    debug = kwargs.pop('debug', False)
    train_split = kwargs.pop('train_split', data_utils.TRAIN)
    val_split = kwargs.pop('val_split', data_utils.VALID)
    train_data = data_utils.get_default_dataset(dataset, split=train_split, seed=seed, **datakwargs)
    val_datakwargs = kwargs.pop('val_datakwargs', datakwargs.copy())
    print("val_datakwargs", val_datakwargs)
    if not kwargs.pop('skip_val', False):
        val_data = data_utils.get_default_dataset(dataset, split=val_split, seed=seed, **val_datakwargs)
    else:
        val_data = None
    short_desc = kwargs.get('short_desc', '')
    print(short_desc)

    assert 'model_kwargs' in kwargs
    assert 'nclass' not in kwargs['model_kwargs']
    kwargs['model_kwargs'] = kwargs['model_kwargs'].copy()
    kwargs['model_kwargs'].setdefault('nclass', len(train_data.LABEL_MAP))

    cb = trainer.CallBack(train_data, val_data, debug=debug, **kwargs)
    key = cb.train(num_workers=4)
    print(key)
    return key



@ptd.persistf(expand_dict_kwargs='all', groupby=[('dataset', 'value_fn', 'cost_fn'), 'perm_seed'])
def _evaluate_main_cache(
    calib_cls, target_cost, delta,
    base_keymode, cost_fn, value_fn, proxy_fn,
    dataset, datakwargs,
    perm_seed=None, split=data_utils.VALIDTEST,
    burn_in=1000,
    calib_cls_kwargs=None,
    **kwargs
    ):
    import utils.eval_utils as eutils
    cost_fn = data_utils.get_set_fn(cost_fn)
    value_fn = data_utils.get_set_fn(value_fn, **kwargs.pop('value_fn_kwargs', {}))
    proxy_fn = data_utils.get_set_fn(proxy_fn, **kwargs.pop('proxy_fn_kwargs', {}))

    if calib_cls_kwargs is None: calib_cls_kwargs = {}
    base_key, mode = base_keymode.split("|")

    if '|cap=' in dataset:
        dataset, debug_max_n = dataset.split("|cap=")
        debug_max_n = int(debug_max_n)
    else:
        debug_max_n = kwargs.pop('debug_max_n', None)

    predsdf = read_prediction(base_key, dataset, split=split,  datakwargs=datakwargs, mode=mode)
    calib_obj = calib_cls(cost_fn, value_fn, proxy_fn, target_cost=target_cost, delta=delta, **calib_cls_kwargs)
    nclass = infer_nclass(predsdf)
    logit = predsdf.reindex(columns=['S%d'%i for i in range(nclass)])
    label = predsdf.reindex(columns=['Y%d'%i for i in range(nclass)])

    # perm_seed is used when we want to re-permute the test set and do not want to re-train the base DNN (which is slow)
    if perm_seed is not None:
        _perm_idx = logit.index[np.random.RandomState(perm_seed).permutation(len(logit))]
        logit, label = logit.reindex(_perm_idx), label.reindex(_perm_idx)

    if debug_max_n is not None:
        logit, label = logit.iloc[:debug_max_n], label.iloc[:debug_max_n]
    if burn_in is None: return eutils.evaluate_debug(calib_obj, logit.values, label.values, cost_fn=cost_fn, value_fn=value_fn)
    _debug_time_cache_path = kwargs.pop('_debug_time_cache_path', None)
    return eutils.evaluate_online(calib_obj, logit.values, label.values, cost_fn=cost_fn, value_fn=value_fn, burn_in=burn_in, update=True, _debug_time_cache_path=_debug_time_cache_path)
