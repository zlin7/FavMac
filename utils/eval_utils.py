import datetime
import os
from typing import Dict, Tuple, Union

import ipdb
import numpy as np
import pandas as pd
import tqdm
from scipy import stats
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)


#===================================================================================================
def create_printable_ser(summ, formatter=lambda m, s: f'{m:.2f}$\pm${s:.2f}', scale=1, pval=0.05):
    ret = {}
    mask = {}
    best_method = summ['rank'].idxmax()
    for method in summ.index:
        assert isinstance(method, str)
        ret[method] = formatter(summ.loc[method, 'mean'] * scale, summ.loc[method, 'std'] * scale)
        mask[method] = 1 if summ.loc[method, 'pval'] > pval else 0
    return pd.Series(ret).astype(str), pd.Series(mask).astype(int)
def create_printable_df(summs, formatter=lambda m, s: f'{m:.2f}$\pm${s:.2f}', scale:Union[float,int,dict]=1, pval=0.05):
    ret = {}
    mask = {}
    for dataset, summ in summs.items():
        _scale = scale[dataset] if isinstance(scale, dict) else scale
        ret[dataset], mask[dataset] = create_printable_ser(summ, formatter, _scale, pval=pval)
    return pd.DataFrame(ret), pd.DataFrame(mask)
def printable_df_to_latex(df, mask_df, mask_format=lambda s: "\\textbf{%s}"%s, table_name=''):
    # Can repeatedly call this function with different formatter and masks
    _fmt = lambda s, flag: mask_format(s) if flag else s
    assert all([df[c].dtype=='O' for c in df.columns])
    assert df.shape == mask_df.shape
    lines = [[table_name] + list(df.columns)]
    for idx in df.index:
        lines.append([str(idx)] + [_fmt(df.loc[idx, c], mask_df.loc[idx, c]) for c in df.columns])
    max_len = max([max(map(len, _)) for _ in lines])
    for line in lines:
        print(" & ".join([_.rjust(max_len) for _ in line]) + "\\\\")
#===================================================================================================

def summarize_mean_std_pval(values_df, paired=False, higher_better=True, target:float=None, twosided=False) -> pd.DataFrame:
    #df[col] is a bunch of random values to compare
    values_df = values_df.copy()
    if target is not None: values_df -= target
    if not higher_better: values_df = - values_df

    summ = values_df.describe().reindex(['count', 'mean', 'std']).T
    summ['count'] = summ['count'].astype(int)
    summ = summ.sort_values('mean', ascending=False)
    msk = summ['count'] == summ['count'].max()
    summ.loc[summ.index[msk], 'rank'] = summ['mean'][msk].rank()
    best_method = summ['rank'].idxmax()
    for method in summ.index[msk]:
        if target is not None:
            assert not paired
            summ.loc[method, 'pval'] = stats.ttest_1samp(values_df[method], 0, alternative='two-sided' if twosided else 'less').pvalue
        else:
            pval_compute = stats.ttest_rel if paired else stats.ttest_ind
            #print(method, pval_compute(values_df[method], values_df[best_method], alternative='less'))
            summ.loc[method, 'pval'] = pval_compute(values_df[method], values_df[best_method], alternative='two-sided' if twosided else 'less').pvalue
            if pd.isnull(summ.loc[method, 'pval']) and summ.loc[method, 'rank'] == summ['rank'].max():
                summ.loc[method, 'pval'] = 1.
    if not higher_better: summ['mean'] = - summ['mean']
    if target is not None: summ['mean'] += target
    return summ

def evaluate_debug(obj, logits, labels, cost_fn, value_fn):
    obj.init_calibrate(logits, labels)

    ret = {"thres": [obj.t],  'cost': [None], 'value': [None], '|S|': [None]}
    for i, (_logit, _label) in tqdm.tqdm(enumerate(zip(logits, labels)), desc='evaluating', total=len(logits)):
        predset, _ = obj(_logit, _label, update=False)
        ret['thres'].append(obj.t)
        ret['cost'].append(cost_fn(predset, _label))
        ret['value'].append(value_fn(predset, _label))
        ret['|S|'].append(predset.sum())
    assert max(ret['cost'][1:]) <= 1 and max(ret['value'][1:]) <= 1
    return ret

def evaluate_online(obj, logits, labels, cost_fn, value_fn,
                    burn_in=200, update=True, _debug_time_cache_path=None):
    """
    obj: intialized (but not calibrated) cc.Calibrator
    logits: np.ndarray of shape (n, K)
    labels: np.ndarray of shape (n, K)
    cost_fn: Callable, cost function
    value_fn: Callable, value function
    burn_in: burn_in period (used to calibrate but evaluated)
    """
    obj.init_calibrate(logits[:burn_in], labels[:burn_in])
    import utils.utils as utils
    logger, log_stride = None, None
    start_time = datetime.datetime.now()
    if _debug_time_cache_path is not None:
        logger = utils.get_logger(os.path.basename(_debug_time_cache_path), _debug_time_cache_path)
        log_stride = int(os.path.basename(_debug_time_cache_path).replace(".log", "").split("_")[-1])
        logger.info(f"|0: {start_time.strftime('%Y%m%d-%H%M%S-%f')}|")

    ret = {"thres": [obj.t],  'cost': [None], 'value': [None], '|S|': [None]}
    for i, (_logit, _label) in tqdm.tqdm(enumerate(zip(logits[burn_in:], labels[burn_in:])), desc='evaluating', total=len(logits)-burn_in):
        predset, _ = obj(_logit, _label, update=update)
        ret['thres'].append(obj.t)
        ret['cost'].append(cost_fn(predset, _label))
        ret['value'].append(value_fn(predset, _label))
        ret['|S|'].append(predset.sum())
        if log_stride is not None and (i+1) % log_stride == 0:
            logger.info(f"|{i+1}: {datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}|")
    if log_stride is not None: logger.info(f"|{i+1}: {datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}|")
    assert max(ret['cost'][1:]) <= 1 and max(ret['value'][1:]) <= 1
    return ret

def _compute_precision_recall(
        gt_label: np.ndarray,
        prediction: np.ndarray,
) -> Tuple[np.array, np.array, np.array, float, Dict[str, np.array]]:

    prediction = prediction.squeeze()

    prec, recall, thresholds = precision_recall_curve(gt_label, prediction)

    # Get best threshold and key metrics.
    f1_score = 2 * (prec * recall) / (prec + recall + 1e-9)
    best_idx = f1_score.argmax()
    best_threshold = thresholds[best_idx]

    return prec, recall, thresholds, best_threshold#, average_precision_score(gt_label, prediction, average=)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def multilabel_eval(gt_label: np.ndarray, logits: np.ndarray, flat=False):
    assert gt_label.shape == logits.shape and len(logits.shape) == 2
    ret = pd.DataFrame(columns=['precision', 'recall', 'F1', 'cnt', 'AP', 'thres'], index=np.arange(gt_label.shape[1]))
    prediction = sigmoid(logits)
    gt_label = gt_label.astype(int)

    keep_classes = np.arange(gt_label.shape[1])[gt_label.sum(0) > 0]
    bin_preds = np.zeros_like(gt_label, dtype=np.bool_)
    #for k in range(gt_label.shape[1]):
    for k in keep_classes:
        _precs, _recalls, _, ret.loc[k, 'thres'] = _compute_precision_recall(gt_label[:, k], prediction[:, k])
        bin_preds[:, k] = (prediction[:, k] >= ret.loc[k, 'thres'])
    bin_preds = bin_preds.astype(int)
    ret['precision'], ret['recall'], ret['F1'], ret['cnt'] = precision_recall_fscore_support(gt_label, bin_preds, labels=np.arange(gt_label.shape[1]))
    ret.loc[keep_classes, 'AP'] = average_precision_score(gt_label[:, keep_classes], prediction[:, keep_classes], average=None)
    for k in keep_classes:
        ret.loc[k, 'AUROC'] = roc_auc_score(gt_label[:, k], prediction[:, k])
    ret = ret.fillna(0)
    mAP = pd.Series({"weighted": np.average(ret['AP'], weights=ret['cnt']), 'macro': ret.loc[keep_classes, 'AP'].mean()})
    F1s = pd.Series({"weighted": np.average(ret['F1'], weights=ret['cnt']), 'macro': ret.loc[keep_classes, 'F1'].mean()})
    AUROCs = pd.Series({"weighted": np.average(ret['AUROC'], weights=ret['cnt']), 'macro': ret.loc[keep_classes, 'AUROC'].mean()})
    summs = {"mAP": mAP, "F1": F1s, 'AUROC': AUROCs}
    if flat:
        summs = pd.Series({f"{prefix}_{metric}": val for metric, _ser in summs.items() for prefix, val in _ser.items()})
    return ret, summs

def singlelabel_eval(gt_label: np.ndarray, pred: np.ndarray, n_class:int, flat=False):
    from sklearn.metrics import classification_report
    assert len(pred.shape) == 2
    pred = np.argmax(pred, 1)
    tmp_report = classification_report(gt_label, pred, output_dict=True)
    label_list = [i for i in range(n_class)]
    tmp_report = [tmp_report[str(i)]['f1-score'] if str(i) in tmp_report else np.NaN for i in label_list]
    f1 = np.mean(tmp_report)
    acc = sum(gt_label == pred) / float(len(gt_label))
    return tmp_report, {"F1": f1, "Acc": acc}

def regression_eval(gt_label: np.ndarray, pred: np.ndarray):
    if len(pred.shape) == len(gt_label.shape) + 1 and pred.shape[-1] == 1: pred = pred[..., 0]
    return {'MSE': np.mean(np.square((pred - gt_label)))}

def merge_mean_std_tables(mean_df, std_df, prec1=4, prec2=4):
    format_ = "{:.%df}({:.%df})"%(prec1, prec2)
    if isinstance(mean_df, pd.DataFrame):
        ndf=  pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
        for c in mean_df.columns:
            ndf[c] = merge_mean_std_tables(mean_df[c], std_df[c], prec1, prec2)
        return ndf
    nser = pd.Series("", index=mean_df.index)
    for i in nser.index:
        nser[i] = format_.format(mean_df[i], std_df[i])
    return nser

if __name__ == '__main__':
    res = singlelabel_eval(np.asarray([0,1,2]), np.asarray([0,1,0]), n_class=3, flat=True)