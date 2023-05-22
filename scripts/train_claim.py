from importlib import reload

import pandas as pd
import torch

import _settings
import data_utils
import models
import pipeline.main as pm
import pipeline.trainer as trainer
import utils.eval_utils as eval_utils
import utils.utils as utils
from loss_functions import get_criterion


def train(
	dataset=_settings.CLAIMDEMOSeq_NAME,
	lr=1e-3, n_epochs=20, batch_size=128, continue_from_key=None,
	train_split=data_utils.TRAIN, val_split=data_utils.VALID, eval_steps=500,
	model_class = models.ClaimModel, model_kwargs=None, gpu_id=0,
	optimizer_class=torch.optim.Adam,
	datakwargs = None,
	criterion='BCE', criterion_kwargs=None,
	#eval and cache embeddings
	eval_datakwargs=None,
	**kwargs):
	"""
	This function trains the base DNN.
	It will return the `key` of the trained model, which can be used to retrieve the model.
	"""
	kwargs.setdefault('short_desc', '')
	if datakwargs is None: datakwargs = {}
	if eval_datakwargs is None: eval_datakwargs = datakwargs.copy()
	if model_kwargs is None: model_kwargs = {}
	if criterion_kwargs is None: criterion_kwargs = {}

	eval_metric = {"loss": "min", "macro_mAP": "max", "weighted_mAP": "max", "weighted_F1": "max", "macro_F1": "max",
				   'weighted_AUROC': "max", 'macro_AUROC': 'max'}

	scheduler_kwargs = {"mode": "min", "factor": 0.5, "patience": 4}
	key = pm._train(dataset, datakwargs,
							   train_split=train_split, val_split=val_split,
							   continue_from_key=continue_from_key,
							   n_epochs=n_epochs, batch_size=batch_size, eval_steps=eval_steps,
							   model_class = model_class, model_kwargs=model_kwargs,
								criterion_class=get_criterion(criterion), criterion_kwargs=criterion_kwargs,
							   optimizer_class=optimizer_class,
							   optimizer_kwargs={"lr": lr},
							   scheduler_kwargs=scheduler_kwargs,
							   task_type=trainer.CallBack.TASK_MULTILABLCLASS, gpu_id=gpu_id,
							   eval_metric=eval_metric,
					val_datakwargs = utils.merge_dict_inline(datakwargs, eval_datakwargs),
					**kwargs
							   )

	eval_res = {}
	if eval_datakwargs is not None:
		if isinstance(gpu_id, tuple) or isinstance(gpu_id, list):
			gpu_id = gpu_id[0]
		for split in [val_split, 'val1', data_utils.VALIDTEST]:
			for mode in ['last', 'loss']:
				eval_kwargs = {"mode": mode}
				preds = pm.read_prediction(key, dataset, split=split, datakwargs=eval_datakwargs, device=utils.gpuid_to_device(gpu_id), **eval_kwargs)
				n_class = pm.infer_nclass(preds)
				tempres = eval_utils.multilabel_eval(preds.reindex(columns=['Y%d'%i for i in range(n_class)]).values,
													preds.reindex(columns=['S%d'%i for i in range(n_class)]).values)
				eval_res[f"{mode}-{split}"] = pd.DataFrame(tempres[1])
	print(eval_res)
	print(key)
	return key, eval_res

class _Configs:
	@classmethod
	def train_few(cls, task_runner:utils.TaskPartitioner = None, debug=False, gpu_id=(0, 1)):
		if task_runner is None: task_runner = utils.TaskPartitioner()
		for dataset in [_settings.CLAIMDEMOSeq_NAME, _settings.CLAIMDEMO_NAME]:
			lr = 1e-5 # 1e-3
			kwargs = {"datakwargs": {'hcc_choice': 'few', 'seq': True, 'topndiags': None}, 'train_split': data_utils.TRAIN}
			kwargs['model_kwargs'] = {'diag_embed_dim': 128, 'hidden_dim': 128}
			task_runner.add_task(train, dataset, debug=debug, gpu_id=gpu_id, batch_size=128, lr=lr, **kwargs)

			kwargs['train_split'] = data_utils.TRAIN1
			task_runner.add_task(train, dataset, debug=debug, gpu_id=gpu_id, batch_size=128, lr=lr, **kwargs)

		return task_runner


	@classmethod
	def train_more(cls, task_runner:utils.TaskPartitioner = None, debug=False, gpu_id=(0, 1)):
		if task_runner is None: task_runner = utils.TaskPartitioner()
		for dataset in [_settings.CLAIMDEMOSeq_NAME, _settings.CLAIMDEMO_NAME]:
			lr = 1e-5 # 1e-3
			kwargs = {"datakwargs": {'hcc_choice': 'more', 'seq': True, 'topndiags': None}, 'train_split': data_utils.TRAIN}
			kwargs['model_kwargs'] = {'diag_embed_dim': 128, 'hidden_dim': 128}
			task_runner.add_task(train, dataset, debug=debug, gpu_id=gpu_id, batch_size=128, lr=lr, **kwargs)

			kwargs['train_split'] = data_utils.TRAIN1
			task_runner.add_task(train, dataset, debug=debug, gpu_id=gpu_id, batch_size=128, lr=lr, **kwargs)

		return task_runner

if __name__ == '__main__':
	res = _Configs.train_few(debug=False).run()