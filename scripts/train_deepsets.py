import torch

import _settings
import models
import pipeline.main as pm
import pipeline.trainer as trainer
import utils.utils as utils
from loss_functions import get_criterion


def train_ds_fp(keymode, dataset=_settings.MIMICIIICompletion_NAME, datakwargs=None, set_fn='sf_FP_cost',
			 n_epochs=20, batch_size=32, eval_steps=None,
			 train_iters=50000, val_iters=5000,
			 lr=1e-3,
			 gpu_id=1,
			 task_type = trainer.CallBack.TASK_CLASS,
			 **kwargs):
	if datakwargs is None: datakwargs = {}
	datakwargs = {"keymode": keymode, 'datakwargs': datakwargs}
	datakwargs['niters'] = train_iters
	key = pm._train(f"LDSFP-{dataset}", datakwargs, train_split='val1', val_split='val',
				n_epochs=n_epochs, batch_size=batch_size, eval_steps=eval_steps,
				model_class = models.DeepSets, model_kwargs={},
				criterion_class=torch.nn.CrossEntropyLoss, criterion_kwargs={},
				optimizer_class=torch.optim.Adam, optimizer_kwargs={"lr": lr},
				task_type=task_type, gpu_id=gpu_id,
				val_datakwargs = utils.merge_dict_inline(datakwargs, {'niters': val_iters}),
								**kwargs)
	print(key)


def train_quantile(keymode, dataset=_settings.MIMICIIICompletion_NAME, datakwargs=None,
					n_epochs=20, batch_size=64, eval_steps=None,
					model_class = models.TrainedThreshold,
					lr=1e-3,
					gpu_id=1,
					task_type = trainer.CallBack.TASK_UNKNOWN,
					**kwargs):
	if datakwargs is None: datakwargs = {}
	datakwargs = {"keymode": keymode, 'datakwargs': datakwargs}
	key = pm._train(f"LOGIT-{dataset}", datakwargs, train_split='val1', val_split='val',
				n_epochs=n_epochs, batch_size=batch_size, eval_steps=eval_steps,
				model_class = model_class, model_kwargs={},
				criterion_class=get_criterion('PinBall'), criterion_kwargs={'q': 0.1},
				optimizer_class=torch.optim.Adam, optimizer_kwargs={"lr": lr},
				task_type=task_type, gpu_id=gpu_id,
				val_datakwargs = utils.merge_dict_inline(datakwargs, {}),
								**kwargs)
	print(key)



class MIMIC_TrainConfigs:
	@classmethod
	def train_ds_few(cls, debug=False, **kwargs):
		from scripts.train_mimic3hcc import _Configs
		key = _Configs.train_few_train1split()[0]
		#key = 'EHRModel-MIMIC-IIICompletion-20230521_181249342352'
		kwargs.update({'datakwargs': {"use_notes": True, 'hcc_choice': 'few'}})
		train_ds_fp(f"{key}|last", debug=debug, **kwargs)
		return
		for _ in ['train_iters', 'val_iters', 'short_desc', 'model_class']: kwargs.pop(_)
		train_quantile(f"{key}|last", debug=debug, **kwargs)

	@classmethod
	def train_ds_more(cls, debug=False, **kwargs):
		from scripts.train_mimic3hcc import _Configs
		key = _Configs.train_more_train1split()[0]
		#key = 'EHRModel-MIMIC-IIICompletion-20230521_161730874913'
		kwargs.update({'datakwargs': {"use_notes": True, 'hcc_choice': 'more'}})
		train_ds_fp(f"{key}|last", debug=debug, **kwargs)
		return
		for _ in ['train_iters', 'val_iters', 'short_desc', 'model_class']: kwargs.pop(_)
		train_quantile(f"{key}|last", debug=debug, **kwargs)

class Claim_TrainConfigs:
	@classmethod
	def train_ds_few(cls, debug=False, **kwargs):
		from scripts.train_claim import _Configs
		key = 'ClaimModel-ClaimPredNew-20230109_194413008551'
		kwargs.update({"dataset": _settings.CLAIMDEMOSeq_NAME})
		kwargs['datakwargs'] = {'hcc_choice': 'few', 'seq': True, 'topndiags': None}
		train_ds_fp(f"{key}|loss", debug=debug, **kwargs)
		return
		for _ in ['train_iters', 'val_iters', 'short_desc', 'model_class']: kwargs.pop(_, None)
		train_quantile(f"{key}|loss", debug=debug, **kwargs)

	@classmethod
	def train_ds_more(cls, debug=False, **kwargs):
		from scripts.train_claim import _Configs
		key = 'ClaimModel-ClaimPredNew-20230109_194413510553'
		kwargs.update({"dataset": _settings.CLAIMDEMOSeq_NAME})
		kwargs['datakwargs'] = {'hcc_choice': 'more', 'seq': True, 'topndiags': None}
		train_ds_fp(f"{key}|loss", debug=debug, **kwargs)
		return

		for _ in ['train_iters', 'val_iters', 'short_desc', 'model_class']: kwargs.pop(_, None)
		train_quantile(f"{key}|loss", debug=debug, **kwargs)

	@classmethod
	def train_ds_few_random(cls, debug=False, **kwargs):
		from scripts.train_claim import _Configs
		key = 'ClaimModel-ClaimPredNew-20230109_194414393208'
		kwargs.update({"dataset": _settings.CLAIMDEMO_NAME})
		kwargs['datakwargs'] = {'hcc_choice': 'few', 'seq': True, 'topndiags': None}
		train_ds_fp(f"{key}|loss", debug=debug, **kwargs)
		return

		for _ in ['train_iters', 'val_iters', 'short_desc', 'model_class']: kwargs.pop(_, None)
		train_quantile(f"{key}|loss", debug=debug, **kwargs)

	@classmethod
	def train_ds_more_random(cls, debug=False, **kwargs):
		from scripts.train_claim import _Configs
		key = 'ClaimModel-ClaimPredNew-20230109_194414301167'
		kwargs.update({"dataset": _settings.CLAIMDEMO_NAME})
		kwargs['datakwargs'] = {'hcc_choice': 'more', 'seq': True, 'topndiags': None}
		train_ds_fp(f"{key}|loss", debug=debug, **kwargs)
		return

		for _ in ['train_iters', 'val_iters', 'short_desc', 'model_class']: kwargs.pop(_, None)
		train_quantile(f"{key}|loss", debug=debug, **kwargs)


class MNIST_TrainConfigs:
	@classmethod
	def train_mnist(cls, debug=False, **kwargs):
		from scripts.train_mnist import _Configs
		key = _Configs.train(debug=False).run()[1][0]
		kwargs.update({"dataset": _settings.MNISTSup_NAME})
		kwargs['datakwargs'] = {'sample_proba': 0.4, 'noise_level': None}
		kwargs['train_iters'] = 100000
		train_ds_fp(f"{key}|loss", debug=debug, **kwargs)
		return

		for _ in ['train_iters', 'val_iters', 'short_desc', 'model_class']: kwargs.pop(_, None)
		train_quantile(f"{key}|loss", debug=debug, **kwargs)


if __name__ == '__main__':
	MNIST_TrainConfigs.train_mnist(debug=False)
	MIMIC_TrainConfigs.train_ds_few(debug=False)
	MIMIC_TrainConfigs.train_ds_more(debug=False)
	if False:
		MIMIC_TrainConfigs.train_ds_few(debug=False)
		MIMIC_TrainConfigs.train_ds_more(debug=False)
		MNIST_TrainConfigs.train_mnist(debug=False)
	if False:
		Claim_TrainConfigs.train_ds_few(debug=False)
		Claim_TrainConfigs.train_ds_more(debug=False)
		Claim_TrainConfigs.train_ds_few_random(debug=False)
		Claim_TrainConfigs.train_ds_more_random(debug=False)
