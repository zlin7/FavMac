import datetime
import os
import shutil
from collections import defaultdict
from typing import Any, Tuple, Union

import ipdb
import numpy as np
# import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import _settings
import utils.utils as utils
from utils import eval_utils


def _to_device(data_or_model, device):
    if isinstance(device, tuple) or isinstance(device, list):
        device = device[0]

    if isinstance(data_or_model, dict):
        return {k: _to_device(v, device) if k !='index' else v for k, v in data_or_model.items()}
    if isinstance(data_or_model, tuple) or isinstance(data_or_model, list):
        return tuple([_to_device(x, device) for x in data_or_model])
    if isinstance(data_or_model, str): return data_or_model
    try:
        return data_or_model.to(device)
    except:
        pass
    return data_or_model

def _state_dict(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()

def move_tfboard_output(src, dst):
    for fname in os.listdir(src):
        if fname.startswith("events.out.tfevents"):
            shutil.copyfile(os.path.join(src, fname),
                            os.path.join(dst, fname))

#=======================================================================================================================
class CallBack(object):
    TASK_CLASS = 'classification'
    TASK_MULTILABLCLASS = 'multilabel-classification'
    TASK_METRIC = 'metric-learning'
    TASK_REGRESSION = 'regression'
    TASK_UNKNOWN = 'unknown'

    _BASE_SAVE_DIR = 'trainer'
    def __init__(self,
                 train_data, valid_data,
                 model_class=None, model_kwargs=None,
                 continue_from_key = None, load_model_only=False,
                 n_epochs=10, batch_size=256, eval_steps=None,
                 gpu_id=0,
                 task_type=TASK_CLASS,
                 short_desc='',
                 debug=False,
                 save_all_ckpt=False,
                 **kwargs,
                 ):
        #NOTE: If we first train 10 epochs and then another 10, the result might be differet from 20 straight due to random seed?
        device = utils.gpuid_to_device(gpu_id)
        name = f'trainer-{train_data.DATASET}'
        if model_class is None or model_kwargs is None:
            assert continue_from_key is not None, "Either pass in a model or continue from a previous training session"

        default_settings = {"last_checkpoint": None, 'task_type': task_type,
                                "model_class": model_class, "model_kwargs": model_kwargs,

                                "criterion_class": {self.TASK_CLASS: torch.nn.CrossEntropyLoss,
                                                    self.TASK_MULTILABLCLASS: torch.nn.BCEWithLogitsLoss}.get(task_type,  torch.nn.MSELoss),
                                "criterion_kwargs": {},

                                "optimizer_class": torch.optim.AdamW,
                                "optimizer_kwargs": {"lr": 1e-3},
                                "scheduler_class": torch.optim.lr_scheduler.ReduceLROnPlateau,
                                "scheduler_kwargs": {"mode": "min", "factor": 0.5, "patience": n_epochs // 20},

                                "eval_metric": {"loss": 'min'}, #used to pick best checkpoint. Assume this is on validation set..
                                }
        if continue_from_key is not None:
            model, old_settings, checkpoint = self.load_state(None, continue_from_key, 'last')
            self.best_val_metrics = checkpoint['best_val_metrics']
            self.epoch, self.step = checkpoint['epoch'], checkpoint['step']

            if load_model_only:
                settings = kwargs.copy()
                for _k in ['model_class', 'model_kwargs']:
                    settings[_k] = old_settings[_k]
                for _setting_key, _setting_val in default_settings.items():
                    settings.setdefault(_setting_key, _setting_val)
            else:
                settings = old_settings
            settings['from_key'] = continue_from_key
        else:
            model = model_class(**model_kwargs)
            settings = kwargs.copy()
            for _setting_key, _setting_val in default_settings.items():
                settings.setdefault(_setting_key, _setting_val)
            self.best_val_metrics = {_metric: ({'min': np.inf, 'max': -np.inf}[_min_or_max], -1, None) for _metric, _min_or_max in settings['eval_metric'].items()}
            #Each value is (value, step, checkpoint_path) tuple
            self.epoch, self.step = 0, 0

        settings.update({"batch_size": batch_size, "eval_steps": eval_steps, "n_epochs": n_epochs})
        self.criterion = settings['criterion_class'](**settings['criterion_kwargs'])
        self.optimizer = settings['optimizer_class'](model.parameters(), **settings['optimizer_kwargs'])
        if settings['scheduler_class'] is not None:
            settings['scheduler_step_on'] = settings['scheduler_kwargs'].pop('step_on', 'val_loss')
            self.scheduler = settings['scheduler_class'](self.optimizer, **settings['scheduler_kwargs'])
        else:
            self.scheduler = None
        if isinstance(device, tuple):
            self.model = torch.nn.DataParallel(model, device_ids=device).to(device[0])
        else:
            self.model = model
            _to_device(self.model, device)
        self.settings = utils.merge_dict_inline(settings, {'name': name, 'best_val_metrics': self.best_val_metrics})
        for _, _mode in self.settings['eval_metric'].items(): assert _mode in {"max", "min"}
        self.task_type = settings['task_type']

        #For this run
        self.key = "%s-%s-%s%s" % (model.__class__.__name__, train_data.DATASET, datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f"), short_desc)
        self.save_path = os.path.join(_settings.WORKSPACE, self._BASE_SAVE_DIR, self.key)
        if not os.path.isdir(self.save_path): os.makedirs(self.save_path)
        if continue_from_key is not None:
            move_tfboard_output(self.save_path.replace(self.key, continue_from_key),
                                          self.save_path)
            shutil.copyfile(os.path.join(self.save_path.replace(self.key, continue_from_key), 'log.log'),
                            os.path.join(self.save_path, 'log.log'))


        #A logger simultaneously writes to many locations.
        self.full_logger = utils.FullLogger(logger=utils.get_logger(name=name + self.key, log_path=os.path.join(self.save_path, 'log.log')),
                                            neptune_ses = utils.get_neptune_logger(self.key, tag=[train_data.DATASET] + ([] if short_desc == '' else [short_desc]), continue_from_key=continue_from_key, debug=debug),
                                            tbwriter=None if debug else SummaryWriter(log_dir=self.save_path))
        self.full_logger.info("Start training ...")


        self.device = device
        self.train_data, self.valid_data = train_data, (valid_data if len(valid_data) else None)
        self.n_class = len(self.train_data.LABEL_MAP)

        self.save_all_ckpt = save_all_ckpt

    def train(self, num_workers=0, seed=_settings.RANDOM_SEED, drop_last_train_batch=True):
        print(f"Training with {num_workers} workers.")
        utils.set_all_seeds(seed)
        batch_size, eval_steps, n_epochs = self.settings['batch_size'], self.settings['eval_steps'], self.settings['n_epochs']
        collate_fn = self.train_data._collate_func if hasattr(self.train_data, '_collate_func') else None
        train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, drop_last=batch_size is not None and drop_last_train_batch)
        if self.valid_data is not None:
            valid_loader = DataLoader(dataset=self.valid_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        model, optimizer, scheduler, criterion = self.model, self.optimizer, self.scheduler, self.criterion
        eval_steps = eval_steps or len(train_loader) #None means every epoch

        self.full_logger[f'parameters/start'] = self.settings.copy()
        self.init_training_history()
        while self.epoch < n_epochs:
            self.epoch += 1
            for all_input_dict in tqdm.tqdm(train_loader, desc='Training Epoch=%d'%self.epoch, ncols=_settings.NCOLS):
                all_input_dict = _to_device(all_input_dict, self.device)
                model.train()

                optimizer.zero_grad()
                output = model(all_input_dict['data'], all_input=all_input_dict)
                # Add any other things the loss function might need to the output as well
                if isinstance(output, dict):
                    output = utils.merge_dict_inline({_[0]: _[1] for _ in all_input_dict.items() if _[0] != 'target'}, output)
                loss = criterion(output, all_input_dict['target'])
                if torch.isnan(loss): ipdb.set_trace()
                if isinstance(loss, torch.Tensor): loss = {'loss': loss}
                loss['loss'].backward()
                optimizer.step()

                self.step += 1
                # bookkeeping
                self.bookkeep_training_step(all_input_dict, output, loss)

                if eval_steps is not None and self.step % eval_steps == 0:
                    self.bookkeep_training_pereval()
                    if self.valid_data is not None: self.on_validation(valid_loader)
                    if self.check_early_stop():
                        self.on_training_end()
                        return self.key
                if not os.path.exists(self.save_path):
                    raise Exception("It seems like the path is deleted. Assuming this means stop!")
        self.on_training_end()
        return self.key

    @classmethod
    def init_history(cls, task_type, prefix=''):
        if task_type == cls.TASK_METRIC:
            ret = {"output": [], 'gt': [], 'loss': defaultdict(list), 'index': []}
        else:
            ret = {"output": [], 'gt': [], 'loss': defaultdict(list), 'index': []}
        return {f"{prefix}{k}": v for k, v in ret.items()}

    @classmethod
    def update_history(cls, history, data, output, loss_dict, task_type, prefix=''):
        extra_output = loss_dict.pop('extra_output', {}) # {"prediction": pred, 'target': target}
        if task_type == cls.TASK_METRIC:
            assert isinstance(output, dict), "This mode should probably have many different types of outputs."
            history[f'{prefix}output'].extend([_ for _ in extra_output['prediction'].detach().cpu().numpy()])
            history[f'{prefix}gt'].extend([_ for _ in data['target'].detach().cpu().numpy()])
            history[f'{prefix}index'].extend([_ for _ in data['index']])
            if loss_dict:
                for k, v in loss_dict.items():
                    history[f'{prefix}loss'][k].append(v.item())
            return history
        if isinstance(output, dict):
            out = output['out']
        elif isinstance(output, tuple) or isinstance(output, list):
            out = output[0] #the rest are extra information
        else:
            out = output
        history[f'{prefix}output'].extend([_ for _ in out.detach().cpu().numpy()])
        history[f'{prefix}gt'].extend([_ for _ in data['target'].detach().cpu().numpy()])
        history[f'{prefix}index'].extend([_ for _ in data['index']])
        if loss_dict:
            for k, v in loss_dict.items():
                history[f'{prefix}loss'][k].append(v.item())
        return history

    def init_training_history(self):
        self.training_history = self.init_history(self.task_type, prefix='train_')

    def bookkeep_training_step(self, data, output, loss_dict):
        self.update_history(self.training_history, data, output, loss_dict, self.task_type, prefix='train_')
        for k, v in loss_dict.items():
            self.full_logger.log_scalar(f'opt/train_{k}', v.item(), self.step)
        return

    def bookkeep_training_pereval(self):
        if self.task_type == self.TASK_CLASS:
            pred = np.asarray(self.training_history['train_output'])
            classwise_res, summs = eval_utils.singlelabel_eval(np.asarray(self.training_history['train_gt']), pred, n_class=self.n_class, flat=True)
        elif self.task_type == self.TASK_MULTILABLCLASS:
            pred = np.asarray(self.training_history['train_output'])
            classwise_res, summs = eval_utils.multilabel_eval(np.asarray(self.training_history['train_gt']), pred, flat=True)
        elif self.task_type == self.TASK_METRIC:
            pred = np.asarray(self.training_history['train_output'])
            classwise_res, summs = eval_utils.singlelabel_eval(np.asarray(self.training_history['train_gt']), pred, n_class=self.n_class, flat=True)
        elif self.task_type == self.TASK_REGRESSION:
            pred = np.asarray(self.training_history['train_output'])
            summs = eval_utils.regression_eval(np.asarray(self.training_history['train_gt']), pred)
        else:
            assert self.task_type == self.TASK_UNKNOWN
            summs = {}
        for _key, _val in summs.items(): self.full_logger.log_scalar(f"eval_train/{_key}", _val, self.step)
        self.init_training_history()

    def on_validation(self, valid_loader):
        _, all_val_gt, val_loss, val_output, val_index = self._eval_model(self.model, valid_loader, self.device, self.criterion,
                                                                    task_type=self.task_type)
        for k, v in val_loss.items():
            self.full_logger.log_scalar(f'opt/val_{k}', v, self.step)
        if self.task_type == self.TASK_CLASS:
            classwise_res, val_metrics = eval_utils.singlelabel_eval(all_val_gt, val_output, n_class=self.n_class, flat=True)
        elif self.task_type == self.TASK_MULTILABLCLASS:
            classwise_res, val_metrics = eval_utils.multilabel_eval(all_val_gt, val_output, flat=True)
        elif self.task_type == self.TASK_METRIC:
            classwise_res, val_metrics = eval_utils.singlelabel_eval(all_val_gt, val_output, n_class=self.n_class, flat=True)
        elif self.task_type == self.TASK_REGRESSION:
            val_metrics = eval_utils.regression_eval(all_val_gt, val_output)
        else:
            assert self.task_type == self.TASK_UNKNOWN
            val_metrics = {}
        val_metrics = utils.merge_dict_inline(dict(val_metrics), {"loss": val_loss['loss']})
        val_metrics = utils.merge_dict_inline(val_metrics, val_loss)
        for _key, _val in val_metrics.items(): self.full_logger.log_scalar(f"eval_valid/{_key}", _val, self.step)

        # update learning rate, if condition
        if self.scheduler is not None:
            step_on = self.settings['scheduler_step_on']
            if step_on is None:
                self.scheduler.step()
            elif step_on == 'val_loss':
                self.scheduler.step(val_loss['loss'])
            elif step_on == 'epoch':
                self.scheduler.step(self.epoch)
            else:
                raise ValueError()

        if hasattr(self.criterion, 'step'): self.criterion.step(epoch=self.epoch)

        # save best checkpoint
        self._save_checkpoint(val_metrics)

    def _save_checkpoint(self, val_metrics=None, save_override=None):
        """
        Compare metrics in val_metrics, and save the best checkpoint basing on self.settings['eval_metric']

        """
        if save_override is None: save_override = self.save_all_ckpt
        save_fnames = {}
        old_fnames = {}
        if val_metrics is not None:
            for metric, mode in self.settings['eval_metric'].items():
                if metric not in val_metrics: continue
                curr_val = val_metrics[metric]
                best_val, _, old_best_path = self.best_val_metrics[metric]
                if (mode == 'min' and curr_val < best_val) or (mode == 'max' and curr_val > best_val):
                    if old_best_path is not None: old_fnames[metric] = old_best_path
                    save_fnames[metric] = f'checkpoint_{self.step}_{metric}_{curr_val:.2e}.pth'
                    self.best_val_metrics[metric] = (curr_val, self.step, save_fnames[metric])
                    self.full_logger.info(f"Best model at {self.step} has {metric}={curr_val}")
        if save_override:
            save_fnames['_'] = f'checkpoint_{self.step}.pth'

        #Actually save the checkpoints (need to happen after we update all "best" metrics)
        if len(save_fnames):
            _curr_ckpt_dict = self._get_curr_ckpt_dict(val_metrics)
            for _, save_fname in save_fnames.items():
                torch.save(_curr_ckpt_dict, os.path.join(self.save_path, save_fname))
        #Remove the old best checkpoints..
        for _, old_path in old_fnames.items():
            old_path = os.path.join(self.save_path, old_path)
            if os.path.isfile(old_path): os.remove(old_path)
        return save_fnames

    def _get_curr_ckpt_dict(self, val_metrics=None):
        return {
                'epoch': self.epoch,
                'step': self.step,
                'val_metrics': val_metrics,
                'best_val_metrics': self.best_val_metrics,
                'state_dict': _state_dict(self.model),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
                'criterion_state_dict': _state_dict(self.criterion),
            }

    def check_early_stop(self):
        try:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.full_logger.log_scalar('opt/lr', current_lr, self.step, log_msg=True)
            if current_lr < 1e-6:
                self.full_logger.info("Early stop!")
                return True
        except:
            pass
        return False

    def on_validation_end(self):
        pass

    @classmethod
    def load_state(cls, name, key, mode='last', device=None) -> Tuple[torch.nn.Module, dict, Any]:
        #mode could be "last" or a metric (in which case we pick the best metric checkpoint)
        if name is not None:
            save_path = os.path.join(_settings.WORKSPACE, name, key)
        else:
            save_path = os.path.join(_settings.WORKSPACE, cls._BASE_SAVE_DIR, key)
        settings = torch.load(os.path.join(save_path, 'settings.pkl'), map_location=device)
        model = settings['model_class'](**settings['model_kwargs'])
        if device: _to_device(model, device)
        if mode == 'last':
            checkpoint_fname = os.path.basename(settings['%s_checkpoint' % mode])
        elif isinstance(mode, int):
            checkpoint_fname = "checkpoint_%d.pth"%mode
        else:
            _, _, checkpoint_fname = settings['best_val_metrics'][mode]
        checkpoint = torch.load(os.path.join(save_path, checkpoint_fname), map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        return model, settings, checkpoint


    def on_training_end(self):
        for _metric, _val in self.best_val_metrics.items():
            self.full_logger.info(f"Best {_metric} on Validation = {_val}")

        self.settings['last_checkpoint'] = self._save_checkpoint(save_override=True)['_']

        torch.save(self.settings, os.path.join(self.save_path, 'settings.pkl'))
        self.full_logger['parameters/final'] = self.settings.copy()
        self.full_logger.stop()

    @classmethod
    def _eval_model(cls, model,  dataloader, device, criterion=None, task_type=TASK_CLASS, forward_kwargs={}):
        model.eval()
        _to_device(model, device)

        val_history = cls.init_history(task_type=task_type, prefix='')
        with torch.no_grad():
            for all_input_dict in tqdm.tqdm(dataloader, ncols=_settings.NCOLS, desc='Evaluating...'):
                all_input_dict = _to_device(all_input_dict, device)
                output = model(all_input_dict['data'], all_input=all_input_dict, **forward_kwargs)
                loss = {}
                if criterion is not None:
                    if isinstance(output, dict):
                        output = utils.merge_dict_inline({_[0]: _[1] for _ in all_input_dict.items() if _[0] != 'target'}, output)
                    loss = criterion(output, all_input_dict['target'])
                    if isinstance(loss, torch.Tensor): loss = {'loss': loss}
                cls.update_history(val_history, all_input_dict, output, loss, task_type=task_type, prefix='')
        val_history = {k: np.asarray(v) if isinstance(v, list) else {kk:np.mean(vv) for kk,vv in v.items()} for k, v in val_history.items()}
        #TODO: compute the prediction for different task_type
        prediction = None
        return prediction, val_history['gt'], val_history['loss'], val_history['output'], val_history['index']

    @classmethod
    def eval_test(cls, key, test_data, device=None, batch_size=256, mode='best', raw_output=False):
        model, settings, checkpoint = cls.load_state(None, key, mode=mode)
        task_type = settings['task_type']

        device = device or utils.gpuid_to_device(0)
        _to_device(model, device)
        collate_fn = test_data._collate_func if hasattr(test_data, '_collate_func') else None
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        criterion = checkpoint['criterion']

        all_test_pred, all_test_gt, test_loss, outputs, all_test_indices = cls._eval_model(model, test_loader, device, criterion, task_type=task_type)
        if not raw_output:
            if task_type == cls.TASK_CLASS:
                classwise_res, summs = eval_utils.singlelabel_eval(all_test_gt, outputs, n_class=len(test_data.LABEL_MAP))
            elif task_type == cls.TASK_MULTILABLCLASS:
                classwise_res, summs = eval_utils.multilabel_eval(all_test_gt, outputs)
            elif task_type == cls.TASK_REGRESSION:
                summs = eval_utils.regression_eval(all_test_gt, outputs)
            else:
                assert task_type == cls.TASK_UNKNOWN
                summs = None
            print(summs)
        return all_test_pred, all_test_gt, test_loss['loss'], outputs, all_test_indices

    @classmethod
    def _clean_unfinished_jobs(cls, dir=None):
        import shutil

        if dir is None:
            dir = os.path.join(_settings.WORKSPACE, cls._BASE_SAVE_DIR)
        del_dir = os.path.join(dir, "..", 'DELETE')
        for key in os.listdir(dir):
            save_path = os.path.join(dir, key)
            setting_path = os.path.join(save_path, 'settings.pkl')
            if not os.path.isfile(setting_path):
                if not os.path.isdir(del_dir): os.makedirs(del_dir)
                shutil.move(save_path, os.path.join(del_dir, key))

if __name__ == '__main__':
    pass