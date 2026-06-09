import argparse
import os
import random

import yaml
import torch
from torch import amp
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import datasets
import models
import utils


class WandbLogger(object):
  def __init__(self, config, args, run_name, run_dir):
    self.enabled = False
    self.run = None
    self._wandb = None

    wandb_config = dict(config.get('wandb') or {})
    enabled = bool(wandb_config.get('enabled', False) or args.wandb)
    if not enabled:
      return

    try:
      import wandb
    except ImportError as exc:
      raise RuntimeError(
        'W&B logging was requested, but wandb is not installed. '
        'Install it with: pip install wandb') from exc

    self._wandb = wandb
    project = args.wandb_project or wandb_config.get('project') or 'ag-meta-2'
    entity = args.wandb_entity or wandb_config.get('entity')
    mode = args.wandb_mode or wandb_config.get('mode') or 'online'
    notes = args.wandb_notes or wandb_config.get('notes')
    save_code = bool(wandb_config.get('save_code', False))

    tags = wandb_config.get('tags') or []
    if isinstance(tags, str):
      tags = [tags]
    else:
      tags = list(tags)
    tags.append('test')
    if args.wandb_tags:
      tags.extend([tag.strip() for tag in args.wandb_tags.split(',')
                   if tag.strip()])

    wandb_run_name = args.wandb_run_name or wandb_config.get('name') or run_name
    self.run = wandb.init(
      project=project,
      entity=entity,
      name=wandb_run_name,
      config=config,
      dir=run_dir,
      mode=mode,
      tags=tags,
      notes=notes,
      save_code=save_code)
    self.enabled = True
    if self.run is not None:
      self.run.summary['run_name'] = wandb_run_name
      self.run.summary['checkpoint_path'] = config.get('load')
    utils.log('wandb: enabled project={} name={} mode={}'.format(
      project, wandb_run_name, mode))

  def log(self, metrics, step=None):
    if self.enabled and self.run is not None:
      self.run.log(self._sanitize_metrics(metrics), step=step)

  def set_summary(self, key, value):
    if self.enabled and self.run is not None:
      self.run.summary[key] = self._sanitize_value(value)

  def update_summary(self, metrics):
    if self.enabled and self.run is not None:
      self.run.summary.update(self._sanitize_metrics(metrics))

  def log_table(self, name, columns, rows):
    if not self.enabled or self._wandb is None:
      return
    table = self._wandb.Table(columns=columns, data=rows)
    self.log({name: table})

  def finish(self):
    if self.enabled and self._wandb is not None:
      self._wandb.finish()

  @staticmethod
  def _sanitize_metrics(metrics):
    clean = {}
    for key, value in metrics.items():
      if value is None:
        continue
      clean[key] = WandbLogger._sanitize_value(value)
    return clean

  @staticmethod
  def _sanitize_value(value):
    if isinstance(value, np.generic):
      return value.item()
    if torch.is_tensor(value):
      return value.detach().cpu().item()
    return value


def make_test_run_name(config, ckpt_config):
  dataset = config.get('dataset', ckpt_config.get('dataset', 'dataset'))
  test_cfg = config.get('test') or {}
  n_way = test_cfg.get('n_way', 'way')
  n_shot = test_cfg.get('n_shot', 'shot')
  ckpt_name = os.path.splitext(os.path.basename(config.get('load', 'ckpt')))[0]
  return '{}_{}_way_{}_shot_{}_test'.format(
    dataset.replace('meta-', ''), n_way, n_shot, ckpt_name)


def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  torch.backends.cudnn.benchmark = True

  dataset = datasets.make(config['dataset'], **config['test'])
  utils.log('meta-test set: {} (x{}), {}'.format(
    dataset[0][0].shape, len(dataset), dataset.n_classes))
  loader = DataLoader(
    dataset, config['test']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=8, pin_memory=True,
    prefetch_factor=4, persistent_workers=True)

  ckpt = torch.load(config['load'])
  inner_args = utils.config_inner_args(config.get('inner_args'))
  ckpt_config = ckpt.get('config', {})
  run_dir = os.path.dirname(config.get('load') or '.') or '.'
  wandb_logger = WandbLogger(
    config, args, make_test_run_name(config, ckpt_config), run_dir)
  use_gradient_transport = config.get(
    'use_gradient_transport',
    ckpt_config.get('use_gradient_transport', False))
  model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
  ckpt_training = ckpt.get('training', {})

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  utils.log('num params: {}'.format(utils.compute_n_params(model)))
  utils.log('gradient transport: {}'.format(
    'enabled' if use_gradient_transport else 'disabled'))
  if 'epoch' in ckpt_training:
    wandb_logger.set_summary('checkpoint/epoch', ckpt_training['epoch'])
  if 'max_va' in ckpt_training:
    wandb_logger.set_summary(
      'checkpoint/max_val_accuracy', ckpt_training['max_va'])
  wandb_logger.set_summary(
    'gradient_transport/enabled', int(bool(use_gradient_transport)))
  if use_gradient_transport:
    if 'gradient_transport_state_dict' not in ckpt:
      utils.log(
        'warning: checkpoint has no gradient transport gates; '
        'using initialized gate values')
    model_for_log = model.module if config.get('_parallel') else model
    gates = model_for_log.get_gradient_transport_gates()
    if len(gates) > 0:
      gate_mean = sum(gates.values()) / len(gates)
      utils.log('gradient transport gate_mean: {:.4f}'.format(gate_mean))

  model.eval()
  aves_va = utils.AverageMeter()
  va_lst = []
  wandb_rows = []

  for epoch in range(1, config['epoch'] + 1):
    for data in tqdm(loader, leave=False):
      x_shot, x_query, y_shot, y_query = data
      x_shot = x_shot.cuda(non_blocking=True)
      y_shot = y_shot.cuda(non_blocking=True)
      x_query = x_query.cuda(non_blocking=True)
      y_query = y_query.cuda(non_blocking=True)

      if inner_args['reset_classifier']:
        if config.get('_parallel'):
          model.module.reset_classifier()
        else:
          model.reset_classifier()

      with amp.autocast('cuda'):
        logits = model(
          x_shot,
          x_query,
          y_shot,
          inner_args,
          meta_train=False,
          use_gradient_transport=use_gradient_transport)
        logits = logits.view(-1, config['test']['n_way'])
        labels = y_query.view(-1)

        pred = torch.argmax(logits, dim=1)
        acc = utils.compute_acc(pred, labels)

      aves_va.update(acc, 1)
      va_lst.append(acc)

    print('test epoch {}: acc={:.2f} +- {:.2f} (%)'.format(
      epoch, aves_va.item() * 100,
      utils.mean_confidence_interval(va_lst) * 100))
    ci95 = float(utils.mean_confidence_interval(va_lst))
    acc_mean = float(aves_va.item())
    acc_percent = acc_mean * 100
    ci95_percent = ci95 * 100
    wandb_rows.append([epoch, acc_mean, acc_percent, ci95, ci95_percent])
    wandb_logger.log({
      'test/epoch': epoch,
      'test/accuracy': acc_mean,
      'test/accuracy_percent': acc_percent,
      'test/confidence_interval_95': ci95,
      'test/confidence_interval_95_percent': ci95_percent,
      'test_accuracy': acc_mean,
      'test_accuracy_percent': acc_percent,
      'test_ci95': ci95,
      'test_ci95_percent': ci95_percent,
      'test/n_way': config['test']['n_way'],
      'test/n_shot': config['test']['n_shot'],
      'test/n_query': config['test']['n_query'],
      'test/n_episode': config['test']['n_episode'],
      'test/n_batch': config['test'].get('n_batch'),
      'gradient_transport/enabled': int(bool(use_gradient_transport)),
    }, step=epoch)

  final_ci95 = float(utils.mean_confidence_interval(va_lst))
  final_acc = float(aves_va.item())
  final_acc_percent = final_acc * 100
  final_ci95_percent = final_ci95 * 100
  wandb_logger.log({
    'test/epoch': config['epoch'] + 1,
    'test/final_accuracy': final_acc,
    'test/final_accuracy_percent': final_acc_percent,
    'test/final_confidence_interval_95': final_ci95,
    'test/final_confidence_interval_95_percent': final_ci95_percent,
    'test_final_accuracy': final_acc,
    'test_final_accuracy_percent': final_acc_percent,
    'test_final_ci95': final_ci95,
    'test_final_ci95_percent': final_ci95_percent,
  }, step=config['epoch'] + 1)
  wandb_logger.update_summary({
    'test/accuracy': final_acc,
    'test/accuracy_percent': final_acc_percent,
    'test/confidence_interval_95': final_ci95,
    'test/confidence_interval_95_percent': final_ci95_percent,
    'test_accuracy': final_acc,
    'test_accuracy_percent': final_acc_percent,
    'test_ci95': final_ci95,
    'test_ci95_percent': final_ci95_percent,
  })
  wandb_logger.log_table(
    'test/results',
    ['epoch', 'accuracy', 'accuracy_percent',
     'confidence_interval_95', 'confidence_interval_95_percent'],
    wandb_rows)
  wandb_logger.finish()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',
                      help='configuration file')
  parser.add_argument('--gpu',
                      help='gpu device number',
                      type=str, default='0')
  parser.add_argument('--efficient',
                      help='if True, enables gradient checkpointing',
                      action='store_true')
  parser.add_argument('--wandb',
                      help='enable Weights & Biases logging',
                      action='store_true')
  parser.add_argument('--wandb-project',
                      help='W&B project name',
                      type=str, default=None)
  parser.add_argument('--wandb-entity',
                      help='W&B team or user entity',
                      type=str, default=None)
  parser.add_argument('--wandb-mode',
                      help='W&B mode: online, offline, or disabled',
                      type=str, default=None,
                      choices=['online', 'offline', 'disabled'])
  parser.add_argument('--wandb-run-name',
                      help='W&B run name',
                      type=str, default=None)
  parser.add_argument('--wandb-tags',
                      help='comma-separated W&B tags',
                      type=str, default=None)
  parser.add_argument('--wandb-notes',
                      help='W&B run notes',
                      type=str, default=None)
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  main(config)
