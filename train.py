import argparse
import os
import random

import yaml
import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.optimizers as optimizers


class NullSummaryWriter(object):
  def add_scalar(self, *args, **kwargs):
    pass

  def add_scalars(self, *args, **kwargs):
    pass

  def flush(self):
    pass

  def close(self):
    pass


class WandbLogger(object):
  def __init__(self, config, args, run_name, run_dir):
    self.enabled = False
    self.run = None
    self._wandb = None
    self.log_artifacts_enabled = False
    self.artifact_name = self._artifact_safe_name(run_name)

    wandb_config = dict(config.get('wandb') or {})
    enabled = bool(wandb_config.get('enabled', False) or args.wandb)
    self.log_artifacts_enabled = bool(
      wandb_config.get('log_artifacts', False) or args.wandb_log_artifacts)

    if not enabled:
      return

    try:
      import wandb
    except ImportError as exc:
      raise RuntimeError(
        'W&B logging was requested, but wandb is not installed. '
        'Install it with: pip install wandb') from exc

    self._wandb = wandb
    project = args.wandb_project or wandb_config.get('project') or 'ag-meta-4'
    entity = args.wandb_entity or wandb_config.get('entity')
    mode = args.wandb_mode or wandb_config.get('mode') or 'online'
    notes = args.wandb_notes or wandb_config.get('notes')
    save_code = bool(wandb_config.get('save_code', False))

    tags = wandb_config.get('tags') or []
    if isinstance(tags, str):
      tags = [tags]
    else:
      tags = list(tags)
    if args.tag:
      tags.append(args.tag)
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
      self.run.summary['checkpoint_dir'] = run_dir
      self.run.summary['run_name'] = wandb_run_name
    utils.log('wandb: enabled project={} name={} mode={}'.format(
      project, wandb_run_name, mode))

  def watch(self, model, config):
    if not self.enabled:
      return
    wandb_config = dict(config.get('wandb') or {})
    if not wandb_config.get('watch_model', False):
      return
    log = wandb_config.get('watch_log', 'gradients')
    log_freq = int(wandb_config.get('watch_log_freq', 100))
    self._wandb.watch(model, log=log, log_freq=log_freq)

  def log(self, metrics, step=None):
    if self.enabled:
      self._wandb.log(metrics, step=step)

  def set_summary(self, key, value):
    if self.enabled and self.run is not None:
      self.run.summary[key] = value

  def log_model_artifacts(self, ckpt_path):
    if not self.enabled or not self.log_artifacts_enabled:
      return
    artifact = self._wandb.Artifact(self.artifact_name, type='model')
    added = False
    for filename in ['epoch-last.pth', 'max-va.pth', 'config.yaml', 'trlog.pth']:
      path = os.path.join(ckpt_path, filename)
      if os.path.exists(path):
        artifact.add_file(path, name=filename)
        added = True
    if added:
      self._wandb.log_artifact(artifact, aliases=['final'])

  def finish(self):
    if self.enabled:
      self._wandb.finish()

  @staticmethod
  def _artifact_safe_name(name):
    safe = ''.join(
      ch if ch.isalnum() or ch in ['-', '_', '.'] else '-'
      for ch in name)
    return safe.strip('.-') or 'model'


def make_ckpt_name(config, args):
  ckpt_name = args.name
  if ckpt_name is None:
    ckpt_name = config['encoder']
    ckpt_name += '_' + config['dataset'].replace('meta-', '')
    ckpt_name += '_{}_way_{}_shot'.format(
      config['train']['n_way'], config['train']['n_shot'])
  if args.tag is not None:
    ckpt_name += '_' + args.tag
  return ckpt_name


def main(config):
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)
  torch.cuda.manual_seed(0)
  torch.backends.cudnn.benchmark = True

  ckpt_name = make_ckpt_name(config, args)
  ckpt_path = os.path.join('./save', ckpt_name)
  utils.ensure_path(ckpt_path)
  utils.set_log_path(ckpt_path)
  if config.get('use_tensorboard', True):
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
  else:
    writer = NullSummaryWriter()
    utils.log('tensorboard: disabled')
  yaml.dump(config, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))
  wandb_logger = WandbLogger(config, args, ckpt_name, ckpt_path)
  use_gradient_transport = config.get('use_gradient_transport', False)

  train_set = datasets.make(config['dataset'], **config['train'])
  utils.log('meta-train set: {} (x{}), {}'.format(
    train_set[0][0].shape, len(train_set), train_set.n_classes))
  train_loader = DataLoader(
    train_set, config['train']['n_episode'],
    collate_fn=datasets.collate_fn, num_workers=8, pin_memory=True,
    prefetch_factor=4, persistent_workers=True)

  eval_val = False
  if config.get('val'):
    eval_val = True
    val_set = datasets.make(config['dataset'], **config['val'])
    utils.log('meta-val set: {} (x{}), {}'.format(
      val_set[0][0].shape, len(val_set), val_set.n_classes))
    val_loader = DataLoader(
      val_set, config['val']['n_episode'],
      collate_fn=datasets.collate_fn, num_workers=4, pin_memory=True,
      prefetch_factor=4, persistent_workers=True)

  inner_args = utils.config_inner_args(config.get('inner_args'))
  if config.get('load'):
    ckpt = torch.load(config['load'])
    config['encoder'] = ckpt['encoder']
    config['encoder_args'] = ckpt['encoder_args']
    config['classifier'] = ckpt['classifier']
    config['classifier_args'] = ckpt['classifier_args']
    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
    optimizer, lr_scheduler = optimizers.load(ckpt, model.parameters())
    start_epoch = ckpt['training']['epoch'] + 1
    max_va = ckpt['training']['max_va']
  else:
    config['encoder_args'] = config.get('encoder_args') or dict()
    config['classifier_args'] = config.get('classifier_args') or dict()
    config['encoder_args']['bn_args']['n_episode'] = config['train']['n_episode']
    config['classifier_args']['n_way'] = config['train']['n_way']
    model = models.make(config['encoder'], config['encoder_args'],
                        config['classifier'], config['classifier_args'])
    optimizer, lr_scheduler = optimizers.make(
      config['optimizer'], model.parameters(), **config['optimizer_args'])
    start_epoch = 1
    max_va = 0.

  if args.efficient:
    model.go_efficient()

  if config.get('_parallel'):
    model = nn.DataParallel(model)

  use_amp = config.get('use_amp', True)
  utils.log('num params: {}'.format(utils.compute_n_params(model)))
  utils.log('gradient transport: {}'.format(
    'enabled' if use_gradient_transport else 'disabled'))
  utils.log('amp (fp16): {}'.format('enabled' if use_amp else 'disabled'))
  wandb_logger.watch(model, config)
  timer_elapsed, timer_epoch = utils.Timer(), utils.Timer()
  scaler = amp.GradScaler('cuda', enabled=use_amp)

  aves_keys = ['tl', 'ta', 'vl', 'va']
  trlog = {k: [] for k in aves_keys}

  for epoch in range(start_epoch, config['epoch'] + 1):
    timer_epoch.start()
    aves = {k: utils.AverageMeter() for k in aves_keys}

    model.train()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    np.random.seed(epoch)

    for data in tqdm(train_loader, desc='meta-train', leave=False):
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

      optimizer.zero_grad(set_to_none=True)
      with amp.autocast('cuda', enabled=use_amp):
        logits = model(
          x_shot,
          x_query,
          y_shot,
          inner_args,
          meta_train=True,
          use_gradient_transport=use_gradient_transport)
        logits = logits.flatten(0, 1)
        labels = y_query.flatten()

        pred = torch.argmax(logits, dim=-1)
        acc = utils.compute_acc(pred, labels)
        loss = F.cross_entropy(logits, labels)

      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
      for param in optimizer.param_groups[0]['params']:
        if param.grad is not None:
          nn.utils.clip_grad_value_(param, 10)
      scaler.step(optimizer)
      scaler.update()

      aves['tl'].update(loss.item(), 1)
      aves['ta'].update(acc, 1)

    if eval_val:
      model.eval()
      np.random.seed(0)

      for data in tqdm(val_loader, desc='meta-val', leave=False):
        x_shot, x_query, y_shot, y_query = data
        x_shot, y_shot = x_shot.cuda(), y_shot.cuda()
        x_query, y_query = x_query.cuda(), y_query.cuda()

        if inner_args['reset_classifier']:
          if config.get('_parallel'):
            model.module.reset_classifier()
          else:
            model.reset_classifier()

        with torch.no_grad(), amp.autocast('cuda', enabled=use_amp):
          logits = model(
            x_shot,
            x_query,
            y_shot,
            inner_args,
            meta_train=False,
            use_gradient_transport=use_gradient_transport)
          logits = logits.flatten(0, 1)
          labels = y_query.flatten()

          pred = torch.argmax(logits, dim=-1)
          acc = utils.compute_acc(pred, labels)
          loss = F.cross_entropy(logits, labels)
        aves['vl'].update(loss.item(), 1)
        aves['va'].update(acc, 1)

    if lr_scheduler is not None:
      lr_scheduler.step()

    for k, avg in aves.items():
      aves[k] = avg.item()
      trlog[k].append(aves[k])

    gate_mean = None
    wandb_metrics = {
      'epoch': epoch,
      'optimizer/lr': optimizer.param_groups[0]['lr'],
      'train/loss': aves['tl'],
      'train/accuracy': aves['ta'],
    }
    model_for_log = model.module if config.get('_parallel') else model
    if use_gradient_transport:
      gates = model_for_log.get_gradient_transport_gates(
        frozen=inner_args['frozen'])
      if len(gates) > 0:
        gate_mean = sum(gates.values()) / len(gates)

        writer.add_scalar('gradient_transport/gate_mean', gate_mean, epoch)
        wandb_metrics['gradient_transport/gate_mean'] = gate_mean

        for gate_name, gate_value in gates.items():
          writer.add_scalar(f'gradient_transport/{gate_name}', gate_value, epoch)
          wandb_metrics[f'gradient_transport/{gate_name}'] = gate_value

    nan_grad_events, nan_grad_total = model_for_log.get_nan_grad_stats()
    nan_grad_rate = nan_grad_events / nan_grad_total if nan_grad_total > 0 else 0.
    writer.add_scalar('inner_loop/nan_grad_rate', nan_grad_rate, epoch)
    writer.add_scalar('inner_loop/nan_grad_events', nan_grad_events, epoch)
    wandb_metrics['inner_loop/nan_grad_rate'] = nan_grad_rate
    wandb_metrics['inner_loop/nan_grad_events'] = nan_grad_events

    epoch_seconds = timer_epoch.end()
    elapsed_seconds = timer_elapsed.end()
    estimate_seconds = (
      elapsed_seconds / (epoch - start_epoch + 1)
      * (config['epoch'] - start_epoch + 1))
    t_epoch = utils.time_str(epoch_seconds)
    t_elapsed = utils.time_str(elapsed_seconds)
    t_estimate = utils.time_str(estimate_seconds)
    wandb_metrics['time/epoch_sec'] = epoch_seconds
    wandb_metrics['time/elapsed_sec'] = elapsed_seconds
    wandb_metrics['time/estimated_total_sec'] = estimate_seconds

    log_str = 'epoch {}, meta-train {:.4f}|{:.4f}'.format(
      str(epoch), aves['tl'], aves['ta'])
    if use_gradient_transport and gate_mean is not None:
      log_str += ', gate_mean {:.4f}'.format(gate_mean)
    if nan_grad_events > 0:
      log_str += ', nan_grad {}/{} ({:.2%})'.format(
        nan_grad_events, nan_grad_total, nan_grad_rate)
    writer.add_scalars('loss', {'meta-train': aves['tl']}, epoch)
    writer.add_scalars('acc', {'meta-train': aves['ta']}, epoch)
    if eval_val:
      log_str += ', meta-val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
      writer.add_scalars('loss', {'meta-val': aves['vl']}, epoch)
      writer.add_scalars('acc', {'meta-val': aves['va']}, epoch)
      wandb_metrics['val/loss'] = aves['vl']
      wandb_metrics['val/accuracy'] = aves['va']
      wandb_metrics['val/best_accuracy'] = max(max_va, aves['va'])
    wandb_logger.log(wandb_metrics, step=epoch)

    log_str += ', {} {}/{}'.format(t_epoch, t_elapsed, t_estimate)
    utils.log(log_str)

    model_ = model.module if config.get('_parallel') else model
    training = {
      'epoch': epoch,
      'max_va': max(max_va, aves['va']),
      'optimizer': config['optimizer'],
      'optimizer_args': config['optimizer_args'],
      'optimizer_state_dict': optimizer.state_dict(),
      'lr_scheduler_state_dict': lr_scheduler.state_dict()
        if lr_scheduler is not None else None,
    }
    ckpt = {
      'file': __file__,
      'config': config,
      'encoder': config['encoder'],
      'encoder_args': config['encoder_args'],
      'encoder_state_dict': model_.encoder.state_dict(),
      'classifier': config['classifier'],
      'classifier_args': config['classifier_args'],
      'classifier_state_dict': model_.classifier.state_dict(),
      'gradient_transport_state_dict':
        model_.gradient_transport_logits.state_dict(),
      'training': training,
    }

    torch.save(ckpt, os.path.join(ckpt_path, 'epoch-last.pth'))
    torch.save(trlog, os.path.join(ckpt_path, 'trlog.pth'))

    if aves['va'] > max_va:
      max_va = aves['va']
      torch.save(ckpt, os.path.join(ckpt_path, 'max-va.pth'))
    wandb_logger.set_summary('best/val_accuracy', max_va)
    wandb_logger.set_summary('last/epoch', epoch)

    writer.flush()

  wandb_logger.log_model_artifacts(ckpt_path)
  wandb_logger.finish()
  writer.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',
                      help='configuration file')
  parser.add_argument('--name',
                      help='model name',
                      type=str, default=None)
  parser.add_argument('--tag',
                      help='auxiliary information',
                      type=str, default=None)
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
  parser.add_argument('--wandb-log-artifacts',
                      help='upload final checkpoints to W&B Artifacts',
                      action='store_true')
  args = parser.parse_args()
  config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  main(config)
