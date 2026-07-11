import argparse
import csv
import json
import math
import os
import random
import sys
from contextlib import nullcontext

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

import yaml
import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

import datasets
import models
import utils


SUPPORTED_DOMAINS = ('PLK', 'RESISC')
ANCHOR_ALIASES = {
  'maml': 'maml',
  'gt': 'gt',
  'gradient_transport': 'gt',
  'gradient transport': 'gt',
  'gradient transport / scalar gate': 'gt',
  'scalar_gate': 'gt',
  'scalar gate': 'gt',
}


def seed_everything(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
  del worker_id
  worker_seed = torch.initial_seed() % (2 ** 32)
  random.seed(worker_seed)
  np.random.seed(worker_seed)


def autocast_for(device, enabled):
  if device.type == 'cuda':
    return amp.autocast('cuda', enabled=enabled)
  return nullcontext()


def canonical_anchor_name(name):
  key = str(name).strip().lower().replace('-', '_')
  return ANCHOR_ALIASES.get(key)


def resolve_domain(config, cli_domain):
  test_domains = (config.get('test') or {}).get('domains')
  config_domain = config.get('domain')

  if cli_domain is not None:
    domain = cli_domain
  elif config_domain is not None:
    domain = config_domain
  elif test_domains and len(test_domains) == 1:
    domain = test_domains[0]
  else:
    raise ValueError(
      'select exactly one Meta-Album domain with --domain or config domain; '
      'PLK and RESISC must not be pooled')

  domain = str(domain).upper()
  if domain not in SUPPORTED_DOMAINS:
    raise ValueError(
      'unsupported Meta-Album analysis domain: {} (choose {})'.format(
        domain, ', '.join(SUPPORTED_DOMAINS)))
  return domain


def resolve_n_tasks(config, cli_n_tasks):
  n_tasks = cli_n_tasks
  if n_tasks is None:
    n_tasks = config.get('n_tasks', 1000)
  n_tasks = int(n_tasks)
  if n_tasks <= 0:
    raise ValueError('n_tasks must be positive')
  return n_tasks


def prepare_test_config(config, domain, n_tasks):
  test_config = dict(config['test'])
  if config.get('data_root') and test_config.get('root_path') is None:
    test_config['root_path'] = config['data_root']

  test_config['domains'] = [domain]
  n_episode = int(test_config.get('n_episode', 4))
  if n_episode <= 0:
    raise ValueError('test.n_episode must be positive')
  test_config['n_episode'] = n_episode
  test_config['n_batch'] = int(math.ceil(n_tasks / n_episode))
  return test_config


def validate_protocol(config, test_config, inner_args):
  if config.get('dataset') != 'meta-album':
    raise ValueError('this analysis config requires dataset: meta-album')

  expected_test = {
    'n_way': 5,
    'n_shot': 1,
    'n_query': 15,
    'normalization': False,
    'transform': None,
  }
  for key, expected in expected_test.items():
    if test_config.get(key) != expected:
      raise ValueError(
        'test.{} must be {!r}, got {!r}'.format(
          key, expected, test_config.get(key)))

  expected_inner = {
    'reset_classifier': False,
    'n_step': 10,
    'encoder_lr': 0.01,
    'classifier_lr': 0.01,
  }
  for key, expected in expected_inner.items():
    if inner_args.get(key) != expected:
      raise ValueError(
        'inner_args.{} must be {!r}, got {!r}'.format(
          key, expected, inner_args.get(key)))

  if 'bn' not in inner_args.get('frozen', []):
    raise ValueError("inner_args.frozen must include 'bn'")


def apply_anchor_overrides(config, args):
  anchor_configs = [dict(anchor) for anchor in config.get('anchors') or []]
  overrides = {
    'maml': {
      'load': args.maml_checkpoint,
      'use_gradient_transport': args.maml_use_gradient_transport,
    },
    'gt': {
      'load': args.gt_checkpoint,
      'use_gradient_transport': args.gt_use_gradient_transport,
    },
  }

  for anchor in anchor_configs:
    kind = canonical_anchor_name(anchor.get('name'))
    if kind not in overrides:
      continue
    for key, value in overrides[kind].items():
      if value is not None:
        anchor[key] = value
  return anchor_configs


def validate_anchors(anchor_configs):
  if len(anchor_configs) != 2:
    raise ValueError('config must define exactly two anchors: MAML and GT')

  by_kind = {}
  for idx, anchor in enumerate(anchor_configs):
    if 'name' not in anchor:
      raise ValueError('anchors[{}] is missing name'.format(idx))
    kind = canonical_anchor_name(anchor['name'])
    if kind is None:
      raise ValueError(
        'unsupported anchor {!r}; use maml and gradient_transport'.format(
          anchor['name']))
    if kind in by_kind:
      raise ValueError('duplicate {} anchor'.format(kind))
    if not anchor.get('load'):
      raise ValueError(
        'anchor {} is missing load; set it in YAML or via CLI'.format(
          anchor['name']))
    by_kind[kind] = anchor

  if set(by_kind) != {'maml', 'gt'}:
    raise ValueError('anchors must contain one MAML and one GT anchor')
  return [by_kind['maml'], by_kind['gt']]


def make_loader(config, test_config, n_tasks, seed, device):
  dataset = datasets.make(config['dataset'], **test_config)
  if dataset.domains != [config['domain']]:
    raise RuntimeError('Meta-Album analysis dataset must contain one domain')

  analysis_dataset = Subset(dataset, range(n_tasks))
  num_workers = int(config.get('num_workers', 0))
  generator = torch.Generator()
  generator.manual_seed(seed)
  loader_kwargs = {
    'batch_size': test_config['n_episode'],
    'collate_fn': datasets.collate_fn,
    'num_workers': num_workers,
    'pin_memory': device.type == 'cuda',
    'shuffle': False,
    'worker_init_fn': seed_worker,
    'generator': generator,
  }
  if num_workers > 0:
    loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 4))
    loader_kwargs['persistent_workers'] = bool(
      config.get('persistent_workers', True))

  utils.log('meta-test domain: {}, tasks: {}, classes: {}'.format(
    config['domain'], n_tasks, dataset.n_classes))
  return DataLoader(analysis_dataset, **loader_kwargs)


def load_anchor_models(config, args, device, anchor_configs):
  inner_args = utils.config_inner_args(dict(config.get('inner_args') or {}))
  anchors = []

  for anchor_config in anchor_configs:
    kind = canonical_anchor_name(anchor_config['name'])
    ckpt_path = anchor_config['load']
    utils.log('loading {} anchor: {}'.format(kind.upper(), ckpt_path))
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_config = ckpt.get('config', {})
    use_gradient_transport = anchor_config.get(
      'use_gradient_transport',
      config.get(
        'use_gradient_transport',
        ckpt_config.get('use_gradient_transport', False)))
    use_gradient_transport = bool(use_gradient_transport)

    if kind == 'maml' and use_gradient_transport:
      raise ValueError('MAML anchor must set use_gradient_transport: false')
    if kind == 'gt' and not use_gradient_transport:
      raise ValueError('GT anchor must set use_gradient_transport: true')

    model = models.load(ckpt, load_clf=(not inner_args['reset_classifier']))
    if args.efficient:
      model.go_efficient()
    if config.get('_parallel'):
      model = nn.DataParallel(model)

    model.eval()
    model.cpu()
    if device.type == 'cuda':
      torch.cuda.empty_cache()

    if use_gradient_transport and 'gradient_transport_state_dict' not in ckpt:
      raise ValueError(
        'GT checkpoint has no gradient_transport_state_dict: {}'.format(
          ckpt_path))
    utils.log('{} params: {}, gradient transport: {}'.format(
      kind.upper(),
      utils.compute_n_params(model),
      'enabled' if use_gradient_transport else 'disabled'))
    anchors.append({
      'kind': kind,
      'name': anchor_config['name'],
      'load': ckpt_path,
      'use_gradient_transport': use_gradient_transport,
      'model': model,
    })

  if args.keep_anchors_on_gpu and device.type == 'cuda':
    for anchor in anchors:
      anchor['model'].to(device)
  return anchors, inner_args


def move_batch_to_device(data, device):
  non_blocking = device.type == 'cuda'
  return tuple(
    tensor.to(device, non_blocking=non_blocking) for tensor in data)


def evaluate_anchor(
        anchor,
        batch,
        inner_args,
        n_way,
        device,
        use_amp,
        keep_on_gpu=False):
  model = anchor['model']
  if not keep_on_gpu:
    model.to(device)
  model.eval()

  x_shot, x_query, y_shot, y_query = batch
  if inner_args['reset_classifier']:
    if isinstance(model, nn.DataParallel):
      model.module.reset_classifier()
    else:
      model.reset_classifier()

  with autocast_for(device, use_amp):
    logits = model(
      x_shot,
      x_query,
      y_shot,
      inner_args,
      meta_train=False,
      use_gradient_transport=anchor['use_gradient_transport'])
    logits = logits.view(-1, n_way)
    labels = y_query.view(-1)

    query_loss = F.cross_entropy(logits, labels, reduction='none')
    loss_per_task = query_loss.view(y_query.size(0), -1).mean(dim=1)
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float()
    accuracy_per_task = correct.view(y_query.size(0), -1).mean(dim=1)

  losses = loss_per_task.detach().float().cpu().tolist()
  accuracies = accuracy_per_task.detach().float().cpu().tolist()

  if not keep_on_gpu:
    model.cpu()
    if device.type == 'cuda':
      torch.cuda.empty_cache()
  return losses, accuracies


def choose_winner(maml_value, gt_value, higher_is_better, tolerance=1e-12):
  if np.isclose(maml_value, gt_value, rtol=0.0, atol=tolerance):
    return 'tie'
  if higher_is_better:
    return 'maml' if maml_value > gt_value else 'gt'
  return 'maml' if maml_value < gt_value else 'gt'


def make_task_rows(domain, losses, accuracies):
  rows = []
  for task_id, values in enumerate(zip(
          losses['maml'], accuracies['maml'],
          losses['gt'], accuracies['gt'])):
    maml_loss, maml_acc, gt_loss, gt_acc = values
    rows.append({
      'task_id': task_id,
      'domain': domain,
      'maml_query_loss': float(maml_loss),
      'maml_query_accuracy': float(maml_acc),
      'gt_query_loss': float(gt_loss),
      'gt_query_accuracy': float(gt_acc),
      'loss_winner': choose_winner(
        maml_loss, gt_loss, higher_is_better=False),
      'accuracy_winner': choose_winner(
        maml_acc, gt_acc, higher_is_better=True),
    })
  return rows


def compute_rates(rows, winner_key):
  total = len(rows)
  counts = {
    name: sum(row[winner_key] == name for row in rows)
    for name in ('maml', 'gt', 'tie')
  }
  return {
    'maml_win_rate': float(counts['maml'] / total),
    'gt_win_rate': float(counts['gt'] / total),
    'tie_rate': float(counts['tie'] / total),
  }


def compute_summary(config, anchors, rows, losses, accuracies, outputs, seed):
  maml_losses = np.asarray(losses['maml'], dtype=np.float64)
  gt_losses = np.asarray(losses['gt'], dtype=np.float64)
  maml_mean_loss = float(np.mean(maml_losses))
  gt_mean_loss = float(np.mean(gt_losses))
  best_anchor = 'maml' if maml_mean_loss <= gt_mean_loss else 'gt'
  best_single_loss = min(maml_mean_loss, gt_mean_loss)
  oracle_loss = float(np.mean(np.minimum(maml_losses, gt_losses)))
  absolute_gain = float(best_single_loss - oracle_loss)
  relative_gain = (
    float(absolute_gain / best_single_loss * 100.0)
    if best_single_loss != 0.0 else None)

  if np.ptp(maml_losses) == 0.0 or np.ptp(gt_losses) == 0.0:
    correlation = None
  else:
    correlation = stats.spearmanr(maml_losses, gt_losses).correlation
    if correlation is None or not np.isfinite(correlation):
      correlation = None
    else:
      correlation = float(correlation)

  anchor_by_kind = {anchor['kind']: anchor for anchor in anchors}
  return {
    'dataset': config['dataset'],
    'domain': config['domain'],
    'seed': seed,
    'n_tasks': len(rows),
    'protocol': {
      'n_way': config['test']['n_way'],
      'n_shot': config['test']['n_shot'],
      'n_query': config['test']['n_query'],
      'n_step': config['inner_args']['n_step'],
      'encoder_lr': config['inner_args']['encoder_lr'],
      'classifier_lr': config['inner_args']['classifier_lr'],
      'frozen': config['inner_args']['frozen'],
      'normalization': config['test']['normalization'],
      'transform': config['test']['transform'],
    },
    'maml': {
      'load': anchor_by_kind['maml']['load'],
      'use_gradient_transport': anchor_by_kind['maml'][
        'use_gradient_transport'],
      'mean_query_loss': maml_mean_loss,
      'mean_query_accuracy': float(np.mean(accuracies['maml'])),
    },
    'gt': {
      'load': anchor_by_kind['gt']['load'],
      'use_gradient_transport': anchor_by_kind['gt'][
        'use_gradient_transport'],
      'mean_query_loss': gt_mean_loss,
      'mean_query_accuracy': float(np.mean(accuracies['gt'])),
    },
    'loss_comparison': compute_rates(rows, 'loss_winner'),
    'accuracy_comparison': compute_rates(rows, 'accuracy_winner'),
    'loss_spearman': correlation,
    'best_single_anchor': best_anchor,
    'best_single_loss': float(best_single_loss),
    'oracle_loss': oracle_loss,
    'absolute_gain': absolute_gain,
    'relative_gain_percent': relative_gain,
    'outputs': outputs,
  }


def write_task_csv(path, rows):
  fieldnames = [
    'task_id',
    'domain',
    'maml_query_loss',
    'maml_query_accuracy',
    'gt_query_loss',
    'gt_query_accuracy',
    'loss_winner',
    'accuracy_winner',
  ]
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_summary_json(path, summary):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, allow_nan=False)


def log_summary(summary):
  utils.log('evaluated {} tasks on {}'.format(
    summary['n_tasks'], summary['domain']))
  utils.log('MAML: mean loss={:.6f}, mean accuracy={:.6f}'.format(
    summary['maml']['mean_query_loss'],
    summary['maml']['mean_query_accuracy']))
  utils.log('GT:   mean loss={:.6f}, mean accuracy={:.6f}'.format(
    summary['gt']['mean_query_loss'],
    summary['gt']['mean_query_accuracy']))

  loss_rates = summary['loss_comparison']
  accuracy_rates = summary['accuracy_comparison']
  utils.log('loss wins: MAML={:.2%}, GT={:.2%}, ties={:.2%}'.format(
    loss_rates['maml_win_rate'],
    loss_rates['gt_win_rate'],
    loss_rates['tie_rate']))
  utils.log('accuracy wins: MAML={:.2%}, GT={:.2%}, ties={:.2%}'.format(
    accuracy_rates['maml_win_rate'],
    accuracy_rates['gt_win_rate'],
    accuracy_rates['tie_rate']))
  utils.log('MAML-GT loss Spearman: {}'.format(summary['loss_spearman']))
  utils.log('best single anchor: {}, loss={:.6f}'.format(
    summary['best_single_anchor'], summary['best_single_loss']))
  utils.log('oracle loss={:.6f}, absolute gain={:.6f}, relative gain={}'.format(
    summary['oracle_loss'],
    summary['absolute_gain'],
    ('{:.4f}%'.format(summary['relative_gain_percent'])
     if summary['relative_gain_percent'] is not None else None)))


def main(config, args):
  seed = int(args.seed if args.seed is not None else config.get('seed', 0))
  seed_everything(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  domain = resolve_domain(config, args.domain)
  n_tasks = resolve_n_tasks(config, args.n_tasks)

  config['domain'] = domain
  test_config = prepare_test_config(config, domain, n_tasks)
  config['test'] = test_config
  inner_args = utils.config_inner_args(dict(config.get('inner_args') or {}))
  config['inner_args'] = inner_args
  validate_protocol(config, test_config, inner_args)

  anchor_configs = validate_anchors(apply_anchor_overrides(config, args))
  loader = make_loader(config, test_config, n_tasks, seed, device)
  anchors, inner_args = load_anchor_models(
    config, args, device, anchor_configs)
  keep_on_gpu = args.keep_anchors_on_gpu and device.type == 'cuda'
  use_amp = bool(config.get('use_amp', False))

  losses = {'maml': [], 'gt': []}
  accuracies = {'maml': [], 'gt': []}
  for data in tqdm(loader, desc='anchor-response {}'.format(domain), leave=False):
    batch = move_batch_to_device(data, device)
    for anchor in anchors:
      batch_losses, batch_accuracies = evaluate_anchor(
        anchor,
        batch,
        inner_args,
        test_config['n_way'],
        device,
        use_amp,
        keep_on_gpu=keep_on_gpu)
      losses[anchor['kind']].extend(batch_losses)
      accuracies[anchor['kind']].extend(batch_accuracies)

  rows = make_task_rows(domain, losses, accuracies)
  output_dir = args.output_dir or config.get('output_dir') or '.'
  os.makedirs(output_dir, exist_ok=True)
  domain_slug = domain.lower()
  outputs = {
    'tasks_csv': os.path.abspath(os.path.join(
      output_dir, 'anchor_response_{}.csv'.format(domain_slug))),
    'summary_json': os.path.abspath(os.path.join(
      output_dir, 'anchor_summary_{}.json'.format(domain_slug))),
  }
  summary = compute_summary(
    config, anchors, rows, losses, accuracies, outputs, seed)
  write_task_csv(outputs['tasks_csv'], rows)
  write_summary_json(outputs['summary_json'], summary)
  log_summary(summary)
  utils.log('task CSV: {}'.format(outputs['tasks_csv']))
  utils.log('summary JSON: {}'.format(outputs['summary_json']))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', required=True,
                      help='analysis configuration file')
  parser.add_argument('--domain', choices=SUPPORTED_DOMAINS,
                      help='single Meta-Album domain to analyze')
  parser.add_argument('--n-tasks', type=int, default=None,
                      help='number of episodic tasks (default: config or 1000)')
  parser.add_argument('--seed', type=int, default=None,
                      help='random seed (default: config or 0)')
  parser.add_argument('--gpu', type=str, default='0',
                      help='GPU device number')
  parser.add_argument('--efficient', action='store_true',
                      help='enable gradient checkpointing')
  parser.add_argument('--output-dir', type=str, default=None,
                      help='directory for analysis CSV/JSON outputs')
  parser.add_argument('--keep-anchors-on-gpu', action='store_true',
                      help='keep both anchor models on GPU')
  parser.add_argument('--maml-checkpoint', type=str, default=None,
                      help='override anchors[maml].load')
  parser.add_argument('--gt-checkpoint', type=str, default=None,
                      help='override anchors[gradient_transport].load')
  parser.add_argument(
    '--maml-use-gradient-transport',
    action=argparse.BooleanOptionalAction,
    default=None,
    help='override MAML use_gradient_transport')
  parser.add_argument(
    '--gt-use-gradient-transport',
    action=argparse.BooleanOptionalAction,
    default=None,
    help='override GT use_gradient_transport')
  args = parser.parse_args()

  with open(args.config, 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  if len(args.gpu.split(',')) > 1:
    config['_parallel'] = True
    config['_gpu'] = args.gpu

  utils.set_gpu(args.gpu)
  main(config, args)
