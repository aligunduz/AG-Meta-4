"""Balanced Meta-Album specialist/router/oracle analysis.

This tool intentionally does not depend on or modify analyze_anchor_response.py.
It evaluates the NATURAL, TECHNICAL, and UNION MAML checkpoints on identical
episodes and also runs a class-balanced query-split/cross-fit analysis.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
from collections import Counter
from contextlib import nullcontext


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

import datasets
import models
import utils


MODEL_NAMES = ('NATURAL', 'TECHNICAL', 'UNION')
CLUSTER_NAMES = ('NATURAL', 'TECHNICAL')
EXPECTED_DOMAINS = {
  'NATURAL': (
    'BRD', 'DOG', 'AWA', 'FLW', 'FNG', 'PLT_NET', 'SPT', 'ACT_40',
    'ACT_410'),
  'TECHNICAL': (
    'RESISC', 'RSICB', 'RSD', 'TEX', 'TEX_ALOT', 'TEX_DTD'),
}
CSV_FLOAT_FIELDS = (
  'natural_query_loss', 'natural_query_accuracy',
  'technical_query_loss', 'technical_query_accuracy',
  'union_query_loss', 'union_query_accuracy',
  'domain_router_query_loss', 'domain_router_query_accuracy',
  'oracle_query_loss', 'oracle_query_accuracy',
)


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


def resolve_project_path(path):
  path = os.path.expanduser(str(path))
  if not os.path.isabs(path):
    path = os.path.join(PROJECT_ROOT, path)
  return os.path.abspath(path)


def autocast_for(device, enabled):
  if device.type == 'cuda':
    return amp.autocast('cuda', enabled=enabled)
  return nullcontext()


def torch_load_checkpoint(path):
  try:
    return torch.load(path, map_location='cpu', weights_only=False)
  except TypeError:
    return torch.load(path, map_location='cpu')


def sha256_file(path, chunk_size=1024 * 1024):
  digest = hashlib.sha256()
  with open(path, 'rb') as f:
    while True:
      chunk = f.read(chunk_size)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def checkpoint_parameter_fingerprint(checkpoint):
  """Fingerprint the model weights used by standard MAML evaluation."""
  digest = hashlib.sha256()
  found = False
  for section in ('encoder_state_dict', 'classifier_state_dict'):
    state_dict = checkpoint.get(section)
    if not isinstance(state_dict, dict):
      continue
    for name in sorted(state_dict):
      value = state_dict[name]
      if not torch.is_tensor(value):
        continue
      found = True
      tensor = value.detach().cpu().contiguous()
      digest.update(section.encode('utf-8'))
      digest.update(name.encode('utf-8'))
      digest.update(str(tensor.dtype).encode('ascii'))
      digest.update(str(tuple(tensor.shape)).encode('ascii'))
      digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
  if not found:
    raise ValueError(
      'checkpoint has no encoder/classifier tensors to fingerprint')
  return digest.hexdigest()


def normalized_config(config):
  config = dict(config)
  config['clusters'] = {
    str(name).upper(): [str(domain).upper() for domain in domains]
    for name, domains in (config.get('clusters') or {}).items()
  }
  config['models'] = {
    str(name).upper(): dict(model_config)
    for name, model_config in (config.get('models') or {}).items()
  }
  config['inner_args'] = utils.config_inner_args(
    dict(config.get('inner_args') or {}))
  return config


def require_equal(label, actual, expected):
  if actual != expected:
    raise ValueError('{} must be {!r}, got {!r}'.format(
      label, expected, actual))


def validate_config(config):
  require_equal('dataset', config.get('dataset'), 'meta-album')
  require_equal('seed', int(config.get('seed', 0)), 0)
  if tuple(config.get('clusters', {}).keys()) != CLUSTER_NAMES:
    raise ValueError(
      'clusters must be ordered NATURAL then TECHNICAL, got {}'.format(
        list(config.get('clusters', {}))))
  for cluster in CLUSTER_NAMES:
    actual = tuple(config['clusters'][cluster])
    require_equal('clusters.{}'.format(cluster), actual,
                  EXPECTED_DOMAINS[cluster])
  all_domains = [
    domain for cluster in CLUSTER_NAMES
    for domain in config['clusters'][cluster]
  ]
  if len(all_domains) != len(set(all_domains)):
    raise ValueError('a domain may belong to only one cluster')

  tasks_per_cluster = int(config.get('tasks_per_cluster', 500))
  require_equal('tasks_per_cluster', tasks_per_cluster, 500)

  test_config = config.get('test') or {}
  expected_test = {
    'split': 'meta-test',
    'n_way': 5,
    'n_shot': 1,
    'n_query': 15,
    'normalization': False,
    'transform': None,
  }
  for key, expected in expected_test.items():
    require_equal('test.{}'.format(key), test_config.get(key), expected)
  if int(test_config.get('n_episode', 0)) <= 0:
    raise ValueError('test.n_episode must be positive')

  query_split = config.get('query_split') or {}
  require_equal('query_split.enabled', query_split.get('enabled'), True)
  require_equal('query_split.n_query', query_split.get('n_query'), 30)
  require_equal('query_split.split_query',
                query_split.get('split_query'), 15)

  expected_inner = {
    'reset_classifier': False,
    'n_step': 10,
    'encoder_lr': 0.01,
    'classifier_lr': 0.01,
  }
  for key, expected in expected_inner.items():
    require_equal('inner_args.{}'.format(key),
                  config['inner_args'].get(key), expected)
  if 'bn' not in config['inner_args'].get('frozen', []):
    raise ValueError("inner_args.frozen must include 'bn'")

  if tuple(config.get('models', {}).keys()) != MODEL_NAMES:
    raise ValueError(
      'models must be ordered NATURAL, TECHNICAL, UNION, got {}'.format(
        list(config.get('models', {}))))
  for name in MODEL_NAMES:
    model_config = config['models'][name]
    if not model_config.get('load'):
      raise ValueError('models.{}.load is required'.format(name))
    require_equal('models.{}.use_gradient_transport'.format(name),
                  model_config.get('use_gradient_transport'), False)

  manifest = config.get('class_split_manifest')
  if not manifest:
    raise ValueError('class_split_manifest is required')
  test_manifest = test_config.get('class_split_manifest', manifest)
  if resolve_project_path(test_manifest) != resolve_project_path(manifest):
    raise ValueError(
      'test.class_split_manifest must match class_split_manifest')

  bootstrap = config.get('bootstrap') or {}
  if int(bootstrap.get('n_resamples', 10000)) <= 0:
    raise ValueError('bootstrap.n_resamples must be positive')
  confidence = float(bootstrap.get('confidence', 0.95))
  require_equal('bootstrap.confidence', confidence, 0.95)


def validate_manifest(config):
  manifest_path = resolve_project_path(config['class_split_manifest'])
  if not os.path.isfile(manifest_path):
    raise FileNotFoundError(
      'class split manifest not found: {}'.format(manifest_path))
  with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)
  if not isinstance(manifest, dict):
    raise ValueError('class split manifest must contain a JSON object')

  counts = {}
  for cluster in CLUSTER_NAMES:
    for domain in config['clusters'][cluster]:
      if domain not in manifest:
        raise KeyError('manifest is missing domain {}'.format(domain))
      entry = manifest[domain]
      if not isinstance(entry, dict):
        raise ValueError('manifest entry {} must be an object'.format(domain))
      split_sets = {}
      for split in ('train', 'val', 'test'):
        values = entry.get(split)
        if not isinstance(values, list):
          raise ValueError(
            'manifest {}.{} must be a list'.format(domain, split))
        if len(values) != len(set(map(str, values))):
          raise ValueError(
            'manifest {}.{} contains duplicates'.format(domain, split))
        split_sets[split] = set(map(str, values))
      overlaps = {
        'train_val': split_sets['train'] & split_sets['val'],
        'train_test': split_sets['train'] & split_sets['test'],
        'val_test': split_sets['val'] & split_sets['test'],
      }
      nonempty_overlaps = {
        key: sorted(value) for key, value in overlaps.items() if value
      }
      if nonempty_overlaps:
        raise ValueError(
          'manifest {} has overlapping splits: {}'.format(
            domain, nonempty_overlaps))
      if len(split_sets['test']) < int(config['test']['n_way']):
        raise ValueError(
          'manifest {} test split has fewer than n_way classes'.format(
            domain))
      counts[domain] = {
        split: len(split_sets[split]) for split in ('train', 'val', 'test')
      }

  return {
    'path': manifest_path,
    'sha256': sha256_file(manifest_path),
    'requested_split': 'meta-test',
    'manifest_key_used': 'test',
    'train_val_test_disjoint': True,
    'all_requested_domains_present': True,
    'class_counts': counts,
  }


def balanced_task_specs(config, seed):
  tasks_per_cluster = int(config['tasks_per_cluster'])
  specs = []
  task_id = 0
  for cluster_index, cluster in enumerate(CLUSTER_NAMES):
    domains = config['clusters'][cluster]
    base, remainder = divmod(tasks_per_cluster, len(domains))
    scheduled_domains = []
    for domain_index, domain in enumerate(domains):
      scheduled_domains.extend(
        [domain] * (base + int(domain_index < remainder)))
    random.Random(seed + 1009 * cluster_index).shuffle(scheduled_domains)
    for domain in scheduled_domains:
      specs.append({
        'task_id': task_id,
        'cluster': cluster,
        'domain': domain,
      })
      task_id += 1
  return specs


def task_balance_report(specs):
  cluster_counts = Counter(spec['cluster'] for spec in specs)
  domain_counts = {
    cluster: Counter(
      spec['domain'] for spec in specs if spec['cluster'] == cluster)
    for cluster in CLUSTER_NAMES
  }
  for cluster in CLUSTER_NAMES:
    values = list(domain_counts[cluster].values())
    if not values or max(values) - min(values) > 1:
      raise RuntimeError(
        '{} task allocation is not maximally balanced'.format(cluster))
  return {
    'total_tasks': len(specs),
    'cluster_counts': dict(cluster_counts),
    'domain_counts': {
      cluster: dict(domain_counts[cluster]) for cluster in CLUSTER_NAMES
    },
    'max_within_cluster_domain_count_difference': {
      cluster: (
        max(domain_counts[cluster].values()) -
        min(domain_counts[cluster].values()))
      for cluster in CLUSTER_NAMES
    },
  }


def make_domain_test_config(config, domain, n_query):
  test_config = dict(config['test'])
  test_config['root_path'] = resolve_project_path(
    test_config.get('root_path') or config.get('data_root'))
  test_config['class_split_manifest'] = resolve_project_path(
    config['class_split_manifest'])
  test_config['split'] = 'meta-test'
  test_config['domains'] = [domain]
  test_config['n_query'] = int(n_query)
  test_config['n_batch'] = 1
  test_config['n_episode'] = 1
  return test_config


class ScheduledSingleDomainEpisodes(Dataset):
  """Episode dataset whose task-to-domain schedule is explicit and balanced."""

  def __init__(self, config, task_specs, n_query):
    super().__init__()
    self.task_specs = list(task_specs)
    self.domain_datasets = {}
    all_domains = [
      domain for cluster in CLUSTER_NAMES
      for domain in config['clusters'][cluster]
    ]
    for domain in all_domains:
      domain_config = make_domain_test_config(config, domain, n_query)
      dataset = datasets.make(config['dataset'], **domain_config)
      if dataset.domains != [domain]:
        raise RuntimeError(
          'task dataset for {} is not single-domain'.format(domain))
      if dataset.split != 'meta-test':
        raise RuntimeError(
          'task dataset for {} is not meta-test'.format(domain))
      if resolve_project_path(dataset.class_split_manifest) != \
          resolve_project_path(config['class_split_manifest']):
        raise RuntimeError(
          'task dataset for {} does not use the requested manifest'.format(
            domain))
      self.domain_datasets[domain] = dataset

  def __len__(self):
    return len(self.task_specs)

  def __getitem__(self, index):
    spec = self.task_specs[index]
    episode = self.domain_datasets[spec['domain']][0]
    return (
      spec['task_id'], spec['cluster'], spec['domain'],
      episode[0], episode[1], episode[2], episode[3])


def collate_scheduled_episodes(items):
  task_ids = [item[0] for item in items]
  clusters = [item[1] for item in items]
  domains = [item[2] for item in items]
  episode_batch = datasets.collate_fn([
    (item[3], item[4], item[5], item[6]) for item in items
  ])
  return task_ids, clusters, domains, episode_batch


def make_loader(config, task_specs, n_query, seed, device):
  dataset = ScheduledSingleDomainEpisodes(config, task_specs, n_query)
  num_workers = int(config.get('num_workers', 0))
  generator = torch.Generator()
  generator.manual_seed(seed)
  loader_kwargs = {
    'dataset': dataset,
    'batch_size': int(config['test']['n_episode']),
    'shuffle': False,
    'collate_fn': collate_scheduled_episodes,
    'num_workers': num_workers,
    'pin_memory': device.type == 'cuda',
    'worker_init_fn': seed_worker,
    'generator': generator,
  }
  if num_workers > 0:
    loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))
    loader_kwargs['persistent_workers'] = bool(
      config.get('persistent_workers', True))
  return DataLoader(**loader_kwargs)


def load_analysis_models(config, args, device):
  records = {}
  parameter_fingerprints = {}
  file_hashes = {}
  resolved_paths = {}

  for name in MODEL_NAMES:
    model_config = config['models'][name]
    checkpoint_path = resolve_project_path(model_config['load'])
    if not os.path.isfile(checkpoint_path):
      raise FileNotFoundError(
        '{} checkpoint not found: {}'.format(name, checkpoint_path))
    resolved_paths[name] = checkpoint_path
    utils.log('loading {} checkpoint: {}'.format(name, checkpoint_path))
    checkpoint = torch_load_checkpoint(checkpoint_path)
    require_equal('{} checkpoint encoder'.format(name),
                  checkpoint.get('encoder'), 'convnet4')
    require_equal('{} checkpoint classifier'.format(name),
                  checkpoint.get('classifier'), 'logistic')
    classifier_args = checkpoint.get('classifier_args') or {}
    require_equal('{} checkpoint classifier n_way'.format(name),
                  int(classifier_args.get('n_way', -1)), 5)

    file_hash = sha256_file(checkpoint_path)
    parameter_fingerprint = checkpoint_parameter_fingerprint(checkpoint)
    file_hashes[name] = file_hash
    parameter_fingerprints[name] = parameter_fingerprint

    model = models.load(checkpoint, load_clf=True)
    if args.efficient:
      model.go_efficient()
    model.eval()
    model.cpu()
    if config.get('_parallel'):
      model = nn.DataParallel(model)

    records[name] = {
      'name': name,
      'load': checkpoint_path,
      'use_gradient_transport': False,
      'file_sha256': file_hash,
      'parameter_sha256': parameter_fingerprint,
      'model': model,
    }
    utils.log('{} params: {}, gradient transport: disabled'.format(
      name, utils.compute_n_params(model)))

  if len(set(resolved_paths.values())) != len(MODEL_NAMES):
    raise ValueError('NATURAL, TECHNICAL, and UNION paths must be distinct')
  if len(set(file_hashes.values())) != len(MODEL_NAMES):
    raise ValueError(
      'NATURAL, TECHNICAL, and UNION checkpoint files are not all distinct')
  if len(set(parameter_fingerprints.values())) != len(MODEL_NAMES):
    raise ValueError(
      'NATURAL, TECHNICAL, and UNION model parameters are not all distinct')

  keep_on_gpu = (
    bool(config.get('keep_models_on_gpu', True))
    if args.keep_models_on_gpu is None else args.keep_models_on_gpu)
  keep_on_gpu = bool(keep_on_gpu and device.type == 'cuda')
  if keep_on_gpu:
    for record in records.values():
      record['model'].to(device)

  checkpoint_report = {
    'all_paths_distinct': True,
    'all_file_hashes_distinct': True,
    'all_parameter_hashes_distinct': True,
    'encoder_verified_as_convnet4': True,
    'classifier_verified_as_5way_logistic': True,
    'gradient_transport_disabled_for_all': True,
    'models': {
      name: {
        'load': records[name]['load'],
        'file_sha256': records[name]['file_sha256'],
        'parameter_sha256': records[name]['parameter_sha256'],
      }
      for name in MODEL_NAMES
    },
  }
  return records, keep_on_gpu, checkpoint_report


def move_batch_to_device(batch, device):
  non_blocking = device.type == 'cuda'
  return tuple(
    tensor.to(device, non_blocking=non_blocking) for tensor in batch)


def batch_storage_signature(batch):
  return tuple(
    (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype))
    for tensor in batch)


def evaluate_model(
        record,
        batch,
        inner_args,
        n_way,
        device,
        use_amp,
        keep_on_gpu):
  model = record['model']
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
      use_gradient_transport=False)
    logits = logits.view(-1, n_way)
    labels = y_query.view(-1)
    element_losses = F.cross_entropy(logits, labels, reduction='none')
    task_losses = element_losses.view(y_query.size(0), -1).mean(dim=1)
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float()
    task_accuracies = correct.view(y_query.size(0), -1).mean(dim=1)

  losses = task_losses.detach().float().cpu().numpy()
  accuracies = task_accuracies.detach().float().cpu().numpy()
  if not np.isfinite(losses).all() or not np.isfinite(accuracies).all():
    raise FloatingPointError(
      '{} produced NaN or Inf query metrics'.format(record['name']))

  if not keep_on_gpu:
    model.cpu()
    if device.type == 'cuda':
      torch.cuda.empty_cache()
  return losses.tolist(), accuracies.tolist()


def split_class_balanced_query_batch(batch, n_way, n_query, split_query):
  x_shot, x_query, y_shot, y_query = batch
  if n_query != split_query * 2:
    raise ValueError(
      'query split requires n_query == 2 * split_query, got {} and {}'.format(
        n_query, split_query))

  query_a, labels_a, query_b, labels_b = [], [], [], []
  for episode in range(x_query.size(0)):
    episode_query_a = []
    episode_labels_a = []
    episode_query_b = []
    episode_labels_b = []
    for cls in range(n_way):
      class_indices = (y_query[episode] == cls).nonzero(
        as_tuple=False).flatten()
      if class_indices.numel() != n_query:
        raise ValueError(
          'task {} class {} has {} query examples; expected {}'.format(
            episode, cls, class_indices.numel(), n_query))
      indices_a = class_indices[:split_query]
      indices_b = class_indices[split_query:]
      episode_query_a.append(x_query[episode, indices_a])
      episode_labels_a.append(y_query[episode, indices_a])
      episode_query_b.append(x_query[episode, indices_b])
      episode_labels_b.append(y_query[episode, indices_b])

    query_a.append(torch.cat(episode_query_a, dim=0))
    labels_a.append(torch.cat(episode_labels_a, dim=0))
    query_b.append(torch.cat(episode_query_b, dim=0))
    labels_b.append(torch.cat(episode_labels_b, dim=0))

  batch_a = (
    x_shot,
    torch.stack(query_a, dim=0),
    y_shot,
    torch.stack(labels_a, dim=0),
  )
  batch_b = (
    x_shot,
    torch.stack(query_b, dim=0),
    y_shot,
    torch.stack(labels_b, dim=0),
  )
  for split_name, split_batch in (('A', batch_a), ('B', batch_b)):
    split_labels = split_batch[3]
    for cls in range(n_way):
      counts = (split_labels == cls).sum(dim=1)
      if not torch.all(counts == split_query):
        raise RuntimeError(
          'query split {} is not class-balanced'.format(split_name))
  return batch_a, batch_b


def evaluate_standard_tasks(
        config,
        model_records,
        loader,
        device,
        keep_on_gpu):
  results = {
    name: {'loss': [], 'accuracy': []} for name in MODEL_NAMES
  }
  evaluated_ids = {name: [] for name in MODEL_NAMES}
  seen_metadata = []
  use_amp = bool(config.get('use_amp', False))
  inner_args = config['inner_args']
  n_way = int(config['test']['n_way'])

  for task_ids, clusters, domains, cpu_batch in tqdm(
      loader, desc='specialist oracle', leave=False):
    batch = move_batch_to_device(cpu_batch, device)
    signature = batch_storage_signature(batch)
    seen_metadata.extend(zip(task_ids, clusters, domains))
    for name in MODEL_NAMES:
      if batch_storage_signature(batch) != signature:
        raise RuntimeError('episode tensors changed between checkpoints')
      losses, accuracies = evaluate_model(
        model_records[name], batch, inner_args, n_way, device, use_amp,
        keep_on_gpu)
      if len(losses) != len(task_ids):
        raise RuntimeError('{} returned the wrong task count'.format(name))
      results[name]['loss'].extend(losses)
      results[name]['accuracy'].extend(accuracies)
      evaluated_ids[name].extend(task_ids)

  return results, evaluated_ids, seen_metadata


def evaluate_query_split_tasks(
        config,
        model_records,
        loader,
        device,
        keep_on_gpu):
  results = {
    name: {
      'A': {'loss': [], 'accuracy': []},
      'B': {'loss': [], 'accuracy': []},
    }
    for name in MODEL_NAMES
  }
  evaluated_ids = {name: [] for name in MODEL_NAMES}
  seen_metadata = []
  use_amp = bool(config.get('use_amp', False))
  inner_args = config['inner_args']
  n_way = int(config['test']['n_way'])
  n_query = int(config['query_split']['n_query'])
  split_query = int(config['query_split']['split_query'])

  for task_ids, clusters, domains, cpu_batch in tqdm(
      loader, desc='query-split cross-fit', leave=False):
    batch = move_batch_to_device(cpu_batch, device)
    batch_a, batch_b = split_class_balanced_query_batch(
      batch, n_way, n_query, split_query)
    signature_a = batch_storage_signature(batch_a)
    signature_b = batch_storage_signature(batch_b)
    seen_metadata.extend(zip(task_ids, clusters, domains))

    for name in MODEL_NAMES:
      if (batch_storage_signature(batch_a) != signature_a or
          batch_storage_signature(batch_b) != signature_b):
        raise RuntimeError(
          'query-split episode tensors changed between checkpoints')
      for split_name, split_batch in (('A', batch_a), ('B', batch_b)):
        losses, accuracies = evaluate_model(
          model_records[name], split_batch, inner_args, n_way, device,
          use_amp, keep_on_gpu)
        if len(losses) != len(task_ids):
          raise RuntimeError(
            '{} split {} returned the wrong task count'.format(
              name, split_name))
        results[name][split_name]['loss'].extend(losses)
        results[name][split_name]['accuracy'].extend(accuracies)
      evaluated_ids[name].extend(task_ids)

  return results, evaluated_ids, seen_metadata


def as_finite_array(values, label):
  array = np.asarray(values, dtype=np.float64)
  if array.ndim != 1 or array.size == 0:
    raise ValueError('{} must be a non-empty vector'.format(label))
  if not np.isfinite(array).all():
    raise FloatingPointError('{} contains NaN or Inf'.format(label))
  return array


def safe_relative_percent(numerator, denominator):
  if denominator == 0.0:
    return None
  return float(numerator / denominator * 100.0)


def estimate_interval(estimate, samples, confidence):
  alpha = (1.0 - confidence) / 2.0
  low, high = np.quantile(samples, [alpha, 1.0 - alpha])
  return {
    'estimate': float(estimate),
    'low': float(low),
    'high': float(high),
  }


def bootstrap_mean_ci(values, n_resamples, confidence, seed):
  values = as_finite_array(values, 'bootstrap values')
  rng = np.random.default_rng(seed)
  samples = np.empty(n_resamples, dtype=np.float64)
  chunk_size = 256
  for start in range(0, n_resamples, chunk_size):
    size = min(chunk_size, n_resamples - start)
    indices = rng.integers(0, values.size, size=(size, values.size))
    samples[start:start + size] = values[indices].mean(axis=1)
  return estimate_interval(float(values.mean()), samples, confidence)


def paired_bootstrap_ci(
        candidate,
        reference,
        n_resamples,
        confidence,
        seed):
  candidate = as_finite_array(candidate, 'paired candidate')
  reference = as_finite_array(reference, 'paired reference')
  if candidate.shape != reference.shape:
    raise ValueError('paired bootstrap arrays must have identical shapes')

  rng = np.random.default_rng(seed)
  candidate_samples = np.empty(n_resamples, dtype=np.float64)
  reference_samples = np.empty(n_resamples, dtype=np.float64)
  relative_samples = np.empty(n_resamples, dtype=np.float64)
  relative_absolute_samples = np.empty(n_resamples, dtype=np.float64)
  chunk_size = 256
  for start in range(0, n_resamples, chunk_size):
    size = min(chunk_size, n_resamples - start)
    indices = rng.integers(0, candidate.size,
                           size=(size, candidate.size))
    candidate_mean = candidate[indices].mean(axis=1)
    reference_mean = reference[indices].mean(axis=1)
    candidate_samples[start:start + size] = candidate_mean
    reference_samples[start:start + size] = reference_mean
    relative_samples[start:start + size] = np.divide(
      reference_mean - candidate_mean,
      reference_mean,
      out=np.full(size, np.nan, dtype=np.float64),
      where=reference_mean != 0.0) * 100.0
    relative_absolute_samples[start:start + size] = np.divide(
      np.abs(candidate_mean - reference_mean),
      reference_mean,
      out=np.full(size, np.nan, dtype=np.float64),
      where=reference_mean != 0.0) * 100.0

  candidate_mean = float(candidate.mean())
  reference_mean = float(reference.mean())
  difference_samples = candidate_samples - reference_samples
  gain_samples = -difference_samples
  finite_relative = relative_samples[np.isfinite(relative_samples)]
  finite_relative_absolute = relative_absolute_samples[
    np.isfinite(relative_absolute_samples)]
  output = {
    'candidate_mean': estimate_interval(
      candidate_mean, candidate_samples, confidence),
    'reference_mean': estimate_interval(
      reference_mean, reference_samples, confidence),
    'candidate_minus_reference': estimate_interval(
      candidate_mean - reference_mean, difference_samples, confidence),
    'absolute_difference': estimate_interval(
      abs(candidate_mean - reference_mean),
      np.abs(difference_samples), confidence),
    'reference_minus_candidate_gain': estimate_interval(
      reference_mean - candidate_mean, gain_samples, confidence),
  }
  point_relative = safe_relative_percent(
    reference_mean - candidate_mean, reference_mean)
  output['relative_gain_percent'] = (
    estimate_interval(point_relative, finite_relative, confidence)
    if point_relative is not None and finite_relative.size else None)
  point_relative_absolute = safe_relative_percent(
    abs(candidate_mean - reference_mean), reference_mean)
  output['relative_absolute_difference_percent'] = (
    estimate_interval(
      point_relative_absolute, finite_relative_absolute, confidence)
    if (point_relative_absolute is not None and
        finite_relative_absolute.size) else None)
  return output


def query_split_noise_floor_bootstrap(
        losses_a,
        losses_b,
        n_resamples,
        confidence,
        seed):
  losses_a = as_finite_array(losses_a, 'noise-floor A losses')
  losses_b = as_finite_array(losses_b, 'noise-floor B losses')
  if losses_a.shape != losses_b.shape:
    raise ValueError('noise-floor arrays must have identical shapes')

  rng = np.random.default_rng(seed)
  best_single_samples = np.empty(n_resamples, dtype=np.float64)
  oracle_samples = np.empty(n_resamples, dtype=np.float64)
  relative_samples = np.empty(n_resamples, dtype=np.float64)
  chunk_size = 256
  for start in range(0, n_resamples, chunk_size):
    size = min(chunk_size, n_resamples - start)
    indices = rng.integers(0, losses_a.size,
                           size=(size, losses_a.size))
    sampled_a = losses_a[indices]
    sampled_b = losses_b[indices]
    best_single = np.minimum(
      sampled_a.mean(axis=1), sampled_b.mean(axis=1))
    oracle = np.minimum(sampled_a, sampled_b).mean(axis=1)
    gain = best_single - oracle
    best_single_samples[start:start + size] = best_single
    oracle_samples[start:start + size] = oracle
    relative_samples[start:start + size] = np.divide(
      gain,
      best_single,
      out=np.full(size, np.nan, dtype=np.float64),
      where=best_single != 0.0) * 100.0

  mean_a = float(losses_a.mean())
  mean_b = float(losses_b.mean())
  best_single = min(mean_a, mean_b)
  oracle = float(np.minimum(losses_a, losses_b).mean())
  gain = best_single - oracle
  gain_samples = best_single_samples - oracle_samples
  finite_relative = relative_samples[np.isfinite(relative_samples)]
  relative = safe_relative_percent(gain, best_single)
  return {
    'best_single_loss': estimate_interval(
      best_single, best_single_samples, confidence),
    'oracle_loss': estimate_interval(oracle, oracle_samples, confidence),
    'absolute_gain': estimate_interval(gain, gain_samples, confidence),
    'relative_gain_percent': (
      estimate_interval(relative, finite_relative, confidence)
      if relative is not None and finite_relative.size else None),
  }


def average_tie_ranks(values):
  values = np.asarray(values, dtype=np.float64)
  order = np.argsort(values, kind='mergesort')
  sorted_values = values[order]
  ranks = np.empty(values.size, dtype=np.float64)
  start = 0
  while start < values.size:
    end = start + 1
    while end < values.size and sorted_values[end] == sorted_values[start]:
      end += 1
    # scipy.stats.rankdata(method='average') uses one-based ranks.
    average_rank = 0.5 * ((start + 1) + end)
    ranks[order[start:end]] = average_rank
    start = end
  return ranks


def spearman_correlation(left, right):
  left = as_finite_array(left, 'Spearman left')
  right = as_finite_array(right, 'Spearman right')
  if left.shape != right.shape:
    raise ValueError('Spearman arrays must have identical shapes')
  if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
    return None
  left_ranks = average_tie_ranks(left)
  right_ranks = average_tie_ranks(right)
  correlation = np.corrcoef(left_ranks, right_ranks)[0, 1]
  if not np.isfinite(correlation):
    return None
  return float(correlation)


def paired_loss_diagnostics_bootstrap(
        natural_losses,
        technical_losses,
        n_resamples,
        confidence,
        seed,
        tolerance=1e-12):
  natural_losses = as_finite_array(
    natural_losses, 'diagnostic NATURAL losses')
  technical_losses = as_finite_array(
    technical_losses, 'diagnostic TECHNICAL losses')
  if natural_losses.shape != technical_losses.shape:
    raise ValueError('diagnostic loss arrays must have identical shapes')

  ties = np.isclose(
    natural_losses, technical_losses, rtol=0.0, atol=tolerance)
  natural_win = ((natural_losses < technical_losses) & ~ties).astype(
    np.float64)
  technical_win = ((technical_losses < natural_losses) & ~ties).astype(
    np.float64)
  tie = ties.astype(np.float64)
  rng = np.random.default_rng(seed)
  natural_samples = np.empty(n_resamples, dtype=np.float64)
  technical_samples = np.empty(n_resamples, dtype=np.float64)
  tie_samples = np.empty(n_resamples, dtype=np.float64)
  correlation_samples = np.empty(n_resamples, dtype=np.float64)
  chunk_size = 128
  for start in range(0, n_resamples, chunk_size):
    size = min(chunk_size, n_resamples - start)
    indices = rng.integers(
      0, natural_losses.size, size=(size, natural_losses.size))
    natural_samples[start:start + size] = natural_win[indices].mean(axis=1)
    technical_samples[start:start + size] = technical_win[indices].mean(
      axis=1)
    tie_samples[start:start + size] = tie[indices].mean(axis=1)
    for offset, sampled_indices in enumerate(indices):
      correlation = spearman_correlation(
        natural_losses[sampled_indices],
        technical_losses[sampled_indices])
      correlation_samples[start + offset] = (
        np.nan if correlation is None else correlation)

  finite_correlation = correlation_samples[
    np.isfinite(correlation_samples)]
  point_correlation = spearman_correlation(
    natural_losses, technical_losses)
  return {
    'natural_win_rate': estimate_interval(
      float(natural_win.mean()), natural_samples, confidence),
    'technical_win_rate': estimate_interval(
      float(technical_win.mean()), technical_samples, confidence),
    'tie_rate': estimate_interval(float(tie.mean()), tie_samples, confidence),
    'loss_spearman': (
      estimate_interval(
        point_correlation, finite_correlation, confidence)
      if point_correlation is not None and finite_correlation.size else None),
  }


def winner_rates(left, right, left_name, right_name, tolerance=1e-12):
  left = as_finite_array(left, 'winner left')
  right = as_finite_array(right, 'winner right')
  if left.shape != right.shape:
    raise ValueError('winner arrays must have identical shapes')
  ties = np.isclose(left, right, rtol=0.0, atol=tolerance)
  left_wins = (left < right) & ~ties
  right_wins = (right < left) & ~ties
  total = left.size
  return {
    '{}_wins'.format(left_name.lower()): int(left_wins.sum()),
    '{}_wins'.format(right_name.lower()): int(right_wins.sum()),
    'ties': int(ties.sum()),
    '{}_win_rate'.format(left_name.lower()): float(left_wins.sum() / total),
    '{}_win_rate'.format(right_name.lower()): float(
      right_wins.sum() / total),
    'tie_rate': float(ties.sum() / total),
  }


def metric_summary(losses, accuracies, bootstrap, seed):
  losses = as_finite_array(losses, 'metric losses')
  accuracies = as_finite_array(accuracies, 'metric accuracies')
  return {
    'mean_query_loss': float(losses.mean()),
    'mean_query_accuracy': float(accuracies.mean()),
    'paired_bootstrap_ci95': {
      'mean_query_loss': bootstrap_mean_ci(
        losses, bootstrap['n_resamples'], bootstrap['confidence'], seed),
      'mean_query_accuracy': bootstrap_mean_ci(
        accuracies, bootstrap['n_resamples'], bootstrap['confidence'],
        seed + 1),
    },
  }


def comparison_summary(
        candidate_losses,
        reference_losses,
        candidate_accuracies,
        reference_accuracies,
        bootstrap,
        seed):
  candidate_losses = as_finite_array(candidate_losses, 'candidate losses')
  reference_losses = as_finite_array(reference_losses, 'reference losses')
  candidate_accuracies = as_finite_array(
    candidate_accuracies, 'candidate accuracies')
  reference_accuracies = as_finite_array(
    reference_accuracies, 'reference accuracies')
  candidate_loss = float(candidate_losses.mean())
  reference_loss = float(reference_losses.mean())
  signed_difference = candidate_loss - reference_loss
  gain = reference_loss - candidate_loss
  candidate_accuracy = float(candidate_accuracies.mean())
  reference_accuracy = float(reference_accuracies.mean())
  return {
    'candidate_mean_query_loss': candidate_loss,
    'reference_mean_query_loss': reference_loss,
    'candidate_minus_reference_loss': signed_difference,
    'absolute_loss_difference': abs(signed_difference),
    'relative_absolute_loss_difference_percent': safe_relative_percent(
      abs(signed_difference), reference_loss),
    'reference_minus_candidate_loss_gain': gain,
    'relative_loss_gain_percent': safe_relative_percent(gain, reference_loss),
    'candidate_mean_query_accuracy': candidate_accuracy,
    'reference_mean_query_accuracy': reference_accuracy,
    'candidate_minus_reference_accuracy': (
      candidate_accuracy - reference_accuracy),
    'paired_bootstrap_ci95': {
      'loss': paired_bootstrap_ci(
        candidate_losses, reference_losses,
        bootstrap['n_resamples'], bootstrap['confidence'], seed),
      'accuracy': paired_bootstrap_ci(
        candidate_accuracies, reference_accuracies,
        bootstrap['n_resamples'], bootstrap['confidence'], seed + 1),
    },
  }


def bootstrap_config(config, args):
  bootstrap = dict(config.get('bootstrap') or {})
  if args.bootstrap_resamples is not None:
    bootstrap['n_resamples'] = args.bootstrap_resamples
  resolved = {
    'n_resamples': int(bootstrap.get('n_resamples', 10000)),
    'confidence': float(bootstrap.get('confidence', 0.95)),
    'seed': int(bootstrap.get('seed', 1729)),
  }
  if resolved['n_resamples'] <= 0:
    raise ValueError('bootstrap resample count must be positive')
  return resolved


def make_standard_rows(task_specs, results):
  rows = []
  for index, spec in enumerate(task_specs):
    natural_loss = float(results['NATURAL']['loss'][index])
    technical_loss = float(results['TECHNICAL']['loss'][index])
    natural_accuracy = float(results['NATURAL']['accuracy'][index])
    technical_accuracy = float(results['TECHNICAL']['accuracy'][index])
    union_loss = float(results['UNION']['loss'][index])
    union_accuracy = float(results['UNION']['accuracy'][index])

    # Exact ties are deterministically routed to NATURAL. The tie flag keeps
    # this convention explicit in both the CSV and summary.
    selected_specialist = (
      'NATURAL' if natural_loss <= technical_loss else 'TECHNICAL')
    specialist_tie = bool(np.isclose(
      natural_loss, technical_loss, rtol=0.0, atol=1e-12))
    domain_specialist = spec['cluster']
    oracle_loss = (
      natural_loss if selected_specialist == 'NATURAL' else technical_loss)
    oracle_accuracy = (
      natural_accuracy
      if selected_specialist == 'NATURAL' else technical_accuracy)
    domain_router_loss = (
      natural_loss if domain_specialist == 'NATURAL' else technical_loss)
    domain_router_accuracy = (
      natural_accuracy
      if domain_specialist == 'NATURAL' else technical_accuracy)

    rows.append({
      'task_id': spec['task_id'],
      'cluster': spec['cluster'],
      'domain': spec['domain'],
      'natural_query_loss': natural_loss,
      'natural_query_accuracy': natural_accuracy,
      'technical_query_loss': technical_loss,
      'technical_query_accuracy': technical_accuracy,
      'union_query_loss': union_loss,
      'union_query_accuracy': union_accuracy,
      'loss_selected_specialist': selected_specialist,
      'specialist_loss_tie': specialist_tie,
      'domain_specialist': domain_specialist,
      'domain_router_query_loss': domain_router_loss,
      'domain_router_query_accuracy': domain_router_accuracy,
      'oracle_query_loss': oracle_loss,
      'oracle_query_accuracy': oracle_accuracy,
    })
  return rows


def make_query_split_rows(task_specs, results):
  rows = []
  for index, spec in enumerate(task_specs):
    values = {}
    for model_name in MODEL_NAMES:
      slug = model_name.lower()
      for split_name in ('A', 'B'):
        split_slug = split_name.lower()
        values['{}_{}_query_loss'.format(slug, split_slug)] = float(
          results[model_name][split_name]['loss'][index])
        values['{}_{}_query_accuracy'.format(slug, split_slug)] = float(
          results[model_name][split_name]['accuracy'][index])

    select_on_a = (
      'NATURAL'
      if values['natural_a_query_loss'] <=
      values['technical_a_query_loss'] else 'TECHNICAL')
    select_on_b = (
      'NATURAL'
      if values['natural_b_query_loss'] <=
      values['technical_b_query_loss'] else 'TECHNICAL')

    a_model = select_on_a.lower()
    b_model = select_on_b.lower()
    a_to_b_loss = values['{}_b_query_loss'.format(a_model)]
    a_to_b_accuracy = values['{}_b_query_accuracy'.format(a_model)]
    b_to_a_loss = values['{}_a_query_loss'.format(b_model)]
    b_to_a_accuracy = values['{}_a_query_accuracy'.format(b_model)]

    oracle_a_model = (
      'natural'
      if values['natural_a_query_loss'] <=
      values['technical_a_query_loss'] else 'technical')
    oracle_b_model = (
      'natural'
      if values['natural_b_query_loss'] <=
      values['technical_b_query_loss'] else 'technical')
    in_sample_oracle_loss = 0.5 * (
      values['{}_a_query_loss'.format(oracle_a_model)] +
      values['{}_b_query_loss'.format(oracle_b_model)])
    in_sample_oracle_accuracy = 0.5 * (
      values['{}_a_query_accuracy'.format(oracle_a_model)] +
      values['{}_b_query_accuracy'.format(oracle_b_model)])

    row = {
      'task_id': spec['task_id'],
      'cluster': spec['cluster'],
      'domain': spec['domain'],
      **values,
      'specialist_selected_on_a': select_on_a,
      'specialist_selected_on_b': select_on_b,
      'a_to_b_query_loss': a_to_b_loss,
      'a_to_b_query_accuracy': a_to_b_accuracy,
      'b_to_a_query_loss': b_to_a_loss,
      'b_to_a_query_accuracy': b_to_a_accuracy,
      'cross_fitted_oracle_query_loss': 0.5 * (
        a_to_b_loss + b_to_a_loss),
      'cross_fitted_oracle_query_accuracy': 0.5 * (
        a_to_b_accuracy + b_to_a_accuracy),
      'cross_fitted_union_query_loss': 0.5 * (
        values['union_a_query_loss'] + values['union_b_query_loss']),
      'cross_fitted_union_query_accuracy': 0.5 * (
        values['union_a_query_accuracy'] +
        values['union_b_query_accuracy']),
      'in_sample_oracle_query_loss': in_sample_oracle_loss,
      'in_sample_oracle_query_accuracy': in_sample_oracle_accuracy,
      'union_split_oracle_query_loss': min(
        values['union_a_query_loss'], values['union_b_query_loss']),
    }
    rows.append(row)
  return rows


def protocol_summary(config, n_query):
  return {
    'phase': 'meta-test',
    'encoder': 'convnet4',
    'n_way': int(config['test']['n_way']),
    'n_shot': int(config['test']['n_shot']),
    'n_query': int(n_query),
    'n_step': int(config['inner_args']['n_step']),
    'encoder_lr': float(config['inner_args']['encoder_lr']),
    'classifier_lr': float(config['inner_args']['classifier_lr']),
    'frozen': list(config['inner_args']['frozen']),
    'normalization': config['test']['normalization'],
    'transform': config['test']['transform'],
    'use_gradient_transport': False,
  }


def compute_standard_summary(
        config,
        rows,
        results,
        outputs,
        checkpoint_report,
        manifest_report,
        validation,
        bootstrap):
  arrays = {
    name: {
      metric: as_finite_array(results[name][metric],
                              '{} {}'.format(name, metric))
      for metric in ('loss', 'accuracy')
    }
    for name in MODEL_NAMES
  }
  router_losses = as_finite_array(
    [row['domain_router_query_loss'] for row in rows], 'router losses')
  router_accuracies = as_finite_array(
    [row['domain_router_query_accuracy'] for row in rows],
    'router accuracies')
  oracle_losses = as_finite_array(
    [row['oracle_query_loss'] for row in rows], 'oracle losses')
  oracle_accuracies = as_finite_array(
    [row['oracle_query_accuracy'] for row in rows], 'oracle accuracies')

  cluster_masks = {
    cluster: np.asarray(
      [row['cluster'] == cluster for row in rows], dtype=bool)
    for cluster in CLUSTER_NAMES
  }
  win_rates = {
    'overall': winner_rates(
      arrays['NATURAL']['loss'], arrays['TECHNICAL']['loss'],
      'NATURAL', 'TECHNICAL'),
  }
  correlations = {
    'overall': spearman_correlation(
      arrays['NATURAL']['loss'], arrays['TECHNICAL']['loss']),
  }
  for cluster, mask in cluster_masks.items():
    win_rates[cluster] = winner_rates(
      arrays['NATURAL']['loss'][mask],
      arrays['TECHNICAL']['loss'][mask],
      'NATURAL', 'TECHNICAL')
    correlations[cluster] = spearman_correlation(
      arrays['NATURAL']['loss'][mask],
      arrays['TECHNICAL']['loss'][mask])

  diagnostic_bootstrap = {}
  diagnostic_groups = {'overall': np.ones(len(rows), dtype=bool)}
  diagnostic_groups.update(cluster_masks)
  for offset, (group_name, mask) in enumerate(diagnostic_groups.items()):
    diagnostic_bootstrap[group_name] = paired_loss_diagnostics_bootstrap(
      arrays['NATURAL']['loss'][mask],
      arrays['TECHNICAL']['loss'][mask],
      bootstrap['n_resamples'], bootstrap['confidence'],
      bootstrap['seed'] + 500 + 10 * offset)

  base_seed = bootstrap['seed']
  model_summaries = {}
  for offset, name in enumerate(MODEL_NAMES):
    model_summaries[name] = {
      'load': checkpoint_report['models'][name]['load'],
      'use_gradient_transport': False,
      **metric_summary(
        arrays[name]['loss'], arrays[name]['accuracy'], bootstrap,
        base_seed + 10 * offset),
    }

  return {
    'mode': 'specialist_oracle',
    'dataset': config['dataset'],
    'seed': int(config.get('seed', 0)),
    'n_tasks': len(rows),
    'n_tasks_per_cluster': int(config['tasks_per_cluster']),
    'clusters': config['clusters'],
    'protocol': protocol_summary(config, config['test']['n_query']),
    'checkpoints': checkpoint_report,
    'models': model_summaries,
    'domain_router': {
      'routing_rule': (
        'NATURAL checkpoint for NATURAL tasks; TECHNICAL checkpoint for '
        'TECHNICAL tasks'),
      **metric_summary(
        router_losses, router_accuracies, bootstrap, base_seed + 100),
    },
    'per_task_oracle': {
      'selection_rule': (
        'lower query loss between NATURAL and TECHNICAL; '
        'exact ties select NATURAL'),
      **metric_summary(
        oracle_losses, oracle_accuracies, bootstrap, base_seed + 110),
    },
    'oracle_vs_union': comparison_summary(
      oracle_losses, arrays['UNION']['loss'],
      oracle_accuracies, arrays['UNION']['accuracy'],
      bootstrap, base_seed + 200),
    'domain_router_vs_union': comparison_summary(
      router_losses, arrays['UNION']['loss'],
      router_accuracies, arrays['UNION']['accuracy'],
      bootstrap, base_seed + 300),
    'specialist_loss_win_rates': win_rates,
    'specialist_loss_spearman': correlations,
    'specialist_diagnostics_paired_bootstrap_ci95': diagnostic_bootstrap,
    'bootstrap': bootstrap,
    'manifest_validation': manifest_report,
    'smoke_validation': validation,
    'outputs': outputs,
  }


def compute_query_split_summary(
        config,
        rows,
        results,
        outputs,
        checkpoint_report,
        manifest_report,
        validation,
        bootstrap,
        standard_summary):
  base_seed = bootstrap['seed'] + 10000
  union_a_loss = as_finite_array(
    results['UNION']['A']['loss'], 'UNION A losses')
  union_b_loss = as_finite_array(
    results['UNION']['B']['loss'], 'UNION B losses')
  union_a_accuracy = as_finite_array(
    results['UNION']['A']['accuracy'], 'UNION A accuracies')
  union_b_accuracy = as_finite_array(
    results['UNION']['B']['accuracy'], 'UNION B accuracies')
  split_oracle_mask = union_a_loss <= union_b_loss
  split_oracle_accuracy = np.where(
    split_oracle_mask, union_a_accuracy, union_b_accuracy)
  best_single_split = (
    'A' if union_a_loss.mean() <= union_b_loss.mean() else 'B')
  best_single_loss = float(min(union_a_loss.mean(), union_b_loss.mean()))
  split_oracle_loss = float(np.minimum(union_a_loss, union_b_loss).mean())
  noise_floor_gain = best_single_loss - split_oracle_loss

  crossfit_losses = as_finite_array(
    [row['cross_fitted_oracle_query_loss'] for row in rows],
    'cross-fitted oracle losses')
  crossfit_accuracies = as_finite_array(
    [row['cross_fitted_oracle_query_accuracy'] for row in rows],
    'cross-fitted oracle accuracies')
  crossfit_union_losses = as_finite_array(
    [row['cross_fitted_union_query_loss'] for row in rows],
    'cross-fitted UNION losses')
  crossfit_union_accuracies = as_finite_array(
    [row['cross_fitted_union_query_accuracy'] for row in rows],
    'cross-fitted UNION accuracies')
  in_sample_losses = as_finite_array(
    [row['in_sample_oracle_query_loss'] for row in rows],
    'query-split in-sample oracle losses')
  in_sample_accuracies = as_finite_array(
    [row['in_sample_oracle_query_accuracy'] for row in rows],
    'query-split in-sample oracle accuracies')

  model_summaries = {}
  for model_offset, name in enumerate(MODEL_NAMES):
    model_summaries[name] = {}
    for split_offset, split_name in enumerate(('A', 'B')):
      model_summaries[name][split_name] = metric_summary(
        results[name][split_name]['loss'],
        results[name][split_name]['accuracy'],
        bootstrap,
        base_seed + 100 * model_offset + 10 * split_offset)

  selection_counts = {
    'selected_on_a': dict(Counter(
      row['specialist_selected_on_a'] for row in rows)),
    'selected_on_b': dict(Counter(
      row['specialist_selected_on_b'] for row in rows)),
  }
  for key in selection_counts:
    counts = selection_counts[key]
    counts['NATURAL'] = int(counts.get('NATURAL', 0))
    counts['TECHNICAL'] = int(counts.get('TECHNICAL', 0))
    counts['NATURAL_rate'] = float(counts['NATURAL'] / len(rows))
    counts['TECHNICAL_rate'] = float(counts['TECHNICAL'] / len(rows))

  noise_floor = {
    'definition': (
      'Same UNION checkpoint and support set; class-balanced A/B query '
      'halves. Gain = best global split mean loss minus per-task minimum '
      'A/B loss, matching analyze_anchor_response.py.'),
    'union_a': {
      'mean_query_loss': float(union_a_loss.mean()),
      'mean_query_accuracy': float(union_a_accuracy.mean()),
    },
    'union_b': {
      'mean_query_loss': float(union_b_loss.mean()),
      'mean_query_accuracy': float(union_b_accuracy.mean()),
    },
    'loss_comparison': winner_rates(
      union_a_loss, union_b_loss, 'A', 'B'),
    'accuracy_comparison_higher_is_better': {},
    'loss_spearman': spearman_correlation(union_a_loss, union_b_loss),
    'best_single_split': best_single_split,
    'best_single_loss': best_single_loss,
    'oracle_loss': split_oracle_loss,
    'oracle_accuracy_corresponding_to_lower_loss': float(
      split_oracle_accuracy.mean()),
    'absolute_gain': noise_floor_gain,
    'relative_gain_percent': safe_relative_percent(
      noise_floor_gain, best_single_loss),
    'paired_bootstrap_ci95': query_split_noise_floor_bootstrap(
      union_a_loss, union_b_loss,
      bootstrap['n_resamples'], bootstrap['confidence'], base_seed + 1000),
  }
  accuracy_ties = np.isclose(
    union_a_accuracy, union_b_accuracy, rtol=0.0, atol=1e-12)
  noise_floor['accuracy_comparison_higher_is_better'] = {
    'a_wins': int(((union_a_accuracy > union_b_accuracy) &
                   ~accuracy_ties).sum()),
    'b_wins': int(((union_b_accuracy > union_a_accuracy) &
                   ~accuracy_ties).sum()),
    'ties': int(accuracy_ties.sum()),
  }

  crossfit_comparison = comparison_summary(
    crossfit_losses, crossfit_union_losses,
    crossfit_accuracies, crossfit_union_accuracies,
    bootstrap, base_seed + 2000)
  in_sample_comparison = comparison_summary(
    in_sample_losses, crossfit_union_losses,
    in_sample_accuracies, crossfit_union_accuracies,
    bootstrap, base_seed + 3000)
  primary_comparison = standard_summary['oracle_vs_union']
  primary_gain = primary_comparison['reference_minus_candidate_loss_gain']
  crossfit_gain = crossfit_comparison['reference_minus_candidate_loss_gain']
  in_sample_gain = in_sample_comparison['reference_minus_candidate_loss_gain']

  comparison_to_noise_floor = {
    'primary_specialist_oracle_gain': primary_gain,
    'primary_specialist_oracle_relative_gain_percent': (
      primary_comparison['relative_loss_gain_percent']),
    'query_split_in_sample_specialist_oracle_gain': in_sample_gain,
    'query_split_in_sample_specialist_oracle_relative_gain_percent': (
      in_sample_comparison['relative_loss_gain_percent']),
    'cross_fitted_specialist_oracle_gain': crossfit_gain,
    'cross_fitted_specialist_oracle_relative_gain_percent': (
      crossfit_comparison['relative_loss_gain_percent']),
    'query_split_noise_floor_gain': noise_floor_gain,
    'query_split_noise_floor_relative_gain_percent': (
      noise_floor['relative_gain_percent']),
    'primary_gain_minus_noise_floor': primary_gain - noise_floor_gain,
    'cross_fitted_gain_minus_noise_floor': crossfit_gain - noise_floor_gain,
    'primary_gain_to_noise_floor_ratio': (
      float(primary_gain / noise_floor_gain)
      if noise_floor_gain != 0.0 else None),
    'cross_fitted_gain_to_noise_floor_ratio': (
      float(crossfit_gain / noise_floor_gain)
      if noise_floor_gain != 0.0 else None),
  }

  return {
    'mode': 'class_balanced_query_split',
    'dataset': config['dataset'],
    'seed': int(config.get('seed', 0)),
    'n_tasks': len(rows),
    'n_tasks_per_cluster': int(config['tasks_per_cluster']),
    'clusters': config['clusters'],
    'protocol': {
      **protocol_summary(config, config['query_split']['n_query']),
      'n_query_per_split_per_class': int(
        config['query_split']['split_query']),
      'class_balanced_splits': True,
    },
    'checkpoints': checkpoint_report,
    'models_by_split': model_summaries,
    'union_query_split_noise_floor': noise_floor,
    'cross_fitted_oracle': {
      'definition': (
        'select NATURAL/TECHNICAL on A and evaluate on B, then select on B '
        'and evaluate on A; average both directions per task'),
      'selection_counts': selection_counts,
      **metric_summary(
        crossfit_losses, crossfit_accuracies, bootstrap, base_seed + 4000),
      'vs_union': crossfit_comparison,
    },
    'query_split_in_sample_oracle': {
      **metric_summary(
        in_sample_losses, in_sample_accuracies, bootstrap, base_seed + 5000),
      'vs_union': in_sample_comparison,
    },
    'specialist_gain_vs_query_split_noise_floor': comparison_to_noise_floor,
    'bootstrap': bootstrap,
    'manifest_validation': manifest_report,
    'smoke_validation': validation,
    'outputs': outputs,
  }


def validate_data_files(config):
  root_path = resolve_project_path(
    config['test'].get('root_path') or config.get('data_root'))
  missing = []
  paths = {}
  for cluster in CLUSTER_NAMES:
    for domain in config['clusters'][cluster]:
      path = os.path.join(root_path, domain + '.pickle')
      paths[domain] = path
      if not os.path.isfile(path):
        missing.append(path)
  if missing:
    raise FileNotFoundError(
      'missing Meta-Album domain pickle(s): {}'.format(', '.join(missing)))
  return {
    'root_path': root_path,
    'all_domain_pickles_present': True,
    'domain_pickle_paths': paths,
  }


def validate_completed_run(
        config,
        task_specs,
        rows,
        results,
        evaluated_ids,
        seen_metadata,
        query_split=False):
  expected_ids = [spec['task_id'] for spec in task_specs]
  expected_metadata = [
    (spec['task_id'], spec['cluster'], spec['domain'])
    for spec in task_specs
  ]
  if list(seen_metadata) != expected_metadata:
    raise RuntimeError('loader task metadata differs from the task schedule')
  for name in MODEL_NAMES:
    if evaluated_ids[name] != expected_ids:
      raise RuntimeError(
        '{} did not evaluate the exact scheduled task IDs'.format(name))

  expected_per_cluster = int(config['tasks_per_cluster'])
  balance = task_balance_report(task_specs)
  expected_cluster_counts = {
    'NATURAL': expected_per_cluster,
    'TECHNICAL': expected_per_cluster,
  }
  if balance['cluster_counts'] != expected_cluster_counts:
    raise RuntimeError(
      'expected 500+500 tasks, got {}'.format(balance['cluster_counts']))
  if len(rows) != 2 * expected_per_cluster:
    raise RuntimeError('output row count does not equal 1000')

  finite = True
  missing = False
  if query_split:
    for name in MODEL_NAMES:
      for split_name in ('A', 'B'):
        for metric in ('loss', 'accuracy'):
          values = results[name][split_name][metric]
          if len(values) != len(task_specs):
            missing = True
          finite = finite and bool(np.isfinite(values).all())
  else:
    for name in MODEL_NAMES:
      for metric in ('loss', 'accuracy'):
        values = results[name][metric]
        if len(values) != len(task_specs):
          missing = True
        finite = finite and bool(np.isfinite(values).all())
    for row in rows:
      missing = missing or any(row.get(field) is None
                               for field in CSV_FLOAT_FIELDS)
      finite = finite and all(
        np.isfinite(float(row[field])) for field in CSV_FLOAT_FIELDS)
  if not finite:
    raise FloatingPointError('completed analysis contains NaN or Inf')
  if missing:
    raise RuntimeError('completed analysis contains missing results')

  return {
    'checkpoints_distinct': True,
    'task_counts_are_500_plus_500': True,
    'task_balance': balance,
    'same_task_ids_used_by_all_three_models': True,
    'same_in_memory_episode_batch_used_by_all_three_models': True,
    'each_task_is_single_domain': True,
    'no_nan_or_infinite_results': True,
    'no_missing_results': True,
    'meta_test_manifest_enforced_by_dataset': True,
    'manifest_train_val_test_disjoint': True,
    'class_balanced_query_split_verified': bool(query_split),
  }


STANDARD_CSV_FIELDS = [
  'task_id', 'cluster', 'domain',
  'natural_query_loss', 'natural_query_accuracy',
  'technical_query_loss', 'technical_query_accuracy',
  'union_query_loss', 'union_query_accuracy',
  'loss_selected_specialist', 'specialist_loss_tie',
  'domain_specialist',
  'domain_router_query_loss', 'domain_router_query_accuracy',
  'oracle_query_loss', 'oracle_query_accuracy',
]

QUERY_SPLIT_CSV_FIELDS = [
  'task_id', 'cluster', 'domain',
  'natural_a_query_loss', 'natural_a_query_accuracy',
  'natural_b_query_loss', 'natural_b_query_accuracy',
  'technical_a_query_loss', 'technical_a_query_accuracy',
  'technical_b_query_loss', 'technical_b_query_accuracy',
  'union_a_query_loss', 'union_a_query_accuracy',
  'union_b_query_loss', 'union_b_query_accuracy',
  'specialist_selected_on_a', 'specialist_selected_on_b',
  'a_to_b_query_loss', 'a_to_b_query_accuracy',
  'b_to_a_query_loss', 'b_to_a_query_accuracy',
  'cross_fitted_oracle_query_loss',
  'cross_fitted_oracle_query_accuracy',
  'cross_fitted_union_query_loss',
  'cross_fitted_union_query_accuracy',
  'in_sample_oracle_query_loss', 'in_sample_oracle_query_accuracy',
  'union_split_oracle_query_loss',
]


def write_csv(path, rows, fieldnames):
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_json(path, value):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(value, f, indent=2, allow_nan=False)


def log_standard_summary(summary):
  utils.log('specialist oracle tasks: {}'.format(summary['n_tasks']))
  for name in MODEL_NAMES:
    metrics = summary['models'][name]
    utils.log('{}: loss={:.6f}, accuracy={:.6f}'.format(
      name, metrics['mean_query_loss'], metrics['mean_query_accuracy']))
  oracle = summary['per_task_oracle']
  oracle_comparison = summary['oracle_vs_union']
  router_comparison = summary['domain_router_vs_union']
  utils.log('oracle: loss={:.6f}, accuracy={:.6f}, UNION gain={:.6f}'.format(
    oracle['mean_query_loss'], oracle['mean_query_accuracy'],
    oracle_comparison['reference_minus_candidate_loss_gain']))
  utils.log('domain-router: loss={:.6f}, UNION gain={:.6f}'.format(
    summary['domain_router']['mean_query_loss'],
    router_comparison['reference_minus_candidate_loss_gain']))


def log_query_split_summary(summary):
  noise = summary['union_query_split_noise_floor']
  crossfit = summary['cross_fitted_oracle']
  utils.log('query-split tasks: {}'.format(summary['n_tasks']))
  utils.log('UNION query-split noise-floor gain={:.6f}'.format(
    noise['absolute_gain']))
  utils.log(
    'cross-fitted oracle: loss={:.6f}, accuracy={:.6f}, UNION gain={:.6f}'
    .format(
      crossfit['mean_query_loss'], crossfit['mean_query_accuracy'],
      crossfit['vs_union']['reference_minus_candidate_loss_gain']))


def run_analysis(config, args):
  validate_config(config)
  seed = int(args.seed if args.seed is not None else config.get('seed', 0))
  require_equal('resolved protocol seed', seed, 0)
  config['seed'] = seed
  bootstrap = bootstrap_config(config, args)
  seed_everything(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  manifest_report = validate_manifest(config)
  data_file_report = validate_data_files(config)
  manifest_report['data_files'] = data_file_report
  model_records, keep_on_gpu, checkpoint_report = load_analysis_models(
    config, args, device)

  task_specs = balanced_task_specs(config, seed)
  balance = task_balance_report(task_specs)
  require_equal('total task count', balance['total_tasks'], 1000)
  require_equal('NATURAL task count',
                balance['cluster_counts'].get('NATURAL'), 500)
  require_equal('TECHNICAL task count',
                balance['cluster_counts'].get('TECHNICAL'), 500)

  if args.validate_only:
    report = {
      'config_valid': True,
      'checkpoints': checkpoint_report,
      'manifest': manifest_report,
      'task_balance': balance,
      'note': (
        'validate-only checks inputs and the deterministic schedule; run '
        'without --validate-only to verify evaluated metrics and shared '
        'episode batches.'),
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return

  output_dir = resolve_project_path(
    args.output_dir or config.get('output_dir') or '.')
  os.makedirs(output_dir, exist_ok=True)
  outputs = {
    'specialist_oracle_tasks_csv': os.path.join(
      output_dir, 'specialist_oracle_tasks.csv'),
    'specialist_oracle_summary_json': os.path.join(
      output_dir, 'specialist_oracle_summary.json'),
    'query_split_tasks_csv': os.path.join(
      output_dir, 'query_split_tasks.csv'),
    'query_split_summary_json': os.path.join(
      output_dir, 'query_split_summary.json'),
  }

  standard_loader = make_loader(
    config, task_specs, int(config['test']['n_query']), seed, device)
  standard_results, standard_ids, standard_metadata = \
    evaluate_standard_tasks(
      config, model_records, standard_loader, device, keep_on_gpu)
  standard_rows = make_standard_rows(task_specs, standard_results)
  standard_validation = validate_completed_run(
    config, task_specs, standard_rows, standard_results,
    standard_ids, standard_metadata, query_split=False)
  standard_summary = compute_standard_summary(
    config, standard_rows, standard_results, outputs,
    checkpoint_report, manifest_report, standard_validation, bootstrap)
  write_csv(
    outputs['specialist_oracle_tasks_csv'], standard_rows,
    STANDARD_CSV_FIELDS)
  write_json(outputs['specialist_oracle_summary_json'], standard_summary)
  log_standard_summary(standard_summary)

  # Release the decoded n_query=15 datasets before constructing their
  # n_query=30 counterparts. This matters for the full 15-domain album.
  del standard_loader
  gc.collect()
  if device.type == 'cuda':
    torch.cuda.empty_cache()

  seed_everything(seed)
  query_loader = make_loader(
    config, task_specs, int(config['query_split']['n_query']), seed, device)
  query_results, query_ids, query_metadata = evaluate_query_split_tasks(
    config, model_records, query_loader, device, keep_on_gpu)
  query_rows = make_query_split_rows(task_specs, query_results)
  query_validation = validate_completed_run(
    config, task_specs, query_rows, query_results,
    query_ids, query_metadata, query_split=True)
  query_summary = compute_query_split_summary(
    config, query_rows, query_results, outputs,
    checkpoint_report, manifest_report, query_validation, bootstrap,
    standard_summary)

  comparison = query_summary[
    'specialist_gain_vs_query_split_noise_floor']
  standard_summary['specialist_gain_vs_query_split_noise_floor'] = comparison
  write_csv(outputs['query_split_tasks_csv'], query_rows,
            QUERY_SPLIT_CSV_FIELDS)
  write_json(outputs['query_split_summary_json'], query_summary)
  write_json(outputs['specialist_oracle_summary_json'], standard_summary)
  log_query_split_summary(query_summary)

  utils.log('specialist task CSV: {}'.format(
    outputs['specialist_oracle_tasks_csv']))
  utils.log('specialist summary JSON: {}'.format(
    outputs['specialist_oracle_summary_json']))
  utils.log('query-split task CSV: {}'.format(
    outputs['query_split_tasks_csv']))
  utils.log('query-split summary JSON: {}'.format(
    outputs['query_split_summary_json']))


def parse_args():
  parser = argparse.ArgumentParser(
    description='Balanced Meta-Album specialist/router/oracle analysis')
  parser.add_argument('--config', required=True,
                      help='specialist-oracle analysis YAML')
  parser.add_argument('--gpu', type=str, default='0',
                      help='GPU device number(s), or -1 for CPU')
  parser.add_argument('--seed', type=int, default=None,
                      help='override config seed (protocol default: 0)')
  parser.add_argument('--output-dir', type=str, default=None,
                      help='override output directory')
  parser.add_argument('--bootstrap-resamples', type=int, default=None,
                      help='override paired bootstrap resample count')
  parser.add_argument('--efficient', action='store_true',
                      help='enable gradient checkpointing')
  parser.add_argument(
    '--keep-models-on-gpu',
    action=argparse.BooleanOptionalAction,
    default=None,
    help='keep all three ConvNet4 models on GPU')
  parser.add_argument(
    '--validate-only', action='store_true',
    help='validate checkpoints, manifest, and the 500+500 task schedule')
  return parser.parse_args()


if __name__ == '__main__':
  cli_args = parse_args()
  with open(cli_args.config, 'r', encoding='utf-8') as f:
    analysis_config = yaml.load(f, Loader=yaml.FullLoader)
  analysis_config = normalized_config(analysis_config)
  if len(cli_args.gpu.split(',')) > 1:
    analysis_config['_parallel'] = True
    analysis_config['_gpu'] = cli_args.gpu
  utils.set_gpu(cli_args.gpu)
  run_analysis(analysis_config, cli_args)
