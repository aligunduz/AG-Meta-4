import json
import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import get_transform


_MANIFEST_SPLITS = ('train', 'val', 'test')
_SPLIT_TO_MANIFEST_KEY = {
  'meta-train': 'train',
  'meta-val': 'val',
  'meta-test': 'test',
}


def _preview_classes(classes, limit=10):
  classes = sorted(classes)
  preview = ', '.join(classes[:limit])
  if len(classes) > limit:
    preview += ', ... ({} total)'.format(len(classes))
  return preview


def _load_class_split_manifest(manifest_path, domains):
  if not os.path.isfile(manifest_path):
    raise FileNotFoundError(
      'Meta-Album class split manifest not found: {}'.format(manifest_path))

  try:
    with open(manifest_path, 'r', encoding='utf-8') as f:
      manifest = json.load(f)
  except json.JSONDecodeError as exc:
    raise ValueError(
      'Invalid Meta-Album class split manifest JSON at {}: {}'.format(
        manifest_path, exc)) from exc

  if not isinstance(manifest, dict):
    raise ValueError(
      'Meta-Album class split manifest must contain a JSON object: {}'.format(
        manifest_path))
  domain_splits = {}
  for domain in domains:
    if domain not in manifest:
      raise KeyError(
        "Meta-Album dataset key '{}' is missing from class split manifest {}"
        .format(domain, manifest_path))
    entry = manifest[domain]

    if not isinstance(entry, dict):
      raise ValueError(
        "Meta-Album manifest entry for dataset '{}' must be an object: {}"
        .format(domain, manifest_path))

    normalized = {}
    for split in _MANIFEST_SPLITS:
      if split not in entry or not isinstance(entry[split], list):
        raise ValueError(
          "Meta-Album manifest dataset '{}' split '{}' must be a JSON array "
          'in {}'.format(domain, split, manifest_path))
      names = [str(value) for value in entry[split]]
      if len(names) != len(set(names)):
        raise ValueError(
          "Meta-Album manifest dataset '{}' split '{}' contains duplicate "
          'classes'.format(domain, split))
      normalized[split] = names

    split_sets = {key: set(values) for key, values in normalized.items()}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
      overlap = split_sets[left] & split_sets[right]
      if overlap:
        raise ValueError(
          "Meta-Album manifest dataset '{}' has overlapping {}/{} classes: {}"
          .format(domain, left, right, _preview_classes(overlap)))
    domain_splits[domain] = normalized

  return domain_splits


def _load_domain(root_path, domain):
  """
  Loads one pre-packed Meta-Album domain (see tools/pack_meta_album.py),
  a pickle with keys 'data' (list/array of HxWxC uint8 images) and
  'labels' (category name per image, same order).
  """
  pickle_file = os.path.join(root_path, domain + '.pickle')
  assert os.path.isfile(pickle_file), (
    'Meta-Album domain "{}" pickle not found at {}. Run '
    'tools/pack_meta_album.py first.'.format(domain, pickle_file))
  with open(pickle_file, 'rb') as f:
    pack = pickle.load(f)
  data = [Image.fromarray(x) for x in pack['data']]
  labels = np.array(pack['labels'])
  return data, labels


@register('meta-album')
class MetaAlbumCrossDomain(Dataset):
  """
  Cross-domain episodic few-shot dataset over Meta-Album.

  By default, meta-train / meta-val / meta-test are defined by disjoint
  Meta-Album datasets, preserving the original domain-holdout behavior.
  When class_split_manifest is supplied, the same datasets may be used in
  every phase and each dataset's class pool is filtered to the manifest's
  train / val / test partition. Each episode still samples its n_way classes
  from one randomly chosen dataset, so a task never mixes datasets.

  Expects root_path/<domain>.pickle for every domain listed in `domains`,
  produced by tools/pack_meta_album.py.

  Args:
    root_path (str): folder containing one <domain>.pickle per domain.
    domains (list of str): Meta-Album domain codes to draw episodes from
      for this split, e.g. ['BRD', 'FLW', 'PLT_VIL'].
    class_split_manifest (str, optional): JSON file containing train, val and
      test class lists for every configured dataset. If omitted, all classes
      in each configured dataset are used, preserving the old behavior. The
      canonical layout is {"BRD": {"train": [...], "val": [...],
      "test": [...]}, ...}.
    domain_groups (list of lists, optional): groups of configured domains used
      for group-balanced episode sampling. When supplied, an episode first
      samples one group uniformly and then one domain uniformly inside that
      group. Every configured domain must appear exactly once. If omitted,
      domains are sampled uniformly as before.
  """
  def __init__(self, root_path, split='train', domains=None, image_size=84,
               normalization=True, transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15,
               class_split_manifest=None, domain_groups=None):
    super(MetaAlbumCrossDomain, self).__init__()
    assert domains, (
      "meta-album requires a 'domains' list in the config's '{}' block, "
      "e.g. domains: [BRD, FLW, PLT_VIL]".format(split))

    self.root_path = root_path
    self.split = split
    self.domains = list(domains)
    self.image_size = image_size
    self.n_batch = n_batch
    self.n_episode = n_episode
    self.n_way = n_way
    self.n_shot = n_shot
    self.n_query = n_query
    self.class_split_manifest = class_split_manifest
    self.domain_groups = None
    self.domain_group_indices = None

    if domain_groups is not None:
      if not isinstance(domain_groups, (list, tuple)) or not domain_groups:
        raise ValueError(
          'meta-album domain_groups must be a non-empty list of groups')

      normalized_groups = []
      grouped_domains = []
      for group_index, group in enumerate(domain_groups):
        if not isinstance(group, (list, tuple)) or not group:
          raise ValueError(
            'meta-album domain_groups[{}] must be a non-empty list'.format(
              group_index))
        normalized_group = [str(domain) for domain in group]
        if len(normalized_group) != len(set(normalized_group)):
          raise ValueError(
            'meta-album domain_groups[{}] contains duplicate domains'.format(
              group_index))
        normalized_groups.append(normalized_group)
        grouped_domains.extend(normalized_group)

      if len(grouped_domains) != len(set(grouped_domains)):
        raise ValueError(
          'meta-album domain_groups contains a domain in multiple groups')

      configured_domains = set(self.domains)
      grouped_domain_set = set(grouped_domains)
      missing = configured_domains - grouped_domain_set
      unknown = grouped_domain_set - configured_domains
      if missing or unknown:
        raise ValueError(
          'meta-album domain_groups must contain every configured domain '
          'exactly once; missing={}, unknown={}'.format(
            sorted(missing), sorted(unknown)))

      domain_to_index = {
        domain: index for index, domain in enumerate(self.domains)
      }
      self.domain_groups = normalized_groups
      self.domain_group_indices = [
        tuple(domain_to_index[domain] for domain in group)
        for group in normalized_groups
      ]

    manifest_split = None
    manifest_domain_splits = None
    if class_split_manifest is not None:
      if split not in _SPLIT_TO_MANIFEST_KEY:
        raise ValueError(
          "Meta-Album class split manifest requires split to be one of {}, "
          "got '{}'".format(sorted(_SPLIT_TO_MANIFEST_KEY), split))
      manifest_split = _SPLIT_TO_MANIFEST_KEY[split]
      manifest_domain_splits = _load_class_split_manifest(
        class_split_manifest, self.domains)

    if normalization:
      self.norm_params = {'mean': [0.485, 0.456, 0.406],
                          'std':  [0.229, 0.224, 0.225]}   # ImageNet statistics
    else:
      self.norm_params = {'mean': [0., 0., 0.],
                          'std':  [1., 1., 1.]}

    self.transform = get_transform(transform, image_size, self.norm_params)
    self.val_transform = get_transform(
      val_transform, image_size, self.norm_params)

    def convert_raw(x):
      mean = torch.tensor(self.norm_params['mean']).view(3, 1, 1).type_as(x)
      std = torch.tensor(self.norm_params['std']).view(3, 1, 1).type_as(x)
      return x * std + mean
    self.convert_raw = convert_raw

    # per-domain decoded images and per-domain, per-class index arrays
    self.domain_data = []
    self.domain_catlocs = []
    self.n_classes = 0
    for domain in self.domains:
      data, labels = _load_domain(root_path, domain)
      all_cat_keys = sorted(np.unique(labels))
      cat_keys = all_cat_keys
      if manifest_domain_splits is not None:
        actual_by_name = {
          str(class_name): class_name for class_name in all_cat_keys
        }
        if len(actual_by_name) != len(all_cat_keys):
          raise ValueError(
            "Meta-Album dataset '{}' contains class labels that collide after "
            'string normalization'.format(domain))

        domain_splits = manifest_domain_splits[domain]
        manifest_classes = set().union(
          *(set(domain_splits[key]) for key in _MANIFEST_SPLITS))
        actual_classes = set(actual_by_name)
        unknown = manifest_classes - actual_classes
        unassigned = actual_classes - manifest_classes
        if unknown:
          raise ValueError(
            "Meta-Album manifest dataset '{}' contains classes not found in "
            'the pickle: {}'.format(domain, _preview_classes(unknown)))
        if unassigned:
          raise ValueError(
            "Meta-Album manifest dataset '{}' does not assign these pickle "
            'classes to train/val/test: {}'.format(
              domain, _preview_classes(unassigned)))

        cat_keys = [
          actual_by_name[class_name]
          for class_name in domain_splits[manifest_split]
        ]

      if len(cat_keys) < n_way:
        raise ValueError(
          "Meta-Album dataset '{}' split '{}' only has {} classes, need at "
          'least n_way={} to sample an episode.'.format(
            domain, split, len(cat_keys), n_way))
      catlocs = tuple(
        np.argwhere(labels == c).reshape(-1) for c in cat_keys)
      self.domain_data.append(data)
      self.domain_catlocs.append(catlocs)
      self.n_classes += len(cat_keys)
      print('{} | split={} | kullanılan sınıf={} | toplam sınıf={}'.format(
        domain, split, len(cat_keys), len(all_cat_keys)))

    if self.domain_group_indices is None:
      print('Meta-Album domain sampling: domain-uniform')
    else:
      print(
        'Meta-Album domain sampling: group-uniform | groups={}'.format(
          self.domain_groups))

  def __len__(self):
    return self.n_batch * self.n_episode

  def _sample_domain_idx(self):
    if self.domain_group_indices is None:
      return int(np.random.randint(len(self.domains)))

    group_idx = int(np.random.randint(len(self.domain_group_indices)))
    group = self.domain_group_indices[group_idx]
    within_group_idx = int(np.random.randint(len(group)))
    return group[within_group_idx]

  def __getitem__(self, index):
    domain_idx = self._sample_domain_idx()
    data = self.domain_data[domain_idx]
    catlocs = self.domain_catlocs[domain_idx]
    cats = np.random.choice(len(catlocs), self.n_way, replace=False)

    shot, query = [], []
    for c in cats:
      idx_list = np.random.choice(
        catlocs[c], self.n_shot + self.n_query, replace=False)
      shot_idx, query_idx = idx_list[:self.n_shot], idx_list[-self.n_query:]
      shot.append(torch.stack([self.transform(data[i]) for i in shot_idx]))
      query.append(torch.stack([
        self.val_transform(data[i]) for i in query_idx]))

    shot = torch.cat(shot, dim=0)             # [n_way * n_shot, C, H, W]
    query = torch.cat(query, dim=0)           # [n_way * n_query, C, H, W]
    cls = torch.arange(self.n_way)[:, None]
    shot_labels = cls.repeat(1, self.n_shot).flatten()    # [n_way * n_shot]
    query_labels = cls.repeat(1, self.n_query).flatten()  # [n_way * n_query]

    return shot, query, shot_labels, query_labels
