import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from .datasets import register
from .transforms import get_transform


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

  Unlike the single-dataset loaders (mini-imagenet, cub200, ...), the
  meta-train / meta-val / meta-test split here is defined by *disjoint
  Meta-Album domains* (e.g. meta-train on ['BRD', 'FLW', 'PLT_VIL'],
  meta-test on entirely unseen domains ['PLK', 'RESISC']) rather than by
  holding out classes within one domain. Each episode samples its n_way
  classes from a single, randomly chosen domain so that a task never mixes
  two unrelated domains.

  Expects root_path/<domain>.pickle for every domain listed in `domains`,
  produced by tools/pack_meta_album.py.

  Args:
    root_path (str): folder containing one <domain>.pickle per domain.
    domains (list of str): Meta-Album domain codes to draw episodes from
      for this split, e.g. ['BRD', 'FLW', 'PLT_VIL'].
  """
  def __init__(self, root_path, split='train', domains=None, image_size=84,
               normalization=True, transform=None, val_transform=None,
               n_batch=200, n_episode=4, n_way=5, n_shot=1, n_query=15):
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
      cat_keys = sorted(np.unique(labels))
      assert len(cat_keys) >= n_way, (
        "Meta-Album domain '{}' only has {} categories, need at least "
        "n_way={} to sample an episode.".format(domain, len(cat_keys), n_way))
      catlocs = tuple(
        np.argwhere(labels == c).reshape(-1) for c in cat_keys)
      self.domain_data.append(data)
      self.domain_catlocs.append(catlocs)
      self.n_classes += len(cat_keys)

  def __len__(self):
    return self.n_batch * self.n_episode

  def __getitem__(self, index):
    domain_idx = np.random.randint(len(self.domains))
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
