import argparse
import glob
import os
import pickle
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image
import torch

from datasets.transforms import get_transform
import models
import utils


DEFAULT_DATA_ROOT = (
  '/content/drive/MyDrive/DOKTORA/meta_learning/datasets/album')
DEFAULT_OUTPUT = 'save/domain_similarity/meta_album_features_maml.npz'


def find_pickle_files(data_root):
  if not os.path.isdir(data_root):
    raise FileNotFoundError(
      'Meta-Album data directory not found: {}'.format(data_root))

  pickle_files = sorted(
    glob.glob(os.path.join(data_root, '*.pickle')),
    key=lambda path: os.path.basename(path).lower())
  if not pickle_files:
    raise FileNotFoundError(
      'no *.pickle files found in Meta-Album data directory: {}'.format(
        data_root))
  return pickle_files


def load_meta_album_pickle(path):
  try:
    with open(path, 'rb') as f:
      pack = pickle.load(f)
  except Exception as exc:
    raise RuntimeError(
      'failed to read Meta-Album pickle {}: {}'.format(path, exc)) from exc

  if not isinstance(pack, dict):
    raise ValueError(
      'invalid Meta-Album pickle {}: expected a dict'.format(path))
  missing = [key for key in ('data', 'labels') if key not in pack]
  if missing:
    raise ValueError(
      'invalid Meta-Album pickle {}: missing key(s) {}'.format(
        path, ', '.join(missing)))

  images = np.asarray(pack['data'])
  labels = np.asarray(pack['labels'])
  if images.ndim != 4 or images.shape[-1] != 3:
    raise ValueError(
      'invalid data in {}: expected [N, H, W, 3], got {}'.format(
        path, images.shape))
  if images.dtype != np.uint8:
    raise ValueError(
      'invalid data dtype in {}: expected uint8, got {}'.format(
        path, images.dtype))
  if labels.ndim != 1:
    labels = labels.reshape(-1)
  if len(images) != len(labels):
    raise ValueError(
      'invalid Meta-Album pickle {}: {} images but {} labels'.format(
        path, len(images), len(labels)))
  if len(images) == 0:
    raise ValueError('Meta-Album pickle contains no images: {}'.format(path))

  try:
    class_labels = labels.astype(str)
  except (TypeError, ValueError) as exc:
    raise ValueError(
      'invalid class labels in {}: labels must be string-convertible'.format(
        path)) from exc
  return images, class_labels


def get_checkpoint_preprocessing(checkpoint):
  config = checkpoint.get('config')
  if not isinstance(config, dict):
    raise ValueError(
      'checkpoint has no config dict; cannot recover Meta-Album preprocessing')
  if config.get('dataset') != 'meta-album':
    raise ValueError(
      'checkpoint dataset must be meta-album, got {!r}'.format(
        config.get('dataset')))

  train_config = config.get('train')
  if not isinstance(train_config, dict):
    raise ValueError(
      'checkpoint config has no train block; cannot recover preprocessing')

  image_size = int(train_config.get('image_size', 84))
  normalization = bool(train_config.get('normalization', True))
  transform_name = train_config.get('transform')

  # These are exactly the normalization branches in datasets/meta_album.py.
  if normalization:
    norm_params = {
      'mean': [0.485, 0.456, 0.406],
      'std': [0.229, 0.224, 0.225],
    }
  else:
    norm_params = {
      'mean': [0.0, 0.0, 0.0],
      'std': [1.0, 1.0, 1.0],
    }

  transform = get_transform(transform_name, image_size, norm_params)
  return transform, {
    'image_size': image_size,
    'normalization': normalization,
    'transform': transform_name,
  }


def load_model(checkpoint_path, device):
  if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
      'MAML checkpoint not found: {}'.format(checkpoint_path))

  try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
  except Exception as exc:
    raise RuntimeError(
      'failed to load MAML checkpoint {}: {}'.format(
        checkpoint_path, exc)) from exc

  required = (
    'encoder',
    'encoder_args',
    'encoder_state_dict',
    'classifier',
    'classifier_args',
    'classifier_state_dict',
  )
  missing = [key for key in required if key not in checkpoint]
  if missing:
    raise ValueError(
      'invalid MAML checkpoint {}: missing key(s) {}'.format(
        checkpoint_path, ', '.join(missing)))

  model = models.load(checkpoint, load_clf=True)
  model.to(device)
  model.eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  return model, checkpoint


def get_encoder_episode(model):
  for module in model.encoder.modules():
    is_episodic = getattr(module, 'is_episodic', None)
    if callable(is_episodic) and is_episodic() and \
        getattr(module, 'track_running_stats', False):
      return 0
  return None


def extract_domain_features(
        images,
        transform,
        model,
        device,
        batch_size,
        encoder_episode):
  feature_batches = []
  with torch.no_grad():
    for start in range(0, len(images), batch_size):
      end = min(start + batch_size, len(images))
      try:
        image_batch = torch.stack([
          transform(Image.fromarray(images[index]))
          for index in range(start, end)
        ])
      except Exception as exc:
        raise RuntimeError(
          'failed to preprocess image index {}: {}'.format(start, exc)) from exc

      image_batch = image_batch.to(device, non_blocking=(device.type == 'cuda'))
      features = model.encoder(
        image_batch,
        params=None,
        episode=encoder_episode)
      if not torch.is_tensor(features) or features.ndim != 2:
        shape = getattr(features, 'shape', None)
        raise RuntimeError(
          'model.encoder must return [batch, feature_dim], got {}'.format(
            shape))
      if features.size(0) != end - start:
        raise RuntimeError(
          'encoder returned {} features for a batch of {}'.format(
            features.size(0), end - start))
      feature_batches.append(features.detach().float().cpu().numpy())

  domain_features = np.concatenate(feature_batches, axis=0)
  return np.asarray(domain_features, dtype=np.float32)


def save_features(
        output_path,
        features,
        dataset_codes,
        class_labels,
        image_indices,
        checkpoint_path):
  if not output_path.lower().endswith('.npz'):
    raise ValueError('output path must end with .npz: {}'.format(output_path))
  output_dir = os.path.dirname(os.path.abspath(output_path))
  os.makedirs(output_dir, exist_ok=True)
  np.savez_compressed(
    output_path,
    features=np.asarray(features, dtype=np.float32),
    dataset_codes=np.asarray(dataset_codes, dtype=str),
    class_labels=np.asarray(class_labels, dtype=str),
    image_indices=np.asarray(image_indices, dtype=np.int64),
    checkpoint=np.asarray(os.path.abspath(checkpoint_path), dtype=str),
  )


def main(args):
  if args.batch_size <= 0:
    raise ValueError('--batch-size must be positive')

  device = torch.device(
    'cuda' if args.gpu != '-1' and torch.cuda.is_available() else 'cpu')
  checkpoint_path = os.path.abspath(args.checkpoint)
  model, checkpoint = load_model(checkpoint_path, device)
  transform, preprocessing = get_checkpoint_preprocessing(checkpoint)
  encoder_episode = get_encoder_episode(model)
  pickle_files = find_pickle_files(args.data_root)

  print('device: {}'.format(device))
  print('checkpoint: {}'.format(checkpoint_path))
  print('encoder feature function: model.encoder(x, params=None, episode={})'.format(
    encoder_episode))
  print('preprocessing: image_size={image_size}, normalization={normalization}, '
        'transform={transform}'.format(**preprocessing))
  print('gradient transport: disabled (encoder-only inference)')

  all_features = []
  all_dataset_codes = []
  all_class_labels = []
  all_image_indices = []
  feature_dim = None

  for pickle_path in pickle_files:
    dataset_code = os.path.splitext(os.path.basename(pickle_path))[0]
    images, class_labels = load_meta_album_pickle(pickle_path)
    try:
      domain_features = extract_domain_features(
        images,
        transform,
        model,
        device,
        args.batch_size,
        encoder_episode)
    except Exception as exc:
      raise RuntimeError(
        'feature extraction failed for {} ({}): {}'.format(
          dataset_code, pickle_path, exc)) from exc

    current_dim = int(domain_features.shape[1])
    if feature_dim is None:
      feature_dim = current_dim
    elif current_dim != feature_dim:
      raise RuntimeError(
        'feature dimension mismatch for {}: expected {}, got {}'.format(
          dataset_code, feature_dim, current_dim))

    n_images = len(images)
    all_features.append(domain_features)
    all_dataset_codes.append(np.asarray([dataset_code] * n_images, dtype=str))
    all_class_labels.append(class_labels)
    all_image_indices.append(np.arange(n_images, dtype=np.int64))
    print('{}: images={}, classes={}, feature_dim={}'.format(
      dataset_code,
      n_images,
      len(np.unique(class_labels)),
      current_dim))

  features = np.concatenate(all_features, axis=0).astype(np.float32, copy=False)
  dataset_codes = np.concatenate(all_dataset_codes)
  class_labels = np.concatenate(all_class_labels)
  image_indices = np.concatenate(all_image_indices)
  total_images = features.shape[0]

  if not (
      len(dataset_codes) == len(class_labels) == len(image_indices) ==
      total_images):
    raise RuntimeError('output metadata lengths do not match feature count')

  save_features(
    args.output,
    features,
    dataset_codes,
    class_labels,
    image_indices,
    checkpoint_path)
  print('total images: {}'.format(total_images))
  print('total datasets: {}'.format(len(pickle_files)))
  print('feature matrix shape: {}'.format(features.shape))
  print('output: {}'.format(os.path.abspath(args.output)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=(
      'Extract fixed MAML encoder features from all Meta-Album pickles.'))
  parser.add_argument(
    '--data-root',
    default=DEFAULT_DATA_ROOT,
    help='directory containing Meta-Album *.pickle files')
  parser.add_argument(
    '--checkpoint',
    required=True,
    help='trained MAML checkpoint path')
  parser.add_argument(
    '--output',
    default=DEFAULT_OUTPUT,
    help='output .npz path')
  parser.add_argument(
    '--batch-size',
    type=int,
    default=256,
    help='encoder inference batch size (default: 256)')
  parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='GPU device number, or -1 for CPU (default: 0)')
  args = parser.parse_args()

  if args.gpu == '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  utils.set_gpu(args.gpu)
  main(args)
