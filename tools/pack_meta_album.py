import csv
import os
import pickle

import numpy as np
from PIL import Image


#DOMAINS = ['BRD', 'FLW', 'PLT_VIL', 'SPT', 'PLK', 'RESISC']
DOMAINS = [
    'DOG',
    'AWA',
    'INS_2',
    'INS',
    'PLT_NET',
    'FNG',
    'MED_LF',
    'PLT_DOC',
    'BCT',
    'PNU',
    'PRT',
    'RSICB',
    'RSD',
    'CRS',
    'APL',
    'BTS',
    'TEX',
    'TEX_DTD',
    'TEX_ALOT',
    'ACT_40',
    'ACT_410',
    'MD_MIX',
    'MD_5_BIS',
    'MD_6',
]


def convert_domain(src_dir, domain, save_dir, image_size=84):
  """
  Meta-Album resmi formatı:
    src_dir/<domain>/labels.csv   (FILE_NAME, CATEGORY, SUPER_CATEGORY)
    src_dir/<domain>/images/*.jpg

  Tek bir <domain>.pickle dosyasına paketler: {'data': uint8 array,
  'labels': category adı array'i}. Etiketler datasets/meta_album.py
  tarafında 0..n_classes-1'e remap edilir (mini_imagenet.py ile aynı desen).
  """
  domain_dir = os.path.join(src_dir, domain)
  labels_csv = os.path.join(domain_dir, 'labels.csv')
  images_dir = os.path.join(domain_dir, 'images')

  with open(labels_csv, 'r', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

  print('[{}] {} görüntü işleniyor...'.format(domain, len(rows)))
  all_images, all_labels = [], []
  for row in rows:
    img_path = os.path.join(images_dir, row['FILE_NAME'])
    try:
      img = Image.open(img_path).convert('RGB').resize(
        (image_size, image_size))
      all_images.append(np.array(img))
      all_labels.append(row['CATEGORY'])
    except Exception as e:
      print('  atlandı {}: {}'.format(img_path, e))

  pack = {
    'data': np.array(all_images, dtype=np.uint8),
    'labels': np.array(all_labels),
  }

  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, domain + '.pickle')
  with open(save_path, 'wb') as f:
    pickle.dump(pack, f)

  n_classes = len(np.unique(all_labels))
  print('  kaydedildi {} ({} görüntü, {} class)'.format(
    save_path, len(all_labels), n_classes))


if __name__ == '__main__':
  # Meta-Album'ü OpenML'den indirip ./materials/_meta_album_raw/<DOMAIN>/
  # altına yerleştirdikten sonra (labels.csv + images/) bu script'i çalıştır.
  # Çıktı, diğer dataset'lerle aynı Drive/DATASET_NAME düzenine uyacak
  # şekilde 'album' klasörüne yazılır (config'te dataset: meta-album ->
  # 'meta-' önekinin auto-default tarafından silinmesiyle 'album' olur).
  src_dir = './materials/_meta_album_raw'
  save_dir = '/content/drive/MyDrive/DOKTORA/meta_learning/datasets/album'

  for domain in DOMAINS:
    convert_domain(src_dir, domain, save_dir)
