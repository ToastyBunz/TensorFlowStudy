import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pathlib


fonts_zip   = tf.keras.utils.get_file(
    'fonts.zip',
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
    cache_dir='.', cache_subdir='fonts',
    extract=True
)

import pathlib
font_csvs = sorted(str(p) for p in pathlib.Path('fonts').glob("*.csv"))

print(font_csvs[:10])
print(len(font_csvs))

fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern= 'fonts/*.csv',
    batch_size=10,
    num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000
)

for features in fonts_ds.take(1):
    for i, (name, value) in enumerate(features.items()):
        if i >15:
            break
        print(f"{name:20s}: {value}")
    print('...')
    print(f"[total: {len(features)} features")

# dont want to work with pixel columns, so we need ot pack them into image tensors

import re

def make_images(features):
    image = [None]*400
    new_feats = {}

    for name, value in features.items():
        match = re.match('r(\d+)c(\d+)', name)
        if match:
          image[int(match.group(1))*20+int(match.group(2))] = value
        else:
          new_feats[name] = value

    image = tf.stack(image, axis=0)
    image = tf.reshape(image, [20, 20, -1])
    new_feats['image'] = image

    return new_feats

# apply function ot each batch in dataset

fonts_image_ds = fonts_ds.map(make_images)

for features in fonts_image_ds.take(1):
    break

from matplotlib import pyplot as plt
plt.figure(figsize=(6, 6), dpi=120)

for n in range(9):
    plt.subplot(3, 3, n+1)
    plt.imshow(features['image'][..., n])
    plt.title(chr(features['m_label'][n]))
    plt.axis('off')

plt.show()
