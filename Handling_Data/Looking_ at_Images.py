import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# print(tf.__version__)

# we are classifying types of flowers
# the flowers contain five sub dirs

# flowers_photos/
#   daisy/
#   dandelion/
#   roses/
#   sunflowers/
#   tulips/

import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                      fname='flower_photos',
                                      untar=True) # should file be extracted

data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print('num images', image_count)

# roses = list(data_dir.glob('roses/*'))
# image = (PIL.Image.open(str(roses[0])))
# image.show()

# tulips = list(data_dir.glob('daisy/*'))
# image = (PIL.Image.open(str(tulips[3])))
# image.show()

# Loading data using keras utility

# Create dataset
batch_size = 32
img_height = 180
img_width =180

# it is good practice to use a validation split when developing a model (80/20 usually)
# Load data off disk
# Breaking up training
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Breaking up training
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset='validation',
    seed = 123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

'''I need to learn how to split image data and use the labels as validation'''

class_names = train_ds.class_names
print(class_names)

# Visualisation

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
        print('labels', class_names[labels[i]])
        print('object', labels[i])
        print('type', type(images))

# plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# convert RGB valuse form 0-250 to 0-1

normalization_layer = tf.keras.layers.Rescaling(1./255)

# 2 ways to use layer.
# 1 apply to dataset by Dataset.map

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

# or you can include the layer inside your model definition to simplify your deployment

# configure the dataset for performance
# buffer prefetching to yield data with out having I/O become blocking.

# Dataset.cache keeps images in memory after they're loaded during first epoch.
# keeps from dataset bottleneck. if dataset is too large to fit into memory
# also option to use this method to create a perfromant on-disk cache.

# Dataset.prefetch overlaps data preprocessing and model execution while training

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Train model

num_classes = 5

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1/255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Rescaling(1/255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Rescaling(1/255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3
)

## Using tf.data for finer control (this is the way I like to do models)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(5):
    print(f.numpy())

# tree structure of teh files can be used to compile a class_names list

class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != 'LICENSE.txt']))
print(class_names)

#dplit dataset into training and validation sets
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# These two lines do the same thing
# print(tf.data.experimental.cardinality(train_ds).numpy())
print(len(train_ds))
print(len(val_ds))


# write a short functino that converts a file path to an (img, label) pair
# convert the path to a list of path components
# the second to last is the class directory
# Integer encode the label

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

# convert compressed string to a 3D unit8 tensor
# resize image to desired size
def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    label = get_label(file_path)
    # Load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# use Dataset.map to create a dataset of image, label pairs:

# set 'num_parrellel_calls' so multiple images are loaded in parallel
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
    print('Image shape: ', image.numpy().shape)
    print('Label: ', label.numpy())

# configure datase for preformance
# well shuffled
# well batched
# batches to be available as soon a possible

def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)


# visualize data
image_batch, label_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = label_batch[i]
    plt.title(class_names[label])
    plt.axis('off')


model.fit(
    train_ds,
    validation_data = val_ds,
    epochs=3
)


# final note this dataset can be pulled from tf datasets
