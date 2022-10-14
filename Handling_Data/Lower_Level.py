import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pathlib
import heartpatients as pd
from keras import layers

np.set_printoptions(precision=3, suppress=True)

# Titanic dataset is about the passengers and the goal of the model is to predict who survived

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
# print(titanic.head())

titanic_features = titanic.copy()
titanic_labels = titanic.pop('survived') # creates pandas series with only the "survived" column
# print(titanic_labels)

titanic_features.pop('survived')


# because of different data types and ranges cannot stack features into numpy array
# and pass it to a Sequential.
# Each column must be handled individually

# could preprocess categorical terms into numerical then train but if model is exported the preprocessing
# in not saved with it (in reality with a little bit of coding this is not a big deal)

# instead for learning sake we are going to use symbolic tensors

# Create a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)

# preform calucualtion using input
result = 2*input+1

# the result doesn't have a value
# print(result)

calc = tf.keras.Model(inputs=input, outputs=result)
# print(calc(1).numpy())
# print(calc(2).numpy())

inputs = {}

for name, column in titanic_features.items():
    dtype = column.dtype
    if dtype == object:
        dtype = tf.string
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

print(inputs) # This tells the dtype for each input column

# first order of business is to take numeric inputs and normalize

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

print(all_numeric_inputs)

# collect all symbolic, to concat later

preprocessed_inputs = [all_numeric_inputs]

# for string inputs use tf.keras.layers.StringLookup to map strings to integer indicies
# Use tf.keras.layers.CatagoryEncoding to convert the indexes into float32 data appropriate for model

# default settings CategoryEncoding creat one hot
# Embedding would also work

for name, input in inputs.items():
    if input.dtype == tf.float32:
        continue

    lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
    one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

    x = lookup(input)
    x = one_hot(x)
    preprocessed_inputs.append(x)

preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)
titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)
# tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

# converting to a dictionary of tensors
titanic_features_dict = {name: np.array(value)
                         for name, value in titanic_features.items()}

features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)

# now building the model

def titanic_model(preprocessing_head, inputs):
    body = tf.keras.Sequential([
        layers.Dense(64),
        layers.Dense(1),
    ])

    preprocessed_inputs = preprocessing_head(inputs)
    result = body(preprocessed_inputs)
    model = tf.keras.Model(inputs, result)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam())

    return model

titanic_model = titanic_model(titanic_preprocessing, inputs)

# when training model use dictionary features as x and labels as y
titanic_model.fit(x=titanic_features_dict,
                   y=titanic_labels,
                   epochs=10)

titanic_model().save('test')
reloaded = tf.keras.model.load_model('test')

features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}

before = titanic_model(features_dict)
after = reloaded(features_dict)
assert (before-after)<1e-3
print(before)
print(after)


# Using tf.data

import itertools
# def slices(features):
#     for i in itertools.count():
#         # for each feature take index 'i'
#         example = {name:values[i] for name, value in features.items}
#         yield example

#
# for example in slices(titanic_features_dict):
#     for name, value in example.items():
#         print(f'{name:19s}: {value}')
#     break

features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)

for example in features_ds:
    for name, value in example.items():
        print(f'{name:19s}: {value}')
    break

titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))
titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)

titanic_model.fit(titanic_batches, epochs=5)

# from a single file
titanic_file_path = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")

titanic_csv_ds = tf.data.experimental.make_csv_dataset(
    titanic_file_path,
    batch_size=5,
    label_name='survived',
    num_epochs=1,
    ignore_errors=True
)

for batch, label in titanic_csv_ds.take(1):
    for key, value in batch.items():
        print(f'{key:20s}: {value}')
    print()
    print(f"{'label':20s}: {label}")




# tf.io.decode_csv: a function for parsing lines of text into a list of CSV column tensors.
# tf.data.experimental.CsvDataset: a lower-level CSV dataset constructor.
# tf.data.experimental.make_csv_dataset

text = pathlib.Path(titanic_file_path).read_txt()
lines = text.split('\n')[1:-1]
all_strings = [str()]*10
all_strings

features = tf.io.decode_csv(lines, record_defaults=all_strings)

for f in features:
    print(f'type: {f.dtype.name}, shape: {f.shape}')

print(lines[0])

titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]
titanic_types

features = tf.io.decode_csv(lines, record_defaults=titanic_types)

for f in features:
    print(f'type: {f.dtype.name}, shape: {f.shape}')

# tf.data.experimental.CsvDataset provides minimasl CSV dataset interface and does not inclued
# column header parsing, column type interface, automatic shuffling, file interleaving
# which are included in tf.data.experamental.make_csv.dataset

simple_titanic = tf.data.experimental.CsvDataset(titanic_file_path, record_defaults=titanic_types, header=True)

for example in simple_titanic.take(1):
    print([e.numpy() for e in example])

# the above body of code is equivalent to:

def decode_titanic_line(line):
  return tf.io.decode_csv(line, titanic_types)

manual_titanic = (
    # Load the lines of text
    tf.data.TextLineDataset(titanic_file_path)
    # Skip the header row.
    .skip(1)
    # Decode the line.
    .map(decode_titanic_line)
)

for example in manual_titanic.take(1):
  print([e.numpy() for e in example])


# multiple files
# first need determine column types for record_defaults. start inspecting first row of one file.

import pathlib
font_csvs =  sorted(str(p) for p in pathlib.Path('fonts').glob("*.csv"))

font_line = pathlib.Path(font_csvs[0].read_text().splitlines()[1])
print(font_line)

num_font_features = font_line.count('.')+1
font_column_types = [str(), str()] + [float()]*(num_font_features-2)

#the tf.data.experimental.CsvDataset can take a list of input files and read the sequentially

print(font_csvs[0])

simple_font_ds = tf.data.experimental.CsvDataset(
    font_csvs,
    record_defaults=font_column_types,
    header=True
)

for row in simple_font_ds.take(10):
    print(row[0].numpy())

font_files = tf.data.Dataset.list_files('fonts/*.csv')

print('Epoch 1:')
for f in list(font_files)[:5]:
    print('    ', f.numpy())
print('    ...')
print()

print('Epoch 2:')
for f in list(font_files)[:5]:
    print('    ', f.numpy())
print('    ...')

def make_font_csv_ds(path):
    return tf.data.experimental.CsvDataset(
        path,
        record_defaults=font_column_types,
        header=True
    )

font_rows = font_files.interleave(make_font_csv_ds,
                                  cycle_length=3)

fonts_dict = {'font_name': [], 'character':[]}

for row in font_rows.take(10):
    fonts_dict['font_name'].append(row[0].numpy().decode())
    fonts_dict['character'].append(chr(row[2].numpy()))

pd.DataFrame(fonts_dict)


# Performance tf.io.decode_csv is more efficient when run on batch of strings

BATCH_SIZE = 2048
fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern= 'fonts/*.csv',
    batch_size=BATCH_SIZE, num_epochs=1,
    num_parallel_reads=100
)

# %%time
for i, batch in enumerate(fonts_ds.take(20)):
    print('.', end='')

print()

# passing batches of text lines to decode_csv runs faster

fonts_files = tf.data.Dataset.list_files('fonts/*.csv')
fonts_lines = fonts_files.interleave(
    lambda fname:tf.data.TextLineDataset(fname).skip(1),
    cycle_length=100).batch(BATCH_SIZE)

fonts_fast = fonts_lines.map(lambda x: tf.io.decode_csv(x, record_defaults=font_column_types))

# %%time
for i, batch in enumerate(fonts_fast.take(20)):
    print('.', end='')

print()

