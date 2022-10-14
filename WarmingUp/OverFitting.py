import tensorflow as tf

print(tf.__version__)

from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile
import pandas as pd

logdir = pathlib.Path(tempfile.mkdtemp())/'tensorboard_logs'
shutil.rmtree(logdir, ignore_errors=True)

gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')

Features = 28

# this line can be used ot read csv directly from a gzip with not intermediate decompression step
ds = tf.data.experimental.CsvDataset(gz, [float(), ]*(Features + 1), compression_type='GZIP')

# repacks scalars into a (feature_vector, label) pair
def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:], 1)
    return features, label

# TF is more efficient when operation on large batches of data
# Makes new dataset that takes batches of 10000 examples and applies pack_row
# then splits them back into individual records
packed_ds = ds.batch(10000).map(pack_row).unbatch()

#inspect

for features, label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins= 101)
    # plt.show()

# use first 1000 samples for validation and 10000 for training

n_validation = int(1e3)
n_train = int(1e4)
buffer_size = int(1e4)
batch_size = 500
steps_per_epoch = n_train//batch_size

# this is creating the test and validation sets
# another way of doing the list splitting like int TF_Q3
validate_ds = packed_ds.take(n_validation).cache()
train_ds = packed_ds.skip(n_validation).take(n_train).cache()

validate_ds = validate_ds.batch(batch_size)
train_ds = train_ds.shuffle(buffer_size).repeat().batch(batch_size)

# Want model to train on general features not super specific
# The simplest way to prevent overfitting is to start with a small model:
# A model with a small number of learnable parameters
# (which is determined by the number of layers and the number of units per layer).
# In deep learning, the number of learnable parameters in a model is often referred to as the model's "capacity".


# Many models train better if you gradually reduce the learning rate during training.

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=steps_per_epoch*1000,
    decay_rate = 1,
    staircase=False
)

def get_optimizer():
    return tf.keras.optimizers.Adam(lr_schedule)

step = np.linspace(0, 100000)
lr = lr_schedule(step)
plt.figure(figsize=(8, 6))
plt.plot(step/steps_per_epoch, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')
# plt.show()

# EpochDots prints a . for each epoch and a full set of metrics every 100 epochs
def get_callbacks(name):
    return [
        # tf.keras.tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir/name)
    ]

def compile_and_fit(model, name, optimizer=None, max_epochs=1000):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[
                      tf.keras.losses.BinaryCrossentropy(from_logits=True,name='binary_crossentropy'), 'accuracy']

                  )
    print(model.summary())

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=2
    )
    return history

tiny_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='elu', input_shape=(Features,)),
    tf.keras.layers.Dense(1)
])

size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

# plotter = tf.keras.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
# plotter.plot(size_histories)
# plt.ylim([0.5, 0.7])

# def plot_bin():
#     hist = pd.DataFrame(history.history)
#     hist['epoch'] = history.epoch
#     plt.plot(history.history['loss'], label='loss')
#     plt.plot(history.history['val_loss'], label='val_loss')
#     plt.ylim([0, 10])
#     plt.xlabel('Epoch')
#     plt.ylabel('Error [MPG]')
#     plt.legend()
#     plt.grid(True)
#     plt.show(block=True)


#small model
small_model = tf.keras.Sequential([
    # input_shape is required here so .summary works
    tf.keras.layers.Dense(16, activation='elu', input_shape=(Features,)),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(1)
])

size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

#Medium model

medium_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='elu', input_shape=(Features)),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(1)
])


size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium')

#Large Model

large_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='elu', input_shape=(Features)),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(1)

])

size_histories['large'] = compile_and_fit(large_model, 'sizes/large')


# plotter.plot(size_histories) #plotter is not working
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim(0.5,0.7)
plt.xlabel('Epochs [Log Scale]')


# View in Tensorboard

# I don't know where this is supposed to go maybe in terminal?
# Load the TensorBoard notebook extension
# %load_ext tensorboard

# Open an embedded TensorBoard viewer
# %tensorboard --logdir {logdir}/sizes

display.IFrame(
    src="https://tensorboard.dev/experiment/vW7jmmF9TmKmy3rbheMQpw/#scalars&_smoothingWeight=0.97",
    width="100%", height="800px"
)

# How to save Tensorboard readouts
# $ tersorboard dev opload --logdir {logdir}/sizes


#Stratedgies to prevent overfitting

# copy training logs for Tiny to use as baseline

shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

# posixpath('/tmpfs/tmp/tmp94hlpmkm/tensorboard_logs/regularizers/Tiny')

regularizers_histories = {}

regularizers_histories['Tiny'] = size_histories['Tiny']


# adding l2

l2_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(Features,)),
    tf.keras.layers.Dense(512, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(Features,)),
    tf.keras.layers.Dense(512, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(Features,)),
    tf.keras.layers.Dense(512, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.0001), input_shape=(Features,)),
    tf.keras.layers.Dense(1)
])

regularizers_histories['l2'] = compile_and_fit(l2_model, 'regularizers/12')

# plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

result = l2_model(features)
regularization_loss= tf.add_n(l2_model.losses)

dropout_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='elu', input_shape=(Features)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(1)
])

regularizers_histories['dropout'] = compile_and_fit(dropout_model, 'regularizers/dropout')

# plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

# combined L2 + Dropout

combined_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.L1L2(0.001), activation='elu', input_shape=(Features)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.L1L2(0.001), activation='elu', input_shape=(Features)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.L1L2(0.001), activation='elu', input_shape=(Features)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.L1L2(0.001), activation='elu', input_shape=(Features)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dropout(1)
])

regularizers_histories['combined'] = compile_and_fit(combined_model, 'regularizers/combined')

# plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])

# View in TensorBoard
# %tensorboard --logdir {logdir}/regularizers

display.IFrame(
    src= 'https://tensorboard.dev/experiment/fGInKDo8TXes1z7HQku9mw/#scalars&_smoothingWeight=0.97',
    width='100%',
    height='800px'
)

# $ tensorboard dev upload --logdir {logdir}/regularizers
