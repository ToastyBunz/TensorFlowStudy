# Multi output multi timestep

# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Sequential


import tensorflow as tf
# from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import gc
import sys

mpl.rcParams['figure.figsize'] = (17, 5)
mpl.rcParams['axes.grid'] = False
sns.set_style("whitegrid")

notebookstart= time.time()

# Data Loader Parameters
BATCH_SIZE = 256
BUFFER_SIZE = 10000
TRAIN_SPLIT = 300000

# LSTM Parameters
EVALUATION_INTERVAL = 200
EPOCHS = 4
PATIENCE = 5

# Reproducibility
SEED = 13
tf.random.set_seed(SEED)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)
print("DataFrame Shape: {} rows, {} columns".format(*df.shape))
# print(df.head())

features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
features = df[features_considered]
features.index = df['Date Time']
print(features.head())

features.plot(subplots=True)
plt.show()

# Standardize

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

past_history = 720
future_target = 72
STEP = 6


def multivariate_multioutput_data(dataset, target, start_index, end_index, history_size,
                                  target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data)[:,:,:,np.newaxis, np.newaxis], np.array(labels)[:,:,:, np.newaxis, np.newaxis]

def create_time_steps(length):
    return list(range(-length, 0))



def multi_step_output_plot(history, true_future, prediction):
    plt.figure(figsize=(18, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    for i, (var, c) in enumerate(zip(features.columns[:2], ['b', 'r'])):
        plt.plot(num_in, np.array(history[:, i]), c, label=var)
        plt.plot(np.arange(num_out)/STEP, np.array(true_future[:, i]), c+'o', markersize=5, alpha=0.5,
                 label=f'True {var.title()}')
        if prediction.any():
            plt.plot(np.arange(num_out)/STEP, np.array(prediction[:, i]), '*', markersize=5, alpha=0.5,
                     label=f'Predicted {var.title()}')

    plt.legend(loc='upper left')
    plt.show()


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

future_target = 144
x_train_multi, y_train_multi = multivariate_multioutput_data(dataset[:,:2], dataset[:,:2], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_multioutput_data(dataset[:,:2], dataset[:, :2],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

BATCH_SIZE = 128

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


def build_model(input_timesteps, output_timesteps, num_links, num_inputs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.BatchNormalization(name='batch_norm_0', input_shape=(input_timesteps, num_inputs, 1, 1)))
    model.add(tf.keras.layers.ConvLSTM2D(name='conv_lstm_1',
                         filters=64, kernel_size=(10, 1),
                         padding='same',
                         return_sequences=False))

    model.add(tf.keras.layers.Dropout(0.30, name='dropout_1'))
    model.add(tf.keras.layers.BatchNormalization(name='batch_norm_1'))

    #     model.add(ConvLSTM2D(name ='conv_lstm_2',
    #                          filters = 64, kernel_size = (5, 1),
    #                          padding='same',
    #                          return_sequences = False))

    #     model.add(Dropout(0.20, name = 'dropout_2'))
    #     model.add(BatchNormalization(name = 'batch_norm_2'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.RepeatVector(output_timesteps))
    model.add(tf.keras.layers.Reshape((output_timesteps, num_inputs, 1, 64)))

    #     model.add(ConvLSTM2D(name ='conv_lstm_3',
    #                          filters = 64, kernel_size = (10, 1),
    #                          padding='same',
    #                          return_sequences = True))

    #     model.add(Dropout(0.20, name = 'dropout_3'))
    #     model.add(BatchNormalization(name = 'batch_norm_3'))

    model.add(tf.keras.layers.ConvLSTM2D(name='conv_lstm_4',
                         filters=64, kernel_size=(5, 1),
                         padding='same',
                         return_sequences=True))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, name='dense_1', activation='relu')))
    model.add(tf.keras.layers.Dense(units=1, name='dense_2'))

    #     optimizer = RMSprop() #lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.9)
    #     optimizer = tf.keras.optimizers.Adam(0.1)
    optimizer = tf.keras.optimizers.RMSprop(lr=0.003, clipvalue=1.0)
    model.compile(loss="mse", optimizer=optimizer, metrics=['mae', 'mse'])
    return model

EPOCHS = 5
steps_per_epoch = 150
validation_steps = 200

modelstart = time.time()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = PATIENCE, restore_best_weights=True)
model = build_model(x_train_multi.shape[1], future_target, y_train_multi.shape[2], x_train_multi.shape[2])
print(model.summary())

# Train
print("\nTRAIN MODEL...")
history = model.fit(train_data_multi,
                    epochs = EPOCHS,
                    validation_data=val_data_multi,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[early_stopping])
model.save('multi-output-timesteps.h5')
print("\nModel Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

plot_train_history(history, 'Multi-Step, Multi-Output Training and validation loss')

for x, y in val_data_multi.take(10):
    multi_step_output_plot(np.squeeze(x[0]), np.squeeze(y[0]), np.squeeze(model.predict(x[0][np.newaxis,:,:,:,:])))