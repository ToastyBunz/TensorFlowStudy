import csv
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# df = pd.read_csv('E:/pythonProject/TensorReview/TF_Questions/household_power_consumption.csv', nrows=86400)

# infer_datetime_format=True, parse_dates={'Datetime':[0,1]}, header=0, low_memory=False,

# NEEDED TO START NOW COMMENTED
# for i in range(2):
#     df.drop(columns=df.columns[0],
#             axis=1,
#             inplace=True)
#     print('1')
#
# df.replace('?', 'NaN', inplace=True)
# # print(df.isna().sum())
# df.dropna(inplace=True)
# df.reset_index(drop=True, inplace=True)
#
# df.to_csv('E:\pythonProject\TensorReview\TF_Questions\HPC_2.csv')

df = pd.read_csv('E:\pythonProject\TensorReview\TF_Questions\HPC_2.csv')

df.drop(columns=['Unnamed: 0'], inplace=True)
print((df['Global_active_power']).mean())
print(df.columns)
# print(df.dtypes)
# print(df.describe().transpose())

column_indacies = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):]

num_features = df.shape[1]

# There are mulitple ways to normalize
def normalize_series_v2(train_df, val_df): # reccomended
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    return train_df, val_df

def normalize_series_v1(train_df, val_df):
    train_min = train_df.min()
    train_max = train_df.max()
    test_min = val_df.min()
    test_max = val_df.max()

    train_df = (train_df - train_min) / train_max
    val_df = (val_df - test_min) / test_max
    return train_df, val_df


train_df, val_df = normalize_series_v1(train_df, val_df)


class WindowGenerator():
    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, label_columns=None):

        #Store data
        self.train_df = train_df
        self.val_df = val_df


        # Work out label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indicies = {name: i for i, name in enumerate(label_columns)}
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width) # from start to slice to the input width
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return'\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes manually.
        # This way the 'tf.data.Datasets' are easier to inpsect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def example(self):
        '''Get a cache an example batch of 'inputs, lables' for plotting'''
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the '.train' dataset
            result = next(iter(self.train))
            self.example = result
            return result


label = ['Global_active_power']
label_str = 'Global_active_power'

# labels_lst = ['Global_active_power', 'Global_reactive_power', 'Voltage',
#        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2',
#        'Sub_metering_3']

w1 = WindowGenerator(input_width=24, label_width=1, shift=24, label_columns=label)

print(w1)

# Split

# def split_window(self, features):
#     inputs = features[:, self.input_slice, :]
#     labels = features[:, self.labels_slice, :]
#     if self.label_columns is not None:
#         labels = tf.stack(
#             [labels[:, :, self.column_indices[name]] for name in self.label_columns],
#             axis=-1)
#
#     # Slicing doesn't preserve static shape information, so set the shapes manually.
#     # This way the 'tf.data.Datasets' are easier to inpsect.
#     inputs.set_shape([None, self.input_width, None])
#     labels.set_shape([None, self.label_width, None])
#
#     return inputs, labels


# split_window = WindowGenerator.split_window

# Stack three slices, the length of teh total window.
example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                           np.array(train_df[100:100+w1.total_window_size]),
                           np.array(train_df[200:200+w1.total_window_size])])


example_inputs, example_labels = w1.split_window(example_window)
w1.example = example_inputs, example_labels

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Label shape: {example_labels.shape}')


# create dataset

# def make_dataset(self, data):
#     data = np.array(data, dtype=np.float32)
#     ds = tf.keras.utils.timeseries_dataset_from_array(
#         data=data,
#         targets=None,
#         sequence_length=self.total_window_size,
#         sequence_stride=1,
#         shuffle=True,
#         batch_size=32)
#
#     ds = ds.map(self.split_window)
#
#     return ds

# WindowGenerator.make_dataset = make_dataset

# windows generator holds the datasets.
# add properties for accessing them using make+dataset method.
# add a standard example batch for easy access and plotting

@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def example(self):
    '''Get a cache an example batch of 'inputs, lables' for plotting'''
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the '.train' dataset
        result = next(iter(self.train))
        self.example = result
        return result


# WindowGenerator.train = train
# WindowGenerator.val = val
# WindowGenerator.example = example

    # WindowGenerator obj gives access to tf.data.Dataset, which makes them easy to iterate

# Dataset.element_spec property tells the structure, datatypes and shape.
# not working
# print(w1.train.element_spec)

# iterating over a dataset yields concrete batches

# for example_inputs, example_labels in w1.train.take(1):
#     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#     print(f'Labels shape (batch, time, features): {example_labels.shape}')

single_step_window = WindowGenerator(
    input_width=1,
    label_width=1,
    shift=1,
    label_columns=label
)

# print(single_step_window)
# for example_inputs, example_labels in single_step_window.train.take(1):
#     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#     print(f'Labels shape (batch, time, features): {example_labels.shape}')

class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

        def call(self, inputs):
            if self.label_index is None:
                return inputs
            result = inputs[:, :, self.label_index]
            return result[:, :, tf.newaxis]


baseline = Baseline(label_index=column_indacies[label_str])
baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val, verbose=0)


print('hello')



