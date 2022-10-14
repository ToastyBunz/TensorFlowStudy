# 2 types
# Forcast for a single time setep
# a single feature
# all features

# Forcast multiple steps:
# Stingle shot make predictions all at once
# autoregressive: make one prediction at a time and feed the output back to the model

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

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

'''I am starting with just typing out everything I have. Then I will itemize. Then if there is time
I will get started otherwise (11:45pm) I am starting at 630 tomorro morning'''


import urllib.request
import zipfile

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np


# this func downloads and extracts to directory that contains this file
# DO NOT CHANGE THIS CODE
# THIS CODE IS NOT WORKING BECAUSE ONLY GRANTED ACCESS DURING TEST
# def download_and_extract_data():
#     url = 'https://storage.googlapis.com/download.tensorflow.org/data/certificate/household_power.zip'
#     urllib.request.urlretrieve(url, 'household_power.zip')
#     with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
#         zip_ref.extractall()

def get_clean():
    df = pd.read_csv('E:/pythonProject/TensorReview/TF_Questions/household_power_consumption.txt', sep=';',
                           infer_datetime_format=True, header=0,
                           parse_dates={'datetime':[0,1]}, low_memory=False)

    zf = pd.read_csv('E:/pythonProject/TensorReview/TF_Questions/household_power_consumption.txt', sep=';',
                     infer_datetime_format=True, header=0,
                     parse_dates={'datetime': [0, 1]}, low_memory=False)


    df.replace('?', np.NaN, inplace=True)
    # print(df.isnull().sum(axis = 0))
    df = df.dropna()
    df = df[5::6]
    df.pop('datetime')
    # print(df.head())
    for i in df.columns[df.isnull().any(axis=0)]:  # ---Applying Only on variables with NaN values
        df[i].fillna(df[i].mean(), inplace=True)
    # print(df.isnull().sum(axis=0))
    # print(df.mean())
    # df.fillna(df.mean())
    return df, zf

df, zf = get_clean()
# print(df.describe().transpose())
print('stop')

# infer_datetime_format=True, index_col=['datetime'],


# # This function normalizes using min max scaling
# #DO NOT CHANGE THIS CODE
# def normalize_series(data, min, max):
#     # data = float(input(data))
#     data = data - min
#     data = data / max
#     return data
#
#
# # this func turns time series into windows
# # DO NOT CHANGE THIS CODE
# def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
#     ds = tf.data.Dataset.from_tensor_slices(series)
#     ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
#     ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
#     ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
#     return ds.batch(batch_size).prefetch(1)
#
#
# #This func:
# # loads the data from CSV
# # normalizes data
# # splits into train and validation
# # uses windowed_dataset to split data into windows of observation and targets
# # model
# # compile
# # fit
# # returns the fit
#
#
# # COMPLETE THIS FUNCTION
# def solution_model():
#     # download and extract data the directory that contains this file
#     # download_and_extract_data()
#     # reads the dataset from the CSV
#     # https://machinelearningmastery.com/how-to-load-and-explore-household-electricity-usage-data/
#     # https: // archive.ics.uci.edu / ml / datasets / individual + household + electric + power + consumption
#     get_clean()
#
#     # number of features in the dataset. We use all features as predictors
#     N_FEATURES = len(df.columns) # DO NOT CHANGE THIS
#
#     # Normalize (Their stuff isnt working so I am going to use my own normalization)
#     data = df.values
#     # data = normalize_series(data, data.min(axis=0), data.max(axis=0))
#     # Mine
#     normalize = tf.keras.layers.Normalization()
#     normalize.adapt(df)
#
#     # splits the data into training and validation sets
#     SPLIT_TIME = int(len(data) * 0.5) # do not change this
#     x_train = data[:SPLIT_TIME]
#     x_valid = data[SPLIT_TIME:]
#
#     # DO NOT CHANGE THIS CODE
#     tf.keras.backend.clear_session()
#     tf.random.set_seed(42)
#     BATCH_SIZE = 32
#     N_PAST = 24
#     N_FUTURE = 24
#     SHIFT = 1
#
#     # code to create window train and validation Do Not Change
#     train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)
#     valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=SHIFT)
#
#     # Your code
#
#     model = tf.keras.model.Sequential([
#
#     ])
#
#
#
#     # There are a bunch of notes here so referance the phone
#     # input shape must be (batch_size, npast=24, nfeatures = 7)
#     # output shape must be (batch_size, n_future, n_features)
#
#     model.compile(loss=tf.keras.losses.Huber(),
#                   optimizer='adam',
#                   metrics=['mae'])
#
#     model.fit()
#
# solution_model()