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


df = pd.read_csv('E:/pythonProject/TensorReview/TF_Questions/household_power_consumption.txt', sep=';',
                           infer_datetime_format=True, parse_dates={'Datetime':[0,1]}, header=0, low_memory=False, nrows=86400)

# parse_dates={'datetime':[0,1]}
# inspect tools
# print(df.head())
# print(df.isna().sum())

df.replace('?', 0, inplace=True)
df.drop(columns=['Datetime'], inplace=True)

for i in df.columns:
    pd.to_numeric(df[i], errors='coerce')

# df.astype(float)

# pd.to_numeric(df, errors='coerce')

types = df.dtypes
print(types)
print(df['Global_active_power'])

print(df['Global_active_power'].mean())


#
# print(df.mean())



# df.dropna(inplace=True)
# df.reset_index(drop=True, inplace=True)

# for i in df.columns[df.isnull().any(axis=0)]:  # ---Applying Only on variables with NaN values
#     df[i].fillna(df[i].mean(), inplace=True)
#
# for i in df.columns:
#     print(df.columns[i].mean())

# print('GI intense', df['Global_intensity'].mean())
#
# nan_values = df[df['Sub_metering_2'].isna()]
# print(nan_values)
#
#
# print(df.loc[[6839]])
#
# print(df.isna().sum())
# # print(nan_values)




print('hello')



# ### data preprocessing

# date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S') # removes timestamps just use index
#
# # plot to understand
# plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
# plot_features = df[plot_cols]
# plot_features.index = date_time
# _ = plot_features.plot(subplots=True)
#
# plot_features = df[plot_cols][:480]
# plot_features.index = date_time[:480]
# _ = plot_features.plot(subplots=True)
#
# # stats to see errors
# stats = df.describe().transpose()
# # remove erronious data
# wv = df['wv (m/s)']
# bad_wv = wv == -9999.0
# wv[bad_wv] = 0.0
#
# max_wv = df['max. wv (m/s)']
# bad_max_wv = max_wv == -9999.0
# max_wv[bad_max_wv] = 0.0
#
# # Convert wind directions to radians
# wv = df.pop('wv (m/s)')
# max_wv = df.pop('max. wv (m/s)')
#
# # Convert to radians.
# wd_rad = df.pop('wd (deg)')*np.pi / 180
#
# # Calculate the wind x and y components.
# df['Wx'] = wv*np.cos(wd_rad)
# df['Wy'] = wv*np.sin(wd_rad)
#
# # Calculate the max wind x and y components.
# df['max Wx'] = max_wv*np.cos(wd_rad)
# df['max Wy'] = max_wv*np.sin(wd_rad)
#
# # convert string time to seconds
# timestamp_s = date_time.map(pd.Timestamp.timestamp)
# print(timestamp_s)