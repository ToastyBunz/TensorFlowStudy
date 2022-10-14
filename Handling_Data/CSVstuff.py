import heartpatients as pd
import numpy as np
import tensorflow as tf
from keras import layers


# make numpy values esier to read
np.set_printoptions(precision=3, suppress=True)

# small CSV: Load memory > as pandas or numpy array

abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Age']
)
# print(abalone_train.head())

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

abalone_features = np.array(abalone_features)
# print(abalone_features)

abalone_model = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

abalone_model.fit(abalone_features, abalone_labels, epochs=10)


## Basic preprocessing

# normalization layer precomuptes the mean and varience of each column and uses these to normalize data

normalize = layers.Normalization()
normalize.adapt(abalone_features)

norm_abalone_model = tf.keras.Sequential([
    normalize,
    layers.Dense(64),
    layers.Dense(1)
])

norm_abalone_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)