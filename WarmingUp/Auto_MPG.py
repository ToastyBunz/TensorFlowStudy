import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import layers



np.set_printoptions(precision=3, suppress=True)

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)


dataset = raw_dataset.copy()
# print(dataset.tail())
# print(dataset.isna().sum())
#
dataset = dataset.dropna()


dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
# print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

chub = sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show(block=True )
# plt.show() # is only showing for a split second then disappearing

# print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# print(train_dataset.describe().transpose()[['mean', 'std']])

normalizer = tf.keras.layers.Normalization(axis=1)
normalizer.adapt(np.array(train_features))
# print(normalizer.mean.numpy())

# predicting MPG from horsepower using linear regression
# step 1: normalize horsepower
# step 2: Apply a linear transformation (y=mx+b) to produce 1 output using a linear layer (tf.keras.layers.Dense)

horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1, ], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

# print(horsepower_model.summary())
# horsepower_model.predict(horsepower[:10])

horsepower_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss = 'mean_absolute_error'
)

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    verbose=0,
    validation_split = 0.2
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

plot_loss(history)


test_results = {}

test_results['Horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels,
    verbose=0)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    plt.show(block=True)

plot_horsepower(x, y)


def multi_input_regression():
#Linear regression with multiple inputs
    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

    linear_model.predict(train_features[:10])

    print(linear_model.layers[1].kernel)

    linear_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )

    history = linear_model.fit(
        train_features,
        train_labels,
        epochs=100,
        verbose=0,
        validation_split=0.2)

    plot_loss(history)

    test_results['linear_model'] = linear_model.evaluate(
        test_features,
        test_labels,
        verbose=0
    )

multi_input_regression()

# DNN model
# def single_input_DNN_model():

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
print(dnn_horsepower_model.summary())

history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100
)

plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)
# plt.show()

test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels,
    verbose=0
)

# DNN with multiple inputs

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100
)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features,
                                               test_labels,
                                               verbose=0)

# Testing preformance of different models

df = pd.DataFrame(test_results, index=['Mean absolute error [MPG}'])
df = df.T

print(df)


test_predictions = dnn_model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Value [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Predictions Error [MPG]')
_ = plt.ylabel('Count')

dnn_model.save('dnn_model')
reloaded = tf.keras.models.load_model('dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features,
    test_labels,
    verbose=0
)

dt2 = pd.DataFrame(test_results, index=['Mean absolute error [MPG]'])
dt2 = dt2.T

