import heartpatients as pd
import numpy as np
import tensorflow as tf
from keras import layers
import pydot
import graphviz

# make numpy values esier to read
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
def slices(features):
    for i in itertools.count():
        # for each feature take index 'i'
        example = {name:values[i] for name, value in features.items}
        yield example


for example in slices(titanic_features_dict):
    for name, value in example.items():
        print(f'{name:19s}: {value}')
    break

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


