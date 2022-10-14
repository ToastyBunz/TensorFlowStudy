import pandas as pd
import tensorflow as tf

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')

df = pd.read_csv(csv_file)
print(df.head())

print(df.dtypes)

# target is going to be our label 1 for cancer 0 for no cancer
target = df.pop('target')


# DataFrame as array
# take the numeric features because numpy can only accept a single dtype

numeric_features_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
numeric_features = df[numeric_features_names]
print(numeric_features.head())

tf.convert_to_tensor(numeric_features)

# in general if an object can be converted to a tensor it can be bassed anywhere you can pass tf.tensor

# with Model.fit

normalizer = tf.keras.layers.Normalization(axis=1)
normalizer.adapt(numeric_features)
# this normalized the rows

normalizer(numeric_features.iloc[:3])

# use the norm layer as the first layer of a simple model:

def get_basic_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

# When youpass the datafram as the x arguent to Model.fit Keras treats the Dataframe as it would be a NumPy

model = get_basic_model()
model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)


# with tf.data
numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

for row in numeric_dataset.take(3):
    print(row)

numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)

model = get_basic_model()
model.fit(numeric_batches, epochs=15)


# Dataframe as a dictionary
# issue arrises because like numpy tf tensors also can only accept 1 data type

numeric_dict_ds = tf.data.Dataset.from_tensor_slices((dict(numeric_features), target))

for row in numeric_dict_ds.take(3):
    print(row)

def stack_dict(inputs, fun=tf.stack):
    values = []
    for key in sorted(inputs.keys()):
      values.append(tf.cast(inputs[key], tf.float32))

    return fun(values, axis=-1)


class MyModel(tf.keras.Model):
    def __int__(self):
        # Create all the interal layers in init
        super().__init__(self)

        self.normalizer = tf.keras.layers.Normalization(axis=-1)

        self.seq = tf.keras.Sequential([
            self.normalizer,
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def adapt(self, inputs):
        # Stack the inputs and 'adapt' the normaliziation layer.
        inputs = stack_dict(inputs)
        self.normalizer.adapt(inputs)


    def call(self, inputs):
        # Stack the inputs
        inputs = stack_dict(inputs)
        # Run them through all the layers
        result = self.seq(inputs)

        return result

model = MyModel()

model.adapt(dict(numeric_features))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(dict(numeric_features), target, epochs=5, batch_size=BATCH_SIZE)

numeric_dict_batches = numeric_dict_ds.shuffe(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)

model.predict(dict(numeric_features.iloc[:3]))


# 2  (the Keras functional style)

inputs = {}
for name, column in numeric_features.items():
    inputs[name] = tf.keras.Input(
        shape=(1,),
        name=name,
        dtype=tf.float32
    )
x = stack_dict(inputs, fun=tf.concat)

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

x = normalizer(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(inputs, x)

model.comile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
    run_eagerly=True
)

tf.keras.utils.plot_model(model, rankdir="LR", show_shapes=True)

# Helpful graph of whats going on
# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe#a_dataframe_as_an_array

model.fit(dict(numeric_features), target,epochs=5, batch_size=BATCH_SIZE)

numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
model.fit(numeric_dict_batches, epochs=5)


### Full Example

binary_feature_names = ['sex', 'fbs', 'exang']
categorical_feature_names = ['cp', 'restecg', 'slope', 'thal', 'ca']

inputs = {}
for name, column in df.items():
    if type(column[0]) == str:
        dtype = tf.string
    elif (name in categorical_feature_names or
        name in binary_feature_names):
        dtype = tf.int64
    else:
        dtype = tf.float32

    inputs[name] = tf.keras.Input(shape=(), name=name, dtype=dtype)

print(inputs)

# binary inputs don't need preprocessing so just add the vector axis and add to the list of preprocessed inputs

preprocessed = []

for name in binary_feature_names:
    inp = inputs[name]
    inp = inp[:, tf.newaxis]
    float_value = tf.cast(inp, tf.float32)
    preprocessed.append(float_value)

print(preprocessed)

# Numeric inputs. Same as before but input as a dict here

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(stack_dict(dict(numeric_features)))

# this stacks the numeric features and runs them through normalization
numeric_inputs = {}
for name in numeric_features_names:
    numeric_inputs[name] = inputs[name]

numeric_inputs = stack_dict(numeric_inputs)
numeric_normalized = normalizer(numeric_inputs)

preprocessed.append(numeric_normalized)

print(preprocessed)

# Categorical features (one hot method)
# example
vocab = ['a', 'b', 'c']
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
print(lookup(['c', 'a', 'a', 'b', 'zzz']))
# there are four one hot encoded columns with the first being an OOV column

# Example 2 it works with numbers too!
vocab = [1, 4, 7, 99]
lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

print(lookup([-1, 4, 1]))


for name  in categorical_feature_names:
    vocab = sorted(set(df[name]))
    print(f'name: {name}')
    print(f'vocab: {vocab}\n')

    if type(vocab[0]) is str:
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
    else:
        lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab, output_mode='onehot')

    x = inputs[name][:, tf.newaxis]
    x = lookup(x)
    preprocessed.append(x)

# asseble the preprocessing head. preprocessed is justa a python list of preprocessing results
print(preprocessed)

# time to concatenate all the preprocessed features along the depth axis
# this will convert the dictionary examples in to single vectors. the vector contains
# Categorical, numeric, and categorical one-hot vectors


preprocessed_result = tf.concat(preprocessed, axis=-1)

# now create the model out of that calculation so it can be reuesed:

preprocessor = tf.keras.Model(inputs, preprocessed_result)
print(tf.keras.utils.plot_model(preprocessor, rankdir='LR', show_shapes=True))

print(preprocessor(dict(df.iloc[:1])))

### create an train model

body = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# now put the two pieces together using keras functional api

print(inputs)
x = preprocessor(inputs)
print(x)

result = body(x)
print(result)

model = tf.keras.Model(inputs, result)
model.compile(optimizer='adam',
              loass=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# this model expects a dictionary fo inputs. Easiest: converty DF to dict and pass as x arg to model.fit
history = model.fit(dict(df), target, epochs=5, batch_size=BATCH_SIZE)

# using tf.data works as well
ds = tf.data.Dataset.from_tensor_slices((
    dict(df),
    target
))

ds = ds.batch(BATCH_SIZE)

import pprint
for x, y in ds.take(1):
    pprint.pprint(x)
    print()
    print(y)

history = model.fit(ds, epochs=5)